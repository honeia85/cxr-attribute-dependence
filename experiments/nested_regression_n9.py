"""
n=9 sensitivity regression combining 6 clean + 3 overlap models.

The architecture-invariance null (ΔR²=0.000) from the n=6 regression has limited
power, so we extend to n=9 by adding the three overlap models (RAD-DINO, CheXzero,
CheSS-ResNet50) whose residualization was performed in
experiments/per_comorbidity_residualization_overlap.py.

Output: results/nested_regression_n9.csv (M1, M2, M5, M6a, M6b, M7a, M7b, M8)
        for comparison with the n=6 version in Table 4 of the manuscript.

Usage:
    PYTHONPATH=. python experiments/nested_regression_n9.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load n=6 clean-model data
clean = pd.read_csv("results/two_factor_regression_data.csv")

# Load n=3 overlap-model residualization output
overlap_resid = pd.read_csv("results/per_comorbidity_residualization_overlap.csv")

# Load OR reference (same for all models - data-only)
or_ref = clean[["attribute", "disease", "log_or", "abs_log_or",
                "odds_ratio"]].drop_duplicates(subset=["attribute", "disease"])

# Load erank reference
geom = pd.read_csv("results/geometry_analysis.csv")
erank_ref = geom[["model", "effective_rank", "ambient_dim"]].copy()
erank_ref["norm_erank"] = erank_ref["effective_rank"] / erank_ref["ambient_dim"]
erank_ref["log_erank"] = np.log(erank_ref["effective_rank"])

# Augment overlap with OR + erank
overlap = overlap_resid.merge(or_ref, on=["attribute", "disease"], how="left")
overlap = overlap.merge(erank_ref[["model", "effective_rank", "log_erank", "norm_erank"]],
                        on="model", how="left")
overlap = overlap.rename(columns={"effective_rank": "erank"})

# Standardize column order
keep = ["attribute", "attribute_type", "model", "disease",
        "auroc_baseline", "auroc_residualized", "auroc_drop",
        "log_or", "abs_log_or", "odds_ratio",
        "erank", "norm_erank", "log_erank"]
clean9 = clean[keep].copy()
overlap9 = overlap[keep].copy()
combined = pd.concat([clean9, overlap9], ignore_index=True)
print(f"Combined n=9 dataset: {len(combined)} obs "
      f"({combined['model'].nunique()} models)")
print(combined.groupby("model")["auroc_drop"].agg(["mean", "count"]))

# Drop rows with missing OR (continuous attrs vs outcome rows where OR undefined)
combined = combined.dropna(subset=["abs_log_or", "auroc_drop"])
print(f"After dropna: {len(combined)} obs")

# Save combined
combined.to_csv("results/two_factor_regression_data_n9.csv", index=False)

# ----- Nested regression -----
X_base = sm.add_constant(combined["abs_log_or"].values)
y = combined["auroc_drop"].values
n = len(combined)

def fit_and_report(name, formula, df):
    model = smf.ols(formula, data=df).fit()
    return {
        "model": name,
        "formula": formula,
        "r_squared": model.rsquared,
        "f_pvalue": model.f_pvalue,
        "n": len(df),
    }

rows = []

# M1: drop ~ |log(OR)|
rows.append(fit_and_report("M1", "auroc_drop ~ abs_log_or", combined))

# M2: + log(erank)
rows.append(fit_and_report("M2", "auroc_drop ~ abs_log_or + log_erank", combined))

# Binary-attribute subset for encoding (M1a)
# Skip M1a here since overlap doesn't have encoding merged; deferred.

# M5: + finding (fixed effect)
rows.append(fit_and_report("M5", "auroc_drop ~ abs_log_or + C(disease)", combined))

# M6a: + finding + log(erank)
rows.append(fit_and_report("M6a", "auroc_drop ~ abs_log_or + C(disease) + log_erank",
                           combined))

# M6b: + finding + model identity
rows.append(fit_and_report("M6b", "auroc_drop ~ abs_log_or + C(disease) + C(model)",
                           combined))

# M7a: |log(OR)| * log(erank) + finding
rows.append(fit_and_report("M7a",
                           "auroc_drop ~ abs_log_or * log_erank + C(disease)",
                           combined))

# M7b: |log(OR)| * model + finding
rows.append(fit_and_report("M7b",
                           "auroc_drop ~ abs_log_or * C(model) + C(disease)",
                           combined))

# M8: + finding + attribute
rows.append(fit_and_report("M8",
                           "auroc_drop ~ abs_log_or + C(disease) + C(attribute)",
                           combined))

out = pd.DataFrame(rows)
# Compute incremental ΔR² from baseline M1
r2_m1 = out.loc[out["model"] == "M1", "r_squared"].values[0]
out["delta_r2_vs_m1"] = out["r_squared"] - r2_m1

print("\n" + "=" * 70)
print(f"n=9 NESTED REGRESSION (n = {n} observations, 9 models)")
print("=" * 70)
print(out[["model", "formula", "r_squared", "delta_r2_vs_m1", "f_pvalue"]].to_string(index=False))

out.to_csv("results/nested_regression_n9.csv", index=False)
print(f"\nSaved: results/nested_regression_n9.csv")
