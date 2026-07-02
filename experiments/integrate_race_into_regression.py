"""
Integrate race as the 24th attribute into the nested regression panel.

Output:
  results/two_factor_regression_data_with_race.csv  (1440 obs = 24 x 10 x 6)
  results/nested_regression_with_race.csv           (M1-M8 with race included)

Usage:
    PYTHONPATH=. python experiments/integrate_race_into_regression.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load existing 23-attribute panel
base = pd.read_csv("results/two_factor_regression_data.csv")

# Load race residualization results
race = pd.read_csv("results/race_residualization_mimic_cxr.csv")
race_or = pd.read_csv("results/race_or_mimic_cxr.csv")

# Load geometry for erank
geom = pd.read_csv("results/geometry_analysis.csv")
erank_ref = geom[["model", "effective_rank"]].copy()
erank_ref["log_erank"] = np.log(erank_ref["effective_rank"])
erank_ref["norm_erank"] = erank_ref["effective_rank"] / geom["ambient_dim"]

# Merge race with OR + erank to match base schema
race_full = race.merge(
    race_or[["disease", "log_or", "abs_log_or", "odds_ratio"]],
    on="disease", how="left")
race_full = race_full.merge(
    erank_ref.rename(columns={"effective_rank": "erank"}),
    on="model", how="left")

# Align columns
common = ["attribute", "attribute_type", "model", "disease",
          "auroc_baseline", "auroc_residualized", "auroc_drop",
          "log_or", "abs_log_or", "odds_ratio",
          "erank", "norm_erank", "log_erank"]
race_full = race_full[common]

combined = pd.concat([base[common], race_full], ignore_index=True)
print(f"Combined: {len(combined)} obs ({combined['attribute'].nunique()} attrs "
      f"x {combined['disease'].nunique()} findings x "
      f"{combined['model'].nunique()} models)")
combined.to_csv("results/two_factor_regression_data_with_race.csv", index=False)

# Nested regression
def fit(name, formula, df):
    m = smf.ols(formula, data=df).fit()
    return {"model": name, "formula": formula,
            "r_squared": m.rsquared, "f_pvalue": m.f_pvalue,
            "n": len(df)}

rows = []
rows.append(fit("M1", "auroc_drop ~ abs_log_or", combined))
rows.append(fit("M2", "auroc_drop ~ abs_log_or + log_erank", combined))
rows.append(fit("M5", "auroc_drop ~ abs_log_or + C(disease)", combined))
rows.append(fit("M6a", "auroc_drop ~ abs_log_or + C(disease) + log_erank", combined))
rows.append(fit("M6b", "auroc_drop ~ abs_log_or + C(disease) + C(model)", combined))
rows.append(fit("M7a", "auroc_drop ~ abs_log_or * log_erank + C(disease)", combined))
rows.append(fit("M7b", "auroc_drop ~ abs_log_or * C(model) + C(disease)", combined))
rows.append(fit("M8", "auroc_drop ~ abs_log_or + C(disease) + C(attribute)", combined))

out = pd.DataFrame(rows)
r2_m1 = out.loc[out["model"] == "M1", "r_squared"].values[0]
out["delta_r2_vs_m1"] = out["r_squared"] - r2_m1
print(out[["model", "r_squared", "delta_r2_vs_m1", "f_pvalue"]].to_string(index=False))
out.to_csv("results/nested_regression_with_race.csv", index=False)
print("Saved: results/nested_regression_with_race.csv")

# Summary: where does race rank among 24 attributes?
rank = (combined.groupby("attribute")["auroc_drop"].mean()
        .sort_values(ascending=False).reset_index())
rank["rank"] = range(1, len(rank) + 1)
race_rank = rank[rank["attribute"] == "race_black_vs_white"]
print(f"\nRace rank among 24 attributes: {race_rank.iloc[0]['rank']} "
      f"of {len(rank)} (mean drop = {race_rank.iloc[0]['auroc_drop']:+.4f})")
