"""
Experiment 1: Inverse-frequency weighted OR → nested regression.
Tests whether repeated imaging inflates the OR→dependence relationship.
"""
import os, sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

COMORBIDITIES = [
    "hypertension", "heart_failure", "atrial_fibrillation",
    "coronary_artery_disease", "stroke", "diabetes", "hyperlipidemia",
    "obesity", "hypothyroidism", "ckd", "aki", "copd", "asthma",
    "respiratory_failure", "pulmonary_fibrosis", "liver_disease",
    "anemia", "smoking_history", "cancer_history", "depression",
]
DEMOGRAPHICS = ["sex", "age", "bmi"]
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Support Devices",
]


def compute_weighted_or(valid, a, w):
    """Weighted 2x2 OR with Haldane correction."""
    d = valid
    tp = w[(a == 1) & (d == 1)].sum()
    fp = w[(a == 1) & (d == 0)].sum()
    fn = w[(a == 0) & (d == 1)].sum()
    tn = w[(a == 0) & (d == 0)].sum()
    if min(tp, fp, fn, tn) < 0.5:
        tp += 0.5; fp += 0.5; fn += 0.5; tn += 0.5
    return (tp * tn) / (fp * fn)


def compute_unweighted_or(a, d):
    """Unweighted 2x2 OR with Haldane correction."""
    tp = ((a == 1) & (d == 1)).sum()
    fp = ((a == 1) & (d == 0)).sum()
    fn = ((a == 0) & (d == 1)).sum()
    tn = ((a == 0) & (d == 0)).sum()
    if min(tp, fp, fn, tn) == 0:
        tp += 0.5; fp += 0.5; fn += 0.5; tn += 0.5
    return (tp * tn) / (fp * fn)


def main():
    print("=" * 80)
    print("  EXPERIMENT 1: INVERSE-FREQUENCY WEIGHTED OR")
    print("  Testing whether repeated imaging inflates the OR->dependence relationship")
    print("=" * 80)

    meta = pd.read_csv(os.path.join(PROJECT, "full_mimic_cxr_metadata.csv"))
    resid = pd.read_csv(os.path.join(PROJECT, "results", "per_comorbidity_residualization.csv"))

    # Compute IPW weight
    img_per_patient = meta.groupby("subject_id").size()
    meta["ipw"] = meta["subject_id"].map(lambda s: 1.0 / img_per_patient[s])
    print(f"Weight range: [{meta.ipw.min():.4f}, {meta.ipw.max():.4f}]")
    print(f"Effective sample size (sum of weights): {meta.ipw.sum():.0f} (= n_patients)")

    train = meta[meta["split"] == "train"].copy()

    rows = []
    for attr in COMORBIDITIES + DEMOGRAPHICS:
        for disease in DISEASES:
            valid = train[train[disease].isin([0.0, 1.0])].copy()
            if len(valid) < 100:
                continue

            if attr == "sex":
                a = valid["gender_binary"].values
            elif attr == "age":
                a = (valid["age"] >= valid["age"].median()).astype(int).values
            elif attr == "bmi":
                bmi_valid = valid[valid["bmi"].notna()]
                if len(bmi_valid) < 100:
                    continue
                valid = bmi_valid
                a = (valid["bmi"] >= valid["bmi"].median()).astype(int).values
            else:
                a = valid[attr].values

            d = (valid[disease] == 1.0).astype(int).values
            w = valid["ipw"].values

            or_wt = compute_weighted_or(d, a, w)
            or_unwt = compute_unweighted_or(a, d)

            rows.append({
                "attribute": attr, "disease": disease,
                "or_weighted": or_wt,
                "abs_log_or_weighted": abs(np.log(or_wt)),
                "or_unweighted": or_unwt,
                "abs_log_or_unweighted": abs(np.log(or_unwt)),
            })

    wor_df = pd.DataFrame(rows)
    print(f"\nComputed {len(wor_df)} weighted OR pairs")

    # Compare weighted vs unweighted
    rho_or, p_or = spearmanr(wor_df["abs_log_or_weighted"], wor_df["abs_log_or_unweighted"])
    print(f"\nWeighted vs Unweighted OR correlation:")
    print(f"  Spearman rho = {rho_or:.4f}, p = {p_or:.2e}")
    delta = (wor_df["abs_log_or_weighted"] - wor_df["abs_log_or_unweighted"]).abs()
    print(f"  Mean |delta log(OR)| = {delta.mean():.4f}")

    # Top changes
    wor_df["or_change"] = wor_df["abs_log_or_weighted"] - wor_df["abs_log_or_unweighted"]
    print("\nTop 10 pairs where weighting changes OR most:")
    top_changes = wor_df.reindex(wor_df["or_change"].abs().nlargest(10).index)
    for _, r in top_changes.iterrows():
        print(f"  {r.attribute:25s} x {r.disease:25s}: "
              f"unwt={r.abs_log_or_unweighted:.4f} -> wt={r.abs_log_or_weighted:.4f} "
              f"(delta={r.or_change:+.4f})")

    # Merge with AUROC drops
    mean_drops = resid.groupby(["attribute", "disease"])["auroc_drop"].mean().reset_index()
    merged = mean_drops.merge(wor_df, on=["attribute", "disease"], how="inner")
    print(f"\nMerged for regression: {len(merged)} attribute-disease pairs")

    # Attribute-disease level regressions
    X_unwt = sm.add_constant(merged["abs_log_or_unweighted"])
    m_unwt = sm.OLS(merged["auroc_drop"], X_unwt).fit()

    X_wt = sm.add_constant(merged["abs_log_or_weighted"])
    m_wt = sm.OLS(merged["auroc_drop"], X_wt).fit()

    print("\n" + "=" * 80)
    print("  REGRESSION COMPARISON (attribute-disease level)")
    print("=" * 80)
    print(f"  Unweighted OR -> drop: R2 = {m_unwt.rsquared:.4f}, "
          f"beta = {m_unwt.params.iloc[1]:.6f}, p = {m_unwt.pvalues.iloc[1]:.2e}")
    print(f"  IPW-Weighted OR -> drop: R2 = {m_wt.rsquared:.4f}, "
          f"beta = {m_wt.params.iloc[1]:.6f}, p = {m_wt.pvalues.iloc[1]:.2e}")
    print(f"  delta R2 = {m_wt.rsquared - m_unwt.rsquared:+.4f}")

    # Full model-level regression
    full_merged = resid.merge(
        wor_df[["attribute", "disease", "abs_log_or_weighted", "abs_log_or_unweighted"]],
        on=["attribute", "disease"], how="inner"
    )
    print(f"\n  Full model-level dataset: {len(full_merged)} observations")

    X_fu = sm.add_constant(full_merged["abs_log_or_unweighted"])
    m_fu = sm.OLS(full_merged["auroc_drop"], X_fu).fit()

    X_fw = sm.add_constant(full_merged["abs_log_or_weighted"])
    m_fw = sm.OLS(full_merged["auroc_drop"], X_fw).fit()

    print(f"  Unweighted OR (model-level): R2 = {m_fu.rsquared:.4f}, "
          f"beta = {m_fu.params.iloc[1]:.6f}")
    print(f"  IPW-Weighted OR (model-level): R2 = {m_fw.rsquared:.4f}, "
          f"beta = {m_fw.params.iloc[1]:.6f}")
    print(f"  delta R2 = {m_fw.rsquared - m_fu.rsquared:+.4f}")

    # Rank stability
    attr_rank_unwt = wor_df.groupby("attribute")["abs_log_or_unweighted"].mean().rank(ascending=False)
    attr_rank_wt = wor_df.groupby("attribute")["abs_log_or_weighted"].mean().rank(ascending=False)
    rho_rank, p_rank = spearmanr(attr_rank_unwt, attr_rank_wt)
    print(f"\n  Attribute rank stability (weighted vs unweighted): "
          f"rho = {rho_rank:.4f}, p = {p_rank:.2e}")

    # Save
    out = os.path.join(PROJECT, "results", "experiment1_ipw_or_comparison.csv")
    wor_df.to_csv(out, index=False, float_format="%.6f")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
