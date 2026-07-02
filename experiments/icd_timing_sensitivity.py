"""
ICD timing sensitivity: stratified OR -> dependence regression on
imaging-adjudicable vs non-adjudicable attributes.

Motivation: MIMIC-IV ICD codes are assigned at discharge, so codes for
CXR-adjudicable conditions (e.g., heart failure) may be partly informed by
the index CXR itself. If this circularity drove our OR -> dependence result,
the slope should be much larger on adjudicable attributes than on
non-adjudicable ones.

Classification:
  ADJUDICABLE (5): conditions where a CXR can meaningfully contribute to the
    diagnosis being coded on this admission
      heart_failure, respiratory_failure, pulmonary_fibrosis, copd,
      cancer_history
  NON_ADJUDICABLE (19): demographics and conditions where the CXR plays no
    role in ICD assignment (diagnosis based on blood test, ECG, BP, BMI,
    psychiatric assessment, brain imaging, etc.)
      age, sex, race_black_vs_white, bmi,
      aki, anemia, asthma, atrial_fibrillation, ckd, coronary_artery_disease,
      depression, diabetes, hyperlipidemia, hypertension, hypothyroidism,
      liver_disease, obesity, smoking_history, stroke

Output:
  results/icd_timing_sensitivity.csv       (stratified regression summary)
  results/icd_timing_sensitivity_full.csv  (per-group OLS + clustered SEs)

Usage:
    PYTHONPATH=. python experiments/icd_timing_sensitivity.py
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ADJUDICABLE = {
    "heart_failure",
    "respiratory_failure",
    "pulmonary_fibrosis",
    "copd",
    "cancer_history",
}

DATA_PATH = "results/two_factor_regression_data_with_race.csv"
OUT_SUMMARY = "results/icd_timing_sensitivity.csv"
OUT_FULL = "results/icd_timing_sensitivity_full.csv"


def fit_regression(df, label):
    X = sm.add_constant(df["abs_log_or"].values)
    y = df["auroc_drop"].values
    ols = sm.OLS(y, X).fit()
    ols_attr = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["attribute"].values},
    )
    ols_find = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["disease"].values},
    )
    beta = ols.params[1]
    r2 = ols.rsquared
    return {
        "group": label,
        "n_attributes": df["attribute"].nunique(),
        "n_observations": len(df),
        "beta": beta,
        "intercept": ols.params[0],
        "r2": r2,
        "p_ols": ols.pvalues[1],
        "se_ols": ols.bse[1],
        "p_cluster_attr": ols_attr.pvalues[1],
        "se_cluster_attr": ols_attr.bse[1],
        "p_cluster_find": ols_find.pvalues[1],
        "se_cluster_find": ols_find.bse[1],
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["adjudicable"] = df["attribute"].isin(ADJUDICABLE)
    print(f"Total observations: {len(df)}")
    print(f"Adjudicable attributes: {sorted(df.loc[df['adjudicable'],'attribute'].unique())}")
    print(f"Non-adjudicable attributes: "
          f"{sorted(df.loc[~df['adjudicable'],'attribute'].unique())}")

    rows = [
        fit_regression(df, "all"),
        fit_regression(df[df["adjudicable"]], "adjudicable"),
        fit_regression(df[~df["adjudicable"]], "non_adjudicable"),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    # Slope equality test: pooled model with interaction term
    sub = df.copy()
    sub["adj_int"] = sub["adjudicable"].astype(int)
    sub["or_x_adj"] = sub["abs_log_or"] * sub["adj_int"]
    X = sm.add_constant(sub[["abs_log_or", "adj_int", "or_x_adj"]].values)
    y = sub["auroc_drop"].values
    pooled = sm.OLS(y, X).fit()
    pooled_cluster = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["attribute"].values},
    )
    slope_diff = pooled.params[3]
    slope_diff_p = pooled.pvalues[3]
    slope_diff_p_cluster = pooled_cluster.pvalues[3]
    print(f"\nSlope difference (adj - nonadj): {slope_diff:+.5f}")
    print(f"  OLS p = {slope_diff_p:.4g}")
    print(f"  Attribute-cluster-robust p = {slope_diff_p_cluster:.4g}")

    full_rows = []
    for r in rows:
        full_rows.append(r)
    full_rows.append({
        "group": "slope_diff_test",
        "n_attributes": 24,
        "n_observations": len(sub),
        "beta": slope_diff,
        "intercept": None,
        "r2": pooled.rsquared,
        "p_ols": slope_diff_p,
        "se_ols": pooled.bse[3],
        "p_cluster_attr": slope_diff_p_cluster,
        "se_cluster_attr": pooled_cluster.bse[3],
        "p_cluster_find": None,
        "se_cluster_find": None,
    })
    pd.DataFrame(full_rows).to_csv(OUT_FULL, index=False)

    print("\n--- Stratified OR -> dependence regression ---")
    for r in rows:
        print(f"{r['group']:16s}  n={r['n_observations']:4d}  "
              f"beta={r['beta']:+.4f}  R^2={r['r2']:.3f}  "
              f"p_clust_attr={r['p_cluster_attr']:.3g}")

    print(f"\nResults: {OUT_SUMMARY}")
    print(f"Full:    {OUT_FULL}")


if __name__ == "__main__":
    main()
