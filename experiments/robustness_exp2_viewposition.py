"""
Experiment 2: View position adjusted OR analysis.
Tests whether AP/PA confounding drives the OR->dependence relationship.
Computes Mantel-Haenszel OR stratified by view position.
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


def get_attribute_binary(valid, attr):
    """Get binary attribute column, return (values, subset_df)."""
    if attr == "sex":
        return valid["gender_binary"].values, valid
    elif attr == "age":
        return (valid["age"] >= valid["age"].median()).astype(int).values, valid
    elif attr == "bmi":
        bmi_valid = valid[valid["bmi"].notna()]
        if len(bmi_valid) < 50:
            return None, None
        return (bmi_valid["bmi"] >= bmi_valid["bmi"].median()).astype(int).values, bmi_valid
    else:
        return valid[attr].values, valid


def main():
    print("=" * 80)
    print("  EXPERIMENT 2: VIEW POSITION ADJUSTED OR ANALYSIS")
    print("  Testing whether AP/PA confounding drives the OR->dependence relationship")
    print("=" * 80)

    meta = pd.read_csv(os.path.join(PROJECT, "full_mimic_cxr_metadata.csv"))
    resid = pd.read_csv(os.path.join(PROJECT, "results", "per_comorbidity_residualization.csv"))
    train = meta[meta["split"] == "train"].copy()

    results = []
    for attr in COMORBIDITIES + DEMOGRAPHICS:
        for disease in DISEASES:
            strata = {}
            for view in ["AP", "PA"]:
                valid = train[(train["ViewPosition"] == view) & train[disease].isin([0.0, 1.0])].copy()
                if len(valid) < 50:
                    continue
                a, valid_sub = get_attribute_binary(valid, attr)
                if a is None or len(valid_sub) < 50:
                    continue
                d = (valid_sub[disease] == 1.0).astype(int).values

                tp = ((a == 1) & (d == 1)).sum()
                fp = ((a == 1) & (d == 0)).sum()
                fn = ((a == 0) & (d == 1)).sum()
                tn = ((a == 0) & (d == 0)).sum()
                strata[view] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(valid_sub)}

            if len(strata) < 2:
                continue

            # Stratum-specific ORs
            or_per_view = {}
            for view, tab in strata.items():
                tp = tab["tp"] + 0.5
                fp = tab["fp"] + 0.5
                fn = tab["fn"] + 0.5
                tn = tab["tn"] + 0.5
                or_per_view[view] = (tp * tn) / (fp * fn)

            # Mantel-Haenszel OR
            mh_num = sum((tab["tp"] * tab["tn"]) / tab["n"] for tab in strata.values())
            mh_den = sum((tab["fp"] * tab["fn"]) / tab["n"] for tab in strata.values())
            or_mh = mh_num / mh_den if mh_den > 0 else np.nan

            # Crude OR
            tp_c = sum(s["tp"] for s in strata.values())
            fp_c = sum(s["fp"] for s in strata.values())
            fn_c = sum(s["fn"] for s in strata.values())
            tn_c = sum(s["tn"] for s in strata.values())
            if min(tp_c, fp_c, fn_c, tn_c) == 0:
                tp_c += 0.5; fp_c += 0.5; fn_c += 0.5; tn_c += 0.5
            or_crude = (tp_c * tn_c) / (fp_c * fn_c)

            def safe_abs_log(x):
                return abs(np.log(x)) if x > 0 and np.isfinite(x) else np.nan

            results.append({
                "attribute": attr, "disease": disease,
                "or_crude": or_crude,
                "abs_log_or_crude": safe_abs_log(or_crude),
                "or_mh": or_mh,
                "abs_log_or_mh": safe_abs_log(or_mh),
                "or_ap": or_per_view.get("AP", np.nan),
                "or_pa": or_per_view.get("PA", np.nan),
                "abs_log_or_ap": safe_abs_log(or_per_view.get("AP", np.nan)),
                "abs_log_or_pa": safe_abs_log(or_per_view.get("PA", np.nan)),
            })

    vp_df = pd.DataFrame(results)
    print(f"Computed {len(vp_df)} attribute-disease pairs with stratified ORs")

    # AP vs PA consistency
    valid_both = vp_df.dropna(subset=["abs_log_or_ap", "abs_log_or_pa"])
    rho, p = spearmanr(valid_both["abs_log_or_ap"], valid_both["abs_log_or_pa"])
    print(f"\nAP vs PA OR consistency:")
    print(f"  Spearman rho = {rho:.4f}, p = {p:.2e}")
    delta_ap_pa = (valid_both["abs_log_or_ap"] - valid_both["abs_log_or_pa"]).abs()
    print(f"  Mean |delta log(OR)|: {delta_ap_pa.mean():.4f}")

    # Crude vs MH
    valid_mh = vp_df.dropna(subset=["abs_log_or_mh"])
    rho2, p2 = spearmanr(valid_mh["abs_log_or_crude"], valid_mh["abs_log_or_mh"])
    print(f"\nCrude vs MH-adjusted OR:")
    print(f"  Spearman rho = {rho2:.4f}, p = {p2:.2e}")
    delta_mh = (valid_mh["abs_log_or_crude"] - valid_mh["abs_log_or_mh"]).abs()
    print(f"  Mean |delta log(OR)|: {delta_mh.mean():.4f}")

    # Top changes
    valid_mh = valid_mh.copy()
    valid_mh["delta_mh"] = valid_mh["abs_log_or_mh"] - valid_mh["abs_log_or_crude"]
    print("\nTop 10 pairs where MH adjustment changes OR most:")
    top_mh = valid_mh.reindex(valid_mh["delta_mh"].abs().nlargest(10).index)
    for _, r in top_mh.iterrows():
        print(f"  {r.attribute:25s} x {r.disease:25s}: "
              f"crude={r.abs_log_or_crude:.4f} -> MH={r.abs_log_or_mh:.4f} "
              f"(delta={r.delta_mh:+.4f})")

    # Regressions
    mean_drops = resid.groupby(["attribute", "disease"])["auroc_drop"].mean().reset_index()
    merged = mean_drops.merge(vp_df, on=["attribute", "disease"], how="inner")
    merged = merged.dropna(subset=["abs_log_or_mh"])
    print(f"\nMerged for regression: {len(merged)} pairs")

    X_crude = sm.add_constant(merged["abs_log_or_crude"])
    m_crude = sm.OLS(merged["auroc_drop"], X_crude).fit()

    X_mh = sm.add_constant(merged["abs_log_or_mh"])
    m_mh = sm.OLS(merged["auroc_drop"], X_mh).fit()

    valid_ap_reg = merged.dropna(subset=["abs_log_or_ap"])
    X_ap = sm.add_constant(valid_ap_reg["abs_log_or_ap"])
    m_ap = sm.OLS(valid_ap_reg["auroc_drop"], X_ap).fit()

    valid_pa_reg = merged.dropna(subset=["abs_log_or_pa"])
    X_pa = sm.add_constant(valid_pa_reg["abs_log_or_pa"])
    m_pa = sm.OLS(valid_pa_reg["auroc_drop"], X_pa).fit()

    print("\n" + "=" * 80)
    print("  REGRESSION COMPARISON (attribute-disease level)")
    print("=" * 80)
    print(f"  Crude OR -> drop:       R2 = {m_crude.rsquared:.4f}, "
          f"beta = {m_crude.params.iloc[1]:.6f}, p = {m_crude.pvalues.iloc[1]:.2e}")
    print(f"  MH-adjusted OR -> drop: R2 = {m_mh.rsquared:.4f}, "
          f"beta = {m_mh.params.iloc[1]:.6f}, p = {m_mh.pvalues.iloc[1]:.2e}")
    print(f"  AP-only OR -> drop:     R2 = {m_ap.rsquared:.4f}, "
          f"beta = {m_ap.params.iloc[1]:.6f}, p = {m_ap.pvalues.iloc[1]:.2e}")
    print(f"  PA-only OR -> drop:     R2 = {m_pa.rsquared:.4f}, "
          f"beta = {m_pa.params.iloc[1]:.6f}, p = {m_pa.pvalues.iloc[1]:.2e}")
    print(f"\n  delta R2 (MH vs Crude): {m_mh.rsquared - m_crude.rsquared:+.4f}")

    # Full model-level
    full = resid.merge(
        vp_df[["attribute", "disease", "abs_log_or_crude", "abs_log_or_mh"]],
        on=["attribute", "disease"], how="inner"
    ).dropna(subset=["abs_log_or_mh"])
    print(f"\n  Full model-level dataset: {len(full)} observations")

    X_fc = sm.add_constant(full["abs_log_or_crude"])
    m_fc = sm.OLS(full["auroc_drop"], X_fc).fit()
    X_fm = sm.add_constant(full["abs_log_or_mh"])
    m_fm = sm.OLS(full["auroc_drop"], X_fm).fit()

    print(f"  Crude OR (model-level):  R2 = {m_fc.rsquared:.4f}")
    print(f"  MH-adj OR (model-level): R2 = {m_fm.rsquared:.4f}")
    print(f"  delta R2 = {m_fm.rsquared - m_fc.rsquared:+.4f}")

    # Rank stability
    rank_crude = vp_df.groupby("attribute")["abs_log_or_crude"].mean().rank(ascending=False)
    rank_mh = vp_df.dropna(subset=["abs_log_or_mh"]).groupby("attribute")["abs_log_or_mh"].mean().rank(ascending=False)
    common = rank_crude.index.intersection(rank_mh.index)
    rho_r, p_r = spearmanr(rank_crude[common], rank_mh[common])
    print(f"\n  Attribute rank stability (crude vs MH): rho = {rho_r:.4f}, p = {p_r:.2e}")

    out = os.path.join(PROJECT, "results", "experiment2_view_position_or.csv")
    vp_df.to_csv(out, index=False, float_format="%.6f")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
