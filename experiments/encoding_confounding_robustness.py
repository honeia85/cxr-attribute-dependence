"""
Encoding vs Confounding: Robustness Analyses.

Two robustness checks on the encoding-vs-confounding classification:
1. Threshold sensitivity — sweep cutoffs and show classification stability
2. Continuous correlation — Spearman rho between encoding and confounding
   (no arbitrary thresholds needed)

Usage:
    cd cxr-metadata-study
    PYTHONPATH=. python experiments/encoding_confounding_robustness.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")

N_BOOTSTRAP = 10000


def classify_pattern(enc_metric, enc_val, conf_val, enc_thresh_auroc, enc_thresh_r2, conf_thresh):
    """Classify into 4-pattern given thresholds."""
    if enc_metric == "auroc":
        high_enc = enc_val > enc_thresh_auroc
    else:
        high_enc = enc_val > enc_thresh_r2
    high_conf = conf_val > conf_thresh

    if high_enc and not high_conf:
        return "ORTHOGONAL"
    elif high_enc and high_conf:
        return "Entangled"
    elif not high_enc and high_conf:
        return "Hidden conf."
    else:
        return "Irrelevant"


def threshold_sensitivity(df):
    """
    Strategy 1: Sweep encoding and confounding thresholds.
    Show how many attributes change pattern across a grid of cutoffs.
    """
    print("=" * 80)
    print("  STRATEGY 1: THRESHOLD SENSITIVITY ANALYSIS")
    print("  How stable is the 4-pattern classification across different cutoffs?")
    print("=" * 80)

    # Define sweep ranges
    auroc_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    r2_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    conf_thresholds = [0.002, 0.003, 0.005, 0.007, 0.010]

    # Reference classification (original: AUROC>0.7, R²>0.1, drop>0.005)
    ref_auroc, ref_r2, ref_conf = 0.70, 0.10, 0.005

    ref_patterns = {}
    for _, row in df.iterrows():
        ref_patterns[row["attribute"]] = classify_pattern(
            row["encoding_metric"], row["encoding_value"],
            row["confounding_drop"], ref_auroc, ref_r2, ref_conf)

    # -- 1a: Vary encoding threshold (fix confounding) --------------------
    print(f"\n  1a. Varying ENCODING threshold (confounding fixed at {ref_conf})")
    print(f"  {'AUROC thresh':>14s}  {'R² thresh':>10s}  {'ORTH':>5s}  {'Entangled':>9s}  "
          f"{'Hidden':>7s}  {'Irrel':>6s}  {'Changed':>8s}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*8}")

    enc_sweep_rows = []
    for auroc_t, r2_t in zip(auroc_thresholds, r2_thresholds):
        patterns = {}
        for _, row in df.iterrows():
            patterns[row["attribute"]] = classify_pattern(
                row["encoding_metric"], row["encoding_value"],
                row["confounding_drop"], auroc_t, r2_t, ref_conf)

        counts = {p: sum(1 for v in patterns.values() if v == p)
                  for p in ["ORTHOGONAL", "Entangled", "Hidden conf.", "Irrelevant"]}
        n_changed = sum(1 for attr in patterns if patterns[attr] != ref_patterns[attr])

        marker = " <- reference" if auroc_t == ref_auroc else ""
        print(f"  {auroc_t:>14.2f}  {r2_t:>10.2f}  {counts['ORTHOGONAL']:>5d}  "
              f"{counts['Entangled']:>9d}  {counts['Hidden conf.']:>7d}  "
              f"{counts['Irrelevant']:>6d}  {n_changed:>8d}{marker}")

        enc_sweep_rows.append({
            "sweep_type": "encoding",
            "auroc_thresh": auroc_t,
            "r2_thresh": r2_t,
            "conf_thresh": ref_conf,
            "n_orthogonal": counts["ORTHOGONAL"],
            "n_entangled": counts["Entangled"],
            "n_hidden": counts["Hidden conf."],
            "n_irrelevant": counts["Irrelevant"],
            "n_changed_from_ref": n_changed,
        })

    # -- 1b: Vary confounding threshold (fix encoding) --------------------
    print(f"\n  1b. Varying CONFOUNDING threshold (encoding fixed at AUROC>{ref_auroc}, R²>{ref_r2})")
    print(f"  {'Conf thresh':>14s}  {'ORTH':>5s}  {'Entangled':>9s}  "
          f"{'Hidden':>7s}  {'Irrel':>6s}  {'Changed':>8s}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*8}")

    conf_sweep_rows = []
    for conf_t in conf_thresholds:
        patterns = {}
        for _, row in df.iterrows():
            patterns[row["attribute"]] = classify_pattern(
                row["encoding_metric"], row["encoding_value"],
                row["confounding_drop"], ref_auroc, ref_r2, conf_t)

        counts = {p: sum(1 for v in patterns.values() if v == p)
                  for p in ["ORTHOGONAL", "Entangled", "Hidden conf.", "Irrelevant"]}
        n_changed = sum(1 for attr in patterns if patterns[attr] != ref_patterns[attr])

        marker = " <- reference" if conf_t == ref_conf else ""
        print(f"  {conf_t:>14.3f}  {counts['ORTHOGONAL']:>5d}  "
              f"{counts['Entangled']:>9d}  {counts['Hidden conf.']:>7d}  "
              f"{counts['Irrelevant']:>6d}  {n_changed:>8d}{marker}")

        conf_sweep_rows.append({
            "sweep_type": "confounding",
            "auroc_thresh": ref_auroc,
            "r2_thresh": ref_r2,
            "conf_thresh": conf_t,
            "n_orthogonal": counts["ORTHOGONAL"],
            "n_entangled": counts["Entangled"],
            "n_hidden": counts["Hidden conf."],
            "n_irrelevant": counts["Irrelevant"],
            "n_changed_from_ref": n_changed,
        })

    # -- 1c: Per-attribute stability --------------------------------------
    print(f"\n  1c. Per-attribute stability (how often does each attribute change pattern?)")
    total_combos = len(auroc_thresholds) * len(conf_thresholds)
    print(f"  Testing {total_combos} threshold combinations")
    print(f"  {'Attribute':>25s}  {'Ref pattern':>15s}  {'Stability':>10s}  {'Most common':>15s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*10}  {'-'*15}")

    stability_rows = []
    for _, row in df.iterrows():
        attr = row["attribute"]
        all_patterns = []
        for auroc_t, r2_t in zip(auroc_thresholds, r2_thresholds):
            for conf_t in conf_thresholds:
                p = classify_pattern(
                    row["encoding_metric"], row["encoding_value"],
                    row["confounding_drop"], auroc_t, r2_t, conf_t)
                all_patterns.append(p)

        ref_pat = ref_patterns[attr]
        n_same = sum(1 for p in all_patterns if p == ref_pat)
        stability = n_same / len(all_patterns)
        most_common = max(set(all_patterns), key=all_patterns.count)

        print(f"  {attr:>25s}  {ref_pat:>15s}  {stability:>9.0%}  {most_common:>15s}")

        stability_rows.append({
            "attribute": attr,
            "reference_pattern": ref_pat,
            "stability": stability,
            "most_common_pattern": most_common,
            "n_combinations": total_combos,
        })

    return enc_sweep_rows + conf_sweep_rows, stability_rows


def continuous_correlation(df):
    """
    Strategy 2: Spearman correlation between encoding and confounding.
    No thresholds needed — treats both as continuous variables.
    """
    print("\n" + "=" * 80)
    print("  STRATEGY 2: CONTINUOUS CORRELATION ANALYSIS")
    print("  Spearman rho between encoding strength and confounding cost")
    print("  (No arbitrary thresholds)")
    print("=" * 80)

    rng = np.random.RandomState(42)

    # Separate binary and continuous attributes
    binary_mask = df["encoding_metric"] == "auroc"
    cont_mask = df["encoding_metric"] == "r2"

    results = []

    for subset_name, mask in [("All (23 attrs)", slice(None)),
                               ("Binary only (21 attrs)", binary_mask),
                               ("Continuous only (2 attrs)", cont_mask)]:
        sub = df[mask].copy()
        if len(sub) < 3:
            print(f"\n  {subset_name}: too few observations for correlation")
            continue

        enc = sub["encoding_value"].values
        conf = sub["confounding_drop"].values

        # Spearman correlation
        rho, pval = stats.spearmanr(enc, conf)

        # Pearson on ranks (same as Spearman, for verification)
        pearson_r, pearson_p = stats.pearsonr(enc, conf)

        # Bootstrap CI for Spearman
        boot_rhos = []
        n = len(enc)
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n, size=n, replace=True)
            if len(np.unique(idx)) < 3:
                continue
            r, _ = stats.spearmanr(enc[idx], conf[idx])
            if np.isfinite(r):
                boot_rhos.append(r)

        boot_rhos = np.array(boot_rhos)
        ci_lo = np.percentile(boot_rhos, 2.5) if len(boot_rhos) > 0 else np.nan
        ci_hi = np.percentile(boot_rhos, 97.5) if len(boot_rhos) > 0 else np.nan

        print(f"\n  {subset_name}:")
        print(f"    Spearman rho = {rho:.4f}  (p = {pval:.4e})")
        print(f"    95% Bootstrap CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"    Pearson r = {pearson_r:.4f}  (p = {pearson_p:.4e})")

        results.append({
            "subset": subset_name,
            "n": len(sub),
            "spearman_rho": rho,
            "spearman_p": pval,
            "bootstrap_ci_lo": ci_lo,
            "bootstrap_ci_hi": ci_hi,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        })

    # -- Rank-based analysis ----------------------------------------------
    print(f"\n  Rank comparison (encoding rank vs confounding rank):")
    df_ranked = df.copy()
    # For binary: higher AUROC = higher encoding rank
    # For continuous: higher R² = higher encoding rank
    df_ranked["enc_rank"] = df_ranked["encoding_value"].rank(ascending=False)
    df_ranked["conf_rank"] = df_ranked["confounding_drop"].rank(ascending=False)

    print(f"  {'Attribute':>25s}  {'Enc rank':>9s}  {'Conf rank':>10s}  {'Diff':>6s}")
    print(f"  {'-'*25}  {'-'*9}  {'-'*10}  {'-'*6}")

    for _, row in df_ranked.sort_values("enc_rank").iterrows():
        diff = abs(row["enc_rank"] - row["conf_rank"])
        marker = "  <- big gap" if diff >= 10 else ""
        print(f"  {row['attribute']:>25s}  {row['enc_rank']:>9.0f}  "
              f"{row['conf_rank']:>10.0f}  {diff:>5.0f}{marker}")

    # -- Key dissociation examples ----------------------------------------
    print(f"\n  Key dissociation examples (encoding rank vs confounding rank):")

    # Biggest positive gap (high encoding, low confounding)
    df_ranked["gap"] = df_ranked["conf_rank"] - df_ranked["enc_rank"]
    top_orthogonal = df_ranked.nlargest(3, "gap")
    print(f"\n    HIGH encoding, LOW confounding (ORTHOGONAL):")
    for _, row in top_orthogonal.iterrows():
        print(f"      {row['attribute']:20s}: enc rank #{row['enc_rank']:.0f}, "
              f"conf rank #{row['conf_rank']:.0f} (gap={row['gap']:+.0f})")

    # Biggest negative gap (low encoding, high confounding)
    top_hidden = df_ranked.nsmallest(3, "gap")
    print(f"\n    LOW encoding, HIGH confounding (HIDDEN CONFOUNDER):")
    for _, row in top_hidden.iterrows():
        print(f"      {row['attribute']:20s}: enc rank #{row['enc_rank']:.0f}, "
              f"conf rank #{row['conf_rank']:.0f} (gap={row['gap']:+.0f})")

    return results, df_ranked[["attribute", "encoding_metric", "encoding_value",
                                "confounding_drop", "enc_rank", "conf_rank", "gap"]].to_dict("records")


def main():
    # Load data
    enc_conf_path = os.path.join(RESULTS_DIR, "encoding_vs_confounding.csv")
    if not os.path.exists(enc_conf_path):
        print(f"ERROR: {enc_conf_path} not found. Run comorbidity_encoding_auroc.py first.")
        sys.exit(1)

    df = pd.read_csv(enc_conf_path)
    print(f"Loaded {len(df)} attributes from {enc_conf_path}\n")

    # Strategy 1
    sweep_rows, stability_rows = threshold_sensitivity(df)

    # Strategy 2
    corr_results, rank_rows = continuous_correlation(df)

    # -- Summary ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("  COMBINED SUMMARY")
    print("=" * 80)

    # Strategy 1 summary
    stab_df = pd.DataFrame(stability_rows)
    mean_stab = stab_df["stability"].mean()
    min_stab = stab_df["stability"].min()
    min_attr = stab_df.loc[stab_df["stability"].idxmin(), "attribute"]
    n_always_stable = (stab_df["stability"] == 1.0).sum()

    print(f"\n  Strategy 1 (Threshold Sensitivity):")
    print(f"    Mean stability: {mean_stab:.1%}")
    print(f"    Always stable: {n_always_stable}/{len(stab_df)} attributes")
    print(f"    Least stable: {min_attr} ({min_stab:.0%})")
    print(f"    Classification is {'ROBUST' if mean_stab > 0.7 else 'SENSITIVE'} to threshold choice")

    # Strategy 2 summary
    if corr_results:
        all_rho = corr_results[0]  # "All" subset
        print(f"\n  Strategy 2 (Continuous Correlation):")
        print(f"    Spearman rho = {all_rho['spearman_rho']:.3f} "
              f"[{all_rho['bootstrap_ci_lo']:.3f}, {all_rho['bootstrap_ci_hi']:.3f}]")
        print(f"    p = {all_rho['spearman_p']:.4e}")

        if all_rho['spearman_p'] < 0.05:
            direction = "positive" if all_rho['spearman_rho'] > 0 else "negative"
            print(f"    Significant {direction} correlation: encoding and confounding "
                  f"are {'partially aligned' if all_rho['spearman_rho'] > 0 else 'inversely related'}")
        else:
            print(f"    No significant correlation: encoding and confounding are decoupled")

        print(f"    This {'SUPPORTS' if all_rho['spearman_rho'] < 0.7 else 'WEAKENS'} "
              f"the dissociation argument (rho < 0.7 = imperfect coupling)")

    # -- Save -------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = os.path.join(RESULTS_DIR, "encoding_confounding_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n  Saved: {sweep_path}")

    stab_path = os.path.join(RESULTS_DIR, "encoding_confounding_stability.csv")
    stab_df.to_csv(stab_path, index=False, float_format="%.4f")
    print(f"  Saved: {stab_path}")

    corr_df = pd.DataFrame(corr_results)
    corr_path = os.path.join(RESULTS_DIR, "encoding_confounding_correlation.csv")
    corr_df.to_csv(corr_path, index=False, float_format="%.6f")
    print(f"  Saved: {corr_path}")

    rank_df = pd.DataFrame(rank_rows)
    rank_path = os.path.join(RESULTS_DIR, "encoding_confounding_ranks.csv")
    rank_df.to_csv(rank_path, index=False, float_format="%.6f")
    print(f"  Saved: {rank_path}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
