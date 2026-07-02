"""
W2 Defense: Proof that debiasing high-OR attributes reduces fairness gaps.

Compares subgroup AUROC gaps (Gender, Age) before and after residualizing
the top confounders identified by our framework.

Conditions:
  1. Baseline (no debiasing)
  2. Top-3 debiased (heart_failure, atrial_fibrillation, age)
  3. All confounders (entangled + hidden = 9 attributes)

For each condition, measures:
  - Per-subgroup AUROC (Gender: Female/Male; Age: <50, 50-70, 70+)
  - AUROC gap (max - min across subgroups)
  - Gap reduction relative to baseline

Uses 6 strict clean models x 10 diseases = 60 observations per condition.

Usage:
    cd cxr-metadata-study
    PYTHONPATH=. python experiments/fairness_debiasing_proof.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import SEED, SUBGROUPS
from experiments.data import (load_metadata, load_canonical_ids, load_split,
                              get_aligned_embeddings, merge_chexpert)

# -- Configuration ------------------------------------------------------
CLEAN_MODELS = [
    "ResNet50-ImageNet",
    "DINOv2-base",
    "BiomedCLIP",
    "XRV-DenseNet-nih",
    "CLIP-ViT-B16",
    "ConvNeXtV2-Base",
]

KEY_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Lung Opacity",
    "Pleural Effusion", "Pneumonia", "Pneumothorax", "Support Devices",
]

# Attribute groups (from encoding_vs_confounding classification)
TOP3 = ["heart_failure", "atrial_fibrillation", "age"]
ENTANGLED = [
    "heart_failure", "atrial_fibrillation", "coronary_artery_disease",
    "ckd", "aki", "hyperlipidemia", "age",
]
HIDDEN_CONF = ["diabetes", "anemia"]
ALL_CONFOUNDERS = ENTANGLED + HIDDEN_CONF  # 9 attributes

STRATEGIES = {
    "baseline":        [],
    "top3":            TOP3,
    "all_confounders": ALL_CONFOUNDERS,
}

ALPHA_RIDGE = 1.0
N_BOOTSTRAP = 1000


def get_attribute_vector(meta_df, attr_name, train_idx):
    """Get attribute vector, with train-fit normalization for continuous vars."""
    if attr_name == "sex":
        return meta_df["gender_binary"].values.astype(float)
    elif attr_name == "age":
        vals = meta_df["age"].values.astype(float)
        m, s = np.nanmean(vals[train_idx]), np.nanstd(vals[train_idx])
        return (vals - m) / s
    elif attr_name == "bmi":
        vals = meta_df["bmi"].values.astype(float)
        m, s = np.nanmean(vals[train_idx]), np.nanstd(vals[train_idx])
        return (vals - m) / s
    else:
        return meta_df[attr_name].values.astype(float)


def residualize(X_train, X_test, D_train, D_test, alpha=ALPHA_RIDGE):
    """Remove attribute signal from embeddings."""
    model = Ridge(alpha=alpha)
    model.fit(D_train, X_train)
    return X_train - model.predict(D_train), X_test - model.predict(D_test)


def assign_subgroup(values, dim_config):
    """Assign subgroup labels based on SUBGROUPS config."""
    if dim_config["bins"] is None:
        labels = dim_config["labels"]
        return np.array([labels[int(v)] if np.isfinite(v) else "missing"
                         for v in values])
    else:
        bins = dim_config["bins"]
        labels = dim_config["labels"]
        result = np.full(len(values), "missing", dtype=object)
        for i, v in enumerate(values):
            if not np.isfinite(v):
                continue
            for j in range(len(bins) - 1):
                if bins[j] <= v < bins[j + 1]:
                    result[i] = labels[j]
                    break
        return result


def subgroup_aurocs(X_train, y_train, X_test, y_test, group_labels_test, groups,
                    min_n=50, min_pos=15):
    """Train LR on full train, compute per-subgroup AUROC on test."""
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]

    results = {}
    for g in groups:
        mask = group_labels_test == g
        yt = y_test[mask]
        yp = prob[mask]
        n_pos = yt.sum()
        n_neg = len(yt) - n_pos
        if len(yt) < min_n or n_pos < min_pos or n_neg < min_pos:
            results[g] = np.nan
            continue
        results[g] = roc_auc_score(yt, yp)

    return results


def bootstrap_gap_reduction(gaps_baseline, gaps_debiased, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap CI for mean gap reduction (baseline - debiased)."""
    rng = np.random.RandomState(seed)
    reductions = gaps_baseline - gaps_debiased
    valid = np.isfinite(reductions)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    reductions = reductions[valid]
    boot_means = np.array([
        rng.choice(reductions, size=len(reductions), replace=True).mean()
        for _ in range(n_boot)
    ])
    return reductions.mean(), np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def main():
    t0 = time.time()

    print("=" * 80)
    print("  W2 DEFENSE: DEBIASING REDUCES FAIRNESS GAPS")
    print("  6 clean models x 10 diseases x 3 strategies x 2 dimensions")
    print("=" * 80)

    # -- Load data ------------------------------------------------------
    print("\n[1/3] Loading data...")
    canonical_ids = load_canonical_ids()
    split = load_split()
    metadata = load_metadata()
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    id_to_order = {did: i for i, did in enumerate(canonical_ids)}
    meta_df = metadata[metadata["dicom_id"].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df["dicom_id"].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)

    merged, labels, masks, all_diseases = merge_chexpert(meta_df)
    disease_indices = {d: all_diseases.index(d) for d in KEY_DISEASES if d in all_diseases}

    # Pre-compute all needed attribute vectors
    all_attr_names = list(set(TOP3 + ALL_CONFOUNDERS))
    attr_vectors = {}
    for attr in all_attr_names:
        attr_vectors[attr] = get_attribute_vector(meta_df, attr, train_idx)

    # Valid mask: all confounder attributes must be finite
    valid_all = np.ones(len(canonical_ids), dtype=bool)
    for attr in all_attr_names:
        valid_all &= np.isfinite(attr_vectors[attr])

    valid_tr = valid_all[train_idx]
    valid_te = valid_all[test_idx]
    tr = train_idx[valid_tr]
    te = test_idx[valid_te]

    print(f"  Dataset: {len(canonical_ids)} images")
    print(f"  Valid (all attrs finite): train={len(tr)}, test={len(te)}")

    # -- Prepare subgroup labels for test set --------------------------
    test_df = meta_df.iloc[te]
    dimensions = {}
    for dim_name, dim_config in SUBGROUPS.items():
        if dim_name == "BMI":
            continue  # Skip BMI — focus on Gender and Age
        col = dim_config["column"]
        values = test_df[col].values.astype(float)
        group_labels = assign_subgroup(values, dim_config)
        dimensions[dim_name] = {
            "labels": dim_config["labels"],
            "group_labels": group_labels,
        }

    print(f"  Dimensions: {list(dimensions.keys())}")
    for dim_name, dim_info in dimensions.items():
        gl = dim_info["group_labels"]
        non_missing = gl != "missing"
        print(f"    {dim_name}: {non_missing.sum()} valid, "
              f"groups = {dim_info['labels']}")

    # -- Main loop ------------------------------------------------------
    print(f"\n[2/3] Running experiment...")
    rows = []

    for model_name in CLEAN_MODELS:
        print(f"\n  MODEL: {model_name}")

        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler()
        scaler.fit(emb[train_idx])
        emb_scaled = scaler.transform(emb)

        X_tr_full = emb_scaled[tr]
        X_te_full = emb_scaled[te]

        for strategy_name, attrs_to_remove in STRATEGIES.items():
            if len(attrs_to_remove) == 0:
                X_tr_s, X_te_s = X_tr_full, X_te_full
            else:
                D_tr = np.column_stack([attr_vectors[a][tr] for a in attrs_to_remove])
                D_te = np.column_stack([attr_vectors[a][te] for a in attrs_to_remove])
                X_tr_s, X_te_s = residualize(X_tr_full, X_te_full, D_tr, D_te)

            for disease in KEY_DISEASES:
                if disease not in disease_indices:
                    continue
                didx = disease_indices[disease]
                mask_tr = masks[tr, didx].astype(bool)
                mask_te = masks[te, didx].astype(bool)
                y_tr = labels[tr, didx]
                y_te = labels[te, didx]

                if mask_tr.sum() < 50 or mask_te.sum() < 50:
                    continue
                if y_tr[mask_tr].sum() < 10 or y_te[mask_te].sum() < 10:
                    continue

                for dim_name, dim_info in dimensions.items():
                    gl_full = dim_info["group_labels"]
                    gl_disease = gl_full[mask_te]
                    non_missing = gl_disease != "missing"

                    if non_missing.sum() < 50:
                        continue

                    try:
                        sg_aurocs = subgroup_aurocs(
                            X_tr_s[mask_tr], y_tr[mask_tr],
                            X_te_s[mask_te][non_missing],
                            y_te[mask_te][non_missing],
                            gl_disease[non_missing],
                            dim_info["labels"])

                        valid_aurocs = [v for v in sg_aurocs.values() if np.isfinite(v)]
                        gap = max(valid_aurocs) - min(valid_aurocs) if len(valid_aurocs) >= 2 else np.nan

                        overall_auroc = roc_auc_score(
                            y_te[mask_te][non_missing],
                            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(
                                X_tr_s[mask_tr], y_tr[mask_tr]
                            ).predict_proba(X_te_s[mask_te][non_missing])[:, 1])

                        row = {
                            "model": model_name,
                            "strategy": strategy_name,
                            "n_attrs_removed": len(attrs_to_remove),
                            "disease": disease,
                            "dimension": dim_name,
                            "overall_auroc": overall_auroc,
                            "auroc_gap": gap,
                        }
                        for g, auroc in sg_aurocs.items():
                            row[f"auroc_{g}"] = auroc

                        rows.append(row)
                    except Exception as e:
                        pass

            print(f"    {strategy_name:20s} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)

    # -- Analysis -------------------------------------------------------
    print(f"\n[3/3] Analyzing gap reductions...")

    # Merge baseline gaps for comparison
    baseline = df[df["strategy"] == "baseline"][
        ["model", "disease", "dimension", "auroc_gap", "overall_auroc"]
    ].rename(columns={"auroc_gap": "gap_baseline", "overall_auroc": "auroc_baseline"})

    compare = df[df["strategy"] != "baseline"].merge(
        baseline, on=["model", "disease", "dimension"], how="left")
    compare["gap_reduction"] = compare["gap_baseline"] - compare["auroc_gap"]
    compare["gap_reduction_pct"] = compare["gap_reduction"] / compare["gap_baseline"] * 100
    compare["auroc_cost"] = compare["auroc_baseline"] - compare["overall_auroc"]

    # -- Summary tables ------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  FAIRNESS GAP REDUCTION SUMMARY")
    print(f"  (positive = gap decreased = fairer)")
    print(f"{'='*90}")

    for dim_name in dimensions:
        print(f"\n  --- {dim_name} ---")
        dim_compare = compare[compare["dimension"] == dim_name]

        for strat in ["top3", "all_confounders"]:
            strat_df = dim_compare[dim_compare["strategy"] == strat]
            valid = strat_df.dropna(subset=["gap_reduction"])

            if len(valid) == 0:
                continue

            # Bootstrap CI for mean gap reduction
            mean_red, ci_lo, ci_hi = bootstrap_gap_reduction(
                valid["gap_baseline"].values, valid["auroc_gap"].values)

            mean_gap_bl = valid["gap_baseline"].mean()
            mean_gap_after = valid["auroc_gap"].mean()
            mean_cost = valid["auroc_cost"].mean()
            pct_improved = (valid["gap_reduction"] > 0).mean() * 100

            n_attrs = valid["n_attrs_removed"].iloc[0]
            print(f"\n  Strategy: {strat} ({n_attrs} attrs)")
            print(f"    Mean gap baseline:    {mean_gap_bl:.4f}")
            print(f"    Mean gap after:       {mean_gap_after:.4f}")
            print(f"    Mean gap reduction:   {mean_red:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")
            print(f"    Gap reduced in:       {pct_improved:.0f}% of cases ({int((valid['gap_reduction']>0).sum())}/{len(valid)})")
            print(f"    Mean AUROC cost:      {mean_cost:+.4f}")

    # -- Per-model summary ---------------------------------------------
    print(f"\n{'='*90}")
    print(f"  PER-MODEL GAP REDUCTION (top3 strategy, averaged across diseases & dimensions)")
    print(f"{'='*90}")

    top3_compare = compare[compare["strategy"] == "top3"]
    for model_name in CLEAN_MODELS:
        m_df = top3_compare[top3_compare["model"] == model_name].dropna(subset=["gap_reduction"])
        if len(m_df) > 0:
            mean_red = m_df["gap_reduction"].mean()
            mean_cost = m_df["auroc_cost"].mean()
            pct_improved = (m_df["gap_reduction"] > 0).mean() * 100
            print(f"  {model_name:25s}: gap_reduction={mean_red:+.4f}, "
                  f"auroc_cost={mean_cost:+.4f}, improved={pct_improved:.0f}%")

    # -- Per-disease summary (top3) ------------------------------------
    print(f"\n{'='*90}")
    print(f"  PER-DISEASE GAP REDUCTION (top3 strategy, Gender dimension)")
    print(f"{'='*90}")

    top3_gender = top3_compare[(top3_compare["dimension"] == "Gender")]
    for disease in KEY_DISEASES:
        d_df = top3_gender[top3_gender["disease"] == disease].dropna(subset=["gap_reduction"])
        if len(d_df) > 0:
            mean_gap_bl = d_df["gap_baseline"].mean()
            mean_gap_after = d_df["auroc_gap"].mean()
            mean_red = d_df["gap_reduction"].mean()
            print(f"  {disease:30s}: {mean_gap_bl:.4f} -> {mean_gap_after:.4f}  "
                  f"(reduction={mean_red:+.4f})")

    # -- Summary statistics for paper ----------------------------------
    print(f"\n{'='*90}")
    print(f"  PAPER-READY SUMMARY")
    print(f"{'='*90}")

    top3_all = compare[compare["strategy"] == "top3"].dropna(subset=["gap_reduction"])
    conf_all = compare[compare["strategy"] == "all_confounders"].dropna(subset=["gap_reduction"])

    if len(top3_all) > 0:
        mean_red, ci_lo, ci_hi = bootstrap_gap_reduction(
            top3_all["gap_baseline"].values, top3_all["auroc_gap"].values)
        print(f"\n  Top-3 debiasing (HF + AFib + Age):")
        print(f"    N observations: {len(top3_all)}")
        print(f"    Mean gap reduction: {mean_red:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    Percentage improved: {(top3_all['gap_reduction']>0).mean():.1%}")
        print(f"    Mean AUROC cost: {top3_all['auroc_cost'].mean():+.4f}")

    if len(conf_all) > 0:
        mean_red, ci_lo, ci_hi = bootstrap_gap_reduction(
            conf_all["gap_baseline"].values, conf_all["auroc_gap"].values)
        print(f"\n  All confounders (9 attributes):")
        print(f"    N observations: {len(conf_all)}")
        print(f"    Mean gap reduction: {mean_red:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    Percentage improved: {(conf_all['gap_reduction']>0).mean():.1%}")
        print(f"    Mean AUROC cost: {conf_all['auroc_cost'].mean():+.4f}")

    # -- Save ----------------------------------------------------------
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    detail_path = os.path.join(outdir, "fairness_debiasing_proof.csv")
    df.to_csv(detail_path, index=False, float_format="%.6f")

    if len(compare) > 0:
        compare_path = os.path.join(outdir, "fairness_debiasing_comparison.csv")
        compare.to_csv(compare_path, index=False, float_format="%.6f")

    elapsed = time.time() - t0
    print(f"\n  Results: {detail_path}")
    print(f"  Comparison: {os.path.join(outdir, 'fairness_debiasing_comparison.csv')}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
