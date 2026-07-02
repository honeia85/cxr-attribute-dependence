"""
Phase 3: Bias & Fairness Analysis - Performance disparities across demographic subgroups.

For each FM and each disease:
  - AUROC per subgroup (gender, age, BMI) with bootstrap CI
  - AUROC gap (max - min) with permutation test
  - AUPRC per subgroup
  - Equalized Odds Difference (|TPR_a - TPR_b| + |FPR_a - FPR_b|)
  - Expected Calibration Error (ECE) per subgroup
  - Brier score per subgroup
  - Effect size reporting (gap + CI) regardless of p-value
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from experiments.config import (
    MODELS, MAIN_MODELS, SEED, MULTI_SEEDS, RESULT_DIR, SUBGROUPS,
    BIAS_MIN_VALID, BIAS_MIN_POS, BIAS_MIN_NEG, ID_COL,
    N_BOOTSTRAP, BOOTSTRAP_CI,
)
from experiments.data import (
    load_metadata, load_canonical_ids, load_split,
    get_aligned_embeddings, merge_chexpert, get_eligible_diseases,
    scale_embeddings,
)
from experiments.stats import (
    bootstrap_auroc, bootstrap_auprc, permutation_gap_test,
    benjamini_hochberg, format_ci, bootstrap_ci,
)
from experiments.models import (
    set_all_seeds, CXROnlyModel, compute_pos_weight,
    train_disease_model, predict_disease, make_loaders,
)


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error.

    Lower is better. Measures how well predicted probabilities
    match actual frequencies.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_accuracy - bin_confidence)
    return ece


def compute_brier(y_true, y_prob):
    """Brier score.

    Lower is better. Measures mean squared probability error.
    """
    return np.mean((y_prob - y_true) ** 2)


def compute_equalized_odds(y_true, y_pred, group_labels, groups, threshold=None):
    """Compute Equalized Odds Difference across subgroups.

    Uses Youden's J optimal threshold if not provided.

    Returns:
        dict with per-group TPR, FPR, and max equalized odds difference
    """
    if threshold is None:
        # Find optimal threshold using Youden's J
        thresholds = np.linspace(0.1, 0.9, 17)
        best_j = -1
        best_t = 0.5
        for t in thresholds:
            pred_binary = (y_pred >= t).astype(int)
            if pred_binary.sum() == 0 or pred_binary.sum() == len(pred_binary):
                continue
            tpr = y_true[pred_binary == 1].mean() if pred_binary.sum() > 0 else 0
            fpr = (1 - y_true[pred_binary == 0].mean()) if (pred_binary == 0).sum() > 0 else 0
            # Correctly: TPR = TP/(TP+FN), FPR = FP/(FP+TN)
            tp = ((pred_binary == 1) & (y_true == 1)).sum()
            fn = ((pred_binary == 0) & (y_true == 1)).sum()
            fp = ((pred_binary == 1) & (y_true == 0)).sum()
            tn = ((pred_binary == 0) & (y_true == 0)).sum()
            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            j = tpr_val - fpr_val
            if j > best_j:
                best_j = j
                best_t = t
        threshold = best_t

    pred_binary = (y_pred >= threshold).astype(int)
    group_metrics = {}

    for g in groups:
        mask = group_labels == g
        yt = y_true[mask]
        yp = pred_binary[mask]

        if len(yt) < 10 or len(np.unique(yt)) < 2:
            group_metrics[g] = {"tpr": np.nan, "fpr": np.nan, "n": len(yt)}
            continue

        tp = ((yp == 1) & (yt == 1)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        group_metrics[g] = {"tpr": tpr, "fpr": fpr, "n": len(yt)}

    # Compute max equalized odds difference
    valid_tprs = [m["tpr"] for m in group_metrics.values() if not np.isnan(m["tpr"])]
    valid_fprs = [m["fpr"] for m in group_metrics.values() if not np.isnan(m["fpr"])]

    if len(valid_tprs) >= 2 and len(valid_fprs) >= 2:
        tpr_gap = max(valid_tprs) - min(valid_tprs)
        fpr_gap = max(valid_fprs) - min(valid_fprs)
        eq_odds_diff = tpr_gap + fpr_gap
    else:
        tpr_gap = np.nan
        fpr_gap = np.nan
        eq_odds_diff = np.nan

    return group_metrics, eq_odds_diff, tpr_gap, fpr_gap, threshold


def compute_subgroup_metrics(y_true, y_pred, group_labels, groups,
                             min_valid=BIAS_MIN_VALID, min_pos=BIAS_MIN_POS,
                             min_neg=BIAS_MIN_NEG):
    """Compute AUROC, AUPRC, ECE, and Brier score per subgroup.

    Returns:
        list of dicts with subgroup metrics
    """
    results = []
    for g in groups:
        mask = group_labels == g
        yt = y_true[mask]
        yp = y_pred[mask]

        n_valid = len(yt)
        n_pos = yt.sum()
        n_neg = n_valid - n_pos

        if n_valid < min_valid or n_pos < min_pos or n_neg < min_neg:
            results.append({
                "subgroup": g, "n": n_valid, "n_pos": int(n_pos),
                "prevalence": n_pos / n_valid if n_valid > 0 else np.nan,
                "auroc": np.nan, "auroc_ci_lo": np.nan, "auroc_ci_hi": np.nan,
                "auprc": np.nan, "auprc_ci_lo": np.nan, "auprc_ci_hi": np.nan,
                "ece": np.nan,
                "brier": np.nan,
                "sufficient_data": False,
            })
            continue

        auroc_pt, auroc_lo, auroc_hi = bootstrap_auroc(yt, yp)
        auprc_pt, auprc_lo, auprc_hi = bootstrap_auprc(yt, yp)
        ece = compute_ece(yt, yp)
        brier = compute_brier(yt, yp)

        results.append({
            "subgroup": g, "n": n_valid, "n_pos": int(n_pos),
            "prevalence": n_pos / n_valid,
            "auroc": auroc_pt, "auroc_ci_lo": auroc_lo, "auroc_ci_hi": auroc_hi,
            "auprc": auprc_pt, "auprc_ci_lo": auprc_lo, "auprc_ci_hi": auprc_hi,
            "ece": ece,
            "brier": brier,
            "sufficient_data": True,
        })

    return results


def bootstrap_gap(y_true, y_pred, group_labels, groups,
                  n_boot=N_BOOTSTRAP, ci=BOOTSTRAP_CI, seed=SEED):
    """Bootstrap CI for AUROC gap (max - min across groups)."""
    rng = np.random.RandomState(seed)
    alpha = (1 - ci) / 2
    n = len(y_true)

    def compute_gap(yt, yp, gl):
        aurocs = []
        for g in groups:
            mask = gl == g
            yt_g, yp_g = yt[mask], yp[mask]
            if len(yt_g) < 10 or len(np.unique(yt_g)) < 2:
                return np.nan
            aurocs.append(roc_auc_score(yt_g, yp_g))
        return max(aurocs) - min(aurocs) if len(aurocs) >= 2 else np.nan

    observed = compute_gap(y_true, y_pred, group_labels)
    boot_gaps = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        g = compute_gap(y_true[idx], y_pred[idx], group_labels[idx])
        if not np.isnan(g):
            boot_gaps.append(g)

    if len(boot_gaps) < 10:
        return observed, np.nan, np.nan

    return observed, np.percentile(boot_gaps, alpha * 100), np.percentile(boot_gaps, (1 - alpha) * 100)


def assign_subgroups(df, dimension_config):
    """Assign subgroup labels based on config.

    Returns:
        np.array of group labels (strings)
    """
    col = dimension_config["column"]
    values = df[col].values

    if dimension_config["bins"] is None:
        labels = dimension_config["labels"]
        # Handle string-valued columns (e.g. race_group) vs numeric (e.g. gender_binary)
        if values.dtype.kind in ('U', 'S', 'O'):
            # String values: use directly if in labels, else "missing"
            label_set = set(labels)
            return np.array([v if v in label_set else "missing" for v in values])
        else:
            return np.array([labels[int(v)] if not np.isnan(v) else "missing" for v in values])
    else:
        bins = dimension_config["bins"]
        labels = dimension_config["labels"]
        result = np.full(len(values), "missing", dtype=object)
        for i, v in enumerate(values):
            if np.isnan(v):
                continue
            for j in range(len(bins) - 1):
                if bins[j] <= v < bins[j + 1]:
                    result[i] = labels[j]
                    break
        return result


def run_phase3(models=None):
    """Run full Phase 3 bias analysis."""
    if models is None:
        models = MAIN_MODELS

    os.makedirs(RESULT_DIR, exist_ok=True)
    canonical_ids = load_canonical_ids()
    split = load_split()

    # Load data
    metadata = load_metadata()
    id_to_order = {did: i for i, did in enumerate(canonical_ids)}
    meta_df = metadata[metadata[ID_COL].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df[ID_COL].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)

    merged_df, labels, masks, diseases = merge_chexpert(meta_df)
    tier1, tier2 = get_eligible_diseases(labels, masks, split, diseases)
    analyze_diseases = tier1 + tier2

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    all_results = []
    all_gap_tests = []
    all_eq_odds = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Phase 3: {model_name}")
        print(f"{'='*60}")

        cxr_dim = MODELS[model_name]["embed_dim"]
        embeddings = get_aligned_embeddings(model_name, canonical_ids)
        train_emb, val_emb, test_emb, _ = scale_embeddings(embeddings, split)

        pw = compute_pos_weight(labels[train_idx], masks[train_idx])

        # Train CXR-Only model with multiple seeds for robustness
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bias_seeds = MULTI_SEEDS[:3]  # use 3 seeds for efficiency
        all_logits = []
        for seed in bias_seeds:
            set_all_seeds(seed)
            model = CXROnlyModel(cxr_dim, len(diseases)).to(device)
            train_loader, val_loader, test_loader = make_loaders(
                train_emb, val_emb, test_emb,
                labels[train_idx], labels[val_idx], labels[test_idx],
                masks[train_idx], masks[val_idx], masks[test_idx],
            )
            model = train_disease_model(model, train_loader, val_loader, device, pw)
            logits_s, test_lab, test_msk = predict_disease(model, test_loader, device)
            all_logits.append(logits_s)
        logits = np.mean(all_logits, axis=0)  # ensemble average
        probs = 1.0 / (1.0 + np.exp(-logits))

        # For each demographic dimension
        for dim_name, dim_config in SUBGROUPS.items():
            test_df = meta_df.iloc[test_idx]
            group_labels = assign_subgroups(test_df, dim_config)
            valid_groups = [g for g in dim_config["labels"]]

            print(f"\n  --- {dim_name} ---")
            for j, disease in enumerate(diseases):
                if disease not in analyze_diseases:
                    continue

                valid = test_msk[:, j].astype(bool)
                y_true = test_lab[valid, j]
                y_pred = probs[valid, j]
                g_labels = group_labels[valid]

                # Filter out "missing" group
                non_missing = g_labels != "missing"
                y_true = y_true[non_missing]
                y_pred = y_pred[non_missing]
                g_labels = g_labels[non_missing]

                # Subgroup metrics (AUROC, AUPRC, ECE, Brier)
                sg_results = compute_subgroup_metrics(y_true, y_pred, g_labels, valid_groups)

                # AUROC gap + bootstrap CI
                aurocs = [r["auroc"] for r in sg_results if not np.isnan(r["auroc"])]
                gap = max(aurocs) - min(aurocs) if len(aurocs) >= 2 else np.nan
                gap_ci_lo, gap_ci_hi = np.nan, np.nan
                if not np.isnan(gap):
                    _, gap_ci_lo, gap_ci_hi = bootstrap_gap(
                        y_true, y_pred, g_labels, valid_groups)

                # Permutation test for gap
                gap_p = np.nan
                if not np.isnan(gap) and gap > 0:
                    _, gap_p = permutation_gap_test(y_true, y_pred, g_labels)

                # Equalized Odds
                eq_metrics, eq_odds_diff, tpr_gap, fpr_gap, threshold = \
                    compute_equalized_odds(y_true, y_pred, g_labels, valid_groups)

                for sg in sg_results:
                    eq_m = eq_metrics.get(sg["subgroup"], {})
                    row = {
                        "model": model_name,
                        "dimension": dim_name,
                        "disease": disease,
                        "subgroup": sg["subgroup"],
                        "n": sg["n"],
                        "n_pos": sg["n_pos"],
                        "prevalence": sg.get("prevalence", np.nan),
                        "auroc": sg["auroc"],
                        "auroc_ci_lo": sg.get("auroc_ci_lo", np.nan),
                        "auroc_ci_hi": sg.get("auroc_ci_hi", np.nan),
                        "auprc": sg.get("auprc", np.nan),
                        "auprc_ci_lo": sg.get("auprc_ci_lo", np.nan),
                        "auprc_ci_hi": sg.get("auprc_ci_hi", np.nan),
                        "ece": sg.get("ece", np.nan),
                        "brier": sg.get("brier", np.nan),
                        "tpr": eq_m.get("tpr", np.nan),
                        "fpr": eq_m.get("fpr", np.nan),
                        "sufficient_data": sg["sufficient_data"],
                    }
                    all_results.append(row)

                all_gap_tests.append({
                    "model": model_name,
                    "dimension": dim_name,
                    "disease": disease,
                    "auroc_gap": gap,
                    "auroc_gap_ci_lo": gap_ci_lo,
                    "auroc_gap_ci_hi": gap_ci_hi,
                    "gap_p_value": gap_p,
                    "n_subgroups_valid": len(aurocs),
                    "eq_odds_diff": eq_odds_diff,
                    "tpr_gap": tpr_gap,
                    "fpr_gap": fpr_gap,
                    "threshold": threshold,
                })

                if not np.isnan(gap):
                    gap_ci_str = f" [{gap_ci_lo:.3f}-{gap_ci_hi:.3f}]" if not np.isnan(gap_ci_lo) else ""
                    p_str = f" p={gap_p:.3f}" if not np.isnan(gap_p) else ""
                    eq_str = f" EqOdds={eq_odds_diff:.3f}" if not np.isnan(eq_odds_diff) else ""
                    print(f"    {disease:30s}: gap={gap:.3f}{gap_ci_str}{p_str}{eq_str}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULT_DIR, "phase3_bias.csv"), index=False)

    gaps_df = pd.DataFrame(all_gap_tests)
    # FDR correction on gap p-values
    valid_p = gaps_df["gap_p_value"].values
    rejected, adjusted_p = benjamini_hochberg(valid_p)
    gaps_df["gap_p_adjusted"] = adjusted_p
    gaps_df["gap_significant"] = rejected
    gaps_df.to_csv(os.path.join(RESULT_DIR, "phase3_gaps.csv"), index=False)

    print(f"\n  Saved: phase3_bias.csv, phase3_gaps.csv")
    print(f"  Significant AUROC gaps (after FDR): {rejected.sum()} / {len(rejected)}")

    # Effect size summary (regardless of p-value)
    print(f"\n  {'='*60}")
    print(f"  Effect Size Summary (top gaps regardless of p-value)")
    print(f"  {'='*60}")
    top_gaps = gaps_df.dropna(subset=["auroc_gap"]).nlargest(15, "auroc_gap")
    for _, row in top_gaps.iterrows():
        ci_str = f"[{row['auroc_gap_ci_lo']:.3f}-{row['auroc_gap_ci_hi']:.3f}]" if not np.isnan(row.get("auroc_gap_ci_lo", np.nan)) else ""
        p_str = f"p={row['gap_p_value']:.3f}" if not np.isnan(row["gap_p_value"]) else ""
        eq_str = f"EqOdds={row['eq_odds_diff']:.3f}" if not np.isnan(row.get("eq_odds_diff", np.nan)) else ""
        print(f"  {row['model']:25s} {row['dimension']:8s} {row['disease']:25s}: "
              f"gap={row['auroc_gap']:.3f} {ci_str} {p_str} {eq_str}")

    return results_df, gaps_df


if __name__ == "__main__":
    run_phase3()
