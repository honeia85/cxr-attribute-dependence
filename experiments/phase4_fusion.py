"""
Phase 4: Metadata Fusion - Does explicit metadata improve disease prediction?

For each FM:
  - CXR-Only, Metadata-Only, Late Fusion (5 seeds each)
  - Per-disease AUROC with bootstrap CI
  - Does fusion reduce fairness gaps?
"""
import os
import numpy as np
import pandas as pd

from experiments.config import (
    MODELS, MAIN_MODELS, SEED, MULTI_SEEDS, RESULT_DIR, ID_COL,
)
from experiments.data import (
    load_metadata, load_canonical_ids, load_split,
    get_aligned_embeddings, merge_chexpert, get_eligible_diseases,
    build_metadata_vectors, scale_embeddings,
)
from experiments.stats import bootstrap_delta, format_delta
from experiments.models import (
    set_all_seeds, CXROnlyModel, MetadataOnlyModel, LateFusionModel,
    compute_pos_weight, train_disease_model, predict_disease,
    compute_per_disease_auroc, make_loaders,
)


def run_phase4(models=None):
    """Run full Phase 4 fusion analysis."""
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
    tier1, _ = get_eligible_diseases(labels, masks, split, diseases)

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    # Metadata vectors (dimension determined by available columns)
    train_meta, val_meta, test_meta, _ = build_metadata_vectors(meta_df, split)
    actual_meta_dim = train_meta.shape[1]
    print(f"  Metadata dimension: {actual_meta_dim}")

    pw = compute_pos_weight(labels[train_idx], masks[train_idx])
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    all_results = []
    bootstrap_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Phase 4: {model_name}")
        print(f"{'='*60}")

        cxr_dim = MODELS[model_name]["embed_dim"]
        embeddings = get_aligned_embeddings(model_name, canonical_ids)
        train_emb, val_emb, test_emb, _ = scale_embeddings(embeddings, split)

        fusion_configs = [
            ("CXR-Only", CXROnlyModel, {"cxr_dim": cxr_dim}, False),
            ("Metadata-Only", MetadataOnlyModel, {"meta_dim": actual_meta_dim}, True),
            ("Late Fusion", LateFusionModel, {"cxr_dim": cxr_dim, "meta_dim": actual_meta_dim}, True),
        ]

        model_predictions = {}  # {(fusion_name, seed): predictions}

        for fusion_name, ModelClass, kwargs, use_meta in fusion_configs:
            print(f"\n  [{fusion_name}]")
            seed_aurocs = []

            for seed in MULTI_SEEDS:
                set_all_seeds(seed)

                model = ModelClass(num_labels=len(diseases), **kwargs).to(device)

                tm = train_meta if use_meta else None
                vm = val_meta if use_meta else None
                tem = test_meta if use_meta else None

                train_loader, val_loader, test_loader = make_loaders(
                    train_emb, val_emb, test_emb,
                    labels[train_idx], labels[val_idx], labels[test_idx],
                    masks[train_idx], masks[val_idx], masks[test_idx],
                    tm, vm, tem,
                )

                model = train_disease_model(model, train_loader, val_loader, device, pw)
                logits, test_lab, test_msk = predict_disease(model, test_loader, device)

                per_disease = compute_per_disease_auroc(logits, test_lab, test_msk, diseases)
                mean_auroc = np.nanmean(list(per_disease.values()))
                seed_aurocs.append(mean_auroc)
                model_predictions[(fusion_name, seed)] = (logits, test_lab, test_msk, per_disease)

            mean_across_seeds = np.mean(seed_aurocs)
            std_across_seeds = np.std(seed_aurocs)
            print(f"    Mean AUROC: {mean_across_seeds:.4f} +/- {std_across_seeds:.4f}")

            # Aggregate per-disease results across seeds
            for disease in diseases:
                disease_aurocs = []
                for seed in MULTI_SEEDS:
                    d_aurocs = model_predictions[(fusion_name, seed)][3]
                    if not np.isnan(d_aurocs.get(disease, np.nan)):
                        disease_aurocs.append(d_aurocs[disease])

                row = {
                    "model": model_name,
                    "fusion": fusion_name,
                    "disease": disease,
                    "auroc_mean": np.mean(disease_aurocs) if disease_aurocs else np.nan,
                    "auroc_std": np.std(disease_aurocs) if disease_aurocs else np.nan,
                    "n_seeds": len(disease_aurocs),
                    "tier1": disease in tier1,
                }
                all_results.append(row)

            # Overall mean
            all_results.append({
                "model": model_name,
                "fusion": fusion_name,
                "disease": "MEAN",
                "auroc_mean": mean_across_seeds,
                "auroc_std": std_across_seeds,
                "n_seeds": len(MULTI_SEEDS),
                "tier1": True,
            })

        # Bootstrap comparison: Late Fusion vs CXR-Only (averaged across all seeds)
        print(f"\n  [Bootstrap: Late Fusion vs CXR-Only (all-seed avg)]")
        cxr_logits = np.mean(
            [model_predictions[("CXR-Only", s)][0] for s in MULTI_SEEDS], axis=0
        )
        lf_logits = np.mean(
            [model_predictions[("Late Fusion", s)][0] for s in MULTI_SEEDS], axis=0
        )
        test_lab = model_predictions[("CXR-Only", MULTI_SEEDS[0])][1]
        test_msk = model_predictions[("CXR-Only", MULTI_SEEDS[0])][2]

        for j, disease in enumerate(diseases):
            valid = test_msk[:, j].astype(bool)
            if valid.sum() < 30 or len(np.unique(test_lab[valid, j])) < 2:
                continue
            y_t = test_lab[valid, j]
            cxr_p = 1.0 / (1.0 + np.exp(-cxr_logits[valid, j]))
            lf_p = 1.0 / (1.0 + np.exp(-lf_logits[valid, j]))

            delta, ci_lo, ci_hi, p_val = bootstrap_delta(y_t, cxr_p, lf_p)
            print(f"    {disease:30s}: {format_delta(delta, ci_lo, ci_hi, p_val)}")
            bootstrap_results.append({
                "model": model_name,
                "disease": disease,
                "delta": delta,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_value": p_val,
                "significant": p_val < 0.05,
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULT_DIR, "phase4_fusion.csv"), index=False)
    print(f"\n  Saved: {os.path.join(RESULT_DIR, 'phase4_fusion.csv')}")

    # Save bootstrap p-values
    if bootstrap_results:
        boot_df = pd.DataFrame(bootstrap_results)
        boot_path = os.path.join(RESULT_DIR, "phase4_bootstrap_pvalues.csv")
        boot_df.to_csv(boot_path, index=False)
        print(f"  Saved: {boot_path}")

        # Summary: sig diseases per model
        sig_summary = boot_df.groupby("model")["significant"].sum().reset_index()
        sig_summary.columns = ["model", "sig_diseases"]
        print("\n  [Sig. Diseases Summary]")
        for _, row in sig_summary.iterrows():
            total = boot_df[boot_df["model"] == row["model"]].shape[0]
            print(f"    {row['model']:30s}: {int(row['sig_diseases'])}/{total}")

    return results_df


if __name__ == "__main__":
    run_phase4()
