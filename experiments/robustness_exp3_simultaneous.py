"""
Experiment 3: Simultaneous residualization of top-3 attributes.
Tests whether collinearity among HF, AFib, Age inflates individual AUROC drops.
Compares: sum(individual drops) vs simultaneous drop.
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from experiments.config import SEED
from experiments.data import (load_metadata, load_canonical_ids, load_split,
                              get_aligned_embeddings, merge_chexpert)

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
    "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Support Devices",
]

TOP3_ATTRS = ["heart_failure", "atrial_fibrillation", "age"]
ALPHA_RIDGE = 1.0


def residualize(X_train, X_test, D_train, D_test, alpha=ALPHA_RIDGE):
    """Remove attribute signal from embeddings via Ridge regression."""
    model = Ridge(alpha=alpha)
    model.fit(D_train, X_train)
    return X_train - model.predict(D_train), X_test - model.predict(D_test)


def disease_auroc(X_train, y_train, X_test, y_test):
    """Logistic regression AUROC for disease prediction."""
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, prob)


def main():
    t0 = time.time()
    print("=" * 80)
    print("  EXPERIMENT 3: SIMULTANEOUS vs INDIVIDUAL RESIDUALIZATION")
    print("  Top-3 attributes: heart_failure, atrial_fibrillation, age")
    print("=" * 80)

    # Load data
    canonical_ids = load_canonical_ids()
    split = load_split()
    metadata = load_metadata()
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    id_to_order = {did: i for i, did in enumerate(canonical_ids)}
    meta_df = metadata[metadata["dicom_id"].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df["dicom_id"].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)

    merged, labels, masks, all_diseases = merge_chexpert(meta_df)
    disease_indices = {d: all_diseases.index(d) for d in KEY_DISEASES}

    # Attribute arrays
    hf_all = meta_df["heart_failure"].values.astype(float)
    af_all = meta_df["atrial_fibrillation"].values.astype(float)
    age_all = meta_df["age"].values.astype(float)
    age_mean = np.nanmean(age_all[train_idx])
    age_std = np.nanstd(age_all[train_idx])

    rows = []
    for model_name in CLEAN_MODELS:
        print(f"\n  MODEL: {model_name}")
        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler()
        scaler.fit(emb[train_idx])
        emb_scaled = scaler.transform(emb)

        X_tr = emb_scaled[train_idx]
        X_te = emb_scaled[test_idx]

        # Build D matrices for individual and simultaneous
        D_hf_tr = hf_all[train_idx].reshape(-1, 1)
        D_hf_te = hf_all[test_idx].reshape(-1, 1)

        D_af_tr = af_all[train_idx].reshape(-1, 1)
        D_af_te = af_all[test_idx].reshape(-1, 1)

        age_tr_norm = ((age_all[train_idx] - age_mean) / age_std).reshape(-1, 1)
        age_te_norm = ((age_all[test_idx] - age_mean) / age_std).reshape(-1, 1)

        # Simultaneous D matrix (3 columns)
        D_sim_tr = np.hstack([D_hf_tr, D_af_tr, age_tr_norm])
        D_sim_te = np.hstack([D_hf_te, D_af_te, age_te_norm])

        # Residualize
        X_tr_hf, X_te_hf = residualize(X_tr, X_te, D_hf_tr, D_hf_te)
        X_tr_af, X_te_af = residualize(X_tr, X_te, D_af_tr, D_af_te)
        X_tr_age, X_te_age = residualize(X_tr, X_te, age_tr_norm, age_te_norm)
        X_tr_sim, X_te_sim = residualize(X_tr, X_te, D_sim_tr, D_sim_te)

        for disease in KEY_DISEASES:
            didx = disease_indices[disease]
            mask_tr = masks[train_idx, didx].astype(bool)
            mask_te = masks[test_idx, didx].astype(bool)
            y_tr = labels[train_idx, didx]
            y_te = labels[test_idx, didx]

            if mask_tr.sum() < 50 or mask_te.sum() < 50:
                continue
            if y_tr[mask_tr].sum() < 10 or y_te[mask_te].sum() < 10:
                continue

            try:
                auroc_base = disease_auroc(X_tr[mask_tr], y_tr[mask_tr],
                                           X_te[mask_te], y_te[mask_te])
                auroc_hf = disease_auroc(X_tr_hf[mask_tr], y_tr[mask_tr],
                                         X_te_hf[mask_te], y_te[mask_te])
                auroc_af = disease_auroc(X_tr_af[mask_tr], y_tr[mask_tr],
                                         X_te_af[mask_te], y_te[mask_te])
                auroc_age = disease_auroc(X_tr_age[mask_tr], y_tr[mask_tr],
                                          X_te_age[mask_te], y_te[mask_te])
                auroc_sim = disease_auroc(X_tr_sim[mask_tr], y_tr[mask_tr],
                                          X_te_sim[mask_te], y_te[mask_te])

                drop_hf = auroc_base - auroc_hf
                drop_af = auroc_base - auroc_af
                drop_age = auroc_base - auroc_age
                drop_sim = auroc_base - auroc_sim
                sum_individual = drop_hf + drop_af + drop_age
                redundancy = sum_individual - drop_sim

                rows.append({
                    "model": model_name, "disease": disease,
                    "auroc_baseline": auroc_base,
                    "auroc_resid_hf": auroc_hf,
                    "auroc_resid_af": auroc_af,
                    "auroc_resid_age": auroc_age,
                    "auroc_resid_simultaneous": auroc_sim,
                    "drop_hf": drop_hf,
                    "drop_af": drop_af,
                    "drop_age": drop_age,
                    "drop_simultaneous": drop_sim,
                    "sum_individual": sum_individual,
                    "redundancy": redundancy,
                    "redundancy_pct": redundancy / sum_individual * 100 if sum_individual > 0 else 0,
                })
            except Exception as e:
                print(f"    SKIP {disease}: {e}")

        elapsed = time.time() - t0
        print(f"    Done ({elapsed:.0f}s)")

    df = pd.DataFrame(rows)

    # Summary
    print("\n" + "=" * 80)
    print("  RESULTS: SIMULTANEOUS vs INDIVIDUAL RESIDUALIZATION")
    print("=" * 80)

    print(f"\n  Total observations: {len(df)}")
    print(f"\n  Mean drops across all model-disease pairs:")
    print(f"    Heart failure (individual): {df.drop_hf.mean():.6f}")
    print(f"    AFib (individual):          {df.drop_af.mean():.6f}")
    print(f"    Age (individual):           {df.drop_age.mean():.6f}")
    print(f"    Sum of individual:          {df.sum_individual.mean():.6f}")
    print(f"    Simultaneous (all 3):       {df.drop_simultaneous.mean():.6f}")
    print(f"    Redundancy:                 {df.redundancy.mean():.6f}")
    print(f"    Redundancy %:               {df.redundancy_pct.mean():.1f}%")

    print(f"\n  Per-model summary:")
    for model in CLEAN_MODELS:
        m = df[df["model"] == model]
        print(f"    {model:25s}: sum_indiv={m.sum_individual.mean():.6f}, "
              f"simultaneous={m.drop_simultaneous.mean():.6f}, "
              f"redundancy={m.redundancy_pct.mean():.1f}%")

    print(f"\n  Per-disease summary:")
    for disease in KEY_DISEASES:
        d = df[df["disease"] == disease]
        if len(d) > 0:
            print(f"    {disease:30s}: sum_indiv={d.sum_individual.mean():.6f}, "
                  f"simultaneous={d.drop_simultaneous.mean():.6f}, "
                  f"redundancy={d.redundancy_pct.mean():.1f}%")

    # Statistical test
    print(f"\n  Is redundancy significantly > 0?")
    from scipy.stats import ttest_1samp
    t, p = ttest_1samp(df["redundancy"], 0)
    print(f"    t = {t:.3f}, p = {p:.2e}")
    print(f"    Mean redundancy = {df.redundancy.mean():.6f} "
          f"[{df.redundancy.quantile(0.025):.6f}, {df.redundancy.quantile(0.975):.6f}]")

    out = os.path.join(PROJECT, "results", "experiment3_simultaneous_residualization.csv")
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"\n  Saved: {out}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
