"""
Per-comorbidity residualization analysis.

For each of 20 comorbidities (+ sex, age, BMI as the non-race demographics),
residualize embeddings and measure AUROC drop on ALL 10 diseases.
Uses the 6 strict clean models (no pretraining overlap with MIMIC-CXR).

The 4th demographic (Black-vs-White race) is the 24th attribute in the
manuscript and is computed separately by
`experiments/race_residualization_mimic_cxr.py` because four-category race
requires a one-hot residualization block. Combining the two scripts'
outputs reproduces the 24 attributes x 10 findings x 6 models = 1,440
observation table (Figure 1, Table S1, Table 3).

Usage:
    cd cxr-metadata-study
    PYTHONPATH=. python experiments/per_comorbidity_residualization.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import SEED
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
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

COMORBIDITIES = [
    "hypertension", "heart_failure", "atrial_fibrillation",
    "coronary_artery_disease", "stroke", "diabetes", "hyperlipidemia",
    "obesity", "hypothyroidism", "ckd", "aki", "copd", "asthma",
    "respiratory_failure", "pulmonary_fibrosis", "liver_disease",
    "anemia", "smoking_history", "cancer_history", "depression",
]

DEMOGRAPHICS = ["sex", "age", "bmi"]  # comparison attributes

ALPHA_RIDGE = 1.0
N_BOOTSTRAP = 1000


def residualize(X_train, X_test, D_train, D_test, alpha=ALPHA_RIDGE):
    """Remove attribute signal from embeddings via Ridge regression."""
    model = Ridge(alpha=alpha)
    model.fit(D_train, X_train)
    X_train_resid = X_train - model.predict(D_train)
    X_test_resid = X_test - model.predict(D_test)
    return X_train_resid, X_test_resid


def disease_auroc(X_train, y_train, X_test, y_test):
    """Logistic regression AUROC for disease prediction."""
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, prob)


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    print("=" * 72)
    print("  PER-COMORBIDITY RESIDUALIZATION ANALYSIS")
    print(f"  {len(CLEAN_MODELS)} clean models x "
          f"{len(DEMOGRAPHICS) + len(COMORBIDITIES)} attributes x "
          f"{len(KEY_DISEASES)} diseases")
    print("  (race is the 4th demographic, computed by "
          "experiments/race_residualization_mimic_cxr.py)")
    print("=" * 72)

    # -- Load data ------------------------------------------------------
    print("\n[1/4] Loading data...")
    canonical_ids = load_canonical_ids()
    split = load_split()
    metadata = load_metadata()
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    # Align metadata to canonical order
    id_to_order = {did: i for i, did in enumerate(canonical_ids)}
    meta_df = metadata[metadata["dicom_id"].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df["dicom_id"].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)

    # Merge disease labels
    merged, labels, masks, all_diseases = merge_chexpert(meta_df)
    disease_indices = {d: all_diseases.index(d) for d in KEY_DISEASES}

    # Extract demographic arrays
    sex_all = meta_df["gender_binary"].values.astype(float)
    age_all = meta_df["age"].values.astype(float)
    bmi_all = meta_df["bmi"].values.astype(float)

    # Train-fit normalization for continuous demographics
    age_mean = np.nanmean(age_all[train_idx])
    age_std = np.nanstd(age_all[train_idx])
    bmi_mean = np.nanmean(bmi_all[train_idx])
    bmi_std = np.nanstd(bmi_all[train_idx])

    # Extract comorbidity arrays
    comorbidity_arrays = {}
    for c in COMORBIDITIES:
        comorbidity_arrays[c] = meta_df[c].values.astype(float)

    # Print comorbidity prevalence
    print(f"\n  Canonical dataset: {len(canonical_ids)} images")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"\n  Comorbidity prevalence (full dataset):")
    for c in COMORBIDITIES:
        vals = comorbidity_arrays[c]
        valid = np.isfinite(vals)
        prev = vals[valid].mean() if valid.sum() > 0 else 0
        print(f"    {c:30s}: {prev:.1%}  (n_valid={valid.sum()})")

    # -- Build all attribute conditions ---------------------------------
    # For demographics, we need valid BMI mask (sex and age are 100% complete)
    demo_valid_tr = (np.isfinite(sex_all[train_idx]) &
                     np.isfinite(age_all[train_idx]) &
                     np.isfinite(bmi_all[train_idx]))
    demo_valid_te = (np.isfinite(sex_all[test_idx]) &
                     np.isfinite(age_all[test_idx]) &
                     np.isfinite(bmi_all[test_idx]))

    # -- Main loop ------------------------------------------------------
    n_attr = len(DEMOGRAPHICS) + len(COMORBIDITIES)
    n_cells = len(CLEAN_MODELS) * n_attr * len(KEY_DISEASES)
    print(f"\n[2/4] Running residualization loop...")
    print(f"  {len(CLEAN_MODELS)} models x ({len(DEMOGRAPHICS)} demographics + "
          f"{len(COMORBIDITIES)} comorbidities) x {len(KEY_DISEASES)} diseases "
          f"= {n_cells} regressions")
    print(f"  x2 (baseline + residualized) = {2 * n_cells} logistic regressions total\n")

    rows = []
    n_done = 0
    n_total = len(CLEAN_MODELS) * (len(DEMOGRAPHICS) + len(COMORBIDITIES)) * len(KEY_DISEASES)

    for model_name in CLEAN_MODELS:
        print(f"\n{'='*68}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*68}")

        # Load and scale embeddings
        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler()
        scaler.fit(emb[train_idx])
        emb_scaled = scaler.transform(emb)

        # -- Demographics (use BMI-valid subset) ------------------------
        tr_demo = train_idx[demo_valid_tr]
        te_demo = test_idx[demo_valid_te]
        X_tr_demo = emb_scaled[tr_demo]
        X_te_demo = emb_scaled[te_demo]

        for attr_name in DEMOGRAPHICS:
            if attr_name == "sex":
                D_tr = sex_all[tr_demo].reshape(-1, 1)
                D_te = sex_all[te_demo].reshape(-1, 1)
            elif attr_name == "age":
                D_tr = ((age_all[tr_demo] - age_mean) / age_std).reshape(-1, 1)
                D_te = ((age_all[te_demo] - age_mean) / age_std).reshape(-1, 1)
            elif attr_name == "bmi":
                D_tr = ((bmi_all[tr_demo] - bmi_mean) / bmi_std).reshape(-1, 1)
                D_te = ((bmi_all[te_demo] - bmi_mean) / bmi_std).reshape(-1, 1)

            X_tr_resid, X_te_resid = residualize(X_tr_demo, X_te_demo, D_tr, D_te)

            for disease in KEY_DISEASES:
                didx = disease_indices[disease]
                mask_tr = masks[tr_demo, didx].astype(bool)
                mask_te = masks[te_demo, didx].astype(bool)
                y_tr = labels[tr_demo, didx]
                y_te = labels[te_demo, didx]

                if mask_tr.sum() < 50 or mask_te.sum() < 50:
                    n_done += 1
                    continue
                if y_tr[mask_tr].sum() < 10 or y_te[mask_te].sum() < 10:
                    n_done += 1
                    continue

                try:
                    auroc_base = disease_auroc(
                        X_tr_demo[mask_tr], y_tr[mask_tr],
                        X_te_demo[mask_te], y_te[mask_te])
                    auroc_resid = disease_auroc(
                        X_tr_resid[mask_tr], y_tr[mask_tr],
                        X_te_resid[mask_te], y_te[mask_te])
                    drop = auroc_base - auroc_resid
                except Exception as e:
                    auroc_base = auroc_resid = drop = np.nan

                rows.append({
                    "attribute": attr_name,
                    "attribute_type": "demographic",
                    "model": model_name,
                    "disease": disease,
                    "auroc_baseline": auroc_base,
                    "auroc_residualized": auroc_resid,
                    "auroc_drop": drop,
                })
                n_done += 1

            print(f"  {attr_name:30s} done  [{n_done}/{n_total}]  "
                  f"({time.time()-t0:.0f}s)")

        # -- Comorbidities (use full dataset, 100% complete) ------------
        X_tr_full = emb_scaled[train_idx]
        X_te_full = emb_scaled[test_idx]

        for comorbidity in COMORBIDITIES:
            c_all = comorbidity_arrays[comorbidity]

            # Build single-column D matrix
            D_tr = c_all[train_idx].reshape(-1, 1)
            D_te = c_all[test_idx].reshape(-1, 1)

            X_tr_resid, X_te_resid = residualize(X_tr_full, X_te_full, D_tr, D_te)

            for disease in KEY_DISEASES:
                didx = disease_indices[disease]
                mask_tr = masks[train_idx, didx].astype(bool)
                mask_te = masks[test_idx, didx].astype(bool)
                y_tr = labels[train_idx, didx]
                y_te = labels[test_idx, didx]

                if mask_tr.sum() < 50 or mask_te.sum() < 50:
                    n_done += 1
                    continue
                if y_tr[mask_tr].sum() < 10 or y_te[mask_te].sum() < 10:
                    n_done += 1
                    continue

                try:
                    auroc_base = disease_auroc(
                        X_tr_full[mask_tr], y_tr[mask_tr],
                        X_te_full[mask_te], y_te[mask_te])
                    auroc_resid = disease_auroc(
                        X_tr_resid[mask_tr], y_tr[mask_tr],
                        X_te_resid[mask_te], y_te[mask_te])
                    drop = auroc_base - auroc_resid
                except Exception as e:
                    auroc_base = auroc_resid = drop = np.nan

                rows.append({
                    "attribute": comorbidity,
                    "attribute_type": "comorbidity",
                    "model": model_name,
                    "disease": disease,
                    "auroc_baseline": auroc_base,
                    "auroc_residualized": auroc_resid,
                    "auroc_drop": drop,
                })
                n_done += 1

            print(f"  {comorbidity:30s} done  [{n_done}/{n_total}]  "
                  f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)

    # -- Summary statistics ---------------------------------------------
    print(f"\n[3/4] Computing summary statistics and bootstrap CIs...")

    # Mean drop per attribute (across CLEAN_MODELS x KEY_DISEASES observations)
    attr_summary = df.groupby("attribute").agg(
        mean_drop=("auroc_drop", "mean"),
        median_drop=("auroc_drop", "median"),
        std_drop=("auroc_drop", "std"),
        n_obs=("auroc_drop", "count"),
    ).reset_index()

    # Bootstrap 95% CI for each attribute's mean drop
    bootstrap_results = {}
    for attr in attr_summary["attribute"].values:
        drops = df[df["attribute"] == attr]["auroc_drop"].dropna().values
        if len(drops) < 2:
            bootstrap_results[attr] = (np.nan, np.nan)
            continue
        boot_means = np.array([
            rng.choice(drops, size=len(drops), replace=True).mean()
            for _ in range(N_BOOTSTRAP)
        ])
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)
        bootstrap_results[attr] = (ci_lo, ci_hi)

    attr_summary["ci_lo"] = attr_summary["attribute"].map(
        lambda a: bootstrap_results[a][0])
    attr_summary["ci_hi"] = attr_summary["attribute"].map(
        lambda a: bootstrap_results[a][1])

    # Add attribute type
    attr_type_map = {a: "demographic" for a in DEMOGRAPHICS}
    attr_type_map.update({c: "comorbidity" for c in COMORBIDITIES})
    attr_summary["attribute_type"] = attr_summary["attribute"].map(attr_type_map)

    # Rank by mean drop (descending)
    attr_summary = attr_summary.sort_values("mean_drop", ascending=False).reset_index(drop=True)
    attr_summary["rank"] = range(1, len(attr_summary) + 1)

    # Classify cost tier
    def classify_tier(drop):
        if abs(drop) < 0.001:
            return "free"
        elif abs(drop) < 0.005:
            return "cheap"
        elif abs(drop) < 0.015:
            return "moderate"
        else:
            return "expensive"

    attr_summary["cost_tier"] = attr_summary["mean_drop"].apply(classify_tier)

    # -- Print results --------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  RANKED ATTRIBUTES BY MEAN AUROC DROP (highest = most costly to debias)")
    print(f"  Each attribute averaged across {len(CLEAN_MODELS)} models x "
          f"{len(KEY_DISEASES)} diseases = up to "
          f"{len(CLEAN_MODELS) * len(KEY_DISEASES)} observations")
    print(f"{'='*80}")
    print(f"  {'Rank':>4s}  {'Attribute':30s}  {'Type':12s}  {'Mean Drop':>10s}  "
          f"{'95% CI':>18s}  {'Tier':>10s}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*12}  {'-'*10}  {'-'*18}  {'-'*10}")

    for _, row in attr_summary.iterrows():
        ci_str = f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]"
        print(f"  {int(row['rank']):4d}  {row['attribute']:30s}  "
              f"{row['attribute_type']:12s}  {row['mean_drop']:+.6f}  "
              f"{ci_str:>18s}  {row['cost_tier']:>10s}")

    # -- Tier summary ---------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  COST TIER SUMMARY")
    print(f"{'='*80}")
    for tier in ["expensive", "moderate", "cheap", "free"]:
        members = attr_summary[attr_summary["cost_tier"] == tier]
        if len(members) > 0:
            names = ", ".join(members["attribute"].values)
            print(f"\n  {tier.upper()} ({len(members)} attributes):")
            print(f"    {names}")
            print(f"    Mean drop range: [{members['mean_drop'].min():+.6f}, "
                  f"{members['mean_drop'].max():+.6f}]")

    # -- Per-model breakdown --------------------------------------------
    print(f"\n{'='*80}")
    print(f"  PER-MODEL MEAN DROP (top 5 attributes)")
    print(f"{'='*80}")
    for model in CLEAN_MODELS:
        model_df = df[df["model"] == model]
        model_attr = model_df.groupby("attribute")["auroc_drop"].mean()
        model_attr = model_attr.sort_values(ascending=False)
        print(f"\n  {model}:")
        for attr, drop in model_attr.head(5).items():
            print(f"    {attr:30s}: {drop:+.6f}")

    # -- Demographics vs comorbidities comparison -----------------------
    print(f"\n{'='*80}")
    print(f"  DEMOGRAPHICS vs COMORBIDITIES")
    print(f"{'='*80}")
    demo_drops = df[df["attribute_type"] == "demographic"]["auroc_drop"]
    comor_drops = df[df["attribute_type"] == "comorbidity"]["auroc_drop"]
    print(f"  Demographics (n={len(demo_drops)}):  "
          f"mean={demo_drops.mean():+.6f}, median={demo_drops.median():+.6f}")
    print(f"  Comorbidities (n={len(comor_drops)}): "
          f"mean={comor_drops.mean():+.6f}, median={comor_drops.median():+.6f}")
    print(f"  Ratio (demo/comor mean): "
          f"{demo_drops.mean() / comor_drops.mean():.1f}x" if comor_drops.mean() != 0
          else "  Comorbidity mean = 0")

    # -- Save -----------------------------------------------------------
    print(f"\n[4/4] Saving results...")
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    # Full detail
    detail_path = os.path.join(outdir, "per_comorbidity_residualization.csv")
    df.to_csv(detail_path, index=False, float_format="%.6f")
    print(f"  Detail: {detail_path}")

    # Summary
    summary_path = os.path.join(outdir, "per_comorbidity_residualization_summary.csv")
    attr_summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"  Summary: {summary_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Total regressions: {n_done}")


if __name__ == "__main__":
    main()
