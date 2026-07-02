"""
Per-comorbidity residualization on the 3 overlap models (RAD-DINO, CheXzero,
CheSS-ResNet50). Extends the n=6 architecture-invariance regression to n=9 with the
overlap models as a sensitivity analysis.

Output: results/per_comorbidity_residualization_overlap.csv
(combined with results/per_comorbidity_residualization.csv for n=9 nested
regression in experiments/nested_regression_n9.py).

Usage:
    PYTHONPATH=. python experiments/per_comorbidity_residualization_overlap.py
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

OVERLAP_MODELS = ["RAD-DINO", "CheXzero", "CheSS-ResNet50"]

KEY_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Support Devices",
]

COMORBIDITIES = [
    "hypertension", "heart_failure", "atrial_fibrillation",
    "coronary_artery_disease", "stroke", "diabetes", "hyperlipidemia",
    "obesity", "hypothyroidism", "ckd", "aki", "copd", "asthma",
    "respiratory_failure", "pulmonary_fibrosis", "liver_disease",
    "anemia", "smoking_history", "cancer_history", "depression",
]

DEMOGRAPHICS = ["sex", "age", "bmi"]
ALPHA_RIDGE = 1.0


def residualize(X_train, X_test, D_train, D_test, alpha=ALPHA_RIDGE):
    m = Ridge(alpha=alpha)
    m.fit(D_train, X_train)
    return X_train - m.predict(D_train), X_test - m.predict(D_test)


def disease_auroc(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


def main():
    print("Loading data...")
    canonical_ids = load_canonical_ids()
    split = load_split()
    metadata = load_metadata()
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    id_to_order = {did: i for i, did in enumerate(canonical_ids)}
    meta_df = metadata[metadata["dicom_id"].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df["dicom_id"].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)

    _, labels, masks, all_diseases = merge_chexpert(meta_df)
    disease_indices = {d: all_diseases.index(d) for d in KEY_DISEASES}

    sex_all = meta_df["gender_binary"].values.astype(float)
    age_all = meta_df["age"].values.astype(float)
    bmi_all = meta_df["bmi"].values.astype(float)
    age_mean = np.nanmean(age_all[train_idx])
    age_std = np.nanstd(age_all[train_idx])
    bmi_mean = np.nanmean(bmi_all[train_idx])
    bmi_std = np.nanstd(bmi_all[train_idx])

    comorb_arrays = {c: meta_df[c].values.astype(float) for c in COMORBIDITIES}

    demo_valid_tr = (np.isfinite(sex_all[train_idx]) &
                     np.isfinite(age_all[train_idx]) &
                     np.isfinite(bmi_all[train_idx]))
    demo_valid_te = (np.isfinite(sex_all[test_idx]) &
                     np.isfinite(age_all[test_idx]) &
                     np.isfinite(bmi_all[test_idx]))

    rows = []
    t0 = time.time()
    for model_name in OVERLAP_MODELS:
        print(f"\n{'='*60}\n  MODEL: {model_name}\n{'='*60}")
        try:
            emb = get_aligned_embeddings(model_name, canonical_ids)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb)

        # Demographics
        tr_demo = train_idx[demo_valid_tr]
        te_demo = test_idx[demo_valid_te]
        X_tr_demo, X_te_demo = emb_s[tr_demo], emb_s[te_demo]
        for attr in DEMOGRAPHICS:
            if attr == "sex":
                D_tr = sex_all[tr_demo].reshape(-1, 1)
                D_te = sex_all[te_demo].reshape(-1, 1)
            elif attr == "age":
                D_tr = ((age_all[tr_demo] - age_mean) / age_std).reshape(-1, 1)
                D_te = ((age_all[te_demo] - age_mean) / age_std).reshape(-1, 1)
            else:
                D_tr = ((bmi_all[tr_demo] - bmi_mean) / bmi_std).reshape(-1, 1)
                D_te = ((bmi_all[te_demo] - bmi_mean) / bmi_std).reshape(-1, 1)
            X_tr_r, X_te_r = residualize(X_tr_demo, X_te_demo, D_tr, D_te)
            for d in KEY_DISEASES:
                di = disease_indices[d]
                mt = masks[tr_demo, di].astype(bool)
                me = masks[te_demo, di].astype(bool)
                yt, ye = labels[tr_demo, di], labels[te_demo, di]
                if mt.sum() < 50 or me.sum() < 50 or yt[mt].sum() < 10 or ye[me].sum() < 10:
                    continue
                try:
                    ab = disease_auroc(X_tr_demo[mt], yt[mt], X_te_demo[me], ye[me])
                    ar = disease_auroc(X_tr_r[mt], yt[mt], X_te_r[me], ye[me])
                    rows.append({
                        "model": model_name, "attribute": attr,
                        "attribute_type": "demographic", "disease": d,
                        "auroc_baseline": ab, "auroc_residualized": ar,
                        "auroc_drop": ab - ar,
                    })
                except Exception as e:
                    print(f"    ERR {attr}/{d}: {e}")

        # Comorbidities (full cohort)
        X_tr_all = emb_s[train_idx]
        X_te_all = emb_s[test_idx]
        for attr in COMORBIDITIES:
            vals_tr = comorb_arrays[attr][train_idx]
            vals_te = comorb_arrays[attr][test_idx]
            if (np.isfinite(vals_tr).sum() < 100 or
                np.isfinite(vals_te).sum() < 100):
                continue
            D_tr = vals_tr.reshape(-1, 1)
            D_te = vals_te.reshape(-1, 1)
            X_tr_r, X_te_r = residualize(X_tr_all, X_te_all, D_tr, D_te)
            for d in KEY_DISEASES:
                di = disease_indices[d]
                mt = masks[train_idx, di].astype(bool)
                me = masks[test_idx, di].astype(bool)
                yt, ye = labels[train_idx, di], labels[test_idx, di]
                if mt.sum() < 50 or me.sum() < 50 or yt[mt].sum() < 10 or ye[me].sum() < 10:
                    continue
                try:
                    ab = disease_auroc(X_tr_all[mt], yt[mt], X_te_all[me], ye[me])
                    ar = disease_auroc(X_tr_r[mt], yt[mt], X_te_r[me], ye[me])
                    rows.append({
                        "model": model_name, "attribute": attr,
                        "attribute_type": "comorbidity", "disease": d,
                        "auroc_baseline": ab, "auroc_residualized": ar,
                        "auroc_drop": ab - ar,
                    })
                except Exception as e:
                    print(f"    ERR {attr}/{d}: {e}")
        print(f"  done in {time.time()-t0:.0f}s")

    out = pd.DataFrame(rows)
    out.to_csv("results/per_comorbidity_residualization_overlap.csv", index=False)
    print(f"\nSaved {len(out)} rows to results/per_comorbidity_residualization_overlap.csv")
    print(out.groupby("model")["auroc_drop"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
