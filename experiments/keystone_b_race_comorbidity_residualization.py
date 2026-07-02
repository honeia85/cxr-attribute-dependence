"""
Direct race-gap test (Section 3.5 / Table S15).

Selecting the top-3 attributes by odds ratio and then testing demographic gaps
conflates a high-OR clinical confounder with a fairness-relevant protected
attribute: the main fairness test (Section 3.4) residualized heart failure /
atrial fibrillation / age and measured sex and age gaps, so race was never in the
residualization set. The direct test residualizes race together with its correlated
cardiac comorbidities (heart failure, atrial fibrillation) and then measures the
four-category race subgroup gap.

We run a decomposition of FOUR conditions per (model, finding) so we can quantify
how much of the race subgroup gap is mediated by race-correlated cardiac disease:
  (0) baseline                 -- no residualization
  (1) race-only                -- residualize 4-category race (one-hot, White ref)
  (2) cardiac-only             -- residualize heart_failure + atrial_fibrillation
  (3) joint race + cardiac     -- residualize race one-hot + HF + afib

Output:
  results/keystone_b_race_gap_decomposition.csv         (per model x finding x condition)
  results/keystone_b_race_gap_summary.csv               (per condition: mean gap, reduction)

Usage:
    PYTHONPATH=. python experiments/keystone_b_race_comorbidity_residualization.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import SEED, MIMIC_IV_ADMISSIONS
from experiments.data import (load_metadata, load_canonical_ids, load_split,
                              get_aligned_embeddings, merge_chexpert)

ADMISSIONS_PATH = MIMIC_IV_ADMISSIONS
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CLEAN_MODELS = [
    "ResNet50-ImageNet", "DINOv2-base", "BiomedCLIP",
    "XRV-DenseNet-nih", "CLIP-ViT-B16", "ConvNeXtV2-Base",
]
OVERLAP_MODELS = ["RAD-DINO", "CheXzero", "CheSS-ResNet50"]
ALL_MODELS = CLEAN_MODELS + OVERLAP_MODELS

KEY_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Support Devices",
]
RACE_CATEGORIES = ["White", "Black", "Asian", "Other"]


def collapse_race(label):
    if pd.isna(label):
        return "Unknown"
    s = str(label).upper()
    if s.startswith("WHITE") or s == "PORTUGUESE":
        return "White"
    if s.startswith("BLACK"):
        return "Black"
    if s.startswith("ASIAN") or s == "SOUTH ASIAN":
        return "Asian"
    if s in {"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"}:
        return "Unknown"
    return "Other"


def patient_race_table():
    adm = pd.read_csv(ADMISSIONS_PATH, usecols=["subject_id", "race"])
    adm["race_group"] = adm["race"].map(collapse_race)

    def majority(group):
        non_unk = group[group != "Unknown"]
        pool = non_unk if len(non_unk) > 0 else group
        return pool.mode().iloc[0]

    return adm.groupby("subject_id")["race_group"].apply(majority).reset_index()


def residualize(X_tr, X_te, D_tr, D_te, alpha=1.0):
    m = Ridge(alpha=alpha)
    m.fit(D_tr, X_tr)
    return X_tr - m.predict(D_tr), X_te - m.predict(D_te)


def race_subgroup_gap(X_tr, y_tr, X_te, y_te, race_te):
    """Train finding classifier; return 4-category race AUROC max-min gap + per-group."""
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    aucs = {}
    for cat in RACE_CATEGORIES:
        cm = race_te == cat
        if cm.sum() < 30 or y_te[cm].sum() < 5 or (1 - y_te[cm]).sum() < 5:
            continue
        try:
            aucs[cat] = roc_auc_score(y_te[cm], p[cm])
        except Exception:
            pass
    if len(aucs) < 2:
        return None
    return max(aucs.values()) - min(aucs.values()), aucs


def main():
    t0 = time.time()
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
    dz_idx = {d: all_diseases.index(d) for d in KEY_DISEASES}

    # Comorbidity vectors (image level)
    hf = meta_df["heart_failure"].fillna(0).values.astype(float)
    af = meta_df["atrial_fibrillation"].fillna(0).values.astype(float)

    # Race per patient -> image
    race_pt = patient_race_table()
    race_map = dict(zip(race_pt["subject_id"], race_pt["race_group"]))
    race_image = np.array([race_map.get(s, "Unknown") for s in meta_df["subject_id"].values])
    has_race = race_image != "Unknown"
    print("Race dist:", pd.Series(race_image).value_counts().to_dict())

    # One-hot race (drop White reference): Black, Asian, Other dummies
    def race_onehot(idx):
        r = race_image[idx]
        return np.column_stack([(r == "Black"), (r == "Asian"), (r == "Other")]).astype(float)

    # Residualization design matrices per condition (on race-coded rows only,
    # so the four conditions share an identical evaluation cohort)
    CONDITIONS = ["baseline", "race_only", "cardiac_only", "joint_race_cardiac"]

    rows = []
    for model_name in ALL_MODELS:
        print(f"\n--- {model_name} ---")
        try:
            emb = get_aligned_embeddings(model_name, canonical_ids)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb)

        # Restrict to race-coded rows (shared cohort across conditions)
        tr = train_idx[has_race[train_idx]]
        te = test_idx[has_race[test_idx]]
        X_tr0, X_te0 = emb_s[tr], emb_s[te]
        race_te = race_image[te]

        # Build per-condition design matrices
        D = {
            "race_only": (np.column_stack([race_onehot(tr)]),
                          np.column_stack([race_onehot(te)])),
            "cardiac_only": (np.column_stack([hf[tr], af[tr]]),
                             np.column_stack([hf[te], af[te]])),
            "joint_race_cardiac": (np.column_stack([race_onehot(tr), hf[tr], af[tr]]),
                                   np.column_stack([race_onehot(te), hf[te], af[te]])),
        }
        # Pre-residualize embeddings once per condition
        Xcond = {"baseline": (X_tr0, X_te0)}
        for c, (Dtr, Dte) in D.items():
            Xcond[c] = residualize(X_tr0, X_te0, Dtr, Dte)

        for d in KEY_DISEASES:
            di = dz_idx[d]
            mt = masks[tr, di]; me = masks[te, di]
            yt_full = labels[tr, di]; ye_full = labels[te, di]
            if mt.sum() < 50 or me.sum() < 50 or yt_full[mt].sum() < 10 or ye_full[me].sum() < 10:
                continue
            row = {"model": model_name, "disease": d}
            ok = True
            for c in CONDITIONS:
                Xt, Xe = Xcond[c]
                res = race_subgroup_gap(Xt[mt], yt_full[mt], Xe[me], ye_full[me], race_te[me])
                if res is None:
                    ok = False
                    break
                row[f"gap_{c}"] = res[0]
            if ok:
                row["red_race_only"] = row["gap_baseline"] - row["gap_race_only"]
                row["red_cardiac_only"] = row["gap_baseline"] - row["gap_cardiac_only"]
                row["red_joint"] = row["gap_baseline"] - row["gap_joint_race_cardiac"]
                rows.append(row)
        sub = [r for r in rows if r["model"] == model_name]
        if sub:
            gb = np.mean([r["gap_baseline"] for r in sub])
            gj = np.mean([r["gap_joint_race_cardiac"] for r in sub])
            print(f"  baseline gap={gb:.4f}  joint gap={gj:.4f}  "
                  f"reduction={gb-gj:+.4f} ({100*(gb-gj)/max(gb,1e-9):+.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "keystone_b_race_gap_decomposition.csv"), index=False)

    # Summary per condition (clean models)
    clean = df[df["model"].isin(CLEAN_MODELS)]
    summ = []
    base = clean["gap_baseline"].mean()
    for c in CONDITIONS:
        g = clean[f"gap_{c}"]
        summ.append({"condition": c, "mean_gap": g.mean(),
                     "gap_reduction_vs_baseline": base - g.mean(),
                     "pct_reduction": 100 * (base - g.mean()) / max(base, 1e-9),
                     "n_cells": len(g)})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(RESULT_DIR, "keystone_b_race_gap_summary.csv"), index=False)

    print("\n" + "=" * 64)
    print("KEYSTONE B SUMMARY (6 clean models, 4-category race subgroup gap)")
    print("=" * 64)
    for _, r in sdf.iterrows():
        print(f"  {r['condition']:22s} gap={r['mean_gap']:.4f}  "
              f"reduction={r['gap_reduction_vs_baseline']:+.4f} "
              f"({r['pct_reduction']:+.1f}%)")
    print(f"\nInterpretation: joint race+cardiac residualization closes "
          f"{sdf[sdf.condition=='joint_race_cardiac']['pct_reduction'].iloc[0]:.1f}% "
          f"of the baseline race gap.")
    print(f"Saved to {RESULT_DIR}/  | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
