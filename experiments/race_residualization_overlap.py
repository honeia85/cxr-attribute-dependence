"""
Race residualization + subgroup fairness on the 3 overlap models (RAD-DINO,
CheXzero, CheSS-ResNet50) — the Yang et al. 2025 Sci Adv replication
(CheXzero is their driving example).

Runs ONLY on the 3 overlap models. The 6 clean-model race results already
live in results/race_residualization_mimic_cxr.csv and
results/race_subgroup_summary_mimic_cxr.csv.

Output:
  results/race_residualization_overlap.csv        (30 rows: 3 x 10)
  results/race_subgroup_summary_overlap.csv       (30 rows: 3 x 10 max-min gap)
  results/race_subgroup_gaps_overlap.csv          (per-group AUROC detail)

Usage:
    PYTHONPATH=. python experiments/race_residualization_overlap.py
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

OVERLAP_MODELS = ["RAD-DINO", "CheXzero", "CheSS-ResNet50"]

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


def disease_auroc(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


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
    disease_indices = {d: all_diseases.index(d) for d in KEY_DISEASES}

    race_pt = patient_race_table()
    race_map = dict(zip(race_pt["subject_id"], race_pt["race_group"]))
    race_image = np.array([race_map.get(s, "Unknown")
                           for s in meta_df["subject_id"].values])
    bw_mask = np.isin(race_image, ["Black", "White"])
    race_binary = (race_image == "Black").astype(float)
    has_race = race_image != "Unknown"
    print(f"Black+White subset: {bw_mask.sum():,} images, {race_binary[bw_mask].mean():.1%} Black")

    resid_rows, sg_rows = [], []

    for model_name in OVERLAP_MODELS:
        print(f"\n--- {model_name} ---")
        try:
            emb = get_aligned_embeddings(model_name, canonical_ids)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb)

        # Part A: Residualization on Black+White subset
        tr_bw = train_idx[bw_mask[train_idx]]
        te_bw = test_idx[bw_mask[test_idx]]
        X_tr, X_te = emb_s[tr_bw], emb_s[te_bw]
        D_tr = race_binary[tr_bw].reshape(-1, 1)
        D_te = race_binary[te_bw].reshape(-1, 1)
        Xr_tr, Xr_te = residualize(X_tr, X_te, D_tr, D_te)

        for d in KEY_DISEASES:
            di = disease_indices[d]
            mt = masks[tr_bw, di].astype(bool)
            me = masks[te_bw, di].astype(bool)
            yt, ye = labels[tr_bw, di], labels[te_bw, di]
            if mt.sum() < 50 or me.sum() < 50 or yt[mt].sum() < 10 or ye[me].sum() < 10:
                continue
            try:
                ab = disease_auroc(X_tr[mt], yt[mt], X_te[me], ye[me])
                ar = disease_auroc(Xr_tr[mt], yt[mt], Xr_te[me], ye[me])
                resid_rows.append({
                    "model": model_name, "attribute": "race_black_vs_white",
                    "disease": d, "auroc_baseline": ab,
                    "auroc_residualized": ar, "auroc_drop": ab - ar,
                })
            except Exception as e:
                print(f"  ERR {d}: {e}")

        # Part B: 4-category race subgroup AUROC gap
        for d in KEY_DISEASES:
            di = disease_indices[d]
            mt = masks[train_idx, di].astype(bool) & has_race[train_idx]
            me = masks[test_idx, di].astype(bool) & has_race[test_idx]
            if mt.sum() < 50 or me.sum() < 50:
                continue
            tr_sel = train_idx[mt]
            te_sel = test_idx[me]
            X_tr_sg, y_tr_sg = emb_s[tr_sel], labels[tr_sel, di]
            X_te_sg, y_te_sg = emb_s[te_sel], labels[te_sel, di]
            if y_tr_sg.sum() < 10 or y_te_sg.sum() < 10:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(X_tr_sg, y_tr_sg)
            p_te = clf.predict_proba(X_te_sg)[:, 1]
            race_te = race_image[te_sel]

            group_aucs = {}
            for cat in RACE_CATEGORIES:
                mk = race_te == cat
                if mk.sum() < 30:
                    continue
                yc, pc = y_te_sg[mk], p_te[mk]
                if yc.sum() < 5 or (1 - yc).sum() < 5:
                    continue
                try:
                    group_aucs[cat] = roc_auc_score(yc, pc)
                except Exception:
                    continue
            if len(group_aucs) < 2:
                continue
            gap = max(group_aucs.values()) - min(group_aucs.values())
            sg_rows.append({
                "model": model_name, "disease": d,
                "n_groups": len(group_aucs),
                "auroc_max": max(group_aucs.values()),
                "auroc_min": min(group_aucs.values()),
                "gap_maxmin": gap,
                **{f"auroc_{c}": group_aucs.get(c, np.nan)
                   for c in RACE_CATEGORIES},
            })

    resid_df = pd.DataFrame(resid_rows)
    sg_df = pd.DataFrame(sg_rows)
    resid_df.to_csv("results/race_residualization_overlap.csv", index=False)
    sg_df.to_csv("results/race_subgroup_summary_overlap.csv", index=False)
    print(f"\nResidualization: {len(resid_df)} rows")
    print(resid_df.groupby("model")["auroc_drop"].agg(["mean", "std"]).round(4))
    print(f"\nSubgroup gaps: {len(sg_df)} rows")
    print(sg_df.groupby("model")["gap_maxmin"].agg(["mean", "max", "count"]).round(4))
    print(f"\nTime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
