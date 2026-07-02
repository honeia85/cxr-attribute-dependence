"""
Race as the 24th attribute on MIMIC-CXR.

Two analyses, both on the 6 clean models:
  (A) Residualization: treat race_black_vs_white (1=Black, 0=White, Asian/Other
      excluded) as a binary attribute, compute AUROC drop per (model, finding)
      on the Black+White subset. Also compute Black-vs-White log(OR) per
      finding for integration into the nested regression.
  (B) Fairness subgroup gap: 4-category race (White, Black, Asian, Other)
      subgroup AUROC per (model, finding), and max-min gap per cell.

Output:
  results/race_residualization_mimic_cxr.csv     (60 rows: 6 x 10)
  results/race_or_mimic_cxr.csv                  (10 rows: one per finding)
  results/race_subgroup_gaps_mimic_cxr.csv       (240 rows: 6 x 10 x 4)
  results/race_subgroup_summary_mimic_cxr.csv    (60 rows: 6 x 10 max-min gap)

Usage:
    PYTHONPATH=. python experiments/race_residualization_mimic_cxr.py
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

CLEAN_MODELS = [
    "ResNet50-ImageNet", "DINOv2-base", "BiomedCLIP",
    "XRV-DenseNet-nih", "CLIP-ViT-B16", "ConvNeXtV2-Base",
]
# Overlap models (pretraining exposure to MIMIC-CXR). Added for the
# Yang et al. 2025 Sci Adv replication: CheXzero is their driving example.
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


def disease_auroc(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


def haldane_or(event_exposed, event_unexposed, n_exposed, n_unexposed):
    """Haldane-corrected odds ratio."""
    a = event_exposed + 0.5
    b = (n_exposed - event_exposed) + 0.5
    c = event_unexposed + 0.5
    d = (n_unexposed - event_unexposed) + 0.5
    return (a * d) / (b * c)


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

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

    # Race labels per patient, then per image
    race_pt = patient_race_table()
    race_map = dict(zip(race_pt["subject_id"], race_pt["race_group"]))
    race_image = np.array([race_map.get(s, "Unknown")
                           for s in meta_df["subject_id"].values])
    print("Race distribution (image level):",
          pd.Series(race_image).value_counts().to_dict())

    # Black-vs-White subset mask (binary attribute)
    bw_mask = np.isin(race_image, ["Black", "White"])
    race_binary = (race_image == "Black").astype(float)
    print(f"Black+White subset: {bw_mask.sum():,} images "
          f"({race_binary[bw_mask].mean():.1%} Black)")

    # ------------------------------------------------------------------
    # Part A: Residualization on Black+White subset
    # ------------------------------------------------------------------
    resid_rows = []
    or_rows = []
    for model_name in ALL_MODELS:
        print(f"\n--- {model_name} ---")
        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb)

        # Apply BW filter on top of existing train/test split
        tr_bw = train_idx[bw_mask[train_idx]]
        te_bw = test_idx[bw_mask[test_idx]]
        X_tr = emb_s[tr_bw]
        X_te = emb_s[te_bw]
        D_tr = race_binary[tr_bw].reshape(-1, 1)
        D_te = race_binary[te_bw].reshape(-1, 1)

        Xr_tr, Xr_te = residualize(X_tr, X_te, D_tr, D_te)

        for d in KEY_DISEASES:
            di = disease_indices[d]
            mt = masks[tr_bw, di].astype(bool)
            me = masks[te_bw, di].astype(bool)
            yt = labels[tr_bw, di]
            ye = labels[te_bw, di]
            if mt.sum() < 50 or me.sum() < 50 or yt[mt].sum() < 10 or ye[me].sum() < 10:
                continue
            try:
                ab = disease_auroc(X_tr[mt], yt[mt], X_te[me], ye[me])
                ar = disease_auroc(Xr_tr[mt], yt[mt], Xr_te[me], ye[me])
                resid_rows.append({
                    "model": model_name,
                    "attribute": "race_black_vs_white",
                    "attribute_type": "demographic",
                    "disease": d,
                    "auroc_baseline": ab,
                    "auroc_residualized": ar,
                    "auroc_drop": ab - ar,
                    "n_train_bw": int(mt.sum()),
                    "n_test_bw": int(me.sum()),
                })
            except Exception as e:
                print(f"  ERR {d}: {e}")

        # Black-vs-White OR per finding (identical across models; compute once,
        # then copy to match the per-model residualization schema).
        if model_name == ALL_MODELS[0]:
            for d in KEY_DISEASES:
                di = disease_indices[d]
                mt = masks[train_idx, di].astype(bool) & bw_mask[train_idx]
                yt = labels[train_idx, di]
                race_tr = race_binary[train_idx]
                ev_black = int(yt[mt & (race_tr == 1)].sum())
                n_black = int((mt & (race_tr == 1)).sum())
                ev_white = int(yt[mt & (race_tr == 0)].sum())
                n_white = int((mt & (race_tr == 0)).sum())
                or_val = haldane_or(ev_black, ev_white, n_black, n_white)
                or_rows.append({
                    "attribute": "race_black_vs_white",
                    "disease": d,
                    "events_black": ev_black, "n_black": n_black,
                    "events_white": ev_white, "n_white": n_white,
                    "odds_ratio": or_val,
                    "log_or": np.log(or_val),
                    "abs_log_or": abs(np.log(or_val)),
                })

    pd.DataFrame(resid_rows).to_csv(
        "results/race_residualization_mimic_cxr.csv", index=False)
    pd.DataFrame(or_rows).to_csv(
        "results/race_or_mimic_cxr.csv", index=False)
    print(f"\nResidualization done: {len(resid_rows)} rows")
    print(f"Race OR table: {len(or_rows)} rows")

    # ------------------------------------------------------------------
    # Part B: Race subgroup fairness gaps (4-category)
    # ------------------------------------------------------------------
    print("\nPart B: subgroup fairness gaps...")
    has_race = race_image != "Unknown"
    sg_rows = []
    for model_name in ALL_MODELS:
        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb)

        for d in KEY_DISEASES:
            di = disease_indices[d]
            # Train a disease classifier on all race-coded training images
            mt = masks[train_idx, di].astype(bool) & has_race[train_idx]
            me = masks[test_idx, di].astype(bool) & has_race[test_idx]
            if mt.sum() < 50 or me.sum() < 50:
                continue
            tr_sel = train_idx[mt]
            te_sel = test_idx[me]
            X_tr = emb_s[tr_sel]
            y_tr = labels[tr_sel, di]
            X_te = emb_s[te_sel]
            y_te = labels[te_sel, di]
            if y_tr.sum() < 10 or y_te.sum() < 10:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(X_tr, y_tr)
            prob_te = clf.predict_proba(X_te)[:, 1]
            # Per-race-group AUROC on test set
            race_te = race_image[te_sel]
            for cat in RACE_CATEGORIES:
                cat_mask = race_te == cat
                if cat_mask.sum() < 30:
                    continue
                y_cat = y_te[cat_mask]
                p_cat = prob_te[cat_mask]
                if y_cat.sum() < 5 or (1 - y_cat).sum() < 5:
                    continue
                try:
                    auc_cat = roc_auc_score(y_cat, p_cat)
                except Exception:
                    continue
                sg_rows.append({
                    "model": model_name,
                    "disease": d,
                    "race_group": cat,
                    "n": int(cat_mask.sum()),
                    "n_pos": int(y_cat.sum()),
                    "auroc": auc_cat,
                })
    sg_df = pd.DataFrame(sg_rows)
    sg_df.to_csv("results/race_subgroup_gaps_mimic_cxr.csv", index=False)

    # Summary per (model, disease): max-min gap across 4 groups
    summary = []
    for (m, d), g in sg_df.groupby(["model", "disease"]):
        if len(g) < 2:
            continue
        gap = g["auroc"].max() - g["auroc"].min()
        summary.append({
            "model": m, "disease": d,
            "n_groups": len(g),
            "max_auroc_race_group": g.loc[g["auroc"].idxmax(), "race_group"],
            "min_auroc_race_group": g.loc[g["auroc"].idxmin(), "race_group"],
            "auroc_max": g["auroc"].max(),
            "auroc_min": g["auroc"].min(),
            "gap_maxmin": gap,
        })
    pd.DataFrame(summary).to_csv(
        "results/race_subgroup_summary_mimic_cxr.csv", index=False)
    print(f"Subgroup gaps: {len(sg_rows)} cells, "
          f"{len(summary)} summary rows")
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
