"""
Per-race-group base rates and calibration (Section 4.1 / Table S17): probes the
label-noise and distribution-shift sources of the race subgroup gap.

Two outputs:
  (1) Base rates (model-free): per finding x race group, positive prevalence.
      Distribution shift in label frequency across groups is a direct,
      model-independent driver of per-group AUROC differences.
  (2) Calibration (per clean model x finding x race group): AUROC, Brier score,
      mean predicted probability vs observed prevalence (calibration gap).
      If subgroup gaps track base-rate / calibration differences rather than a
      demographic-to-prediction pathway, that supports the distribution-shift
      reading over a label-noise-only or explicit-bias reading.

Usage:
    PYTHONPATH=. python experiments/m4c_race_baserate_calibration.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import MIMIC_IV_ADMISSIONS
from experiments.data import (load_metadata, load_canonical_ids, load_split,
                              get_aligned_embeddings, merge_chexpert)

ADMISSIONS_PATH = MIMIC_IV_ADMISSIONS
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CLEAN_MODELS = ["ResNet50-ImageNet", "DINOv2-base", "BiomedCLIP",
                "XRV-DenseNet-nih", "CLIP-ViT-B16", "ConvNeXtV2-Base"]
KEY_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
                "Pneumonia", "Pneumothorax", "Support Devices"]
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


def main():
    t0 = time.time()
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

    race_pt = patient_race_table()
    race_map = dict(zip(race_pt["subject_id"], race_pt["race_group"]))
    race_image = np.array([race_map.get(s, "Unknown") for s in meta_df["subject_id"].values])

    # ---------- (1) Base rates (test set, model-free) ----------
    base_rows = []
    for d in KEY_DISEASES:
        di = dz_idx[d]
        me = masks[test_idx, di]
        ye = labels[test_idx, di]
        rte = race_image[test_idx]
        for cat in RACE_CATEGORIES:
            cm = me & (rte == cat)
            n = int(cm.sum())
            if n < 30:
                continue
            base_rows.append({"disease": d, "race_group": cat, "n": n,
                              "n_pos": int(ye[cm].sum()),
                              "prevalence": float(ye[cm].mean())})
    base = pd.DataFrame(base_rows)
    base.to_csv(os.path.join(RESULT_DIR, "m4c_race_baserate.csv"), index=False)

    # prevalence spread across groups per finding
    spread = (base.groupby("disease")["prevalence"]
              .agg(["min", "max"]).assign(spread=lambda x: x["max"] - x["min"])
              .sort_values("spread", ascending=False))
    print("Per-finding positive-prevalence spread across race groups (test set):")
    print(spread.round(3).to_string())

    # ---------- (2) Calibration (per clean model x finding x race group) ----------
    cal_rows = []
    for model_name in CLEAN_MODELS:
        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb)
        has_race = race_image != "Unknown"
        for d in KEY_DISEASES:
            di = dz_idx[d]
            mt = masks[train_idx, di] & has_race[train_idx]
            me = masks[test_idx, di] & has_race[test_idx]
            if mt.sum() < 50 or me.sum() < 50:
                continue
            trs = train_idx[mt]; tes = test_idx[me]
            yt = labels[trs, di]; ye = labels[tes, di]
            if yt.sum() < 10 or ye.sum() < 10:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(emb_s[trs], yt)
            p = clf.predict_proba(emb_s[tes])[:, 1]
            rte = race_image[tes]
            for cat in RACE_CATEGORIES:
                cm = rte == cat
                if cm.sum() < 30 or ye[cm].sum() < 5 or (1 - ye[cm]).sum() < 5:
                    continue
                try:
                    auc = roc_auc_score(ye[cm], p[cm])
                except Exception:
                    continue
                cal_rows.append({
                    "model": model_name, "disease": d, "race_group": cat,
                    "n": int(cm.sum()), "auroc": auc,
                    "brier": brier_score_loss(ye[cm], p[cm]),
                    "mean_pred": float(p[cm].mean()),
                    "prevalence": float(ye[cm].mean()),
                    "calib_gap": float(p[cm].mean() - ye[cm].mean()),
                })
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(os.path.join(RESULT_DIR, "m4c_race_calibration.csv"), index=False)

    # Correlate per-group AUROC with per-group base rate -> is gap a base-rate effect?
    print("\nCalibration summary (mean across clean models x findings, per race group):")
    g = cal.groupby("race_group").agg(
        auroc=("auroc", "mean"), brier=("brier", "mean"),
        calib_gap=("calib_gap", "mean"), prevalence=("prevalence", "mean")).round(4)
    print(g.to_string())

    # Spearman: does per-cell AUROC track per-cell prevalence (distribution shift)?
    from scipy.stats import spearmanr
    rho, pval = spearmanr(cal["auroc"], cal["prevalence"])
    print(f"\nSpearman(per-cell AUROC, per-cell prevalence) = {rho:.3f} (p={pval:.2e})")
    print(f"Saved to {RESULT_DIR}/  | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
