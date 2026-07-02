"""
Race probing on MIMIC-CXR with 6 clean foundation models.

Companion experiment establishing race encoding on MIMIC-CXR itself, directly
comparable to Gichoya et al. (Lancet Digit Health 2022), who reported near-ceiling
race detection on MIMIC-CXR.

This script reproduces Gichoya's setting on MIMIC-CXR with our 6 frozen encoders
(linear probe) and reports:
  - Black vs White binary AUROC (Gichoya's primary metric)
  - One-vs-rest multiclass AUROC for {White, Black, Asian, Other}

Patient-level 70/30 train/test split, sex-stratified (matching the rest of the
manuscript's analysis protocol).

Usage:
    python -m experiments.race_probing_mimic_cxr
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import CLEAN_MODELS, EMBEDDING_DIR, METADATA_CSV, ID_COL, SEED, MIMIC_IV_ADMISSIONS


ADMISSIONS_PATH = Path(MIMIC_IV_ADMISSIONS)
RESULTS_PATH = Path(__file__).parent.parent / "results" / "race_probing_mimic_cxr.csv"


def collapse_race(label):
    """Map MIMIC-IV admissions race label to {White, Black, Asian, Other, Unknown}."""
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
    print("Loading MIMIC-IV admissions for race labels ...")
    adm = pd.read_csv(ADMISSIONS_PATH, usecols=["subject_id", "race"])
    adm["race_group"] = adm["race"].map(collapse_race)
    # If a patient has multiple admissions, take the most frequent label
    # (excluding Unknown if any non-Unknown admission exists).
    def majority(group):
        non_unk = group[group != "Unknown"]
        pool = non_unk if len(non_unk) > 0 else group
        return pool.mode().iloc[0]

    per_pt = adm.groupby("subject_id")["race_group"].apply(majority).reset_index()
    print(f"  {len(per_pt):,} patients with a race label")
    print(per_pt["race_group"].value_counts())
    return per_pt


def load_embeddings(model_name, dicom_ids_target):
    emb = np.load(Path(EMBEDDING_DIR) / f"{model_name}_embeddings.npy")
    ids = np.load(Path(EMBEDDING_DIR) / f"{model_name}_dicom_ids.npy", allow_pickle=True)
    id_to_row = {d: i for i, d in enumerate(ids)}
    rows = np.array([id_to_row.get(d, -1) for d in dicom_ids_target])
    valid = rows >= 0
    print(f"  {model_name}: {valid.sum():,}/{len(dicom_ids_target):,} aligned")
    return emb[rows[valid]], valid


def main():
    rng = np.random.default_rng(SEED)

    # ----- 1. Load metadata + race labels ---------------------------------
    meta = pd.read_csv(METADATA_CSV, usecols=[ID_COL, "subject_id", "gender_binary"])
    race = patient_race_table()
    meta = meta.merge(race, on="subject_id", how="inner")
    meta = meta[meta["race_group"].isin(["White", "Black", "Asian", "Other"])].copy()

    # Patient-level 70/30 sex-stratified split
    patients = meta[["subject_id", "gender_binary"]].drop_duplicates()
    pos = patients[patients["gender_binary"] == 1]["subject_id"].values
    neg = patients[patients["gender_binary"] == 0]["subject_id"].values
    rng.shuffle(pos)
    rng.shuffle(neg)
    train_subj = set(np.concatenate([pos[: int(0.7 * len(pos))],
                                     neg[: int(0.7 * len(neg))]]))
    test_subj = set(np.concatenate([pos[int(0.7 * len(pos)):],
                                    neg[int(0.7 * len(neg)):]]))
    meta["split"] = meta["subject_id"].apply(
        lambda s: "train" if s in train_subj else ("test" if s in test_subj else "drop")
    )
    meta = meta[meta["split"] != "drop"].reset_index(drop=True)
    print(f"\nTotal images: {len(meta):,} ({meta['subject_id'].nunique():,} patients)")
    print("Train/test by image:", meta["split"].value_counts().to_dict())
    print("Race distribution (image level):")
    print(meta["race_group"].value_counts())

    dicom_ids = meta[ID_COL].values
    race_y = meta["race_group"].values
    is_train = (meta["split"].values == "train")

    classes = np.array(["White", "Black", "Asian", "Other"])

    rows = []
    for model_name in CLEAN_MODELS:
        print(f"\n===== {model_name} =====")
        emb, valid = load_embeddings(model_name, dicom_ids)
        if valid.sum() == 0:
            continue
        y = race_y[valid]
        tr = is_train[valid]
        scaler = StandardScaler().fit(emb[tr])
        emb_s = scaler.transform(emb)
        X_tr, y_tr = emb_s[tr], y[tr]
        X_te, y_te = emb_s[~tr], y[~tr]

        # Multiclass linear probe
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=2000, n_jobs=-1, random_state=SEED,
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        sklearn_classes = list(clf.classes_)
        macro_auroc = roc_auc_score(y_te, proba, multi_class="ovr",
                                    average="macro", labels=sklearn_classes)
        per_class = {}
        for cls in classes:
            try:
                per_class[cls] = roc_auc_score(
                    (y_te == cls).astype(int),
                    proba[:, sklearn_classes.index(cls)],
                )
            except Exception:
                per_class[cls] = float("nan")

        # Black-vs-White binary (Gichoya's primary metric)
        bw_mask_tr = np.isin(y_tr, ["Black", "White"])
        bw_mask_te = np.isin(y_te, ["Black", "White"])
        clf_bw = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                    max_iter=2000, n_jobs=-1, random_state=SEED)
        clf_bw.fit(X_tr[bw_mask_tr], (y_tr[bw_mask_tr] == "Black").astype(int))
        proba_bw = clf_bw.predict_proba(X_te[bw_mask_te])[:, 1]
        bw_auroc = roc_auc_score((y_te[bw_mask_te] == "Black").astype(int), proba_bw)

        print(f"  Macro AUROC (4-class OvR): {macro_auroc:.3f}")
        print(f"  Black vs White AUROC:      {bw_auroc:.3f}")
        for cls in classes:
            print(f"    {cls:6s} OvR AUROC: {per_class[cls]:.3f}")

        rows.append({
            "model": model_name,
            "n_train": int(tr.sum()),
            "n_test": int((~tr).sum()),
            "macro_auroc_4class": macro_auroc,
            "black_vs_white_auroc": bw_auroc,
            **{f"auroc_{cls}_ovr": per_class[cls] for cls in classes},
        })

    out = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved: {RESULTS_PATH}")
    print("\nSummary:")
    print(out[["model", "macro_auroc_4class", "black_vs_white_auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()
