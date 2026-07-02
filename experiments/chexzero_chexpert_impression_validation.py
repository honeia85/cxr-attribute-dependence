"""
CheXzero sanity/external check on CheXpert Plus using impression labels.

The previously prepared chexpert_plus_metadata.csv used report_fixed.json labels.
Those labels fail a basic finding-classification gate across models. CheXBERT
labels from impression_fixed.json recover expected finding AUROC, so this script
keeps demographics/splits from chexpert_plus_metadata.csv but replaces finding
labels from impression_fixed.json.

Outputs:
  results_chexpert_impression/chexzero_finding_baseline.csv
  results_chexpert_impression/chexzero_encoding.csv
  results_chexpert_impression/chexzero_demographic_dependence.csv
  results_chexpert_impression/chexzero_race_subgroup_gaps.csv
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(PROJECT_DIR)
EMB_DIR = os.path.join(PROJECT_DIR, "embeddings_chexpert")
META_CSV = os.path.join(PROJECT_DIR, "chexpert_plus_metadata.csv")
LABEL_JSON = os.path.join(ROOT_DIR, "chexpert_source", "impression_fixed.json")
OUT_DIR = os.path.join(PROJECT_DIR, "results_chexpert_impression")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "CheXzero"
FINDINGS = [
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
RACE_CATEGORIES = ["White", "Black", "Asian", "Other"]


def load_impression_labels(ids):
    wanted = set(ids)
    records = []
    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["path_to_image"] in wanted:
                records.append(rec)
    labels = pd.DataFrame(records).set_index("path_to_image")
    labels = labels.reindex(ids)
    missing = int(labels[FINDINGS].isna().all(axis=1).sum())
    if missing:
        print(f"WARNING: {missing} image ids missing all impression labels")
    return labels


def disease_auroc(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


def residualize(X_tr, X_te, D_tr, D_te, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(D_tr, X_tr)
    return X_tr - model.predict(D_tr), X_te - model.predict(D_te)


def valid_binary(values):
    return np.isin(values.astype(float), [0.0, 1.0])


def main():
    t0 = time.time()
    emb = np.load(os.path.join(EMB_DIR, f"{MODEL}_embeddings.npy"))
    ids = np.load(os.path.join(EMB_DIR, f"{MODEL}_image_ids.npy"), allow_pickle=True)
    ids = np.array([str(x) for x in ids])
    print(f"{MODEL}: embeddings {emb.shape}, ids {len(ids)}")

    meta = pd.read_csv(META_CSV).set_index("path_to_image").reindex(ids)
    labels = load_impression_labels(ids)
    split = meta["split"].values
    train_idx = np.flatnonzero(np.isin(split, ["train", "val"]))
    test_idx = np.flatnonzero(split == "test")
    print(f"train+val={len(train_idx):,} test={len(test_idx):,}")

    scaler = StandardScaler().fit(emb[train_idx])
    X = scaler.transform(emb)

    # Finding baseline gate.
    baseline_rows = []
    print("\n[Finding baseline, impression labels]")
    for finding in FINDINGS:
        y_raw = labels[finding].values.astype(float)
        mask = valid_binary(y_raw)
        y = (y_raw == 1.0).astype(int)
        mt = np.isin(np.arange(len(ids)), train_idx) & mask
        me = np.isin(np.arange(len(ids)), test_idx) & mask
        if mt.sum() < 100 or me.sum() < 50 or y[me].sum() < 10 or (1 - y[me]).sum() < 10:
            continue
        auc = disease_auroc(X[mt], y[mt], X[me], y[me])
        row = {
            "model": MODEL,
            "label_source": "impression_fixed.json",
            "finding": finding,
            "train_n": int(mt.sum()),
            "test_n": int(me.sum()),
            "test_prevalence": float(y[me].mean()),
            "auroc": float(auc),
        }
        baseline_rows.append(row)
        print(f"  {finding:28s} AUROC={auc:.3f} test_n={me.sum():5d} prev={y[me].mean():.2f}")
    pd.DataFrame(baseline_rows).to_csv(
        os.path.join(OUT_DIR, "chexzero_finding_baseline.csv"), index=False
    )

    # Attribute encoding.
    sex = meta["gender_binary"].values.astype(float)
    age = meta["age"].values.astype(float)
    bmi = meta["bmi"].values.astype(float)
    race_group = meta["race_group"].astype(str).values
    bw_mask = np.isin(race_group, ["Black", "White"])
    race_bin = (race_group == "Black").astype(float)
    has_race = np.isin(race_group, RACE_CATEGORIES)

    enc = {"model": MODEL, "label_source": "impression_fixed.json"}
    tr = train_idx[~np.isnan(sex[train_idx])]
    te = test_idx[~np.isnan(sex[test_idx])]
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], sex[tr])
    enc["enc_sex_auroc"] = float(roc_auc_score(sex[te], clf.predict_proba(X[te])[:, 1]))

    tr = train_idx[bw_mask[train_idx]]
    te = test_idx[bw_mask[test_idx]]
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], race_bin[tr])
    enc["enc_race_black_white_auroc"] = float(
        roc_auc_score(race_bin[te], clf.predict_proba(X[te])[:, 1])
    )

    tr = train_idx[~np.isnan(age[train_idx])]
    te = test_idx[~np.isnan(age[test_idx])]
    rg = Ridge(alpha=1.0).fit(X[tr], age[tr])
    enc["enc_age_r2"] = float(r2_score(age[te], rg.predict(X[te])))

    tr = train_idx[~np.isnan(bmi[train_idx])]
    te = test_idx[~np.isnan(bmi[test_idx])]
    rg = Ridge(alpha=1.0).fit(X[tr], bmi[tr])
    enc["enc_bmi_r2"] = float(r2_score(bmi[te], rg.predict(X[te])))

    print("\n[Encoding]")
    print(
        "  sex={enc_sex_auroc:.3f} race={enc_race_black_white_auroc:.3f} "
        "age_R2={enc_age_r2:.3f} bmi_R2={enc_bmi_r2:.3f}".format(**enc)
    )
    pd.DataFrame([enc]).to_csv(os.path.join(OUT_DIR, "chexzero_encoding.csv"), index=False)

    # Demographic dependence by linear residualization.
    dep_rows = []
    attr_specs = [
        ("sex", sex, None),
        ("race_black_vs_white", race_bin, bw_mask),
        ("age", age, None),
        ("bmi", bmi, None),
    ]
    print("\n[Demographic dependence]")
    for attr_name, attr, subset in attr_specs:
        valid = ~np.isnan(attr)
        if subset is not None:
            valid &= subset
        tr = train_idx[valid[train_idx]]
        te = test_idx[valid[test_idx]]
        D_tr = attr[tr].reshape(-1, 1)
        D_te = attr[te].reshape(-1, 1)
        if attr_name in ("age", "bmi"):
            mu = D_tr.mean()
            sd = D_tr.std() + 1e-9
            D_tr = (D_tr - mu) / sd
            D_te = (D_te - mu) / sd
        X_tr, X_te = X[tr], X[te]
        Xr_tr, Xr_te = residualize(X_tr, X_te, D_tr, D_te)
        drops = []
        for finding in FINDINGS:
            y_raw = labels[finding].values.astype(float)
            y = (y_raw == 1.0).astype(int)
            mt = valid_binary(y_raw[tr])
            me = valid_binary(y_raw[te])
            if mt.sum() < 100 or me.sum() < 50 or y[te][me].sum() < 10 or (1 - y[te][me]).sum() < 10:
                continue
            auc_base = disease_auroc(X_tr[mt], y[tr][mt], X_te[me], y[te][me])
            auc_resid = disease_auroc(Xr_tr[mt], y[tr][mt], Xr_te[me], y[te][me])
            drop = auc_base - auc_resid
            drops.append(drop)
            dep_rows.append(
                {
                    "model": MODEL,
                    "label_source": "impression_fixed.json",
                    "attribute": attr_name,
                    "finding": finding,
                    "auroc_baseline": float(auc_base),
                    "auroc_residualized": float(auc_resid),
                    "auroc_drop": float(drop),
                }
            )
        if drops:
            print(f"  {attr_name:22s} mean_drop={np.mean(drops):.4f} n={len(drops)}")
    pd.DataFrame(dep_rows).to_csv(
        os.path.join(OUT_DIR, "chexzero_demographic_dependence.csv"), index=False
    )

    # Race subgroup AUROC gaps.
    gap_rows = []
    print("\n[Race subgroup AUROC gaps]")
    for finding in FINDINGS:
        y_raw = labels[finding].values.astype(float)
        y = (y_raw == 1.0).astype(int)
        mt = valid_binary(y_raw) & np.isin(np.arange(len(ids)), train_idx) & has_race
        me = valid_binary(y_raw) & np.isin(np.arange(len(ids)), test_idx) & has_race
        if mt.sum() < 100 or me.sum() < 50 or y[me].sum() < 10 or (1 - y[me]).sum() < 10:
            continue
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[mt], y[mt])
        p = clf.predict_proba(X[me])[:, 1]
        y_te = y[me]
        race_te = race_group[me]
        aucs = {}
        for cat in RACE_CATEGORIES:
            cm = race_te == cat
            if cm.sum() < 30 or y_te[cm].sum() < 5 or (1 - y_te[cm]).sum() < 5:
                continue
            aucs[cat] = float(roc_auc_score(y_te[cm], p[cm]))
        if len(aucs) >= 2:
            gap = max(aucs.values()) - min(aucs.values())
            gap_rows.append(
                {
                    "model": MODEL,
                    "label_source": "impression_fixed.json",
                    "finding": finding,
                    "n_groups": len(aucs),
                    "gap_maxmin": float(gap),
                    **{f"auroc_{k}": v for k, v in aucs.items()},
                }
            )
            print(f"  {finding:28s} gap={gap:.3f} groups={len(aucs)}")
    pd.DataFrame(gap_rows).to_csv(
        os.path.join(OUT_DIR, "chexzero_race_subgroup_gaps.csv"), index=False
    )

    if baseline_rows:
        mean_auc = np.mean([r["auroc"] for r in baseline_rows])
        print(f"\nMean finding AUROC = {mean_auc:.3f}")
    if dep_rows and gap_rows:
        dep = pd.DataFrame(dep_rows)
        gaps = pd.DataFrame(gap_rows)
        race_drop = dep[dep["attribute"] == "race_black_vs_white"]["auroc_drop"].mean()
        race_gap = gaps["gap_maxmin"].mean()
        print(f"Race dependence mean drop = {race_drop:.4f}")
        print(f"Race subgroup gap mean    = {race_gap:.4f}")
        print(f"Gap/drop ratio            = {race_gap / max(race_drop, 1e-9):.1f}x")
    print(f"\nSaved to {OUT_DIR} | elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
