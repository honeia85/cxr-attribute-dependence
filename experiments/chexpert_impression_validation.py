"""
CheXpert Plus external sanity analysis using impression-level CheXBERT labels.

This reuses already-extracted CheXpert Plus embeddings and replaces the finding
labels from chexpert_plus_metadata.csv with chexbert impression labels from
chexpert_source/impression_fixed.json. The report-level labels fail a finding
classification sanity gate; impression-level labels recover expected AUROC.

Outputs are written incrementally to results_chexpert_impression/:
  finding_baseline.csv
  encoding.csv
  demographic_dependence.csv
  race_subgroup_gaps.csv
  summary.csv
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

MODEL_GROUPS = {
    "ResNet50-ImageNet": "clean",
    "DINOv2-base": "clean",
    "BiomedCLIP": "clean",
    "XRV-DenseNet-nih": "clean",
    "CLIP-ViT-B16": "clean",
    "ConvNeXtV2-Base": "clean",
    "RAD-DINO": "overlap",
    "CheXzero": "overlap",
    "CheSS-ResNet50": "overlap",
}

PREFERRED_ORDER = [
    "ResNet50-ImageNet",
    "DINOv2-base",
    "BiomedCLIP",
    "XRV-DenseNet-nih",
    "CLIP-ViT-B16",
    "ConvNeXtV2-Base",
    "RAD-DINO",
    "CheXzero",
    "CheSS-ResNet50",
]

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


def available_models():
    models = []
    for model in PREFERRED_ORDER:
        emb_path = os.path.join(EMB_DIR, f"{model}_embeddings.npy")
        ids_path = os.path.join(EMB_DIR, f"{model}_image_ids.npy")
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            models.append(model)
    return models


def load_impression_labels():
    records = []
    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    labels = pd.DataFrame(records).set_index("path_to_image")
    return labels[FINDINGS]


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


def append_write(rows, filename):
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, filename), index=False)


def run_model(model_name, meta_all, labels_all):
    t0 = time.time()
    group = MODEL_GROUPS.get(model_name, "unknown")
    emb = np.load(os.path.join(EMB_DIR, f"{model_name}_embeddings.npy"))
    ids = np.load(os.path.join(EMB_DIR, f"{model_name}_image_ids.npy"), allow_pickle=True)
    ids = np.array([str(x) for x in ids])
    print(f"\n--- {model_name} ({group}) emb={emb.shape} ---", flush=True)

    meta = meta_all.reindex(ids)
    labels = labels_all.reindex(ids)
    missing = int(labels[FINDINGS].isna().all(axis=1).sum())
    if missing:
        print(f"  warning: {missing} ids missing all impression labels", flush=True)

    split = meta["split"].values
    train_mask = np.isin(split, ["train", "val"])
    test_mask = split == "test"
    train_idx = np.flatnonzero(train_mask)
    print(f"  train+val={train_mask.sum():,} test={test_mask.sum():,}", flush=True)

    scaler = StandardScaler().fit(emb[train_idx])
    X = scaler.transform(emb)

    baseline_rows = []
    for finding in FINDINGS:
        y_raw = labels[finding].values.astype(float)
        mask = valid_binary(y_raw)
        y = (y_raw == 1.0).astype(int)
        mt = train_mask & mask
        me = test_mask & mask
        if mt.sum() < 100 or me.sum() < 50 or y[me].sum() < 10 or (1 - y[me]).sum() < 10:
            continue
        auc = disease_auroc(X[mt], y[mt], X[me], y[me])
        baseline_rows.append(
            {
                "model": model_name,
                "model_group": group,
                "label_source": "impression_fixed.json",
                "finding": finding,
                "train_n": int(mt.sum()),
                "test_n": int(me.sum()),
                "test_prevalence": float(y[me].mean()),
                "auroc": float(auc),
            }
        )
    if baseline_rows:
        print(
            f"  finding mean AUROC={np.mean([r['auroc'] for r in baseline_rows]):.3f} "
            f"(n={len(baseline_rows)})",
            flush=True,
        )

    sex = meta["gender_binary"].values.astype(float)
    age = meta["age"].values.astype(float)
    bmi = meta["bmi"].values.astype(float)
    race_group = meta["race_group"].astype(str).values
    bw_mask = np.isin(race_group, ["Black", "White"])
    race_bin = (race_group == "Black").astype(float)
    has_race = np.isin(race_group, RACE_CATEGORIES)

    enc = {
        "model": model_name,
        "model_group": group,
        "label_source": "impression_fixed.json",
    }
    tr = train_mask & ~np.isnan(sex)
    te = test_mask & ~np.isnan(sex)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], sex[tr])
    enc["enc_sex_auroc"] = float(roc_auc_score(sex[te], clf.predict_proba(X[te])[:, 1]))

    tr = train_mask & bw_mask
    te = test_mask & bw_mask
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], race_bin[tr])
    enc["enc_race_black_white_auroc"] = float(
        roc_auc_score(race_bin[te], clf.predict_proba(X[te])[:, 1])
    )

    tr = train_mask & ~np.isnan(age)
    te = test_mask & ~np.isnan(age)
    rg = Ridge(alpha=1.0).fit(X[tr], age[tr])
    enc["enc_age_r2"] = float(r2_score(age[te], rg.predict(X[te])))

    tr = train_mask & ~np.isnan(bmi)
    te = test_mask & ~np.isnan(bmi)
    rg = Ridge(alpha=1.0).fit(X[tr], bmi[tr])
    enc["enc_bmi_r2"] = float(r2_score(bmi[te], rg.predict(X[te])))
    print(
        "  encoding sex={enc_sex_auroc:.3f} race={enc_race_black_white_auroc:.3f} "
        "age_R2={enc_age_r2:.3f} bmi_R2={enc_bmi_r2:.3f}".format(**enc),
        flush=True,
    )

    dep_rows = []
    attr_specs = [
        ("sex", sex, None),
        ("race_black_vs_white", race_bin, bw_mask),
        ("age", age, None),
        ("bmi", bmi, None),
    ]
    for attr_name, attr, subset in attr_specs:
        valid_attr = ~np.isnan(attr)
        if subset is not None:
            valid_attr &= subset
        tr = train_mask & valid_attr
        te = test_mask & valid_attr
        D_tr = attr[tr].reshape(-1, 1)
        D_te = attr[te].reshape(-1, 1)
        if attr_name in ("age", "bmi"):
            mu = D_tr.mean()
            sd = D_tr.std() + 1e-9
            D_tr = (D_tr - mu) / sd
            D_te = (D_te - mu) / sd
        X_tr, X_te = X[tr], X[te]
        Xr_tr, Xr_te = residualize(X_tr, X_te, D_tr, D_te)
        idx_tr = np.flatnonzero(tr)
        idx_te = np.flatnonzero(te)
        drops = []
        for finding in FINDINGS:
            y_raw = labels[finding].values.astype(float)
            y = (y_raw == 1.0).astype(int)
            mt = valid_binary(y_raw[idx_tr])
            me = valid_binary(y_raw[idx_te])
            y_tr = y[idx_tr]
            y_te = y[idx_te]
            if mt.sum() < 100 or me.sum() < 50 or y_te[me].sum() < 10 or (1 - y_te[me]).sum() < 10:
                continue
            auc_base = disease_auroc(X_tr[mt], y_tr[mt], X_te[me], y_te[me])
            auc_resid = disease_auroc(Xr_tr[mt], y_tr[mt], Xr_te[me], y_te[me])
            drop = auc_base - auc_resid
            drops.append(drop)
            dep_rows.append(
                {
                    "model": model_name,
                    "model_group": group,
                    "label_source": "impression_fixed.json",
                    "attribute": attr_name,
                    "finding": finding,
                    "auroc_baseline": float(auc_base),
                    "auroc_residualized": float(auc_resid),
                    "auroc_drop": float(drop),
                }
            )
        if drops:
            print(f"  dep[{attr_name}] mean_drop={np.mean(drops):.4f}", flush=True)

    gap_rows = []
    for finding in FINDINGS:
        y_raw = labels[finding].values.astype(float)
        mask = valid_binary(y_raw) & has_race
        y = (y_raw == 1.0).astype(int)
        mt = train_mask & mask
        me = test_mask & mask
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
            gap_rows.append(
                {
                    "model": model_name,
                    "model_group": group,
                    "label_source": "impression_fixed.json",
                    "finding": finding,
                    "n_groups": len(aucs),
                    "gap_maxmin": float(max(aucs.values()) - min(aucs.values())),
                    **{f"auroc_{k}": v for k, v in aucs.items()},
                }
            )
    if gap_rows:
        print(
            f"  race subgroup gap mean={np.mean([r['gap_maxmin'] for r in gap_rows]):.3f}",
            flush=True,
        )

    print(f"  elapsed={time.time() - t0:.0f}s", flush=True)
    return baseline_rows, [enc], dep_rows, gap_rows


def main():
    t0 = time.time()
    models = available_models()
    print(f"Available full models: {models}", flush=True)
    meta_all = pd.read_csv(META_CSV).set_index("path_to_image")
    labels_all = load_impression_labels()

    baseline_all, enc_all, dep_all, gap_all = [], [], [], []
    for model_name in models:
        baseline, enc, dep, gap = run_model(model_name, meta_all, labels_all)
        baseline_all.extend(baseline)
        enc_all.extend(enc)
        dep_all.extend(dep)
        gap_all.extend(gap)
        append_write(baseline_all, "finding_baseline.csv")
        append_write(enc_all, "encoding.csv")
        append_write(dep_all, "demographic_dependence.csv")
        append_write(gap_all, "race_subgroup_gaps.csv")

    baseline_df = pd.DataFrame(baseline_all)
    dep_df = pd.DataFrame(dep_all)
    gap_df = pd.DataFrame(gap_all)
    summary_rows = []
    for model_name in models:
        b = baseline_df[baseline_df["model"] == model_name]
        d = dep_df[(dep_df["model"] == model_name) & (dep_df["attribute"] == "race_black_vs_white")]
        g = gap_df[gap_df["model"] == model_name]
        summary_rows.append(
            {
                "model": model_name,
                "model_group": MODEL_GROUPS.get(model_name, "unknown"),
                "mean_finding_auroc": float(b["auroc"].mean()) if len(b) else np.nan,
                "race_dependence_mean_drop": float(d["auroc_drop"].mean()) if len(d) else np.nan,
                "race_gap_mean": float(g["gap_maxmin"].mean()) if len(g) else np.nan,
                "race_gap_drop_ratio": float(g["gap_maxmin"].mean() / max(d["auroc_drop"].mean(), 1e-9))
                if len(d) and len(g)
                else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    print("\nSummary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved to {OUT_DIR} | total elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
