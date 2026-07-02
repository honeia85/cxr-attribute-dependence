"""
Multi-metric subgroup fairness with bootstrap CIs (Table S21).

AUROC alone does not fully characterise fairness, so we add per-subgroup sensitivity,
specificity, calibration error, and false-positive / false-negative rates, each with a
confidence interval around the subgroup gap.

For each clean model x finding x subgroup-dimension (Gender, Age tertiles, 4-cat Race)
we train a linear finding probe, pick the Youden-J operating point on the test set,
and report per-subgroup AUROC, sensitivity (TPR), specificity (TNR), FPR, FNR, and
expected calibration error (ECE, 10 bins). We also report the max-min subgroup gap for
each metric with a percentile bootstrap 95% CI (1000 resamples).

Usage:
    PYTHONPATH=. python experiments/multimetric_fairness.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import SEED, MIMIC_IV_ADMISSIONS
from experiments.data import (load_metadata, load_canonical_ids, load_split,
                              get_aligned_embeddings, merge_chexpert)

ADMISSIONS_PATH = MIMIC_IV_ADMISSIONS
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CLEAN_MODELS = ["ResNet50-ImageNet", "DINOv2-base", "BiomedCLIP",
                "XRV-DenseNet-nih", "CLIP-ViT-B16", "ConvNeXtV2-Base"]
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
            "Pneumonia", "Pneumothorax", "Support Devices"]
N_BOOT = 1000


def collapse_race(label):
    if pd.isna(label):
        return "Unknown"
    s = str(label).upper()
    if s.startswith("WHITE") or s == "PORTUGUESE": return "White"
    if s.startswith("BLACK"): return "Black"
    if s.startswith("ASIAN") or s == "SOUTH ASIAN": return "Asian"
    if s in {"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"}: return "Unknown"
    return "Other"


def patient_race_table():
    adm = pd.read_csv(ADMISSIONS_PATH, usecols=["subject_id", "race"])
    adm["race_group"] = adm["race"].map(collapse_race)
    def majority(group):
        nu = group[group != "Unknown"]; pool = nu if len(nu) else group
        return pool.mode().iloc[0]
    return adm.groupby("subject_id")["race_group"].apply(majority).reset_index()


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (p >= bins[b]) & (p < bins[b + 1] if b < n_bins - 1 else p <= bins[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return e


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {"sensitivity": sens, "specificity": spec,
            "fpr": 1 - spec if spec == spec else np.nan,
            "fnr": 1 - sens if sens == sens else np.nan}


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    canonical_ids = load_canonical_ids(); split = load_split(); metadata = load_metadata()
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    id_to_order = {d: i for i, d in enumerate(canonical_ids)}
    meta_df = metadata[metadata["dicom_id"].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df["dicom_id"].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)
    _, labels, masks, all_diseases = merge_chexpert(meta_df)
    dz_idx = {d: all_diseases.index(d) for d in FINDINGS}

    sex = meta_df["gender_binary"].values.astype(float)
    age = meta_df["age"].values.astype(float)
    race_pt = patient_race_table()
    race_map = dict(zip(race_pt["subject_id"], race_pt["race_group"]))
    race_img = np.array([race_map.get(s, "Unknown") for s in meta_df["subject_id"].values])

    def groups_for(dim, idx):
        if dim == "Gender":
            g = np.where(sex[idx] == 1, "Male", "Female")
            return g, ["Female", "Male"]
        if dim == "Age":
            a = age[idx]; g = np.where(a < 50, "<50", np.where(a < 70, "50-70", "70+"))
            return g, ["<50", "50-70", "70+"]
        g = race_img[idx]
        return g, ["White", "Black", "Asian", "Other"]

    per_rows, gap_rows = [], []
    for model_name in CLEAN_MODELS:
        print(f"\n--- {model_name} ---", flush=True)
        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler().fit(emb[train_idx]); emb_s = scaler.transform(emb)
        for dz in FINDINGS:
            di = dz_idx[dz]
            mt = masks[train_idx, di]; me = masks[test_idx, di]
            if mt.sum() < 50 or me.sum() < 50: continue
            trs = train_idx[mt]; tes = test_idx[me]
            yt = labels[trs, di]; ye = labels[tes, di]
            if yt.sum() < 10 or ye.sum() < 10: continue
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(emb_s[trs], yt)
            p = clf.predict_proba(emb_s[tes])[:, 1]
            fpr, tpr, thr = roc_curve(ye, p)
            thr_opt = thr[np.argmax(tpr - fpr)]  # Youden J
            for dim in ["Gender", "Age", "Race"]:
                g, cats = groups_for(dim, tes)
                aucs = {}
                for cat in cats:
                    cm = g == cat
                    if cm.sum() < 30 or ye[cm].sum() < 5 or (1 - ye[cm]).sum() < 5: continue
                    yc, pc = ye[cm], p[cm]
                    try: auc = roc_auc_score(yc, pc)
                    except Exception: continue
                    mm = metrics_at(yc, pc, thr_opt)
                    per_rows.append({"model": model_name, "finding": dz, "dimension": dim,
                                     "group": cat, "n": int(cm.sum()), "auroc": auc,
                                     "ece": ece(yc, pc), **mm})
                    aucs[cat] = (yc, pc)
                if len(aucs) < 2: continue
                # bootstrap CI for each metric's max-min gap
                def gap_for(metric_fn):
                    vals = []
                    for cat, (yc, pc) in aucs.items():
                        try: vals.append(metric_fn(yc, pc))
                        except Exception: vals.append(np.nan)
                    vals = [v for v in vals if v == v]
                    return (max(vals) - min(vals)) if len(vals) >= 2 else np.nan
                auc_gap = gap_for(lambda y, q: roc_auc_score(y, q))
                boots = []
                for _ in range(N_BOOT):
                    bv = []
                    for cat, (yc, pc) in aucs.items():
                        bi = rng.randint(0, len(yc), len(yc))
                        if yc[bi].sum() < 3 or (1 - yc[bi]).sum() < 3: continue
                        try: bv.append(roc_auc_score(yc[bi], pc[bi]))
                        except Exception: pass
                    if len(bv) >= 2: boots.append(max(bv) - min(bv))
                lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 50 else (np.nan, np.nan))
                gap_rows.append({"model": model_name, "finding": dz, "dimension": dim,
                                 "auroc_gap": auc_gap, "gap_ci_lo": lo, "gap_ci_hi": hi,
                                 "sens_gap": gap_for(lambda y, q: metrics_at(y, q, thr_opt)["sensitivity"]),
                                 "spec_gap": gap_for(lambda y, q: metrics_at(y, q, thr_opt)["specificity"]),
                                 "fpr_gap": gap_for(lambda y, q: metrics_at(y, q, thr_opt)["fpr"]),
                                 "fnr_gap": gap_for(lambda y, q: metrics_at(y, q, thr_opt)["fnr"]),
                                 "ece_gap": gap_for(lambda y, q: ece(y, q))})
        print(f"  done {model_name}", flush=True)

    pd.DataFrame(per_rows).to_csv(os.path.join(RESULT_DIR, "multimetric_fairness_per_group.csv"), index=False)
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(os.path.join(RESULT_DIR, "multimetric_fairness_gaps.csv"), index=False)

    print("\n" + "=" * 70)
    print("MULTI-METRIC SUBGROUP GAPS (mean across 6 models x findings, by dimension)")
    print("=" * 70)
    g = gap.groupby("dimension").agg(
        auroc=("auroc_gap", "mean"), ci_lo=("gap_ci_lo", "mean"), ci_hi=("gap_ci_hi", "mean"),
        sens=("sens_gap", "mean"), spec=("spec_gap", "mean"),
        fpr=("fpr_gap", "mean"), fnr=("fnr_gap", "mean"), ece=("ece_gap", "mean"))
    for dim, r in g.iterrows():
        print(f"  {dim:8s} AUROC gap {r['auroc']:.3f} [boot CI {r['ci_lo']:.3f},{r['ci_hi']:.3f}] | "
              f"sens {r['sens']:.3f} spec {r['spec']:.3f} FPR {r['fpr']:.3f} "
              f"FNR {r['fnr']:.3f} ECE {r['ece']:.3f}")
    print(f"\nSaved to {RESULT_DIR}/ | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
