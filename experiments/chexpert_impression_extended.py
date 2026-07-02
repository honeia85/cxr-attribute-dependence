"""
CheXpert Plus external sanity check — full-strength extension.

Runs, on the already-extracted CheXpert Plus embeddings + impression-level CheXBERT
labels, the SAME analyses used for the primary MIMIC cohort, so the external check
matches the primary cohort's rigor:

  (A) Multi-metric subgroup fairness  -> mirrors Table S21 / multimetric_fairness.py
      Youden-J operating point; per-subgroup AUROC, sensitivity, specificity, FPR,
      FNR, ECE; max-min subgroup gap with percentile bootstrap 95% CI (1000).
      Dimensions: Gender, Age tertiles, 4-category Race.

  (B) Nonlinear concept erasure (demographics) -> mirrors Table S19 / nonlinear_concept_erasure.py
      INLP in (1) raw embedding and (2) RBF random-Fourier-feature space, with a
      matched random-direction control; net_drop = concept_drop - random_drop;
      decodability after erasure by a linear AND an MLP probe.
      Demographic attributes only (CheXpert Plus has no ICD comorbidities):
      sex, race (Black vs White), age (median split, used-attribute positive control).

Outputs to results_chexpert_impression/:
  chexpert_multimetric_per_group.csv
  chexpert_multimetric_gaps.csv
  chexpert_nonlinear_concept_erasure.csv
  chexpert_extended_summary.txt
"""
import json
import os
import time

import numpy as np
import pandas as pd
from numpy.linalg import norm, qr
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(PROJECT_DIR)
EMB_DIR = os.path.join(PROJECT_DIR, "embeddings_chexpert")
META_CSV = os.path.join(PROJECT_DIR, "chexpert_plus_metadata.csv")
LABEL_JSON = os.path.join(ROOT_DIR, "chexpert_source", "impression_fixed.json")
OUT_DIR = os.path.join(PROJECT_DIR, "results_chexpert_impression")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_GROUPS = {
    "ResNet50-ImageNet": "clean", "DINOv2-base": "clean", "BiomedCLIP": "clean",
    "XRV-DenseNet-nih": "clean", "CLIP-ViT-B16": "clean", "ConvNeXtV2-Base": "clean",
    "RAD-DINO": "overlap", "CheXzero": "overlap",
}
PREFERRED_ORDER = ["ResNet50-ImageNet", "DINOv2-base", "BiomedCLIP",
                   "XRV-DenseNet-nih", "CLIP-ViT-B16", "ConvNeXtV2-Base",
                   "RAD-DINO", "CheXzero"]
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
            "Pneumonia", "Pneumothorax", "Support Devices"]
NL_FINDINGS = ["Cardiomegaly", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"]
RACE_CATEGORIES = ["White", "Black", "Asian", "Other"]
N_BOOT = 1000
N_RFF = 1500
SUB_TR = 35000
SUB_TE = 18000
INLP_MAX_ITER = 25
INLP_TARGET = 0.530


def available_models():
    out = []
    for m in PREFERRED_ORDER:
        if (os.path.exists(os.path.join(EMB_DIR, f"{m}_embeddings.npy"))
                and os.path.exists(os.path.join(EMB_DIR, f"{m}_image_ids.npy"))):
            out.append(m)
    return out


def load_impression_labels():
    recs = []
    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return pd.DataFrame(recs).set_index("path_to_image")[FINDINGS]


def valid_binary(v):
    return np.isin(v.astype(float), [0.0, 1.0])


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


def inlp_directions(Xtr, atr, Xva, ava, max_iter=INLP_MAX_ITER, target=INLP_TARGET):
    Xt, Xv = Xtr.copy(), Xva.copy()
    dirs = []
    for _ in range(max_iter):
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xt, atr)
        auc = roc_auc_score(ava, clf.predict_proba(Xv)[:, 1])
        if auc < target:
            break
        w = clf.coef_[0]; w = w / (norm(w) + 1e-12)
        dirs.append(w)
        Xt = Xt - np.outer(Xt @ w, w)
        Xv = Xv - np.outer(Xv @ w, w)
    if not dirs:
        return np.zeros((Xtr.shape[1], 0)), 0
    Q, _ = qr(np.array(dirs).T)
    return Q[:, :len(dirs)], len(dirs)


def project_out(X, Q):
    return X if Q.shape[1] == 0 else X - (X @ Q) @ Q.T


def lin_auroc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def mlp_decode(Xtr, atr, Xte, ate):
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=60, early_stopping=True,
                        random_state=SEED).fit(Xtr, atr)
    return roc_auc_score(ate, clf.predict_proba(Xte)[:, 1])


def multimetric(model_name, group, X, train_mask, test_mask, labels, sex, age, race_group, rng):
    per_rows, gap_rows = [], []

    def groups_for(dim, sel):
        if dim == "Gender":
            return np.where(sex[sel] == 1, "Male", "Female"), ["Female", "Male"]
        if dim == "Age":
            a = age[sel]
            return np.where(a < 50, "<50", np.where(a < 70, "50-70", "70+")), ["<50", "50-70", "70+"]
        return race_group[sel], RACE_CATEGORIES

    for dz in FINDINGS:
        y_raw = labels[dz].values.astype(float)
        mask = valid_binary(y_raw)
        y = (y_raw == 1.0).astype(int)
        mt = train_mask & mask; me = test_mask & mask
        if mt.sum() < 100 or me.sum() < 50 or y[me].sum() < 10 or (1 - y[me]).sum() < 10:
            continue
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[mt], y[mt])
        sel = np.flatnonzero(me)
        p = clf.predict_proba(X[sel])[:, 1]
        ye = y[sel]
        fpr, tpr, thr = roc_curve(ye, p)
        thr_opt = thr[np.argmax(tpr - fpr)]
        for dim in ["Gender", "Age", "Race"]:
            g, cats = groups_for(dim, sel)
            aucs = {}
            for cat in cats:
                cm = g == cat
                if cm.sum() < 30 or ye[cm].sum() < 5 or (1 - ye[cm]).sum() < 5:
                    continue
                yc, pc = ye[cm], p[cm]
                try:
                    auc = roc_auc_score(yc, pc)
                except Exception:
                    continue
                mm = metrics_at(yc, pc, thr_opt)
                per_rows.append({"model": model_name, "model_group": group, "finding": dz,
                                 "dimension": dim, "group": cat, "n": int(cm.sum()),
                                 "auroc": auc, "ece": ece(yc, pc), **mm})
                aucs[cat] = (yc, pc)
            if len(aucs) < 2:
                continue

            def gap_for(fn):
                vals = []
                for _c, (yc, pc) in aucs.items():
                    try:
                        vals.append(fn(yc, pc))
                    except Exception:
                        vals.append(np.nan)
                vals = [v for v in vals if v == v]
                return (max(vals) - min(vals)) if len(vals) >= 2 else np.nan

            auc_gap = gap_for(lambda y_, q_: roc_auc_score(y_, q_))
            boots = []
            for _ in range(N_BOOT):
                bv = []
                for _c, (yc, pc) in aucs.items():
                    bi = rng.randint(0, len(yc), len(yc))
                    if yc[bi].sum() < 3 or (1 - yc[bi]).sum() < 3:
                        continue
                    try:
                        bv.append(roc_auc_score(yc[bi], pc[bi]))
                    except Exception:
                        pass
                if len(bv) >= 2:
                    boots.append(max(bv) - min(bv))
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 50 else (np.nan, np.nan))
            gap_rows.append({"model": model_name, "model_group": group, "finding": dz, "dimension": dim,
                             "auroc_gap": auc_gap, "gap_ci_lo": lo, "gap_ci_hi": hi,
                             "sens_gap": gap_for(lambda y_, q_: metrics_at(y_, q_, thr_opt)["sensitivity"]),
                             "spec_gap": gap_for(lambda y_, q_: metrics_at(y_, q_, thr_opt)["specificity"]),
                             "fpr_gap": gap_for(lambda y_, q_: metrics_at(y_, q_, thr_opt)["fpr"]),
                             "fnr_gap": gap_for(lambda y_, q_: metrics_at(y_, q_, thr_opt)["fnr"]),
                             "ece_gap": gap_for(lambda y_, q_: ece(y_, q_))})
    return per_rows, gap_rows


def nonlinear(model_name, group, emb_s, train_idx, test_idx, labels, sex, age, race_group, rng):
    rows = []
    age_bin = (age > np.nanmedian(age)).astype(float)
    bw = np.isin(race_group, ["Black", "White"])
    race_bin = (race_group == "Black").astype(float)
    attrs = [("sex", sex, None), ("race_black_vs_white", race_bin, bw), ("age_bin", age_bin, None)]

    tr_sub = rng.choice(train_idx, min(SUB_TR, len(train_idx)), replace=False)
    te_sub = rng.choice(test_idx, min(SUB_TE, len(test_idx)), replace=False)
    n_va = min(9000, len(tr_sub) // 4)
    va_sub = tr_sub[:n_va]; trc_sub = tr_sub[n_va:]
    need = np.unique(np.concatenate([trc_sub, va_sub, te_sub]))
    loc = {int(a): i for i, a in enumerate(need)}

    def to_local(idx):
        return np.array([loc[int(i)] for i in idx])

    d = emb_s.shape[1]
    rff = RBFSampler(n_components=N_RFF, gamma=1.0 / d, random_state=SEED)
    rff.fit(emb_s[trc_sub])
    phi_need = rff.transform(emb_s[need]).astype(np.float32)

    label_mat = {dz: (labels[dz].values.astype(float)) for dz in NL_FINDINGS}

    def getX(space, idx):
        return emb_s[idx] if space == "raw" else phi_need[to_local(idx)]

    for attr_name, attr, submask in attrs:
        vsel = (~np.isnan(attr)) if submask is None else (submask & ~np.isnan(attr))
        trc = trc_sub[vsel[trc_sub]]; va = va_sub[vsel[va_sub]]; te = te_sub[vsel[te_sub]]
        atr, ava, ate = attr[trc], attr[va], attr[te]
        if atr.sum() < 50 or (1 - atr).sum() < 50 or ate.sum() < 30:
            print(f"  [{attr_name}] skip (balance)", flush=True)
            continue
        for space in ["raw", "rff"]:
            Xtr = getX(space, trc); Xva = getX(space, va); Xte = getX(space, te)
            dim = Xtr.shape[1]
            Q, niter = inlp_directions(Xtr, atr, Xva, ava)
            if niter == 0:
                print(f"  [{attr_name}/{space}] not decodable (0 dirs)", flush=True)
                continue
            G = rng.randn(dim, niter); Qr, _ = qr(G); Qr = Qr[:, :niter]
            Xtr_c = project_out(Xtr, Q); Xte_c = project_out(Xte, Q)
            Xtr_r = project_out(Xtr, Qr); Xte_r = project_out(Xte, Qr)
            dec_lin = lin_auroc(Xtr_c, atr, Xte_c, ate)
            try:
                dec_mlp = mlp_decode(Xtr_c, atr.astype(int), Xte_c, ate.astype(int))
            except Exception:
                dec_mlp = np.nan
            d_concept, d_random = [], []
            for dz in NL_FINDINGS:
                yv = label_mat[dz]
                mt = valid_binary(yv[trc]); me = valid_binary(yv[te])
                yt = (yv[trc] == 1.0).astype(int); ye = (yv[te] == 1.0).astype(int)
                if mt.sum() < 50 or me.sum() < 50 or yt[mt].sum() < 10 or ye[me].sum() < 10 \
                        or (1 - ye[me]).sum() < 10:
                    continue
                try:
                    base = lin_auroc(Xtr[mt], yt[mt], Xte[me], ye[me])
                    dc = base - lin_auroc(Xtr_c[mt], yt[mt], Xte_c[me], ye[me])
                    dr = base - lin_auroc(Xtr_r[mt], yt[mt], Xte_r[me], ye[me])
                    d_concept.append(dc); d_random.append(dr)
                except Exception:
                    pass
            mc = float(np.mean(d_concept)) if d_concept else np.nan
            mr = float(np.mean(d_random)) if d_random else np.nan
            rows.append({"model": model_name, "model_group": group, "attribute": attr_name,
                         "space": space, "n_dirs": niter, "decode_lin_after": dec_lin,
                         "decode_mlp_after": dec_mlp, "drop_concept": mc, "drop_random": mr,
                         "net_drop": mc - mr})
            print(f"  [{attr_name}/{space}] dirs={niter} decode lin={dec_lin:.3f} "
                  f"mlp={dec_mlp:.3f} | concept={mc:+.4f} random={mr:+.4f} NET={mc-mr:+.4f}",
                  flush=True)
    return rows


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    models = available_models()
    print(f"Models: {models}", flush=True)
    meta_all = pd.read_csv(META_CSV).set_index("path_to_image")
    labels_all = load_impression_labels()

    per_all, gap_all, nl_all = [], [], []
    for model_name in models:
        group = MODEL_GROUPS.get(model_name, "unknown")
        emb = np.load(os.path.join(EMB_DIR, f"{model_name}_embeddings.npy"))
        ids = np.load(os.path.join(EMB_DIR, f"{model_name}_image_ids.npy"), allow_pickle=True)
        ids = np.array([str(x) for x in ids])
        print(f"\n===== {model_name} ({group}) emb={emb.shape} =====", flush=True)
        meta = meta_all.reindex(ids)
        labels = labels_all.reindex(ids)
        split = meta["split"].values
        train_mask = np.isin(split, ["train", "val"])
        test_mask = split == "test"
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        scaler = StandardScaler().fit(emb[train_idx])
        X = scaler.transform(emb).astype(np.float32)

        sex = meta["gender_binary"].values.astype(float)
        age = meta["age"].values.astype(float)
        race_group = meta["race_group"].astype(str).values

        pr, gr = multimetric(model_name, group, X, train_mask, test_mask, labels, sex, age, race_group, rng)
        per_all.extend(pr); gap_all.extend(gr)
        pd.DataFrame(per_all).to_csv(os.path.join(OUT_DIR, "chexpert_multimetric_per_group.csv"), index=False)
        pd.DataFrame(gap_all).to_csv(os.path.join(OUT_DIR, "chexpert_multimetric_gaps.csv"), index=False)
        print(f"  [multimetric] done ({len(gr)} gap rows)", flush=True)

        nl = nonlinear(model_name, group, X, train_idx, test_idx, labels, sex, age, race_group, rng)
        nl_all.extend(nl)
        pd.DataFrame(nl_all).to_csv(os.path.join(OUT_DIR, "chexpert_nonlinear_concept_erasure.csv"), index=False)
        print(f"  [nonlinear] done ({len(nl)} rows) | elapsed={time.time()-t0:.0f}s", flush=True)

    lines = []
    gap = pd.DataFrame(gap_all)
    if len(gap):
        g = gap.groupby("dimension").agg(
            auroc=("auroc_gap", "mean"), ci_lo=("gap_ci_lo", "mean"), ci_hi=("gap_ci_hi", "mean"),
            sens=("sens_gap", "mean"), spec=("spec_gap", "mean"), fpr=("fpr_gap", "mean"),
            fnr=("fnr_gap", "mean"), ece=("ece_gap", "mean"))
        lines.append("MULTI-METRIC SUBGROUP GAPS (mean across models x findings, by dimension)")
        for dim, r in g.iterrows():
            lines.append(f"  {dim:8s} AUROC {r['auroc']:.3f} [CI {r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
                         f"sens {r['sens']:.3f} spec {r['spec']:.3f} FPR {r['fpr']:.3f} "
                         f"FNR {r['fnr']:.3f} ECE {r['ece']:.3f}")
    nl = pd.DataFrame(nl_all)
    if len(nl):
        lines.append("\nNONLINEAR CONCEPT ERASURE (collateral-corrected NET drop), mean across models")
        order = ["age_bin", "race_black_vs_white", "sex"]
        for space in ["raw", "rff"]:
            lines.append(f"[{space.upper()}] attr: decode(lin/mlp) | concept/random/NET")
            sub = nl[nl["space"] == space].groupby("attribute").agg(
                dl=("decode_lin_after", "mean"), dm=("decode_mlp_after", "mean"),
                dc=("drop_concept", "mean"), drnd=("drop_random", "mean"),
                net=("net_drop", "mean")).reindex(order)
            for a, r in sub.iterrows():
                lines.append(f"  {a:22s} {r['dl']:.3f}/{r['dm']:.3f} | "
                             f"{r['dc']:+.4f}/{r['drnd']:+.4f}/NET {r['net']:+.4f}")
    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "chexpert_extended_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + f"\n\ntotal elapsed={time.time()-t0:.0f}s\n")
    print("\n" + summary, flush=True)
    print(f"\nSaved to {OUT_DIR} | total elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
