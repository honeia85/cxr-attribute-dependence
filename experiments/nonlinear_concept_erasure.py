"""
Nonlinear concept erasure with a random-direction control (Table S19).

Question: is the dependence hierarchy / "sex and race are not used" an artifact of
LINEAR residualization? We erase each attribute by Iterative Nullspace Projection
(INLP) in (1) the raw embedding (linear erasure) and (2) an RBF random-Fourier-
feature (RFF) space (linear-in-RFF = nonlinear-in-embedding -> nonlinear erasure).

Key fix vs a naive version: INLP removes whole directions and can collaterally
remove finding-predictive signal, inflating the finding-AUROC drop. We therefore
add a RANDOM-DIRECTION CONTROL: for each attribute we also remove the SAME NUMBER
of random orthonormal directions and measure the finding drop. The collateral-
corrected dependence is

    net_drop = drop(concept-directions) - drop(random-directions, same count).

For an unused attribute (sex/race) the concept directions are uncorrelated with the
finding, so removing them costs about as much as random directions -> net ~ 0. For a
used attribute (heart failure/age) the concept directions are finding-correlated ->
net > 0. This isolates use from collateral damage and is robust to incomplete erasure.

We also report attribute decodability after erasure by BOTH a linear and an MLP probe
(a direct check of whether nonlinear attribute information is removed).

Usage:
    PYTHONPATH=. python experiments/nonlinear_concept_erasure.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from numpy.linalg import norm, qr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import SEED, MIMIC_IV_ADMISSIONS
from experiments.data import (load_metadata, load_canonical_ids, load_split,
                              get_aligned_embeddings, merge_chexpert)

ADMISSIONS_PATH = MIMIC_IV_ADMISSIONS
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

MODELS = ["ResNet50-ImageNet", "DINOv2-base"]
FINDINGS = ["Cardiomegaly", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"]
N_RFF = 1500
SUB_TR = 35000
SUB_TE = 18000
INLP_MAX_ITER = 25
INLP_TARGET = 0.530


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


def inlp_directions(Xtr, atr, Xva, ava, max_iter=INLP_MAX_ITER, target=INLP_TARGET):
    """Return an orthonormal basis (d x k) of the directions INLP removes."""
    Xt, Xv = Xtr.copy(), Xva.copy()
    dirs = []
    auc = np.nan
    for _ in range(max_iter):
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xt, atr)
        auc = roc_auc_score(ava, clf.predict_proba(Xv)[:, 1])
        if auc < target:
            break
        w = clf.coef_[0]
        w = w / (norm(w) + 1e-12)
        dirs.append(w)
        Xt = Xt - np.outer(Xt @ w, w)
        Xv = Xv - np.outer(Xv @ w, w)
    if not dirs:
        return np.zeros((Xtr.shape[1], 0)), auc, 0
    Q, _ = qr(np.array(dirs).T)  # orthonormalize (d x k)
    return Q[:, :len(dirs)], auc, len(dirs)


def project_out(X, Q):
    if Q.shape[1] == 0:
        return X
    return X - (X @ Q) @ Q.T


def lin_auroc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def mlp_decode(Xtr, atr, Xte, ate):
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=60, early_stopping=True,
                        random_state=SEED).fit(Xtr, atr)
    return roc_auc_score(ate, clf.predict_proba(Xte)[:, 1])


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    canonical_ids = load_canonical_ids()
    split = load_split()
    metadata = load_metadata()
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    id_to_order = {d: i for i, d in enumerate(canonical_ids)}
    meta_df = metadata[metadata["dicom_id"].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df["dicom_id"].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)
    _, labels, masks, all_diseases = merge_chexpert(meta_df)
    dz_idx = {d: all_diseases.index(d) for d in FINDINGS}

    sex = meta_df["gender_binary"].values.astype(float)
    age = meta_df["age"].values.astype(float)
    age_bin = (age > np.nanmedian(age)).astype(float)
    hf = meta_df["heart_failure"].fillna(0).values.astype(float)
    af = meta_df["atrial_fibrillation"].fillna(0).values.astype(float)
    race_pt = patient_race_table()
    race_map = dict(zip(race_pt["subject_id"], race_pt["race_group"]))
    race_img = np.array([race_map.get(s, "Unknown") for s in meta_df["subject_id"].values])
    bw = np.isin(race_img, ["Black", "White"])
    race_bin = (race_img == "Black").astype(float)

    tr_sub = rng.choice(train_idx, min(SUB_TR, len(train_idx)), replace=False)
    te_sub = rng.choice(test_idx, min(SUB_TE, len(test_idx)), replace=False)
    n_va = min(9000, len(tr_sub) // 4)
    va_sub = tr_sub[:n_va]
    trc_sub = tr_sub[n_va:]
    need = np.unique(np.concatenate([trc_sub, va_sub, te_sub]))
    loc = {int(a): i for i, a in enumerate(need)}

    def to_local(idx):
        return np.array([loc[int(i)] for i in idx])

    ATTRS = [
        ("sex", sex, None),
        ("race_black_vs_white", race_bin, bw),
        ("age_bin", age_bin, None),
        ("heart_failure", hf, None),
        ("atrial_fibrillation", af, None),
    ]

    rows = []
    for model_name in MODELS:
        print(f"\n===== {model_name} =====", flush=True)
        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb).astype(np.float32)
        d = emb_s.shape[1]
        rff = RBFSampler(n_components=N_RFF, gamma=1.0 / d, random_state=SEED)
        rff.fit(emb_s[trc_sub])
        phi_need = rff.transform(emb_s[need]).astype(np.float32)  # only needed rows

        def getX(space, idx):
            return emb_s[idx] if space == "raw" else phi_need[to_local(idx)]

        for attr_name, attr, submask in ATTRS:
            vsel = (~np.isnan(attr)) if submask is None else (submask & ~np.isnan(attr))
            trc = trc_sub[vsel[trc_sub]]; va = va_sub[vsel[va_sub]]; te = te_sub[vsel[te_sub]]
            atr, ava, ate = attr[trc], attr[va], attr[te]
            if atr.sum() < 50 or (1 - atr).sum() < 50 or ate.sum() < 30:
                print(f"  [{attr_name}] skip (balance)", flush=True); continue

            for space in ["raw", "rff"]:
                Xtr = getX(space, trc); Xva = getX(space, va); Xte = getX(space, te)
                dim = Xtr.shape[1]
                Q, dec_pre, niter = inlp_directions(Xtr, atr, Xva, ava)
                if niter == 0:
                    print(f"  [{attr_name}/{space}] not decodable (0 dirs)", flush=True); continue
                # random control: same number of orthonormal directions
                G = rng.randn(dim, niter)
                Qr, _ = qr(G); Qr = Qr[:, :niter]

                Xtr_c = project_out(Xtr, Q);  Xte_c = project_out(Xte, Q)
                Xtr_r = project_out(Xtr, Qr); Xte_r = project_out(Xte, Qr)

                dec_lin = lin_auroc(Xtr_c, atr, Xte_c, ate)
                try:
                    dec_mlp = mlp_decode(Xtr_c, atr.astype(int), Xte_c, ate.astype(int))
                except Exception:
                    dec_mlp = np.nan

                d_concept, d_random = [], []
                for dz in FINDINGS:
                    di = dz_idx[dz]
                    mt = masks[trc, di]; me = masks[te, di]
                    if mt.sum() < 50 or me.sum() < 50:
                        continue
                    yt = labels[trc, di]; ye = labels[te, di]
                    if yt[mt].sum() < 10 or ye[me].sum() < 10:
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
                rows.append({"model": model_name, "attribute": attr_name, "space": space,
                             "n_dirs": niter, "decode_lin_after": dec_lin,
                             "decode_mlp_after": dec_mlp, "drop_concept": mc,
                             "drop_random": mr, "net_drop": mc - mr})
                print(f"  [{attr_name}/{space}] dirs={niter} decode lin={dec_lin:.3f} "
                      f"mlp={dec_mlp:.3f} | concept={mc:+.4f} random={mr:+.4f} "
                      f"NET={mc-mr:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "nonlinear_concept_erasure.csv"), index=False)

    print("\n" + "=" * 72)
    print("NONLINEAR CONCEPT ERASURE (collateral-corrected) — mean across models")
    print("=" * 72)
    order = ["heart_failure", "atrial_fibrillation", "age_bin", "race_black_vs_white", "sex"]
    for space in ["raw", "rff"]:
        print(f"\n[{space.upper()}]  attr: decode(lin/mlp) | concept / random / NET drop")
        sub = df[df["space"] == space]
        g = sub.groupby("attribute").agg(
            dl=("decode_lin_after", "mean"), dm=("decode_mlp_after", "mean"),
            dc=("drop_concept", "mean"), drnd=("drop_random", "mean"),
            net=("net_drop", "mean")).reindex(order)
        for a, r in g.iterrows():
            print(f"  {a:22s} {r['dl']:.3f}/{r['dm']:.3f} | "
                  f"{r['dc']:+.4f} / {r['drnd']:+.4f} / NET {r['net']:+.4f}")
    print(f"\nSaved {RESULT_DIR}/nonlinear_concept_erasure.csv | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
