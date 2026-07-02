"""
Attribute dependence measured with the Matthews Correlation Coefficient (MCC),
Table S20.

We recompute attribute dependence with MCC alongside AUROC to show the dependence
hierarchy is not an artifact of the chosen metric, and to support discussion of the
clinical meaningfulness of the (small) effect sizes. For each (model, attribute,
finding) we residualize the attribute from the embedding, train a linear finding
probe on the original and residualized embeddings, choose the MCC-maximizing
threshold on the training predictions, and report the MCC drop (baseline -
residualized) and the AUROC drop. We report the per-attribute means and the Spearman
correlation between the AUROC-based and MCC-based attribute rankings.

MCC reference: Chicco D. & Jurman G. (2025), Engineering Applications of Artificial
Intelligence, https://doi.org/10.1016/j.engappai.2025.113347

Usage:
    PYTHONPATH=. python experiments/mcc_dependence.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import SEED, MIMIC_IV_ADMISSIONS
from experiments.data import (load_metadata, load_canonical_ids, load_split,
                              get_aligned_embeddings, merge_chexpert)

ADMISSIONS_PATH = MIMIC_IV_ADMISSIONS
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CLEAN_MODELS = ["ResNet50-ImageNet", "DINOv2-base", "BiomedCLIP",
                "XRV-DenseNet-nih", "CLIP-ViT-B16", "ConvNeXtV2-Base"]
FINDINGS = ["Cardiomegaly", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"]
THRESH_GRID = np.linspace(0.05, 0.95, 19)


def collapse_race(label):
    if pd.isna(label): return "Unknown"
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


def residualize(Xtr, Xte, Dtr, Dte):
    m = Ridge(alpha=1.0).fit(Dtr, Xtr)
    return Xtr - m.predict(Dtr), Xte - m.predict(Dte)


def best_mcc(y_tr, p_tr, y_te, p_te):
    # choose MCC-max threshold on train, apply to test
    best_t, best = 0.5, -2
    for t in THRESH_GRID:
        mc = matthews_corrcoef(y_tr, (p_tr >= t).astype(int))
        if mc > best:
            best, best_t = mc, t
    return matthews_corrcoef(y_te, (p_te >= best_t).astype(int))


def fit_eval(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    ptr = clf.predict_proba(Xtr)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, pte), best_mcc(ytr, ptr, yte, pte)


def main():
    t0 = time.time()
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
    hf = meta_df["heart_failure"].fillna(0).values.astype(float)
    af = meta_df["atrial_fibrillation"].fillna(0).values.astype(float)
    race_pt = patient_race_table()
    race_map = dict(zip(race_pt["subject_id"], race_pt["race_group"]))
    race_img = np.array([race_map.get(s, "Unknown") for s in meta_df["subject_id"].values])
    bw = np.isin(race_img, ["Black", "White"])
    race_bin = (race_img == "Black").astype(float)

    ATTRS = [("heart_failure", hf, None), ("atrial_fibrillation", af, None),
             ("age", age, None), ("sex", sex, None),
             ("race_black_vs_white", race_bin, bw)]

    rows = []
    for model_name in CLEAN_MODELS:
        print(f"\n--- {model_name} ---", flush=True)
        emb = get_aligned_embeddings(model_name, canonical_ids)
        scaler = StandardScaler().fit(emb[train_idx]); emb_s = scaler.transform(emb)
        for attr_name, attr, submask in ATTRS:
            vsel = (~np.isnan(attr)) if submask is None else (submask & ~np.isnan(attr))
            tr = train_idx[vsel[train_idx]]; te = test_idx[vsel[test_idx]]
            Dtr = attr[tr].reshape(-1, 1); Dte = attr[te].reshape(-1, 1)
            if attr_name == "age":
                mu, sd = Dtr.mean(), Dtr.std() + 1e-9
                Dtr = (Dtr - mu) / sd; Dte = (Dte - mu) / sd
            Xtr, Xte = emb_s[tr], emb_s[te]
            Xr_tr, Xr_te = residualize(Xtr, Xte, Dtr, Dte)
            for dz in FINDINGS:
                di = dz_idx[dz]
                mt = masks[tr, di]; me = masks[te, di]
                if mt.sum() < 50 or me.sum() < 50: continue
                yt = labels[tr, di]; ye = labels[te, di]
                if yt[mt].sum() < 10 or ye[me].sum() < 10: continue
                try:
                    a_b, m_b = fit_eval(Xtr[mt], yt[mt], Xte[me], ye[me])
                    a_r, m_r = fit_eval(Xr_tr[mt], yt[mt], Xr_te[me], ye[me])
                    rows.append({"model": model_name, "attribute": attr_name, "finding": dz,
                                 "auroc_drop": a_b - a_r, "mcc_drop": m_b - m_r,
                                 "mcc_base": m_b, "auroc_base": a_b})
                except Exception as e:
                    print(f"   ERR {attr_name}/{dz}: {e}", flush=True)
        print(f"  done {model_name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "mcc_dependence.csv"), index=False)

    print("\n" + "=" * 64)
    print("MCC vs AUROC DEPENDENCE (mean across 6 models x 5 findings)")
    print("=" * 64)
    g = df.groupby("attribute").agg(auroc_drop=("auroc_drop", "mean"),
                                    mcc_drop=("mcc_drop", "mean")).reindex(
        ["heart_failure", "atrial_fibrillation", "age", "race_black_vs_white", "sex"])
    for a, r in g.iterrows():
        print(f"  {a:22s} AUROC drop {r['auroc_drop']:+.4f} | MCC drop {r['mcc_drop']:+.4f}")
    rho, p = spearmanr(g["auroc_drop"], g["mcc_drop"])
    print(f"\n  Spearman rho (AUROC-drop vs MCC-drop attribute ranking) = {rho:.3f} (p={p:.3f})")
    print(f"\nSaved {RESULT_DIR}/mcc_dependence.csv | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
