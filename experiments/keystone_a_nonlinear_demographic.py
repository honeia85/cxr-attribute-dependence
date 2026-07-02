"""
Is the encoding-dependence dissociation an artifact of LINEARITY? (Section 2.4 /
Table S18).

Linear residualization captures only the linear pathway, so "dependence" is a lower
bound and is reported as "linear use". This matters for one claim in particular:
that sex/race are not used. We test it two
non-degenerate ways (for binary attributes, MLP residualization of X on the
attribute reduces to the linear group-mean map, so the informative tests are):

  PART 1 -- Nonlinear USE test (the decisive one). For sex and race, train a
  NONLINEAR (MLP) finding classifier on the raw embedding X and on the linearly
  residualized embedding X_res; nonlinear dependence = AUROC(MLP,X) -
  AUROC(MLP,X_res). If ~0, sex/race are not used even by a nonlinear classifier.

  PART 2 -- Linear vs MLP residualization ranking. For the two CONTINUOUS
  attributes (age, BMI) where MLP residualization is genuinely nonlinear,
  recompute the dependence and check rank preservation; report the Spearman rho
  between the linear and (linear-for-binary / MLP-for-continuous) 24-attribute
  rankings, replacing the manuscript's unquantified "qualitatively similar".

Train is subsampled to SUB for MLP speed; AUROC is on the full test set.

Usage:
    PYTHONPATH=. python experiments/keystone_a_nonlinear_demographic.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import SEED, MIMIC_IV_ADMISSIONS
from experiments.data import (load_metadata, load_canonical_ids, load_split,
                              get_aligned_embeddings, merge_chexpert)

ADMISSIONS_PATH = MIMIC_IV_ADMISSIONS
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# XRV-DenseNet-nih excluded from MLP: erank 5.6 -> MLP diverges (manuscript S.M.3)
MLP_MODELS = ["ResNet50-ImageNet", "DINOv2-base", "BiomedCLIP",
              "CLIP-ViT-B16", "ConvNeXtV2-Base"]
KEY_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
                "Pneumonia", "Pneumothorax", "Support Devices"]
# Part 1 (expensive MLP finding-classifier) uses 5 representative findings
P1_FINDINGS = ["Cardiomegaly", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"]
SUB = 25000  # train subsample for MLP speed
MLP_HIDDEN = 128  # fixed small hidden layer (d//2 too slow at d=2048)

# Linear per-attribute mean drops from Table S1 (24 attributes), for the
# linear-vs-MLP ranking comparison (Part 2).
LINEAR_DROP_S1 = {
    "heart_failure": 0.0178, "atrial_fibrillation": 0.0147, "age": 0.0120,
    "aki": 0.0109, "anemia": 0.0088, "ckd": 0.0069, "coronary_artery_disease": 0.0057,
    "bmi": 0.0042, "copd": 0.0039, "hyperlipidemia": 0.0038, "respiratory_failure": 0.0036,
    "diabetes": 0.0030, "obesity": 0.0027, "race_black_vs_white": 0.0015,
    "hypertension": 0.0015, "cancer_history": 0.0015, "hypothyroidism": 0.0011,
    "stroke": 0.0010, "pulmonary_fibrosis": 0.0007, "liver_disease": 0.0003,
    "sex": 0.0002, "asthma": 0.0001, "depression": 0.0000, "smoking_history": -0.0000,
}


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


def lin_resid(X_tr, X_te, D_tr, D_te):
    m = LinearRegression().fit(D_tr, X_tr)
    return X_tr - m.predict(D_tr), X_te - m.predict(D_te)


def mlp_resid(X_tr, X_te, D_tr, D_te, d):
    h = MLP_HIDDEN
    m = MLPRegressor(hidden_layer_sizes=(h,), activation="relu", max_iter=100,
                     learning_rate_init=1e-3, alpha=1e-4, random_state=SEED)
    m.fit(D_tr, X_tr)
    return X_tr - m.predict(D_tr), X_te - m.predict(D_te)


def mlp_auroc(X_tr, y_tr, X_te, y_te, d):
    h = MLP_HIDDEN
    clf = MLPClassifier(hidden_layer_sizes=(h,), activation="relu", max_iter=80,
                        early_stopping=True, learning_rate_init=1e-3, alpha=1e-4,
                        random_state=SEED)
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


def lin_auroc(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)
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

    sex = meta_df["gender_binary"].values.astype(float)
    age = meta_df["age"].values.astype(float)
    bmi = meta_df["bmi"].values.astype(float)
    race_pt = patient_race_table()
    race_map = dict(zip(race_pt["subject_id"], race_pt["race_group"]))
    race_image = np.array([race_map.get(s, "Unknown") for s in meta_df["subject_id"].values])
    bw_mask = np.isin(race_image, ["Black", "White"])
    race_bin = (race_image == "Black").astype(float)

    def subsample(idx):
        if len(idx) <= SUB:
            return idx
        return rng.choice(idx, SUB, replace=False)

    part1_rows, part2_rows, decode_rows = [], [], []

    for model_name in MLP_MODELS:
        print(f"\n--- {model_name} ---", flush=True)
        emb = get_aligned_embeddings(model_name, canonical_ids)
        d = emb.shape[1]
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb)

        # ===== PART 1: nonlinear USE of sex / race (MLP classifier dependence) =====
        for attr_name, attr, sub in [("sex", sex, None),
                                     ("race_black_vs_white", race_bin, bw_mask)]:
            valid = ~np.isnan(attr) if sub is None else (sub & ~np.isnan(attr))
            tr = train_idx[valid[train_idx]]; te = test_idx[valid[test_idx]]
            D_tr = attr[tr].reshape(-1, 1); D_te = attr[te].reshape(-1, 1)
            Xl_tr, Xl_te = lin_resid(emb_s[tr], emb_s[te], D_tr, D_te)  # linear residualized
            # decodability probe: can an MLP recover the attribute after linear residualization?
            try:
                pp = subsample(np.arange(len(tr)))
                dec_base = mlp_auroc(emb_s[tr][pp], attr[tr][pp].round().astype(int),
                                     emb_s[te], attr[te].round().astype(int), d)
                dec_res = mlp_auroc(Xl_tr[pp], attr[tr][pp].round().astype(int),
                                    Xl_te, attr[te].round().astype(int), d)
                decode_rows.append({"model": model_name, "attribute": attr_name,
                                    "decode_auroc_X": dec_base, "decode_auroc_Xres": dec_res})
                print(f"  {attr_name}: MLP-decode X={dec_base:.3f} -> X_res={dec_res:.3f}", flush=True)
            except Exception as e:
                print(f"    decode ERR {attr_name}: {e}", flush=True)
            for dz in P1_FINDINGS:
                di = dz_idx[dz]
                mt = masks[tr, di]; me = masks[te, di]
                if mt.sum() < 50 or me.sum() < 50:
                    continue
                yt = labels[tr, di]; ye = labels[te, di]
                if yt[mt].sum() < 10 or ye[me].sum() < 10:
                    continue
                tr_s = subsample(np.where(mt)[0])
                Xb_tr = emb_s[tr][tr_s]; yb = yt[tr_s]
                Xr_tr = Xl_tr[tr_s]
                try:
                    a_base_mlp = mlp_auroc(Xb_tr, yb, emb_s[te][me], ye[me], d)
                    a_res_mlp = mlp_auroc(Xr_tr, yb, Xl_te[me], ye[me], d)
                    a_base_lin = lin_auroc(Xb_tr, yb, emb_s[te][me], ye[me])
                    a_res_lin = lin_auroc(Xr_tr, yb, Xl_te[me], ye[me])
                    part1_rows.append({
                        "model": model_name, "attribute": attr_name, "disease": dz,
                        "mlp_dependence": a_base_mlp - a_res_mlp,
                        "linear_dependence": a_base_lin - a_res_lin,
                        "mlp_base_auroc": a_base_mlp, "mlp_res_auroc": a_res_mlp})
                except Exception as e:
                    print(f"    ERR {attr_name}/{dz}: {e}", flush=True)
            sub_r = [r for r in part1_rows if r["model"] == model_name and r["attribute"] == attr_name]
            if sub_r:
                print(f"  {attr_name}: MLP dep={np.mean([r['mlp_dependence'] for r in sub_r]):+.4f} "
                      f"linear dep={np.mean([r['linear_dependence'] for r in sub_r]):+.4f} "
                      f"(n={len(sub_r)})", flush=True)

        # ===== PART 2: MLP residualization for continuous attrs (age, bmi) =====
        for attr_name, attr in [("age", age), ("bmi", bmi)]:
            valid = ~np.isnan(attr)
            tr = train_idx[valid[train_idx]]; te = test_idx[valid[test_idx]]
            a_tr = attr[tr]; a_te = attr[te]
            mu, sd = a_tr.mean(), a_tr.std() + 1e-9
            D_tr = ((a_tr - mu) / sd).reshape(-1, 1); D_te = ((a_te - mu) / sd).reshape(-1, 1)
            Xl_tr, Xl_te = lin_resid(emb_s[tr], emb_s[te], D_tr, D_te)
            Xm_tr, Xm_te = mlp_resid(emb_s[tr], emb_s[te], D_tr, D_te, d)
            for dz in KEY_DISEASES:
                di = dz_idx[dz]
                mt = masks[tr, di]; me = masks[te, di]
                if mt.sum() < 50 or me.sum() < 50:
                    continue
                yt = labels[tr, di]; ye = labels[te, di]
                if yt[mt].sum() < 10 or ye[me].sum() < 10:
                    continue
                try:
                    base = lin_auroc(emb_s[tr][mt], yt[mt], emb_s[te][me], ye[me])
                    lin_d = base - lin_auroc(Xl_tr[mt], yt[mt], Xl_te[me], ye[me])
                    mlp_d = base - lin_auroc(Xm_tr[mt], yt[mt], Xm_te[me], ye[me])
                    part2_rows.append({"model": model_name, "attribute": attr_name,
                                       "disease": dz, "linear_drop": lin_d, "mlp_drop": mlp_d})
                except Exception as e:
                    print(f"    ERR {attr_name}/{dz}: {e}", flush=True)

    p1 = pd.DataFrame(part1_rows); p1.to_csv(os.path.join(RESULT_DIR, "keystone_a_nonlinear_use.csv"), index=False)
    p2 = pd.DataFrame(part2_rows); p2.to_csv(os.path.join(RESULT_DIR, "keystone_a_mlp_residualization_continuous.csv"), index=False)
    pd_dec = pd.DataFrame(decode_rows); pd_dec.to_csv(os.path.join(RESULT_DIR, "keystone_a_decodability.csv"), index=False)

    print("\n" + "=" * 64)
    print("KEYSTONE A SUMMARY")
    print("=" * 64)
    print("\n[Part 1] Nonlinear (MLP) vs linear dependence for demographics:")
    for a in ["sex", "race_black_vs_white"]:
        s = p1[p1["attribute"] == a]
        if len(s):
            print(f"  {a:22s} MLP dep = {s['mlp_dependence'].mean():+.4f} "
                  f"[{s['mlp_dependence'].quantile(.025):+.4f}, {s['mlp_dependence'].quantile(.975):+.4f}] | "
                  f"linear dep = {s['linear_dependence'].mean():+.4f} (n={len(s)})")
    print("\n[Decodability] MLP decode AUROC of attribute from X vs linearly-residualized X_res:")
    for a in ["sex", "race_black_vs_white"]:
        s = pd_dec[pd_dec["attribute"] == a]
        if len(s):
            print(f"  {a:22s} X = {s['decode_auroc_X'].mean():.3f} -> X_res = {s['decode_auroc_Xres'].mean():.3f}")

    print("\n[Part 2] MLP residualization (continuous attrs), mean drop:")
    mlp_drop_by_attr = dict(LINEAR_DROP_S1)  # start from linear; override age/bmi with MLP
    for a in ["age", "bmi"]:
        s = p2[p2["attribute"] == a]
        if len(s):
            lin_m = s["linear_drop"].mean(); mlp_m = s["mlp_drop"].mean()
            mlp_drop_by_attr[a] = mlp_m
            print(f"  {a:6s} linear={lin_m:.4f}  MLP={mlp_m:.4f}")
    attrs = list(LINEAR_DROP_S1.keys())
    lin_rank = [LINEAR_DROP_S1[a] for a in attrs]
    hyb_rank = [mlp_drop_by_attr[a] for a in attrs]
    rho, pval = spearmanr(lin_rank, hyb_rank)
    print(f"\n  Spearman rho (linear ranking vs linear/MLP-hybrid ranking, 24 attrs) "
          f"= {rho:.4f} (p={pval:.2e})")
    print(f"\nSaved to {RESULT_DIR}/  | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
