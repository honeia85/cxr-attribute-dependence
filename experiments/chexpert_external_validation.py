"""
External validation on CheXpert Plus (Stanford).

CheXpert Plus has NO ICD linkage, so the OR -> dependence regression CANNOT be
replicated. What CAN be replicated on a second institution with a different
source population are the two demographic dissociations that anchor the paper:

  (1) Encoding-dependence DISSOCIATION: demographics are strongly encoded
      (high linear-probe AUROC / R^2) yet contribute little to finding
      prediction (small AUROC drop after linear residualization).
  (2) Dependence-subgroup-gap DECOUPLING: race residualization drop is far
      smaller than the 4-category race subgroup AUROC gap.

Attributes: sex, race (Black-vs-White), age, bmi  (no comorbidities -- no ICD).
Findings: 10 (CheXpert labels minus Fracture, Lung Lesion).
Models: 6 strict-clean (residualization) + 3 overlap (subgroup gap only).

Usage:
    PYTHONPATH=. python experiments/chexpert_external_validation.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEX_DIR = os.path.join(PROJECT_DIR, "embeddings_chexpert")
META_CSV = os.path.join(PROJECT_DIR, "chexpert_plus_metadata.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "results_chexpert")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42

CLEAN_MODELS = [
    "ResNet50-ImageNet", "DINOv2-base", "BiomedCLIP",
    "XRV-DenseNet-nih", "CLIP-ViT-B16", "ConvNeXtV2-Base",
]
OVERLAP_MODELS = ["RAD-DINO", "CheXzero", "CheSS-ResNet50"]

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Support Devices",
]
RACE_CATEGORIES = ["White", "Black", "Asian", "Other"]


def load_meta_aligned(canonical_ids):
    md = pd.read_csv(META_CSV)
    md = md.set_index("path_to_image")
    md = md.reindex([str(x) for x in canonical_ids])
    md = md.reset_index().rename(columns={"index": "path_to_image"})
    return md


def get_aligned_embeddings(model_name, canonical_ids):
    emb = np.load(os.path.join(CHEX_DIR, f"{model_name}_embeddings.npy"))
    ids = np.load(os.path.join(CHEX_DIR, f"{model_name}_image_ids.npy"),
                  allow_pickle=True)
    id_to_idx = {str(d): i for i, d in enumerate(ids)}
    aligned = [id_to_idx[str(d)] for d in canonical_ids]
    return emb[aligned]


def build_labels(md):
    """1 -> pos, 0 -> neg, else (NaN, -1 uncertain) -> masked. (U-Ignore)"""
    L = md[FINDINGS].values.astype(float)
    labels = np.where(L == 1.0, 1.0, 0.0).astype(np.float32)
    masks = np.isin(L, [0.0, 1.0])
    return labels, masks


def residualize(X_tr, X_te, D_tr, D_te, alpha=1.0):
    m = Ridge(alpha=alpha)
    m.fit(D_tr, X_tr)
    return X_tr - m.predict(D_tr), X_te - m.predict(D_te)


def disease_auroc(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


def main():
    t0 = time.time()
    canonical_ids = np.load(os.path.join(CHEX_DIR, "canonical_image_ids.npy"),
                            allow_pickle=True)
    split = np.load(os.path.join(CHEX_DIR, "canonical_split.npz"))
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    md = load_meta_aligned(canonical_ids)
    labels, masks = build_labels(md)
    dz_idx = {d: i for i, d in enumerate(FINDINGS)}

    # Attribute vectors (image level)
    sex = md["gender_binary"].values.astype(float)
    age = md["age"].values.astype(float)
    bmi = md["bmi"].values.astype(float)
    race_grp = md["race_group"].astype(str).values
    bw_mask = np.isin(race_grp, ["Black", "White"])
    race_bin = (race_grp == "Black").astype(float)
    has_race = race_grp != "Unknown"
    print(f"N={len(canonical_ids):,} | train={len(train_idx):,} test={len(test_idx):,}")
    print(f"BW subset: {bw_mask.sum():,} ({race_bin[bw_mask].mean():.1%} Black) | "
          f"race-coded: {has_race.sum():,}")

    enc_rows, dep_rows, sg_rows = [], [], []

    for model_name in CLEAN_MODELS + OVERLAP_MODELS:
        is_clean = model_name in CLEAN_MODELS
        print(f"\n--- {model_name} ({'clean' if is_clean else 'overlap'}) ---")
        try:
            emb = get_aligned_embeddings(model_name, canonical_ids)
        except Exception as e:
            print(f"  SKIP (embeddings): {e}")
            continue
        scaler = StandardScaler().fit(emb[train_idx])
        emb_s = scaler.transform(emb)

        # ---------- ENCODING (linear probe) ----------
        if is_clean:
            # sex
            tr = train_idx[~np.isnan(sex[train_idx])]
            te = test_idx[~np.isnan(sex[test_idx])]
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(emb_s[tr], sex[tr])
            enc_sex = roc_auc_score(sex[te], clf.predict_proba(emb_s[te])[:, 1])
            # race BW
            trb = train_idx[bw_mask[train_idx]]
            teb = test_idx[bw_mask[test_idx]]
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(emb_s[trb], race_bin[trb])
            enc_race = roc_auc_score(race_bin[teb], clf.predict_proba(emb_s[teb])[:, 1])
            # age, bmi (Ridge R^2)
            tra = train_idx[~np.isnan(age[train_idx])]; tea = test_idx[~np.isnan(age[test_idx])]
            rg = Ridge(alpha=1.0).fit(emb_s[tra], age[tra])
            enc_age = r2_score(age[tea], rg.predict(emb_s[tea]))
            trm = train_idx[~np.isnan(bmi[train_idx])]; tem = test_idx[~np.isnan(bmi[test_idx])]
            rg = Ridge(alpha=1.0).fit(emb_s[trm], bmi[trm])
            enc_bmi = r2_score(bmi[tem], rg.predict(emb_s[tem]))
            enc_rows.append({"model": model_name, "enc_sex_auroc": enc_sex,
                             "enc_race_auroc": enc_race, "enc_age_r2": enc_age,
                             "enc_bmi_r2": enc_bmi})
            print(f"  encoding: sex={enc_sex:.3f} race={enc_race:.3f} "
                  f"age_R2={enc_age:.3f} bmi_R2={enc_bmi:.3f}")

            # ---------- DEPENDENCE (residualization drop) ----------
            for attr_name, attr, sub in [
                ("sex", sex, None), ("race_black_vs_white", race_bin, bw_mask),
                ("age", age, None), ("bmi", bmi, None)]:
                # rows valid for this attribute
                valid = ~np.isnan(attr) if sub is None else (sub & ~np.isnan(attr))
                tr = train_idx[valid[train_idx]]; te = test_idx[valid[test_idx]]
                D_tr = attr[tr].reshape(-1, 1); D_te = attr[te].reshape(-1, 1)
                if attr_name in ("age", "bmi"):  # standardize continuous
                    mu, sd = D_tr.mean(), D_tr.std() + 1e-9
                    D_tr = (D_tr - mu) / sd; D_te = (D_te - mu) / sd
                X_tr, X_te = emb_s[tr], emb_s[te]
                Xr_tr, Xr_te = residualize(X_tr, X_te, D_tr, D_te)
                for d in FINDINGS:
                    di = dz_idx[d]
                    mt = masks[tr, di]; me = masks[te, di]
                    yt = labels[tr, di]; ye = labels[te, di]
                    if mt.sum() < 50 or me.sum() < 50 or yt[mt].sum() < 10 or ye[me].sum() < 10:
                        continue
                    try:
                        ab = disease_auroc(X_tr[mt], yt[mt], X_te[me], ye[me])
                        ar = disease_auroc(Xr_tr[mt], yt[mt], Xr_te[me], ye[me])
                        dep_rows.append({"model": model_name, "attribute": attr_name,
                                         "disease": d, "auroc_baseline": ab,
                                         "auroc_residualized": ar, "auroc_drop": ab - ar})
                    except Exception as e:
                        print(f"    ERR {attr_name}/{d}: {e}")
                sub_d = [r for r in dep_rows if r["model"] == model_name and r["attribute"] == attr_name]
                if sub_d:
                    print(f"    dep[{attr_name}] mean drop = "
                          f"{np.mean([r['auroc_drop'] for r in sub_d]):.4f} (n={len(sub_d)})")

        # ---------- RACE SUBGROUP GAP (all 9 models) ----------
        for d in FINDINGS:
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
            rte = race_grp[tes]
            aucs = {}
            for cat in RACE_CATEGORIES:
                cm = rte == cat
                if cm.sum() < 30 or ye[cm].sum() < 5 or (1 - ye[cm]).sum() < 5:
                    continue
                try:
                    aucs[cat] = roc_auc_score(ye[cm], p[cm])
                except Exception:
                    pass
            if len(aucs) >= 2:
                sg_rows.append({"model": model_name, "disease": d,
                                "n_groups": len(aucs),
                                "auroc_max": max(aucs.values()),
                                "auroc_min": min(aucs.values()),
                                "gap_maxmin": max(aucs.values()) - min(aucs.values())})

    pd.DataFrame(enc_rows).to_csv(os.path.join(OUT_DIR, "chexpert_encoding.csv"), index=False)
    pd.DataFrame(dep_rows).to_csv(os.path.join(OUT_DIR, "chexpert_dependence.csv"), index=False)
    sg = pd.DataFrame(sg_rows)
    sg.to_csv(os.path.join(OUT_DIR, "chexpert_race_subgroup_gaps.csv"), index=False)

    # ---------- SUMMARY ----------
    dep = pd.DataFrame(dep_rows)
    print("\n" + "=" * 64)
    print("CheXpert Plus EXTERNAL VALIDATION SUMMARY (6 clean models)")
    print("=" * 64)
    enc = pd.DataFrame(enc_rows)
    print("\n[Encoding] mean across clean models:")
    print(f"  sex  AUROC = {enc['enc_sex_auroc'].mean():.3f}")
    print(f"  race AUROC = {enc['enc_race_auroc'].mean():.3f}")
    print(f"  age  R^2   = {enc['enc_age_r2'].mean():.3f}")
    print(f"  bmi  R^2   = {enc['enc_bmi_r2'].mean():.3f}")
    print("\n[Dependence] mean AUROC drop across clean models x findings:")
    for a in ["sex", "race_black_vs_white", "age", "bmi"]:
        s = dep[dep["attribute"] == a]["auroc_drop"]
        if len(s):
            print(f"  {a:22s} = {s.mean():.4f}  (95% CI [{s.quantile(.025):.4f}, {s.quantile(.975):.4f}], n={len(s)})")
    race_dep = dep[(dep["attribute"] == "race_black_vs_white") &
                   (dep["model"].isin(CLEAN_MODELS))]["auroc_drop"].mean()
    race_gap_clean = sg[sg["model"].isin(CLEAN_MODELS)]["gap_maxmin"].mean()
    race_gap_all = sg["gap_maxmin"].mean()
    print("\n[Decoupling] race dependence vs 4-category race subgroup gap:")
    print(f"  race dependence (clean)      = {race_dep:.4f}")
    print(f"  race subgroup gap (clean)    = {race_gap_clean:.4f}  ({race_gap_clean/max(race_dep,1e-9):.0f}x)")
    print(f"  race subgroup gap (9 models) = {race_gap_all:.4f}")
    print(f"\nSaved to {OUT_DIR}/  | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
