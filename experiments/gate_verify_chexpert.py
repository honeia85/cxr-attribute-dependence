"""
Gate check after re-extracting CheXpert chunk-0 embeddings with the 16-bit fix.

Loads a freshly-extracted per-chunk embedding file (e.g. ResNet50-ImageNet_chunk0),
matches it to the CheXpert Plus metadata by path_to_image, and trains a linear
finding probe (using the official train/val vs test split) for a couple of
findings. If the 16-bit PNG fix worked, finding AUROC should recover from the
broken ~0.52 to roughly the MIMIC range (~0.80-0.86). Self-contained; no canonical
files needed (operates on whatever images are in chunk 0).

Usage:
    PYTHONPATH=. python experiments/gate_verify_chexpert.py ResNet50-ImageNet
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(PROJ, "embeddings_chexpert")
META = os.path.join(PROJ, "chexpert_plus_metadata.csv")
FINDINGS = ["Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia"]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ResNet50-ImageNet"
    emb = np.load(os.path.join(EMB, f"{model}_chunk0_emb.npy"))
    ids = np.load(os.path.join(EMB, f"{model}_chunk0_ids.npy"), allow_pickle=True)
    ids = np.array([str(x) for x in ids])
    print(f"{model} chunk0: emb {emb.shape}, ids {len(ids)}")

    md = pd.read_csv(META).set_index("path_to_image")
    md = md.reindex(ids)  # align metadata rows to embedding order
    split = md["split"].values
    is_tr = np.isin(split, ["train", "val"])
    is_te = split == "test"
    print(f"matched train+val={is_tr.sum()} test={is_te.sum()} "
          f"(unmatched meta rows: {md['split'].isna().sum()})")

    sc = StandardScaler().fit(emb[is_tr])
    X = sc.transform(emb)
    print(f"\n{'finding':28s} {'train_n':>8s} {'test_n':>7s} {'prev':>5s} {'AUROC':>6s}")
    aucs = []
    for f in FINDINGS:
        y = md[f].values.astype(float)
        m = np.isin(y, [0.0, 1.0])
        yy = (y == 1.0).astype(int)
        mt = is_tr & m; me = is_te & m
        if mt.sum() < 100 or me.sum() < 50 or yy[me].sum() < 10:
            print(f"{f:28s} (insufficient labels)")
            continue
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[mt], yy[mt])
        a = roc_auc_score(yy[me], clf.predict_proba(X[me])[:, 1])
        aucs.append(a)
        print(f"{f:28s} {int(mt.sum()):8d} {int(me.sum()):7d} {yy[me].mean():5.2f} {a:6.3f}")

    if aucs:
        mean = float(np.mean(aucs))
        print(f"\nMean finding AUROC = {mean:.3f}")
        if mean >= 0.75:
            print(">>> GATE PASS: findings recovered (fix works). Proceed to full re-extraction.")
        else:
            print(">>> GATE FAIL: findings still near chance. Diagnose further (inversion? windowing?) before full run.")


if __name__ == "__main__":
    main()
