"""
Uncertainty-label sensitivity for the odds ratios (U-Ignore vs U-Zero).

The main analysis uses the U-Ignore policy (uncertain CheXpert labels masked). Here we
recompute the attribute-finding odds ratios under U-Zero (uncertain treated as negative)
and check whether the attribute OR ranking is preserved. If the Spearman correlation
between the U-Ignore and U-Zero attribute rankings is high, the uncertainty-label policy
does not drive the OR-dependence result.

Usage:
    PYTHONPATH=. python experiments/u_zero_or_sensitivity.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.config import CHEXPERT_ALL, CHEXPERT_EXCLUDE, BINARY_FEATURES, MIMIC_IV_ADMISSIONS
from experiments.data import load_metadata, load_canonical_ids
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ADMISSIONS_PATH = MIMIC_IV_ADMISSIONS

FINDINGS = [d for d in CHEXPERT_ALL if d not in CHEXPERT_EXCLUDE]


def collapse_race(label):
    if pd.isna(label): return "Unknown"
    s = str(label).upper()
    if s.startswith("WHITE") or s == "PORTUGUESE": return "White"
    if s.startswith("BLACK"): return "Black"
    if s.startswith("ASIAN") or s == "SOUTH ASIAN": return "Asian"
    if s in {"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"}: return "Unknown"
    return "Other"


def haldane_logor(a_pos_exp, a_pos_unexp, n_exp, n_unexp):
    a = a_pos_exp + 0.5; b = (n_exp - a_pos_exp) + 0.5
    c = a_pos_unexp + 0.5; d = (n_unexp - a_pos_unexp) + 0.5
    return abs(np.log((a * d) / (b * c)))


def main():
    canonical_ids = load_canonical_ids()
    md = load_metadata()
    md = md[md["dicom_id"].isin(set(canonical_ids))].reset_index(drop=True)

    # race
    adm = pd.read_csv(ADMISSIONS_PATH, usecols=["subject_id", "race"])
    adm["rg"] = adm["race"].map(collapse_race)
    rmap = adm[adm["rg"] != "Unknown"].groupby("subject_id")["rg"].agg(
        lambda s: s.mode().iloc[0]).to_dict()
    race = md["subject_id"].map(lambda s: rmap.get(s, "Unknown")).values

    # attributes (binary): 20 comorbidities + sex; plus age, bmi median-split; plus race BW
    attrs = {}
    for c in BINARY_FEATURES:
        if c in md.columns:
            attrs[c] = md[c].fillna(0).values.astype(float)
    attrs["age"] = (md["age"].values > md["age"].median()).astype(float)
    attrs["bmi"] = (md["bmi"].values > md["bmi"].median()).astype(float)
    bw = np.isin(race, ["Black", "White"])
    attrs["race_black_vs_white"] = np.where(race == "Black", 1.0, 0.0)

    rows = []
    for policy in ["U-Ignore", "U-Zero"]:
        for attr_name, a in attrs.items():
            sub = bw if attr_name == "race_black_vs_white" else np.ones(len(md), bool)
            for f in FINDINGS:
                y_raw = md[f].values  # 1, 0, -1, NaN
                if policy == "U-Ignore":
                    valid = sub & np.isin(y_raw, [0.0, 1.0])
                    y = (y_raw == 1.0).astype(float)
                else:  # U-Zero: uncertain(-1) and blank(NaN) -> 0
                    valid = sub & ~np.isnan(a)
                    y = (y_raw == 1.0).astype(float)  # everything not ==1 is negative
                m = valid & ~np.isnan(a)
                av = a[m].astype(bool); yv = y[m]
                if av.sum() < 20 or (~av).sum() < 20:
                    continue
                lor = haldane_logor(int(yv[av].sum()), int(yv[~av].sum()),
                                    int(av.sum()), int((~av).sum()))
                rows.append({"policy": policy, "attribute": attr_name, "finding": f, "abs_log_or": lor})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "u_zero_or_sensitivity.csv"), index=False)
    piv = df.groupby(["policy", "attribute"])["abs_log_or"].mean().unstack("policy")
    piv = piv.dropna()
    rho, p = spearmanr(piv["U-Ignore"], piv["U-Zero"])
    print("Mean |log(OR)| per attribute under each uncertainty policy:")
    print(piv.sort_values("U-Ignore", ascending=False).round(3).to_string())
    print(f"\nSpearman rho (U-Ignore vs U-Zero attribute ranking, n={len(piv)} attrs) = {rho:.3f} (p={p:.2e})")
    print(f"Pearson r = {np.corrcoef(piv['U-Ignore'], piv['U-Zero'])[0,1]:.3f}")
    print(f"Saved {RESULT_DIR}/u_zero_or_sensitivity.csv")


if __name__ == "__main__":
    main()
