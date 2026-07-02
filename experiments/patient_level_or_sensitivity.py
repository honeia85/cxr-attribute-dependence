"""
Patient-level OR sensitivity analysis.

Image-level ORs treat repeated images from the same patient as independent,
potentially inflating associations. Here we recompute ORs at the patient level
(one random image per subject_id) and check whether the main regression finding
(R^2) changes materially.

Output: results/patient_level_or_sensitivity.csv + console summary.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

SEED = 42
HALDANE = 0.5  # Haldane correction for zero cells

# -- Attribute mapping --------------------------------------------------
# Map regression-data attribute names → metadata column names
# Most are identical; only 'sex' differs.
ATTR_TO_COL = {
    'sex': 'gender_binary',  # 0=F, 1=M in metadata
}

BINARY_ATTRS = [
    'hypertension', 'heart_failure', 'atrial_fibrillation',
    'coronary_artery_disease', 'stroke', 'diabetes', 'hyperlipidemia',
    'obesity', 'hypothyroidism', 'ckd', 'aki', 'copd', 'asthma',
    'respiratory_failure', 'pulmonary_fibrosis', 'liver_disease',
    'anemia', 'smoking_history', 'cancer_history', 'depression', 'sex',
]

CONTINUOUS_ATTRS = ['age', 'bmi']

ALL_ATTRS = BINARY_ATTRS + CONTINUOUS_ATTRS  # 23 total

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Lung Opacity', 'Pleural Effusion',
    'Pneumonia', 'Pneumothorax', 'Support Devices',
]


def compute_or_matched(a_col, d_binary, haldane=HALDANE):
    """
    Compute OR matching the original two_factor_regression.py approach.

    a_col: attribute values (binary 0/1 or continuous-already-binarized)
    d_binary: disease column already preprocessed to binary (1=pos, 0=not-pos)
              via the original approach: dropna then (==1).astype(int)

    Haldane correction applied only when a cell is zero.
    """
    mask = a_col.notna() & (a_col.isin([0, 1]))
    a = a_col[mask].astype(int)
    d = d_binary.reindex(a.index)  # align
    valid = a.index.intersection(d.index)
    a = a.loc[valid]
    d = d.loc[valid]
    if len(a) < 10:
        return np.nan, np.nan, np.nan

    tp = ((a == 1) & (d == 1)).sum()
    fp = ((a == 1) & (d == 0)).sum()
    fn = ((a == 0) & (d == 1)).sum()
    tn = ((a == 0) & (d == 0)).sum()

    if min(tp, fp, fn, tn) == 0:
        tp, fp, fn, tn = tp + haldane, fp + haldane, fn + haldane, tn + haldane

    OR = (tp * tn) / (fp * fn)
    log_or = np.log(OR)
    return OR, log_or, abs(log_or)


def binarize_continuous(series):
    """Binarize by median split: >= median → 1, else 0."""
    valid = series.dropna()
    if len(valid) == 0:
        return series
    med = valid.median()
    return (series >= med).astype(float).where(series.notna())


def main():
    print("=" * 70)
    print("PATIENT-LEVEL OR SENSITIVITY ANALYSIS")
    print("=" * 70)

    # -- Load data ------------------------------------------------------
    meta = pd.read_csv('full_mimic_cxr_metadata.csv')
    reg_data = pd.read_csv('results/two_factor_regression_data.csv')
    # `image_or` was previously a separate file (`odds_ratios_attr_disease.csv`);
    # since OR is image-level (model-independent), it can be recovered from
    # the per-cell regression-data file by dropping duplicates over (attr, disease).
    image_or = (
        reg_data[['attribute', 'disease', 'abs_log_or']]
        .drop_duplicates(subset=['attribute', 'disease'])
        .reset_index(drop=True)
    )

    print(f"\nImages: {len(meta):,}")
    print(f"Patients: {meta['subject_id'].nunique():,}")

    # -- Patient-level subsample ----------------------------------------
    np.random.seed(SEED)
    patient_df = meta.groupby('subject_id').apply(
        lambda g: g.sample(1, random_state=SEED)
    ).reset_index(drop=True)
    print(f"Patient-level subset: {len(patient_df):,} images (1 per patient)")

    # -- Compute ORs matching original approach ---------------------------
    # The original script: for each disease, dropna on that column, then
    # disease_binary = (val == 1.0).astype(int).  This treats -1 and 0 both
    # as "not positive".  For attributes, it uses isin([0,1]) filter.
    # For age/bmi, it computes median on the valid subset per disease.

    results = []
    for disease in DISEASES:
        # -- Image-level disease preprocessing (matching original) --
        dis_img = meta[disease].dropna()
        valid_idx_img = dis_img.index
        d_binary_img = (dis_img == 1.0).astype(int)

        # -- Patient-level disease preprocessing --
        dis_pat = patient_df[disease].dropna()
        valid_idx_pat = dis_pat.index
        d_binary_pat = (dis_pat == 1.0).astype(int)

        for attr in BINARY_ATTRS:
            col = ATTR_TO_COL.get(attr, attr)
            # Image
            OR_i, log_or_i, abs_log_or_i = compute_or_matched(
                meta.loc[valid_idx_img, col], d_binary_img
            )
            # Patient
            OR_p, log_or_p, abs_log_or_p = compute_or_matched(
                patient_df.loc[valid_idx_pat, col], d_binary_pat
            )
            results.append({
                'attribute': attr, 'disease': disease,
                'or_image': OR_i, 'log_or_image': log_or_i,
                'abs_log_or_image': abs_log_or_i,
                'or_patient': OR_p, 'log_or_patient': log_or_p,
                'abs_log_or_patient': abs_log_or_p,
            })

        # Age (median split, computed per disease subset)
        for cont_attr, cont_col in [('age', 'age'), ('bmi', 'bmi')]:
            # Image
            vals_img = meta.loc[valid_idx_img, cont_col]
            vm_img = vals_img.notna()
            if vm_img.sum() > 100:
                med_img = vals_img[vm_img].median()
                a_img = (vals_img[vm_img] >= med_img).astype(int)
                OR_i, log_or_i, abs_log_or_i = compute_or_matched(
                    a_img, d_binary_img.reindex(a_img.index)
                )
            else:
                OR_i, log_or_i, abs_log_or_i = np.nan, np.nan, np.nan

            # Patient
            vals_pat = patient_df.loc[valid_idx_pat, cont_col]
            vm_pat = vals_pat.notna()
            if vm_pat.sum() > 100:
                med_pat = vals_pat[vm_pat].median()
                a_pat = (vals_pat[vm_pat] >= med_pat).astype(int)
                OR_p, log_or_p, abs_log_or_p = compute_or_matched(
                    a_pat, d_binary_pat.reindex(a_pat.index)
                )
            else:
                OR_p, log_or_p, abs_log_or_p = np.nan, np.nan, np.nan

            results.append({
                'attribute': cont_attr, 'disease': disease,
                'or_image': OR_i, 'log_or_image': log_or_i,
                'abs_log_or_image': abs_log_or_i,
                'or_patient': OR_p, 'log_or_patient': log_or_p,
                'abs_log_or_patient': abs_log_or_p,
            })

    or_df = pd.DataFrame(results)

    # -- Verify image-level ORs match existing file ---------------------
    merged_check = or_df[['attribute', 'disease', 'abs_log_or_image']].merge(
        image_or[['attribute', 'disease', 'abs_log_or']], on=['attribute', 'disease']
    )
    corr_check = np.corrcoef(
        merged_check['abs_log_or_image'],
        merged_check['abs_log_or']
    )[0, 1]
    print(f"\nImage-level OR verification (corr with existing): {corr_check:.6f}")

    # -- Compare image vs patient ORs -----------------------------------
    valid = or_df.dropna(subset=['abs_log_or_image', 'abs_log_or_patient'])
    rho_or, p_or = stats.spearmanr(valid['abs_log_or_image'], valid['abs_log_or_patient'])
    r_pearson = np.corrcoef(valid['abs_log_or_image'], valid['abs_log_or_patient'])[0, 1]

    print(f"\n{'-' * 50}")
    print(f"Image vs Patient |log(OR)| comparison (n={len(valid)} pairs):")
    print(f"  Spearman rho  = {rho_or:.4f}  (p = {p_or:.2e})")
    print(f"  Pearson r     = {r_pearson:.4f}")
    print(f"  Mean |log(OR)| image:   {valid['abs_log_or_image'].mean():.4f}")
    print(f"  Mean |log(OR)| patient: {valid['abs_log_or_patient'].mean():.4f}")

    # -- Replace OR in regression data → refit models -------------------
    # Merge patient-level ORs into 920-row regression data
    or_lookup = or_df[['attribute', 'disease', 'abs_log_or_patient', 'log_or_patient']].copy()
    reg = reg_data.merge(or_lookup, on=['attribute', 'disease'], how='left')

    # Sanity: check merge
    n_na = reg['abs_log_or_patient'].isna().sum()
    if n_na > 0:
        print(f"\n  WARNING: {n_na} rows could not be matched (will drop)")
    reg = reg.dropna(subset=['abs_log_or_patient'])

    # -- M1: drop ~ |log(OR)| only -------------------------------------
    # Image-level (original)
    m1_image = smf.ols('auroc_drop ~ abs_log_or', data=reg).fit()
    # Patient-level
    m1_patient = smf.ols('auroc_drop ~ abs_log_or_patient', data=reg).fit()

    print(f"\n{'=' * 50}")
    print(f"M1: auroc_drop ~ |log(OR)|")
    print(f"  Image-level  R² = {m1_image.rsquared:.4f}")
    print(f"  Patient-level R² = {m1_patient.rsquared:.4f}")
    print(f"  Delta R²        = {m1_patient.rsquared - m1_image.rsquared:+.4f}")

    # -- M5: drop ~ |log(OR)| + disease intercepts ---------------------
    m5_image = smf.ols('auroc_drop ~ abs_log_or + C(disease)', data=reg).fit()
    m5_patient = smf.ols('auroc_drop ~ abs_log_or_patient + C(disease)', data=reg).fit()

    print(f"\nM5: auroc_drop ~ |log(OR)| + C(disease)")
    print(f"  Image-level  R² = {m5_image.rsquared:.4f}")
    print(f"  Patient-level R² = {m5_patient.rsquared:.4f}")
    print(f"  Delta R²        = {m5_patient.rsquared - m5_image.rsquared:+.4f}")

    # -- Beta coefficients ----------------------------------------------
    print(f"\nM1 beta (|log(OR)|):")
    print(f"  Image:   {m1_image.params['abs_log_or']:.6f}  (p={m1_image.pvalues['abs_log_or']:.2e})")
    print(f"  Patient: {m1_patient.params['abs_log_or_patient']:.6f}  (p={m1_patient.pvalues['abs_log_or_patient']:.2e})")

    print(f"\nM5 beta (|log(OR)|):")
    print(f"  Image:   {m5_image.params['abs_log_or']:.6f}  (p={m5_image.pvalues['abs_log_or']:.2e})")
    print(f"  Patient: {m5_patient.params['abs_log_or_patient']:.6f}  (p={m5_patient.pvalues['abs_log_or_patient']:.2e})")

    # -- Per-disease breakdown ------------------------------------------
    print(f"\n{'-' * 50}")
    print("Per-disease Spearman rho (image vs patient |log(OR)|):")
    for disease in DISEASES:
        sub = valid[valid['disease'] == disease]
        if len(sub) >= 5:
            rho_d, _ = stats.spearmanr(sub['abs_log_or_image'], sub['abs_log_or_patient'])
            print(f"  {disease:35s}  rho = {rho_d:.4f}  (n={len(sub)})")

    # -- Top-5 attribute ranking stability ------------------------------
    print(f"\n{'-' * 50}")
    print("Top-5 attributes by mean |log(OR)| across diseases:")
    mean_image = valid.groupby('attribute')['abs_log_or_image'].mean().sort_values(ascending=False)
    mean_patient = valid.groupby('attribute')['abs_log_or_patient'].mean().sort_values(ascending=False)
    rank_rho, _ = stats.spearmanr(mean_image, mean_patient)
    print(f"\n  {'Image-level':35s}  {'Patient-level':35s}")
    for i in range(5):
        ai = mean_image.index[i]
        ap = mean_patient.index[i]
        print(f"  {i+1}. {ai:30s}  {i+1}. {ap:30s}")
    print(f"\n  Rank correlation (all 23): rho = {rank_rho:.4f}")

    # -- Bootstrap stability (100 resamples of patient selection) -------
    print(f"\n{'-' * 50}")
    print("Bootstrap stability: 100 random patient-sample seeds")
    r2_m1_list = []
    r2_m5_list = []
    rho_list = []

    # Pre-build per-patient index lists for fast sampling
    patient_groups = meta.groupby('subject_id').apply(lambda g: g.index.tolist())

    for seed_i in range(100):
        rng = np.random.RandomState(seed_i)
        # Fast: pick one random index per patient
        idx = patient_groups.apply(lambda indices: rng.choice(indices)).values
        pdf_i = meta.iloc[idx].copy()
        pdf_i = pdf_i.reset_index(drop=True)

        res_i = []
        for disease in DISEASES:
            dis_i = pdf_i[disease].dropna()
            valid_idx_i = dis_i.index
            d_bin_i = (dis_i == 1.0).astype(int)

            for attr in BINARY_ATTRS:
                col = ATTR_TO_COL.get(attr, attr)
                _, _, alor = compute_or_matched(pdf_i.loc[valid_idx_i, col], d_bin_i)
                res_i.append({'attribute': attr, 'disease': disease, 'abs_log_or_p': alor})

            for cont_attr, cont_col in [('age', 'age'), ('bmi', 'bmi')]:
                vals = pdf_i.loc[valid_idx_i, cont_col]
                vm = vals.notna()
                if vm.sum() > 100:
                    med = vals[vm].median()
                    a_bin = (vals[vm] >= med).astype(int)
                    _, _, alor = compute_or_matched(a_bin, d_bin_i.reindex(a_bin.index))
                else:
                    alor = np.nan
                res_i.append({'attribute': cont_attr, 'disease': disease, 'abs_log_or_p': alor})

        or_i = pd.DataFrame(res_i)
        reg_i = reg_data.merge(or_i, on=['attribute', 'disease'], how='left').dropna(subset=['abs_log_or_p'])

        m1_i = smf.ols('auroc_drop ~ abs_log_or_p', data=reg_i).fit()
        m5_i = smf.ols('auroc_drop ~ abs_log_or_p + C(disease)', data=reg_i).fit()
        r2_m1_list.append(m1_i.rsquared)
        r2_m5_list.append(m5_i.rsquared)

        # Spearman rho with image-level
        merged_i = or_i.merge(
            image_or[['attribute', 'disease', 'abs_log_or']],
            on=['attribute', 'disease']
        ).dropna()
        rho_i, _ = stats.spearmanr(merged_i['abs_log_or'], merged_i['abs_log_or_p'])
        rho_list.append(rho_i)

        if (seed_i + 1) % 20 == 0:
            print(f"  ... {seed_i + 1}/100 done")

    r2_m1_arr = np.array(r2_m1_list)
    r2_m5_arr = np.array(r2_m5_list)
    rho_arr = np.array(rho_list)

    print(f"  M1 R²: {r2_m1_arr.mean():.4f} [{np.percentile(r2_m1_arr, 2.5):.4f}, {np.percentile(r2_m1_arr, 97.5):.4f}]")
    print(f"  M5 R²: {r2_m5_arr.mean():.4f} [{np.percentile(r2_m5_arr, 2.5):.4f}, {np.percentile(r2_m5_arr, 97.5):.4f}]")
    print(f"  OR rho: {rho_arr.mean():.4f} [{np.percentile(rho_arr, 2.5):.4f}, {np.percentile(rho_arr, 97.5):.4f}]")

    # -- Save results ---------------------------------------------------
    or_df.to_csv('results/patient_level_or_sensitivity.csv', index=False)
    print(f"\nSaved: results/patient_level_or_sensitivity.csv ({len(or_df)} rows)")

    # -- Final conclusion -----------------------------------------------
    print(f"\n{'=' * 70}")
    print("CONCLUSION:")
    delta_m1 = m1_patient.rsquared - m1_image.rsquared
    delta_m5 = m5_patient.rsquared - m5_image.rsquared

    # Interpretation: direction matters — positive means patient-level is STRONGER
    print(f"  Image-level  M1 R² = {m1_image.rsquared:.4f}, M5 R² = {m5_image.rsquared:.4f}")
    print(f"  Patient-level M1 R² = {m1_patient.rsquared:.4f}, M5 R² = {m5_patient.rsquared:.4f}")
    print(f"  Delta: M1 {delta_m1:+.4f}, M5 {delta_m5:+.4f}")
    print(f"  Image-to-patient OR rank correlation: Spearman rho = {rho_or:.4f}")
    print(f"  Bootstrap 95% CI: M1 R² [{np.percentile(r2_m1_arr, 2.5):.4f}, {np.percentile(r2_m1_arr, 97.5):.4f}]")
    print(f"                    M5 R² [{np.percentile(r2_m5_arr, 2.5):.4f}, {np.percentile(r2_m5_arr, 97.5):.4f}]")
    print()
    if abs(delta_m5) < 0.05:
        print(f"  ROBUST: Patient-level ORs preserve the main finding.")
        print(f"  M5 R² changes by only {delta_m5:+.4f} (within noise).")
        if delta_m1 > 0:
            print(f"  M1 R² actually IMPROVES ({delta_m1:+.4f}), suggesting repeated images")
            print(f"  add noise that attenuates the OR-drop relationship.")
    else:
        print(f"  MATERIAL CHANGE in M5: {delta_m5:+.4f} — further investigation needed.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
