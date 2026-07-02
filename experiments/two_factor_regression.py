"""
Two-Factor Regression: Confounding = f(Epidemiology) × g(Geometry)

Tests the hypothesis:
  AUROC_drop(attr, model, disease) = β₁·OR(attr,disease) + β₂·erank(model) + β₃·OR×erank + ε

If β₃ < 0 and significant: high erank buffers against epidemiological confounders.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Repo root (this file lives in <root>/experiments/).
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -- 1. Load existing results --
resid = pd.read_csv(f"{BASE}/results/per_comorbidity_residualization.csv")
geom = pd.read_csv(f"{BASE}/results/geometry_analysis.csv")
meta = pd.read_csv(f"{BASE}/full_mimic_cxr_metadata.csv")

print(f"Residualization: {len(resid)} rows")
print(f"Models: {resid['model'].unique()}")
print(f"Diseases: {resid['disease'].unique()}")
print(f"Attributes: {resid['attribute'].nunique()}")

# -- 2. Compute Odds Ratios (attr × disease) --
diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Lung Opacity', 'Pleural Effusion',
            'Pneumonia', 'Pneumothorax', 'Support Devices']
binary_attrs = [
    'hypertension', 'heart_failure', 'atrial_fibrillation', 'coronary_artery_disease',
    'stroke', 'diabetes', 'hyperlipidemia', 'obesity', 'hypothyroidism',
    'ckd', 'aki', 'copd', 'asthma', 'respiratory_failure', 'pulmonary_fibrosis',
    'liver_disease', 'anemia', 'smoking_history', 'cancer_history', 'depression'
]

# For sex: gender_binary (0=F, 1=M)
# For age/bmi: use median split to create binary

or_records = []

for disease in diseases:
    disease_col = meta[disease].dropna()
    valid_idx = disease_col.index
    disease_binary = (disease_col == 1.0).astype(int)

    # Binary comorbidity attributes
    for attr in binary_attrs:
        attr_vals = meta.loc[valid_idx, attr]
        valid_mask = attr_vals.notna() & (attr_vals.isin([0, 1]))
        if valid_mask.sum() < 100:
            continue
        a = attr_vals[valid_mask].astype(int)
        d = disease_binary[valid_mask]

        # 2x2 table
        tp = ((a == 1) & (d == 1)).sum()
        fp = ((a == 1) & (d == 0)).sum()
        fn = ((a == 0) & (d == 1)).sum()
        tn = ((a == 0) & (d == 0)).sum()

        if min(tp, fp, fn, tn) == 0:
            # Haldane correction
            tp, fp, fn, tn = tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5

        odds_ratio = (tp * tn) / (fp * fn)
        log_or = np.log(odds_ratio)

        or_records.append({
            'attribute': attr, 'disease': disease,
            'odds_ratio': odds_ratio, 'log_or': log_or,
            'abs_log_or': abs(log_or)
        })

    # Sex
    sex_vals = meta.loc[valid_idx, 'gender_binary']
    valid_mask = sex_vals.notna()
    a = sex_vals[valid_mask].astype(int)
    d = disease_binary[valid_mask]
    tp = ((a == 1) & (d == 1)).sum()
    fp = ((a == 1) & (d == 0)).sum()
    fn = ((a == 0) & (d == 1)).sum()
    tn = ((a == 0) & (d == 0)).sum()
    if min(tp, fp, fn, tn) == 0:
        tp, fp, fn, tn = tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5
    odds_ratio = (tp * tn) / (fp * fn)
    or_records.append({
        'attribute': 'sex', 'disease': disease,
        'odds_ratio': odds_ratio, 'log_or': np.log(odds_ratio),
        'abs_log_or': abs(np.log(odds_ratio))
    })

    # Age (median split)
    age_vals = meta.loc[valid_idx, 'age']
    valid_mask = age_vals.notna()
    median_age = age_vals[valid_mask].median()
    a = (age_vals[valid_mask] >= median_age).astype(int)
    d = disease_binary[valid_mask]
    tp = ((a == 1) & (d == 1)).sum()
    fp = ((a == 1) & (d == 0)).sum()
    fn = ((a == 0) & (d == 1)).sum()
    tn = ((a == 0) & (d == 0)).sum()
    if min(tp, fp, fn, tn) == 0:
        tp, fp, fn, tn = tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5
    odds_ratio = (tp * tn) / (fp * fn)
    or_records.append({
        'attribute': 'age', 'disease': disease,
        'odds_ratio': odds_ratio, 'log_or': np.log(odds_ratio),
        'abs_log_or': abs(np.log(odds_ratio))
    })

    # BMI (median split)
    bmi_vals = meta.loc[valid_idx, 'bmi']
    valid_mask = bmi_vals.notna()
    if valid_mask.sum() > 100:
        median_bmi = bmi_vals[valid_mask].median()
        a = (bmi_vals[valid_mask] >= median_bmi).astype(int)
        d = disease_binary[valid_mask]
        tp = ((a == 1) & (d == 1)).sum()
        fp = ((a == 1) & (d == 0)).sum()
        fn = ((a == 0) & (d == 1)).sum()
        tn = ((a == 0) & (d == 0)).sum()
        if min(tp, fp, fn, tn) == 0:
            tp, fp, fn, tn = tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5
        odds_ratio = (tp * tn) / (fp * fn)
        or_records.append({
            'attribute': 'bmi', 'disease': disease,
            'odds_ratio': odds_ratio, 'log_or': np.log(odds_ratio),
            'abs_log_or': abs(np.log(odds_ratio))
        })

or_df = pd.DataFrame(or_records)

print("\n" + "="*70)
print("ODDS RATIOS (top 15 by |log(OR)|)")
print("="*70)
or_summary = or_df.groupby('attribute')['abs_log_or'].mean().sort_values(ascending=False)
for attr, val in or_summary.head(15).items():
    print(f"  {attr:30s}  mean|log(OR)| = {val:.3f}")

# -- 3. Build merged dataset --
# Map model → normalized erank
erank_map = {}
for _, row in geom.iterrows():
    model = row['model']
    erank = row['effective_rank']
    dim = row['ambient_dim']
    erank_map[model] = {
        'erank': erank,
        'norm_erank': erank / dim,  # normalized
        'log_erank': np.log(erank)
    }

print("\n" + "="*70)
print("MODEL GEOMETRY")
print("="*70)
for m, v in erank_map.items():
    print(f"  {m:25s}  erank={v['erank']:.1f}  norm={v['norm_erank']:.3f}  log={v['log_erank']:.2f}")

# Merge: resid ← or_df ← erank
merged = resid.merge(or_df[['attribute', 'disease', 'log_or', 'abs_log_or', 'odds_ratio']],
                      on=['attribute', 'disease'], how='left')

# Add erank
merged['erank'] = merged['model'].map(lambda m: erank_map.get(m, {}).get('erank', np.nan))
merged['norm_erank'] = merged['model'].map(lambda m: erank_map.get(m, {}).get('norm_erank', np.nan))
merged['log_erank'] = merged['model'].map(lambda m: erank_map.get(m, {}).get('log_erank', np.nan))

# Drop rows with missing OR (shouldn't be many)
before = len(merged)
merged = merged.dropna(subset=['abs_log_or', 'erank'])
print(f"\nMerged dataset: {before} → {len(merged)} rows (dropped {before - len(merged)} missing)")

# -- 4. TWO-FACTOR REGRESSION --
print("\n" + "="*70)
print("TWO-FACTOR REGRESSION ANALYSIS")
print("="*70)

# Standardize predictors for interpretability
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
merged[['abs_log_or_z', 'log_erank_z']] = scaler.fit_transform(
    merged[['abs_log_or', 'log_erank']])
merged['interaction_z'] = merged['abs_log_or_z'] * merged['log_erank_z']

# Model 1: Epidemiology only
print("\n--- Model 1: AUROC_drop ~ |log(OR)| ---")
m1 = smf.ols('auroc_drop ~ abs_log_or_z', data=merged).fit()
print(f"  R² = {m1.rsquared:.4f}, Adj R² = {m1.rsquared_adj:.4f}")
print(f"  β(|log(OR)|) = {m1.params['abs_log_or_z']:.6f}, p = {m1.pvalues['abs_log_or_z']:.2e}")

# Model 2: Geometry only
print("\n--- Model 2: AUROC_drop ~ log(erank) ---")
m2 = smf.ols('auroc_drop ~ log_erank_z', data=merged).fit()
print(f"  R² = {m2.rsquared:.4f}, Adj R² = {m2.rsquared_adj:.4f}")
print(f"  β(log_erank) = {m2.params['log_erank_z']:.6f}, p = {m2.pvalues['log_erank_z']:.2e}")

# Model 3: Additive
print("\n--- Model 3: AUROC_drop ~ |log(OR)| + log(erank) ---")
m3 = smf.ols('auroc_drop ~ abs_log_or_z + log_erank_z', data=merged).fit()
print(f"  R² = {m3.rsquared:.4f}, Adj R² = {m3.rsquared_adj:.4f}")
print(f"  β(|log(OR)|) = {m3.params['abs_log_or_z']:.6f}, p = {m3.pvalues['abs_log_or_z']:.2e}")
print(f"  β(log_erank) = {m3.params['log_erank_z']:.6f}, p = {m3.pvalues['log_erank_z']:.2e}")

# Model 4: Full interaction
print("\n--- Model 4: AUROC_drop ~ |log(OR)| + log(erank) + |log(OR)|×log(erank) ---")
m4 = smf.ols('auroc_drop ~ abs_log_or_z + log_erank_z + interaction_z', data=merged).fit()
print(f"  R² = {m4.rsquared:.4f}, Adj R² = {m4.rsquared_adj:.4f}")
print(f"  β(|log(OR)|)     = {m4.params['abs_log_or_z']:.6f}, p = {m4.pvalues['abs_log_or_z']:.2e}")
print(f"  β(log_erank)     = {m4.params['log_erank_z']:.6f}, p = {m4.pvalues['log_erank_z']:.2e}")
print(f"  β(interaction)   = {m4.params['interaction_z']:.6f}, p = {m4.pvalues['interaction_z']:.2e}")

# F-test: Model 4 vs Model 1
print(f"\n  ΔR² (M4 vs M1): {m4.rsquared - m1.rsquared:.4f}")
print(f"  ΔR² (M4 vs M3): {m4.rsquared - m3.rsquared:.4f}")

# AIC/BIC comparison
print(f"\n  AIC: M1={m1.aic:.1f}, M2={m2.aic:.1f}, M3={m3.aic:.1f}, M4={m4.aic:.1f}")
print(f"  BIC: M1={m1.bic:.1f}, M2={m2.bic:.1f}, M3={m3.bic:.1f}, M4={m4.bic:.1f}")

# -- 5. Per-model analysis --
print("\n" + "="*70)
print("PER-MODEL: Epidemiology → Confounding (within-model regressions)")
print("="*70)

for model_name in sorted(merged['model'].unique()):
    sub = merged[merged['model'] == model_name]
    r, p = stats.spearmanr(sub['abs_log_or'], sub['auroc_drop'])
    slope, intercept, r_val, p_val, se = stats.linregress(sub['abs_log_or'], sub['auroc_drop'])
    erank_val = sub['erank'].iloc[0]
    print(f"  {model_name:25s} erank={erank_val:6.1f}  ρ={r:.3f} (p={p:.2e})  slope={slope:.4f}")

# -- 6. Per-attribute ranking consistency across models --
print("\n" + "="*70)
print("ATTRIBUTE RANKING CONSISTENCY ACROSS MODELS")
print("="*70)

attr_model_drop = merged.groupby(['attribute', 'model'])['auroc_drop'].mean().unstack()
# Rank within each model
attr_model_rank = attr_model_drop.rank(ascending=False)
# Kendall's W (concordance)
from scipy.stats import friedmanchisquare

ranks = attr_model_rank.values
n_attrs = ranks.shape[0]
n_models = ranks.shape[1]
mean_rank = ranks.mean(axis=1)
SSB = n_models * np.sum((mean_rank - mean_rank.mean())**2)
W = 12 * SSB / (n_models**2 * (n_attrs**3 - n_attrs))
print(f"  Kendall's W = {W:.3f} (0=no agreement, 1=perfect)")
print(f"  n_attributes={n_attrs}, n_models={n_models}")

# Show top 5 per model
print("\n  Top 5 confounders per model:")
for model_name in attr_model_drop.columns:
    top5 = attr_model_drop[model_name].sort_values(ascending=False).head(5)
    attrs = ", ".join([f"{a}({v:.4f})" for a, v in top5.items()])
    print(f"    {model_name:25s}: {attrs}")

# -- 7. Predicted vs Actual scatter data --
print("\n" + "="*70)
print("MODEL FIT DIAGNOSTICS")
print("="*70)

merged['predicted_drop'] = m4.predict(merged)
residuals = merged['auroc_drop'] - merged['predicted_drop']
print(f"  Residual mean: {residuals.mean():.6f}")
print(f"  Residual std:  {residuals.std():.6f}")
print(f"  Max overpredict: {residuals.min():.6f}")
print(f"  Max underpredict: {residuals.max():.6f}")

# Largest residuals (outliers)
merged['residual'] = residuals
outliers = merged.nlargest(10, 'residual')[['attribute', 'model', 'disease', 'auroc_drop', 'predicted_drop', 'residual']]
print("\n  Top 10 underpredicted (model underestimates confounding):")
for _, row in outliers.iterrows():
    print(f"    {row['attribute']:20s} × {row['disease']:20s} ({row['model']:20s}): "
          f"actual={row['auroc_drop']:.4f}, pred={row['predicted_drop']:.4f}, resid={row['residual']:.4f}")

# -- 8. Save everything --
or_df.to_csv(f"{BASE}/results/odds_ratios_attr_disease.csv", index=False)
merged.to_csv(f"{BASE}/results/two_factor_regression_data.csv", index=False)

# Summary table
summary = pd.DataFrame({
    'model': ['M1: Epidemiology', 'M2: Geometry', 'M3: Additive', 'M4: Interaction'],
    'formula': [
        'drop ~ |log(OR)|',
        'drop ~ log(erank)',
        'drop ~ |log(OR)| + log(erank)',
        'drop ~ |log(OR)| + log(erank) + |log(OR)|×log(erank)'
    ],
    'R2': [m1.rsquared, m2.rsquared, m3.rsquared, m4.rsquared],
    'adj_R2': [m1.rsquared_adj, m2.rsquared_adj, m3.rsquared_adj, m4.rsquared_adj],
    'AIC': [m1.aic, m2.aic, m3.aic, m4.aic],
    'BIC': [m1.bic, m2.bic, m3.bic, m4.bic]
})
summary.to_csv(f"{BASE}/results/two_factor_model_comparison.csv", index=False)

print("\n" + "="*70)
print("FILES SAVED")
print("="*70)
print(f"  {BASE}/results/odds_ratios_attr_disease.csv")
print(f"  {BASE}/results/two_factor_regression_data.csv")
print(f"  {BASE}/results/two_factor_model_comparison.csv")

# -- 9. Final Verdict --
print("\n" + "="*70)
print("VERDICT")
print("="*70)

beta_or_p = m4.pvalues['abs_log_or_z']
beta_erank_p = m4.pvalues['log_erank_z']
beta_int_p = m4.pvalues['interaction_z']
beta_int_val = m4.params['interaction_z']

verdict_parts = []
if beta_or_p < 0.05:
    verdict_parts.append("Epidemiology (|log(OR)|) significantly predicts confounding cost")
else:
    verdict_parts.append("Epidemiology NOT significant")

if beta_erank_p < 0.05:
    verdict_parts.append("Geometry (log erank) significantly predicts confounding cost")
else:
    verdict_parts.append("Geometry NOT significant")

if beta_int_p < 0.05:
    direction = "BUFFERS" if beta_int_val < 0 else "AMPLIFIES"
    verdict_parts.append(f"Interaction significant: high erank {direction} epidemiological confounding")
else:
    verdict_parts.append("Interaction NOT significant")

for v in verdict_parts:
    print(f"  {v}")

print(f"\n  Full model R² = {m4.rsquared:.4f}")
print(f"  Epidemiology alone R² = {m1.rsquared:.4f}")
print(f"  Geometry adds ΔR² = {m3.rsquared - m1.rsquared:.4f}")
print(f"  Interaction adds ΔR² = {m4.rsquared - m3.rsquared:.4f}")
