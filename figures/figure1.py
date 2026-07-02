"""
Regenerate Figure 1 with proper mathtext for p-value (fixes Unicode superscript box).

Inputs (relative to code_release/):
    results/per_comorbidity_residualization_summary.csv  (23 non-race attrs mean drop)
    results/race_residualization_mimic_cxr.csv           (race rows -> mean for 24th attr)
    results/phase1_probing.csv                           (encoding AUROC / R^2)
    results/race_probing_mimic_cxr.csv                   (race encoding)
    results/two_factor_regression_data_with_race.csv     (1440 obs for Fig 1B)
Output:
    Figure1_dissociation.pdf
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'pdf.fonttype': 42,   # TrueType (avoid Type 3 font box-rendering issue)
    'ps.fonttype': 42,
})

DEMO_COLOR = '#1f77b4'
COMO_COLOR = '#ff7f0e'

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'results')

# -- Panel A data ------------------------------------------------------
summ = pd.read_csv(os.path.join(RES, 'per_comorbidity_residualization_summary.csv'))
race_resid = pd.read_csv(os.path.join(RES, 'race_residualization_mimic_cxr.csv'))
race_dep_mean = race_resid['auroc_drop'].mean()  # 0.001546

probe = pd.read_csv(os.path.join(RES, 'phase1_probing.csv'))
race_probe = pd.read_csv(os.path.join(RES, 'race_probing_mimic_cxr.csv'))
race_enc_mean = race_probe['black_vs_white_auroc'].mean()  # 0.829

# Build encoding column matched to summary attribute names
attr_enc = {}
# binary attrs -> linear_auroc; continuous (age, bmi) -> linear_r2 (used as encoding metric)
for _, r in probe.iterrows():
    attr = r['target']
    val = r.get('linear_auroc') if (r.get('task') == 'classification' or pd.notna(r.get('linear_auroc'))) else r.get('linear_r2')
    if pd.notna(val):
        attr_enc.setdefault(attr, []).append(float(val))
attr_enc_mean = {k: float(np.mean(v)) for k, v in attr_enc.items()}

# Compose the 24-row table
rowsA = []
for _, r in summ.iterrows():
    attr = r['attribute']
    enc = attr_enc_mean.get(attr, np.nan)
    rowsA.append({
        'attribute': attr,
        'attribute_type': r['attribute_type'],
        'encoding': enc,
        'mean_drop': r['mean_drop'],
    })
rowsA.append({'attribute': 'race', 'attribute_type': 'demographic',
              'encoding': race_enc_mean, 'mean_drop': race_dep_mean})
A = pd.DataFrame(rowsA).dropna(subset=['encoding', 'mean_drop'])

# -- Panel B data ------------------------------------------------------
B = pd.read_csv(os.path.join(RES, 'two_factor_regression_data_with_race.csv'))

# OLS fit (M1)
x = B['abs_log_or'].values
y = B['auroc_drop'].values
slope, intercept = np.polyfit(x, y, 1)
ss_res = np.sum((y - (slope * x + intercept)) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"Panel B M1: beta={slope:.4f} intercept={intercept:.4f} R^2={r2:.3f} n={len(B)}")

# -- Plot --------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A
axA = axes[0]
for tp, color in [('demographic', DEMO_COLOR), ('comorbidity', COMO_COLOR)]:
    sub = A[A['attribute_type'] == tp]
    axA.scatter(sub['encoding'], sub['mean_drop'], s=40, c=color,
                label=tp.capitalize(), edgecolors='black', linewidths=0.5, zorder=3)

# Annotations for sex / race / heart_failure
for attr_label, attr_key, dy in [('Sex', 'sex', 0.001),
                                  ('Race (B-vs-W)', 'race', 0.001),
                                  ('Heart failure', 'heart_failure', 0.001)]:
    row = A[A['attribute'] == attr_key]
    if not row.empty:
        x0, y0 = float(row['encoding'].iloc[0]), float(row['mean_drop'].iloc[0])
        axA.annotate(attr_label, (x0, y0), xytext=(x0 + 0.02, y0 + dy),
                     fontsize=8.5, ha='left',
                     arrowprops=dict(arrowstyle='-', lw=0.4, color='gray'))

axA.set_xlabel('Encoding strength (AUROC for binary, $R^2$ for continuous)')
axA.set_ylabel('Mean attribute dependence (AUROC drop)')
axA.set_title('A. Encoding and dependence dissociate', loc='left', fontweight='bold')
axA.legend(loc='upper left', frameon=False, fontsize=9)
axA.axhline(0, color='lightgray', lw=0.5, zorder=1)
axA.grid(True, alpha=0.3, linewidth=0.4)

# Panel B
axB = axes[1]
axB.scatter(B['abs_log_or'], B['auroc_drop'], s=8, alpha=0.35,
            c='#444444', edgecolors='none', zorder=2)
xfit = np.linspace(0, B['abs_log_or'].max() * 1.02, 100)
yfit = slope * xfit + intercept
axB.plot(xfit, yfit, '-', color='#d62728', lw=1.6, zorder=4,
         label=fr'$y = {slope:.3f}\,x {intercept:+.4f}$')

# p-value: use mathtext to avoid Unicode superscript box-rendering issue
stats_label = (
    rf'$\beta = {slope:.3f}$' '\n'
    rf'$R^2 = {r2:.3f}$' '\n'
    r'$p < 10^{-15}$' '\n'
    rf'$n = {len(B)}$'
)
axB.text(0.97, 0.05, stats_label, transform=axB.transAxes,
         ha='right', va='bottom', fontsize=9.5,
         bbox=dict(facecolor='white', edgecolor='lightgray',
                   boxstyle='round,pad=0.4', lw=0.5))

axB.set_xlabel(r'$|\log(\mathrm{OR})|$')
axB.set_ylabel('AUROC drop upon residualization')
axB.set_title('B. Attribute--finding ORs covary with dependence',
              loc='left', fontweight='bold')
axB.axhline(0, color='lightgray', lw=0.5, zorder=1)
axB.grid(True, alpha=0.3, linewidth=0.4)

plt.tight_layout()
out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Figure1_dissociation.pdf')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Wrote: {out_pdf}")
