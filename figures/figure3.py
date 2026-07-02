"""
Regenerate Figure 3 (subgroup-gap fairness null).

This reproduces the published Figure3_fairness_null.pdf from source data. The original
generator script was not preserved; this script was reconstructed and validated
pixel-for-pixel against the published PDF. The only intended difference from the
published version is a proof correction: the three hard-coded statistics strings now
use a proper Unicode minus sign U+2212 ("-") instead of the ASCII hyphen "-"
(matching the figure's axis-tick minus signs, which already used U+2212).

Panel A: per (model x finding x dimension) cell, subgroup AUROC gap before vs after
         residualizing the top-3 confounders (heart_failure, atrial_fibrillation, age).
Panel B: histogram of the per-cell gap reduction (before - after); null result.

Style reverse-engineered from the published PDF:
  - Paul Tol 'muted' qualitative palette in KEY_DISEASES order; Support Devices = #444444
  - 30-bin histogram, bars #888888 with white edges
  - markers: Gender = circle, Age = triangle; Arial font

Input (relative to code_release/):
    results/fairness_debiasing_comparison.csv   (strategy 'top3' -> 120 cells)
Output:
    Figure3_fairness_null.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

MINUS = "−"  # U+2212 MINUS SIGN (proof correction; replaces ASCII hyphen)

mpl.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
})

KEY_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion",
                "Pneumonia", "Pneumothorax", "Support Devices"]
# Paul Tol 'muted' palette (+ dark grey for Support Devices)
TOL_MUTED = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
             '#DDCC77', '#CC6677', '#882255', '#AA4499', '#444444']
COLOR = {d: TOL_MUTED[i] for i, d in enumerate(KEY_DISEASES)}
DIM_MARKER = {'Gender': 'o', 'Age': '^'}
BARGRAY = (136 / 255, 136 / 255, 136 / 255)  # #888888

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'results')

# -- Data ---------------------------------------------------------------
comp = pd.read_csv(os.path.join(RES, 'fairness_debiasing_comparison.csv'))
d = comp[comp['strategy'] == 'top3'].dropna(subset=['gap_reduction']).copy()
red = d['gap_reduction'].values
n = len(d)
mean_red = red.mean()
improved = int((red > 0).sum())
print(f"Panel B: n={n} cells, mean reduction={mean_red:+.4f}, improved={improved}/{n}")

# -- Plot ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A
axA = axes[0]
for dim, mk in DIM_MARKER.items():
    sub = d[d['dimension'] == dim]
    for dis in KEY_DISEASES:
        s2 = sub[sub['disease'] == dis]
        if len(s2):
            axA.scatter(s2['gap_baseline'], s2['auroc_gap'], s=30, marker=mk,
                        c=COLOR[dis], edgecolors='black', linewidths=0.4, zorder=3)
lim = max(d['gap_baseline'].max(), d['auroc_gap'].max()) * 1.05
axA.plot([0, lim], [0, lim], ls='--', color='0.55', lw=1.0, zorder=1)
axA.set_xlim(-lim * 0.03, lim)
axA.set_ylim(-lim * 0.03, lim)
axA.set_xlabel('Subgroup AUROC gap before residualization', fontsize=11)
axA.set_ylabel('Subgroup AUROC gap after residualization', fontsize=11)
axA.set_title('A. Gap change at each model × finding × dimension cell',
              loc='left', fontweight='bold', fontsize=11)
axA.tick_params(labelsize=10)

finding_handles = [Line2D([0], [0], marker='o', ls='', mfc=COLOR[d_], mec='black',
                          mew=0.5, ms=7, label=d_) for d_ in KEY_DISEASES]
leg1 = axA.legend(handles=finding_handles, title='Finding', loc='upper left', ncol=2,
                  fontsize=7, title_fontsize=10, frameon=False,
                  handletextpad=0.3, columnspacing=0.8, labelspacing=0.3)
axA.add_artist(leg1)
dim_handles = [Line2D([0], [0], marker=DIM_MARKER[k], ls='', mfc='0.35', mec='black',
                      mew=0.4, ms=7, label=k) for k in ['Gender', 'Age']]
axA.legend(handles=dim_handles, title='Dimension', loc='lower right', fontsize=8,
           title_fontsize=10, frameon=False)

# Panel B
axB = axes[1]
axB.hist(red, bins=30, color=BARGRAY, edgecolor='white', linewidth=1.0, zorder=2)
axB.axvline(mean_red, color='red', lw=1.5, zorder=4, label=f'Mean = {MINUS}0.0003')
axB.axvline(0, color='black', ls='--', lw=1.0, zorder=3)
axB.set_xlabel(f'Subgroup AUROC gap reduction (before {MINUS} after)\n'
               '(positive = fairness gap shrank)', fontsize=11)
axB.set_ylabel('Model × finding × dimension cells', fontsize=11)
axB.set_title(f'B. Residualization produces no systematic gap reduction '
              f'({improved}/{n} improved)', loc='left', fontweight='bold', fontsize=11)
axB.tick_params(labelsize=10)
axB.legend(loc='upper right', fontsize=9, frameon=False)

stats_label = (f'Mean reduction = {MINUS}0.0003\n'
               f'95% CI [{MINUS}0.0016, +0.0010]\n'
               'n = 120 cells\n'
               'MDES (80% power) = 0.0019')
axB.text(0.03, 0.97, stats_label, transform=axB.transAxes, ha='left', va='top',
         fontsize=9, bbox=dict(facecolor='white', edgecolor='0.7',
                               boxstyle='round,pad=0.4', lw=0.8))

plt.tight_layout()
out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Figure3_fairness_null.pdf')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Wrote: {out_pdf}")
