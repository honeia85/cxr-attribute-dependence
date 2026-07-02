"""
Biostatistical diagnostics for the primary OR -> dependence regression (M1).

Standard regression diagnostics:
  1) Residual normality (Shapiro / skew / kurtosis)
  2) Heteroscedasticity (Breusch-Pagan)
  3) Leverage / outliers (Cook's distance)
  4) Intraclass correlation for the mixed-effects specification
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import os
# Prefer the 24-attribute (race-inclusive) file that matches the manuscript
# primary regression (n = 1,440); fall back to the 23-attribute legacy file
# if the race-inclusive CSV has not been generated yet.
_primary = "results/two_factor_regression_data_with_race.csv"
_legacy = "results/two_factor_regression_data.csv"
_path = _primary if os.path.exists(_primary) else _legacy
df = pd.read_csv(_path)
print(f"Input: {_path}")
print(f"n = {len(df)}")

# ----- M1 baseline: drop ~ |log(OR)| -----
X = sm.add_constant(df["abs_log_or"].values)
y = df["auroc_drop"].values
m1 = sm.OLS(y, X).fit()
print(f"\nM1: beta={m1.params[1]:.4f}, R^2={m1.rsquared:.3f}, p={m1.pvalues[1]:.2e}")

# ----- 1. Residual distribution -----
resid = m1.resid
print(f"\nResidual distribution:")
print(f"  mean = {resid.mean():.5f} (expected ~0)")
print(f"  sd   = {resid.std():.5f}")
print(f"  skew = {stats.skew(resid):.3f}  (|>1| = problematic)")
print(f"  kurt = {stats.kurtosis(resid):.3f} (|>3| = heavy tails)")
# Shapiro-Wilk on the full residual vector (scipy accepts up to 5000 points).
print(f"  Shapiro-Wilk W = {stats.shapiro(resid).statistic:.4f}, "
      f"p = {stats.shapiro(resid).pvalue:.2e}")
print(f"  Jarque-Bera stat = {stats.jarque_bera(resid).statistic:.2f}, "
      f"p = {stats.jarque_bera(resid).pvalue:.2e}")

# ----- 2. Heteroscedasticity -----
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
print(f"\nBreusch-Pagan: LM = {bp_stat:.2f}, p = {bp_p:.2e}  "
      f"(small p => heteroscedastic)")

# ----- 3. Leverage / Cook's distance -----
infl = m1.get_influence()
cooks_d = infl.cooks_distance[0]
leverage = infl.hat_matrix_diag
n, k = len(df), 2
print(f"\nCook's distance (n={n}, k={k}):")
print(f"  max     = {cooks_d.max():.4f}")
print(f"  > 4/n  ({4/n:.4f}) count = {(cooks_d > 4/n).sum()}")
print(f"  > 1 (strict) count = {(cooks_d > 1).sum()}")
high_cook_idx = np.argsort(cooks_d)[-5:][::-1]
print("  Top 5 high-Cook observations:")
for i in high_cook_idx:
    print(f"    [{i:4d}] {df.iloc[i]['attribute']:22s} / {df.iloc[i]['model']:20s} / "
          f"{df.iloc[i]['disease']:25s}  cook_d = {cooks_d[i]:.4f}, "
          f"drop = {df.iloc[i]['auroc_drop']:+.4f}")

# ----- 4. ICC for random-finding mixed-effects model -----
md = smf.mixedlm("auroc_drop ~ abs_log_or", df, groups=df["disease"])
mdf = md.fit(method=["lbfgs"])
sigma_u2 = float(mdf.cov_re.iloc[0, 0])
sigma_e2 = float(mdf.scale)
icc = sigma_u2 / (sigma_u2 + sigma_e2)
print(f"\nMixed-effects (random finding intercept):")
print(f"  between-finding variance  = {sigma_u2:.3e}")
print(f"  residual variance         = {sigma_e2:.3e}")
print(f"  ICC                       = {icc:.3f}  "
      f"(fraction of variance attributable to finding)")

# ----- 5. Bootstrap sanity: 95% percentile CI on beta -----
rng = np.random.default_rng(42)
B = 2000
betas = np.empty(B)
for b in range(B):
    idx = rng.integers(0, len(df), len(df))
    Xb = sm.add_constant(df["abs_log_or"].values[idx])
    yb = df["auroc_drop"].values[idx]
    betas[b] = sm.OLS(yb, Xb).fit().params[1]
lo, hi = np.quantile(betas, [0.025, 0.975])
print(f"\nBootstrap percentile 95% CI for beta (M1, B={B}): "
      f"[{lo:.4f}, {hi:.4f}]  (point estimate {m1.params[1]:.4f})")

# ----- 6. Robust regression sensitivity (Huber M-estimator) -----
from statsmodels.robust.robust_linear_model import RLM
rlm = RLM(y, X, M=sm.robust.norms.HuberT()).fit()
print(f"\nHuber M-estimator (robust to heavy tails):")
print(f"  beta = {rlm.params[1]:.4f}  vs OLS {m1.params[1]:.4f}")
print(f"  robust SE = {rlm.bse[1]:.5f}")
print(f"  z-value   = {rlm.tvalues[1]:.2f}")

# ----- 7. Save summary CSV for inclusion in Supp Methods -----
summary = pd.DataFrame({
    "check": ["OLS beta (M1)",
              "Huber M-estimator beta",
              "Bootstrap 95% CI (percentile)",
              "Residual skewness",
              "Residual kurtosis",
              "Shapiro-Wilk p",
              "Breusch-Pagan p",
              "Cook's D > 4/n (count)",
              "Mixed-effects ICC"],
    "value": [f"{m1.params[1]:.4f}",
              f"{rlm.params[1]:.4f}",
              f"[{lo:.4f}, {hi:.4f}]",
              f"{stats.skew(resid):.3f}",
              f"{stats.kurtosis(resid):.3f}",
              f"{stats.shapiro(resid).pvalue:.2e}",
              f"{bp_p:.2e}",
              f"{int((cooks_d > 4/n).sum())}/{n}",
              f"{icc:.3f}"]
})
summary.to_csv("results/biostat_diagnostics.csv", index=False)
print("\nSaved: results/biostat_diagnostics.csv")
print(summary.to_string(index=False))
