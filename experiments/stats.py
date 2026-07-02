"""
Statistical utilities: Bootstrap CI, DeLong test, FDR correction, permutation tests.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats as scipy_stats

from experiments.config import N_BOOTSTRAP, BOOTSTRAP_CI, FDR_Q, N_PERMUTATIONS, SEED


# ============================================================
# Bootstrap Confidence Intervals
# ============================================================
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOTSTRAP,
                 ci=BOOTSTRAP_CI, seed=SEED):
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true, y_pred: arrays
        metric_fn: callable(y_true, y_pred) -> float
        n_boot: number of bootstrap iterations
        ci: confidence level (e.g. 0.95)

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    alpha = (1 - ci) / 2
    n = len(y_true)

    point = metric_fn(y_true, y_pred)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_vals.append(metric_fn(yt, yp))

    if len(boot_vals) < 10:
        return point, np.nan, np.nan

    return point, np.percentile(boot_vals, alpha * 100), np.percentile(boot_vals, (1 - alpha) * 100)


def bootstrap_auroc(y_true, y_pred, **kwargs):
    """Bootstrap CI specifically for AUROC."""
    return bootstrap_ci(y_true, y_pred, roc_auc_score, **kwargs)


def bootstrap_auprc(y_true, y_pred, **kwargs):
    """Bootstrap CI specifically for AUPRC."""
    return bootstrap_ci(y_true, y_pred, average_precision_score, **kwargs)


def bootstrap_delta(y_true, y_pred_a, y_pred_b, metric_fn=roc_auc_score,
                    n_boot=N_BOOTSTRAP, ci=BOOTSTRAP_CI, seed=SEED):
    """Bootstrap CI for the difference between two models' metrics.

    Returns:
        (mean_delta, ci_low, ci_high, p_value)
        where delta = metric(B) - metric(A)
    """
    rng = np.random.RandomState(seed)
    alpha = (1 - ci) / 2
    n = len(y_true)

    point_a = metric_fn(y_true, y_pred_a)
    point_b = metric_fn(y_true, y_pred_b)
    point_delta = point_b - point_a

    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        ma = metric_fn(yt, y_pred_a[idx])
        mb = metric_fn(yt, y_pred_b[idx])
        deltas.append(mb - ma)

    deltas = np.array(deltas)
    if len(deltas) < 10:
        return point_delta, np.nan, np.nan, np.nan

    ci_low = np.percentile(deltas, alpha * 100)
    ci_high = np.percentile(deltas, (1 - alpha) * 100)
    # Two-sided p-value: proportion of bootstrap deltas on the other side of 0
    if point_delta >= 0:
        p_val = (deltas < 0).mean() * 2
    else:
        p_val = (deltas > 0).mean() * 2
    p_val = min(p_val, 1.0)

    return point_delta, ci_low, ci_high, p_val


# ============================================================
# DeLong Test for Paired AUROC Comparison
# ============================================================
def _compute_midrank(x):
    """Compute midranks for DeLong test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n - 1 and z[k + 1] == z[k]:
            k += 1
        for l in range(i, k + 1):
            rank[j[l]] = 0.5 * (i + k) + 1
        i = k + 1
    return rank


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong AUC computation. Based on Sun & Xu (2014)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]
    aucs = np.zeros(k)
    tx = np.zeros(k)
    tz = np.zeros(k)

    for r in range(k):
        all_scores = np.concatenate([positive_examples[r], negative_examples[r]])
        ranks = _compute_midrank(all_scores)
        aucs[r] = (np.sum(ranks[:m]) - m * (m + 1) / 2) / (m * n)

    # Structural components
    v01 = np.zeros((k, m))
    v10 = np.zeros((k, n))
    for r in range(k):
        for i in range(m):
            v01[r, i] = (negative_examples[r] < positive_examples[r, i]).sum() / n
        for j in range(n):
            v10[r, j] = (positive_examples[r] > negative_examples[r, j]).sum() / m

    s01 = np.cov(v01) if m > 1 else np.zeros((k, k))
    s10 = np.cov(v10) if n > 1 else np.zeros((k, k))

    if isinstance(s01, np.floating):
        s01 = np.array([[s01]])
    if isinstance(s10, np.floating):
        s10 = np.array([[s10]])

    s = s01 / m + s10 / n
    return aucs, s


def delong_test(y_true, y_pred_a, y_pred_b):
    """DeLong test for comparing two AUROCs on the same data.

    Returns:
        (auroc_a, auroc_b, z_stat, p_value)
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    # Sort by true labels (positives first)
    order = np.argsort(-y_true)  # 1s first, then 0s
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())

    if label_1_count == 0 or label_1_count == len(y_true):
        return np.nan, np.nan, np.nan, np.nan

    predictions = np.vstack([y_pred_a[order], y_pred_b[order]])
    aucs, sigma = _fast_delong(predictions, label_1_count)

    diff = aucs[0] - aucs[1]
    var = sigma[0, 0] + sigma[1, 1] - 2 * sigma[0, 1]

    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0

    z = diff / np.sqrt(var)
    p = 2 * scipy_stats.norm.sf(abs(z))

    return aucs[0], aucs[1], z, p


# ============================================================
# Multiple Testing Correction
# ============================================================
def benjamini_hochberg(p_values, q=FDR_Q):
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: array of p-values
        q: FDR threshold

    Returns:
        (rejected, adjusted_p): boolean array and adjusted p-values
    """
    p = np.asarray(p_values)
    n = len(p)
    valid = ~np.isnan(p)

    adjusted = np.full(n, np.nan)
    rejected = np.full(n, False)

    if valid.sum() == 0:
        return rejected, adjusted

    # Work with valid p-values only
    p_valid = p[valid]
    n_valid = len(p_valid)
    order = np.argsort(p_valid)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n_valid + 1)

    # Adjusted p-values
    adj = p_valid * n_valid / ranks
    # Enforce monotonicity
    adj_sorted = adj[order]
    for i in range(n_valid - 2, -1, -1):
        adj_sorted[i + 1] = min(adj_sorted[i + 1], 1.0)
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj_final = np.empty(n_valid)
    adj_final[order] = adj_sorted

    adjusted[valid] = adj_final
    rejected[valid] = adj_final < q

    return rejected, adjusted


# ============================================================
# Permutation Test
# ============================================================
def permutation_auroc(y_true, y_pred, n_perm=N_PERMUTATIONS, seed=SEED):
    """Permutation test for AUROC significance.

    Returns:
        (observed_auroc, p_value, null_mean, null_std)
    """
    rng = np.random.RandomState(seed)
    observed = roc_auc_score(y_true, y_pred)

    null_aurocs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        if len(np.unique(y_perm)) < 2:
            continue
        null_aurocs.append(roc_auc_score(y_perm, y_pred))

    null_aurocs = np.array(null_aurocs)
    p_value = (null_aurocs >= observed).mean()

    return observed, p_value, null_aurocs.mean(), null_aurocs.std()


# ============================================================
# Fairness Gap Statistics
# ============================================================
def permutation_gap_test(y_true, y_pred, group_labels, metric_fn=roc_auc_score,
                         n_perm=N_PERMUTATIONS, seed=SEED):
    """Permutation test for fairness gap significance.

    Tests whether the observed AUROC gap across groups is larger than chance.

    Returns:
        (observed_gap, p_value)
    """
    rng = np.random.RandomState(seed)
    groups = np.unique(group_labels)

    def compute_gap(g_labels):
        aurocs = []
        for g in groups:
            mask = g_labels == g
            yt, yp = y_true[mask], y_pred[mask]
            if len(np.unique(yt)) < 2 or len(yt) < 20:
                return np.nan
            aurocs.append(metric_fn(yt, yp))
        return max(aurocs) - min(aurocs)

    observed_gap = compute_gap(group_labels)
    if np.isnan(observed_gap):
        return observed_gap, np.nan

    null_gaps = []
    for _ in range(n_perm):
        perm_groups = rng.permutation(group_labels)
        g = compute_gap(perm_groups)
        if not np.isnan(g):
            null_gaps.append(g)

    if len(null_gaps) < 10:
        return observed_gap, np.nan

    p_value = (np.array(null_gaps) >= observed_gap).mean()
    return observed_gap, p_value


# ============================================================
# Formatting Helpers
# ============================================================
def format_ci(point, low, high, decimals=3):
    """Format metric with CI: '0.XXX (0.XXX-0.XXX)'"""
    if np.isnan(low) or np.isnan(high):
        return f"{point:.{decimals}f}"
    return f"{point:.{decimals}f} ({low:.{decimals}f}-{high:.{decimals}f})"


def format_delta(delta, low, high, p, decimals=3):
    """Format delta with CI and p-value."""
    sig = "*" if p < 0.05 else ""
    if np.isnan(low):
        return f"{delta:+.{decimals}f}"
    return f"{delta:+.{decimals}f} ({low:+.{decimals}f}, {high:+.{decimals}f}) p={p:.3f}{sig}"
