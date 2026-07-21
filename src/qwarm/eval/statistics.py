"""Statistical tests for paired warm-vs-cold comparisons.

Uses paired t-test, Wilcoxon signed-rank, and Cohen's d so the thesis can
report effect size alongside p-values.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_seed_test(
    metric_a: list[float],
    metric_b: list[float],
) -> dict:
    """Compute paired t-test, Wilcoxon signed-rank, and Cohen's d for two lists.

    Inf values are replaced with the finite maximum × 10 before testing so
    failed episodes don't silently drop pairs.

    Args:
        metric_a: values for condition A (e.g., warm_cost per cell).
        metric_b: values for condition B (e.g., cold_cost per cell).

    Returns dict with:
        n, mean_a, mean_b, mean_diff (a - b), std_diff,
        t_stat, t_pvalue, w_stat, w_pvalue, cohens_d
    """
    a = np.array(metric_a, dtype=float)
    b = np.array(metric_b, dtype=float)

    # Replace inf with large finite value for statistical tests
    finite_max = np.nanmax(np.where(np.isfinite(a), a, np.nan))
    finite_max = max(finite_max, np.nanmax(np.where(np.isfinite(b), b, np.nan)), 1.0)
    a = np.where(np.isinf(a), finite_max * 10, a)
    b = np.where(np.isinf(b), finite_max * 10, b)

    diff = a - b
    n = len(diff)

    t_stat, t_pvalue = stats.ttest_rel(a, b)
    try:
        w_stat, w_pvalue = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_pvalue = float("nan"), float("nan")

    std_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    cohens_d = float(np.mean(diff) / std_diff) if std_diff > 0 else float("nan")

    return {
        "n": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a, ddof=1)) if n > 1 else float("nan"),
        "std_b": float(np.std(b, ddof=1)) if n > 1 else float("nan"),
        "mean_diff": float(np.mean(diff)),
        "std_diff": std_diff,
        "t_stat": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "w_stat": float(w_stat),
        "w_pvalue": float(w_pvalue),
        "cohens_d": cohens_d,
    }
