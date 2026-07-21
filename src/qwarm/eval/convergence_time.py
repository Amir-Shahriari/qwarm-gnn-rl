"""Convergence-time metric: iterations to reach 95 % of optimal goal-reach rate.

Proposal §4.4 metric 4: warm-start should reach 95 % optimal in <= 0.5x the
iterations that cold-start requires.
"""
from __future__ import annotations


def episodes_to_reach_95pct_optimal(
    logs: dict,
    metric_key: str = "goal_reach_rate",
    threshold_fraction: float = 0.95,
) -> int:
    """Return the first iteration (1-indexed) where metric >= threshold_fraction
    of the best value seen in logs[metric_key].

    If the threshold is never reached, returns len(logs[metric_key]) + 1
    (one past the end) — meaning convergence was not achieved in budget.

    Args:
        logs: training log dict from train_gnn_dqn (keys: iteration,
              goal_reach_rate, mean_return, mean_loss, ...).
        metric_key: which logged metric to track convergence on.
              Default: 'goal_reach_rate'.
        threshold_fraction: fraction of the series maximum that counts as
              "converged". Default 0.95 = 95 % optimal.

    Returns:
        int — first iteration index (1-indexed) at which convergence was hit,
        or len(series) + 1 if never reached.
    """
    series = logs.get(metric_key, [])
    if not series:
        return 1

    best = max(series)
    if best <= 0:
        return len(series) + 1

    target = threshold_fraction * best
    for i, val in enumerate(series):
        if val >= target:
            return i + 1

    return len(series) + 1


def convergence_ratio(
    warm_logs: dict,
    cold_logs: dict,
    metric_key: str = "goal_reach_rate",
    threshold_fraction: float = 0.95,
) -> dict:
    """Compute convergence ratio warm / cold.

    A ratio <= 0.5 means warm reached 95 % optimal in at most half the
    iterations cold required (proposal §4.4 metric M4).

    Returns dict with warm_iters, cold_iters, ratio, m4_pass (ratio <= 0.5).
    """
    warm_iters = episodes_to_reach_95pct_optimal(warm_logs, metric_key, threshold_fraction)
    cold_iters = episodes_to_reach_95pct_optimal(cold_logs, metric_key, threshold_fraction)
    ratio = warm_iters / max(cold_iters, 1)
    return {
        "warm_iters_to_95pct": warm_iters,
        "cold_iters_to_95pct": cold_iters,
        "convergence_ratio": ratio,
        "m4_pass": ratio <= 0.5,
        "threshold_fraction": threshold_fraction,
        "metric_key": metric_key,
    }
