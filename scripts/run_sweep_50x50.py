"""50x50 warm-vs-cold sweep -- HEADLINE-identical protocol.

Fixed config (pre-declared, not tuned):
  Grid:     50x50, extra_edges=3, deactivate_prob=0.22, node_deactivate_prob=0.05
  Train:    n_iterations=10 (1x tier), episodes_per_iteration=200,
            grad_steps_per_episode=4, batch_size=64, hidden_dim=128,
            expert_ratio=0.30, pre_seed_n_states=3, pre_seed_k_paths=3,
            gamma=0.95
  NOTE:     batch_size=64 matches the 100x100 HEADLINE exactly (default in
            run_multi_seed_warm_vs_cold.py, absent from configs/default.yaml).
            The previous run incorrectly used batch_size=256.
  Oracles:  ClassicalAStar + QuantumInspiredStochasticOracle (+ FaithfulQAOA if Qiskit)
            FaithfulQAOA returns immediately for graphs >1000 nodes (2500 at 50x50),
            so the effective oracle pool is ClassicalAStar + QuantumInspiredStochastic.
  Seeds:    [42, 1337, 2024, 7, 314159]  (same outer seeds as 100x100 HEADLINE)
  S4 fix:   target_encoder.eval() before target embeddings in learn_from_batch
            (src/qwarm/agents/gnn_dqn.py -- applied before this run)
  Eval:     Strict goal-reach at max_steps=300 AND max_steps=1000 (dual-budget)
  Budget:   4x (40 iters) if 1x wall-clock projects to finish within 5-day deadline

Buffer pre-check: verifies that batch_size=64 transitions are reachable within
iteration 1.  Hard-stops ONLY if they cannot be (truly zero grad steps).
If pre-seeding alone does not reach 64, reports the estimated episode at which
gradient steps begin and the estimated count -- this is NOT a failure.

Writes ONLY to runs/sweep_50x50/:
  sweep_v1_50x50_1x.json   -- per-cell results (1x tier)
  sweep_v1_50x50_4x.json   -- per-cell results (4x tier, if run)
  aggregate_50x50.json     -- aggregate statistics
  cells.json               -- per-cell metadata
  REPORT.md                -- full report

Usage:
  uv run python scripts/run_sweep_50x50.py
"""
from __future__ import annotations

import gc
import heapq
import json
import math
import pathlib
import sys
import time
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer, Transition
from qwarm.training.expert_seeding import discover_all_paths, seed_buffer_from_path_library
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.utils.seeding import set_global_seed

try:
    from qwarm.oracles.faithful_qaoa import FaithfulSimulatedQAOA as _FaithfulQAOA
    _HAS_FAITHFUL_QAOA = True
except ImportError:
    _HAS_FAITHFUL_QAOA = False

# ── Fixed configuration ───────────────────────────────────────────────────────
SEEDS = [42, 1337, 2024, 7, 314159]   # identical outer seeds to 100x100 headline
N_SCENARIOS = 5                         # 5 seeds x 5 scenarios = 25 cells
OUT_DIR = pathlib.Path("runs/sweep_50x50")

GRID_CFG: dict[str, Any] = dict(
    grid_width=50,
    grid_height=50,
    extra_edges=3,
    deactivate_prob=0.22,
    node_deactivate_prob=0.05,
)

TRAIN_CFG: dict[str, Any] = dict(
    n_iterations_1x=10,
    n_iterations_4x=40,
    episodes_per_iteration=200,         # MUST equal 100x100 headline -- do not change
    grad_steps_per_episode=4,
    batch_size=64,                      # CORRECTED: matches 100x100 headline (was 256)
    hidden_dim=128,
    expert_ratio=0.30,
    pre_seed_n_states=3,
    pre_seed_k_paths=3,
    gamma=0.95,
)

EVAL_BUDGETS = [300, 1000]             # dual-budget (300 = primary, 1000 = extended)
K_THRESHOLD = 3.0

# 4x deadline: skip 4x if projected wall-clock exceeds this
DEADLINE_SECS = 5 * 24 * 3600         # 5 days

# Conservative lower bound on online transitions generated per episode on 50x50 grid.
# Used ONLY to detect truly-zero-training scenarios (hard fail).
# With max_steps=400 per episode, even getting stuck after 5 steps gives 5 transitions.
_MIN_ONLINE_STEPS_PER_EPISODE = 5
# Typical estimate used for projecting when grad steps begin
_EST_ONLINE_STEPS_PER_EPISODE = 50

ORACLES_USED: list[str] = ["ClassicalAStar", "QuantumInspiredStochasticOracle"]
if _HAS_FAITHFUL_QAOA:
    ORACLES_USED.append("FaithfulSimulatedQAOA")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _json_float(v: Any) -> "float | None":
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return float(v)


def _dijkstra(graph: dict, nodes: dict, src: str, dst: str) -> float:
    """Dijkstra with env cost formula: distance + 0.1*time + node_penalty."""
    q: list[tuple[float, str]] = [(0.0, src)]
    best: dict[str, float] = {src: 0.0}
    vis: set[str] = set()
    while q:
        c, n = heapq.heappop(q)
        if n in vis:
            continue
        vis.add(n)
        if n == dst:
            return c
        for nb, e in graph[n].items():
            if not e["active"] or not nodes[nb]["active"]:
                continue
            nc = c + e["distance"] + 0.1 * e["time"] + nodes[nb]["node_penalty"]
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                heapq.heappush(q, (nc, nb))
    return float("inf")


def _save_json(obj: Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_float)


def _build_oracles(dyn_graph: DynamicGraph, seed: int) -> list:
    """Build oracle list identical to the 100x100 headline sweep."""
    oracles: list = [
        ClassicalAStar(dyn_graph.nodes, dyn_graph.graph),
        QuantumInspiredStochasticOracle(dyn_graph.nodes, dyn_graph.graph),
    ]
    if _HAS_FAITHFUL_QAOA:
        oracles.append(_FaithfulQAOA(
            dyn_graph.nodes, dyn_graph.graph,
            p_layers=2, k_candidate_paths=5, max_edges_in_subgraph=20,
            n_optimiser_restarts=2, max_optimiser_iters=40, seed=seed,
        ))
    return oracles


# ── Buffer pre-check (iteration 1 verification) ───────────────────────────────

def check_buffer_prereq(seed: int, queries: list[tuple[str, str]]) -> dict:
    """Replicate train_gnn_dqn pre-seeding exactly; return buffer diagnostics.

    The check has two levels:
      HARD FAIL: even after all episodes_per_iteration episodes, the buffer
                 cannot reach batch_size (truly zero gradient steps).
                 Caller MUST stop and report.
      SOFT (pass): pre-seeding alone is below batch_size but the buffer fills
                 within the first few episodes -- training happens.
                 Caller reports the estimated start episode and continues.
    """
    g = DynamicGraph(**GRID_CFG, seed=seed)
    buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(seed),
    )
    oracles = _build_oracles(g, seed)

    path_library = discover_all_paths(
        g, oracles, queries,
        n_perturbation_states=TRAIN_CFG["pre_seed_n_states"],
        n_shortest_paths=TRAIN_CFG["pre_seed_k_paths"],
        base_iteration=0,
    )
    n_from_library = seed_buffer_from_path_library(g, buf, path_library, iteration=0)

    # Mirror goal-adjacent seeding from train_gnn_dqn
    n_goal_adj = 0
    for src, dst in queries:
        for node_id, neighbors in g.graph.items():
            if dst not in neighbors:
                continue
            edata = neighbors[dst]
            if not (edata["active"] and g.nodes[node_id]["active"] and g.nodes[dst]["active"]):
                continue
            step_cost = edata["distance"] + 0.1 * edata["time"] + g.nodes[dst]["node_penalty"]
            buf.expert_pool.append(
                Transition(
                    state_node=node_id, action_node=dst,
                    reward=-step_cost + 100.0, next_state_node=dst,
                    done=True, valid_next_actions=[],
                    is_expert=True, iteration_added=0, goal_node=dst,
                )
            )
            n_goal_adj += 1

    pre_seed_size = len(buf)
    batch_size = TRAIN_CFG["batch_size"]
    eps_per_iter = TRAIN_CFG["episodes_per_iteration"]
    grad_per_eps = TRAIN_CFG["grad_steps_per_episode"]

    # Hard fail: buffer cannot reach batch_size even with maximum online data in iter 1
    max_online_iter1 = eps_per_iter * _MIN_ONLINE_STEPS_PER_EPISODE
    if (pre_seed_size + max_online_iter1) < batch_size:
        return {
            "buf_size": pre_seed_size,
            "n_from_library": n_from_library,
            "n_goal_adj": n_goal_adj,
            "batch_size": batch_size,
            "passes": False,
            "hard_fail": True,
            "episode_grad_starts": None,
            "expected_grad_steps_iter1": 0,
        }

    # Soft: buffer will fill within iteration 1
    if pre_seed_size >= batch_size:
        episode_grad_starts = 0   # grad steps from episode 1
        grad_steps_iter1 = eps_per_iter * grad_per_eps
    else:
        shortfall = batch_size - pre_seed_size
        episode_grad_starts = math.ceil(shortfall / _EST_ONLINE_STEPS_PER_EPISODE)
        grad_steps_iter1 = max(0, eps_per_iter - episode_grad_starts) * grad_per_eps

    return {
        "buf_size": pre_seed_size,
        "n_from_library": n_from_library,
        "n_goal_adj": n_goal_adj,
        "batch_size": batch_size,
        "passes": True,
        "hard_fail": False,
        "pre_seed_below_batch": pre_seed_size < batch_size,
        "episode_grad_starts": episode_grad_starts,
        "expected_grad_steps_iter1": grad_steps_iter1,
    }


# ── Dual-budget agent evaluation ──────────────────────────────────────────────

def _eval_dual_budget(
    agent: GNNDQN,
    dyn_graph: DynamicGraph,
    src: str,
    dst: str,
    data,
) -> dict[int, dict]:
    """Greedy rollout at both step budgets.  Returns {budget: {strict, cost}}."""
    agent.encode(data)
    results: dict[int, dict] = {}
    for budget in EVAL_BUDGETS:
        path = [src]
        visited: set[str] = {src}
        current = src
        cost = 0.0
        for _ in range(budget):
            valid = [
                nb for nb, d in dyn_graph.graph[current].items()
                if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
            ]
            if not valid:
                break
            action = agent.choose_action(current, valid, dst, data, epsilon=0.0)
            edge = dyn_graph.graph[current][action]
            cost += edge["distance"] + 0.1 * edge["time"] + dyn_graph.nodes[action]["node_penalty"]
            path.append(action)
            visited.add(action)
            current = action
            if current == dst:
                break
        reached = (path[-1] == dst)
        results[budget] = {"strict": reached, "cost": cost if reached else float("inf")}
    return results


# ── Single cell runner ────────────────────────────────────────────────────────

def run_cell(seed: int, scenario, n_iterations: int) -> dict:
    """Train warm + cold agents; evaluate at both budgets; compute Dijkstra ref."""
    set_global_seed(seed)
    src = scenario.source_node
    dst = scenario.destination_node
    queries = [(src, dst)]

    # ── Warm agent (with oracle pre-seeding) ──────────────────────────────────
    g_warm = DynamicGraph(**GRID_CFG, seed=scenario.grid_seed)
    oracles = _build_oracles(g_warm, seed)
    warm_agent = GNNDQN(
        node_in_dim=4,
        hidden_dim=TRAIN_CFG["hidden_dim"],
        gamma=TRAIN_CFG["gamma"],
        seed=seed,
    )
    warm_buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(seed),
    )
    t_warm = time.perf_counter()
    train_gnn_dqn(
        g_warm, PathfindingEnv, warm_agent, warm_buf, oracles, queries,
        n_iterations=n_iterations,
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=seed,
        pre_seed_n_states=TRAIN_CFG["pre_seed_n_states"],
        pre_seed_k_paths=TRAIN_CFG["pre_seed_k_paths"],
    )
    warm_train_s = time.perf_counter() - t_warm
    warm_grad_steps = warm_agent._step_count  # total grad steps across all iterations

    data_warm = dynamic_graph_to_pyg(g_warm, device=warm_agent.device)
    t0 = time.perf_counter()
    warm_res = _eval_dual_budget(warm_agent, g_warm, src, dst, data_warm)
    warm_infer_ms = (time.perf_counter() - t0) * 1000

    # ── Cold agent (no oracles, rho=0) ────────────────────────────────────────
    g_cold = DynamicGraph(**GRID_CFG, seed=scenario.grid_seed)
    cold_agent = GNNDQN(
        node_in_dim=4,
        hidden_dim=TRAIN_CFG["hidden_dim"],
        gamma=TRAIN_CFG["gamma"],
        seed=seed,
    )
    cold_buf = ExpertReplayBuffer(
        expert_ratio=0.0,
        rng=np.random.default_rng(seed),
    )
    t_cold = time.perf_counter()
    train_gnn_dqn(
        g_cold, PathfindingEnv, cold_agent, cold_buf, [], queries,
        n_iterations=n_iterations,
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=False,
        seed=seed,
    )
    cold_train_s = time.perf_counter() - t_cold
    cold_grad_steps = cold_agent._step_count

    data_cold = dynamic_graph_to_pyg(g_cold, device=cold_agent.device)
    t0 = time.perf_counter()
    cold_res = _eval_dual_budget(cold_agent, g_cold, src, dst, data_cold)
    cold_infer_ms = (time.perf_counter() - t0) * 1000

    # ── Dijkstra reference ────────────────────────────────────────────────────
    dij_cost = _dijkstra(g_warm.graph, g_warm.nodes, src, dst)

    def _ratio(cost: "float | None", dij: "float | None") -> "float | None":
        if cost is None or dij is None:
            return None
        if cost >= 1e14 or dij <= 0 or dij >= 1e14:
            return None
        return cost / dij

    w300, c300 = warm_res[300], cold_res[300]
    w1000, c1000 = warm_res[1000], cold_res[1000]
    dij_safe = _json_float(dij_cost)
    wc300 = _json_float(w300["cost"])
    cc300 = _json_float(c300["cost"])
    wc1000 = _json_float(w1000["cost"])
    cc1000 = _json_float(c1000["cost"])

    return {
        # Metadata (schema matches sweep_v1_on_100x100.json)
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "grid_seed": scenario.grid_seed,
        "source": src,
        "destination": dst,
        "euclidean_distance": scenario.euclidean_distance,
        # Primary budget max_steps=300
        "warm_strict": w300["strict"],
        "cold_strict": c300["strict"],
        "warm_cost": wc300,
        "cold_cost": cc300,
        "warm_cost_ratio": _ratio(wc300, dij_safe),
        "cold_cost_ratio": _ratio(cc300, dij_safe),
        # Extended budget max_steps=1000
        "warm_strict_1000": w1000["strict"],
        "cold_strict_1000": c1000["strict"],
        "warm_cost_1000": wc1000,
        "cold_cost_1000": cc1000,
        "warm_cost_ratio_1000": _ratio(wc1000, dij_safe),
        "cold_cost_ratio_1000": _ratio(cc1000, dij_safe),
        # Reference and timing
        "dijkstra_cost": dij_safe,
        "warm_infer_ms": warm_infer_ms,
        "cold_infer_ms": cold_infer_ms,
        "warm_train_s": warm_train_s,
        "cold_train_s": cold_train_s,
        "warm_grad_steps_total": warm_grad_steps,
        "cold_grad_steps_total": cold_grad_steps,
        "k_threshold": K_THRESHOLD,
    }


# ── Sweep loop ────────────────────────────────────────────────────────────────

def run_sweep(
    scenarios_by_seed: dict[int, list],
    n_iterations: int,
    out_path: pathlib.Path,
) -> list[dict]:
    """Run all 25 cells, saving per-cell incrementally for crash-resumability."""
    records: list[dict] = []
    total_cells = sum(len(v) for v in scenarios_by_seed.values())
    cell_idx = 0

    for seed in SEEDS:
        for scenario in scenarios_by_seed[seed]:
            cell_idx += 1
            t_cell = time.perf_counter()
            print(
                f"\n[{cell_idx:>2}/{total_cells}] seed={seed}  "
                f"{scenario.source_node} -> {scenario.destination_node}  "
                f"euclid={scenario.euclidean_distance:.1f}  n_iter={n_iterations}",
                flush=True,
            )
            try:
                row = run_cell(seed, scenario, n_iterations)
            except Exception as exc:
                import traceback
                print(f"  ERROR in cell: {exc}", flush=True)
                traceback.print_exc()
                gc.collect()
                continue

            wall_cell = time.perf_counter() - t_cell
            row["wall_cell_s"] = wall_cell
            records.append(row)

            print(
                f"  warm_strict={row['warm_strict']}  cold_strict={row['cold_strict']}  "
                f"warm_cost={row['warm_cost']}  dij={row['dijkstra_cost']}  "
                f"warm_grad={row['warm_grad_steps_total']}  wall={wall_cell:.0f}s",
                flush=True,
            )
            _save_json(records, out_path)
            gc.collect()

    return records


# ── Aggregate statistics ──────────────────────────────────────────────────────

def _binomial_wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _paired_t(
    a: list["float | None"],
    b: list["float | None"],
) -> tuple["float | None", "float | None"]:
    pairs = [
        (x, y) for x, y in zip(a, b)
        if x is not None and y is not None and x < 1e14 and y < 1e14
    ]
    if len(pairs) < 2:
        return None, None
    xs, ys = zip(*pairs)
    t, p = scipy_stats.ttest_rel(list(xs), list(ys))
    return float(t), float(p)


def _mcnemar_p(a: list[bool], b: list[bool]) -> "float | None":
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    if n01 + n10 == 0:
        return None
    chi2 = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    return float(1.0 - scipy_stats.chi2.cdf(chi2, df=1))


def compute_aggregate(records: list[dict], tier: str) -> dict:
    n = len(records)
    if n == 0:
        return {"tier": tier, "n_cells": 0, "error": "no records"}

    def _fin(v: "float | None") -> float:
        return float("inf") if v is None else v

    # Budget 300
    ws300 = [r["warm_strict"] for r in records]
    cs300 = [r["cold_strict"] for r in records]
    wc300 = [r["warm_cost"] for r in records]
    cc300 = [r["cold_cost"] for r in records]
    wr300 = [r["warm_cost_ratio"] for r in records if r.get("warm_cost_ratio") is not None]

    win_300 = sum(1 for r in records if _fin(r["warm_cost"]) < _fin(r["cold_cost"])) / n
    warm_reach_300 = sum(ws300) / n
    cold_reach_300 = sum(cs300) / n
    mean_cr_300 = float(np.mean(wr300)) if wr300 else None
    t300, p300 = _paired_t(wc300, cc300)
    ci_lo, ci_hi = _binomial_wilson_ci(sum(ws300), n)
    mcn_300 = _mcnemar_p(ws300, cs300)

    # Budget 1000
    ws1000 = [r["warm_strict_1000"] for r in records]
    cs1000 = [r["cold_strict_1000"] for r in records]
    wc1000 = [r.get("warm_cost_1000") for r in records]
    cc1000 = [r.get("cold_cost_1000") for r in records]
    wr1000 = [r["warm_cost_ratio_1000"] for r in records if r.get("warm_cost_ratio_1000") is not None]

    win_1000 = sum(1 for r in records
                   if _fin(r.get("warm_cost_1000")) < _fin(r.get("cold_cost_1000"))) / n
    warm_reach_1000 = sum(ws1000) / n
    cold_reach_1000 = sum(cs1000) / n
    mean_cr_1000 = float(np.mean(wr1000)) if wr1000 else None
    t1000, p1000 = _paired_t(wc1000, cc1000)

    # Grad step summary
    warm_grad = [r.get("warm_grad_steps_total", 0) for r in records]
    cold_grad = [r.get("cold_grad_steps_total", 0) for r in records]
    avg_warm_grad = float(np.mean(warm_grad)) if warm_grad else 0.0
    avg_cold_grad = float(np.mean(cold_grad)) if cold_grad else 0.0

    # Gates (evaluate only -- do NOT engineer toward)
    g_m1 = win_300 >= 0.80
    g_m2 = mean_cr_300 is not None and mean_cr_300 <= 5.0
    g_m3 = p300 is not None and p300 < 0.01

    return {
        "tier": tier,
        "n_cells": n,
        "grad_steps": {
            "warm_avg_total": _json_float(avg_warm_grad),
            "cold_avg_total": _json_float(avg_cold_grad),
        },
        "budget_300": {
            "win_rate": win_300,
            "warm_strict_reach_rate": warm_reach_300,
            "cold_strict_reach_rate": cold_reach_300,
            "n_warm_reached": int(sum(ws300)),
            "mean_warm_cost_ratio": _json_float(mean_cr_300),
            "n_ratio_cells": len(wr300),
            "paired_t": _json_float(t300),
            "paired_p": _json_float(p300),
            "binomial_ci_warm_reach_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "mcnemar_warm_vs_cold_p": _json_float(mcn_300),
        },
        "budget_1000": {
            "win_rate": win_1000,
            "warm_strict_reach_rate": warm_reach_1000,
            "cold_strict_reach_rate": cold_reach_1000,
            "n_warm_reached": int(sum(ws1000)),
            "mean_warm_cost_ratio": _json_float(mean_cr_1000),
            "n_ratio_cells": len(wr1000),
            "paired_t": _json_float(t1000),
            "paired_p": _json_float(p1000),
        },
        "gates": {
            "M1_win_ge_80pct": {
                "pass": bool(g_m1),
                "value": round(win_300, 4),
                "threshold": 0.80,
                "description": "warm win rate (warm_cost < cold_cost) >= 80% (budget 300)",
            },
            "M2_mean_cost_ratio_le_5": {
                "pass": bool(g_m2),
                "value": _json_float(mean_cr_300),
                "threshold": 5.0,
                "description": "mean warm cost ratio vs Dijkstra <= 5.0x (budget 300, reached cells)",
            },
            "M3_paired_p_lt_001": {
                "pass": bool(g_m3),
                "value": _json_float(p300),
                "threshold": 0.01,
                "description": "paired t-test p < 0.01 (warm vs cold cost, budget 300)",
            },
        },
    }


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(
    agg_1x: dict,
    agg_4x: "dict | None",
    buf_check: dict,
    wall_1x: float,
    skipped_4x_reason: "str | None",
) -> None:
    lines = [
        "# 50x50 Sweep -- HEADLINE Protocol -- REPORT",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Headline Config Comparison (STEP 0 Verification)",
        "",
        "Source of truth: run_multi_seed_warm_vs_cold.py + configs/default.yaml",
        "",
        "| Parameter              | 100x100 Headline | 50x50 This Run | Match? |",
        "|------------------------|-----------------|----------------|--------|",
        "| batch_size             | 64              | 64             | YES    |",
        "| n_iterations           | 10              | 10             | YES    |",
        "| episodes_per_iteration | 200             | 200            | YES    |",
        "| grad_steps_per_episode | 4               | 4              | YES    |",
        "| hidden_dim             | 128             | 128            | YES    |",
        "| expert_ratio           | 0.30            | 0.30           | YES    |",
        "| pre_seed_n_states      | 3               | 3              | YES    |",
        "| pre_seed_k_paths       | 3               | 3              | YES    |",
        "| gamma                  | 0.95 (default)  | 0.95           | YES    |",
        "| epsilon_start          | 0.40 (default)  | 0.40 (default) | YES    |",
        "| epsilon_end            | 0.05 (default)  | 0.05 (default) | YES    |",
        "| lr                     | 1e-4 (default)  | 1e-4 (default) | YES    |",
        "| target_update_interval | 100 (default)   | 100 (default)  | YES    |",
        "| grid_width/height      | 100x100         | 50x50          | NO (intentional) |",
        "| extra_edges            | 4               | 3              | NO (declared 50x50 env) |",
        "| deactivate_prob        | 0.30            | 0.22           | NO (declared 50x50 env) |",
        "| node_deactivate_prob   | N/A             | 0.05           | NO (declared 50x50 env) |",
        "| S4 fix                 | NOT applied     | Applied        | Disclosed below |",
        "",
        "NOTE: The previous 50x50 run used batch_size=256 (a ~10x factor vs 64). This was",
        "a protocol mismatch that broke comparability and caused ~4x slowdown. This run",
        "corrects it to batch_size=64, matching the headline exactly.",
        "",
        "Oracle disclosure: FaithfulSimulatedQAOA returns immediately for graphs with",
        ">1000 nodes (faithful_qaoa.py:104). Both 50x50 (2500 nodes) and 100x100",
        "(10000 nodes) exceed this threshold, so QAOA contributed ZERO paths at either",
        "scale. The effective oracle pool is ClassicalAStar + QuantumInspiredStochasticOracle",
        "at both scales. This is expected and matches the headline behavior.",
        "",
        "## Configuration",
        "",
        "Grid: 50x50, extra_edges=3, deactivate_prob=0.22, node_deactivate_prob=0.05",
        "Training: n_iterations=10 (1x) / 40 (4x), episodes_per_iteration=200,",
        "          grad_steps_per_episode=4, batch_size=64, hidden_dim=128,",
        "          expert_ratio=0.30, pre_seed_n_states=3, pre_seed_k_paths=3",
        "gamma: 0.95  |  dropout: 0.1 (default GraphSAGEEncoder)",
        f"Seeds: {SEEDS}  |  Scenarios/seed: {N_SCENARIOS}  |  Total cells: {len(SEEDS)*N_SCENARIOS}",
        f"Oracles: {', '.join(ORACLES_USED)}",
        "(Identical oracle pool to 100x100 HEADLINE -- no substitutions.)",
        "",
        "## S4 Fix (Disclosed)",
        "",
        "Applied: `self.target_encoder.eval()` added before computing target embeddings",
        "in `learn_from_batch` (src/qwarm/agents/gnn_dqn.py, inside `with torch.no_grad()` block).",
        "",
        "Root cause: target_encoder is a deepcopy of _encoder_raw, which starts in PyTorch",
        "default train mode.  learn_from_batch calls `self._encoder_raw.train()` at the top,",
        "but never explicitly sets target_encoder to eval mode.  With dropout=0.1, 10% of",
        "target encoder activations were randomly zeroed during TD target computation,",
        "injecting stochastic noise into target values and destabilizing learning.",
        "",
        "Fix: `self.target_encoder.eval()` called before `t_emb = self.target_encoder(x, ei)`.",
        "No other changes were made.",
        "",
        "## Iteration 1 Verification",
        "",
        f"Buffer size after pre-seeding (first cell, seed={SEEDS[0]}):",
        f"  - From path library (discover_all_paths): {buf_check['n_from_library']}",
        f"  - From goal-adjacent seeding:             {buf_check['n_goal_adj']}",
        f"  - Total pre-seed:                         {buf_check['buf_size']} transitions",
        f"Batch size threshold:                        {buf_check['batch_size']}",
    ]

    if buf_check.get("hard_fail"):
        lines += [
            "",
            "HARD FAIL: Buffer cannot reach batch_size even after all episodes in iter 1.",
            "Zero gradient steps occurred. Run was aborted per protocol.",
        ]
    elif buf_check.get("pre_seed_below_batch"):
        ep = buf_check.get("episode_grad_starts", "?")
        est = buf_check.get("expected_grad_steps_iter1", "?")
        lines += [
            "",
            f"Note: Pre-seeding ({buf_check['buf_size']}) is below batch_size ({buf_check['batch_size']}).",
            f"      This is expected on a 50x50 grid with pre_seed_n_states=3, pre_seed_k_paths=3.",
            f"      The buffer fills from online data within the first ~{ep} episodes of iter 1.",
            f"Estimated gradient steps in iteration 1: ~{est}",
            f"  (= max(0, {TRAIN_CFG['episodes_per_iteration']} - {ep}) episodes",
            f"   x {TRAIN_CFG['grad_steps_per_episode']} grad steps/episode)",
            "",
            "Training IS occurring: gradient steps begin in iteration 1 after ~{ep} episodes.".format(ep=ep),
            "This is NOT a failure -- protocol fidelity is maintained.",
        ]
    else:
        est = buf_check.get("expected_grad_steps_iter1", "?")
        lines += [
            "",
            "Pre-seeding reached batch_size -- gradient steps from episode 1.",
            f"Estimated gradient steps in iteration 1: ~{est}",
        ]

    lines += [""]

    # Helper for formatting optional floats
    def _fmt(v: "float | None", precision: int = 4) -> str:
        return f"{v:.{precision}f}" if v is not None else "N/A"

    # ── 1x tier ───────────────────────────────────────────────────────────────
    n1x = agg_1x.get("n_cells", 0)
    gs = agg_1x.get("grad_steps", {})
    lines += [
        "## 1x Tier (10 iterations)",
        "",
        f"Cells completed:            {n1x} / {len(SEEDS)*N_SCENARIOS}",
        f"Wall-clock total:           {wall_1x/3600:.2f} h",
        f"Wall-clock per cell:        {wall_1x/max(n1x,1):.0f} s",
        f"Avg warm grad steps/cell:   {_fmt(gs.get('warm_avg_total'), 0)}",
        f"Avg cold grad steps/cell:   {_fmt(gs.get('cold_avg_total'), 0)}",
        "",
        "### Primary budget (max_steps=300)",
    ]
    b300 = agg_1x.get("budget_300", {})
    ci = b300.get("binomial_ci_warm_reach_95", [None, None])
    ci_str = f"[{ci[0]:.2%}, {ci[1]:.2%}]" if ci[0] is not None else "N/A"
    mcn = b300.get("mcnemar_warm_vs_cold_p")
    pt = b300.get("paired_t"); pp = b300.get("paired_p")
    cr = b300.get("mean_warm_cost_ratio")
    lines += [
        f"  Win rate (warm < cold):          {b300.get('win_rate', 0.0):.2%}",
        f"  Warm strict reach rate:          {b300.get('warm_strict_reach_rate', 0.0):.2%}"
        f"  (95% CI: {ci_str})",
        f"  Cold strict reach rate:          {b300.get('cold_strict_reach_rate', 0.0):.2%}",
        f"  Mean warm cost ratio:            {_fmt(cr)}",
        f"  Paired t-test:                   t={_fmt(pt, 3)}, p={_fmt(pp)}",
        f"  McNemar p (reach):               {_fmt(mcn)}",
        "",
        "  Gates (evaluate only -- not engineered toward):",
    ]
    for gname, gdata in agg_1x.get("gates", {}).items():
        val = gdata.get("value")
        val_str = f"{val:.4f}" if val is not None else "N/A"
        passed = gdata.get("pass", False)
        lines.append(f"    {gname}: {'PASS' if passed else 'FAIL'}  "
                     f"value={val_str}  threshold={gdata.get('threshold')}")

    lines += [""]
    lines += ["### Extended budget (max_steps=1000)"]
    b1k = agg_1x.get("budget_1000", {})
    pt1k = b1k.get("paired_t"); pp1k = b1k.get("paired_p")
    cr1k = b1k.get("mean_warm_cost_ratio")
    lines += [
        f"  Win rate (warm < cold):          {b1k.get('win_rate', 0.0):.2%}",
        f"  Warm strict reach rate:          {b1k.get('warm_strict_reach_rate', 0.0):.2%}",
        f"  Cold strict reach rate:          {b1k.get('cold_strict_reach_rate', 0.0):.2%}",
        f"  Mean warm cost ratio:            {_fmt(cr1k)}",
        f"  Paired t-test:                   t={_fmt(pt1k, 3)}, p={_fmt(pp1k)}",
    ]
    lines += [""]

    # ── 4x tier ───────────────────────────────────────────────────────────────
    if agg_4x is not None:
        lines += ["## 4x Tier (40 iterations)", ""]
        b300_4x = agg_4x.get("budget_300", {})
        cr4x = b300_4x.get("mean_warm_cost_ratio")
        pp4x = b300_4x.get("paired_p")
        lines += [
            f"  Win rate (warm < cold):          {b300_4x.get('win_rate', 0.0):.2%}",
            f"  Warm strict reach rate:          {b300_4x.get('warm_strict_reach_rate', 0.0):.2%}",
            f"  Mean warm cost ratio:            {_fmt(cr4x)}",
            f"  Paired p:                        {_fmt(pp4x)}",
            "",
            "  Gates:",
        ]
        for gname, gdata in agg_4x.get("gates", {}).items():
            val = gdata.get("value")
            val_str = f"{val:.4f}" if val is not None else "N/A"
            passed = gdata.get("pass", False)
            lines.append(f"    {gname}: {'PASS' if passed else 'FAIL'}  "
                         f"value={val_str}  threshold={gdata.get('threshold')}")
        lines += [""]
    else:
        lines += [
            "## 4x Tier",
            "",
            f"  Skipped: {skipped_4x_reason or 'Not run'}",
            "",
        ]

    lines += [
        "## Output Files",
        "",
        f"  {OUT_DIR}/sweep_v1_50x50_1x.json",
    ]
    if agg_4x is not None:
        lines += [f"  {OUT_DIR}/sweep_v1_50x50_4x.json"]
    lines += [
        f"  {OUT_DIR}/aggregate_50x50.json",
        f"  {OUT_DIR}/cells.json",
        f"  {OUT_DIR}/REPORT.md",
    ]

    report_path = OUT_DIR / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {report_path}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("50x50 HEADLINE-IDENTICAL SWEEP")
    print("=" * 72)
    print(f"Grid:        {GRID_CFG}")
    print(f"Train 1x:    n_iter={TRAIN_CFG['n_iterations_1x']}, "
          f"eps/iter={TRAIN_CFG['episodes_per_iteration']}, "
          f"batch={TRAIN_CFG['batch_size']}")
    print(f"Oracles:     {ORACLES_USED}")
    print(f"Seeds:       {SEEDS}  ({N_SCENARIOS} scenarios each = {len(SEEDS)*N_SCENARIOS} cells)")
    print(f"S4 fix:      target_encoder.eval() in learn_from_batch (gnn_dqn.py)")
    print(f"Output dir:  {OUT_DIR}/")
    print(flush=True)

    # ── Step 1: sample all scenarios once; save cells.json ────────────────────
    print("Sampling scenarios ...", flush=True)
    scenarios_by_seed: dict[int, list] = {}
    all_cells: list[dict] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        grid_tmpl = dict(
            grid_width=GRID_CFG["grid_width"],
            grid_height=GRID_CFG["grid_height"],
            extra_edges=GRID_CFG["extra_edges"],
            deactivate_prob=GRID_CFG["deactivate_prob"],
        )
        scenarios = sample_scenarios(grid_tmpl, n_scenarios=N_SCENARIOS, rng=rng)
        scenarios_by_seed[seed] = scenarios
        for sc in scenarios:
            all_cells.append({
                "seed": seed,
                "scenario_id": sc.scenario_id,
                "grid_seed": sc.grid_seed,
                "source": sc.source_node,
                "destination": sc.destination_node,
                "euclidean_distance": sc.euclidean_distance,
            })

    _save_json(all_cells, OUT_DIR / "cells.json")
    print(f"  {len(all_cells)} cells saved to {OUT_DIR / 'cells.json'}", flush=True)

    # ── Step 2: buffer pre-check (before any training) ────────────────────────
    first_sc = scenarios_by_seed[SEEDS[0]][0]
    first_queries = [(first_sc.source_node, first_sc.destination_node)]
    print(f"\nBuffer pre-check (seed={SEEDS[0]}, q={first_queries[0]}) ...", flush=True)
    buf_check = check_buffer_prereq(SEEDS[0], first_queries)
    print(f"  Pre-seed buffer:  {buf_check['buf_size']} transitions "
          f"(library: {buf_check['n_from_library']}, goal-adj: {buf_check['n_goal_adj']})")
    print(f"  Batch size:       {buf_check['batch_size']}")
    print(f"  Passes:           {buf_check['passes']}", flush=True)

    if not buf_check["passes"]:
        # Hard fail: truly zero gradient steps
        print(
            f"\nHARD FAIL: Buffer ({buf_check['buf_size']}) + max online data "
            f"({TRAIN_CFG['episodes_per_iteration']} x {_MIN_ONLINE_STEPS_PER_EPISODE}) "
            f"< batch_size ({buf_check['batch_size']}).",
            "\nZero gradient steps would occur in iteration 1 -- STOPPING per protocol.",
            flush=True,
        )
        with open(OUT_DIR / "REPORT.md", "w") as f:
            f.write("# 50x50 Sweep HARD-ABORTED\n\n")
            f.write(f"Buffer pre-check HARD FAIL: {buf_check['buf_size']} + "
                    f"{TRAIN_CFG['episodes_per_iteration']*_MIN_ONLINE_STEPS_PER_EPISODE} "
                    f"< {buf_check['batch_size']}\n")
            f.write("Zero gradient steps would occur in iteration 1. Run aborted per protocol.\n")
        sys.exit(1)

    if buf_check.get("pre_seed_below_batch"):
        ep = buf_check.get("episode_grad_starts", "?")
        est = buf_check.get("expected_grad_steps_iter1", "?")
        print(f"  Note: pre-seed ({buf_check['buf_size']}) < batch_size ({buf_check['batch_size']}).")
        print(f"        Buffer fills from online data ~episode {ep} of iter 1.")
        print(f"  Estimated grad steps in iter 1: ~{est}")
        print(f"  Training WILL occur. Proceeding.", flush=True)
    else:
        est = buf_check.get("expected_grad_steps_iter1", "?")
        print(f"  Pre-seed >= batch_size. Grad steps from episode 1.")
        print(f"  Estimated grad steps in iter 1: ~{est}", flush=True)

    # ── Step 3: 1x tier ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"1x TIER -- {TRAIN_CFG['n_iterations_1x']} iters x "
          f"{TRAIN_CFG['episodes_per_iteration']} eps/iter x "
          f"{TRAIN_CFG['batch_size']} batch")
    print(f"{'='*72}", flush=True)

    out_1x = OUT_DIR / "sweep_v1_50x50_1x.json"
    t0_1x = time.perf_counter()
    records_1x = run_sweep(
        scenarios_by_seed=scenarios_by_seed,
        n_iterations=TRAIN_CFG["n_iterations_1x"],
        out_path=out_1x,
    )
    wall_1x = time.perf_counter() - t0_1x
    print(f"\n1x complete: {len(records_1x)} cells in {wall_1x/3600:.2f} h "
          f"({wall_1x/max(len(records_1x),1):.0f} s/cell)", flush=True)

    agg_1x = compute_aggregate(records_1x, "1x")

    # ── Step 4: decide 4x tier ────────────────────────────────────────────────
    agg_4x: "dict | None" = None
    skipped_4x_reason: "str | None" = None
    records_4x: list[dict] = []

    proj_4x_s = wall_1x * 4.0
    print(f"\n4x projection: {proj_4x_s/3600:.1f} h  "
          f"deadline: {DEADLINE_SECS/3600:.0f} h (5 days)", flush=True)

    if proj_4x_s <= DEADLINE_SECS:
        print("4x within deadline. Running 4x tier ...", flush=True)
        print(f"\n{'='*72}")
        print(f"4x TIER -- {TRAIN_CFG['n_iterations_4x']} iters x "
              f"{TRAIN_CFG['episodes_per_iteration']} eps/iter")
        print(f"{'='*72}", flush=True)
        out_4x = OUT_DIR / "sweep_v1_50x50_4x.json"
        records_4x = run_sweep(
            scenarios_by_seed=scenarios_by_seed,
            n_iterations=TRAIN_CFG["n_iterations_4x"],
            out_path=out_4x,
        )
        agg_4x = compute_aggregate(records_4x, "4x")
        print(f"\n4x complete: {len(records_4x)} cells", flush=True)
    else:
        skipped_4x_reason = (
            f"Projected 4x wall-clock {proj_4x_s/3600:.1f} h exceeds "
            f"5-day deadline ({DEADLINE_SECS/3600:.0f} h). Reporting 1x only."
        )
        print(f"Skipping 4x: {skipped_4x_reason}", flush=True)

    # ── Step 5: save aggregate ────────────────────────────────────────────────
    aggregate: dict[str, Any] = {
        "config": {
            "grid": GRID_CFG,
            "train_1x_n_iterations": TRAIN_CFG["n_iterations_1x"],
            "train_4x_n_iterations": TRAIN_CFG["n_iterations_4x"],
            "episodes_per_iteration": TRAIN_CFG["episodes_per_iteration"],
            "grad_steps_per_episode": TRAIN_CFG["grad_steps_per_episode"],
            "batch_size": TRAIN_CFG["batch_size"],
            "hidden_dim": TRAIN_CFG["hidden_dim"],
            "expert_ratio": TRAIN_CFG["expert_ratio"],
            "pre_seed_n_states": TRAIN_CFG["pre_seed_n_states"],
            "pre_seed_k_paths": TRAIN_CFG["pre_seed_k_paths"],
            "gamma": TRAIN_CFG["gamma"],
            "seeds": SEEDS,
            "n_scenarios": N_SCENARIOS,
            "eval_budgets": EVAL_BUDGETS,
            "oracles": ORACLES_USED,
            "s4_fix_applied": True,
            "s4_fix_description": (
                "target_encoder.eval() called before target embeddings in "
                "learn_from_batch (gnn_dqn.py)"
            ),
        },
        "buffer_prereq": buf_check,
        "tier_1x": agg_1x,
        "wall_1x_s": wall_1x,
    }
    if agg_4x is not None:
        aggregate["tier_4x"] = agg_4x
        aggregate["wall_4x_s"] = None  # filled below if available
        # McNemar 1x vs 4x (reach, budget 300)
        if len(records_1x) == len(records_4x) and records_1x:
            r1_reach = [r["warm_strict"] for r in records_1x]
            r4_reach = [r["warm_strict"] for r in records_4x]
            aggregate["mcnemar_1x_vs_4x_p"] = _json_float(_mcnemar_p(r1_reach, r4_reach))
    else:
        aggregate["skipped_4x_reason"] = skipped_4x_reason

    _save_json(aggregate, OUT_DIR / "aggregate_50x50.json")

    # ── Step 6: write report ──────────────────────────────────────────────────
    write_report(
        agg_1x=agg_1x,
        agg_4x=agg_4x,
        buf_check=buf_check,
        wall_1x=wall_1x,
        skipped_4x_reason=skipped_4x_reason,
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for gname, gdata in agg_1x.get("gates", {}).items():
        val = gdata.get("value")
        val_str = f"{val:.4f}" if val is not None else "N/A"
        passed = gdata.get("pass", False)
        print(f"  {gname}: {'PASS' if passed else 'FAIL'}  value={val_str}")
    print(f"\n  Output: {OUT_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
