"""Reward-shaping control — potential-based Dijkstra distance-to-goal shaping,
demonstration-free, lambda_shape swept over {0.5, 1, 2, 4, 8}.

Protocol (mirrors scripts/run_multi_seed_warm_vs_cold.py so results are
directly comparable to the stored warm/cold arms in runs/sweep_phase3_final.json):
  - 25x25 smoke config: GRID_TEMPLATE / TRAIN_CFG below are byte-identical to
    run_multi_seed_warm_vs_cold.py's.
  - Same 5 seeds x 5 scenarios (DEFAULT_SEEDS, sample_scenarios with the same
    min_euclidean_fraction=0.6), so scenario_ids line up 1:1 with
    sweep_phase3_final.json's cells for a paired test against stored warm/cold.
  - Demonstration-free: oracles=[], expert_ratio=0.0, re_seed_experts_each_iteration=False
    (identical to the cold arm's training config in _run_cell) — the only
    difference from cold is env_kwargs={"lambda_shape": lm} on the training
    environment. lambda_shape is defined in PathfindingEnv
    (src/qwarm/env/pathfinding_env.py) as the potential-based term
    lambda_shape * (d_star(v_cur) - d_star(v_next)), d_star = static reverse
    Dijkstra distance-to-goal, already unit-tested in
    tests/test_reward_shaping.py. train_gnn_dqn() previously had no path to
    reach this parameter (hardcoded env_class(..., max_steps=400) with no
    kwargs passthrough) — env_kwargs was added to unblock this script.
  - Evaluation identical to run_multi_seed_warm_vs_cold.py: evaluate_with_reasonableness
    on the SAME evaluator, same reachability semantics (strict/reasonable),
    same max_steps default (300). lambda_shape only affects the training
    reward signal, never the evaluation env, matching standard potential-based
    shaping practice (Ng et al. 1999) and this repo's own eval calls, which
    never pass lambda_shape.

Usage:
  uv run python scripts/run_shaping_control.py --train --lambda-values 0.5 1 2 4 8

Output:
  runs/shaping_control/lambda_<lm>.json   — one per arm, 25 records each,
    same schema as sweep_phase3_final.json plus "lambda_shape".
  runs/shaping_control/aggregate.json     — per-arm reach counts and exact
    binomial McNemar tests (paired on seed+scenario_id, the same test
    variant used for the paper's other McNemar claim, the cold-4x control)
    against the stored warm (24/25) and cold (13/25) arms in
    runs/sweep_phase3_final.json, plus a "provenance" block (torch version,
    device, platform, python version, git tip hash) so the artefact
    self-documents the environment it was produced in — relevant because the
    stored warm/cold reference was produced under a different environment
    (torch 2.11+cu128 GPU vs whatever this run used).
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time

import numpy as np

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.eval.metrics import evaluate_with_reasonableness
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.utils.seeding import set_global_seed

ROOT = pathlib.Path(__file__).parent.parent
OUT_DIR = ROOT / "runs" / "shaping_control"
SWEEP_25 = ROOT / "runs" / "sweep_phase3_final.json"


def _provenance() -> dict:
    """Environment fingerprint so the aggregate self-documents what produced
    it — the stored warm/cold reference was produced under torch 2.11+cu128
    GPU; this script runs under whatever's currently installed, which may
    differ (see the 2026-07-20 env-parity check referenced in the wall-clock
    estimate below)."""
    import platform
    import subprocess

    try:
        git_tip = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        git_tip = None

    try:
        import torch
        torch_version = torch.__version__
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_version = None
        device = None

    return {
        "torch_version": torch_version,
        "device": device,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_tip": git_tip,
    }

# Byte-identical to scripts/run_multi_seed_warm_vs_cold.py's smoke config,
# so shaping-arm cells are trained/evaluated under the same conditions as
# the stored warm/cold arms.
GRID_TEMPLATE = {
    "grid_width": 25,
    "grid_height": 25,
    "extra_edges": 2,
    "deactivate_prob": 0.15,
}
TRAIN_CFG = {
    "n_iterations": 5,
    "episodes_per_iteration": 100,
    "grad_steps_per_episode": 4,
    "batch_size": 64,
    "hidden_dim": 64,
}
DEFAULT_SEEDS = [42, 1337, 2024, 7, 314159]
DEFAULT_N_SCENARIOS = 5
DEFAULT_LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]


def _run_cell(seed: int, scenario, lambda_shape: float) -> dict:
    """Train a demonstration-free (oracles=[], expert_ratio=0.0) agent with
    potential-based reward shaping and evaluate it — the shaping-control arm.

    Mirrors the cold half of run_multi_seed_warm_vs_cold.py's _run_cell
    exactly, except env_kwargs threads lambda_shape into the training
    environment. Evaluation never receives lambda_shape (shaping is a
    training-time signal only).
    """
    set_global_seed(seed)

    src = scenario.source_node
    dst = scenario.destination_node
    queries = [(src, dst)]

    g = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=scenario.grid_seed,
    )
    agent = GNNDQN(node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=seed)
    buf = ExpertReplayBuffer(expert_ratio=0.0, rng=np.random.default_rng(seed))

    t0 = time.perf_counter()
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, [], queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=False,
        seed=seed,
        env_kwargs={"lambda_shape": lambda_shape},
    )
    train_s = time.perf_counter() - t0

    data = dynamic_graph_to_pyg(g, device=agent.device)
    v = evaluate_with_reasonableness(g, agent, src, dst, k_threshold=3.0, data=data)

    return {
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "source": src,
        "destination": dst,
        "lambda_shape": lambda_shape,
        "shaped_cost": v.cost,
        "shaped_strict": v.reached_goal_strict,
        "shaped_reasonable": v.reached_goal_reasonable,
        "shaped_cost_ratio": v.cost_ratio,
        "dijkstra_cost": v.dijkstra_reference_cost,
        "train_s": train_s,
    }


def run_arm(lambda_shape: float, seeds: list[int], n_scenarios: int) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        scenarios = sample_scenarios(
            GRID_TEMPLATE, n_scenarios=n_scenarios, rng=rng,
            min_euclidean_fraction=0.6,
        )
        for scenario in scenarios:
            row = _run_cell(seed, scenario, lambda_shape)
            rows.append(row)
            gc.collect()
    return rows


def _mcnemar_exact_binomial(a: list[bool], b: list[bool]) -> "float | None":
    """Exact binomial McNemar on paired booleans — same test variant used for
    the paper's cold-4x-control McNemar claim."""
    from scipy import stats as scipy_stats
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    disc = n01 + n10
    if disc == 0:
        return None
    k = min(n01, n10)
    return float(2 * scipy_stats.binom.cdf(k, disc, 0.5))


def build_aggregate(by_lambda: dict[float, list[dict]]) -> dict:
    if not SWEEP_25.exists():
        raise SystemExit(f"Missing {SWEEP_25} — cannot pair-test against stored warm/cold.")
    stored = {(r["seed"], r["scenario_id"]): r for r in json.loads(SWEEP_25.read_text())}

    per_arm = {}
    best_lm, best_reach = None, -1
    for lm, rows in sorted(by_lambda.items()):
        n = len(rows)
        n_reach = sum(1 for r in rows if r["shaped_strict"])
        shaped_bool, warm_bool, cold_bool = [], [], []
        for r in rows:
            key = (r["seed"], r["scenario_id"])
            if key not in stored:
                continue
            shaped_bool.append(bool(r["shaped_strict"]))
            warm_bool.append(bool(stored[key].get("warm_strict")))
            cold_bool.append(bool(stored[key].get("cold_strict")))

        per_arm[lm] = {
            "n_cells": n,
            "reach": n_reach,
            "reach_pct": n_reach / n if n else float("nan"),
            "mcnemar_vs_warm_p": _mcnemar_exact_binomial(shaped_bool, warm_bool),
            "mcnemar_vs_cold_p": _mcnemar_exact_binomial(shaped_bool, cold_bool),
        }
        if n_reach > best_reach:
            best_lm, best_reach = lm, n_reach

    return {
        "arms": per_arm,
        "best_arm_lambda_shape": best_lm,
        "best_arm_reach": best_reach,
        "warm_reach_reference": sum(1 for r in stored.values() if r.get("warm_strict")),
        "cold_reach_reference": sum(1 for r in stored.values() if r.get("cold_strict")),
        "n_cells_reference": len(stored),
        "provenance": _provenance(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-values", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--n-scenarios", type=int, default=DEFAULT_N_SCENARIOS)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--train", action="store_true",
                     help="Actually launch training. Without this flag, the script "
                          "only prints the wall-clock estimate and exits — this "
                          "scaffold does not launch training by default.")
    args = ap.parse_args()

    n_cells = len(args.seeds) * args.n_scenarios
    n_runs = n_cells * len(args.lambda_values)

    # Demonstration-free training (oracles=[], expert_ratio=0.0) matches the
    # cold arm's config exactly. The stored sweep's cold_train_s average
    # (162s/cell, on the original torch 2.11+cu128 GPU env) is NOT used here:
    # a same-environment env-parity check (2026-07-20, this repo's torch
    # 2.13 CPU env, 3 cells) observed cold-arm training at 36-38s/cell —
    # 4-4.5x faster, plausibly because this small (hidden_dim=64) model's
    # per-batch GPU kernel-launch overhead outweighs any GPU throughput
    # advantage, so CPU wins for a workload this small. That check is n=3,
    # not part of this repo's committed artefacts (throwaway diagnostic),
    # and does not include the shaping term's own per-step overhead
    # (lambda_shape * (d_prev - d_next), a small dict lookup + arithmetic
    # per env.step() — expected negligible, but unmeasured at production
    # scale). Treat this estimate as directional, not a committed number.
    OBSERVED_COLD_TRAIN_S = [38, 37, 36]  # 2026-07-20 env-parity check, n=3
    per_cell_s = sum(OBSERVED_COLD_TRAIN_S) / len(OBSERVED_COLD_TRAIN_S)

    print(f"Shaping control: {len(args.lambda_values)} arms x {n_cells} cells = {n_runs} runs")
    total_s = per_cell_s * n_runs
    print(f"Estimated wall-clock (serial, based on tonight's n=3 env-parity check, "
          f"NOT the stored sweep's 162s/cell historical average, mean={per_cell_s:.0f}s/cell "
          f"observed cold-arm training time in this torch 2.13 CPU env): "
          f"{total_s/3600:.1f} h")
    print("  Caveat: n=3 extrapolation; shaping-term per-step overhead not measured "
          "and not included in this estimate.")

    if not args.train:
        print("\nNOT launching training (pass --train to actually run this on the GPU machine).")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_lambda: dict[float, list[dict]] = {}
    for lm in sorted(args.lambda_values):
        print(f"\n=== lambda_shape={lm} ===")
        rows = run_arm(lm, args.seeds, args.n_scenarios)
        by_lambda[lm] = rows
        out_path = pathlib.Path(args.out_dir) / f"lambda_{lm}.json"
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"  reach: {sum(1 for r in rows if r['shaped_strict'])}/{len(rows)}  -> {out_path}")

    agg = build_aggregate(by_lambda)
    agg_path = pathlib.Path(args.out_dir) / "aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2))
    print(f"\nAggregate written to {agg_path}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
