"""Phase 3 — Multi-seed x multi-scenario warm-vs-cold sweep.

Runs 5 seeds x 5 scenarios = 25 paired (warm, cold) comparisons.

Default config (smoke test, ~38 min on 25×25):
    uv run python scripts/run_multi_seed_warm_vs_cold.py

Headline config (4-8 h on 100×100, RTX 5080):
    uv run python scripts/run_multi_seed_warm_vs_cold.py \\
        --config configs/default.yaml \\
        --out runs/sweep_headline_v4.json

Other examples:
    uv run python scripts/run_multi_seed_warm_vs_cold.py --seeds 42 1337 2024 --n-scenarios 3
    uv run python scripts/run_multi_seed_warm_vs_cold.py --out runs/sweep.json
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time

import numpy as np
import pandas as pd

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.replay.expert_library import ExpertLibrary
from qwarm.oracles.pool import build_oracle_pool, pool_pre_seed_k_paths
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.eval.metrics import evaluate_with_reasonableness
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.eval.statistics import paired_seed_test
from qwarm.utils.seeding import set_global_seed

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
    "expert_ratio": 0.40,
    "pre_seed_n_states": 5,   # smoke default: n_iterations
    "pre_seed_k_paths": 10,   # smoke default
    "oracle_pool": "full",    # full | classical_only | quantum_only
}
DEFAULT_SEEDS = [42, 1337, 2024, 7, 314159]
DEFAULT_N_SCENARIOS = 5


def _run_cell(
    seed: int,
    scenario,
    k_threshold: float = 3.0,
    library=None,
    lambda_retr: float = 0.5,
    trace_root: pathlib.Path | None = None,
) -> dict:
    """Train warm + cold agents on separate graph instances and evaluate both.

    Each agent gets its own DynamicGraph seeded from scenario.grid_seed so both
    see the same perturbation sequence (states 0→5) and are evaluated at state 5.
    This avoids the embedding distribution mismatch that occurs when cold trains
    on the post-warm graph (states 5→10) and both are evaluated at state 10.
    """
    set_global_seed(seed)

    src = scenario.source_node
    dst = scenario.destination_node
    queries = [(src, dst)]

    warm_trace = cold_trace = None
    if trace_root is not None:
        cell_dir = trace_root / f"seed{seed}_{scenario.scenario_id}"
        warm_trace = cell_dir / "warm"
        cold_trace = cell_dir / "cold"

    # --- Warm agent: own graph instance, trains and evaluates at states 0→5 ---
    g_warm = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=scenario.grid_seed,
    )
    oracles = build_oracle_pool(
        g_warm.nodes, g_warm.graph,
        pool=TRAIN_CFG["oracle_pool"], seed=seed,
    )
    warm_agent = GNNDQN(
        node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=seed
    )
    warm_buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(seed),
    )
    print("         [warm: pre-seeding...]", flush=True)
    t_warm = time.perf_counter()
    train_gnn_dqn(
        g_warm, PathfindingEnv, warm_agent, warm_buf, oracles, queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=seed,
        pre_seed_n_states=TRAIN_CFG["pre_seed_n_states"],
        pre_seed_k_paths=pool_pre_seed_k_paths(
            TRAIN_CFG["oracle_pool"], TRAIN_CFG["pre_seed_k_paths"]
        ),
        trace_dir=warm_trace,
    )
    warm_train_s = time.perf_counter() - t_warm
    print(f"         [warm: done in {warm_train_s:.0f}s, evaluating...]", flush=True)

    data_warm = dynamic_graph_to_pyg(g_warm, device=warm_agent.device)
    t0 = time.perf_counter()
    warm_v = evaluate_with_reasonableness(
        g_warm, warm_agent, src, dst, k_threshold=k_threshold, data=data_warm,
        library=library, lambda_retr=lambda_retr,
    )
    warm_infer_ms = (time.perf_counter() - t0) * 1000

    # --- Cold agent: separate graph (same seed = same perturbation sequence) ---
    g_cold = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=scenario.grid_seed,
    )
    cold_agent = GNNDQN(
        node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=seed
    )
    cold_buf = ExpertReplayBuffer(
        expert_ratio=0.0,
        rng=np.random.default_rng(seed),
    )
    print("         [cold: training...]", flush=True)
    t_cold = time.perf_counter()
    train_gnn_dqn(
        g_cold, PathfindingEnv, cold_agent, cold_buf, [], queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=False,
        seed=seed,
        trace_dir=cold_trace,
    )
    cold_train_s = time.perf_counter() - t_cold
    print(f"         [cold: done in {cold_train_s:.0f}s]", flush=True)

    data_cold = dynamic_graph_to_pyg(g_cold, device=cold_agent.device)
    t0 = time.perf_counter()
    cold_v = evaluate_with_reasonableness(
        g_cold, cold_agent, src, dst, k_threshold=k_threshold, data=data_cold
    )
    cold_infer_ms = (time.perf_counter() - t0) * 1000

    return {
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "source": src,
        "destination": dst,
        "euclidean_distance": scenario.euclidean_distance,
        "warm_cost": warm_v.cost,
        "cold_cost": cold_v.cost,
        "warm_strict": warm_v.reached_goal_strict,
        "cold_strict": cold_v.reached_goal_strict,
        "warm_reasonable": warm_v.reached_goal_reasonable,
        "cold_reasonable": cold_v.reached_goal_reasonable,
        "warm_cost_ratio": warm_v.cost_ratio,
        "cold_cost_ratio": cold_v.cost_ratio,
        "dijkstra_cost": warm_v.dijkstra_reference_cost,
        "warm_infer_ms": warm_infer_ms,
        "cold_infer_ms": cold_infer_ms,
        "warm_train_s": warm_train_s,
        "cold_train_s": cold_train_s,
        "k_threshold": k_threshold,
    }


def _run_cell_isolated(grid_template: dict, train_cfg: dict,
                       lib_path: str | None, trace_root: str | None,
                       **cell_kwargs) -> dict:
    """Subprocess entry point: restore --config overrides (a spawn child
    re-imports this module fresh, losing parent-side global mutations),
    then run the cell."""
    GRID_TEMPLATE.update(grid_template)
    TRAIN_CFG.update(train_cfg)
    library = ExpertLibrary.load(lib_path) if lib_path else None
    return _run_cell(
        library=library,
        trace_root=pathlib.Path(trace_root) if trace_root else None,
        **cell_kwargs,
    )


def run_sweep(
    seeds: list[int],
    n_scenarios: int,
    k_threshold: float = 3.0,
    checkpoint_path: pathlib.Path | None = None,
    use_library: bool = False,
    lambda_retr: float = 0.5,
    trace_root: pathlib.Path | None = None,
    cell_timeout_s: float | None = None,
    resume_rows: list[dict] | None = None,
) -> pd.DataFrame:
    rows = list(resume_rows) if resume_rows else []
    done_keys = {(r["seed"], r["scenario_id"]) for r in rows}
    total_cells = len(seeds) * n_scenarios
    cell_idx = 0

    for seed in seeds:
        # Load per-seed expert library when --use-library is set.
        # Gracefully skips if the library file doesn't exist for this seed.
        library = None
        if use_library:
            lib_path = pathlib.Path(f"runs/libraries/seed_{seed}.pt")
            if lib_path.exists():
                library = ExpertLibrary.load(str(lib_path))
                print(f"  [V2] Loaded library for seed {seed}: {len(library)} entries "
                      f"({lib_path})", flush=True)
            else:
                print(f"  [V2] No library found for seed {seed} at {lib_path}; "
                      f"falling back to standard inference.", flush=True)

        rng = np.random.default_rng(seed)
        scenarios = sample_scenarios(
            GRID_TEMPLATE, n_scenarios=n_scenarios, rng=rng,
            min_euclidean_fraction=0.6,
        )
        for scenario in scenarios:
            cell_idx += 1
            if (seed, scenario.scenario_id) in done_keys:
                print(f"  [{cell_idx:>2}/{total_cells}] seed={seed} "
                      f"{scenario.scenario_id} — already in checkpoint, skipped",
                      flush=True)
                continue
            t0 = time.perf_counter()
            print(
                f"  [{cell_idx:>2}/{total_cells}] seed={seed} "
                f"{scenario.source_node}->{scenario.destination_node} "
                f"(dist={scenario.euclidean_distance:.1f})",
                flush=True,
            )
            if cell_timeout_s:
                from cell_watchdog import run_isolated
                status, payload = run_isolated(
                    "run_multi_seed_warm_vs_cold", "_run_cell_isolated",
                    {
                        "grid_template": GRID_TEMPLATE, "train_cfg": TRAIN_CFG,
                        "lib_path": str(lib_path) if library is not None else None,
                        "trace_root": str(trace_root) if trace_root else None,
                        "seed": seed, "scenario": scenario,
                        "k_threshold": k_threshold, "lambda_retr": lambda_retr,
                    },
                    timeout_s=cell_timeout_s,
                )
                if status != "ok":
                    detail = (f"after {cell_timeout_s:.0f}s" if status == "timeout"
                              else payload)
                    print(f"         CELL {status.upper()} ({detail}) — "
                          f"SKIPPED, sweep continues", flush=True)
                    gc.collect()
                    continue
                row = payload
            else:
                try:
                    row = _run_cell(seed, scenario, k_threshold=k_threshold,
                                    library=library, lambda_retr=lambda_retr,
                                    trace_root=trace_root)
                except Exception as exc:
                    print(f"         ERROR: {exc}", flush=True)
                    gc.collect()
                    continue
            elapsed = time.perf_counter() - t0
            wc = row["warm_cost"] if row["warm_cost"] is not None else float("inf")
            cc = row["cold_cost"] if row["cold_cost"] is not None else float("inf")
            warm_beat = wc < cc
            print(
                f"         warm={wc:.0f}  cold={cc:.0f}  "
                f"{'warm<cold' if warm_beat else 'COLD<warm'}  ({elapsed:.0f}s)",
                flush=True,
            )
            rows.append(row)
            # Flush progress so a crash doesn't lose completed cells
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w") as fh:
                    json.dump(rows, fh, indent=2, default=lambda x: None if x != x else x)
            gc.collect()

    return pd.DataFrame(rows)


def _print_aggregate(df: pd.DataFrame) -> None:
    n = len(df)
    finite = df[df["warm_cost"] < float("inf")]
    win_rate = (df["warm_cost"] < df["cold_cost"]).mean()

    warm_reasonable_rate = df["warm_reasonable"].mean()
    cold_reasonable_rate = df["cold_reasonable"].mean()

    stats_cost = paired_seed_test(df["warm_cost"].tolist(), df["cold_cost"].tolist())
    stats_infer = paired_seed_test(df["warm_infer_ms"].tolist(), df["cold_infer_ms"].tolist())

    print("\n" + "=" * 65)
    print("  MULTI-SEED x MULTI-SCENARIO SWEEP — AGGREGATE RESULTS")
    print("=" * 65)
    print(f"  Cells: {n}  ({len(df['seed'].unique())} seeds x "
          f"{n // len(df['seed'].unique())} scenarios)")
    print(f"  Training config: {GRID_TEMPLATE['grid_width']}x{GRID_TEMPLATE['grid_height']} "
          f"grid, {TRAIN_CFG['n_iterations']} iters x "
          f"{TRAIN_CFG['episodes_per_iteration']} eps, "
          f"expert_ratio={TRAIN_CFG['expert_ratio']}")
    print("-" * 65)
    print(f"  {'Metric':<40} {'Warm':>10} {'Cold':>10}  p-val")
    print(f"  {'-'*40} {'-'*10} {'-'*10}  ------")

    wc_mean = df["warm_cost"].replace(float("inf"), float("nan")).mean()
    cc_mean = df["cold_cost"].replace(float("inf"), float("nan")).mean()
    wc_std  = df["warm_cost"].replace(float("inf"), float("nan")).std()
    cc_std  = df["cold_cost"].replace(float("inf"), float("nan")).std()
    tp = stats_cost["t_pvalue"]
    print(f"  {'Composite cost (mean+/-std)':<40} "
          f"{wc_mean:>8.0f}+-{wc_std:.0f}  "
          f"{cc_mean:>8.0f}+-{cc_std:.0f}  "
          f"{tp:.4f}")

    print(f"  {'Reasonable-reach rate (k=3)':<40} "
          f"{warm_reasonable_rate:>10.2%}  "
          f"{cold_reasonable_rate:>10.2%}")

    wi_mean = df["warm_infer_ms"].mean(); wi_std = df["warm_infer_ms"].std()
    ci_mean = df["cold_infer_ms"].mean(); ci_std = df["cold_infer_ms"].std()
    ip = stats_infer["t_pvalue"]
    print(f"  {'Inference latency ms (mean+/-std)':<40} "
          f"{wi_mean:>7.1f}+-{wi_std:.1f}  "
          f"{ci_mean:>7.1f}+-{ci_std:.1f}  "
          f"{ip:.4f}")

    print(f"  {'Win rate (warm < cold)':<40} {win_rate:>10.2%}")
    print("-" * 65)

    # Gate checks
    mean_warm_ratio = finite["warm_cost_ratio"].mean() if len(finite) > 0 else float("inf")
    print(f"\n  Gates:")
    g_m1 = win_rate >= 0.80
    g_m2 = mean_warm_ratio <= 5.0
    g_m3 = tp < 0.01
    print(f"    M1  win_rate >= 0.80:             {win_rate:.2%}  {'PASS' if g_m1 else 'FAIL'}")
    print(f"    M2  mean warm/dijkstra <= 5.0:    {mean_warm_ratio:.2f}  {'PASS' if g_m2 else 'FAIL'}")
    print(f"    M3  paired t-test p < 0.01:       {tp:.4f}  {'PASS' if g_m3 else 'FAIL'}")
    print("=" * 65)


def _apply_config(config_path: str) -> None:
    """Override GRID_TEMPLATE and TRAIN_CFG from a YAML config file."""
    global GRID_TEMPLATE, TRAIN_CFG
    if not _HAS_YAML:
        raise RuntimeError("PyYAML not installed; run: uv add pyyaml")
    with open(config_path) as fh:
        cfg = _yaml.safe_load(fh)
    grid_keys = {"grid_width", "grid_height", "extra_edges", "deactivate_prob"}
    train_keys = {"n_iterations", "episodes_per_iteration", "hidden_dim", "expert_ratio",
                  "grad_steps_per_episode", "batch_size", "pre_seed_n_states", "pre_seed_k_paths",
                  "oracle_pool"}
    for k, v in cfg.items():
        if k in grid_keys:
            GRID_TEMPLATE[k] = v
        elif k in train_keys:
            TRAIN_CFG[k] = v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config to override grid/train params (e.g. configs/default.yaml)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n-scenarios", type=int, default=DEFAULT_N_SCENARIOS)
    parser.add_argument("--k-threshold", type=float, default=3.0)
    parser.add_argument("--out", type=str, default=None,
                        help="Path for JSON output (default: runs/sweep_<timestamp>.json)")
    parser.add_argument("--use-library", action="store_true",
                        help="Enable Option B memory-augmented inference for the warm agent.")
    parser.add_argument("--lambda-retr", type=float, default=0.5,
                        help="Retrieval blending weight (used only with --use-library).")
    parser.add_argument("--trace-root", type=str, default=None,
                        help="Directory for per-cell training traces "
                             "(<root>/seed<seed>_<scenario_id>/{warm,cold}/episodes.jsonl).")
    parser.add_argument("--oracle-pool", type=str, default=None,
                        choices=["full", "classical_only", "quantum_only"],
                        help="Warm-arm oracle ensemble (default: full = legacy behaviour).")
    parser.add_argument("--cell-timeout", type=float, default=5400,
                        help="Hard per-cell timeout in seconds; each cell runs in an "
                             "isolated subprocess and a hung/crashed cell is skipped "
                             "loudly. 0 disables the watchdog (legacy in-process mode).")
    parser.add_argument("--resume", action="store_true",
                        help="Load completed cells from --out (the checkpoint) and "
                             "skip them. Bit-compatible: every cell re-seeds itself.")
    args = parser.parse_args()

    if args.oracle_pool:
        TRAIN_CFG["oracle_pool"] = args.oracle_pool

    if args.config:
        _apply_config(args.config)
        print(f"  Config loaded from {args.config}")
        print(f"  Grid: {GRID_TEMPLATE}")
        print(f"  Train: {TRAIN_CFG}")

    print(f"\nPhase 3 — Multi-seed x Multi-scenario Sweep")
    print(f"Seeds: {args.seeds}  Scenarios/seed: {args.n_scenarios}")
    print(f"Total cells: {len(args.seeds) * args.n_scenarios}  "
          f"k_threshold={args.k_threshold}\n")

    out_path = pathlib.Path(args.out) if args.out else (
        pathlib.Path("runs") / f"sweep_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_library:
        print(f"  [V2] --use-library enabled (lambda_retr={args.lambda_retr})")

    resume_rows = None
    if args.resume and out_path.exists():
        with open(out_path) as fh:
            resume_rows = json.load(fh)
        print(f"  Resuming: {len(resume_rows)} cells loaded from {out_path}")

    t_start = time.perf_counter()
    df = run_sweep(
        args.seeds, args.n_scenarios,
        k_threshold=args.k_threshold,
        checkpoint_path=out_path,
        use_library=args.use_library,
        lambda_retr=args.lambda_retr,
        trace_root=pathlib.Path(args.trace_root) if args.trace_root else None,
        cell_timeout_s=args.cell_timeout or None,
        resume_rows=resume_rows,
    )
    elapsed = time.perf_counter() - t_start
    print(f"\nSweep complete in {elapsed/60:.1f} min")

    if df.empty:
        raise SystemExit(
            "\nNo cells completed (all skipped/failed) — nothing to "
            "aggregate; existing checkpoint left untouched."
        )

    _print_aggregate(df)

    df.to_json(out_path, orient="records", indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
