"""Demonstration-source ablation: do quantum-derived demonstrations contribute
beyond classical ones?

Four arms over the SAME 25 cells (5 seeds x 5 scenarios, 25x25 smoke config,
1x budget) as runs/sweep_phase3_final.json:

    full           — warm runs of the traced sweep (runs/sweep_phase3_traced.json);
                     NOT retrained here (warm == full oracle pool by definition)
    cold           — cold runs of the traced sweep; NOT retrained here
    classical_only — trained by this script (A* + Yen k-shortest demonstrations)
    quantum_only   — trained by this script (faithful QAOA + stochastic A*,
                     Yen disabled)

All arms share identical graph perturbation realisations per cell because
DynamicGraph carries its own RNG seeded from scenario.grid_seed.

Pipeline (crash-resumable, keyed on (seed, scenario_id, arm)):
  1. Train any missing (cell, arm) runs for the two new arms, with per-episode
     tracing into <traces-root>/seed<seed>_<scenario_id>/<arm>/ and
     demonstration-diversity metrics recorded at buffer-seeding time.
  2. Recompute full-arm seeding diversity offline (discovery only, no training)
     for cells whose warm trace predates diversity recording.
  3. Sanity check: traced-sweep warm costs vs the published
     runs/sweep_phase3_final.json per cell. Aborts the join when more cells
     differ than the documented reproduction baseline (5/25), unless --force-join.
  4. Join all four arms into runs/demo_source_ablation_results.json
     (sweep_phase3 per-cell schema, un-prefixed, plus "arm" + diversity).
  5. Aggregate into runs/demo_source_ablation_aggregate.json: per-arm reach,
     win-rate vs cold, mean/median cost ratio, pairwise Wilcoxon (+ matched-pairs
     rank-biserial, Cohen's d) on cost, exact McNemar (+ odds ratio) on reach,
     solvable cells only (the known-unreachable cell is excluded from rates),
     plus human-readable verdict lines.

Usage:
    uv run python scripts/run_demo_source_ablation.py            # full pipeline
    uv run python scripts/run_demo_source_ablation.py --no-train # join/aggregate only
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_multi_seed_warm_vs_cold import GRID_TEMPLATE, TRAIN_CFG, DEFAULT_SEEDS

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.oracles.pool import build_oracle_pool, pool_pre_seed_k_paths
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.training.expert_seeding import discover_all_paths, compute_demo_diversity
from qwarm.eval.metrics import evaluate_with_reasonableness
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.eval.statistics import paired_seed_test
from qwarm.utils.seeding import set_global_seed

NEW_ARMS = ("classical_only", "quantum_only")
ALL_ARMS = ("full", "classical_only", "quantum_only", "cold")
K_THRESHOLD = 3.0
# Documented reproduction baseline: 5/25 warm cells changed cost on a plain
# (untraced) rerun — see runs_rerun/REPRODUCTION_REPORT.md.
SANITY_BASELINE_DIFFS = 5


def _cells() -> list[tuple[int, object]]:
    """The exact (seed, scenario) grid of the published sweep."""
    cells = []
    for seed in DEFAULT_SEEDS:
        rng = np.random.default_rng(seed)
        scenarios = sample_scenarios(
            GRID_TEMPLATE, n_scenarios=5, rng=rng, min_euclidean_fraction=0.6,
        )
        cells.extend((seed, sc) for sc in scenarios)
    return cells


def _load_unreachable(audit_path: pathlib.Path) -> set[tuple[int, str]]:
    with open(audit_path) as fh:
        audit = json.load(fh)
    return {
        (rec["seed"], rec["scenario_id"])
        for rec in audit
        if rec["scale"] == "25x25" and "tier" not in rec and not rec["reachable"]
    }


def _cell_dir(traces_root: pathlib.Path, seed: int, scenario_id: str) -> pathlib.Path:
    return traces_root / f"seed{seed}_{scenario_id}"


def _train_arm_cell(seed: int, scenario, arm: str, traces_root: pathlib.Path) -> dict:
    """Train one (cell, arm) run; mirrors the warm path of the sweep's _run_cell."""
    set_global_seed(seed)
    src, dst = scenario.source_node, scenario.destination_node
    queries = [(src, dst)]

    g = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=scenario.grid_seed,
    )
    oracles = build_oracle_pool(g.nodes, g.graph, pool=arm, seed=seed)
    agent = GNNDQN(node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=seed)
    buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(seed),
    )

    trace_dir = _cell_dir(traces_root, seed, scenario.scenario_id) / arm
    t0 = time.perf_counter()
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles, queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=seed,
        pre_seed_n_states=TRAIN_CFG["pre_seed_n_states"],
        pre_seed_k_paths=pool_pre_seed_k_paths(arm, TRAIN_CFG["pre_seed_k_paths"]),
        trace_dir=trace_dir,
    )
    train_s = time.perf_counter() - t0

    # Persist weights alongside the trace so the demo can load this arm later.
    ck_path = trace_dir / "agent.pt"
    torch.save({
        "encoder_raw_state_dict": agent._encoder_raw.state_dict(),
        "q_head_state_dict": agent._q_head_raw.state_dict(),
        "hidden_dim": TRAIN_CFG["hidden_dim"],
        "node_in_dim": 4,
        "arm": arm,
        "seed": seed,
        "scenario_id": scenario.scenario_id,
    }, ck_path)

    data = dynamic_graph_to_pyg(g, device=agent.device)
    t0 = time.perf_counter()
    v = evaluate_with_reasonableness(
        g, agent, src, dst, k_threshold=K_THRESHOLD, data=data,
    )
    infer_ms = (time.perf_counter() - t0) * 1000

    diversity = None
    div_path = trace_dir / "seeding_diversity.json"
    if div_path.exists():
        with open(div_path) as fh:
            diversity = json.load(fh)

    return {
        "arm": arm,
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "source": src,
        "destination": dst,
        "euclidean_distance": scenario.euclidean_distance,
        "cost": v.cost,
        "strict": v.reached_goal_strict,
        "reasonable": v.reached_goal_reasonable,
        "cost_ratio": v.cost_ratio,
        "dijkstra_cost": v.dijkstra_reference_cost,
        "infer_ms": infer_ms,
        "train_s": train_s,
        "k_threshold": K_THRESHOLD,
        "diversity": diversity,
    }


def _train_arm_cell_isolated(grid_template: dict, train_cfg: dict,
                             traces_root: str, **kwargs) -> dict:
    """Subprocess entry point for the cell watchdog: restore config globals
    (shared by import with run_multi_seed_warm_vs_cold) in the fresh child,
    then train the (cell, arm) run."""
    GRID_TEMPLATE.update(grid_template)
    TRAIN_CFG.update(train_cfg)
    return _train_arm_cell(traces_root=pathlib.Path(traces_root), **kwargs)


def _recompute_full_diversity_isolated(grid_template: dict, train_cfg: dict,
                                       seed: int, scenario) -> dict:
    """Subprocess entry point for the offline diversity recompute."""
    GRID_TEMPLATE.update(grid_template)
    TRAIN_CFG.update(train_cfg)
    return recompute_full_diversity(seed, scenario)


def recompute_full_diversity(seed: int, scenario) -> dict:
    """Re-run ONLY the pre-seed discovery for the full pool (no training).

    Mirrors the RNG call order inside train_gnn_dqn: graph fresh from
    scenario.grid_seed, then set_global_seed(seed) immediately before
    discovery — reproducing the library the sweep's warm run seeded from.
    """
    src, dst = scenario.source_node, scenario.destination_node
    g = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=scenario.grid_seed,
    )
    oracles = build_oracle_pool(g.nodes, g.graph, pool="full", seed=seed)
    set_global_seed(seed)
    library = discover_all_paths(
        g, oracles, [(src, dst)],
        n_perturbation_states=TRAIN_CFG["pre_seed_n_states"],
        n_shortest_paths=TRAIN_CFG["pre_seed_k_paths"],
        base_iteration=0,
    )
    metrics = compute_demo_diversity(library)
    metrics["recomputed_offline"] = True
    return metrics


# ── Statistics ─────────────────────────────────────────────────────────────────

def _inf_replace(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_max = max(
        float(np.nanmax(np.where(np.isfinite(a), a, np.nan))),
        float(np.nanmax(np.where(np.isfinite(b), b, np.nan))),
        1.0,
    )
    return (
        np.where(np.isinf(a), finite_max * 10, a),
        np.where(np.isinf(b), finite_max * 10, b),
    )


def wilcoxon_rank_biserial(a: list[float], b: list[float]) -> float | None:
    """Matched-pairs rank-biserial correlation: (W+ - W-) / (W+ + W-).

    Positive values mean a > b on the paired differences.
    """
    x, y = _inf_replace(np.asarray(a, float), np.asarray(b, float))
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return None
    ranks = sps.rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    return (w_plus - w_minus) / (w_plus + w_minus)


def mcnemar_exact(reach_a: list[bool], reach_b: list[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (binomial on discordant pairs)."""
    b = sum(1 for x, y in zip(reach_a, reach_b) if x and not y)
    c = sum(1 for x, y in zip(reach_a, reach_b) if y and not x)
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        p = min(1.0, 2.0 * float(sps.binom.cdf(min(b, c), n_disc, 0.5)))
    return {
        "b_only_a_reaches": b,
        "c_only_b_reaches": c,
        "n_discordant": n_disc,
        "p_exact": p,
        "odds_ratio": (b / c) if c > 0 else (float("inf") if b > 0 else None),
        "prop_diff": (sum(map(bool, reach_a)) - sum(map(bool, reach_b))) / len(reach_a),
    }


def compare_arms(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Pairwise stats between two arms over the same ordered solvable cells."""
    cost_a = [r["cost"] if r["cost"] is not None else float("inf") for r in rows_a]
    cost_b = [r["cost"] if r["cost"] is not None else float("inf") for r in rows_b]
    base = paired_seed_test(cost_a, cost_b)  # t, Wilcoxon, Cohen's d
    return {
        "n_pairs": base["n"],
        "cost_mean_a": base["mean_a"],
        "cost_mean_b": base["mean_b"],
        "cost_wilcoxon_p": base["w_pvalue"],
        "cost_wilcoxon_stat": base["w_stat"],
        "cost_rank_biserial": wilcoxon_rank_biserial(cost_a, cost_b),
        "cost_cohens_d": base["cohens_d"],
        "cost_t_p": base["t_pvalue"],
        "mcnemar_strict": mcnemar_exact(
            [r["strict"] for r in rows_a], [r["strict"] for r in rows_b]
        ),
        "mcnemar_reasonable": mcnemar_exact(
            [r["reasonable"] for r in rows_a], [r["reasonable"] for r in rows_b]
        ),
    }


def _arm_summary(rows: list[dict], cold_rows: list[dict]) -> dict:
    costs = np.array(
        [r["cost"] if r["cost"] is not None else float("inf") for r in rows]
    )
    cold_costs = np.array(
        [r["cost"] if r["cost"] is not None else float("inf") for r in cold_rows]
    )
    ratios = np.array(
        [r["cost_ratio"] for r in rows if r["cost_ratio"] is not None
         and np.isfinite(r["cost_ratio"])]
    )
    div_rows = [r for r in rows if r.get("diversity")]
    div_mean = (
        {
            k: float(np.mean([r["diversity"][k] for r in div_rows
                              if r["diversity"].get(k) is not None]))
            for k in ("n_unique_paths", "mean_pairwise_jaccard_distance",
                      "n_unique_state_actions", "n_transitions",
                      "n_qaoa_paths", "n_quantum_paths", "n_classical_paths")
        }
        if div_rows else None
    )
    return {
        "n_cells": len(rows),
        "strict_reach_rate": float(np.mean([r["strict"] for r in rows])),
        "reasonable_reach_rate": float(np.mean([r["reasonable"] for r in rows])),
        "win_rate_vs_cold": float(np.mean(costs < cold_costs)),
        "cost_ratio_mean": float(np.mean(ratios)) if len(ratios) else None,
        "cost_ratio_median": float(np.median(ratios)) if len(ratios) else None,
        "n_finite_cost_ratio": int(len(ratios)),
        "mean_train_s": float(np.mean([r["train_s"] for r in rows
                                       if r.get("train_s") is not None])),
        "diversity_mean": div_mean,
    }


# ── Join / aggregate ───────────────────────────────────────────────────────────

def sanity_check(traced_path: pathlib.Path, published_path: pathlib.Path) -> dict:
    with open(traced_path) as fh:
        traced = {(r["seed"], r["scenario_id"]): r for r in json.load(fh)}
    with open(published_path) as fh:
        published = {(r["seed"], r["scenario_id"]): r for r in json.load(fh)}

    diffs, missing = [], []
    for key, pub in published.items():
        tr = traced.get(key)
        if tr is None:
            missing.append(list(key))
            continue
        pc = pub["warm_cost"] if pub["warm_cost"] is not None else float("inf")
        tc = tr["warm_cost"] if tr["warm_cost"] is not None else float("inf")
        if np.isinf(pc) and np.isinf(tc):
            rel = 0.0
        elif np.isinf(pc) or np.isinf(tc):
            rel = float("inf")
        else:
            rel = abs(tc - pc) / max(abs(pc), 1e-9)
        strict_flip = bool(pub["warm_strict"]) != bool(tr["warm_strict"])
        if rel > 0.01 or strict_flip:
            diffs.append({
                "seed": key[0], "scenario_id": key[1],
                "published_warm_cost": pub["warm_cost"],
                "traced_warm_cost": tr["warm_cost"],
                "rel_diff": None if np.isinf(rel) else rel,
                "strict_flip": strict_flip,
            })
    return {
        "n_compared": len(published) - len(missing),
        "n_missing_in_traced": len(missing),
        "missing": missing,
        "n_diff_cells": len(diffs),
        "diff_cells": diffs,
        "baseline_untraced_rerun_diffs": SANITY_BASELINE_DIFFS,
        "within_baseline": len(diffs) <= SANITY_BASELINE_DIFFS and not missing,
    }


def rows_from_traced_sweep(traced_path: pathlib.Path,
                           full_diversity: dict[tuple[int, str], dict]) -> list[dict]:
    with open(traced_path) as fh:
        sweep_rows = json.load(fh)
    out = []
    for r in sweep_rows:
        key = (r["seed"], r["scenario_id"])
        common = {k: r[k] for k in (
            "seed", "scenario_id", "source", "destination",
            "euclidean_distance", "dijkstra_cost", "k_threshold",
        )}
        out.append({
            "arm": "full", **common,
            "cost": r["warm_cost"], "strict": r["warm_strict"],
            "reasonable": r["warm_reasonable"], "cost_ratio": r["warm_cost_ratio"],
            "infer_ms": r["warm_infer_ms"], "train_s": r["warm_train_s"],
            "diversity": full_diversity.get(key),
        })
        out.append({
            "arm": "cold", **common,
            "cost": r["cold_cost"], "strict": r["cold_strict"],
            "reasonable": r["cold_reasonable"], "cost_ratio": r["cold_cost_ratio"],
            "infer_ms": r["cold_infer_ms"], "train_s": r["cold_train_s"],
            "diversity": None,
        })
    return out


def _json_sanitize(obj):
    """NaN -> None, +/-inf -> 'inf'/'-inf' so output files are strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return None
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
    return obj


def _fmt_p(p) -> str:
    return "nan" if p is None or not np.isfinite(p) else f"{p:.4f}"


def build_verdicts(per_arm: dict, comparisons: dict) -> list[str]:
    verdicts = []
    for a, b in (("full", "classical_only"), ("quantum_only", "classical_only"),
                 ("full", "quantum_only")):
        c = comparisons[f"{a}_vs_{b}"]
        verdicts.append(
            f"{a} vs {b}: reach {per_arm[a]['strict_reach_rate']:.1%} vs "
            f"{per_arm[b]['strict_reach_rate']:.1%} (McNemar p="
            f"{_fmt_p(c['mcnemar_strict']['p_exact'])}, OR="
            f"{c['mcnemar_strict']['odds_ratio']}), cost Wilcoxon p="
            f"{_fmt_p(c['cost_wilcoxon_p'])} (rank-biserial r="
            f"{c['cost_rank_biserial']:+.2f}, Cohen's d={c['cost_cohens_d']:+.2f})"
        )
    warm_bits = []
    for arm in ("full", "classical_only", "quantum_only"):
        c = comparisons[f"{arm}_vs_cold"]
        warm_bits.append(
            f"{arm}: reach {per_arm[arm]['strict_reach_rate']:.1%} vs "
            f"{per_arm['cold']['strict_reach_rate']:.1%}, win-rate "
            f"{per_arm[arm]['win_rate_vs_cold']:.1%}, cost Wilcoxon p="
            f"{_fmt_p(c['cost_wilcoxon_p'])} (r={c['cost_rank_biserial']:+.2f})"
        )
    verdicts.append("all warm arms vs cold — " + "; ".join(warm_bits))
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-root", type=str, default="runs/traces_25x25")
    parser.add_argument("--traced-sweep", type=str, default="runs/sweep_phase3_traced.json")
    parser.add_argument("--published", type=str, default="runs/sweep_phase3_final.json")
    parser.add_argument("--audit", type=str, default="runs/eval_reachability_audit.json")
    parser.add_argument("--checkpoint", type=str, default="runs/demo_source_ablation_partial.json")
    parser.add_argument("--out-results", type=str, default="runs/demo_source_ablation_results.json")
    parser.add_argument("--out-aggregate", type=str, default="runs/demo_source_ablation_aggregate.json")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training; join + aggregate from existing artifacts.")
    parser.add_argument("--skip-full-diversity", action="store_true",
                        help="Skip the offline full-arm diversity recompute.")
    parser.add_argument("--force-join", action="store_true",
                        help="Join even when the sanity check exceeds the baseline.")
    parser.add_argument("--cell-timeout", type=float, default=5400,
                        help="Hard per-(cell, arm) timeout in seconds; each run "
                             "executes in an isolated subprocess and a hung/crashed "
                             "run is skipped loudly. 0 disables the watchdog.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Train ONE quantum_only cell (seed=DEFAULT_SEEDS[0], scenario s1), "
                             "save its checkpoint, print wall-clock time + reach + cost ratio, "
                             "then exit.  Use this to gauge per-cell training time before "
                             "launching the full retrain.")
    parser.add_argument("--filter-seed", type=int, default=None,
                        help="Restrict training to cells with this training seed only. "
                             "Use 42 to retrain just the demo scenarios (grid seed 191664964).")
    parser.add_argument("--skip-if-checkpoint", action="store_true",
                        help="Skip any (arm, seed, scenario) triple whose traces directory "
                             "already contains a saved agent.pt checkpoint.  When set, the "
                             "JSON metrics checkpoint is ignored as the done-set; only the "
                             "filesystem agent.pt controls what gets skipped.")
    parser.add_argument("--train-only", action="store_true",
                        help="Exit after the training step (step 1).  Skips offline diversity "
                             "recompute, sanity check, join, and aggregate.  Use with "
                             "--filter-seed / --skip-if-checkpoint for targeted retrains.")
    args = parser.parse_args()

    traces_root = pathlib.Path(args.traces_root)
    checkpoint_path = pathlib.Path(args.checkpoint)
    cells = _cells()

    # ── Smoke-test mode ──────────────────────────────────────────────────────
    if args.smoke_test:
        sm_arm = "quantum_only"
        sm_seed = DEFAULT_SEEDS[0]
        sm_candidates = [(s, sc) for s, sc in cells if s == sm_seed]
        if len(sm_candidates) < 2:
            print(f"[smoke-test] ERROR: could not find s1 for seed={sm_seed}", flush=True)
            sys.exit(1)
        _, sm_sc = sm_candidates[1]  # index 1 → scenario s1
        print(f"[smoke-test] Training {sm_arm}  seed={sm_seed}  "
              f"{sm_sc.source_node} -> {sm_sc.destination_node}  "
              f"scenario_id={sm_sc.scenario_id}", flush=True)
        t0 = time.perf_counter()
        row = _train_arm_cell(sm_seed, sm_sc, sm_arm, traces_root)
        elapsed = time.perf_counter() - t0
        ck_path = _cell_dir(traces_root, sm_seed, sm_sc.scenario_id) / sm_arm / "agent.pt"
        print("", flush=True)
        print("[smoke-test] -- RESULT -----------------------------------------------", flush=True)
        print(f"[smoke-test]  wall_clock  = {elapsed:.0f}s  ({elapsed / 60:.1f} min)", flush=True)
        print(f"[smoke-test]  reached     = {row['strict']}", flush=True)
        print(f"[smoke-test]  cost        = {row['cost']:.2f}", flush=True)
        print(f"[smoke-test]  cost_ratio  = {row['cost_ratio']}", flush=True)
        print(f"[smoke-test]  checkpoint  = {ck_path}", flush=True)
        print("[smoke-test] --------------------------------------------------------", flush=True)
        sys.exit(0)
    # ── End smoke-test mode ──────────────────────────────────────────────────

    unreachable = _load_unreachable(pathlib.Path(args.audit))
    print(f"Cells: {len(cells)}  unreachable (excluded from rates): "
          f"{sorted(unreachable)}", flush=True)

    # ── 1. Train missing (cell, arm) runs ────────────────────────────────────
    rows_new: list[dict] = []
    if checkpoint_path.exists() and not args.skip_if_checkpoint:
        with open(checkpoint_path) as fh:
            rows_new = json.load(fh)
        print(f"Resuming: {len(rows_new)} (cell, arm) runs already done", flush=True)
    elif args.skip_if_checkpoint:
        print("--skip-if-checkpoint: ignoring JSON checkpoint; "
              "using agent.pt filesystem presence instead.", flush=True)
    done = {(r["seed"], r["scenario_id"], r["arm"]) for r in rows_new}

    if not args.no_train:
        todo = [
            (seed, sc, arm)
            for arm in NEW_ARMS
            for seed, sc in cells
            if (seed, sc.scenario_id, arm) not in done
        ]
        if args.filter_seed is not None:
            before = len(todo)
            todo = [(s, sc, a) for s, sc, a in todo if s == args.filter_seed]
            print(f"  --filter-seed {args.filter_seed}: {before} -> {len(todo)} cells", flush=True)
        if args.skip_if_checkpoint:
            def _has_ck(s, sc, a):
                return (_cell_dir(traces_root, s, sc.scenario_id) / a / "agent.pt").exists()
            before = len(todo)
            skipped = [(s, sc, a) for s, sc, a in todo if _has_ck(s, sc, a)]
            todo = [(s, sc, a) for s, sc, a in todo if not _has_ck(s, sc, a)]
            if skipped:
                print(f"  --skip-if-checkpoint: skipping {len(skipped)} cells with existing agent.pt:", flush=True)
                for s, sc, a in skipped:
                    print(f"    {a:<14} seed={s}  {sc.scenario_id}", flush=True)
        print(f"Training {len(todo)} remaining (cell, arm) runs", flush=True)
        for i, (seed, sc, arm) in enumerate(todo, 1):
            print(f"  [{i:>3}/{len(todo)}] {arm:<14} seed={seed} "
                  f"{sc.source_node}->{sc.destination_node}", flush=True)
            t0 = time.perf_counter()
            if args.cell_timeout:
                from cell_watchdog import run_isolated
                status, payload = run_isolated(
                    "run_demo_source_ablation", "_train_arm_cell_isolated",
                    {
                        "grid_template": GRID_TEMPLATE, "train_cfg": TRAIN_CFG,
                        "traces_root": str(traces_root),
                        "seed": seed, "scenario": sc, "arm": arm,
                    },
                    timeout_s=args.cell_timeout,
                )
                if status != "ok":
                    detail = (f"after {args.cell_timeout:.0f}s"
                              if status == "timeout" else payload)
                    print(f"          RUN {status.upper()} ({detail}) — "
                          f"SKIPPED, ablation continues", flush=True)
                    gc.collect()
                    continue
                row = payload
            else:
                try:
                    row = _train_arm_cell(seed, sc, arm, traces_root)
                except Exception as exc:
                    print(f"          ERROR: {exc}", flush=True)
                    gc.collect()
                    continue
            print(f"          cost={row['cost'] if row['cost'] is not None else float('inf'):.0f} "
                  f"strict={row['strict']}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            rows_new.append(row)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as fh:
                json.dump(rows_new, fh, indent=2)
            gc.collect()

    if args.train_only:
        print("\n--train-only: skipping diversity recompute, sanity check, join, aggregate.", flush=True)
        return

    # ── 2. Offline full-arm diversity (discovery only, no training) ──────────
    full_diversity: dict[tuple[int, str], dict] = {}
    if not args.skip_full_diversity:
        for seed, sc in cells:
            div_path = _cell_dir(traces_root, seed, sc.scenario_id) / "warm" / "seeding_diversity.json"
            if div_path.exists():
                with open(div_path) as fh:
                    full_diversity[(seed, sc.scenario_id)] = json.load(fh)
                continue
            print(f"  [full diversity] recomputing seed={seed} {sc.scenario_id}", flush=True)
            if args.cell_timeout:
                from cell_watchdog import run_isolated
                status, metrics = run_isolated(
                    "run_demo_source_ablation", "_recompute_full_diversity_isolated",
                    {"grid_template": GRID_TEMPLATE, "train_cfg": TRAIN_CFG,
                     "seed": seed, "scenario": sc},
                    timeout_s=args.cell_timeout,
                )
                if status != "ok":
                    print(f"  [full diversity] {status.upper()} for seed={seed} "
                          f"{sc.scenario_id} — SKIPPED (cell keeps diversity=None)",
                          flush=True)
                    continue
            else:
                metrics = recompute_full_diversity(seed, sc)
            div_path.parent.mkdir(parents=True, exist_ok=True)
            with open(div_path, "w") as fh:
                json.dump(metrics, fh, indent=2)
            full_diversity[(seed, sc.scenario_id)] = metrics

    # ── 3. Sanity check traced sweep vs published results ────────────────────
    sanity = sanity_check(pathlib.Path(args.traced_sweep), pathlib.Path(args.published))
    sanity_path = pathlib.Path("runs/demo_source_ablation_sanity.json")
    with open(sanity_path, "w") as fh:
        json.dump(sanity, fh, indent=2)
    print(f"\nSanity check: {sanity['n_diff_cells']}/{sanity['n_compared']} warm cells "
          f"differ >1% from published (untraced-rerun baseline: "
          f"{SANITY_BASELINE_DIFFS}); report: {sanity_path}", flush=True)
    if not sanity["within_baseline"] and not args.force_join:
        print("ABORTING join: differences exceed the documented reproduction "
              "baseline — review the sanity report (use --force-join to override).",
              flush=True)
        sys.exit(2)

    # ── 4. Join ───────────────────────────────────────────────────────────────
    all_rows = rows_from_traced_sweep(pathlib.Path(args.traced_sweep), full_diversity)
    all_rows.extend(rows_new)
    by_arm_count = {arm: sum(1 for r in all_rows if r["arm"] == arm) for arm in ALL_ARMS}
    print(f"Joined rows per arm: {by_arm_count}", flush=True)
    dump_rows = []
    for r in all_rows:
        r = dict(r)
        # Match the published sweep schema: unreachable cost / ratio stored as null
        for k in ("cost", "cost_ratio"):
            if r.get(k) is not None and np.isinf(r[k]):
                r[k] = None
        dump_rows.append(_json_sanitize(r))
    with open(args.out_results, "w") as fh:
        json.dump(dump_rows, fh, indent=2)

    # ── 5. Aggregate on solvable cells, paired by (seed, scenario_id) ────────
    solvable_keys = sorted(
        {(r["seed"], r["scenario_id"]) for r in all_rows}
        - unreachable
    )
    indexed_by_arm = {
        arm: {(r["seed"], r["scenario_id"]): r for r in all_rows if r["arm"] == arm}
        for arm in ALL_ARMS
    }
    for arm in ALL_ARMS:
        missing = [k for k in solvable_keys if k not in indexed_by_arm[arm]]
        if missing:
            print(f"WARNING: arm {arm} missing cells {missing}; they are dropped "
                  f"from all paired stats", flush=True)
            solvable_keys = [k for k in solvable_keys if k not in missing]
    arm_rows = {
        arm: [indexed_by_arm[arm][k] for k in solvable_keys] for arm in ALL_ARMS
    }

    per_arm = {arm: _arm_summary(arm_rows[arm], arm_rows["cold"]) for arm in ALL_ARMS}
    comparisons = {}
    for a, b in (("full", "classical_only"), ("quantum_only", "classical_only"),
                 ("full", "quantum_only"), ("full", "cold"),
                 ("classical_only", "cold"), ("quantum_only", "cold")):
        comparisons[f"{a}_vs_{b}"] = compare_arms(arm_rows[a], arm_rows[b])

    verdicts = build_verdicts(per_arm, comparisons)
    aggregate = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"grid": GRID_TEMPLATE, "train": TRAIN_CFG,
                   "k_threshold": K_THRESHOLD},
        "n_cells_total": len(cells),
        "n_solvable": len(solvable_keys),
        "excluded_unreachable": sorted(list(map(list, unreachable))),
        "sources": {
            "full_and_cold": str(args.traced_sweep),
            "new_arms_checkpoint": str(args.checkpoint),
            "sanity_report": str(sanity_path),
        },
        "sanity": {k: sanity[k] for k in ("n_diff_cells", "n_compared", "within_baseline")},
        "per_arm": per_arm,
        "comparisons": comparisons,
        "verdicts": verdicts,
    }
    with open(args.out_aggregate, "w") as fh:
        json.dump(_json_sanitize(aggregate), fh, indent=2)

    print("\n" + "=" * 72)
    print("  DEMONSTRATION-SOURCE ABLATION — VERDICTS (solvable cells only, "
          f"n={len(solvable_keys)})")
    print("=" * 72)
    for v in verdicts:
        print("  " + v)
    print("=" * 72)
    print(f"\n  Results:   {args.out_results}")
    print(f"  Aggregate: {args.out_aggregate}")


if __name__ == "__main__":
    main()
