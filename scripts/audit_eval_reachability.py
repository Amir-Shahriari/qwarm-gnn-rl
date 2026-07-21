"""Audit: is each sweep cell's destination Dijkstra-reachable on the FINAL
evaluation graph state?

Background: the reward ablation showed cell seed42/s1 (100x100) is unreachable
on the graph state used at evaluation, which silently caps the achievable
goal-reach rate. This script measures how widespread that is, per scale.

Eval-state reconstruction (verified against the sweep scripts):
  - run_multi_seed_warm_vs_cold.py / run_sweep_50x50.py evaluate on g_warm
    immediately after train_gnn_dqn, with no further perturbation.
  - train_gnn_dqn applies exactly n_iterations x update_graph(iteration=i+1)
    to the graph. The pre-seeding discovery (discover_all_paths) perturbs the
    graph but restores full state INCLUDING the RNG (save_state/restore_state
    carry rng_state), and nothing else consumes the graph's RNG. So:
        final state = DynamicGraph(**grid_cfg, seed=grid_seed)
                      + n_iterations x update_graph(iteration=i+1)
  - Cross-validated per cell against the dijkstra_cost RECORDED by the sweep
    (computed on the true final state with the same env cost formula):
    any mismatch is flagged loudly as a reconstruction failure.

Sweeps audited:
  100x100: runs/sweep_v1_on_100x100.json      (configs/default.yaml, 10 iters)
  50x50:   runs/sweep_50x50/sweep_v1_50x50_1x.json (run_sweep_50x50.py, 10 iters)
  25x25:   runs/sweep_phase3_final.json       (smoke defaults, 5 iters)

Output:
  runs/eval_reachability_audit.json -- per cell {scale, seed, scenario_id,
      reachable, dijkstra_cost (null if unreachable), warm_strict} plus
      validation fields {recorded_dijkstra_cost, recon_matches_recorded}.

Read-only analysis: no training, writes only the audit JSON under runs/.

Usage:
    uv run python scripts/audit_eval_reachability.py
"""
from __future__ import annotations

import heapq
import json
import math
import pathlib
from typing import Any

from qwarm.env.dynamic_graph import DynamicGraph

OUT_PATH = pathlib.Path("runs/eval_reachability_audit.json")

# Per-scale: sweep results file + the grid/training config that produced it.
SWEEPS: dict[str, dict[str, Any]] = {
    "25x25": {
        "file": "runs/sweep_phase3_final.json",
        # run_multi_seed_warm_vs_cold.py smoke defaults
        "grid": dict(grid_width=25, grid_height=25, extra_edges=2, deactivate_prob=0.15),
        "n_iterations": 5,
    },
    "50x50": {
        "file": "runs/sweep_50x50/sweep_v1_50x50_1x.json",
        # run_sweep_50x50.py GRID_CFG
        "grid": dict(grid_width=50, grid_height=50, extra_edges=3,
                     deactivate_prob=0.22, node_deactivate_prob=0.05),
        "n_iterations": 10,
    },
    "100x100": {
        "file": "runs/sweep_v1_on_100x100.json",
        # configs/default.yaml
        "grid": dict(grid_width=100, grid_height=100, extra_edges=4, deactivate_prob=0.30),
        "n_iterations": 10,
    },
}


def _grid_seed(cell: dict) -> int:
    if "grid_seed" in cell:
        return int(cell["grid_seed"])
    return int(cell["scenario_id"].split("_s")[0].replace("seed", ""))


def _dijkstra(graph: dict, nodes: dict, src: str, dst: str) -> float:
    """Env cost formula: distance + 0.1*time + node_penalty (same as the sweeps)."""
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


def _norm_recorded(v: Any) -> "float | None":
    """Recorded dijkstra_cost: None / NaN / inf all mean 'not finite'."""
    if v is None:
        return None
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def reconstruct_final_state(grid_cfg: dict, grid_seed: int, n_iterations: int) -> DynamicGraph:
    g = DynamicGraph(**grid_cfg, seed=grid_seed)
    for i in range(n_iterations):
        g.update_graph(iteration=i + 1)
    return g


def main() -> None:
    records: list[dict] = []
    contradictions: list[str] = []
    recon_failures: list[str] = []

    for scale, cfg in SWEEPS.items():
        cells = json.loads(pathlib.Path(cfg["file"]).read_text())
        print(f"\n{scale}: {len(cells)} cells from {cfg['file']} "
              f"({cfg['n_iterations']} perturbation iterations)")

        for cell in cells:
            seed = cell["seed"]
            sid = cell["scenario_id"]
            src, dst = cell["source"], cell["destination"]
            gseed = _grid_seed(cell)

            g = reconstruct_final_state(cfg["grid"], gseed, cfg["n_iterations"])
            dij = _dijkstra(g.graph, g.nodes, src, dst)
            reachable = dij < float("inf")

            recorded = _norm_recorded(cell.get("dijkstra_cost"))
            if recorded is not None and reachable:
                match = abs(dij - recorded) <= 1e-6 * max(1.0, abs(recorded))
            else:
                # both not-finite -> consistent; one finite, one not -> mismatch
                match = (recorded is None) == (not reachable)
            if not match:
                recon_failures.append(
                    f"{scale} seed={seed} {sid}: recomputed dijkstra="
                    f"{dij if reachable else 'inf'} vs recorded={recorded}"
                )

            warm_strict = bool(cell.get("warm_strict", False))
            if warm_strict and not reachable:
                contradictions.append(
                    f"{scale} seed={seed} {sid}: warm_strict=True but destination "
                    f"UNREACHABLE in reconstructed eval state"
                )

            records.append({
                "scale": scale,
                "seed": seed,
                "scenario_id": sid,
                "reachable": reachable,
                "dijkstra_cost": dij if reachable else None,
                "warm_strict": warm_strict,
                # validation extras
                "recorded_dijkstra_cost": recorded,
                "recon_matches_recorded": match,
            })

        n_un = sum(1 for r in records if r["scale"] == scale and not r["reachable"])
        print(f"  unreachable on final eval state: {n_un}/{len(cells)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {len(records)} cell audits to {OUT_PATH}")

    # ── Reconstruction validation ─────────────────────────────────────────────
    print("\n" + "=" * 74)
    if recon_failures:
        print(f"!! RECONSTRUCTION MISMATCHES ({len(recon_failures)}) -- the recipe did "
              f"NOT reproduce the recorded eval state for these cells:")
        for m in recon_failures:
            print(f"   {m}")
        print("   (Reachability verdicts for these cells are NOT trustworthy.)")
    else:
        print("Reconstruction validated: recomputed Dijkstra matches the recorded "
              "dijkstra_cost on every cell (where finite).")

    # ── Sanity check (warm_strict=True must be reachable) ────────────────────
    if contradictions:
        print(f"\n!! CONTRADICTIONS ({len(contradictions)}):")
        for m in contradictions:
            print(f"   {m}")
    else:
        print("Sanity check passed: every warm_strict=True cell is reachable.")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print(f"{'scale':<9} {'cells':>5} {'unreach':>7} {'published reach':>16} "
          f"{'reach over solvable':>20}")
    print("-" * 74)
    for scale in SWEEPS:
        rows = [r for r in records if r["scale"] == scale]
        n = len(rows)
        solvable = [r for r in rows if r["reachable"]]
        n_un = n - len(solvable)
        pub = sum(r["warm_strict"] for r in rows) / n
        adj = (sum(r["warm_strict"] for r in solvable) / len(solvable)
               if solvable else float("nan"))
        print(f"{scale:<9} {n:>5} {n_un:>7} {pub:>15.1%} "
              f"{adj:>19.1%}  ({sum(r['warm_strict'] for r in solvable)}/{len(solvable)})")
    print("=" * 74)


if __name__ == "__main__":
    main()
