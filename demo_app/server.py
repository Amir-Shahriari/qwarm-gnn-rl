"""FastAPI backend for the ICDM 2026 GNN-DQN warm-start pathfinding demo.

Endpoints:
  GET  /                     → serve demo_app/static/index.html
  GET  /cells                → list of cell summary dicts
  POST /reach                → body: {"cell_id": "25x25_s1"} → full rollout result
  GET  /sandbox/scenarios    → list of sandbox scenarios with training pair indices
  GET  /sandbox/grid         → query: ?cell_id=25x25_s1 → grid wire format + train src/dst
  POST /sandbox              → body: {cell_id, blocked, src_idx, dst_idx} → race result

Run with: uv run python -m demo_app.server
"""
from __future__ import annotations

import heapq
import json
import math
import pathlib
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.utils.device import resolve_device

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).parent.parent  # repo root
_MANIFEST_PATH = _ROOT / "demo_agents" / "manifest.json"
_DEMO_AGENTS_DIR = _ROOT / "demo_agents"
_STATIC_DIR = pathlib.Path(__file__).parent / "static"
_TRACES_ROOT = _ROOT / "runs" / "traces_25x25"
_ORACLE_SEED = 42

# ---------------------------------------------------------------------------
# In-memory cache populated at startup
# ---------------------------------------------------------------------------

# cell_id → precomputed wire-format dict (the heavy /reach response)
_CELL_CACHE: dict[str, dict] = {}
# cell_id → lightweight summary dict (for /cells)
_CELL_SUMMARIES: list[dict] = []

# Sandbox: all evaluated scenarios kept alive for interactive race.
# cell_id → {warm_agent, cold_agent, graph, node_ids, idx_map,
#             train_src, train_dst, train_src_idx, train_dst_idx, scale}
_SANDBOX_CELLS: dict[str, dict] = {}

# Oracle-source mode: per (cell_id, arm) precomputed wire formats.
# Only populated for 25×25 solvable scenarios.
# key: "25x25_s1|classical_only", etc.
_ORACLE_CACHE: dict[str, dict] = {}
# cell_id → {arm → GNNDQN} — kept alive so we can build OOD variants later.
_ORACLE_AGENTS: dict[str, dict[str, "GNNDQN"]] = {}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(path: pathlib.Path, device: torch.device) -> GNNDQN:
    """Load a GNNDQN agent from a checkpoint file.

    Handles both demo_v1 format (encoder_raw_state_dict / q_head_state_dict)
    and the old format (encoder / q_head).
    """
    ck = torch.load(str(path), map_location=device, weights_only=False)
    if "encoder_raw_state_dict" in ck:
        enc_sd = ck["encoder_raw_state_dict"]
        qh_sd = ck["q_head_state_dict"]
        hidden_dim = ck["hidden_dim"]
        node_in_dim = ck["node_in_dim"]
    else:
        enc_sd = ck["encoder"]
        qh_sd = ck["q_head"]
        w = enc_sd.get("conv1.lin_l.weight") or enc_sd.get("conv1.lin.weight")
        hidden_dim = w.shape[0] if w is not None else 128
        node_in_dim = w.shape[1] if w is not None else 4

    agent = GNNDQN(node_in_dim=node_in_dim, hidden_dim=hidden_dim, device=device, seed=0)
    agent._encoder_raw.load_state_dict(enc_sd)
    agent._q_head_raw.load_state_dict(qh_sd)
    agent.update_target()
    agent._encoder_raw.eval()
    agent._q_head_raw.eval()
    return agent


# ---------------------------------------------------------------------------
# Graph reconstruction
# ---------------------------------------------------------------------------

def _rebuild_graph(scale_entry: dict, grid_seed: int) -> DynamicGraph:
    """Reconstruct the post-training graph deterministically from config + seed."""
    grid = scale_entry["grid"]
    n_iters = scale_entry["train"]["n_iterations"]
    g = DynamicGraph(
        grid_width=grid["grid_width"],
        grid_height=grid["grid_height"],
        extra_edges=grid.get("extra_edges", 2),
        deactivate_prob=grid.get("deactivate_prob", 0.15),
        node_deactivate_prob=grid.get("node_deactivate_prob", 0.05),
        seed=grid_seed,
    )
    for i in range(1, n_iters + 1):
        g.update_graph(iteration=i)
    return g


# ---------------------------------------------------------------------------
# Dijkstra (inline)
# ---------------------------------------------------------------------------

def _dijkstra(g: DynamicGraph, src: str, dst: str) -> tuple[float, list[str]]:
    """Dijkstra on DynamicGraph using the same cost formula as rollout.

    Step cost: edge["distance"] + 0.1 * edge["time"] + g.nodes[nb]["node_penalty"]
    Returns (cost, path) where path is a list of node_ids from src to dst.
    If unreachable, returns (float("inf"), [src]).
    """
    dist: dict[str, float] = {src: 0.0}
    prev: dict[str, str | None] = {src: None}
    heap: list[tuple[float, str]] = [(0.0, src)]

    while heap:
        cost, node = heapq.heappop(heap)
        if node == dst:
            path: list[str] = []
            cur: str | None = dst
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            path.reverse()
            return cost, path

        if cost > dist.get(node, float("inf")):
            continue

        if not g.nodes[node]["active"]:
            continue

        for nb, edata in g.graph.get(node, {}).items():
            if not edata["active"]:
                continue
            if not g.nodes[nb]["active"]:
                continue
            step_cost = edata["distance"] + 0.1 * edata["time"] + g.nodes[nb]["node_penalty"]
            new_cost = cost + step_cost
            if new_cost < dist.get(nb, float("inf")):
                dist[nb] = new_cost
                prev[nb] = node
                heapq.heappush(heap, (new_cost, nb))

    return float("inf"), [src]


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _rollout(
    agent: GNNDQN,
    g: DynamicGraph,
    src: str,
    dst: str,
    max_steps: int = 500,
) -> dict:
    """Greedy rollout with epsilon=0.0.

    Returns dict with:
      path, reached, steps, cost, raw_cost, encode_ms, mean_latency_ms
    """
    data = dynamic_graph_to_pyg(g, device=agent.device)

    t0 = time.perf_counter()
    agent.encode(data)
    encode_ms = (time.perf_counter() - t0) * 1000

    node_id_to_idx = data.node_id_to_idx

    current = src
    path = [current]
    visited: set[str] = {current}
    total_cost = 0.0
    decide_times: list[float] = []

    hit_step_budget = False
    for step_i in range(max_steps):
        if current == dst:
            break

        valid_actions: list[str] = []
        for nb, edata in g.graph.get(current, {}).items():
            if (
                edata["active"]
                and g.nodes[current]["active"]
                and g.nodes[nb]["active"]
                and nb not in visited
                and nb in node_id_to_idx
            ):
                valid_actions.append(nb)

        if not valid_actions:
            break

        t1 = time.perf_counter()
        chosen = agent.choose_action(current, valid_actions, dst, data, epsilon=0.0)
        decide_times.append((time.perf_counter() - t1) * 1000)

        edata = g.graph[current][chosen]
        step_cost = edata["distance"] + 0.1 * edata["time"] + g.nodes[chosen]["node_penalty"]
        total_cost += step_cost

        current = chosen
        path.append(current)
        visited.add(current)
    else:
        # for/else: loop completed all max_steps without a `break`
        hit_step_budget = (current != dst) and bool(valid_actions)

    reached = current == dst
    mean_latency_ms = float(sum(decide_times) / len(decide_times)) if decide_times else 0.0

    return {
        "path": path,
        "reached": reached,
        "steps": len(path) - 1,
        "cost": total_cost if reached else float("inf"),
        "raw_cost": total_cost,
        "encode_ms": encode_ms,
        "mean_latency_ms": mean_latency_ms,
        "hit_step_budget": hit_step_budget,
    }


def _dijkstra_blocked(
    g: DynamicGraph,
    src: str,
    dst: str,
    blocked: frozenset[str],
) -> tuple[float, list[str]]:
    """Dijkstra that treats nodes in `blocked` as inactive."""
    dist: dict[str, float] = {src: 0.0}
    prev: dict[str, str | None] = {src: None}
    heap: list[tuple[float, str]] = [(0.0, src)]

    while heap:
        cost, node = heapq.heappop(heap)
        if node == dst:
            path: list[str] = []
            cur: str | None = dst
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            path.reverse()
            return cost, path
        if cost > dist.get(node, float("inf")):
            continue
        if not g.nodes[node]["active"] or node in blocked:
            continue
        for nb, edata in g.graph.get(node, {}).items():
            if not edata["active"]:
                continue
            if not g.nodes[nb]["active"] or nb in blocked:
                continue
            step_cost = edata["distance"] + 0.1 * edata["time"] + g.nodes[nb]["node_penalty"]
            new_cost = cost + step_cost
            if new_cost < dist.get(nb, float("inf")):
                dist[nb] = new_cost
                prev[nb] = node
                heapq.heappush(heap, (new_cost, nb))

    return float("inf"), [src]


def _rollout_blocked(
    agent: GNNDQN,
    g: DynamicGraph,
    src: str,
    dst: str,
    blocked: frozenset[str],
    max_steps: int = 500,
) -> dict:
    """Greedy rollout that treats nodes in `blocked` as inactive."""
    saved_active = {nid: g.nodes[nid]["active"] for nid in blocked if nid in g.nodes}
    for nid in saved_active:
        g.nodes[nid]["active"] = False

    data = dynamic_graph_to_pyg(g, device=agent.device)

    t0 = time.perf_counter()
    agent.encode(data)
    encode_ms = (time.perf_counter() - t0) * 1000

    for nid, was in saved_active.items():
        g.nodes[nid]["active"] = was

    node_id_to_idx = data.node_id_to_idx

    current = src
    path = [current]
    visited: set[str] = {current}
    total_cost = 0.0
    decide_times: list[float] = []

    hit_step_budget = False
    for step_i in range(max_steps):
        if current == dst:
            break
        valid_actions: list[str] = []
        for nb, edata in g.graph.get(current, {}).items():
            if (
                edata["active"]
                and g.nodes[current]["active"]
                and g.nodes[nb]["active"]
                and nb not in visited
                and nb not in blocked
                and nb in node_id_to_idx
            ):
                valid_actions.append(nb)
        if not valid_actions:
            break
        t1 = time.perf_counter()
        chosen = agent.choose_action(current, valid_actions, dst, data, epsilon=0.0)
        decide_times.append((time.perf_counter() - t1) * 1000)
        edata = g.graph[current][chosen]
        step_cost = edata["distance"] + 0.1 * edata["time"] + g.nodes[chosen]["node_penalty"]
        total_cost += step_cost
        current = chosen
        path.append(current)
        visited.add(current)
    else:
        # for/else: loop completed all max_steps without a `break`
        hit_step_budget = (current != dst) and bool(valid_actions)

    reached = current == dst
    mean_latency_ms = float(sum(decide_times) / len(decide_times)) if decide_times else 0.0
    return {
        "path": path,
        "reached": reached,
        "steps": len(path) - 1,
        "cost": total_cost if reached else float("inf"),
        "raw_cost": total_cost,
        "encode_ms": encode_ms,
        "mean_latency_ms": mean_latency_ms,
        "hit_step_budget": hit_step_budget,
    }


# ---------------------------------------------------------------------------
# Wire format builder
# ---------------------------------------------------------------------------

def _build_wire_format(
    scale_entry: dict,
    scenario: dict,
    warm_agent: GNNDQN,
    cold_agent: GNNDQN,
    g: DynamicGraph,
) -> dict:
    """Build the full /reach wire format dict from agents + graph."""
    src = scenario["source"]
    dst = scenario["destination"]

    node_ids: list[str] = list(g.nodes.keys())
    idx_map: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    nodes_list: list[list[float]] = []
    for nid in node_ids:
        info = g.nodes[nid]
        cx, cy = info["coords"]
        active_f = 1.0 if info["active"] else 0.0
        nodes_list.append([cx, cy, active_f])

    seen_edges: set[tuple[int, int]] = set()
    edges_list: list[list[int]] = []
    for nid in node_ids:
        if not g.nodes[nid]["active"]:
            continue
        for nb, edata in g.graph[nid].items():
            if edata["active"] and g.nodes[nb]["active"]:
                u = idx_map[nid]
                v = idx_map[nb]
                key = (min(u, v), max(u, v))
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges_list.append([u, v])

    src_idx = idx_map.get(src, -1)
    dst_idx = idx_map.get(dst, -1)

    grid_section = {
        "w": scale_entry["grid"]["grid_width"],
        "h": scale_entry["grid"]["grid_height"],
        "nodes": nodes_list,
        "edges": edges_list,
        "src_idx": src_idx,
        "dst_idx": dst_idx,
    }

    dijk_cost, dijk_path = _dijkstra(g, src, dst)
    dijk_path_idx = [idx_map[n] for n in dijk_path if n in idx_map]

    dijkstra_section = {
        "path": dijk_path_idx,
        "cost": dijk_cost if dijk_cost != float("inf") else None,
        "structurally_unsolvable": dijk_cost == float("inf"),
    }

    warm_res = _rollout(warm_agent, g, src, dst)
    warm_path_idx = [idx_map[n] for n in warm_res["path"] if n in idx_map]
    if warm_res["reached"] and dijk_cost > 0 and dijk_cost != float("inf"):
        warm_ratio = warm_res["cost"] / dijk_cost
    else:
        warm_ratio = None

    warm_section = {
        "path": warm_path_idx,
        "reached": warm_res["reached"],
        "steps": warm_res["steps"],
        "cost": warm_res["cost"],
        "raw_cost": warm_res["raw_cost"],
        "cost_ratio_vs_dijkstra": warm_ratio,
        "mean_latency_ms": warm_res["mean_latency_ms"],
        "encode_ms": warm_res["encode_ms"],
        "hit_step_budget": warm_res["hit_step_budget"],
    }

    cold_res = _rollout(cold_agent, g, src, dst)
    cold_path_idx = [idx_map[n] for n in cold_res["path"] if n in idx_map]
    if cold_res["reached"] and dijk_cost > 0 and dijk_cost != float("inf"):
        cold_ratio = cold_res["cost"] / dijk_cost
    else:
        cold_ratio = None

    cold_section = {
        "path": cold_path_idx,
        "reached": cold_res["reached"],
        "steps": cold_res["steps"],
        "cost": cold_res["cost"],
        "raw_cost": cold_res["raw_cost"],
        "cost_ratio_vs_dijkstra": cold_ratio,
        "mean_latency_ms": cold_res["mean_latency_ms"],
        "encode_ms": cold_res["encode_ms"],
        "hit_step_budget": cold_res["hit_step_budget"],
    }

    return {
        "grid": grid_section,
        "warm": warm_section,
        "cold": cold_section,
        "dijkstra": dijkstra_section,
    }


# ---------------------------------------------------------------------------
# Startup / lifespan
# ---------------------------------------------------------------------------

def _load_one_scenario(
    scale_key: str,
    scale_entry: dict,
    sc_idx: int,
    scenario: dict,
    device: torch.device,
) -> dict | None:
    """Compute everything for one manifest scenario: graph rebuild, checkpoint
    loading, rollouts, and (for solvable 25x25 cells) oracle-arm rollouts.

    Pure w.r.t. module-level state: reads only its arguments plus the
    read-only path constants (_DEMO_AGENTS_DIR, _TRACES_ROOT, _ORACLE_SEED),
    and mutates no module-level dict/list. Each scenario builds its own
    DynamicGraph (seeded independently) and its own GNNDQN instances. Called
    sequentially, one scenario at a time, by lifespan() below - see that
    function's docstring for why this stays sequential rather than running
    scenarios concurrently.

    Returns a dict of everything the caller should fold into the module-level
    caches, or None if the scenario's checkpoints are missing (mirrors the
    old sequential loop's `continue`).
    """
    sc_seed = scenario["grid_seed"]
    cell_id = f"{scale_key}_s{sc_idx}"

    print(f"[startup] {cell_id}: rebuilding graph...", flush=True)
    g = _rebuild_graph(scale_entry, sc_seed)

    warm_pt = _DEMO_AGENTS_DIR / scenario["warm"]
    cold_pt = _DEMO_AGENTS_DIR / scenario["cold"]

    if not warm_pt.exists() or not cold_pt.exists():
        print(f"[startup] SKIP {cell_id}: checkpoint missing ({warm_pt.name} or {cold_pt.name})")
        return None

    print(f"[startup] {cell_id}: loading agents...", flush=True)
    warm_agent = _load_checkpoint(warm_pt, device)
    cold_agent = _load_checkpoint(cold_pt, device)

    print(f"[startup] {cell_id}: running rollouts...", flush=True)
    wire = _build_wire_format(scale_entry, scenario, warm_agent, cold_agent, g)

    # Keep all scenario agents + graphs for the sandbox tab
    node_ids = list(g.nodes.keys())
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    src_id = scenario["source"]
    dst_id = scenario["destination"]
    sandbox_entry = {
        "warm_agent": warm_agent,
        "cold_agent": cold_agent,
        "graph": g,
        "node_ids": node_ids,
        "idx_map": idx_map,
        "train_src": src_id,
        "train_dst": dst_id,
        "train_src_idx": idx_map.get(src_id, -1),
        "train_dst_idx": idx_map.get(dst_id, -1),
        "scale": scale_key,
        "scale_entry": scale_entry,
        "scenario": scenario,
        "grid_seed": sc_seed,
    }

    # Oracle-source mode: load classical_only + quantum_only for solvable 25×25 cells
    oracle_agents: dict[str, "GNNDQN"] | None = None
    oracle_cache_entries: dict[str, dict] = {}
    oracle_available = False
    if scale_key == "25x25" and not scenario.get("unsolvable", False):
        sc_id = scenario["scenario_id"]  # e.g. "seed191664964_s1"
        cell_oracle: dict[str, GNNDQN] = {"full_pool": warm_agent}

        for arm in ("classical_only", "quantum_only"):
            ck = _TRACES_ROOT / f"seed{_ORACLE_SEED}_{sc_id}" / arm / "agent.pt"
            if ck.exists():
                cell_oracle[arm] = _load_checkpoint(ck, device)
                print(f"[startup] {cell_id}: oracle {arm} loaded", flush=True)
            else:
                print(
                    f"[startup] {cell_id}: WARNING no oracle checkpoint for {arm} at {ck}",
                    flush=True,
                )

        oracle_agents = cell_oracle

        # Pre-build wire formats, reusing cold/dijkstra/grid from base wire
        dijk_cost_v = wire["dijkstra"]["cost"] or 0.0
        for arm, oagent in cell_oracle.items():
            if arm == "full_pool":
                oracle_cache_entries[f"{cell_id}|{arm}"] = wire
            else:
                ores = _rollout(oagent, g, src_id, dst_id)
                opath = [idx_map[n] for n in ores["path"] if n in idx_map]
                oratio = (
                    ores["cost"] / dijk_cost_v
                    if ores["reached"] and dijk_cost_v > 0
                    else None
                )
                oracle_cache_entries[f"{cell_id}|{arm}"] = {
                    **wire,
                    "warm": {
                        "path": opath,
                        "reached": ores["reached"],
                        "steps": ores["steps"],
                        "cost": ores["cost"],
                        "raw_cost": ores["raw_cost"],
                        "cost_ratio_vs_dijkstra": oratio,
                        "mean_latency_ms": ores["mean_latency_ms"],
                        "encode_ms": ores["encode_ms"],
                        "hit_step_budget": ores["hit_step_budget"],
                    },
                }

        oracle_available = bool(cell_oracle)

    dijk_cost = wire["dijkstra"]["cost"] or 0.0
    euc_dist = scenario.get("euclidean_distance", 0.0)

    summary = {
        "cell_id": cell_id,
        "scale": scale_key,
        "scenario_id": scenario.get("scenario_id", cell_id),
        "source": scenario["source"],
        "destination": scenario["destination"],
        "grid_seed": sc_seed,
        "warm_reached": wire["warm"]["reached"],
        "warm_ratio": wire["warm"]["cost_ratio_vs_dijkstra"],
        "cold_reached": wire["cold"]["reached"],
        "cold_ratio": wire["cold"]["cost_ratio_vs_dijkstra"],
        "euclidean_distance": euc_dist,
        "unsolvable": scenario.get("unsolvable", False),
        "oracle_available": oracle_available,
    }

    print(
        f"[startup] {cell_id}: done "
        f"(warm_reached={wire['warm']['reached']}, cold_reached={wire['cold']['reached']})",
        flush=True,
    )

    return {
        "cell_id": cell_id,
        "wire": wire,
        "sandbox_entry": sandbox_entry,
        "oracle_agents": oracle_agents,
        "oracle_cache_entries": oracle_cache_entries,
        "summary": summary,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all agents and precompute wire-format responses at startup.

    Each scenario's computation (graph rebuild, checkpoint load, rollouts,
    oracle-arm rollouts) is independent of every other scenario, but is run
    sequentially, one scenario at a time: PyTorch's CPU intra-op thread pool
    already parallelizes each GNNDQN forward pass internally, so running
    multiple scenarios concurrently (e.g. via a ThreadPoolExecutor) causes
    CPU thread oversubscription and contention rather than a speedup - this
    was tried and measured to make startup slower, not faster. Reassembly
    into the four module-level caches happens inline as each scenario
    finishes.
    """
    global _SANDBOX_CELLS

    device = resolve_device("auto")
    print(f"[startup] Using device: {device}")

    if not _MANIFEST_PATH.exists():
        raise RuntimeError(f"[startup] FATAL: manifest.json not found at {_MANIFEST_PATH}")
    manifest = json.loads(_MANIFEST_PATH.read_text())

    jobs = [
        (scale_key, scale_entry, sc_idx, scenario)
        for scale_key, scale_entry in manifest.items()
        for sc_idx, scenario in enumerate(scale_entry["scenarios"])
    ]
    print(f"[startup] Loading {len(jobs)} scenarios across {len(manifest)} scales...")

    for job in jobs:
        r = _load_one_scenario(*job, device=device)
        if r is None:
            continue
        cell_id = r["cell_id"]
        _CELL_CACHE[cell_id] = r["wire"]
        _SANDBOX_CELLS[cell_id] = r["sandbox_entry"]
        if r["oracle_agents"] is not None:
            _ORACLE_AGENTS[cell_id] = r["oracle_agents"]
        for k, v in r["oracle_cache_entries"].items():
            _ORACLE_CACHE[k] = v
        _CELL_SUMMARIES.append(r["summary"])

    print("[startup] All startup tasks complete. Server ready.", flush=True)

    yield

    # Cleanup (nothing to do)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ICDM 2026 GNN-DQN Demo", lifespan=lifespan)


from fastapi.requests import Request


@app.exception_handler(Exception)
async def _friendly_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a Python traceback to the browser — always a small JSON error."""
    print(f"[error] {request.method} {request.url.path}: {type(exc).__name__}: {exc}", flush=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "code": type(exc).__name__,
            "message": "Something went wrong running this scenario. Click Reset and try again.",
        },
    )


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ReachRequest(BaseModel):
    cell_id: str
    perturb_multiplier: float = 1.0  # 1.0 = training level (paper); other values = OOD live rollout


class OracleRequest(BaseModel):
    cell_id: str
    oracle_arm: str = "full_pool"  # "full_pool" | "classical_only" | "quantum_only"


class SandboxRequest(BaseModel):
    blocked: list[int] = []
    src_idx: int
    dst_idx: int
    cell_id: str = "25x25_s1"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    """Serve the static HTML demo page."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"message": "Demo UI not yet built. Use /cells and /reach APIs."})
    return FileResponse(str(index_path))


@app.get("/cells")
async def get_cells() -> list[dict]:
    """Return a list of cell summary dicts (lightweight, no heavy objects)."""
    return _CELL_SUMMARIES


@app.post("/reach")
async def post_reach(req: ReachRequest):
    """Return rollout result for a given cell_id.

    perturb_multiplier=1.0 (default) → precomputed cache at training-level perturbation.
    Any other value → live rollout on graph rebuilt with deactivate_prob scaled by that factor.
    """
    if req.perturb_multiplier == 1.0:
        wire = _CELL_CACHE.get(req.cell_id)
        if wire is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown cell_id: {req.cell_id!r}. Valid IDs: {sorted(_CELL_CACHE.keys())}",
            )
        json_bytes = json.dumps(wire, allow_nan=True).encode("utf-8")
        return Response(content=json_bytes, media_type="application/json")

    # Non-training perturbation: rebuild graph with modified deactivate_prob and run live.
    info = _SANDBOX_CELLS.get(req.cell_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown cell_id: {req.cell_id!r}. Valid IDs: {sorted(_SANDBOX_CELLS.keys())}",
        )

    scale_entry = info["scale_entry"]
    base_deact = scale_entry["grid"].get("deactivate_prob", 0.15)
    new_deact = min(0.95, base_deact * req.perturb_multiplier)
    modified_grid = {**scale_entry["grid"], "deactivate_prob": new_deact}
    if "node_deactivate_prob" in modified_grid:
        modified_grid["node_deactivate_prob"] = min(
            0.95, modified_grid["node_deactivate_prob"] * req.perturb_multiplier,
        )
    modified_entry = {**scale_entry, "grid": modified_grid}

    g = _rebuild_graph(modified_entry, info["grid_seed"])
    wire = _build_wire_format(
        modified_entry, info["scenario"],
        info["warm_agent"], info["cold_agent"], g,
    )
    json_bytes = json.dumps(wire, allow_nan=True).encode("utf-8")
    return Response(content=json_bytes, media_type="application/json")


@app.post("/oracle_reach")
async def post_oracle_reach(req: OracleRequest):
    """Return oracle-source arm rollout for a solvable 25x25 scenario.

    oracle_arm: "full_pool" | "classical_only" | "quantum_only"
    Only available for 25x25 s1-s4 (not s0 which has an inactive destination).
    """
    valid_arms = ("full_pool", "classical_only", "quantum_only")
    if req.oracle_arm not in valid_arms:
        raise HTTPException(400, detail=f"Unknown oracle_arm {req.oracle_arm!r}. Valid: {valid_arms}")

    cache_key = f"{req.cell_id}|{req.oracle_arm}"
    wire = _ORACLE_CACHE.get(cache_key)
    if wire is None:
        raise HTTPException(
            404,
            detail=f"No oracle cache for {cache_key!r}. Available: {sorted(_ORACLE_CACHE.keys())}",
        )
    json_bytes = json.dumps(wire, allow_nan=True).encode("utf-8")
    return Response(content=json_bytes, media_type="application/json")


@app.get("/sandbox/scenarios")
async def get_sandbox_scenarios() -> list[dict]:
    """Return all sandbox scenarios with their training src/dst indices."""
    return [
        {
            "cell_id": cid,
            "scale": info["scale"],
            "train_src_idx": info["train_src_idx"],
            "train_dst_idx": info["train_dst_idx"],
        }
        for cid, info in _SANDBOX_CELLS.items()
    ]


@app.get("/sandbox/grid")
async def get_sandbox_grid(cell_id: str = "25x25_s1"):
    """Return the base grid wire format for a sandbox scenario.

    Includes train_src_idx and train_dst_idx so the frontend can default
    the race to the agent's training pair and show an OOD warning when the
    user picks a different pair.
    """
    info = _SANDBOX_CELLS.get(cell_id)
    if info is None:
        # Fall back to first available cell rather than crashing
        cell_id = next(iter(_SANDBOX_CELLS), None)
        if cell_id is None:
            raise HTTPException(status_code=503, detail="Sandbox not ready")
        info = _SANDBOX_CELLS[cell_id]

    g = info["graph"]
    node_ids = info["node_ids"]
    idx_map = info["idx_map"]
    scale = info["scale"]

    nodes_list = []
    for nid in node_ids:
        node_info = g.nodes[nid]
        cx, cy = node_info["coords"]
        nodes_list.append([cx, cy, 1.0 if node_info["active"] else 0.0])

    seen: set[tuple[int, int]] = set()
    edges_list: list[list[int]] = []
    for nid in node_ids:
        if not g.nodes[nid]["active"]:
            continue
        u = idx_map[nid]
        for nb, edata in g.graph[nid].items():
            if edata["active"] and g.nodes[nb]["active"]:
                v = idx_map[nb]
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    edges_list.append([u, v])

    parts = scale.split("x")
    w_dim = int(parts[0]) if len(parts) == 2 else 25
    h_dim = int(parts[1]) if len(parts) == 2 else 25

    return {
        "w": w_dim,
        "h": h_dim,
        "nodes": nodes_list,
        "edges": edges_list,
        "train_src_idx": info["train_src_idx"],
        "train_dst_idx": info["train_dst_idx"],
    }


@app.post("/sandbox")
async def post_sandbox(req: SandboxRequest):
    """Run warm vs cold agent race on user-defined obstacle layout."""
    info = _SANDBOX_CELLS.get(req.cell_id)
    if info is None:
        raise HTTPException(
            status_code=503,
            detail=f"Sandbox cell {req.cell_id!r} not ready. "
                   f"Available: {sorted(_SANDBOX_CELLS.keys())}",
        )

    g = info["graph"]
    node_ids = info["node_ids"]
    idx_map = info["idx_map"]
    warm_agent = info["warm_agent"]
    cold_agent = info["cold_agent"]
    scale = info["scale"]

    n = len(node_ids)
    if req.src_idx < 0 or req.src_idx >= n:
        raise HTTPException(status_code=400, detail=f"src_idx out of range [0,{n})")
    if req.dst_idx < 0 or req.dst_idx >= n:
        raise HTTPException(status_code=400, detail=f"dst_idx out of range [0,{n})")
    if req.src_idx == req.dst_idx:
        raise HTTPException(status_code=400, detail="src_idx and dst_idx must differ")

    blocked_set = frozenset(node_ids[i] for i in req.blocked if 0 <= i < n)

    src = node_ids[req.src_idx]
    dst = node_ids[req.dst_idx]

    if src in blocked_set:
        raise HTTPException(status_code=400, detail="Source node is blocked")
    if dst in blocked_set:
        raise HTTPException(status_code=400, detail="Destination node is blocked")

    # Grid wire format (reflects user's blocked set)
    nodes_list = []
    for nid in node_ids:
        node_info = g.nodes[nid]
        cx, cy = node_info["coords"]
        active_f = 0.0 if (nid in blocked_set or not node_info["active"]) else 1.0
        nodes_list.append([cx, cy, active_f])
    seen: set[tuple[int, int]] = set()
    edges_list: list[list[int]] = []
    for nid in node_ids:
        if not g.nodes[nid]["active"] or nid in blocked_set:
            continue
        u = idx_map[nid]
        for nb, edata in g.graph[nid].items():
            if edata["active"] and g.nodes[nb]["active"] and nb not in blocked_set:
                v = idx_map[nb]
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    edges_list.append([u, v])

    parts = scale.split("x")
    w_dim = int(parts[0]) if len(parts) == 2 else 25
    h_dim = int(parts[1]) if len(parts) == 2 else 25

    grid_section = {
        "w": w_dim,
        "h": h_dim,
        "nodes": nodes_list,
        "edges": edges_list,
        "src_idx": req.src_idx,
        "dst_idx": req.dst_idx,
    }

    # Dijkstra
    dijk_cost, dijk_path = _dijkstra_blocked(g, src, dst, blocked_set)
    dijk_path_idx = [idx_map[nd] for nd in dijk_path if nd in idx_map]
    dijkstra_section = {
        "path": dijk_path_idx,
        "cost": dijk_cost if dijk_cost != float("inf") else None,
        "structurally_unsolvable": dijk_cost == float("inf"),
    }

    # Warm rollout
    warm_res = _rollout_blocked(warm_agent, g, src, dst, blocked_set)
    warm_path_idx = [idx_map[nd] for nd in warm_res["path"] if nd in idx_map]
    warm_ratio = (
        warm_res["cost"] / dijk_cost
        if warm_res["reached"] and dijk_cost > 0 and dijk_cost != float("inf")
        else None
    )
    warm_section = {
        "path": warm_path_idx,
        "reached": warm_res["reached"],
        "steps": warm_res["steps"],
        "cost": warm_res["cost"],
        "raw_cost": warm_res["raw_cost"],
        "cost_ratio_vs_dijkstra": warm_ratio,
        "mean_latency_ms": warm_res["mean_latency_ms"],
        "encode_ms": warm_res["encode_ms"],
        "hit_step_budget": warm_res["hit_step_budget"],
    }

    # Cold rollout
    cold_res = _rollout_blocked(cold_agent, g, src, dst, blocked_set)
    cold_path_idx = [idx_map[nd] for nd in cold_res["path"] if nd in idx_map]
    cold_ratio = (
        cold_res["cost"] / dijk_cost
        if cold_res["reached"] and dijk_cost > 0 and dijk_cost != float("inf")
        else None
    )
    cold_section = {
        "path": cold_path_idx,
        "reached": cold_res["reached"],
        "steps": cold_res["steps"],
        "cost": cold_res["cost"],
        "raw_cost": cold_res["raw_cost"],
        "cost_ratio_vs_dijkstra": cold_ratio,
        "mean_latency_ms": cold_res["mean_latency_ms"],
        "encode_ms": cold_res["encode_ms"],
        "hit_step_budget": cold_res["hit_step_budget"],
    }

    result = {
        "grid": grid_section,
        "warm": warm_section,
        "cold": cold_section,
        "dijkstra": dijkstra_section,
    }
    json_bytes = json.dumps(result, allow_nan=True).encode("utf-8")
    return Response(content=json_bytes, media_type="application/json")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_app.server:app", host="127.0.0.1", port=8765, reload=False)
