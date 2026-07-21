"""Disk cache for expert paths keyed on (seed, source, dest).

Eliminates the 400-2700s per-cell QAOA regeneration by persisting results to
runs/expert_cache/seed_<seed>/source_<s>_dest_<d>.pkl between sweep runs.

Cache layout:
    runs/expert_cache/
        seed_42/
            source_537_dest_54.pkl
            source_476_dest_449.pkl
"""
from __future__ import annotations

import pathlib
import pickle
from typing import List, Tuple

_DEFAULT_CACHE_ROOT = pathlib.Path("runs/expert_cache")


def _cache_path(
    seed: int,
    source: str,
    dest: str,
    cache_root: pathlib.Path = _DEFAULT_CACHE_ROOT,
) -> pathlib.Path:
    src_safe = source.replace("/", "_").replace("\\", "_")
    dst_safe = dest.replace("/", "_").replace("\\", "_")
    return cache_root / f"seed_{seed}" / f"source_{src_safe}_dest_{dst_safe}.pkl"


def get_expert_paths(
    seed: int,
    source: str,
    dest: str,
    env,
    oracles: list,
    force_regenerate: bool = False,
    cache_root: pathlib.Path = _DEFAULT_CACHE_ROOT,
) -> List[Tuple[float, List[str]]]:
    """Return cached (cost, path) pairs for (seed, source, dest); generate if missing.

    Each entry is a (cost, path) tuple where path is a list of node-ID strings.
    Returns an empty list if no oracle found a valid path.

    Args:
        seed:             Training seed (used as cache partition key).
        source:           Source node ID string.
        dest:             Destination node ID string.
        env:              DynamicGraph or similar — passed to oracles.
        oracles:          List of oracle objects with .find_optimized_route().
        force_regenerate: Ignore existing cache entry and recompute.
        cache_root:       Root directory for cache files.
    """
    cp = _cache_path(seed, source, dest, cache_root)

    if not force_regenerate and cp.exists():
        with open(cp, "rb") as f:
            return pickle.load(f)

    results: List[Tuple[float, List[str]]] = []
    for oracle in oracles:
        cost, path, _ = oracle.find_optimized_route(source, dest)
        if cost < float("inf") and len(path) >= 2:
            results.append((cost, list(path)))

    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump(results, f)

    return results
