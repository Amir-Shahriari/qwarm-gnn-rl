"""Oracle-pool assembly with a config flag for demonstration-source ablations.

oracle_pool values:
    "full"           — ClassicalAStar + QuantumInspiredStochasticOracle
                       (+ FaithfulSimulatedQAOA when Qiskit is available).
                       Reproduces the legacy inline assembly bit-for-bit.
    "classical_only" — ClassicalAStar only. Yen k-shortest paths (run inside
                       discover_all_paths) remain enabled: classical = A* + Yen.
    "quantum_only"   — QuantumInspiredStochasticOracle + FaithfulSimulatedQAOA.
                       Yen k-shortest must be disabled by the caller via
                       pool_pre_seed_k_paths() so no classical demonstrations
                       leak into the buffer.
"""
from __future__ import annotations

from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle

try:
    from qwarm.oracles.faithful_qaoa import FaithfulSimulatedQAOA as _FaithfulQAOA
    _HAS_FAITHFUL_QAOA = True
except ImportError:
    _HAS_FAITHFUL_QAOA = False

ORACLE_POOLS = ("full", "classical_only", "quantum_only")

# Matches the inline assembly in scripts/run_multi_seed_warm_vs_cold.py.
DEFAULT_QAOA_PARAMS = dict(
    p_layers=2,
    k_candidate_paths=5,
    max_edges_in_subgraph=20,
    n_optimiser_restarts=2,
    max_optimiser_iters=40,
)

# Oracle/provenance names considered quantum-derived vs classical-derived.
QUANTUM_SOURCES = ("faithful_qaoa", "quantum_inspired_stochastic")
CLASSICAL_SOURCES = ("ClassicalAStar", "KShortestPaths")


def build_oracle_pool(
    nodes: dict,
    graph: dict,
    pool: str = "full",
    seed: int = 0,
    qaoa_params: dict | None = None,
) -> list:
    """Assemble the oracle ensemble for the given pool flag.

    Default ("full") returns the same oracle types, order, and parameters as
    the legacy inline assembly, so existing sweeps are reproduced exactly.
    """
    if pool not in ORACLE_POOLS:
        raise ValueError(
            f"Unknown oracle_pool {pool!r}; expected one of {ORACLE_POOLS}"
        )
    params = dict(DEFAULT_QAOA_PARAMS, **(qaoa_params or {}))

    if pool == "quantum_only" and not _HAS_FAITHFUL_QAOA:
        # Silently dropping QAOA would turn the quantum arm into
        # stochastic-A*-only and invalidate the ablation.
        raise RuntimeError(
            "oracle_pool='quantum_only' requires qiskit/qiskit-aer for "
            "FaithfulSimulatedQAOA, which is not importable."
        )

    oracles: list = []
    if pool in ("full", "classical_only"):
        oracles.append(ClassicalAStar(nodes, graph))
    if pool in ("full", "quantum_only"):
        oracles.append(QuantumInspiredStochasticOracle(nodes, graph))
        if _HAS_FAITHFUL_QAOA:
            oracles.append(_FaithfulQAOA(nodes, graph, seed=seed, **params))
    return oracles


def pool_pre_seed_k_paths(pool: str, default_k: int) -> int:
    """Yen k-shortest budget for the pre-seed discovery step.

    Yen's algorithm is a classical demonstration source, so it is disabled
    (k=0) for the quantum_only arm and kept at the configured budget otherwise.
    """
    if pool not in ORACLE_POOLS:
        raise ValueError(
            f"Unknown oracle_pool {pool!r}; expected one of {ORACLE_POOLS}"
        )
    return 0 if pool == "quantum_only" else default_k
