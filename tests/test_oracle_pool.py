"""Tests for the oracle_pool config flag (default reproduces legacy behaviour)."""
import pytest

from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.oracles.pool import (
    DEFAULT_QAOA_PARAMS,
    _HAS_FAITHFUL_QAOA,
    build_oracle_pool,
    pool_pre_seed_k_paths,
)
from qwarm.training.expert_seeding import discover_all_paths
from qwarm.utils.seeding import set_global_seed


# ── Default (legacy) behaviour unchanged ──────────────────────────────────────

def test_default_pool_is_full(tiny_graph):
    qiskit = pytest.importorskip("qiskit")
    qiskit_aer = pytest.importorskip("qiskit_aer")
    oracles = build_oracle_pool(tiny_graph.nodes, tiny_graph.graph)
    expected = [ClassicalAStar, QuantumInspiredStochasticOracle]
    if _HAS_FAITHFUL_QAOA:
        from qwarm.oracles.faithful_qaoa import FaithfulSimulatedQAOA
        expected.append(FaithfulSimulatedQAOA)
    # Same types in the same order as the legacy inline assembly
    assert [type(o) for o in oracles] == expected


def test_full_pool_qaoa_params_match_legacy(tiny_graph):
    qiskit = pytest.importorskip("qiskit")
    qiskit_aer = pytest.importorskip("qiskit_aer")
    qaoa = build_oracle_pool(tiny_graph.nodes, tiny_graph.graph, seed=7)[-1]
    assert qaoa.p == DEFAULT_QAOA_PARAMS["p_layers"]
    assert qaoa.k == DEFAULT_QAOA_PARAMS["k_candidate_paths"]
    assert qaoa.max_edges == DEFAULT_QAOA_PARAMS["max_edges_in_subgraph"]
    assert qaoa.n_restarts == DEFAULT_QAOA_PARAMS["n_optimiser_restarts"]
    assert qaoa.max_iters == DEFAULT_QAOA_PARAMS["max_optimiser_iters"]


def test_oracles_share_graph_references(tiny_graph):
    qiskit = pytest.importorskip("qiskit")
    qiskit_aer = pytest.importorskip("qiskit_aer")
    for oracle in build_oracle_pool(tiny_graph.nodes, tiny_graph.graph):
        assert oracle.nodes is tiny_graph.nodes
        assert oracle.graph is tiny_graph.graph


def test_unknown_pool_rejected(tiny_graph):
    with pytest.raises(ValueError):
        build_oracle_pool(tiny_graph.nodes, tiny_graph.graph, pool="bogus")
    with pytest.raises(ValueError):
        pool_pre_seed_k_paths("bogus", 10)


# ── classical_only: A* oracle + Yen (Yen budget untouched) ────────────────────

def test_classical_only_composition(tiny_graph):
    oracles = build_oracle_pool(tiny_graph.nodes, tiny_graph.graph, pool="classical_only")
    assert [type(o) for o in oracles] == [ClassicalAStar]


def test_classical_only_keeps_yen_budget():
    assert pool_pre_seed_k_paths("classical_only", 10) == 10
    assert pool_pre_seed_k_paths("full", 10) == 10


# ── quantum_only: stochastic A* + QAOA, Yen disabled ──────────────────────────

def test_quantum_only_composition(tiny_graph):
    qiskit = pytest.importorskip("qiskit")
    qiskit_aer = pytest.importorskip("qiskit_aer")
    from qwarm.oracles.faithful_qaoa import FaithfulSimulatedQAOA
    oracles = build_oracle_pool(tiny_graph.nodes, tiny_graph.graph, pool="quantum_only")
    assert [type(o) for o in oracles] == [
        QuantumInspiredStochasticOracle, FaithfulSimulatedQAOA
    ]


def test_quantum_only_disables_yen():
    assert pool_pre_seed_k_paths("quantum_only", 10) == 0


def test_discover_with_zero_k_paths_has_no_classical_entries(tiny_graph):
    set_global_seed(0)
    nodes = sorted(tiny_graph.nodes)
    queries = [(nodes[0], nodes[-1])]
    oracles = [QuantumInspiredStochasticOracle(tiny_graph.nodes, tiny_graph.graph)]
    library = discover_all_paths(
        tiny_graph, oracles, queries,
        n_perturbation_states=2, n_shortest_paths=0,
    )
    assert all(e["oracle"] == "quantum_inspired_stochastic" for e in library)


# ── Factory pool == manual assembly for the discovery step ────────────────────

def test_factory_classical_discovery_matches_manual(tiny_graph):
    """classical_only via the factory yields the identical path library as a
    manually assembled [ClassicalAStar] list under the same seed."""
    nodes = sorted(tiny_graph.nodes)
    queries = [(nodes[0], nodes[-1])]

    set_global_seed(0)
    state = tiny_graph.save_state()
    lib_factory = discover_all_paths(
        tiny_graph,
        build_oracle_pool(tiny_graph.nodes, tiny_graph.graph, pool="classical_only"),
        queries, n_perturbation_states=3, n_shortest_paths=5,
    )
    tiny_graph.restore_state(state)

    set_global_seed(0)
    lib_manual = discover_all_paths(
        tiny_graph,
        [ClassicalAStar(tiny_graph.nodes, tiny_graph.graph)],
        queries, n_perturbation_states=3, n_shortest_paths=5,
    )

    assert [e["path"] for e in lib_factory] == [e["path"] for e in lib_manual]
    assert [e["cost"] for e in lib_factory] == [e["cost"] for e in lib_manual]
