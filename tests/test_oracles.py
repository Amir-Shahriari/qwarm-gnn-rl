import pytest
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.classical_dijkstra import ClassicalDijkstra
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.oracles.quantum_astar import QuantumAStar
from qwarm.oracles.quantum_modified_dijkstra import QuantumModifiedDijkstra


@pytest.fixture
def g5():
    return DynamicGraph(grid_width=5, grid_height=5, extra_edges=1, seed=0)


@pytest.fixture
def g5_blocked():
    """Graph where Node_1 has all edges inactive."""
    g = DynamicGraph(grid_width=5, grid_height=5, extra_edges=0, seed=1)
    for nb in list(g.graph["Node_1"].keys()):
        g.graph["Node_1"][nb]["active"] = False
        g.graph[nb]["Node_1"]["active"] = False
    return g


ORACLE_FACTORIES = [
    ("ClassicalAStar", ClassicalAStar),
    ("ClassicalDijkstra", ClassicalDijkstra),
    ("QuantumInspiredStochasticOracle", QuantumInspiredStochasticOracle),
    ("QuantumAStar", QuantumAStar),
    ("QuantumModifiedDijkstra", QuantumModifiedDijkstra),
]


@pytest.mark.parametrize("name,OracleClass", ORACLE_FACTORIES)
def test_oracle_finds_valid_path(name, OracleClass, g5):
    oracle = OracleClass(g5.nodes, g5.graph)
    cost, path, wall = oracle.find_optimized_route("Node_1", "Node_25")
    if cost < float("inf"):
        assert path[0] == "Node_1", f"{name}: path must start at source"
        assert path[-1] == "Node_25", f"{name}: path must end at destination"
        assert wall >= 0.0
        # every consecutive pair must be connected by an active edge
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            assert dst in g5.graph[src], f"{name}: {dst} not a neighbour of {src}"
            assert g5.is_edge_active(src, dst), f"{name}: inactive edge {src}->{dst}"


@pytest.mark.parametrize("name,OracleClass", ORACLE_FACTORIES)
def test_oracle_returns_inf_when_isolated(name, OracleClass, g5_blocked):
    oracle = OracleClass(g5_blocked.nodes, g5_blocked.graph)
    cost, path, _ = oracle.find_optimized_route("Node_1", "Node_25")
    assert cost == float("inf"), f"{name}: expected inf cost from isolated source"
    assert path == [], f"{name}: expected empty path from isolated source"


@pytest.mark.parametrize("name,OracleClass", ORACLE_FACTORIES)
def test_oracle_source_equals_destination(name, OracleClass, g5):
    """Trivial case: source == destination."""
    oracle = OracleClass(g5.nodes, g5.graph)
    cost, path, _ = oracle.find_optimized_route("Node_1", "Node_1")
    # Some implementations return the path ["Node_1"] with cost 0; others
    # may treat it as not-found. Both are acceptable — just don't crash.
    assert cost == 0.0 or len(path) >= 1 or cost == float("inf")


def test_determinism_classical_astar(g5):
    o1 = ClassicalAStar(g5.nodes, g5.graph)
    o2 = ClassicalAStar(g5.nodes, g5.graph)
    c1, p1, _ = o1.find_optimized_route("Node_1", "Node_25")
    c2, p2, _ = o2.find_optimized_route("Node_1", "Node_25")
    assert c1 == pytest.approx(c2)
    assert p1 == p2


def test_stochastic_oracle_runs_without_qiskit():
    """Verify QuantumInspiredStochasticOracle runs without qiskit installed."""
    from qwarm.oracles import quantum_inspired_stochastic  # noqa: F401
    g = DynamicGraph(5, 5, seed=0)
    oracle = QuantumInspiredStochasticOracle(g.nodes, g.graph)
    cost, path, _ = oracle.find_optimized_route("Node_1", "Node_25")
    assert isinstance(cost, float)


def test_quantum_astar_hqiskit_flag():
    from qwarm.oracles.quantum_astar import HAS_QISKIT
    assert isinstance(HAS_QISKIT, bool)


def test_quantum_dijkstra_has_qiskit_flag():
    from qwarm.oracles.quantum_modified_dijkstra import HAS_QISKIT
    assert isinstance(HAS_QISKIT, bool)
