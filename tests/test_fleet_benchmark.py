import pytest
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.eval.fleet_benchmark import run_fleet_benchmark


@pytest.fixture
def trained_agent():
    g = DynamicGraph(5, 5, extra_edges=1, seed=0)
    agent = GNNDQN(node_in_dim=4, hidden_dim=32, seed=0)
    from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
    data = dynamic_graph_to_pyg(g)
    agent.encode(data)
    return g, agent


def test_fleet_benchmark_returns_dict(trained_agent):
    g, agent = trained_agent
    result = run_fleet_benchmark(g, agent, n_queries=20, seed=0)
    expected_keys = {
        "n_queries", "astar_total_s", "gnn_total_s",
        "astar_per_query_ms", "gnn_per_query_ms", "throughput_ratio",
    }
    assert expected_keys.issubset(result.keys())


def test_fleet_benchmark_both_produce_results(trained_agent):
    """Both A* and GNN should complete all queries without crashing."""
    g, agent = trained_agent
    result = run_fleet_benchmark(g, agent, n_queries=100, seed=0)
    # Both timings should be positive; G5 gate is measured on the large config,
    # not the 5×5 test graph (PyTorch overhead dominates at tiny scale).
    assert result["astar_total_s"] > 0.0
    assert result["gnn_total_s"] > 0.0
    assert result["n_queries"] > 0


def test_fleet_benchmark_throughput_ratio_positive(trained_agent):
    g, agent = trained_agent
    result = run_fleet_benchmark(g, agent, n_queries=10, seed=1)
    assert result["throughput_ratio"] > 0.0
