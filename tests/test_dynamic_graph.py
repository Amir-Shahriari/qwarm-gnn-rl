import pytest
from qwarm.env.dynamic_graph import DynamicGraph


def test_node_count(tiny_graph):
    assert len(tiny_graph.nodes) == 25


def test_all_nodes_active_at_init(tiny_graph):
    for info in tiny_graph.nodes.values():
        assert info["active"] is True


def test_node_ids_are_strings(tiny_graph):
    for nid in tiny_graph.nodes:
        assert isinstance(nid, str)
        assert nid.startswith("Node_")


def test_is_edge_active_checks_both_endpoints(tiny_graph):
    node_list = list(tiny_graph.nodes.keys())
    n = node_list[0]
    neighbors = list(tiny_graph.graph[n].keys())
    assert len(neighbors) > 0
    neighbor = neighbors[0]
    tiny_graph.nodes[neighbor]["active"] = False
    assert not tiny_graph.is_edge_active(n, neighbor)
    # restore
    tiny_graph.nodes[neighbor]["active"] = True
    assert tiny_graph.is_edge_active(n, neighbor)


def test_is_edge_active_checks_edge_flag(tiny_graph):
    n = list(tiny_graph.nodes.keys())[0]
    nb = list(tiny_graph.graph[n].keys())[0]
    tiny_graph.graph[n][nb]["active"] = False
    assert not tiny_graph.is_edge_active(n, nb)


def test_update_graph_perturbs_weights(tiny_graph):
    first_distances = {
        (n, nb): tiny_graph.graph[n][nb]["distance"]
        for n in list(tiny_graph.graph)[:3]
        for nb in tiny_graph.graph[n]
        if tiny_graph.graph[n][nb]["active"]
    }
    tiny_graph.update_graph(iteration=1)
    any_changed = any(
        tiny_graph.graph[n][nb]["distance"] != first_distances[(n, nb)]
        for (n, nb) in first_distances
        if tiny_graph.graph[n][nb].get("active")
    )
    assert any_changed


def test_update_graph_can_deactivate_edges(tiny_graph):
    # run many updates with very high deactivation prob
    g = DynamicGraph(5, 5, extra_edges=0, deactivate_prob=1.0, seed=7)
    g.update_graph(iteration=1)
    all_inactive = all(
        not data["active"]
        for node in g.graph
        for data in g.graph[node].values()
    )
    assert all_inactive


def test_get_node_penalty_returns_inf_when_inactive(tiny_graph):
    n = list(tiny_graph.nodes.keys())[0]
    tiny_graph.nodes[n]["active"] = False
    assert tiny_graph.get_node_penalty(n) == float("inf")


def test_determinism():
    g1 = DynamicGraph(grid_width=5, grid_height=5, extra_edges=1, seed=42)
    g2 = DynamicGraph(grid_width=5, grid_height=5, extra_edges=1, seed=42)
    for _ in range(10):
        g1.update_graph(iteration=1)
        g2.update_graph(iteration=1)
    for n in g1.graph:
        for nb in g1.graph[n]:
            assert g1.graph[n][nb]["distance"] == pytest.approx(g2.graph[n][nb]["distance"])
            assert g1.graph[n][nb]["active"] == g2.graph[n][nb]["active"]


def test_different_seeds_differ():
    g1 = DynamicGraph(5, 5, seed=0)
    g2 = DynamicGraph(5, 5, seed=999)
    dists1 = [tiny_graph_data["distance"] for n in g1.graph for tiny_graph_data in g1.graph[n].values()]
    dists2 = [tiny_graph_data["distance"] for n in g2.graph for tiny_graph_data in g2.graph[n].values()]
    assert dists1 != dists2
