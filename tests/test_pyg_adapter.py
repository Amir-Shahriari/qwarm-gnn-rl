import torch
import pytest
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg


def test_pyg_data_shape(tiny_graph):
    data = dynamic_graph_to_pyg(tiny_graph)
    n = len(tiny_graph.nodes)
    assert data.x.shape == (n, 4), f"Expected [{n}, 4] node features, got {data.x.shape}"
    assert data.edge_index.shape[0] == 2
    assert data.node_id_to_idx is not None
    assert len(data.node_id_to_idx) == n


def test_node_features_normalized(tiny_graph):
    data = dynamic_graph_to_pyg(tiny_graph, normalize_coords=True)
    # x[:, 0] and x[:, 1] should be in [0, 1]
    assert float(data.x[:, 0].min()) >= -1e-6
    assert float(data.x[:, 0].max()) <= 1.0 + 1e-6
    assert float(data.x[:, 1].min()) >= -1e-6
    assert float(data.x[:, 1].max()) <= 1.0 + 1e-6


def test_node_features_no_normalization(tiny_graph):
    data = dynamic_graph_to_pyg(tiny_graph, normalize_coords=False)
    # raw coords: 5x5 grid so x in [0,4], y in [0,4]
    assert float(data.x[:, 0].max()) == pytest.approx(4.0)
    assert float(data.x[:, 1].max()) == pytest.approx(4.0)


def test_active_node_reflected(tiny_graph):
    data = dynamic_graph_to_pyg(tiny_graph)
    idx = data.node_id_to_idx["Node_1"]
    assert data.x[idx, 2].item() == pytest.approx(1.0)


def test_inactive_node_reflected(tiny_graph):
    tiny_graph.nodes["Node_1"]["active"] = False
    data = dynamic_graph_to_pyg(tiny_graph)
    idx = data.node_id_to_idx["Node_1"]
    assert data.x[idx, 2].item() == pytest.approx(0.0)


def test_inactive_edges_excluded(tiny_graph):
    """Edges where either endpoint is inactive should be excluded."""
    # deactivate all edges incident to Node_1
    for nb in list(tiny_graph.graph["Node_1"].keys()):
        tiny_graph.graph["Node_1"][nb]["active"] = False
    data_before = dynamic_graph_to_pyg(tiny_graph)
    # re-run: edges from Node_1 should not appear
    n1_idx = data_before.node_id_to_idx["Node_1"]
    src_nodes = data_before.edge_index[0].tolist()
    dst_nodes = data_before.edge_index[1].tolist()
    assert n1_idx not in src_nodes


def test_idx_roundtrip(tiny_graph):
    data = dynamic_graph_to_pyg(tiny_graph)
    for nid, idx in data.node_id_to_idx.items():
        assert data.idx_to_node_id[idx] == nid


def test_transfer_normalization_comparable():
    """Two different-sized grids normalized per-grid have compatible coordinate ranges."""
    g_small = DynamicGraph(5, 5, seed=0)
    g_large = DynamicGraph(10, 10, seed=0)
    d_small = dynamic_graph_to_pyg(g_small, normalize_coords=True)
    d_large = dynamic_graph_to_pyg(g_large, normalize_coords=True)
    # Both grids: coords in [0,1]
    assert float(d_small.x[:, 0].max()) <= 1.0 + 1e-6
    assert float(d_large.x[:, 0].max()) <= 1.0 + 1e-6
