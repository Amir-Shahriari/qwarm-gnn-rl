"""Convert DynamicGraph to torch_geometric.Data.

Coordinate normalization defaults to per-grid [0,1]² (A2 design decision).
This ensures structural features are scale-invariant when the GNN transfers
between grids of different sizes (e.g., training on 50x50, testing on 75x75).

The optional `device` parameter constructs all tensors directly on the target
device — avoids the CPU-build → .to(GPU) slow path that dominated training time
on large grids.
"""
from __future__ import annotations

import torch
import torch_geometric.data as pyg_data

from qwarm.env.dynamic_graph import DynamicGraph


def dynamic_graph_to_pyg(
    dyn_graph: DynamicGraph,
    normalize_coords: bool = True,
    device: torch.device | str | None = None,
) -> pyg_data.Data:
    """Build a PyG Data object from the current DynamicGraph state.

    Node features (4 per node): [x_coord, y_coord, active_float, node_penalty]
    Coordinates are normalized to [0, 1]² per-grid by default.
    Only edges where both endpoints and the edge itself are active are included.

    Args:
        dyn_graph: The source graph.
        normalize_coords: Whether to normalise coordinates to [0, 1]².
        device: Target device for all tensors. None → CPU (same as before).
    """
    nodes = dyn_graph.nodes
    graph = dyn_graph.graph
    node_ids: list[str] = list(nodes.keys())
    idx_map: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    kw: dict = {"dtype": torch.float}
    if device is not None:
        kw["device"] = device

    # ---- Node features ----
    raw_coords = torch.tensor(
        [[nodes[n]["coords"][0], nodes[n]["coords"][1]] for n in node_ids],
        **kw,
    )
    if normalize_coords:
        mins = raw_coords.min(0).values
        maxs = raw_coords.max(0).values
        denom = (maxs - mins).clamp(min=1e-8)
        coords = (raw_coords - mins) / denom
    else:
        coords = raw_coords

    active = torch.tensor(
        [[float(nodes[n]["active"])] for n in node_ids], **kw
    )
    penalty = torch.tensor(
        [[nodes[n]["node_penalty"]] for n in node_ids], **kw
    )
    x = torch.cat([coords, active, penalty], dim=1)  # [N, 4]

    # ---- Edges (only active) ----
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_attr_list: list[list[float]] = []

    for n in node_ids:
        if not nodes[n]["active"]:
            continue
        for nb, data in graph[n].items():
            if data["active"] and nodes[nb]["active"]:
                src_list.append(idx_map[n])
                dst_list.append(idx_map[nb])
                edge_attr_list.append([data["distance"], data["time"]])

    long_kw: dict = {"dtype": torch.long}
    if device is not None:
        long_kw["device"] = device

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], **long_kw)
        edge_attr = torch.tensor(edge_attr_list, **kw)
    else:
        edge_index = torch.zeros(2, 0, **long_kw)
        edge_attr = torch.zeros(0, 2, **kw)

    pyg = pyg_data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    pyg.node_id_to_idx = idx_map        # type: ignore[assignment]
    pyg.idx_to_node_id = node_ids       # type: ignore[assignment]
    return pyg
