"""Goal-conditioned GNN-DQN agent with GraphSAGE encoder.

The agent uses a two-headed architecture:
- GraphSAGEEncoder: produces per-node embeddings from the live graph state.
- QHead: computes Q(current, action, goal) = MLP([h_cur || h_action || h_goal]).

Embeddings are cached after each encode() call and reused across q_values() /
choose_action() calls until the next encode(). The training loop calls encode()
once per iteration after dyn_graph.update_graph() to keep embeddings fresh.

Vectorised batch_infer advances B queries in parallel on the GPU:
- One encode() call, one adjacency-tensor build.
- Each rollout step computes Q-values for ALL (B, max_deg) pairs in a single
  MLP forward pass — no Python loop over queries.
- Finished queries are masked out; the step loop exits when all are done.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as pyg_data
from torch_geometric.nn import SAGEConv

from qwarm.replay.expert_replay_buffer import Transition
from qwarm.utils.compile import maybe_compile
from qwarm.utils.device import assert_on, resolve_device
from qwarm.utils.seeding import set_global_seed


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # aggr='sum' uses scatter_add (deterministic on CUDA with
        # use_deterministic_algorithms). 'mean' uses scatter_reduce(mean)
        # which is non-deterministic on CUDA even with the flag.
        self.conv1 = SAGEConv(node_in_dim, hidden_dim, aggr="sum")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="sum")
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.norm1(F.relu(self.conv1(x, edge_index)))
        h = self.dropout(h)
        h = self.norm2(F.relu(self.conv2(h, edge_index)))
        return h


class QHead(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h_cur: torch.Tensor,
        h_actions: torch.Tensor,
        h_goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_cur:     [hidden_dim]
        h_actions: [K, hidden_dim]
        h_goal:    [hidden_dim]
        returns:   [K]  Q-values for each candidate action
        """
        k = h_actions.shape[0]
        cur_exp = h_cur.unsqueeze(0).expand(k, -1)
        goal_exp = h_goal.unsqueeze(0).expand(k, -1)
        combined = torch.cat([cur_exp, h_actions, goal_exp], dim=1)
        return self.fc2(F.relu(self.fc1(combined))).squeeze(-1)


class GNNDQN:
    def __init__(
        self,
        node_in_dim: int = 4,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.95,
        target_update_interval: int = 100,
        grad_clip_norm: float = 1.0,
        imitation_margin: float = 0.8,
        imitation_lambda: float = 1.0,
        device: str | torch.device = "auto",
        seed: int = 0,
    ) -> None:
        set_global_seed(seed)
        self.device = resolve_device(device)

        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.grad_clip_norm = grad_clip_norm
        self.imitation_margin = imitation_margin
        self.imitation_lambda = imitation_lambda
        self._step_count = 0

        # Raw (uncompiled) refs kept for optimizer + grad clipping
        self._encoder_raw = GraphSAGEEncoder(node_in_dim, hidden_dim).to(self.device)
        self._q_head_raw = QHead(hidden_dim).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self._encoder_raw.parameters()) + list(self._q_head_raw.parameters()),
            lr=lr,
            weight_decay=1e-5,
        )

        # Target networks (not compiled — only online nets benefit from compile)
        self.target_encoder = copy.deepcopy(self._encoder_raw).to(self.device)
        self.target_q_head = copy.deepcopy(self._q_head_raw).to(self.device)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_q_head.parameters():
            p.requires_grad_(False)

        # Online nets — optionally compiled (no-op on Windows / CPU)
        self.encoder = maybe_compile(self._encoder_raw)
        self.q_head = maybe_compile(self._q_head_raw)

        # Embedding cache (keyed by id(data))
        self._cached_embeddings: torch.Tensor | None = None
        self._cached_data_id: int | None = None

        # Adjacency tensor cache for vectorised batch_infer (keyed by id(data))
        self._cached_adj_idx: torch.Tensor | None = None
        self._cached_adj_mask: torch.Tensor | None = None
        self._cached_adj_data_id: int | None = None

        # Diagnostic trace (JSONL file, one line per gradient step)
        self._trace_writer = None

    # ------------------------------------------------------------------ #
    #  Diagnostic trace                                                   #
    # ------------------------------------------------------------------ #

    def open_trace(self, trace_path: str | pathlib.Path) -> None:
        """Open a JSONL trace file; one line per gradient step is appended."""
        p = pathlib.Path(trace_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._trace_writer = open(p, "w", buffering=1)

    def close_trace(self) -> None:
        if self._trace_writer is not None:
            self._trace_writer.close()
            self._trace_writer = None

    # ------------------------------------------------------------------ #
    #  Encoding                                                            #
    # ------------------------------------------------------------------ #

    def encode(self, data: pyg_data.Data) -> torch.Tensor:
        """Run encoder on the graph snapshot and cache the result."""
        self.encoder.eval()
        with torch.no_grad():
            x = data.x.to(self.device)
            ei = data.edge_index.to(self.device)
            emb = self.encoder(x, ei)
        self._cached_embeddings = emb
        self._cached_data_id = id(data)
        return emb

    def _get_emb(self, data: pyg_data.Data) -> torch.Tensor:
        if self._cached_data_id != id(data) or self._cached_embeddings is None:
            return self.encode(data)
        return self._cached_embeddings

    def _node_emb(self, node_id: str, data: pyg_data.Data, emb: torch.Tensor) -> torch.Tensor:
        return emb[data.node_id_to_idx[node_id]]

    # ------------------------------------------------------------------ #
    #  Inference (single query)                                            #
    # ------------------------------------------------------------------ #

    def q_values(
        self,
        state: str,
        valid_actions: list[str],
        goal: str,
        data: pyg_data.Data,
    ) -> torch.Tensor:
        emb = self._get_emb(data)
        h_cur = self._node_emb(state, data, emb)
        h_goal = self._node_emb(goal, data, emb)
        h_acts = torch.stack([self._node_emb(a, data, emb) for a in valid_actions])
        self.q_head.eval()
        with torch.no_grad():
            return self.q_head(h_cur, h_acts, h_goal)

    def choose_action(
        self,
        state: str,
        valid_actions: list[str],
        goal: str,
        data: pyg_data.Data,
        epsilon: float = 0.05,
    ) -> str:
        if random.random() < epsilon:
            return random.choice(valid_actions)
        q = self.q_values(state, valid_actions, goal, data)
        return valid_actions[int(q.argmax().item())]

    @torch.no_grad()
    def select_action_with_library(
        self,
        state: str,
        valid_actions: list[str],
        goal: str,
        data: pyg_data.Data,
        library,
        lambda_retr: float = 0.5,
        adaptive_lambda: bool = False,
        lambda_max: float = 1.0,
        return_diagnostics: bool = False,
    ) -> "str | tuple[str, dict]":
        """Memory-augmented inference: blend network Q-values with library Q-values.

        Used by the warm-start agent at evaluation; cold agent still uses choose_action.

        Fixed-lambda mode (default, V1/V2 compatible):
            Q_final(s, a) = Q_network(s, a) + lambda_retr * Q_retrieval(s, a)

        Adaptive-lambda mode (V3):
            lambda_eff = lambda_max * max(0, cosine_sim(h_s, library))
            Q_final(s, a) = Q_network(s, a) + lambda_eff * Q_retrieval(s, a)

        Args:
            adaptive_lambda:    when True, scale lambda by max library similarity.
            lambda_max:         ceiling for the adaptive lambda (default 1.0).
            return_diagnostics: when True, return (action, diag_dict) instead of
                                just the action string.  diag_dict keys:
                                lambda_effective, max_similarity, q_network,
                                q_retrieval, q_final.
        """
        emb = self._get_emb(data)
        cur_idx = data.node_id_to_idx[state]
        goal_idx = data.node_id_to_idx[goal]
        h_cur = emb[cur_idx]
        h_goal = emb[goal_idx]
        h_actions = torch.stack([emb[data.node_id_to_idx[a]] for a in valid_actions])

        self.q_head.eval()
        Q_network = self.q_head(h_cur, h_actions, h_goal)          # (K,)

        if adaptive_lambda:
            Q_retrieval, max_sim = library.query(
                h_cur, h_actions, k=5, return_max_similarity=True
            )
            lambda_effective = lambda_max * max(0.0, max_sim)
        else:
            Q_retrieval = library.query(h_cur, h_actions, k=5)     # (K,)
            lambda_effective = lambda_retr
            max_sim = None

        Q_final = Q_network + lambda_effective * Q_retrieval
        action = valid_actions[int(Q_final.argmax().item())]

        if return_diagnostics:
            return action, {
                "lambda_effective": lambda_effective,
                "max_similarity": max_sim,
                "q_network": Q_network.tolist(),
                "q_retrieval": Q_retrieval.tolist(),
                "q_final": Q_final.tolist(),
            }
        return action

    # ------------------------------------------------------------------ #
    #  Vectorised batch inference (GPU-parallel across B queries)          #
    # ------------------------------------------------------------------ #

    def _build_adj_tensors(
        self,
        dyn_graph,
        data: pyg_data.Data,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build [N, max_deg] adjacency index and active-mask tensors.

        Cached by id(data) — rebuild only when the PyG data object changes.
        adj_idx:  [N, max_deg] int64 — padded neighbor indices (0 for padding)
        adj_mask: [N, max_deg] bool  — True iff edge+both endpoints active
        """
        if self._cached_adj_data_id == id(data) and self._cached_adj_idx is not None:
            return self._cached_adj_idx, self._cached_adj_mask  # type: ignore[return-value]

        node_id_to_idx = data.node_id_to_idx
        idx_to_node: list[str] = data.idx_to_node_id  # type: ignore[assignment]

        adj_lists: list[list[int]] = [[] for _ in range(N)]
        mask_lists: list[list[bool]] = [[] for _ in range(N)]

        for node_i, node in enumerate(idx_to_node):
            for nb, edata in dyn_graph.graph.get(node, {}).items():
                if nb not in node_id_to_idx:
                    continue
                nb_i = node_id_to_idx[nb]
                adj_lists[node_i].append(nb_i)
                mask_lists[node_i].append(
                    edata["active"]
                    and dyn_graph.nodes[node]["active"]
                    and dyn_graph.nodes[nb]["active"]
                )

        max_deg = max((len(l) for l in adj_lists), default=1)

        # Build on CPU numpy arrays then transfer once — avoids ~90K individual
        # CUDA scalar writes which each incur kernel-launch overhead.
        adj_idx_np = np.zeros((N, max_deg), dtype=np.int64)
        adj_mask_np = np.zeros((N, max_deg), dtype=np.bool_)
        for i, (nbrs, masks) in enumerate(zip(adj_lists, mask_lists)):
            k = len(nbrs)
            if k:
                adj_idx_np[i, :k] = nbrs
                adj_mask_np[i, :k] = masks

        adj_idx = torch.from_numpy(adj_idx_np).to(self.device)
        adj_mask = torch.from_numpy(adj_mask_np).to(self.device)

        self._cached_adj_idx = adj_idx
        self._cached_adj_mask = adj_mask
        self._cached_adj_data_id = id(data)
        return adj_idx, adj_mask

    def batch_infer(
        self,
        queries: list[tuple[str, str]],
        dyn_graph,
        data: pyg_data.Data,
        max_steps: int = 200,
    ) -> list[list[str]]:
        """Vectorised greedy rollout: B queries advance in parallel on the GPU.

        GPU-optimised: step tensors accumulate on-device; a single .cpu() call
        after the loop transfers all path data.  Per-step CUDA syncs are
        eliminated; done.all() is checked only every _CHECK_EVERY steps.
        """
        B = len(queries)
        if B == 0:
            return []

        h = self._get_emb(data)
        N, D = h.shape
        node_id_to_idx = data.node_id_to_idx
        idx_to_node: list[str] = data.idx_to_node_id  # type: ignore[assignment]

        adj_idx, adj_mask = self._build_adj_tensors(dyn_graph, data, N)
        max_deg = adj_idx.shape[1]

        sources_cpu = [node_id_to_idx.get(s, -1) for s, _ in queries]
        goals_cpu   = [node_id_to_idx.get(g, -1) for _, g in queries]

        current = torch.tensor(
            [max(s, 0) for s in sources_cpu], dtype=torch.long, device=self.device
        )
        goals_t = torch.tensor(
            [max(g, 0) for g in goals_cpu], dtype=torch.long, device=self.device
        )
        done = torch.tensor(
            [s < 0 or g < 0 for s, g in zip(sources_cpu, goals_cpu)],
            dtype=torch.bool, device=self.device,
        )

        visited = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        visited.scatter_(1, current.unsqueeze(1).clamp(0, N - 1), True)

        self._q_head_raw.eval()
        ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )

        # GPU tensors accumulated per step; transferred to CPU once after loop.
        # steps_incl counts real moves per query (running sum, avoids stacking
        # a separate moved_flags list and slicing via prefix instead of per-step if).
        step_nodes: list[torch.Tensor] = []
        steps_incl = torch.zeros(B, dtype=torch.long, device=self.device)
        _CHECK_EVERY = 16  # syncs at most max_steps//16 times for early exit

        with torch.no_grad(), ctx:
            for _step in range(max_steps):
                if _step % _CHECK_EVERY == 0 and done.all():
                    break

                # Snapshot before no_move update. `|` creates a new tensor so
                # was_done remains the pre-update reference.
                was_done = done

                nbrs    = adj_idx[current]
                edge_ok = adj_mask[current]
                not_vis = ~visited.gather(1, nbrs.clamp(0, N - 1))
                valid   = edge_ok & not_vis & ~done.unsqueeze(1)

                no_move = ~valid.any(dim=1)
                done    = done | no_move  # new tensor — was_done unaffected

                h_cur  = h[current]
                h_goal = h[goals_t]
                h_nbrs = h[nbrs.clamp(0, N - 1)]
                combined = torch.cat([
                    h_cur.unsqueeze(1).expand(-1, max_deg, -1),
                    h_nbrs,
                    h_goal.unsqueeze(1).expand(-1, max_deg, -1),
                ], dim=-1)

                q = self._q_head_raw.fc2(
                    F.relu(self._q_head_raw.fc1(combined))
                ).squeeze(-1)
                q = q.masked_fill(~valid, float("-inf"))

                best_local = q.argmax(dim=1)
                next_nodes = nbrs.gather(1, best_local.unsqueeze(1)).squeeze(1)
                next_nodes = torch.where(done, current, next_nodes)

                just_finished = (next_nodes == goals_t) & ~done
                done = done | just_finished  # new tensor

                visited.scatter_(1, next_nodes.unsqueeze(1).clamp(0, N - 1), True)

                # Queries that actually moved: were active & had a valid neighbour.
                # (just_finished queries have no_move=False so they are included.)
                moved = ~was_done & ~no_move
                step_nodes.append(next_nodes)
                steps_incl = steps_incl + moved.long()

                current = next_nodes

        # Single GPU→CPU transfer ------------------------------------------------
        # moves form a contiguous prefix in each row of all_steps because done[i]
        # is monotonically non-decreasing, so non-move entries come after all moves.
        if not step_nodes:
            return [[queries[i][0]] for i in range(B)]

        all_steps_cpu = torch.stack(step_nodes, dim=1).cpu().tolist()   # [B, S]
        steps_incl_cpu = steps_incl.cpu().tolist()                       # [B]

        paths: list[list[str]] = []
        for i in range(B):
            src_name = queries[i][0]
            if sources_cpu[i] < 0:
                paths.append([src_name])
                continue
            n = int(steps_incl_cpu[i])
            paths.append([src_name] + [idx_to_node[idx] for idx in all_steps_cpu[i][:n]])

        return paths

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def learn_from_batch(
        self,
        batch: list[Transition],
        data: pyg_data.Data,
        gamma: float | None = None,
        goal_node: str | None = None,
        trace_iteration: int = -1,
        trace_expert_buf_size: int = -1,
        trace_online_buf_size: int = -1,
    ) -> float:
        if not batch:
            return 0.0
        if gamma is None:
            gamma = self.gamma

        self._encoder_raw.train()
        self._q_head_raw.train()

        x = data.x.to(self.device)
        ei = data.edge_index.to(self.device)
        emb = self._encoder_raw(x, ei)

        with torch.no_grad():
            self.target_encoder.eval()  # S4 fix: prevent stochastic dropout in target values
            t_emb = self.target_encoder(x, ei)

        if os.environ.get("QWARM_DEBUG") == "1":
            assert_on(self.device, self._encoder_raw, self._q_head_raw)

        loss_terms: list[torch.Tensor] = []
        imitation_terms: list[torch.Tensor] = []
        valid_transitions: list[Transition] = []
        q_pred_vals: list[float] = []
        target_vals: list[float] = []

        for t in batch:
            if (
                t.state_node not in data.node_id_to_idx
                or t.action_node not in data.node_id_to_idx
            ):
                continue

            goal = goal_node
            if goal is None and hasattr(t, "goal_node"):
                goal = t.goal_node
            if goal is None or goal not in data.node_id_to_idx:
                continue

            s_idx = data.node_id_to_idx[t.state_node]
            a_idx = data.node_id_to_idx[t.action_node]
            g_idx = data.node_id_to_idx[goal]

            h_cur = emb[s_idx]
            h_act = emb[a_idx].unsqueeze(0)
            h_goal = emb[g_idx]
            q_pred = self._q_head_raw(h_cur, h_act, h_goal).squeeze()

            if t.done or not t.valid_next_actions:
                target_val = torch.tensor(t.reward, dtype=torch.float, device=self.device)
            else:
                valid_next = [
                    a for a in t.valid_next_actions if a in data.node_id_to_idx
                ]
                if not valid_next or t.next_state_node not in data.node_id_to_idx:
                    target_val = torch.tensor(t.reward, dtype=torch.float, device=self.device)
                else:
                    h_ns = t_emb[data.node_id_to_idx[t.next_state_node]]
                    h_ns_goal = t_emb[g_idx]
                    h_ns_acts = t_emb[[data.node_id_to_idx[a] for a in valid_next]]
                    next_q = self.target_q_head(h_ns, h_ns_acts, h_ns_goal).max()
                    target_val = (
                        torch.tensor(t.reward, dtype=torch.float, device=self.device)
                        + gamma * next_q
                    )

            loss_terms.append(F.mse_loss(q_pred, target_val.detach()))
            valid_transitions.append(t)
            q_pred_vals.append(float(q_pred.detach().item()))
            target_vals.append(float(target_val.detach().item()))

            # DQfD large-margin supervised imitation loss (expert transitions only).
            # Forces Q(s, a_E) >= Q(s, a) + margin for every non-expert neighbour,
            # preventing the agent from preferring circuitous non-expert paths.
            if t.is_expert and self.imitation_lambda > 0.0:
                nbr_mask = ei[0] == s_idx
                nbr_idx = ei[1, nbr_mask]          # active neighbours from edge_index
                if nbr_idx.numel() > 1:             # need at least one non-expert option
                    expert_pos_mask = nbr_idx == a_idx
                    if expert_pos_mask.any():
                        h_nbrs = emb[nbr_idx]       # [K, D]
                        q_all = self._q_head_raw(h_cur, h_nbrs, h_goal)  # [K]
                        q_exp = q_all[expert_pos_mask][0]
                        # max Q over non-expert neighbours
                        other_mask = ~expert_pos_mask
                        q_others_max = q_all[other_mask].max()
                        margin_loss = torch.relu(
                            q_others_max + self.imitation_margin - q_exp
                        )
                        imitation_terms.append(margin_loss)

        if not loss_terms:
            return 0.0

        td_loss = torch.stack(loss_terms).mean()
        if imitation_terms and self.imitation_lambda > 0.0:
            loss = td_loss + self.imitation_lambda * torch.stack(imitation_terms).mean()
        else:
            loss = td_loss
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = float(
            nn.utils.clip_grad_norm_(
                list(self._encoder_raw.parameters()) + list(self._q_head_raw.parameters()),
                self.grad_clip_norm,
            )
        )
        self.optimizer.step()

        if self._trace_writer is not None:
            n_exp = sum(1 for t in valid_transitions if t.is_expert)
            n_valid = len(valid_transitions)
            expert_losses = [lt.item() for lt, t in zip(loss_terms, valid_transitions) if t.is_expert]
            online_losses = [lt.item() for lt, t in zip(loss_terms, valid_transitions) if not t.is_expert]
            td_errs = [abs(qp - tv) for qp, tv in zip(q_pred_vals, target_vals)]
            q_abs = [abs(qp) for qp in q_pred_vals]
            entry = {
                "step": self._step_count,
                "iteration": trace_iteration,
                "expert_fraction_in_batch": n_exp / n_valid if n_valid else 0.0,
                "loss": float(loss.item()),
                "loss_expert": float(np.mean(expert_losses)) if expert_losses else None,
                "loss_online": float(np.mean(online_losses)) if online_losses else None,
                "td_error_mean_abs": float(np.mean(td_errs)) if td_errs else None,
                "td_error_max_abs": float(np.max(td_errs)) if td_errs else None,
                "q_value_mean": float(np.mean(q_pred_vals)) if q_pred_vals else None,
                "q_value_std": float(np.std(q_pred_vals)) if q_pred_vals else None,
                "q_value_max_abs": float(np.max(q_abs)) if q_abs else None,
                "grad_norm": total_norm,
                "grad_norm_clipped": min(total_norm, self.grad_clip_norm),
                "online_buffer_size": trace_online_buf_size,
                "expert_buffer_size": trace_expert_buf_size,
            }
            self._trace_writer.write(json.dumps(entry) + "\n")

        self._step_count += 1
        if self._step_count % self.target_update_interval == 0:
            self.update_target()

        self.encode(data)
        return float(loss.item())

    def update_target(self) -> None:
        self.target_encoder.load_state_dict(self._encoder_raw.state_dict())
        self.target_q_head.load_state_dict(self._q_head_raw.state_dict())

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder": self._encoder_raw.state_dict(),
                "q_head": self._q_head_raw.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self._encoder_raw.load_state_dict(ckpt["encoder"])
        self._q_head_raw.load_state_dict(ckpt["q_head"])
        self.update_target()
        self._cached_embeddings = None
        self._cached_data_id = None
        self._cached_adj_idx = None
        self._cached_adj_mask = None
        self._cached_adj_data_id = None
