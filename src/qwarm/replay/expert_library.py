"""ExpertLibrary — persistent, queryable store of expert (state, action, q_target) tuples.

Used at runtime by the warm-start agent to bias Q-values toward known-good demonstrations.

Deviation from plan spec: since the actual GNNDQN uses variable action sets (string node IDs
rather than fixed integer indices), `query` takes both state_emb and action_embs to return
per-candidate-action Q-values of shape (K,) for K valid neighbours.
"""
from __future__ import annotations

import pickle
from typing import List, Optional

import torch
import torch.nn.functional as F


class ExpertLibrary:
    def __init__(
        self,
        embed_dim: int = 128,
        max_size: int = 5000,
        similarity: str = "cosine",
        device: str = "cpu",
    ):
        self.embed_dim = embed_dim
        self.max_size = max_size
        self.similarity = similarity
        self.device = device

        self._state_embs: List[torch.Tensor] = []   # each (D,)
        self._action_embs: List[torch.Tensor] = []  # each (D,)
        self._q_targets: List[float] = []
        self._meta: List[dict] = []

    def add(
        self,
        state_emb: torch.Tensor,
        action_emb: torch.Tensor,
        q_target: float,
        meta: Optional[dict] = None,
    ) -> None:
        if len(self._state_embs) >= self.max_size:
            # FIFO eviction
            self._state_embs.pop(0)
            self._action_embs.pop(0)
            self._q_targets.pop(0)
            self._meta.pop(0)
        self._state_embs.append(state_emb.detach().cpu())
        self._action_embs.append(action_emb.detach().cpu())
        self._q_targets.append(float(q_target))
        self._meta.append(meta or {})

    def add_path(
        self,
        encoder,
        data,
        path_nodes: List[str],
        q_targets: List[float],
    ) -> None:
        """Bulk-add an expert trajectory given string node IDs and a pyg Data object."""
        if len(path_nodes) < 2:
            return
        device = next(encoder.parameters()).device
        with torch.no_grad():
            emb = encoder(data.x.to(device), data.edge_index.to(device))
        for s, a, q in zip(path_nodes[:-1], path_nodes[1:], q_targets):
            if s not in data.node_id_to_idx or a not in data.node_id_to_idx:
                continue
            s_emb = emb[data.node_id_to_idx[s]]
            a_emb = emb[data.node_id_to_idx[a]]
            self.add(s_emb, a_emb, q, meta={"state": s, "action": a})

    def query(
        self,
        state_emb: torch.Tensor,
        action_embs: torch.Tensor,
        k: int = 5,
        return_max_similarity: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, float]":
        """Return (K,) similarity-weighted retrieval Q-values for K candidate actions.

        Finds the k stored entries most similar to state_emb, then credits each
        retrieved Q-value to whichever candidate action embedding is most similar
        to the stored action embedding. Returns zeros if the library is empty
        (graceful degradation to pure-DQN behaviour).

        Args:
            state_emb:            (D,) embedding of the current node.
            action_embs:          (K, D) embeddings of K valid neighbour actions.
            k:                    number of nearest expert entries to retrieve.
            return_max_similarity: when True return (q_ret, max_sim) where max_sim
                                  is the maximum similarity between state_emb and
                                  ALL stored library state embeddings (a float).
                                  For cosine mode, max_sim is in [-1, 1].
                                  For dot/euclidean modes, max_sim is normalised
                                  via tanh then clipped to [0, 1] — these modes
                                  are not the headline path and the normalisation
                                  is only approximate.
        """
        K = action_embs.shape[0]
        if not self._state_embs:
            if return_max_similarity:
                return torch.zeros(K, device=state_emb.device), 0.0
            return torch.zeros(K, device=state_emb.device)

        all_s = torch.stack(self._state_embs).to(state_emb.device)
        if self.similarity == "cosine":
            sims = F.cosine_similarity(state_emb.unsqueeze(0), all_s, dim=-1)
        elif self.similarity == "dot":
            sims = (all_s @ state_emb)
        elif self.similarity == "euclidean":
            sims = -torch.norm(all_s - state_emb.unsqueeze(0), dim=-1)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")

        # Compute max similarity over ALL entries before top-k selection.
        if return_max_similarity:
            if self.similarity == "cosine":
                max_sim: float = float(sims.max().item())  # already in [-1, 1]
            else:
                # Non-cosine modes: normalise via tanh so the result is roughly
                # in [0, 1].  Not the headline path — cosine is recommended.
                max_sim = float(torch.tanh(sims.max()).clamp(0.0, 1.0).item())

        k_eff = min(k, len(self._state_embs))
        top_sims, top_idx = torch.topk(sims, k_eff)
        weights = F.softmax(top_sims, dim=0)

        q_ret = torch.zeros(K, device=state_emb.device)
        for w, idx in zip(weights, top_idx):
            a_stored = self._action_embs[idx.item()].to(state_emb.device)
            q_val = self._q_targets[idx.item()]
            act_sims = F.cosine_similarity(a_stored.unsqueeze(0), action_embs, dim=-1)
            best = int(act_sims.argmax().item())
            q_ret[best] += w.item() * q_val

        if return_max_similarity:
            return q_ret, max_sim
        return q_ret

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "embed_dim": self.embed_dim,
                    "max_size": self.max_size,
                    "similarity": self.similarity,
                    "state_embs": self._state_embs,
                    "action_embs": self._action_embs,
                    "q_targets": self._q_targets,
                    "meta": self._meta,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ExpertLibrary":
        with open(path, "rb") as f:
            d = pickle.load(f)
        lib = cls(
            embed_dim=d["embed_dim"],
            max_size=d["max_size"],
            similarity=d["similarity"],
            device=device,
        )
        lib._state_embs = d["state_embs"]
        lib._action_embs = d["action_embs"]
        lib._q_targets = d["q_targets"]
        lib._meta = d["meta"]
        return lib

    def __len__(self) -> int:
        return len(self._state_embs)
