"""Unit tests for ExpertLibrary (src/qwarm/replay/expert_library.py).

Tests:
  1. add / query round-trip
  2. save / load round-trip
  3. query returns correct shape (num_actions,)
  4. empty library returns zero-vector (graceful degradation)
  5. determinism: same query state → same Q-retrieval output
"""
import os
import tempfile

import pytest
import torch

from qwarm.replay.expert_library import ExpertLibrary

D = 32   # small embed dim for speed


def _rand(seed: int, *shape) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ── 1. add / query round-trip ────────────────────────────────────────────────

def test_add_and_len():
    lib = ExpertLibrary(embed_dim=D)
    assert len(lib) == 0
    lib.add(_rand(0, D), _rand(1, D), q_target=1.0)
    lib.add(_rand(2, D), _rand(3, D), q_target=2.0)
    assert len(lib) == 2


def test_query_round_trip():
    lib = ExpertLibrary(embed_dim=D)
    s = _rand(0, D)
    a = _rand(1, D)
    lib.add(s, a, q_target=5.0)

    action_embs = torch.stack([a, _rand(2, D)])     # K=2 candidates
    q_ret = lib.query(s, action_embs, k=1)

    assert q_ret.shape == (2,)
    # The stored action should receive a positive Q contribution
    assert q_ret[0] > 0.0 or q_ret[1] > 0.0


# ── 2. save / load round-trip ────────────────────────────────────────────────

def test_save_load_round_trip(tmp_path):
    lib = ExpertLibrary(embed_dim=D)
    for i in range(5):
        lib.add(_rand(i, D), _rand(i + 100, D), q_target=float(i))
    save_path = str(tmp_path / "lib.pkl")
    lib.save(save_path)

    lib2 = ExpertLibrary.load(save_path)
    assert len(lib2) == len(lib)
    assert lib2.embed_dim == D
    assert lib2.similarity == lib.similarity
    # Q-targets should round-trip exactly
    assert lib2._q_targets == lib._q_targets


# ── 3. query returns correct shape ───────────────────────────────────────────

def test_query_shape_matches_action_count():
    lib = ExpertLibrary(embed_dim=D)
    for i in range(10):
        lib.add(_rand(i, D), _rand(i + 50, D), q_target=float(i))

    for K in (1, 3, 7):
        s_emb = _rand(99, D)
        a_embs = _rand(200 + K, K, D)
        q_ret = lib.query(s_emb, a_embs, k=5)
        assert q_ret.shape == (K,), f"Expected ({K},), got {q_ret.shape}"


# ── 4. empty library returns zero-vector ────────────────────────────────────

def test_empty_library_returns_zeros():
    lib = ExpertLibrary(embed_dim=D)
    s_emb = _rand(0, D)
    a_embs = _rand(1, 4, D)
    q_ret = lib.query(s_emb, a_embs, k=5)
    assert q_ret.shape == (4,)
    assert torch.all(q_ret == 0.0)


# ── 5. determinism ───────────────────────────────────────────────────────────

def test_query_determinism():
    lib = ExpertLibrary(embed_dim=D)
    for i in range(8):
        lib.add(_rand(i, D), _rand(i + 20, D), q_target=float(i) * 0.5)

    s_emb = _rand(99, D)
    a_embs = _rand(77, 3, D)
    q1 = lib.query(s_emb, a_embs, k=5)
    q2 = lib.query(s_emb, a_embs, k=5)
    assert torch.allclose(q1, q2)


# ── 6. FIFO eviction respects max_size ───────────────────────────────────────

def test_fifo_eviction():
    lib = ExpertLibrary(embed_dim=D, max_size=3)
    for i in range(5):
        lib.add(_rand(i, D), _rand(i + 10, D), q_target=float(i))
    assert len(lib) == 3
    # Only the last 3 q-targets should remain (FIFO → 2, 3, 4)
    assert lib._q_targets == [2.0, 3.0, 4.0]


# ── 7. similarity variants don't crash ───────────────────────────────────────

@pytest.mark.parametrize("sim", ["cosine", "dot", "euclidean"])
def test_similarity_variants(sim):
    lib = ExpertLibrary(embed_dim=D, similarity=sim)
    for i in range(4):
        lib.add(_rand(i, D), _rand(i + 30, D), q_target=1.0)
    q = lib.query(_rand(9, D), _rand(5, 2, D), k=3)
    assert q.shape == (2,)
    assert torch.isfinite(q).all()


# ── 8. add_path bulk-adds transitions ────────────────────────────────────────

def test_add_path():
    """add_path should add len(path)-1 transitions from a minimal mock encoder."""
    import types

    class MockEncoder:
        def __call__(self, x, edge_index):
            # Return fixed embeddings indexed by node position
            return torch.eye(x.shape[0], D)

        def parameters(self):
            yield torch.zeros(1)   # satisfies next(encoder.parameters()).device

    class MockData:
        def __init__(self, n: int):
            self.x = torch.zeros(n, 4)
            self.edge_index = torch.zeros(2, 0, dtype=torch.long)
            self.node_id_to_idx = {f"n{i}": i for i in range(n)}

    lib = ExpertLibrary(embed_dim=D)
    enc = MockEncoder()
    data = MockData(5)
    path = ["n0", "n1", "n2", "n3"]
    q_targets = [1.0, 0.9, 0.8]     # one per hop
    lib.add_path(enc, data, path, q_targets)
    assert len(lib) == 3


# ── 9. return_max_similarity — shape regression ───────────────────────────────

def test_query_shape_regression_with_max_similarity():
    """query() without return_max_similarity must return same shape as before."""
    lib = ExpertLibrary(embed_dim=D)
    for i in range(5):
        lib.add(_rand(i, D), _rand(i + 50, D), q_target=float(i))

    s_emb = _rand(99, D)
    a_embs = _rand(200, 3, D)

    q_ret_old = lib.query(s_emb, a_embs, k=5)
    assert q_ret_old.shape == (3,), f"Expected (3,), got {q_ret_old.shape}"


# ── 10. return_max_similarity — tuple return and range ────────────────────────

def test_query_return_max_similarity_tuple():
    """return_max_similarity=True must return (tensor of shape (K,), float)."""
    lib = ExpertLibrary(embed_dim=D)
    for i in range(10):
        lib.add(_rand(i, D), _rand(i + 50, D), q_target=float(i))

    s_emb = _rand(99, D)
    a_embs = _rand(200, 4, D)

    result = lib.query(s_emb, a_embs, k=5, return_max_similarity=True)
    assert isinstance(result, tuple), "Expected a tuple when return_max_similarity=True"
    q_ret, max_sim = result
    assert q_ret.shape == (4,), f"Expected (4,), got {q_ret.shape}"
    assert isinstance(max_sim, float), f"max_sim should be float, got {type(max_sim)}"
    assert -1.0 <= max_sim <= 1.0, f"Cosine max_sim out of range: {max_sim}"


# ── 11. return_max_similarity — empty library returns 0.0 ────────────────────

def test_query_return_max_similarity_empty_library():
    """Empty library must return (zeros, 0.0) when return_max_similarity=True."""
    lib = ExpertLibrary(embed_dim=D)
    s_emb = _rand(0, D)
    a_embs = _rand(1, 3, D)

    result = lib.query(s_emb, a_embs, k=5, return_max_similarity=True)
    assert isinstance(result, tuple)
    q_ret, max_sim = result
    assert q_ret.shape == (3,)
    assert torch.all(q_ret == 0.0)
    assert max_sim == 0.0


# ── 12. return_max_similarity — cosine max_sim is max over all entries ────────

def test_query_max_similarity_is_global_max():
    """max_sim should equal the maximum cosine similarity over all stored entries."""
    import torch.nn.functional as F
    lib = ExpertLibrary(embed_dim=D, similarity="cosine")
    stored = [_rand(i, D) for i in range(8)]
    for i, s in enumerate(stored):
        lib.add(s, _rand(i + 100, D), q_target=1.0)

    s_emb = _rand(99, D)
    a_embs = _rand(77, 3, D)

    _, max_sim = lib.query(s_emb, a_embs, k=5, return_max_similarity=True)

    all_s = torch.stack(stored)
    sims = F.cosine_similarity(s_emb.unsqueeze(0), all_s, dim=-1)
    expected = float(sims.max().item())
    assert abs(max_sim - expected) < 1e-5, f"max_sim {max_sim} != expected {expected}"
