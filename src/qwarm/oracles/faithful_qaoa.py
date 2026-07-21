"""Faithful classical simulation of the QAOA variational quantum circuit.

Implements shortest-path routing via QAOA (Farhi et al. 2014) on a candidate
subgraph derived from K-shortest-paths decomposition. Encodes the routing
problem as a QUBO (Lucas 2014, §6.4) and runs the variational ansatz on
Qiskit's state-vector simulator with COBYLA classical optimisation.

Candidate-subgraph restriction (≤ 20 edges / qubits) keeps state-vector
simulation tractable on classical hardware.

For bulk fast path generation, see `quantum_inspired_stochastic.py`.

Parameters `diversity_scale` and `congestion_scale` in `find_optimized_route`
are accepted for interface compatibility but not used; QAOA edge weights are
taken directly from the live graph at solve time.
"""
from __future__ import annotations

import itertools
import time
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False


class FaithfulSimulatedQAOA:
    """Faithful classical simulation of QAOA for shortest-path routing.

    Follows the existing oracle interface:
      oracle = FaithfulSimulatedQAOA(nodes, graph, ...)
      cost, path, wall_time = oracle.find_optimized_route(source, destination)

    Uses the same mutable-reference semantics as other oracles: `nodes` and
    `graph` are stored as references, so DynamicGraph.update_graph() mutations
    are automatically visible to subsequent find_optimized_route calls.
    """

    name = "faithful_qaoa"

    def __init__(
        self,
        nodes: dict,
        graph: dict,
        p_layers: int = 2,
        k_candidate_paths: int = 5,
        max_edges_in_subgraph: int = 20,
        n_optimiser_restarts: int = 3,
        shots: int = 1024,
        flow_penalty_multiplier: float = 10.0,
        optimiser: str = "COBYLA",
        max_optimiser_iters: int = 50,
        seed: int = 0,
    ) -> None:
        if not _HAS_QISKIT:
            raise ImportError(
                "FaithfulSimulatedQAOA requires qiskit and qiskit-aer. "
                "Install with: uv sync --extra qiskit_backend"
            )
        self.nodes = nodes
        self.graph = graph
        self.p = p_layers
        self.k = k_candidate_paths
        self.max_edges = max_edges_in_subgraph
        self.n_restarts = n_optimiser_restarts
        self.shots = shots
        self.flow_penalty_mult = flow_penalty_multiplier
        self.optimiser = optimiser
        self.max_iters = max_optimiser_iters
        self.rng = np.random.default_rng(seed)
        self._backend = AerSimulator(method="statevector")
        self._estimator = StatevectorEstimator()

    # ------------------------------------------------------------------ #
    # Public interface (matches other oracles)                            #
    # ------------------------------------------------------------------ #

    def find_optimized_route(
        self,
        source: str,
        destination: str,
        diversity_scale: float = 1.0,  # accepted for interface compat, unused
        congestion_scale: float = 1.0,  # accepted for interface compat, unused
    ) -> tuple[float, list[str], float]:
        """Return (cost, path, wall_time_s) via QAOA on the active subgraph."""
        t0 = time.perf_counter()

        if source == destination:
            return 0.0, [source], 0.0

        # QAOA subgraph decomposition is only tractable for small graphs
        # (~hundreds of nodes). On large graphs K-shortest-paths enumeration
        # is too slow; return inf so the caller falls back to other oracles.
        if len(self.nodes) > 1000:
            return float("inf"), [], 0.0

        # Step 1: build candidate subgraph from K-shortest paths
        G_nx = self._to_networkx_active()
        if not nx.has_path(G_nx, source, destination):
            return float("inf"), [], time.perf_counter() - t0

        candidate_paths = self._k_shortest_paths(G_nx, source, destination)
        if not candidate_paths:
            return float("inf"), [], time.perf_counter() - t0

        candidate_edges = self._edges_from_paths(candidate_paths)

        if len(candidate_edges) > self.max_edges:
            candidate_edges = self._prune_edges(candidate_edges, candidate_paths)

        if len(candidate_edges) == 0:
            best = min(candidate_paths, key=lambda p: self._path_cost(p))
            return self._path_cost(best), best, time.perf_counter() - t0

        # Step 2: build QUBO
        edge_index = {e: i for i, e in enumerate(candidate_edges)}
        node_set = sorted({n for e in candidate_edges for n in e})
        node_index = {n: i for i, n in enumerate(node_set)}
        h, J = self._build_qubo(candidate_edges, edge_index, node_index, source, destination)

        # Step 3: build cost Hamiltonian
        n_qubits = len(candidate_edges)
        cost_op = self._qubo_to_sparse_pauli(h, J, n_qubits)

        # Step 4: optimise QAOA parameters
        best_params, best_value = self._optimise_qaoa(cost_op, n_qubits)

        # Step 5: sample at optimal parameters
        bitstrings = self._sample(best_params, cost_op, n_qubits)

        # Step 6: decode to path
        path, cost = self._decode_to_path(bitstrings, candidate_edges, source, destination)

        elapsed = time.perf_counter() - t0

        if path is None:
            # Fallback: best classical K-shortest path
            best = min(candidate_paths, key=lambda p: self._path_cost(p))
            return self._path_cost(best), best, elapsed

        return cost, path, elapsed

    # ------------------------------------------------------------------ #
    # Subgraph construction                                               #
    # ------------------------------------------------------------------ #

    def _to_networkx_active(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for node, ndata in self.nodes.items():
            if ndata["active"]:
                G.add_node(node)
        for u, neighbours in self.graph.items():
            for v, edata in neighbours.items():
                if (
                    edata["active"]
                    and self.nodes[u]["active"]
                    and self.nodes[v]["active"]
                ):
                    cost = edata["distance"] + 0.1 * edata["time"]
                    G.add_edge(u, v, cost=cost)
        return G

    def _k_shortest_paths(self, G: nx.DiGraph, src: str, dst: str) -> list[list[str]]:
        # islice stops the generator after self.k paths — avoids exhausting Yen's
        # algorithm on large graphs where list()[: k] would enumerate all paths.
        try:
            return list(itertools.islice(
                nx.shortest_simple_paths(G, src, dst, weight="cost"), self.k
            ))
        except nx.NetworkXNoPath:
            return []

    def _edges_from_paths(self, paths: list[list[str]]) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        edges = []
        for path in paths:
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) not in seen:
                    edges.append((u, v))
                    seen.add((u, v))
        return edges

    def _prune_edges(
        self, edges: list[tuple[str, str]], paths: list[list[str]]
    ) -> list[tuple[str, str]]:
        freq: Counter[tuple[str, str]] = Counter()
        for path in paths:
            for u, v in zip(path[:-1], path[1:]):
                freq[(u, v)] += 1
        return [e for e, _ in freq.most_common(self.max_edges)]

    # ------------------------------------------------------------------ #
    # QUBO formulation (Lucas 2014, §6.4)                                 #
    # ------------------------------------------------------------------ #

    def _edge_cost(self, u: str, v: str) -> float:
        d = self.graph[u][v]
        return d["distance"] + 0.1 * d["time"]

    def _path_cost(self, path: list[str]) -> float:
        return sum(self._edge_cost(u, v) for u, v in zip(path[:-1], path[1:]))

    def _build_qubo(
        self,
        edges: list[tuple[str, str]],
        edge_index: dict[tuple[str, str], int],
        node_index: dict[str, int],
        src: str,
        dst: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(edges)
        max_w = max(self._edge_cost(u, v) for u, v in edges)
        P = self.flow_penalty_mult * max_w

        h = np.array([self._edge_cost(u, v) for u, v in edges], dtype=float)
        J = np.zeros((n, n), dtype=float)

        for node in node_index:
            b_v = 1 if node == src else (-1 if node == dst else 0)
            out_idx = [edge_index[e] for e in edges if e[0] == node]
            in_idx = [edge_index[e] for e in edges if e[1] == node]

            # Linear contributions from flow conservation penalty
            for i in out_idx:
                h[i] += P * (1 - 2 * b_v)
            for i in in_idx:
                h[i] += P * (1 + 2 * b_v)

            # Quadratic contributions
            for i in out_idx:
                for j in out_idx:
                    if i != j:
                        J[min(i, j), max(i, j)] += P
            for i in in_idx:
                for j in in_idx:
                    if i != j:
                        J[min(i, j), max(i, j)] += P
            for i in out_idx:
                for j in in_idx:
                    J[min(i, j), max(i, j)] -= 2 * P

        return h, J

    # ------------------------------------------------------------------ #
    # QUBO → Ising → SparsePauliOp (x_i = (I - Z_i)/2)                  #
    # ------------------------------------------------------------------ #

    def _qubo_to_sparse_pauli(self, h: np.ndarray, J: np.ndarray, n: int) -> SparsePauliOp:
        consolidated: dict[str, float] = defaultdict(float)

        def _z_str(qubit: int) -> str:
            s = ["I"] * n
            s[qubit] = "Z"
            return "".join(reversed(s))

        def _zz_str(i: int, j: int) -> str:
            s = ["I"] * n
            s[i] = "Z"
            s[j] = "Z"
            return "".join(reversed(s))

        # Linear: h_i * x_i = h_i * (I - Z_i)/2 → coeff -h_i/2 on Z_i
        for i in range(n):
            if abs(h[i]) > 1e-12:
                consolidated[_z_str(i)] += -h[i] / 2

        # Quadratic: J_ij * x_i * x_j = J_ij * (I-Z_i)(I-Z_j)/4
        # Expansion: I/4 - Z_i/4 - Z_j/4 + Z_iZ_j/4 (constant dropped)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    consolidated[_z_str(i)] += -J[i, j] / 4
                    consolidated[_z_str(j)] += -J[i, j] / 4
                    consolidated[_zz_str(i, j)] += J[i, j] / 4

        terms = [(s, c) for s, c in consolidated.items() if abs(c) > 1e-12]
        if not terms:
            # Zero Hamiltonian — add identity with zero coeff to avoid empty op
            terms = [("I" * n, 0.0)]
        return SparsePauliOp.from_list(terms)

    # ------------------------------------------------------------------ #
    # QAOA circuit                                                         #
    # ------------------------------------------------------------------ #

    def _build_qaoa_circuit(
        self, gammas: np.ndarray, betas: np.ndarray, n: int, cost_op: SparsePauliOp
    ) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for k in range(self.p):
            self._apply_cost_layer(qc, float(gammas[k]), cost_op, n)
            for i in range(n):
                qc.rx(2.0 * float(betas[k]), i)
        return qc

    def _apply_cost_layer(
        self, qc: QuantumCircuit, gamma: float, cost_op: SparsePauliOp, n: int
    ) -> None:
        for pauli, coeff in zip(cost_op.paulis, cost_op.coeffs):
            c = float(np.real(coeff))
            # Qiskit stores Paulis little-endian; iterate from qubit 0
            z_qubits = [i for i in range(n) if str(pauli)[n - 1 - i] == "Z"]
            if len(z_qubits) == 1:
                qc.rz(2.0 * gamma * c, z_qubits[0])
            elif len(z_qubits) == 2:
                q0, q1 = z_qubits
                qc.cx(q0, q1)
                qc.rz(2.0 * gamma * c, q1)
                qc.cx(q0, q1)

    # ------------------------------------------------------------------ #
    # Classical optimisation                                               #
    # ------------------------------------------------------------------ #

    def _optimise_qaoa(
        self, cost_op: SparsePauliOp, n: int
    ) -> tuple[np.ndarray, float]:
        from scipy.optimize import minimize

        best_value = np.inf
        best_params: np.ndarray = np.zeros(2 * self.p)

        def objective(params: np.ndarray) -> float:
            gammas, betas = params[: self.p], params[self.p :]
            qc = self._build_qaoa_circuit(gammas, betas, n, cost_op)
            # StatevectorEstimator handles the expectation value computation
            # correctly in qiskit 1.x (avoids the non-contiguous array bug
            # in Statevector.expectation_value with Aer 0.13.3 output).
            pub = (qc, cost_op)
            result = self._estimator.run([pub]).result()
            return float(result[0].data.evs)

        for _ in range(self.n_restarts):
            x0 = self.rng.uniform(0, np.pi, size=2 * self.p)
            res = minimize(
                objective,
                x0,
                method=self.optimiser,
                options={"maxiter": self.max_iters, "rhobeg": 0.2},
            )
            if res.fun < best_value:
                best_value = res.fun
                best_params = res.x

        return best_params, best_value

    def _sample(
        self, params: np.ndarray, cost_op: SparsePauliOp, n: int
    ) -> list[tuple[str, int]]:
        gammas, betas = params[: self.p], params[self.p :]
        qc = self._build_qaoa_circuit(gammas, betas, n, cost_op)
        qc.measure_all()
        result = self._backend.run(transpile(qc, self._backend), shots=self.shots).result()
        counts = result.get_counts()
        return sorted(counts.items(), key=lambda kv: -kv[1])

    def _decode_to_path(
        self,
        bitstrings: list[tuple[str, int]],
        edges: list[tuple[str, str]],
        source: str,
        destination: str,
    ) -> tuple[list[str] | None, float]:
        for bs, _ in bitstrings[:10]:
            bs_clean = bs.replace(" ", "")[::-1]  # Qiskit little-endian → edge index order
            selected = [
                edges[i] for i, ch in enumerate(bs_clean) if i < len(edges) and ch == "1"
            ]
            path = self._reconstruct_path(selected, source, destination)
            if path is not None:
                return path, self._path_cost(path)
        return None, float("inf")

    def _reconstruct_path(
        self,
        selected: list[tuple[str, str]],
        src: str,
        dst: str,
    ) -> list[str] | None:
        if not selected:
            return None
        adj: dict[str, list[str]] = {}
        for u, v in selected:
            adj.setdefault(u, []).append(v)

        if src not in adj:
            return None

        path = [src]
        visited = {src}
        while path[-1] != dst:
            cur = path[-1]
            if cur not in adj:
                return None
            nexts = [v for v in adj[cur] if v not in visited]
            if not nexts:
                return None
            path.append(nexts[0])
            visited.add(nexts[0])
            if len(path) > 1000:
                return None
        return path
