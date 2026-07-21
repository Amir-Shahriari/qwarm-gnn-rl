"""Expert path library with oracle-source provenance tracking.

Holds pre-decomposition expert paths collected from one or more oracles,
tagged with the oracle that produced each path. The `oracle_source` tag
is kept at the path level — it never enters the ExpertReplayBuffer's
Transition objects, and therefore never enters the training loop.

Usage:
    library = build_expert_library(dyn_graph, oracles, source, destination)
    print(library.composition_report())  # {'faithful_qaoa': 3, 'quantum_inspired_stochastic': 5}
    library.seed_replay_buffer(replay_buf, dyn_graph, PathfindingEnv, iteration=0)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer


@dataclass
class ExpertPath:
    nodes: tuple[str, ...]
    cost: float
    snapshot_id: int
    oracle_source: str
    extra: dict = field(default_factory=dict)


class ExpertLibrary:
    """Collection of ExpertPath objects with oracle-provenance tracking."""

    def __init__(self) -> None:
        self._paths: list[ExpertPath] = []

    def add(self, path: ExpertPath) -> None:
        self._paths.append(path)

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self):
        return iter(self._paths)

    def composition_report(self) -> dict[str, int]:
        """Return {oracle_source: n_paths} counts for methodology reporting."""
        return dict(Counter(p.oracle_source for p in self._paths))

    def seed_replay_buffer(
        self,
        replay_buffer: "ExpertReplayBuffer",
        dyn_graph,
        env_class,
        iteration: int,
        gamma: float = 0.95,
    ) -> int:
        """Decompose all stored paths into Transition objects and add to buffer.

        Returns the total number of transitions added. The oracle_source tag
        is discarded at this boundary — it is a path-level metadata concept
        and has no role in the training loss computation.
        """
        total = 0
        for ep in self._paths:
            total += replay_buffer.add_expert_path(
                dyn_graph,
                env_class,
                list(ep.nodes),
                iteration,
                goal_node=ep.nodes[-1] if ep.nodes else None,
                gamma=gamma,
            )
        return total


def build_expert_library(
    dyn_graph,
    oracles: list,
    source: str,
    destination: str,
    iteration: int = 0,
) -> ExpertLibrary:
    """Query each oracle for source→destination and collect results in an ExpertLibrary.

    Each path is tagged with `oracle.name` (falls back to the class name).
    Returns an ExpertLibrary regardless of whether any oracle found a path.
    """
    library = ExpertLibrary()
    for oracle in oracles:
        oracle_source = getattr(oracle, "name", type(oracle).__name__)
        cost, path, _ = oracle.find_optimized_route(source, destination)
        if cost < float("inf") and len(path) >= 2:
            library.add(ExpertPath(
                nodes=tuple(path),
                cost=cost,
                snapshot_id=iteration,
                oracle_source=oracle_source,
            ))
    return library
