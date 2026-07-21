"""Two-pool replay buffer that prevents catastrophic forgetting of expert demonstrations.

The expert pool is persistent: it is never randomly evicted and its transitions are
mixed into every gradient batch at a fixed ratio (default 30%). This means the
network sees oracle demonstrations in EVERY gradient update, preventing the v1
wash-out where one-shot incorporate_expert_path writes were overwritten by 1500
online TD updates.

Design follows Wei et al. DRL-QER (2024).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state_node: str
    action_node: str
    reward: float
    next_state_node: str
    done: bool
    valid_next_actions: list[str]
    is_expert: bool
    iteration_added: int
    goal_node: str = ""  # Destination node for goal-conditioned RL


class ExpertReplayBuffer:
    def __init__(
        self,
        expert_capacity: int = 50_000,
        online_capacity: int = 100_000,
        expert_ratio: float = 0.30,
        staleness_window: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.expert_pool: deque[Transition] = deque(maxlen=expert_capacity)
        self.online_pool: deque[Transition] = deque(maxlen=online_capacity)
        self.expert_ratio = expert_ratio
        self.staleness_window = staleness_window
        self.rng = rng if rng is not None else np.random.default_rng()

    def add_expert_path(
        self,
        dyn_graph,
        env_class,
        path: list[str],
        iteration: int,
        goal_node: str | None = None,
        gamma: float = 0.95,
    ) -> int:
        """Convert an oracle path into MC-return transitions and store in expert_pool.

        Each hop is stored with its full discounted return (done=True) so the agent
        directly learns Q(s_t, a_t) = MC_return without needing multi-step Bellman
        backup across N hops.

        Args:
            dyn_graph: The DynamicGraph at the state where the path was computed
            env_class: PathfindingEnv or similar
            path: The oracle-computed path [src, ..., dst]
            iteration: Current iteration number
            goal_node: The destination node (usually path[-1])
            gamma: Discount factor — must match the agent's gamma (default 0.95)

        Returns the count of transitions actually stored.
        """
        if len(path) < 2:
            return 0

        if goal_node is None:
            goal_node = path[-1]

        env = env_class(dyn_graph.graph, dyn_graph.nodes, path[0], path[-1], max_steps=9999, lambda_shape=0.0)
        env.reset()

        # Collect per-step rewards; stop at first inactive/invalid edge.
        # Invalid moves leave the agent in place (next_state != a) in every
        # invalid_penalty_mode, so this check is mode-agnostic — unlike the
        # old `reward == -5.0` sentinel which only held in legacy mode.
        steps: list[tuple[str, str, float]] = []
        for i in range(len(path) - 1):
            s, a = path[i], path[i + 1]
            env.current_node = s
            env.visited_nodes = {s}
            next_state, reward, done = env.step(a)
            if next_state != a:  # inactive edge — truncate
                break
            steps.append((s, a, reward))
            if done:
                break

        if not steps:
            return 0

        # Compute full discounted MC returns: G[t] = r[t] + γ·G[t+1]
        T = len(steps)
        mc_returns = [0.0] * T
        mc_returns[T - 1] = steps[T - 1][2]
        for t in range(T - 2, -1, -1):
            mc_returns[t] = steps[t][2] + gamma * mc_returns[t + 1]

        for (s, a, _), G_t in zip(steps, mc_returns):
            self.expert_pool.append(
                Transition(
                    state_node=s,
                    action_node=a,
                    reward=G_t,
                    next_state_node=a,   # unused — done=True skips bootstrapping
                    done=True,
                    valid_next_actions=[],
                    is_expert=True,
                    iteration_added=iteration,
                    goal_node=goal_node,
                )
            )

        return len(steps)

    def add_online_transition(self, t: Transition) -> None:
        self.online_pool.append(t)

    def prune_stale(self, current_iteration: int) -> int:
        """Drop expert transitions older than staleness_window iterations.

        Returns the number dropped. No-op if staleness_window is None.
        """
        if self.staleness_window is None:
            return 0
        cutoff = current_iteration - self.staleness_window
        before = len(self.expert_pool)
        self.expert_pool = deque(
            (t for t in self.expert_pool if t.iteration_added >= cutoff),
            maxlen=self.expert_pool.maxlen,
        )
        return before - len(self.expert_pool)

    def sample(self, batch_size: int) -> list[Transition]:
        """Return a mixed batch of expert + online transitions.

        The ratio is best-effort: if one pool is empty, the other fills the deficit.
        """
        n_expert_target = int(round(batch_size * self.expert_ratio))
        n_online_target = batch_size - n_expert_target

        n_expert_avail = min(n_expert_target, len(self.expert_pool))
        n_online_avail = min(n_online_target, len(self.online_pool))

        # Backfill deficit from whichever pool has extra
        deficit = batch_size - (n_expert_avail + n_online_avail)
        if deficit > 0:
            extra_expert = min(deficit, len(self.expert_pool) - n_expert_avail)
            n_expert_avail += extra_expert
            deficit -= extra_expert
        if deficit > 0:
            extra_online = min(deficit, len(self.online_pool) - n_online_avail)
            n_online_avail += extra_online

        expert_list = list(self.expert_pool)
        online_list = list(self.online_pool)

        expert_idx = (
            self.rng.choice(len(expert_list), size=n_expert_avail, replace=False)
            if n_expert_avail > 0
            else []
        )
        online_idx = (
            self.rng.choice(len(online_list), size=n_online_avail, replace=False)
            if n_online_avail > 0
            else []
        )

        return [expert_list[i] for i in expert_idx] + [online_list[i] for i in online_idx]

    def __len__(self) -> int:
        return len(self.expert_pool) + len(self.online_pool)
