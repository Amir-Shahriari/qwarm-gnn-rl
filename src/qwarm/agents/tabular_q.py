"""Tabular Q-learning agent — verbatim from legacy baseline."""
import random


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple[str, str], float] = {}

    def get_q_value(self, state: str, action: str) -> float:
        return self.q_table.get((state, action), 0.0)

    def set_q_value(self, state: str, action: str, value: float) -> None:
        self.q_table[(state, action)] = value

    def choose_action(self, state: str, valid_actions: list[str]) -> str:
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = [(self.get_q_value(state, a), a) for a in valid_actions]
        return max(q_vals, key=lambda x: x[0])[1]

    def update_q(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_valid_actions: list[str],
    ) -> None:
        old_q = self.get_q_value(state, action)
        future_q = (
            max(self.get_q_value(next_state, a) for a in next_valid_actions)
            if next_valid_actions
            else 0.0
        )
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.set_q_value(state, action, new_q)
