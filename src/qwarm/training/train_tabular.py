"""Tabular training loop — verbatim from legacy baseline."""
from qwarm.agents.tabular_q import QLearningAgent
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv


def run_episode(
    env: PathfindingEnv,
    agent: QLearningAgent,
) -> tuple[float, list[str]]:
    state = env.reset()
    done = False
    path = [state]
    total_reward = 0.0

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        action = agent.choose_action(state, valid_actions)
        next_state, reward, done = env.step(action)
        total_reward += reward
        next_valid = env.get_valid_actions() if not done else []
        agent.update_q(state, action, reward, next_state, next_valid)
        if next_state != state:
            path.append(next_state)
        state = next_state

    return total_reward, path


def train_qlearning_agent(
    dyn_graph: DynamicGraph,
    source: str,
    destination: str,
    episodes: int = 3000,
    agent: QLearningAgent | None = None,
) -> QLearningAgent:
    env = PathfindingEnv(
        dyn_graph.graph, dyn_graph.nodes, source, destination, max_steps=200
    )
    if agent is None:
        agent = QLearningAgent(alpha=0.05, gamma=0.95, epsilon=0.2)

    for ep in range(episodes):
        run_episode(env, agent)
        if ep % 500 == 0 and agent.epsilon > 0.01:
            agent.epsilon *= 0.99

    return agent
