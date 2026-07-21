"""Shared edge-cost function — verbatim from legacy baseline."""


def calculate_cost(
    distance: float,
    travel_time: float,
    quantum_value: float = 0.0,
    previous_cost: float = 0.0,
    path_length: int = 1,
    degree: int = 1,
    node_penalty: float = 0.0,
    diversity_scale: float = 1.0,
    congestion_scale: float = 1.0,
) -> float:
    quantum_scale_factor = 5.0
    step_cost = (
        distance
        + 0.1 * travel_time
        + (quantum_scale_factor * quantum_value)
        + node_penalty
    )
    diversity_penalty = diversity_scale * (path_length * 0.05)
    congestion_penalty = congestion_scale * (degree * 0.1)
    return previous_cost + step_cost + diversity_penalty + congestion_penalty
