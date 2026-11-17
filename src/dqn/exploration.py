"""Contrato 3.2 — exploração ε-greedy. Implementação: Pessoa B. Uso: Pessoa A (agent.act)."""

from __future__ import annotations

import numpy as np


class LinearEpsilon:
    """ε(step): cai linearmente de `start` a `end` em `decay_steps` passos e fica em `end`."""

    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def __call__(self, step: int) -> float:
        if step >= self.decay_steps:
            return float(self.end)
        
        progress = step / self.decay_steps
        
        # Decaimento linear entre start e end
        return float(self.start + progress * (self.end - self.start))


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Com prob. ε, ação uniforme em range(len(q_values)); senão argmax(q_values)."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))
