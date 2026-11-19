"""Contrato 3.2 — exploração ε-greedy. Implementação: Pessoa B. Uso: Pessoa A (agent.act)."""

from __future__ import annotations

import numpy as np


class LinearEpsilon:
    """ε(step): cai linearmente de `start` a `end` em `decay_steps` passos e fica em `end`."""

    def __init__(self, start: float, end: float, decay_steps: int):
        if not isinstance(decay_steps, (int, np.integer)) or isinstance(
            decay_steps, (bool, np.bool_)
        ):
            raise TypeError("decay_steps deve ser um inteiro.")
        if decay_steps <= 0:
            raise ValueError("decay_steps deve ser positivo.")

        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def __call__(self, step: int) -> float:
        if step < 0:
            raise ValueError("Step deve ser maior ou igual a zero.")

        if step >= self.decay_steps:
            return float(self.end)
        
        progress = step / self.decay_steps
        
        # Decaimento linear entre start e end
        return float(self.start + progress * (self.end - self.start))


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Com prob. ε, ação uniforme em range(len(q_values)); senão argmax(q_values)."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon deve estar entre 0 e 1.")
    if q_values.ndim != 1 or q_values.size == 0:
        raise ValueError("q_values deve ser um vetor não vazio.")

    if rng.random() < epsilon:
        return int(rng.integers(q_values.size))
    return int(np.argmax(q_values))
