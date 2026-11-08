"""Contrato 3.2 — exploração ε-greedy. Implementação: Pessoa B. Uso: Pessoa A (agent.act)."""

from __future__ import annotations

import numpy as np


class LinearEpsilon:
    """ε(step): cai linearmente de `start` a `end` em `decay_steps` passos e fica em `end`."""

    def __init__(self, start: float, end: float, decay_steps: int):
        raise NotImplementedError

    def __call__(self, step: int) -> float:
        raise NotImplementedError


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Com prob. ε, ação uniforme em range(len(q_values)); senão argmax(q_values)."""
    raise NotImplementedError
