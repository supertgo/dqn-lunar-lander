from __future__ import annotations

from pathlib import Path

import numpy as np

from dqn.config import Hyperparameters


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hp: Hyperparameters,
        rng: np.random.Generator,
    ):
        raise NotImplementedError

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        raise NotImplementedError

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        """Guarda a transição no replay buffer."""
        raise NotImplementedError

    def update(self) -> float:
        """
        Alvo: y = r + γ · (1 − done) · max_a' Q_alvo(s', a')
        """
        raise NotImplementedError

    def sync_target(self) -> None:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError
