from __future__ import annotations
import copy
from pathlib import Path

import numpy as np
import torch

from dqn.config import Hyperparameters
from dqn.network import QNetwork


def _epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    # Trocar por `from dqn.exploration import epsilon_greedy`
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hp: Hyperparameters,
        rng: np.random.Generator,
    ):
        self.n_actions = n_actions
        self.hp = hp
        self.rng = rng

        torch.manual_seed(int(rng.integers(2**31)))

        self.q = QNetwork(obs_dim, n_actions, hp.hidden_sizes)
        self.q_target = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=hp.learning_rate)

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.q(obs_t).numpy()

        return _epsilon_greedy(q_values, epsilon, self.rng)

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
