from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray  # (B, obs_dim) float32
    actions: np.ndarray  # (B,)         int64
    rewards: np.ndarray  # (B,)         float32
    next_obs: np.ndarray  # (B, obs_dim) float32
    dones: np.ndarray  # (B,)         float32 — 1.0 só se terminated (truncated NÃO conta)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        raise NotImplementedError

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> Batch:
        """Amostra uniforme, com reposição ou não, usando self.rng (reprodutível)."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Número de transições armazenadas (≤ capacity)."""
        raise NotImplementedError
