from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class Transition(NamedTuple):
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_size: int,
        rng: np.random.Generator,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)):
            raise TypeError("capacity deve ser um inteiro.")
        if capacity <= 0:
            raise ValueError("capacity deve ser maior que zero.")
        if isinstance(obs_size, bool) or not isinstance(obs_size, (int, np.integer)):
            raise TypeError("obs_size deve ser um inteiro.")
        if obs_size <= 0:
            raise ValueError("obs_size deve ser maior que zero.")

        self.capacity = int(capacity)
        self.obs_size = obs_size
        self.rng = rng
        self.buffer: list[Transition] = []
        self._next_idx = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        if obs.shape != (self.obs_size,) or next_obs.shape != (self.obs_size,):
            raise ValueError("Dimensão dos vetores de estado diferente do esperado.")

        transition = Transition(obs.copy(), action, reward, next_obs.copy(), done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self.capacity

    def sample(self, batch_size: int) -> Batch:
        """Amostragem uniforme sem reposição."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise TypeError("batch_size deve ser um inteiro.")
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("Batch size inválido.")

        ids = self.rng.choice(len(self), size=batch_size, replace=False)

        batch = [self.buffer[i] for i in ids]

        obs = np.asarray([t.obs for t in batch], dtype=np.float32)
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_obs = np.asarray([t.next_obs for t in batch], dtype=np.float32)
        dones = np.asarray([t.done for t in batch], dtype=np.float32)

        return Batch(obs, actions, rewards, next_obs, dones)

    def __len__(self) -> int:
        return len(self.buffer)
