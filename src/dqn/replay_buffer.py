
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import NamedTuple
import numpy as np


@dataclass
class Batch:
    obs: np.ndarray       # (B, obs_dim) float32
    actions: np.ndarray   # (B,) int64
    rewards: np.ndarray   # (B,) float32
    next_obs: np.ndarray # (B, obs_dim) float32
    dones: np.ndarray     # (B,) float32


class Transition(NamedTuple):
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.buffer = deque(maxlen=capacity)
        self.rng = rng

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        transition = Transition(
            obs.copy(),
            action,
            reward,
            next_obs.copy(),
            done
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Batch:
        """Amostragem uniforme sem reposição."""

        if batch_size > len(self):
            raise ValueError("Batch size maior que o Replay Buffer.")

        ids = self.rng.choice(
            len(self),
            size=batch_size,
            replace=False
        )

        batch = [self.buffer[i] for i in ids]

        obs = np.asarray([t.obs for t in batch], dtype=np.float32)
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_obs = np.asarray([t.next_obs for t in batch], dtype=np.float32)
        dones = np.asarray([t.done for t in batch], dtype=np.float32)

        return Batch(obs, actions, rewards, next_obs, dones)

    def __len__(self) -> int:
        return len(self.buffer)