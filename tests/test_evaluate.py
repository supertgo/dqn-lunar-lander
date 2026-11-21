from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from dqn.config import EvalConfig
from dqn.evaluate import evaluate

class FakeEnv:
    """Ambiente determinístico para testar o contrato de evaluate()."""

    def __init__(self, episodes: list[list[tuple[float, bool, bool]]]) -> None:
        self._episodes = episodes
        self._ep_idx = -1
        self._step_idx = 0
        self.reset_seeds: list[int | None] = []
        self.received_actions: list[int] = []
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        self._ep_idx += 1
        self._step_idx = 0
        return np.array([float(self._ep_idx), 0.0], dtype=np.float32), {}

    def step(self, action: int):
        self.received_actions.append(action)
        reward, terminated, truncated = self._episodes[self._ep_idx][self._step_idx]
        self._step_idx += 1
        obs = np.array([float(self._ep_idx), float(self._step_idx)], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        self.closed = True
