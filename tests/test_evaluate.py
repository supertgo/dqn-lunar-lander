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
        
def _make_eval_config(**overrides) -> EvalConfig:
    defaults = {
        "env_id": "FakeEnv-v0",
        "eval_freq": 1000,
        "eval_episodes": 2,
        "final_eval_episodes": 5,
        "eval_seed": 100,
        "solved_threshold": 200.0,
    }
    defaults.update(overrides)
    return EvalConfig(**defaults)


@pytest.mark.parametrize("invalid_episodes", [0, -1, -10])
def test_evaluate_raises_value_error_for_non_positive_episodes(invalid_episodes: int) -> None:
    cfg = _make_eval_config()

    with pytest.raises(ValueError, match="positivo"):
        evaluate(policy=lambda obs: 0, cfg=cfg, n_episodes=invalid_episodes)


