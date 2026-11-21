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


def test_evaluate_computes_exact_metrics_and_threshold_boundary(monkeypatch: pytest.MonkeyPatch,) -> None:
    # Episódio 0: retorno = 200.0 (exatamente no threshold -> sucesso), tamanho = 2
    # Episódio 1: retorno = 100.0 (abaixo do threshold -> falha), tamanho = 1
    # Episódio 2: retorno = 300.0 (acima do threshold -> sucesso), tamanho = 3
    fake_env = FakeEnv(
        episodes=[
            [(120.0, False, False), (80.0, True, False)],
            [(100.0, True, False)],
            [(100.0, False, False), (100.0, False, False), (100.0, True, False)],
        ]
    )
    monkeypatch.setattr("dqn.evaluate.gym.make", lambda env_id: fake_env)
    cfg = _make_eval_config(solved_threshold=200.0)

    result = evaluate(policy=lambda obs: 1, cfg=cfg, n_episodes=3)

    expected_std = float(np.std([200.0, 100.0, 300.0]))
    assert result["mean_return"] == pytest.approx(200.0)
    assert result["std_return"] == pytest.approx(expected_std)
    assert result["success_rate"] == pytest.approx(2.0 / 3.0)
    assert result["mean_length"] == pytest.approx(2.0)
    assert result["n_episodes"] == 3
    assert fake_env.closed is True


def test_evaluate_stops_episode_on_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeEnv(
        episodes=[
            [(10.0, False, False), (15.0, False, True), (999.0, True, False)],
        ]
    )
    monkeypatch.setattr("dqn.evaluate.gym.make", lambda env_id: fake_env)
    cfg = _make_eval_config()

    result = evaluate(policy=lambda obs: 0, cfg=cfg, n_episodes=1)

    assert result["mean_return"] == pytest.approx(25.0)
    assert result["mean_length"] == pytest.approx(2.0)


def test_evaluate_uses_sequential_deterministic_seeds(monkeypatch: pytest.MonkeyPatch,) -> None:
    fake_env = FakeEnv(
        episodes=[
            [(1.0, True, False)],
            [(2.0, True, False)],
            [(3.0, True, False)],
        ]
    )
    monkeypatch.setattr("dqn.evaluate.gym.make", lambda env_id: fake_env)
    cfg = _make_eval_config(eval_seed=42)

    evaluate(policy=lambda obs: 0, cfg=cfg, n_episodes=3)

    assert fake_env.reset_seeds == [42, 43, 44]
