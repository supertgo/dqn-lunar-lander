from __future__ import annotations

import numpy as np
import pytest

from dqn.config import EvalConfig
from dqn.evaluate import evaluate

class FakeEnv:
    """Ambiente determinístico para testar o contrato de evaluate()."""

    def __init__(self, episodes: list[list[tuple[float, bool, bool]]], *, return_list_obs: bool = False,) -> None:
        self._episodes = episodes
        self._return_list_obs = return_list_obs
        self._ep_idx = -1
        self._step_idx = 0
        self.reset_seeds: list[int | None] = []
        self.received_actions: list[int] = []
        self.closed = False

    def _format_obs(self, ep: int, step: int):
        values = [float(ep), float(step)]
        return values if self._return_list_obs else np.array(values, dtype=np.float32)

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
    # Retornos: [200.0 (no limiar -> sucesso), 100.0 (falha), 300.0 (sucesso)]
    # Tamanhos: [2, 1, 3]
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
    
    assert result["mean_return"] == pytest.approx(200.0)
    assert result["std_return"] == pytest.approx(81.6496580927726)
    assert result["success_rate"] == pytest.approx(2.0 / 3.0)
    assert result["mean_length"] == pytest.approx(2.0)
    assert result["n_episodes"] == 3

def test_evaluate_returns_zero_std_for_single_episode(monkeypatch: pytest.MonkeyPatch,) -> None:
    fake_env = FakeEnv(episodes=[[(42.5, True, False)]])
    monkeypatch.setattr("dqn.evaluate.gym.make", lambda env_id: fake_env)
    cfg = _make_eval_config()

    result = evaluate(policy=lambda obs: 0, cfg=cfg, n_episodes=1)

    assert result["mean_return"] == pytest.approx(42.5)
    assert result["std_return"] == pytest.approx(0.0)

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

def test_evaluate_creates_environment_from_config_env_id(monkeypatch: pytest.MonkeyPatch,) -> None:
    fake_env = FakeEnv(episodes=[[(1.0, True, False)]])
    requested_env_ids: list[str] = []

    def fake_make(env_id: str) -> FakeEnv:
        requested_env_ids.append(env_id)
        return fake_env

    monkeypatch.setattr("dqn.evaluate.gym.make", fake_make)
    cfg = _make_eval_config(env_id="CustomLunarLander-v3")

    evaluate(policy=lambda obs: 0, cfg=cfg, n_episodes=1)

    assert requested_env_ids == ["CustomLunarLander-v3"]

def test_evaluate_converts_observations_and_actions(monkeypatch: pytest.MonkeyPatch,) -> None:
    fake_env = FakeEnv(
        episodes=[[(1.0, False, False), (2.0, True, False)]],
        return_list_obs=True,
    )
    monkeypatch.setattr("dqn.evaluate.gym.make", lambda env_id: fake_env)
    cfg = _make_eval_config()
    observed_types: list[type] = []

    def numpy_action_policy(obs: np.ndarray) -> int:
        observed_types.append(type(obs))
        return np.int64(2)  # type: ignore[return-value]

    evaluate(policy=numpy_action_policy, cfg=cfg, n_episodes=1)

    assert observed_types == [np.ndarray, np.ndarray]
    assert fake_env.received_actions == [2, 2]
    assert all(type(action) is int for action in fake_env.received_actions)

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

def test_evaluate_closes_environment_after_successful_run(monkeypatch: pytest.MonkeyPatch,) -> None:
    fake_env = FakeEnv(episodes=[[(1.0, True, False)]])
    monkeypatch.setattr("dqn.evaluate.gym.make", lambda env_id: fake_env)
    cfg = _make_eval_config()

    evaluate(policy=lambda obs: 0, cfg=cfg, n_episodes=1)

    assert fake_env.closed is True


def test_evaluate_closes_environment_even_when_policy_fails(monkeypatch: pytest.MonkeyPatch,) -> None:
    fake_env = FakeEnv(episodes=[[(1.0, True, False)]])
    monkeypatch.setattr("dqn.evaluate.gym.make", lambda env_id: fake_env)
    cfg = _make_eval_config()

    def broken_policy(obs: np.ndarray) -> int:
        raise RuntimeError("falha simulada na política")

    with pytest.raises(RuntimeError, match="falha simulada"):
        evaluate(policy=broken_policy, cfg=cfg, n_episodes=1)

    assert fake_env.closed is True
    
@pytest.mark.slow
def test_evaluate_is_deterministic_on_real_lunar_lander_env() -> None:
    cfg = _make_eval_config(env_id="LunarLander-v3", eval_seed=123)
    constant_policy = lambda obs: 0

    first_run = evaluate(policy=constant_policy, cfg=cfg, n_episodes=2)
    second_run = evaluate(policy=constant_policy, cfg=cfg, n_episodes=2)

    assert first_run == second_run
    assert first_run["n_episodes"] == 2
    assert first_run["mean_length"] > 0
