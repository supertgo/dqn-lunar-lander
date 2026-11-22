"""Estabilização opcional do treino: clipping do gradiente e decaimento da taxa de aprendizado.

Os dois campos são opcionais (None = desligado) e não entram no config_id quando
desligados, para que as runs já feitas continuem com os mesmos identificadores.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from dqn.agent import DQNAgent
from dqn.config import RunConfig, dqn_run_config, load_evaluation, load_hyperparameters
from dqn.train import learning_rate_at
from test_agent import OBS_DIM, make_hp


def _agent(**hp) -> DQNAgent:
    agent = DQNAgent(OBS_DIM, 4, make_hp(**hp), np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for _ in range(20):
        agent.store(rng.normal(size=OBS_DIM).astype(np.float32), int(rng.integers(4)),
                    float(rng.normal() * 100), rng.normal(size=OBS_DIM).astype(np.float32), False)
    return agent


def test_new_fields_default_to_off():
    hp = load_hyperparameters()
    assert hp.max_grad_norm is None
    assert hp.learning_rate_end is None


def test_config_id_unchanged_when_new_fields_are_off():
    cfg = dqn_run_config(seed=0)
    legacy = {k: v for k, v in cfg.hyperparameters.items()
              if k not in ("max_grad_norm", "learning_rate_end")}
    old = RunConfig(algo="dqn", hyperparameters=legacy, evaluation=load_evaluation(), seed=0)
    assert cfg.config_id == old.config_id


def test_config_id_changes_when_a_new_field_is_on():
    assert (dqn_run_config(0, {"max_grad_norm": 10.0}).config_id
            != dqn_run_config(0).config_id)


def test_update_clips_gradient_norm(monkeypatch):
    seen = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *args, **kwargs):
        params = list(params)
        seen.append(max_norm)
        return real(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    _agent(max_grad_norm=0.5).update()
    assert seen == [0.5]
    total = torch.norm(torch.stack([p.grad.norm() for p in _clipped_grads()]))
    assert total <= 0.5 + 1e-5


def _clipped_grads():
    agent = _agent(max_grad_norm=0.5)
    agent.update()
    return [p for p in agent.q.parameters() if p.grad is not None]


def test_update_without_clipping_does_not_call_it(monkeypatch):
    called = []
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: called.append(1))
    _agent().update()
    assert called == []


def test_set_learning_rate_changes_optimizer():
    agent = _agent()
    agent.set_learning_rate(1e-5)
    assert all(g["lr"] == 1e-5 for g in agent.optimizer.param_groups)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 1e-3), (500, 5.5e-4), (1000, 1e-4)],
)
def test_learning_rate_decays_linearly_over_training(step, expected):
    hp = make_hp(learning_rate=1e-3, learning_rate_end=1e-4, total_timesteps=1000)
    assert learning_rate_at(step, hp) == pytest.approx(expected)


def test_learning_rate_constant_when_decay_off():
    hp = make_hp(learning_rate=1e-3)
    assert learning_rate_at(700, hp) == pytest.approx(1e-3)


def test_reference_yaml_still_loads_all_fields():
    assert set(asdict(load_hyperparameters())) >= {"max_grad_norm", "learning_rate_end"}
