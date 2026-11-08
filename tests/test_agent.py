import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.config import Hyperparameters

from typing import Any

OBS_DIM = 8
N_ACTIONS = 4


def make_hp(**overrides) -> Hyperparameters:
    values: dict[str, Any] = dict(
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=8,
        buffer_size=100,
        learning_starts=0,
        target_update_interval=10,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100,
        total_timesteps=1000,
        train_freq=1,
        hidden_sizes=(16, 16),
    )

    values.update(overrides)
    return Hyperparameters(**values)


def make_agent(seed: int = 0) -> DQNAgent:
    return DQNAgent(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS, hp=make_hp(), rng=np.random.default_rng(seed)
    )


def test_target_network_starts_with_same_weights_as_online():
    agent = make_agent()

    for p_online, p_target in zip(agent.q.parameters(), agent.q_target.parameters()):
        assert torch.equal(p_online, p_target)


def test_target_network_is_independent_copy():
    agent = make_agent()

    with torch.no_grad():
        next(agent.q.parameters()).add_(1.0)

    p_online = next(agent.q.parameters())
    p_target = next(agent.q_target.parameters())

    assert not torch.equal(p_online, p_target)


def test_same_seed_gives_same_weights():
    a = make_agent(seed=0)
    b = make_agent(seed=0)

    for p_a, p_b in zip(a.q.parameters(), b.q.parameters()):
        assert torch.equal(p_a, p_b)


def test_different_seed_gives_different_weights():
    a = make_agent(seed=0)
    b = make_agent(seed=1)

    for p_a, p_b in zip(a.q.parameters(), b.q.parameters()):
        assert not torch.equal(p_a, p_b)


def test_act_returns_valid_python_int():
    agent = make_agent()
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    for epsilon in (0.0, 1.0):
        action = agent.act(obs, epsilon)

        assert type(action) is int
        assert 0 <= action < N_ACTIONS


def test_act_with_zero_epsilon_is_greedy():
    agent = make_agent()
    rng = np.random.default_rng(42)

    for _ in range(20):
        obs = rng.standard_normal(OBS_DIM).astype(np.float32)
        with torch.no_grad():
            expected = agent.q(torch.as_tensor(obs)).argmax().item()
        assert agent.act(obs, 0.0) == expected


def test_act_with_epsilon_one_is_exploratory():
    agent = make_agent()
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    actions = {agent.act(obs, 1.0) for _ in range(200)}
    assert actions == set(range(N_ACTIONS))
