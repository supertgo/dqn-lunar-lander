import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.config import Hyperparameters

from typing import Any


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
    return DQNAgent(obs_dim=8, n_actions=4, hp=make_hp(), rng=np.random.default_rng(seed))


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
