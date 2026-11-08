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


def make_agent(seed: int = 0, **hp_overrides) -> DQNAgent:
    return DQNAgent(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hp=make_hp(**hp_overrides),
        rng=np.random.default_rng(seed),
    )


def make_transition(rng, terminated=False):
    obs = rng.standard_normal(OBS_DIM).astype(np.float32)
    next_obs = rng.standard_normal(OBS_DIM).astype(np.float32)
    action = int(rng.integers(N_ACTIONS))
    reward = float(rng.standard_normal())

    return obs, action, reward, next_obs, terminated


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


def test_store_adds_transitions_to_buffer():
    agent = make_agent()
    rng = np.random.default_rng(0)

    for i in range(3):
        agent.store(*make_transition(rng))
        assert len(agent.buffer) == i + 1


def test_buffer_respects_capacity():
    capacity = 5
    agent = make_agent(buffer_size=capacity)
    rng = np.random.default_rng(0)

    for i in range(12):
        agent.store(*make_transition(rng))

    assert len(agent.buffer) == capacity


def test_sample_returns_batch_with_contract_shapes_and_dtypes():
    agent = make_agent()
    rng = np.random.default_rng(0)
    batch_size = 4

    for _ in range(10):
        agent.store(*make_transition(rng))

    batch = agent.buffer.sample(batch_size)

    assert batch.obs.shape == (batch_size, OBS_DIM)
    assert batch.obs.dtype == np.float32
    assert batch.actions.shape == (batch_size,)
    assert batch.actions.dtype == np.int64
    assert batch.rewards.shape == (batch_size,)
    assert batch.rewards.dtype == np.float32
    assert batch.next_obs.shape == (batch_size, OBS_DIM)
    assert batch.next_obs.dtype == np.float32
    assert batch.dones.shape == (batch_size,)
    assert batch.dones.dtype == np.float32


def test_sample_converts_terminated_to_done_flag():
    for terminated, expected in ((True, 1.0), (False, 0.0)):
        agent = make_agent()
        agent.store(*make_transition(np.random.default_rng(0), terminated=terminated))

        batch = agent.buffer.sample(1)

        assert batch.dones[0] == expected
