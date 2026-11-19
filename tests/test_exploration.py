import numpy as np
import pytest

from dqn.exploration import LinearEpsilon, epsilon_greedy


def test_linear_epsilon_decreases_and_stays_at_end():
    get_epsilon = LinearEpsilon(start=10.0, end=2.0, decay_steps=2)

    assert get_epsilon(step=0) == 10.0
    assert get_epsilon(step=1) == 6.0
    assert get_epsilon(step=2) == 2.0
    assert get_epsilon(step=5) == 2.0


def test_linear_epsilon_increases_when_end_is_larger():
    get_epsilon = LinearEpsilon(start=0.0, end=1.0, decay_steps=4)

    assert get_epsilon(step=2) == 0.5


@pytest.mark.parametrize("decay_steps", [0, -1])
def test_linear_epsilon_rejects_non_positive_decay_steps(decay_steps):
    with pytest.raises(ValueError):
        LinearEpsilon(start=1.0, end=0.0, decay_steps=decay_steps)


@pytest.mark.parametrize("decay_steps", [1.5, True])
def test_linear_epsilon_rejects_non_integer_decay_steps(decay_steps):
    with pytest.raises(TypeError):
        LinearEpsilon(start=1.0, end=0.0, decay_steps=decay_steps)


def test_linear_epsilon_rejects_negative_step():
    get_epsilon = LinearEpsilon(start=1.0, end=0.0, decay_steps=2)

    with pytest.raises(ValueError):
        get_epsilon(step=-1)


def test_exploitation_epsilon_greedy():
    q_values = np.array([0.0, 1.0, 2.0])

    best_action = epsilon_greedy(
        q_values=q_values,
        epsilon=0.0,
        rng=np.random.default_rng(seed=42),
    )

    assert best_action == 2


def test_exploration_epsilon_greedy():
    q_values = np.array([0.0, 1.0, 2.0])

    best_action = epsilon_greedy(
        q_values=q_values,
        epsilon=1.0,
        rng=np.random.default_rng(seed=42),
    )

    assert best_action == 1


def test_epsilon_greedy_returns_first_action_when_q_values_are_tied():
    action = epsilon_greedy(
        q_values=np.array([3.0, 3.0, 1.0]),
        epsilon=0.0,
        rng=np.random.default_rng(seed=42),
    )

    assert action == 0


def test_epsilon_greedy_accepts_single_action():
    action = epsilon_greedy(
        q_values=np.array([5.0]),
        epsilon=1.0,
        rng=np.random.default_rng(seed=42),
    )

    assert action == 0


@pytest.mark.parametrize("epsilon", [-0.01, 1.01, float("nan")])
def test_epsilon_greedy_rejects_epsilon_outside_valid_range(epsilon):
    with pytest.raises(ValueError):
        epsilon_greedy(
            q_values=np.array([0.0, 1.0]),
            epsilon=epsilon,
            rng=np.random.default_rng(seed=42),
        )


@pytest.mark.parametrize("q_values", [np.array([]), np.array([[0.0, 1.0]])])
def test_epsilon_greedy_rejects_empty_or_non_vector_q_values(q_values):
    with pytest.raises(ValueError):
        epsilon_greedy(
            q_values=q_values,
            epsilon=0.0,
            rng=np.random.default_rng(seed=42),
        )
