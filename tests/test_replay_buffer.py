import numpy as np
import pytest

from dqn.replay_buffer import ReplayBuffer, Transition


def make_replay_buffer(capacity: int, obs_size: int, seed: int = 42):
    rng = np.random.default_rng(seed=seed)
    replay_buffer = ReplayBuffer(capacity=capacity, obs_size=obs_size, rng=rng)

    return replay_buffer


def mock_transition(obs_size: int) -> Transition:
    obs = np.random.rand(obs_size).astype(np.float32)
    action = np.random.randint(0, 2)
    reward = float(np.random.rand())
    next_obs = np.random.rand(obs_size).astype(np.float32)
    done = bool(np.random.randint(0, 2))

    return obs, action, reward, next_obs, done


def add_mocks_transitions_in_buffer(replay_buffer: ReplayBuffer, quantity_of_transitions: int = 10):

    for _ in range(quantity_of_transitions):
        obs, action, reward, next_obs, done = mock_transition(obs_size=replay_buffer.obs_size)
        replay_buffer.add(obs, action, reward, next_obs, done)


def make_replay_buffer_and_add_transitions(
    capacity: int, obs_size: int, seed: int = 42, quantity_of_transitions: int = 10
):
    replay_buffer = make_replay_buffer(capacity=capacity, obs_size=obs_size, seed=seed)

    add_mocks_transitions_in_buffer(replay_buffer, quantity_of_transitions)

    return replay_buffer


def test_length_of_buffer_zero_or_more():
    replay_buffer = make_replay_buffer(capacity=3, obs_size=3)

    assert len(replay_buffer) == 0

    obs, action, reward, next_obs, done = mock_transition(3)
    replay_buffer.add(obs, action, reward, next_obs, done)

    assert len(replay_buffer) == 1


def test_save_transition_right():
    replay_buffer = make_replay_buffer(capacity=3, obs_size=3)
    obs, action, reward, next_obs, done = mock_transition(3)
    replay_buffer.add(obs, action, reward, next_obs, done)

    expected_transition = Transition(obs, action, reward, next_obs, done)
    actual = replay_buffer.buffer[0]

    np.testing.assert_array_equal(expected_transition.obs, actual.obs)
    assert expected_transition.action == actual.action
    assert expected_transition.reward == actual.reward
    np.testing.assert_array_equal(expected_transition.next_obs, actual.next_obs)
    assert expected_transition.done == actual.done


def test_dont_save_transition_with_wrong_obs_shape():
    replay_buffer = make_replay_buffer(capacity=3, obs_size=3)
    obs, action, reward, next_obs, done = mock_transition(obs_size=2)

    with pytest.raises(ValueError):
        replay_buffer.add(obs, action, reward, next_obs, done)


def test_dont_save_transition_with_wrong_next_obs_shape():
    replay_buffer = make_replay_buffer(capacity=3, obs_size=3)
    obs, action, reward, _, done = mock_transition(obs_size=3)
    next_obs = np.zeros(2, dtype=np.float32)

    with pytest.raises(ValueError):
        replay_buffer.add(obs, action, reward, next_obs, done)


def test_stored_transition_dont_change_if_original_array_change():
    replay_buffer = make_replay_buffer(capacity=3, obs_size=3)
    obs, action, reward, next_obs, done = mock_transition(3)
    replay_buffer.add(obs, action, reward, next_obs, done)

    original_obs = obs.copy()
    original_next_obs = next_obs.copy()
    obs -= 0.01
    next_obs += 0.01
    stored_obs = replay_buffer.buffer[0].obs
    stored_next_obs = replay_buffer.buffer[0].next_obs

    np.testing.assert_array_equal(stored_obs, original_obs)
    np.testing.assert_array_equal(stored_next_obs, original_next_obs)


def test_sample_gives_length_needed():
    replay_buffer = make_replay_buffer_and_add_transitions(
        capacity=3, obs_size=3, quantity_of_transitions=3
    )

    batch = replay_buffer.sample(batch_size=3)

    assert batch.actions.shape == (3,)


def test_sample_returns_batch_with_expected_shapes_and_dtypes():
    replay_buffer = make_replay_buffer_and_add_transitions(
        capacity=3, obs_size=2, quantity_of_transitions= 3
    )

    batch = replay_buffer.sample(batch_size=2)

    assert batch.obs.shape == (2, 2)
    assert batch.obs.dtype == np.float32
    assert batch.actions.shape == (2,)
    assert batch.actions.dtype == np.int64
    assert batch.rewards.shape == (2,)
    assert batch.rewards.dtype == np.float32
    assert batch.next_obs.shape == (2, 2)
    assert batch.next_obs.dtype == np.float32
    assert batch.dones.shape == (2,)
    assert batch.dones.dtype == np.float32
    assert len(np.unique(batch.obs[:, 0])) == 2


def test_buffer_keeps_most_recent_transitions_after_capacity_is_reached():
    replay_buffer = make_replay_buffer(capacity=3, obs_size=1)
    for i in range(5):
        replay_buffer.add(
            obs=np.array([i], dtype=np.float32),
            action=i,
            reward=float(i),
            next_obs=np.array([i + 1], dtype=np.float32),
            done=False,
        )

    batch = replay_buffer.sample(batch_size=3)

    np.testing.assert_array_equal(np.sort(batch.obs[:, 0]), [2, 3, 4])
    assert len(replay_buffer) == 3


def test_sample_fail_if_dont_have_enough_stored_transitions():
    replay_buffer = make_replay_buffer_and_add_transitions(
        capacity=3, obs_size=3, quantity_of_transitions=0
    )

    with pytest.raises(ValueError):
        replay_buffer.sample(batch_size=1)


def test_sample_fail_if_batch_size_is_zero_or_negative():
    replay_buffer = make_replay_buffer_and_add_transitions(
        capacity=3, obs_size=3, quantity_of_transitions=3
    )

    with pytest.raises(ValueError):
        replay_buffer.sample(batch_size=0)

    with pytest.raises(ValueError):
        replay_buffer.sample(batch_size=-1)


def test_replay_buffer_rejects_non_positive_dimensions():
    with pytest.raises(ValueError):
        make_replay_buffer(capacity=0, obs_size=3)

    with pytest.raises(ValueError):
        make_replay_buffer(capacity=3, obs_size=0)


def test_sample_rejects_non_integer_batch_size():
    replay_buffer = make_replay_buffer_and_add_transitions(
        capacity=3, obs_size=3, quantity_of_transitions=3
    )

    with pytest.raises(TypeError):
        replay_buffer.sample(batch_size=1.5)
