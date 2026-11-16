from __future__ import annotations

import copy
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dqn.config import Hyperparameters
from dqn.network import QNetwork
from dqn.replay_buffer import Batch


def _epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    # Trocar por `from dqn.exploration import epsilon_greedy`
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


class _ListReplayBuffer:
    # TODO(integração): trocar por `from dqn.replay_buffer import ReplayBuffer`
    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator) -> None:
        self.memory = deque(maxlen=capacity)
        self.rng = rng

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        self.memory.append((obs, action, reward, next_obs, terminated))

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(len(self.memory), size=batch_size)
        transitions = [self.memory[i] for i in idx]

        return Batch(
            obs=np.array([t[0] for t in transitions], dtype=np.float32),
            actions=np.array([t[1] for t in transitions], dtype=np.int64),
            rewards=np.array([t[2] for t in transitions], dtype=np.float32),
            next_obs=np.array([t[3] for t in transitions], dtype=np.float32),
            dones=np.array([t[4] for t in transitions], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hp: Hyperparameters,
        rng: np.random.Generator,
    ):
        self.n_actions = n_actions
        self.hp = hp
        self.rng = rng

        torch.manual_seed(int(rng.integers(2**31)))

        self.q = QNetwork(obs_dim, n_actions, hp.hidden_sizes)
        self.q_target = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=hp.learning_rate)
        self.buffer = _ListReplayBuffer(hp.buffer_size, obs_dim, rng)

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.q(obs_t).numpy()

        return _epsilon_greedy(q_values, epsilon, self.rng)

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, terminated)

    def update(self) -> float:
        """
        Alvo: y = r + γ · (1 − done) · max_a' Q_alvo(s', a')
        """
        batch = self.buffer.sample(self.hp.batch_size)
        obs = torch.as_tensor(batch.obs)
        actions = torch.as_tensor(batch.actions)
        rewards = torch.as_tensor(batch.rewards)
        dones = torch.as_tensor(batch.dones)
        next_obs = torch.as_tensor(batch.next_obs)

        q_all = self.q(obs)
        q_sa = q_all[torch.arange(len(actions)), actions]

        with torch.no_grad():
            next_q = self.q_target(next_obs).max(dim=1).values
            y = rewards + self.hp.gamma * (1 - dones) * next_q

        loss = F.smooth_l1_loss(q_sa, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def sync_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str | Path) -> None:
        torch.save(self.q.state_dict(), path)
