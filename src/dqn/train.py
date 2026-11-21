from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from dqn.agent import DQNAgent
from dqn.config import RunConfig
from dqn.evaluate import evaluate
from dqn.exploration import LinearEpsilon
from dqn.run_logger import RunLogger


def train(cfg: RunConfig, results_dir: Path) -> dict[str, Any]:
    hp = cfg.hp
    eval_cfg = cfg.evaluation
    rng = np.random.default_rng(cfg.seed)

    env = gym.make(eval_cfg.env_id)
    try:
        assert env.observation_space.shape is not None
        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)

        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, hp=hp, rng=rng)
        epsilon_schedule = LinearEpsilon(start=hp.epsilon_start, end=hp.epsilon_end,decay_steps=hp.epsilon_decay_steps,)

        def greedy_policy(obs: np.ndarray) -> int:
            return agent.act(obs, epsilon=0.0)

        with RunLogger(cfg, results_dir) as logger:
            obs, _ = env.reset(seed=cfg.seed)
            obs = np.asarray(obs, dtype=np.float32)

            episode = 0
            ep_return = 0.0
            ep_length = 0
            ep_losses: list[float] = []

            
    finally:
        env.close()

