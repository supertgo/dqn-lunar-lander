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

            for step in range(1, hp.total_timesteps + 1):
                epsilon = epsilon_schedule(step - 1)
                action = agent.act(obs, epsilon)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32)

                agent.store(obs, action, float(reward), next_obs, bool(terminated))

                ep_return += float(reward)
                ep_length += 1

                if (
                    step >= hp.learning_starts
                    and step % hp.train_freq == 0
                    and len(agent.buffer) >= hp.batch_size
                ):
                    ep_losses.append(agent.update())

                if step % hp.target_update_interval == 0:
                    agent.sync_target()

                
            agent.save(logger.dir / "model.pt")
            final_eval = evaluate(greedy_policy, eval_cfg, eval_cfg.final_eval_episodes)
            return logger.finish(final_eval)

    finally:
        env.close()

