from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from dqn.config import (
    RESULTS_DIR,
    EvalConfig,
    RunConfig,
    load_evaluation,
    load_hyperparameters,
)
from dqn.evaluate import PolicyFn, evaluate
from dqn.run_logger import RunLogger, is_complete, run_dir


class _SB3LoggingCallback(BaseCallback):
    def __init__(self, run_logger: RunLogger, eval_cfg: EvalConfig, policy_fn: PolicyFn,) -> None:
        super().__init__()
        self.run_logger = run_logger
        self.eval_cfg = eval_cfg
        self.policy_fn = policy_fn
        self.episode = 0
        self.ep_return = 0.0
        self.ep_length = 0

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self.ep_return += reward
        self.ep_length += 1

        if done:
            self.episode += 1
            epsilon = float(getattr(self.model, "exploration_rate", float("nan")))
            loss = float(self.model.logger.name_to_value.get("train/loss", float("nan")))
            self.run_logger.log_episode(
                step=self.num_timesteps,
                episode=self.episode,
                **{"return": self.ep_return},
                length=self.ep_length,
                epsilon=epsilon,
                loss=loss,
            )
            self.ep_return = 0.0
            self.ep_length = 0

        if self.eval_cfg.eval_freq > 0 and self.num_timesteps % self.eval_cfg.eval_freq == 0:
            eval_res = evaluate(self.policy_fn, self.eval_cfg, self.eval_cfg.eval_episodes)
            self.run_logger.log_eval(
                step=self.num_timesteps,
                mean_return=eval_res["mean_return"],
                std_return=eval_res["std_return"],
                success_rate=eval_res["success_rate"],
                mean_length=eval_res["mean_length"],
            )

        return True
