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

    

