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
    
