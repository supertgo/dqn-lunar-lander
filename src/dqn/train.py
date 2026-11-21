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
    
