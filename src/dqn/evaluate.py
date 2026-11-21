from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import gymnasium as gym
import numpy as np

from dqn.config import EvalConfig

PolicyFn = Callable[[np.ndarray], int]


class EvalResult(TypedDict):
    mean_return: float
    std_return: float
    success_rate: float
    mean_length: float
    n_episodes: int


def evaluate(policy: PolicyFn, cfg: EvalConfig, n_episodes: int) -> EvalResult:
    if n_episodes <= 0:
        raise ValueError(f"n_episodes deve ser positivo, recebido: {n_episodes}")

    returns: list[float] = []
    lengths: list[int] = []

    env = gym.make(cfg.env_id)
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=cfg.eval_seed + ep)
            done = False
            ep_return = 0.0
            ep_length = 0

            while not done:
                action = int(policy(np.asarray(obs)))
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                ep_length += 1
                done = bool(terminated or truncated)

            returns.append(ep_return)
            lengths.append(ep_length)
    finally:
        env.close()

    returns_arr = np.asarray(returns, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)

    return EvalResult(
        mean_return=float(returns_arr.mean()),
        std_return=float(returns_arr.std()),
        success_rate=float(np.mean(returns_arr >= cfg.solved_threshold)),
        mean_length=float(lengths_arr.mean()),
        n_episodes=int(n_episodes),
    )
