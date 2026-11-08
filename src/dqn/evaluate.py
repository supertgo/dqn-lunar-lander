from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

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
    raise NotImplementedError
