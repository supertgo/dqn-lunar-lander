from __future__ import annotations

from pathlib import Path
from typing import Any

from dqn.config import RunConfig


def train(cfg: RunConfig, results_dir: Path) -> dict[str, Any]:
    raise NotImplementedError
