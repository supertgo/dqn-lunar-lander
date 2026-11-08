from __future__ import annotations

from pathlib import Path
from typing import Any

from dqn.config import RunConfig
from dqn.evaluate import EvalResult

TRAIN_COLUMNS = ["step", "episode", "return", "length", "epsilon", "loss"]
EVAL_COLUMNS = ["step", "mean_return", "std_return", "success_rate", "mean_length"]


def run_dir(cfg: RunConfig, results_dir: Path) -> Path:
    """results_dir / cfg.run_id"""
    raise NotImplementedError


def is_complete(run_dir: Path) -> bool:
    """True se summary.json existe."""
    raise NotImplementedError


class RunLogger:
    def __init__(self, cfg: RunConfig, results_dir: Path):
        """Cria o diretório da run e grava config.json e meta.json."""
        raise NotImplementedError

    def log_episode(self, **row: Any) -> None:
        """Uma linha de train.csv; chaves = TRAIN_COLUMNS."""
        raise NotImplementedError

    def log_eval(self, **row: Any) -> None:
        """Uma linha de eval.csv; chaves = EVAL_COLUMNS."""
        raise NotImplementedError

    def finish(self, final_eval: EvalResult) -> dict[str, Any]:
        """Grava summary.json (final_eval + wall_time_s), fecha os arquivos e retorna o summary."""
        raise NotImplementedError
