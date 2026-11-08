from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
HYPERPARAMETERS_FILE = CONFIGS_DIR / "hyperparameters.yaml"
EVALUATION_FILE = CONFIGS_DIR / "evaluation.yaml"
RESULTS_DIR = ROOT / "results" / "runs"


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    gamma: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    total_timesteps: int
    train_freq: int
    hidden_sizes: tuple[int, ...]


@dataclass(frozen=True)
class EvalConfig:
    env_id: str
    eval_freq: int
    eval_episodes: int
    final_eval_episodes: int
    eval_seed: int
    solved_threshold: float


@dataclass(frozen=True)
class RunConfig:
    algo: str  # "dqn" ou "sb3"
    hyperparameters: dict[str, Any]  # dqn: campos de Hyperparameters; sb3: {"total_timesteps": N}
    evaluation: EvalConfig
    seed: int

    @property
    def hp(self) -> Hyperparameters:
        raise NotImplementedError

    @property
    def config_id(self) -> str:
        raise NotImplementedError

    @property
    def run_id(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class Sweep:
    name: str
    seeds: list[int]
    variants: list[dict[str, Any]]  # cada variant sobrescreve parte de hyperparameters.yaml
    evaluation: dict[str, Any]  # overrides opcionais de evaluation.yaml
    results_dir: Path

    def run_configs(self) -> list[RunConfig]:
        """Uma RunConfig por (variant, seed)."""
        raise NotImplementedError


def load_hyperparameters(overrides: dict[str, Any] | None = None) -> Hyperparameters:
    raise NotImplementedError


def load_evaluation(overrides: dict[str, Any] | None = None) -> EvalConfig:
    raise NotImplementedError


def dqn_run_config(
    seed: int,
    hp_overrides: dict[str, Any] | None = None,
    eval_overrides: dict[str, Any] | None = None,
) -> RunConfig:
    raise NotImplementedError


def load_sweep(path: str | Path) -> Sweep:
    raise NotImplementedError
