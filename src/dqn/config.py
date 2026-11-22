from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

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
    # Estabilização opcional (None = desligado). Desligados, não entram no config_id.
    max_grad_norm: float | None = None  # clipping da norma do gradiente
    learning_rate_end: float | None = None  # lr decai linearmente até este valor


OPTIONAL_HYPERPARAMETERS = ("max_grad_norm", "learning_rate_end")


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
        if self.algo != "dqn":
            raise ValueError(f"Hyperparameters do DQN indisponíveis para algo={self.algo!r}")
        values = dict(self.hyperparameters)

        if "hidden_sizes" in values:
            values["hidden_sizes"] = tuple(int(size) for size in values["hidden_sizes"])

        return Hyperparameters(**values)

    @property
    def config_id(self) -> str:
        payload = {
            "algo": self.algo,
            "hyperparameters": {
                k: v
                for k, v in self.hyperparameters.items()
                if not (k in OPTIONAL_HYPERPARAMETERS and v is None)
            },
            "evaluation": asdict(self.evaluation),
        }

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:6]

        return f"{self.algo}-{digest}"

    @property
    def run_id(self) -> str:
        return f"{self.config_id}_s{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "hyperparameters": dict(self.hyperparameters),
            "evaluation": asdict(self.evaluation),
            "seed": self.seed,
            "config_id": self.config_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class Sweep:
    name: str
    seeds: list[int]
    variants: list[dict[str, Any]]  # cada variant sobrescreve parte de hyperparameters.yaml
    evaluation: dict[str, Any]  # overrides opcionais de evaluation.yaml
    results_dir: Path

    def run_configs(self) -> list[RunConfig]:
        """Uma RunConfig por (variant, seed)."""
        configs: list[RunConfig] = []

        for variant in self.variants:
            for seed in self.seeds:
                hp = load_hyperparameters(variant)
                evaluation = load_evaluation(self.evaluation)
                configs.append(
                    RunConfig(
                        algo="dqn",
                        hyperparameters=asdict(hp),
                        evaluation=evaluation,
                        seed=int(seed),
                    )
                )

        return configs


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Arquivo YAML deve conter um mapeamento: {path}")

    return data


def load_hyperparameters(overrides: dict[str, Any] | None = None) -> Hyperparameters:
    values = _load_yaml(HYPERPARAMETERS_FILE)
    if overrides:
        values.update(overrides)

    if "hidden_sizes" in values:
        values["hidden_sizes"] = tuple(int(size) for size in values["hidden_sizes"])

    try:
        return Hyperparameters(**values)
    except TypeError as exc:
        raise ValueError("Hiperparâmetros incompletos ou inválidos") from exc


def load_evaluation(overrides: dict[str, Any] | None = None) -> EvalConfig:
    values = _load_yaml(EVALUATION_FILE)
    if overrides:
        values.update(overrides)

    try:
        return EvalConfig(**values)
    except TypeError as exc:
        raise ValueError("Configuração de avaliação incompleta ou inválida") from exc


def dqn_run_config(
    seed: int,
    hp_overrides: dict[str, Any] | None = None,
    eval_overrides: dict[str, Any] | None = None,
) -> RunConfig:
    hp = load_hyperparameters(hp_overrides)
    evaluation = load_evaluation(eval_overrides)
    
    return RunConfig(
        algo="dqn",
        hyperparameters=asdict(hp),
        evaluation=evaluation,
        seed=int(seed),
    )


def load_sweep(path: str | Path) -> Sweep:
    sweep_path = Path(path)
    data = _load_yaml(sweep_path)

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Sweep sem nome válido: {sweep_path}")

    seeds = data.get("seeds")
    if seeds is None:
        seeds = []
    if not isinstance(seeds, list):
        raise ValueError(f"'seeds' deve ser uma lista: {sweep_path}")

    variants = data.get("variants")
    if variants is None:
        variants = []
    if not isinstance(variants, list):
        raise ValueError(f"'variants' deve ser uma lista: {sweep_path}")
    if not all(isinstance(variant, dict) for variant in variants):
        raise ValueError(f"'variants' deve conter apenas mapeamentos: {sweep_path}")

    evaluation = data.get("evaluation")
    if evaluation is None:
        evaluation = {}
    if not isinstance(evaluation, dict):
        raise ValueError(f"'evaluation' deve ser um mapeamento: {sweep_path}")

    raw_results_dir = data.get("results_dir")
    if raw_results_dir:
        results_dir = Path(raw_results_dir)
        if not results_dir.is_absolute():
            results_dir = ROOT / results_dir
    else:
        results_dir = RESULTS_DIR

    return Sweep(
        name=name,
        seeds=[int(seed) for seed in seeds],
        variants=variants,
        evaluation=evaluation,
        results_dir=results_dir,
    )
