from __future__ import annotations

import csv
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any

from dqn.config import RunConfig
from dqn.evaluate import EvalResult

TRAIN_COLUMNS = ["step", "episode", "return", "length", "epsilon", "loss"]
EVAL_COLUMNS = ["step", "mean_return", "std_return", "success_rate", "mean_length"]


def run_dir(cfg: RunConfig, results_dir: Path) -> Path:
    """results_dir / cfg.run_id"""
    return Path(results_dir) / cfg.run_id


def is_complete(run_dir: Path) -> bool:
    """True se summary.json existe."""
    return (Path(run_dir) / "summary.json").is_file()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str | None]:
    packages = ["torch", "numpy", "gymnasium", "pandas", "pyyaml", "stable-baselines3"]
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(tmp_path, path)


class RunLogger:
    def __init__(self, cfg: RunConfig, results_dir: Path):
        """Cria o diretório da run e grava config.json e meta.json."""
        self.cfg = cfg
        self.dir = run_dir(cfg, results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()

        _write_json_atomic(self.dir / "config.json", cfg.to_dict())

        meta = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "versions": _package_versions(),
        }
        _write_json_atomic(self.dir / "meta.json", meta)

        self._train_file = (self.dir / "train.csv").open("w", newline="", encoding="utf-8")
        self._train_writer = csv.DictWriter(self._train_file, fieldnames=TRAIN_COLUMNS)
        self._train_writer.writeheader()
        self._train_file.flush()

        self._eval_file = (self.dir / "eval.csv").open("w", newline="", encoding="utf-8")
        self._eval_writer = csv.DictWriter(self._eval_file, fieldnames=EVAL_COLUMNS)
        self._eval_writer.writeheader()
        self._eval_file.flush()

        self._closed = False

    def _validate_row(self, row: dict[str, Any], expected_columns: list[str]) -> None:
        keys = set(row.keys())
        expected = set(expected_columns)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(f"Colunas inválidas: faltando={missing}, extras={extra}")

    def log_episode(self, **row: Any) -> None:
        """Uma linha de train.csv; chaves = TRAIN_COLUMNS."""
        self._validate_row(row, TRAIN_COLUMNS)
        self._train_writer.writerow(row)
        self._train_file.flush()

    def log_eval(self, **row: Any) -> None:
        """Uma linha de eval.csv; chaves = EVAL_COLUMNS."""
        self._validate_row(row, EVAL_COLUMNS)
        self._eval_writer.writerow(row)
        self._eval_file.flush()

    def close(self) -> None:
        """Fecha os arquivos CSV sem gravar summary.json."""
        if not self._closed:
            self._train_file.close()
            self._eval_file.close()
            self._closed = True

    def finish(self, final_eval: EvalResult) -> dict[str, Any]:
        """Grava summary.json (final_eval + wall_time_s), fecha os arquivos e retorna o summary."""
        self.close()
        wall_time_s = time.perf_counter() - self._start_time
        summary: dict[str, Any] = {
            **final_eval,
            "wall_time_s": round(wall_time_s, 3),
        }
        _write_json_atomic(self.dir / "summary.json", summary)
        return summary

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
