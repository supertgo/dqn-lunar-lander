from __future__ import annotations

import argparse
import csv
import json
import random
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dqn.config import RunConfig, Sweep, load_sweep

TRAIN_COLUMNS = ["step", "episode", "return", "length", "epsilon", "loss"]
EVAL_COLUMNS = ["step", "mean_return", "std_return", "success_rate", "mean_length"]


def _train(cfg: RunConfig, results_dir: Path) -> dict[str, Any]:
    """Mock temporário do treino; substitua por dqn.train.train na integração."""
    run_dir = results_dir / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "git_commit": "mock",
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    hp = cfg.hyperparameters
    total_steps = max(1, int(hp.get("total_timesteps", 1)))
    n_episodes = min(10, total_steps)
    epsilon_start = float(hp.get("epsilon_start", 1.0))
    epsilon_end = float(hp.get("epsilon_end", 0.05))
    decay_steps = max(1, int(hp.get("epsilon_decay_steps", total_steps)))
    rng = random.Random(cfg.seed)

    train_rows = []
    for episode in range(1, n_episodes + 1):
        step = max(1, total_steps * episode // n_episodes)
        progress = episode / n_episodes
        train_rows.append(
            {
                "step": step,
                "episode": episode,
                "return": -200.0 + 400.0 * progress + rng.uniform(-20.0, 20.0),
                "length": rng.randint(100, 500),
                "epsilon": max(
                    epsilon_end,
                    epsilon_start
                    + (epsilon_end - epsilon_start) * min(step / decay_steps, 1.0),
                ),
                "loss": max(0.0, 1.0 / episode + rng.uniform(0.0, 0.05)),
            }
        )

    with (run_dir / "train.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAIN_COLUMNS)
        writer.writeheader()
        writer.writerows(train_rows)

    eval_freq = max(1, int(cfg.evaluation.eval_freq))
    eval_steps = list(range(eval_freq, total_steps + 1, eval_freq))
    eval_steps = eval_steps[:10]
    eval_rows = [
        {
            "step": step,
            "mean_return": -200.0 + 400.0 * step / total_steps,
            "std_return": 20.0,
            "success_rate": min(1.0, step / total_steps),
            "mean_length": 300.0,
        }
        for step in eval_steps
    ]
    with (run_dir / "eval.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        writer.writerows(eval_rows)

    final_eval = {
        "mean_return": float(train_rows[-1]["return"]),
        "std_return": 20.0,
        "success_rate": 1.0,
        "mean_length": 300.0,
        "n_episodes": int(cfg.evaluation.final_eval_episodes),
        "wall_time_s": 0.0,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(final_eval, indent=2),
        encoding="utf-8",
    )
    return final_eval


def _run_one(cfg: RunConfig, results_dir: Path) -> dict[str, Any]:
    return _train(cfg, results_dir)


def run_sweep(sweep: Sweep, workers: int = 1) -> list[dict[str, Any]]:
    """Execute cada configuração pendente do sweep e retorna os summaries."""
    if workers < 1:
        raise ValueError("workers deve ser pelo menos 1")

    sweep.results_dir.mkdir(parents=True, exist_ok=True)

    expanded_configs = sweep.run_configs()
    if not expanded_configs:
        raise ValueError(
            f"Sweep {sweep.name!r} não gera nenhuma run; configure seeds e variants."
        )

    configs: list[RunConfig] = []
    seen_run_ids: set[str] = set()
    for cfg in expanded_configs:
        if cfg.run_id in seen_run_ids:
            continue
        seen_run_ids.add(cfg.run_id)

        run_dir = sweep.results_dir / cfg.run_id
        if (run_dir / "summary.json").is_file():
            print(f"[skip] {cfg.run_id} já concluída", flush=True)
            continue
        configs.append(cfg)

    print(
        f"Sweep {sweep.name}: {len(configs)} run(s) pendente(s), "
        f"{workers} worker(s).",
        flush=True,
    )

    if workers == 1:
        summaries: list[dict[str, Any]] = []
        for index, cfg in enumerate(configs, start=1):
            print(f"[{index}/{len(configs)}] Iniciando {cfg.run_id}", flush=True)
            summary = _run_one(cfg, sweep.results_dir)
            summaries.append(summary)
            print(f"[{index}/{len(configs)}] Concluída {cfg.run_id}", flush=True)
        return summaries

    summaries_by_id: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_configs = {
            executor.submit(_run_one, cfg, sweep.results_dir): cfg for cfg in configs
        }
        for future in as_completed(future_configs):
            cfg = future_configs[future]
            summary = future.result()
            summaries_by_id[cfg.run_id] = summary
            print(f"[concluída] {cfg.run_id}", flush=True)

    return [summaries_by_id[cfg.run_id] for cfg in configs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Executa sequencialmente (ou em paralelo) as runs de um sweep."
    )
    parser.add_argument(
        "sweep",
        type=Path,
        help="caminho do arquivo YAML do sweep",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="número de treinos simultâneos (padrão: 1, sequencial)",
    )
    args = parser.parse_args(argv)

    sweep = load_sweep(args.sweep)
    run_sweep(sweep, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())