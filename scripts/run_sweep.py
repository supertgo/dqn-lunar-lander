from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dqn.config import RunConfig, Sweep, load_sweep
from dqn.train import train


def _train(cfg: RunConfig, results_dir: Path) -> dict[str, Any]:
    return train(cfg, results_dir)


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