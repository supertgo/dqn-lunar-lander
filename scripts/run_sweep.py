from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dqn.config import RunConfig, Sweep, load_sweep
from dqn.train import train


def _train(cfg: RunConfig, results_dir: Path) -> dict[str, Any]:
    """Treina uma run com dqn.train.train usando 1 thread do torch.

    O paralelismo vem de rodar várias runs em processos (--workers); sem limitar as
    threads, cada processo tentaria usar todos os núcleos e eles disputariam a CPU.
    """
    import torch

    torch.set_num_threads(1)
    return train(cfg, results_dir)


def _run_one(cfg: RunConfig, results_dir: Path) -> dict[str, Any]:
    return _train(cfg, results_dir)


def run_sweep(sweep: Sweep | _CombinedSweep, workers: int = 1) -> list[dict[str, Any]]:
    """Execute cada configuração pendente do sweep e retorna os summaries."""
    if workers < 1:
        raise ValueError("workers deve ser pelo menos 1")

    sweep.results_dir.mkdir(parents=True, exist_ok=True)

    expanded_configs = sweep.run_configs()
    if not expanded_configs:
        raise ValueError(f"Sweep {sweep.name!r} não gera nenhuma run; configure seeds e variants.")

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
        f"Sweep {sweep.name}: {len(configs)} run(s) pendente(s), {workers} worker(s).",
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
        future_configs = {executor.submit(_run_one, cfg, sweep.results_dir): cfg for cfg in configs}
        for future in as_completed(future_configs):
            cfg = future_configs[future]
            summary = future.result()
            summaries_by_id[cfg.run_id] = summary
            print(f"[concluída] {cfg.run_id}", flush=True)

    return [summaries_by_id[cfg.run_id] for cfg in configs]


@dataclass(frozen=True)
class _CombinedSweep:
    """Vários sweeps numa fila só: os workers não ficam ociosos entre um sweep e outro."""

    sweeps: list[Sweep]

    @property
    def name(self) -> str:
        return "+".join(sweep.name for sweep in self.sweeps)

    @property
    def results_dir(self) -> Path:
        return self.sweeps[0].results_dir

    def run_configs(self) -> list[RunConfig]:
        return [cfg for sweep in self.sweeps for cfg in sweep.run_configs()]


def _combine(sweeps: list[Sweep]) -> Sweep | _CombinedSweep:
    if len(sweeps) == 1:
        return sweeps[0]
    if len({sweep.results_dir for sweep in sweeps}) > 1:
        raise ValueError("Sweeps combinados precisam do mesmo results_dir")
    return _CombinedSweep(sweeps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Executa sequencialmente (ou em paralelo) as runs de um sweep."
    )
    parser.add_argument(
        "sweeps",
        type=Path,
        nargs="+",
        help="arquivo(s) YAML de sweep; vários rodam numa fila só, sem repetir runs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="número de treinos simultâneos (padrão: 1, sequencial)",
    )
    args = parser.parse_args(argv)

    sweep = _combine([load_sweep(path) for path in args.sweeps])
    run_sweep(sweep, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
