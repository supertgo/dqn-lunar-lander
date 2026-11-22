"""Análise extra: avalia o modelo do último passo e o melhor checkpoint de cada run.

O melhor checkpoint é escolhido pelas avaliações intermediárias (eval_seed do
protocolo, poucos episódios). Para não medir essa escolha com os mesmos episódios,
os dois modelos são avaliados aqui em episódios novos (--eval-seed, padrão 20000).
O resultado vai para checkpoints.json de cada run; summary.json não é alterado.

Uso: uv run python scripts/eval_checkpoints.py results/extra/best_checkpoint --workers 8
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from dqn.config import EvalConfig
from dqn.evaluate import PolicyFn, evaluate

MODEL_FILES = {"dqn": ("model.pt", "best_model.pt"), "sb3": ("model.zip", "best_model.zip")}


def _dqn_policy(path: Path, hp: dict[str, Any], obs_dim: int, n_actions: int) -> PolicyFn:
    import torch

    from dqn.network import QNetwork

    q = QNetwork(obs_dim, n_actions, tuple(int(h) for h in hp["hidden_sizes"]))
    q.load_state_dict(torch.load(path, map_location="cpu"))
    q.eval()

    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax())

    return policy


def _sb3_policy(path: Path) -> PolicyFn:
    from stable_baselines3 import DQN

    model = DQN.load(path, device="cpu")

    def policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return policy


def _policy(algo: str, path: Path, cfg: dict[str, Any], eval_cfg: EvalConfig) -> PolicyFn:
    if algo == "sb3":
        return _sb3_policy(path)
    import gymnasium as gym

    env = gym.make(eval_cfg.env_id)
    try:
        assert env.observation_space.shape is not None
        obs_dim, n_actions = int(env.observation_space.shape[0]), int(env.action_space.n)
    finally:
        env.close()
    return _dqn_policy(path, cfg["hyperparameters"], obs_dim, n_actions)


def evaluate_run(run: Path, eval_seed: int = 20000, n_episodes: int = 100) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    cfg = json.loads((run / "config.json").read_text())
    algo = cfg["algo"]
    eval_cfg = replace(EvalConfig(**cfg["evaluation"]), eval_seed=eval_seed)
    last_file, best_file = MODEL_FILES[algo]
    best = json.loads((run / "best_checkpoint.json").read_text())

    out: dict[str, Any] = {
        "run_id": cfg["run_id"],
        "config_id": cfg["config_id"],
        "algo": algo,
        "seed": cfg["seed"],
        "eval_seed": eval_seed,
        "best_step": best["step"],
        "best_selection_return": best["eval_mean_return"],
    }
    for key, name in (("last", last_file), ("best", best_file)):
        out[key] = dict(evaluate(_policy(algo, run / name, cfg, eval_cfg), eval_cfg, n_episodes))
    (run / "checkpoints.json").write_text(json.dumps(out, indent=2))
    return out


def pending_runs(results_dir: Path, force: bool = False) -> list[Path]:
    runs = []
    for run in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if not (run / "summary.json").is_file() or not (run / "best_checkpoint.json").is_file():
            continue
        if force or not (run / "checkpoints.json").is_file():
            runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia último modelo e melhor checkpoint em episódios novos.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--eval-seed", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs = pending_runs(args.results_dir, args.force)
    print(f"{len(runs)} run(s) para avaliar em {args.results_dir}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_run, r, args.eval_seed, args.episodes) for r in runs]
        for fut in futures:
            o = fut.result()
            print(
                f"[ok] {o['run_id']}: último {o['last']['mean_return']:.1f} | "
                f"melhor (passo {o['best_step']}) {o['best']['mean_return']:.1f}"
            )


if __name__ == "__main__":
    main()
