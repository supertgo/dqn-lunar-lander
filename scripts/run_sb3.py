from __future__ import annotations

import json

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from dqn.config import (
    RESULTS_DIR,
    EvalConfig,
    RunConfig,
    load_evaluation,
    load_hyperparameters,
)
from dqn.evaluate import PolicyFn, evaluate
from dqn.run_logger import RunLogger, is_complete, run_dir


class _SB3LoggingCallback(BaseCallback):
    def __init__(self, run_logger: RunLogger, eval_cfg: EvalConfig, policy_fn: PolicyFn,) -> None:
        super().__init__()
        self.run_logger = run_logger
        self.eval_cfg = eval_cfg
        self.policy_fn = policy_fn
        self.episode = 0
        self.ep_return = 0.0
        self.ep_length = 0
        self.best_eval_return = float("-inf")

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self.ep_return += reward
        self.ep_length += 1

        if done:
            self.episode += 1
            epsilon = float(getattr(self.model, "exploration_rate", float("nan")))
            loss = float(self.model.logger.name_to_value.get("train/loss", float("nan")))
            self.run_logger.log_episode(
                step=self.num_timesteps,
                episode=self.episode,
                **{"return": self.ep_return},
                length=self.ep_length,
                epsilon=epsilon,
                loss=loss,
            )
            self.ep_return = 0.0
            self.ep_length = 0

        if self.eval_cfg.eval_freq > 0 and self.num_timesteps % self.eval_cfg.eval_freq == 0:
            eval_res = evaluate(self.policy_fn, self.eval_cfg, self.eval_cfg.eval_episodes)
            self.run_logger.log_eval(
                step=self.num_timesteps,
                mean_return=eval_res["mean_return"],
                std_return=eval_res["std_return"],
                success_rate=eval_res["success_rate"],
                mean_length=eval_res["mean_length"],
            )
            # Análise extra: mesmo critério de melhor checkpoint do nosso DQN.
            if eval_res["mean_return"] > self.best_eval_return:
                self.best_eval_return = eval_res["mean_return"]
                self.model.save(self.run_logger.dir / "best_model.zip")
                (self.run_logger.dir / "best_checkpoint.json").write_text(
                    json.dumps({"step": self.num_timesteps, "eval_mean_return": self.best_eval_return})
                )

        return True
        
def train_sb3(cfg: RunConfig, results_dir: Path) -> dict[str, Any]:
    if cfg.algo != "sb3":
        raise ValueError(f"Esperado algo='sb3', recebido: {cfg.algo!r}")

    total_timesteps = int(cfg.hyperparameters["total_timesteps"])
    eval_cfg = cfg.evaluation

    env = gym.make(eval_cfg.env_id)
    try:
        model = DQN("MlpPolicy", env, seed=cfg.seed, verbose=0)

        def greedy_policy(obs: np.ndarray) -> int:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        with RunLogger(cfg, results_dir) as logger:
            callback = _SB3LoggingCallback(logger, eval_cfg, greedy_policy)
            model.learn(total_timesteps=total_timesteps, callback=callback)
            model.save(logger.dir / "model.zip")
            final_eval = evaluate(greedy_policy, eval_cfg, eval_cfg.final_eval_episodes)
            return logger.finish(final_eval)
    finally:
        env.close()

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o baseline DQN do Stable Baselines3 (configuração padrão).")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Lista de seeds para execução (padrão: 0 1 2).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total de passos de treino (padrão: total_timesteps de configs/hyperparameters.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Diretório de saída das runs (padrão: results/runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reexecuta mesmo se summary.json já existir.",
    )
    return parser.parse_args()
    
def main() -> None:
    args = _parse_args()
    total_timesteps = (
        args.total_timesteps
        if args.total_timesteps is not None
        else load_hyperparameters().total_timesteps
    )
    eval_cfg = load_evaluation()

    for seed in args.seeds:
        cfg = RunConfig(
            algo="sb3",
            hyperparameters={"total_timesteps": int(total_timesteps)},
            evaluation=eval_cfg,
            seed=int(seed),
        )
        out_dir = run_dir(cfg, args.results_dir)
        if not args.force and is_complete(out_dir):
            print(f"[skip] {cfg.run_id} já concluída.")
            continue

        print(f"[run]  {cfg.run_id} (total_timesteps={total_timesteps})...")
        summary = train_sb3(cfg, args.results_dir)
        print(
            f"[done] {cfg.run_id} | mean_return={summary['mean_return']:.2f} "
            f"| success_rate={summary['success_rate']:.2%} "
            f"| wall_time={summary['wall_time_s']:.1f}s"
        )


if __name__ == "__main__":
    main()

