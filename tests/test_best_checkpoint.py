"""Análise extra: melhor checkpoint vs. modelo do último passo.

O protocolo oficial (summary.json) continua sendo o do último modelo; estes testes
garantem que o treino também guarda o melhor checkpoint e que eval_checkpoints
avalia os dois em episódios separados dos usados para escolher o checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import eval_checkpoints
from dqn.config import dqn_run_config
from dqn.train import train

TINY_HP = {
    "total_timesteps": 600,
    "learning_starts": 100,
    "batch_size": 16,
    "buffer_size": 1000,
    "epsilon_decay_steps": 300,
    "target_update_interval": 100,
    "hidden_sizes": [8],
}
TINY_EVAL = {"eval_freq": 200, "eval_episodes": 1, "final_eval_episodes": 1}


def _tiny_run(tmp_path: Path) -> Path:
    cfg = dqn_run_config(seed=0, hp_overrides=TINY_HP, eval_overrides=TINY_EVAL)
    train(cfg, tmp_path)
    return tmp_path / cfg.run_id


def test_train_saves_best_checkpoint_at_an_intermediate_eval(tmp_path: Path):
    run = _tiny_run(tmp_path)

    assert (run / "model.pt").is_file()
    assert (run / "best_model.pt").is_file()
    best = json.loads((run / "best_checkpoint.json").read_text())
    assert best["step"] in {200, 400, 600}
    evals = [line.split(",") for line in (run / "eval.csv").read_text().splitlines()[1:]]
    assert best["eval_mean_return"] == max(float(row[1]) for row in evals)


def test_train_summary_still_uses_last_model(tmp_path: Path):
    run = _tiny_run(tmp_path)

    summary = json.loads((run / "summary.json").read_text())
    assert summary["n_episodes"] == TINY_EVAL["final_eval_episodes"]


def test_eval_checkpoints_evaluates_last_and_best_on_held_out_episodes(tmp_path: Path):
    run = _tiny_run(tmp_path)

    out = eval_checkpoints.evaluate_run(run, eval_seed=20000, n_episodes=2)

    saved = json.loads((run / "checkpoints.json").read_text())
    assert saved == out
    assert saved["eval_seed"] == 20000
    assert saved["best_step"] == json.loads((run / "best_checkpoint.json").read_text())["step"]
    for key in ("last", "best"):
        assert saved[key]["n_episodes"] == 2
        assert set(saved[key]) >= {"mean_return", "success_rate"}


def test_eval_checkpoints_skips_runs_without_best_model(tmp_path: Path):
    run = tmp_path / "dqn-old_s0"
    run.mkdir()
    (run / "summary.json").write_text("{}")

    assert eval_checkpoints.pending_runs(tmp_path) == []
