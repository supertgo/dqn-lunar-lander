from pathlib import Path
from types import SimpleNamespace

import pytest
import run_sweep


def test_run_sweep_runs_pending_configs_sequentially_and_skips_completed(
    tmp_path: Path, monkeypatch, capsys
):
    completed = SimpleNamespace(run_id="dqn-done_s0")
    first = SimpleNamespace(run_id="dqn-first_s0")
    duplicate = SimpleNamespace(run_id="dqn-first_s0")
    second = SimpleNamespace(run_id="dqn-second_s1")
    (tmp_path / completed.run_id).mkdir()
    (tmp_path / completed.run_id / "summary.json").write_text("{}", encoding="utf-8")
    sweep = SimpleNamespace(
        name="test",
        results_dir=tmp_path,
        run_configs=lambda: [completed, first, duplicate, second],
    )
    calls = []

    def fake_train(cfg, results_dir):
        calls.append((cfg.run_id, results_dir))
        return {"run_id": cfg.run_id}

    monkeypatch.setattr(run_sweep, "_train", fake_train)

    summaries = run_sweep.run_sweep(sweep)

    assert [run_id for run_id, _ in calls] == [first.run_id, second.run_id]
    assert all(results_dir == tmp_path for _, results_dir in calls)
    assert summaries == [{"run_id": first.run_id}, {"run_id": second.run_id}]
    output = capsys.readouterr().out
    assert f"[skip] {completed.run_id}" in output
    assert "2 run(s) pendente(s), 1 worker(s)" in output


def test_train_delegates_to_real_dqn_training_with_one_thread(tmp_path: Path, monkeypatch):
    import torch

    calls = []

    def fake_dqn_train(cfg, results_dir):
        calls.append((cfg, results_dir, torch.get_num_threads()))
        return {"mean_return": 1.0}

    monkeypatch.setattr(run_sweep, "train", fake_dqn_train)
    original_threads = torch.get_num_threads()
    try:
        cfg = SimpleNamespace(run_id="dqn-x_s0")

        summary = run_sweep._train(cfg, tmp_path)
    finally:
        torch.set_num_threads(original_threads)

    assert summary == {"mean_return": 1.0}
    assert calls == [(cfg, tmp_path, 1)]


def test_run_sweep_uses_dqn_train_function():
    from dqn.train import train

    assert run_sweep.train is train


def test_train_writes_run_files(tmp_path: Path):
    from dqn.config import EvalConfig, RunConfig

    cfg = RunConfig(
        algo="dqn",
        hyperparameters={
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "batch_size": 4,
            "buffer_size": 100,
            "learning_starts": 20,
            "target_update_interval": 10,
            "total_timesteps": 20,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 10,
            "train_freq": 1,
            "hidden_sizes": [64, 64],
        },
        evaluation=EvalConfig(
            env_id="LunarLander-v3",
            eval_freq=5,
            eval_episodes=2,
            final_eval_episodes=3,
            eval_seed=1,
            solved_threshold=200.0,
        ),
        seed=7,
    )

    summary = run_sweep._train(cfg, tmp_path)
    run_dir = tmp_path / cfg.run_id

    assert set(summary) == {
        "mean_return",
        "std_return",
        "success_rate",
        "mean_length",
        "n_episodes",
        "wall_time_s",
    }
    assert {
        "config.json",
        "meta.json",
        "train.csv",
        "eval.csv",
        "summary.json",
        "model.pt",
    } <= {path.name for path in run_dir.iterdir()}
    assert (run_dir / "summary.json").is_file()


def test_run_sweep_rejects_non_positive_workers(tmp_path: Path):
    sweep = SimpleNamespace(name="test", results_dir=tmp_path, run_configs=list)

    with pytest.raises(ValueError, match="workers"):
        run_sweep.run_sweep(sweep, workers=0)


def test_run_sweep_rejects_empty_sweep(tmp_path: Path):
    sweep = SimpleNamespace(name="empty", results_dir=tmp_path, run_configs=list)

    with pytest.raises(ValueError, match="não gera nenhuma run"):
        run_sweep.run_sweep(sweep)


def test_main_loads_sweep_and_uses_requested_workers(monkeypatch, tmp_path: Path):
    sweep = object()
    loaded = []
    runs = []

    def fake_load_sweep(path):
        loaded.append(path)
        return sweep

    def fake_run_sweep(value, workers):
        runs.append((value, workers))

    monkeypatch.setattr(run_sweep, "load_sweep", fake_load_sweep)
    monkeypatch.setattr(run_sweep, "run_sweep", fake_run_sweep)

    assert run_sweep.main([str(tmp_path / "sweep.yaml"), "--workers", "3"]) == 0
    assert loaded == [tmp_path / "sweep.yaml"]
    assert runs == [(sweep, 3)]


def test_main_runs_several_sweeps_in_one_queue_without_duplicates(monkeypatch, tmp_path: Path):
    shared = SimpleNamespace(run_id="dqn-base_s0")
    sweeps = {
        tmp_path / "a.yaml": SimpleNamespace(
            name="a",
            results_dir=tmp_path,
            run_configs=lambda: [shared, SimpleNamespace(run_id="dqn-a_s0")],
        ),
        tmp_path / "b.yaml": SimpleNamespace(
            name="b",
            results_dir=tmp_path,
            run_configs=lambda: [shared, SimpleNamespace(run_id="dqn-b_s0")],
        ),
    }
    trained = []
    monkeypatch.setattr(run_sweep, "load_sweep", lambda path: sweeps[path])
    monkeypatch.setattr(
        run_sweep, "_train", lambda cfg, results_dir: trained.append(cfg.run_id) or {}
    )

    assert run_sweep.main([str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")]) == 0
    assert trained == ["dqn-base_s0", "dqn-a_s0", "dqn-b_s0"]


def test_main_rejects_sweeps_with_different_results_dirs(monkeypatch, tmp_path: Path):
    sweeps = {
        tmp_path / "a.yaml": SimpleNamespace(
            name="a", results_dir=tmp_path / "x", run_configs=list
        ),
        tmp_path / "b.yaml": SimpleNamespace(
            name="b", results_dir=tmp_path / "y", run_configs=list
        ),
    }
    monkeypatch.setattr(run_sweep, "load_sweep", lambda path: sweeps[path])

    with pytest.raises(ValueError, match="results_dir"):
        run_sweep.main([str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")])
