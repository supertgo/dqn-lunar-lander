import json
from os import write

import pandas as pd

from pytest import approx
from dqn.evaluate import EvalResult
from dqn.run_logger import EVAL_COLUMNS, TRAIN_COLUMNS
from fake_runs import write_fake_run

HP = {"learning_rate": 1e-3, "gamma": 0.99}


def test_fake_run_create_all_contract_files(tmp_path):
    run = write_fake_run(tmp_path, "dqn-abc123", HP, seed=0)

    assert run == tmp_path / "dqn-abc123_s0"

    for name in ("config.json", "meta.json", "train.csv", "eval.csv", "summary.json"):
        assert (run / name).is_file()


def test_fake_run_csvs_have_contract_columns(tmp_path):
    run = write_fake_run(tmp_path, "dqn-abc123", HP, seed=0)

    assert list(pd.read_csv(run / "train.csv").columns) == TRAIN_COLUMNS
    assert list(pd.read_csv(run / "eval.csv").columns) == EVAL_COLUMNS


def test_fake_run_summary_has_eval_result_keys_and_wall_time(tmp_path):
    run = write_fake_run(tmp_path, "dqn-abc123", HP, seed=0)

    summary = json.loads((run / "summary.json").read_text())

    assert set(summary) == set(EvalResult.__annotations__) | {"wall_time_s"}


def test_fake_run_config_identifies_the_run(tmp_path):
    run = write_fake_run(tmp_path, "dqn-abc123", HP, seed=3)

    config = json.loads((run / "config.json").read_text())

    assert config["run_id"] == "dqn-abc123_s3"
    assert config["config_id"] == "dqn-abc123"
    assert config["seed"] == 3
    assert config["hyperparameters"] == HP


def test_fake_run_is_reproducible_by_seed(tmp_path):
    a = pd.read_csv(write_fake_run(tmp_path / "a", "dqn-abc123", HP, seed=0) / "train.csv")
    b = pd.read_csv(write_fake_run(tmp_path / "b", "dqn-abc123", HP, seed=0) / "train.csv")
    c = pd.read_csv(write_fake_run(tmp_path / "c", "dqn-abc123", HP, seed=1) / "train.csv")

    assert a.equals(b)
    assert not a.equals(c)


def test_fake_run_learning_curve_converges_to_final_return(tmp_path):
    for final_return in (200.0, -50.0):
        run = write_fake_run(tmp_path, f"dqn-{final_return}", HP, seed=0, final_return=final_return)

        train = pd.read_csv(run / "train.csv")

        assert train["return"].tail(50).mean() == approx(final_return, abs=30)
