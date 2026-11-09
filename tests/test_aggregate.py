from pytest import approx

from aggregate import load_runs, summarize
from fake_runs import write_fake_run

HP_A = {"learning_rate": 1e-3, "gamma": 0.99}
HP_B = {"learning_rate": 1e-2, "gamma": 0.99}


def write_sweep(results_dir, seeds=(0, 1, 2)):
    """Duas configurações × três seeds."""
    for seed in seeds:
        write_fake_run(results_dir, "dqn-aaa", HP_A, seed, final_return=200.0)
        write_fake_run(results_dir, "dqn-bbb", HP_B, seed, final_return=-50.0)


def test_load_runs_returns_one_row_per_completed_run(tmp_path):
    write_sweep(tmp_path)

    runs = load_runs(tmp_path)

    assert len(runs) == 6
    assert set(runs["config_id"]) == {"dqn-aaa", "dqn-bbb"}
    assert {"run_id", "seed", "mean_return", "success_rate"} <= set(runs.columns)


def test_summarize_groups_seeds_by_config(tmp_path):
    write_sweep(tmp_path)

    configs = summarize(load_runs(tmp_path))

    assert len(configs) == 2
    assert list(configs["n_seeds"]) == [3, 3]


def test_load_runs_ignores_incomplete_runs(tmp_path):
    write_sweep(tmp_path)
    (tmp_path / "dqn-bbb_s1" / "summary.json").unlink()

    runs = load_runs(tmp_path)

    assert len(runs) == 5
    assert "dqn-bbb_s1" not in set(runs["run_id"])


def test_load_runs_ignores_files_in_results_dir(tmp_path):
    write_sweep(tmp_path)
    (tmp_path / ".gitkeep").touch()

    assert len(load_runs(tmp_path)) == 6


def test_load_runs_turns_hyperparameters_into_columns(tmp_path):
    write_sweep(tmp_path)

    runs = load_runs(tmp_path)

    assert {"learning_rate", "gamma"} <= set(runs.columns)
    row = runs[runs["run_id"] == "dqn-bbb_s0"].iloc[0]
    assert row["learning_rate"] == 1e-2


def test_summarize_computes_mean_and_std_across_seeds(tmp_path):
    write_sweep(tmp_path)
    runs = load_runs(tmp_path)

    configs = summarize(runs)

    seeds_a = runs[runs["config_id"] == "dqn-aaa"]["mean_return"]
    row_a = configs[configs["config_id"] == "dqn-aaa"].iloc[0]
    assert row_a["mean_return"] == approx(seeds_a.mean())
    assert row_a["mean_return_std"] == approx(seeds_a.std())


def test_summarize_keeps_hyperparameters(tmp_path):
    write_sweep(tmp_path)

    configs = summarize(load_runs(tmp_path))

    row_b = configs[configs["config_id"] == "dqn-bbb"].iloc[0]
    assert row_b["learning_rate"] == 1e-2
    assert row_b["gamma"] == 0.99


def test_empty_results_dir_gives_empty_tables(tmp_path):
    runs = load_runs(tmp_path)

    assert len(runs) == 0
    assert len(summarize(runs)) == 0
