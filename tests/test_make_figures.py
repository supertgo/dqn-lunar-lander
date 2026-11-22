from types import SimpleNamespace

import pandas as pd
from fake_runs import write_fake_run
from make_figures import (
    format_steps,
    load_curves,
    mean_curve,
    seeds_note,
    sweep_figure,
    variant_labels,
)
from pytest import approx

HP_A = {"learning_rate": 1e-3, "gamma": 0.99}
HP_B = {"learning_rate": 1e-2, "gamma": 0.99}


def write_sweep(results_dir, seeds=(0, 1, 2)):
    for seed in seeds:
        write_fake_run(results_dir, "dqn-aaa", HP_A, seed, final_return=200.0)
        write_fake_run(results_dir, "dqn-bbb", HP_B, seed, final_return=-50.0)


def test_load_curves_returns_only_requested_configs_with_seed(tmp_path):
    write_sweep(tmp_path)

    curves = load_curves(tmp_path, ["dqn-aaa"])

    assert set(curves["config_id"]) == {"dqn-aaa"}
    assert set(curves["seed"]) == {0, 1, 2}
    assert {"step", "mean_return"} <= set(curves.columns)


def test_load_curves_ignores_incomplete_runs(tmp_path):
    write_sweep(tmp_path)
    (tmp_path / "dqn-aaa_s1" / "summary.json").unlink()

    curves = load_curves(tmp_path, ["dqn-aaa"])

    assert set(curves["seed"]) == {0, 2}


def make_curves(seed_curves):
    """{seed: (steps, valores)} -> DataFrame no formato de load_curves."""
    rows = [
        {"config_id": "c", "seed": seed, "step": s, "mean_return": v}
        for seed, (steps, values) in seed_curves.items()
        for s, v in zip(steps, values)
    ]
    return pd.DataFrame(rows)


def test_mean_curve_averages_seeds_on_common_grid():
    curves = make_curves({0: ([0, 100], [0.0, 100.0]), 1: ([0, 100], [100.0, 200.0])})

    m = mean_curve(curves)

    assert m["mean"].iloc[0] == approx(50.0)
    assert m["mean"].iloc[-1] == approx(150.0)
    assert m["n_seeds"].iloc[0] == 2


def test_mean_curve_of_identical_seeds_has_zero_std():
    curves = make_curves({0: ([0, 100], [5.0, 9.0]), 1: ([0, 100], [5.0, 9.0])})

    assert (mean_curve(curves)["std"] == 0).all()


def test_mean_curve_aligns_seeds_evaluated_at_different_steps():
    curves = make_curves({0: ([0, 100, 200], [0.0, 100.0, 200.0]), 1: ([0, 150], [0.0, 150.0])})

    m = mean_curve(curves)

    assert m["step"].iloc[-1] == approx(150.0)  # não passa do menor último passo
    assert m["mean"].iloc[-1] == approx(150.0)  # as duas seeds valem 150 em step=150


def test_variant_labels_names_each_config_once_in_variant_order():
    cfgs = [
        SimpleNamespace(config_id=cid, hyperparameters={"learning_rate": lr, "gamma": 0.99})
        for cid, lr in [("dqn-bbb", 1e-2), ("dqn-aaa", 1e-3), ("dqn-bbb", 1e-2)]
    ]

    labels = variant_labels(cfgs, ["learning_rate"])

    assert labels == {"dqn-bbb": "0.01", "dqn-aaa": "0.001"}


def test_variant_labels_joins_multiple_keys():
    cfgs = [
        SimpleNamespace(
            config_id="dqn-aaa", hyperparameters={"epsilon_start": 1.0, "epsilon_end": 0.05}
        )
    ]

    assert variant_labels(cfgs, ["epsilon_start", "epsilon_end"]) == {"dqn-aaa": "1.0 / 0.05"}


def test_variant_labels_without_variant_keys_uses_default():
    cfgs = [SimpleNamespace(config_id="dqn-aaa", hyperparameters={"gamma": 0.99})]

    assert variant_labels(cfgs, [], default="DQN (nosso)") == {"dqn-aaa": "DQN (nosso)"}


def test_sweep_figure_writes_file(tmp_path):
    runs_dir = tmp_path / "runs"
    write_sweep(runs_dir)
    labels = {"dqn-aaa": "0.001", "dqn-bbb": "0.01"}

    out = sweep_figure(runs_dir, labels, "learning_rate", tmp_path / "fig" / "learning_rate.pdf")

    assert out.is_file()
    assert out.stat().st_size > 0


def test_format_steps_keeps_distinct_ticks_distinct():
    assert format_steps(1500) == "1.5k"
    assert format_steps(2000) == "2k"
    assert format_steps(250_000) == "250k"
    assert format_steps(1_500_000) == "1.5M"
    assert format_steps(500) == "500"
    assert format_steps(0) == "0"


def test_seeds_note_uses_singular_and_ranges():
    assert seeds_note(1, 1) == "1 seed"
    assert seeds_note(3, 3) == "3 seeds"
    assert seeds_note(2, 5) == "2–5 seeds"
