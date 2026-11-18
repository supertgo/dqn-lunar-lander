from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

import dqn.config as config
from dqn.config import EvalConfig, Hyperparameters, RunConfig, Sweep


@pytest.fixture
def hp_data() -> dict:
    return {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 100_000,
        "learning_starts": 1_000,
        "target_update_interval": 500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 10_000,
        "total_timesteps": 500_000,
        "train_freq": 4,
        "hidden_sizes": [64, 64],
    }


@pytest.fixture
def eval_data() -> dict:
    return {
        "env_id": "LunarLander-v3",
        "eval_freq": 5_000,
        "eval_episodes": 10,
        "final_eval_episodes": 20,
        "eval_seed": 1_234,
        "solved_threshold": 200.0,
    }


@pytest.fixture
def config_files(tmp_path: Path, monkeypatch, hp_data: dict, eval_data: dict):
    hp_file = tmp_path / "hyperparameters.yaml"
    eval_file = tmp_path / "evaluation.yaml"
    hp_file.write_text(yaml.safe_dump(hp_data), encoding="utf-8")
    eval_file.write_text(yaml.safe_dump(eval_data), encoding="utf-8")

    monkeypatch.setattr(config, "HYPERPARAMETERS_FILE", hp_file)
    monkeypatch.setattr(config, "EVALUATION_FILE", eval_file)

    return hp_file, eval_file


def make_eval(**overrides) -> EvalConfig:
    values = {
        "env_id": "LunarLander-v3",
        "eval_freq": 5_000,
        "eval_episodes": 10,
        "final_eval_episodes": 20,
        "eval_seed": 1_234,
        "solved_threshold": 200.0,
    }
    values.update(overrides)
    return EvalConfig(**values)


def test_load_hyperparameters_reads_yaml(config_files):
    hp = config.load_hyperparameters()

    assert hp.learning_rate == 0.001
    assert hp.gamma == 0.99
    assert hp.hidden_sizes == (64, 64)


def test_load_hyperparameters_applies_overrides_without_mutating_input(config_files):
    overrides = {"learning_rate": 0.0005, "hidden_sizes": [32, 32]}

    hp = config.load_hyperparameters(overrides)

    assert hp.learning_rate == 0.0005
    assert hp.hidden_sizes == (32, 32)
    assert overrides == {"learning_rate": 0.0005, "hidden_sizes": [32, 32]}


def test_load_evaluation_reads_yaml_and_applies_overrides(config_files):
    evaluation = config.load_evaluation(
        {"eval_freq": 1_000, "eval_episodes": 3}
    )

    assert evaluation.env_id == "LunarLander-v3"
    assert evaluation.eval_freq == 1_000
    assert evaluation.eval_episodes == 3
    assert evaluation.final_eval_episodes == 20


def test_load_yaml_helpers_reject_missing_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "HYPERPARAMETERS_FILE", tmp_path / "missing.yaml")

    with pytest.raises(FileNotFoundError):
        config.load_hyperparameters()


def test_load_hyperparameters_rejects_missing_fields(config_files):
    config.HYPERPARAMETERS_FILE.write_text("learning_rate: 0.001\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Hiperparâmetros"):
        config.load_hyperparameters()


def test_load_evaluation_rejects_missing_fields(config_files):
    config.EVALUATION_FILE.write_text("env_id: LunarLander-v3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="avaliação"):
        config.load_evaluation()


def test_dqn_run_config_builds_typed_hyperparameters(config_files):
    run = config.dqn_run_config(
        seed=7,
        hp_overrides={"batch_size": 128},
        eval_overrides={"eval_episodes": 4},
    )

    assert run.algo == "dqn"
    assert run.seed == 7
    assert run.hyperparameters["batch_size"] == 128
    assert run.evaluation.eval_episodes == 4
    assert run.hp.batch_size == 128
    assert run.hp.hidden_sizes == (64, 64)


def test_run_config_hp_returns_hyperparameters_with_list_hidden_sizes():
    evaluation = make_eval()
    run = RunConfig(
        algo="dqn",
        hyperparameters={
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 64,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "target_update_interval": 500,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 10_000,
            "total_timesteps": 500_000,
            "train_freq": 4,
            "hidden_sizes": [32, 32],
        },
        evaluation=evaluation,
        seed=1,
    )

    assert run.hp.hidden_sizes == (32, 32)


def test_run_config_hp_rejects_sb3():
    run = RunConfig(
        algo="sb3",
        hyperparameters={"total_timesteps": 500_000},
        evaluation=make_eval(),
        seed=1,
    )

    with pytest.raises(ValueError, match="DQN"):
        _ = run.hp


def test_config_id_is_deterministic_and_excludes_seed():
    evaluation = make_eval()
    hp = {"learning_rate": 0.001, "gamma": 0.99}

    a = RunConfig("dqn", hp, evaluation, seed=1)
    b = RunConfig("dqn", hp, evaluation, seed=999)

    assert a.config_id == b.config_id
    assert a.config_id.startswith("dqn-")
    assert len(a.config_id.removeprefix("dqn-")) == 6
    assert a.run_id == f"{a.config_id}_s1"
    assert b.run_id == f"{b.config_id}_s999"


def test_config_id_changes_with_configuration_but_not_seed():
    evaluation = make_eval()
    base = RunConfig("dqn", {"learning_rate": 0.001}, evaluation, seed=1)
    changed_hp = RunConfig(
        "dqn", {"learning_rate": 0.0001}, evaluation, seed=1
    )
    changed_eval = RunConfig(
        "dqn",
        {"learning_rate": 0.001},
        make_eval(eval_episodes=5),
        seed=1,
    )
    changed_algo = RunConfig(
        "sb3", {"learning_rate": 0.001}, evaluation, seed=1
    )

    assert base.config_id != changed_hp.config_id
    assert base.config_id != changed_eval.config_id
    assert base.config_id != changed_algo.config_id


def test_run_config_to_dict_is_json_serializable(config_files):
    run = config.dqn_run_config(
        seed=5,
        hp_overrides={"learning_rate": 0.0002},
    )
    data = run.to_dict()

    assert data["algo"] == "dqn"
    assert data["seed"] == 5
    assert data["config_id"] == run.config_id
    assert data["run_id"] == run.run_id
    assert data["hyperparameters"] == run.hyperparameters
    assert data["evaluation"] == asdict(run.evaluation)


def test_sweep_run_configs_is_variant_times_seed(config_files):
    sweep = Sweep(
        name="test",
        seeds=[1, 2, 3],
        variants=[{"batch_size": 32}, {"batch_size": 128}],
        evaluation={"eval_episodes": 2},
        results_dir=Path("/tmp/results"),
    )

    runs = sweep.run_configs()

    assert len(runs) == 6
    assert {(run.seed, run.hyperparameters["batch_size"]) for run in runs} == {
        (1, 32),
        (2, 32),
        (3, 32),
        (1, 128),
        (2, 128),
        (3, 128),
    }
    assert all(run.evaluation.eval_episodes == 2 for run in runs)
    assert len({run.run_id for run in runs}) == 6


def test_load_sweep_reads_fields(tmp_path: Path):
    sweep_file = tmp_path / "epsilon.yaml"
    sweep_file.write_text(
        yaml.safe_dump(
            {
                "name": "epsilon",
                "seeds": [1, 2],
                "variants": [
                    {"epsilon_start": 1.0},
                    {"epsilon_end": 0.01},
                ],
                "evaluation": {"eval_freq": 2_000},
                "results_dir": "results/custom",
            }
        ),
        encoding="utf-8",
    )

    sweep = config.load_sweep(sweep_file)

    assert sweep.name == "epsilon"
    assert sweep.seeds == [1, 2]
    assert len(sweep.variants) == 2
    assert sweep.evaluation == {"eval_freq": 2_000}
    assert sweep.results_dir == config.ROOT / "results/custom"


def test_load_sweep_uses_defaults(tmp_path: Path):
    sweep_file = tmp_path / "smoke.yaml"
    sweep_file.write_text(
        yaml.safe_dump({"name": "smoke"}),
        encoding="utf-8",
    )

    sweep = config.load_sweep(sweep_file)

    assert sweep.seeds == []
    assert sweep.variants == []
    assert sweep.evaluation == {}
    assert sweep.results_dir == config.RESULTS_DIR
    assert sweep.run_configs() == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", "1"),
        ("variants", "invalid"),
        ("variants", ["invalid"]),
        ("evaluation", "invalid"),
    ],
)
def test_load_sweep_rejects_invalid_field_types(tmp_path: Path, field: str, value):
    sweep_file = tmp_path / "invalid.yaml"
    sweep_file.write_text(
        yaml.safe_dump({"name": "invalid", field: value}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        config.load_sweep(sweep_file)


def test_load_sweep_rejects_missing_name(tmp_path: Path):
    sweep_file = tmp_path / "invalid.yaml"
    sweep_file.write_text(yaml.safe_dump({"seeds": [1]}), encoding="utf-8")

    with pytest.raises(ValueError, match="nome"):
        config.load_sweep(sweep_file)


def test_load_sweep_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        config.load_sweep(tmp_path / "missing.yaml")