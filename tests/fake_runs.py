import json
from pathlib import Path

from dqn.run_logger import TRAIN_COLUMNS

import numpy as np
import pandas as pd

from dqn.run_logger import EVAL_COLUMNS, TRAIN_COLUMNS

START_RETURN = -200.0
NOISE_STD = 30.0
SOLVED_THRESHOLD = 200.0


def write_fake_run(
    results_dir: Path,
    config_id: str,
    hyperparameters: dict,
    seed: int,
    *,
    algo: str = "dqn",
    final_return: float = 200.0,
    n_episodes: int = 300,
    eval_every: int = 10,
) -> Path:
    rng = np.random.default_rng(seed)
    run_id = f"{config_id}_s{seed}"
    run = Path(results_dir) / run_id
    run.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(n_episodes)
    progress = 1 - np.exp(-episodes / (n_episodes / 5))
    curve = START_RETURN + (final_return - START_RETURN) * progress
    returns = curve + rng.normal(0, NOISE_STD, size=n_episodes)
    lengths = rng.integers(100, 1000, size=n_episodes)
    steps = np.cumsum(lengths)

    train = pd.DataFrame(
        {
            "step": steps,
            "episode": episodes,
            "return": returns,
            "length": lengths,
            "epsilon": np.linspace(1.0, 0.05, n_episodes),
            "loss": 1.0 / (1 + episodes) + rng.uniform(0, 0.05, size=n_episodes),
        }
    )
    train[TRAIN_COLUMNS].to_csv(run / "train.csv", index=False)

    # avaliação periódica: uma linha a cada eval_every episódios
    eval_idx = episodes[eval_every - 1 :: eval_every]
    eval_df = pd.DataFrame(
        {
            "step": steps[eval_idx],
            "mean_return": curve[eval_idx] + rng.normal(0, NOISE_STD / 3, size=len(eval_idx)),
            "std_return": rng.uniform(10, 60, size=len(eval_idx)),
            "success_rate": np.clip(progress[eval_idx] * (final_return / SOLVED_THRESHOLD), 0, 1),
            "mean_length": lengths[eval_idx].astype(float),
        }
    )
    eval_df[EVAL_COLUMNS].to_csv(run / "eval.csv", index=False)

    config = {
        "algo": algo,
        "hyperparameters": hyperparameters,
        "evaluation": {},
        "seed": seed,
        "run_id": run_id,
        "config_id": config_id,
    }
    meta = {"hostname": "fake", "git_commit": "fake", "created_at": "1970-01-01T00:00:00"}
    last = eval_df.iloc[-1]
    summary = {
        "mean_return": float(last["mean_return"]),
        "std_return": float(last["std_return"]),
        "success_rate": float(last["success_rate"]),
        "mean_length": float(last["mean_length"]),
        "n_episodes": 100,
        "wall_time_s": 60.0,
    }

    for name, content in (("config.json", config), ("meta.json", meta), ("summary.json", summary)):
        (run / name).write_text(json.dumps(content, indent=2))

    return run
