import json
from pathlib import Path

import pandas as pd

from dqn.config import RESULTS_DIR


ID_COLUMNS = ["run_id", "config_id", "algo", "seed"]
SUMMARY_METRICS = [
    "mean_return",
    "std_return",
    "success_rate",
    "mean_length",
    "n_episodes",
    "wall_time_s",
]
METRICS_TO_AGGREGATE = ["mean_return", "success_rate", "mean_length", "wall_time_s"]


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows = []

    for run in sorted(Path(results_dir).iterdir()):
        summary_path = run / "summary.json"

        if not summary_path.is_file():
            continue

        config = json.loads((run / "config.json").read_text())
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "run_id": config["run_id"],
                "config_id": config["config_id"],
                "algo": config["algo"],
                "seed": config["seed"],
                **config["hyperparameters"],
                **summary,
            }
        )

    if not rows:
        return pd.DataFrame(columns=[*ID_COLUMNS, *SUMMARY_METRICS])

    return pd.DataFrame(rows)


def hyperparameter_columns(runs: pd.DataFrame) -> list[str]:
    known = set(ID_COLUMNS) | set(SUMMARY_METRICS)

    return [c for c in runs.columns if c not in known]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "n_algo": ("algo", "first"),
        "n_seeds": ("seed", "size"),
        **{hp: (hp, "first") for hp in hyperparameter_columns(runs)},
    }

    for metric in METRICS_TO_AGGREGATE:
        aggregations[metric] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    return runs.groupby("config_id").agg(**aggregations).reset_index()


def main() -> None:
    runs = load_runs(RESULTS_DIR)
    out = RESULTS_DIR.parent / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summarize(runs).to_csv(out / "config.csv", index=False)

    print(f"{len(runs)} runs concluidas -> {out}")


if __name__ == "__main__":
    main()
