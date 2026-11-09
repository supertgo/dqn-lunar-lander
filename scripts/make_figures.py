"""Gera as figuras do relatório (report/figures/) a partir das runs em results/runs/."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aggregate import load_runs, summarize
from matplotlib.ticker import FuncFormatter

from dqn.config import CONFIGS_DIR, RESULTS_DIR, ROOT

FIGURES_DIR = ROOT / "report" / "figures"
SOLVED_THRESHOLD = 200.0
N_GRID_POINTS = 100

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]  # 2º canal além da cor (impressão P&B, daltonismo)
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load_curves(results_dir: Path, config_ids: list[str]) -> pd.DataFrame:
    frames = []

    for run in sorted(Path(results_dir).iterdir()):
        if not (run / "summary.json").is_file():
            continue

        config = json.loads((run / "config.json").read_text())

        if config["config_id"] not in config_ids:
            continue

        curve = pd.read_csv(run / "eval.csv")
        curve.insert(0, "seed", config["seed"])
        curve.insert(0, "config_id", config["config_id"])
        frames.append(curve)

    if not frames:
        return pd.DataFrame(columns=["config_id", "seed", "step", "mean_return"])

    return pd.concat(frames, ignore_index=True)


def mean_curve(curves: pd.DataFrame, metric: str = "mean_return") -> pd.DataFrame:
    per_seed = [g.sort_values("step") for _, g in curves.groupby("seed")]
    start = max(g["step"].iloc[0] for g in per_seed)
    end = min(g["step"].iloc[-1] for g in per_seed)
    grid = np.linspace(start, end, N_GRID_POINTS)

    values = np.stack([np.interp(grid, g["step"], g[metric]) for g in per_seed])
    std = values.std(axis=0, ddof=1) if len(per_seed) > 1 else np.zeros_like(grid)

    return pd.DataFrame(
        {"step": grid, "mean": values.mean(axis=0), "std": std, "n_seeds": len(per_seed)}
    )


def plot_learning_curves(ax, curves: pd.DataFrame, labels: dict[str, str]) -> None:
    for i, (config_id, label) in enumerate(labels.items()):
        one = curves[curves["config_id"] == config_id]

        if one.empty:
            continue

        m = mean_curve(one)
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.fill_between(
            m["step"],
            m["mean"] - m["std"],
            m["mean"] + m["std"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            m["step"],
            m["mean"],
            color=color,
            linewidth=1.5,
            label=label,
            marker=MARKERS[i % len(MARKERS)],
            markersize=4,
            markevery=N_GRID_POINTS // 5,
        )

    ax.axhline(SOLVED_THRESHOLD, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    ax.annotate(
        "resolvido",
        xy=(0, SOLVED_THRESHOLD),
        xycoords=("axes fraction", "data"),
        xytext=(2, 2),
        textcoords="offset points",
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_xlabel("passos de treino")
    ax.set_ylabel("retorno médio (avaliação)")


def plot_final_performance(ax, configs: pd.DataFrame, labels: dict[str, str]) -> None:
    for i, (config_id, label) in enumerate(labels.items()):
        row = configs[configs["config_id"] == config_id]

        if row.empty:
            continue

        row = row.iloc[0]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.errorbar(
            i,
            row["mean_return"],
            yerr=row["mean_return_std"],
            fmt=MARKERS[i % len(MARKERS)],
            color=color,
            markersize=6,
            capsize=3,
            linewidth=1.2,
        )

    ax.axhline(SOLVED_THRESHOLD, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(labels)), list(labels.values()), fontsize=8)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("retorno médio (avaliação final)")


def sweep_figure(
    results_dir: Path, labels: dict[str, str], title: str, out_path: Path, legend_title: str = ""
) -> Path:
    runs = load_runs(results_dir)
    configs = summarize(runs)
    curves = load_curves(results_dir, list(labels))

    fig, (ax_curves, ax_final) = plt.subplots(
        1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [5, 3]}
    )
    plot_learning_curves(ax_curves, curves, labels)
    plot_final_performance(ax_final, configs, labels)
    handles, names = ax_curves.get_legend_handles_labels()
    fig.legend(
        handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(names), fontsize=8
    )
    ax_final.set_xlabel(legend_title)
    n_seeds = configs[configs["config_id"].isin(labels)]["n_seeds"]
    seeds_note = (
        f"{n_seeds.min()}–{n_seeds.max()}" if n_seeds.min() != n_seeds.max() else f"{n_seeds.min()}"
    )
    fig.suptitle(
        f"{title}  ({seeds_note} seeds; faixa e barras = ±1 desvio)",
        color=TEXT_PRIMARY,
        fontsize=10,
        y=1.02,
    )
    fig.subplots_adjust(top=0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def variant_labels(
    run_configs, variant_keys: list[str], default: str = "referência"
) -> dict[str, str]:
    labels: dict[str, str] = {}

    for cfg in run_configs:
        if cfg.config_id in labels:
            continue

        values = [str(cfg.hyperparameters[k]) for k in variant_keys]
        labels[cfg.config_id] = " / ".join(values) if values else default

    return labels


def main() -> None:
    from dqn.config import load_sweep  # só o main depende do config.py (Pessoa B)

    for path in sorted((CONFIGS_DIR / "sweeps").glob("*.yaml")):
        sweep = load_sweep(path)

        if sweep.name == "smoke":
            continue

        keys = list(dict.fromkeys(k for variant in sweep.variants for k in variant))
        labels = variant_labels(sweep.run_configs(), keys, default="DQN (nosso)")

        if sweep.name == "final":
            runs = load_runs(RESULTS_DIR)

            for config_id in runs[runs["algo"] == "sb3"]["config_id"].unique():
                labels[config_id] = "DQN (SB3)"

        out = sweep_figure(
            RESULTS_DIR, labels, sweep.name, FIGURES_DIR / f"{sweep.name}.pdf", " / ".join(keys)
        )
        print(f"{sweep.name}: {out}")


if __name__ == "__main__":
    main()
