"""Gera as figuras e os números do relatório (report/figures/ e report/figures/numeros.json).

Separação central: seeds 0–9 foram usadas para escolher configurações ("seeds da busca");
seeds >= 10 nunca participaram da escolha ("seeds novas") e dão a estimativa sem viés de seleção.
Intervalos: IC de 95% por bootstrap percentil da média entre seeds.

Uso: uv run python scripts/report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import make_figures as mf  # aplica o estilo (rcParams) e as cores das figuras do projeto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aggregate import load_runs
from matplotlib.ticker import FuncFormatter

from dqn.config import CONFIGS_DIR, RESULTS_DIR, ROOT, load_sweep

OUT = ROOT / "report" / "figures"
EXTRA = ROOT / "results" / "extra" / "best_checkpoint"
FRESH = 10  # seeds >= FRESH não participaram da seleção
REF = "dqn-b3a10c"
WINNER = "dqn-b2e60e"
C_REF, C_ALT, C_SB3, C_GRAY = mf.SERIES_COLORS[0], mf.SERIES_COLORS[1], mf.SERIES_COLORS[2], "#9b9a94"

AXES = [  # (arquivo do sweep de um fator, título do painel, rótulo de cada variante)
    ("learning_rate", r"taxa de aprendizado $\alpha$", lambda hp: f"{hp['learning_rate']:g}"),
    ("gamma", r"fator de desconto $\gamma$", lambda hp: f"{hp['gamma']:g}"),
    ("buffer_size", r"tamanho do buffer $|\mathcal{D}|$", lambda hp: mf.format_steps(hp["buffer_size"])),
    ("batch_size", r"tamanho do batch $B$", lambda hp: f"{hp['batch_size']}"),
    ("target_update", r"atualização do alvo $C$", lambda hp: mf.format_steps(hp["target_update_interval"])),
    (
        "epsilon",
        r"$\varepsilon$: $T_\varepsilon$ / $\varepsilon_{\min}$",
        lambda hp: f"{mf.format_steps(hp['epsilon_decay_steps'])}/{hp['epsilon_end']:g}",
    ),
]


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(x, size=(n, len(x))).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def stats(runs: pd.DataFrame) -> dict:
    r = runs["mean_return"].to_numpy()
    lo, hi = boot_ci(r)
    return {
        "n": int(len(r)),
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "ci": [lo, hi],
        "median": float(np.median(r)),
        "min": float(r.min()),
        "success": float(runs["success_rate"].mean()),
        "seeds": sorted(int(s) for s in runs["seed"]),
    }


def sweep_ids(name: str) -> list[tuple[str, dict]]:
    sweep = load_sweep(CONFIGS_DIR / "sweeps" / f"{name}.yaml")
    seen: dict[str, dict] = {}
    for cfg in sweep.run_configs():
        seen.setdefault(cfg.config_id, cfg.hyperparameters)
    return list(seen.items())


def round_ids(name: str) -> list[tuple[str, dict]]:
    sweep = load_sweep(CONFIGS_DIR / "sweeps" / "rounds" / f"{name}.yaml")
    seen: dict[str, dict] = {}
    for cfg in sweep.run_configs():
        seen.setdefault(cfg.config_id, cfg.hyperparameters)
    return list(seen.items())


def _panel(ax, groups: list[tuple[str, pd.DataFrame, bool]], title: str) -> None:
    """groups: (rótulo, runs, é_referência). Pontos = seeds; marcador + barra = média e IC 95%."""
    rng = np.random.default_rng(0)
    for i, (label, g, is_ref) in enumerate(groups):
        if g.empty:
            continue
        y = g["mean_return"].to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=7, color=C_GRAY, alpha=0.7, zorder=2, linewidths=0)
        lo, hi = boot_ci(y)
        color = C_REF if is_ref else C_ALT
        ax.errorbar(i, y.mean(), yerr=[[y.mean() - lo], [hi - y.mean()]], fmt="o", color=color,
                    markersize=5, capsize=3, linewidth=1.3, zorder=3)
    ax.axhline(mf.SOLVED_THRESHOLD, color=mf.TEXT_SECONDARY, linewidth=0.7, linestyle="--")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=7)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, fontsize=8.5, color=mf.TEXT_PRIMARY)


def fig_sweeps(runs: pd.DataFrame) -> None:
    search = runs[runs["seed"].isin([0, 1, 2])]  # desenho de um fator por vez: seeds 0–2
    fresh = runs[runs["seed"] >= FRESH]
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.6), sharey=True)
    for ax, (name, title, fmt) in zip(axes.flat, AXES):
        groups = [(fmt(hp), search[search["config_id"] == cid], cid == REF) for cid, hp in sweep_ids(name)]
        _panel(ax, groups, title)

    steps = [(REF, {"total_timesteps": 300000})] + round_ids("round_09_steps")
    steps.sort(key=lambda t: t[1]["total_timesteps"])
    groups = [(mf.format_steps(hp["total_timesteps"]), fresh[fresh["config_id"] == cid], cid == REF) for cid, hp in steps]
    _panel(axes.flat[6], groups, "passos de treino $T$ (seeds novas)")

    stab = round_ids("round_09_stability")
    names = {(10.0, None): "clip", (None, 5e-05): "lr↓", (10.0, 5e-05): "clip+lr↓"}
    groups = [("nenhuma", fresh[fresh["config_id"] == REF], True)] + [
        (names[(hp["max_grad_norm"], hp["learning_rate_end"])], fresh[fresh["config_id"] == cid], False)
        for cid, hp in stab
    ]
    _panel(axes.flat[7], groups, "estabilização (seeds novas)")
    for ax in axes[:, 0]:
        ax.set_ylabel("retorno (aval. final)")
    axes[0, 0].set_ylim(-420, 320)
    fig.tight_layout()
    fig.savefig(OUT / "fig-sweeps.pdf")
    plt.close(fig)


SELECTION = [  # configurações escolhidas na busca e reavaliadas em seeds novas
    (REF, "referência"),
    ("dqn-e61d39", "B256 · tf2 · 512² · ε 0.01"),
    (WINNER, "B256 · tf2 · 512² (vencedora)"),
    ("dqn-5d51ac", "B256 · tf2 · 512² · alvo 500"),
]


def fig_selection(runs: pd.DataFrame) -> dict:
    out = {}
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, (cid, label) in enumerate(SELECTION):
        r = runs[runs["config_id"] == cid]
        s, f = r[r["seed"] < FRESH], r[r["seed"] >= FRESH]
        out[cid] = {"label": label, "busca": stats(s), "novas": stats(f)}
        y = len(SELECTION) - 1 - i
        for g, color, dy in ((s, C_GRAY, 0.12), (f, C_REF, -0.12)):
            lo, hi = boot_ci(g["mean_return"])
            m = g["mean_return"].mean()
            ax.errorbar(m, y + dy, xerr=[[m - lo], [hi - m]], fmt="o", color=color, markersize=5, capsize=3)
        ax.annotate("", xy=(f["mean_return"].mean(), y - 0.12), xytext=(s["mean_return"].mean(), y + 0.12),
                    arrowprops={"arrowstyle": "->", "color": mf.TEXT_SECONDARY, "linewidth": 0.8})
    ax.set_yticks(range(len(SELECTION)), [lab for _, lab in reversed(SELECTION)], fontsize=7.5)
    ax.axvline(mf.SOLVED_THRESHOLD, color=mf.TEXT_SECONDARY, linewidth=0.7, linestyle="--")
    ax.set_xlabel("retorno médio (aval. final), IC 95%")
    ax.plot([], [], "o", color=C_GRAY, label="seeds da busca (0–9)")
    ax.plot([], [], "o", color=C_REF, label="seeds novas (≥10)")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig-selecao.pdf")
    plt.close(fig)
    return out


def _ci_band(curves: pd.DataFrame) -> pd.DataFrame:
    per_seed = [g.sort_values("step") for _, g in curves.groupby("seed")]
    grid = np.array(sorted(set.intersection(*(set(g["step"]) for g in per_seed))))
    vals = np.stack([np.interp(grid, g["step"], g["mean_return"]) for g in per_seed])
    idx = np.random.default_rng(0).integers(0, len(per_seed), size=(4000, len(per_seed)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"step": grid, "mean": vals.mean(axis=0), "lo": lo, "hi": hi})


FINAL = "dqn-39b3e6"  # configs/sweeps/final.yaml (seeds 30–49)
STAB_1M = "dqn-0bde89"  # final anterior: referência + estabilizações, 1M passos
SB3_300K, SB3_1M = "sb3-104563", "sb3-72ba93"
CONTROL_SEEDS = range(30, 50)  # seeds em que a final e o controle rodaram juntos


def fig_final(runs: pd.DataFrame) -> dict:
    series = [
        (FINAL, "DQN (nosso), final", C_REF, CONTROL_SEEDS),
        (STAB_1M, "DQN (nosso), lr 5e-4 e alvo 1000", C_ALT, CONTROL_SEEDS),
        (SB3_1M, "DQN (SB3), padrão", C_SB3, range(0, 5)),
    ]
    out = {}
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, (cid, label, color, seed_range) in enumerate(series):
        seeds = runs[(runs["config_id"] == cid) & runs["seed"].isin(list(seed_range))]
        curves = mf.load_curves(RESULTS_DIR, [cid])
        curves = curves[curves["seed"].isin(seeds["seed"])]
        b = _ci_band(curves)
        ax.fill_between(b["step"], b["lo"], b["hi"], color=color, alpha=0.18, linewidth=0)
        ax.plot(b["step"], b["mean"], color=color, linewidth=1.4, marker=mf.MARKERS[i], markersize=3.5,
                markevery=6, label=f"{label} ({len(seeds)} seeds)")
        out[cid] = {"label": label, **stats(seeds)}
    ax.axhline(mf.SOLVED_THRESHOLD, color=mf.TEXT_SECONDARY, linewidth=0.7, linestyle="--")
    ax.xaxis.set_major_formatter(FuncFormatter(mf.format_steps))
    ax.set_xlabel("passos de treino")
    ax.set_ylabel("retorno médio (avaliação)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig-final.pdf")
    plt.close(fig)
    return out


def fig_checkpoint() -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(EXTRA.glob("*/checkpoints.json"))]
    if not rows:
        return {}
    df = pd.DataFrame(
        [{"config_id": r["config_id"], "seed": r["seed"], "last": r["last"]["mean_return"],
          "best": r["best"]["mean_return"], "last_succ": r["last"]["success_rate"],
          "best_succ": r["best"]["success_rate"], "best_step": r["best_step"]} for r in rows]
    )
    groups = [(REF, "referência"), (WINNER, "vencedora da busca"), (df[df.config_id.str.startswith("sb3")].config_id.iloc[0] if (df.config_id.str.startswith("sb3")).any() else "", "SB3 padrão")]
    groups = [(c, lab) for c, lab in groups if c and (df["config_id"] == c).any()]
    out = {}
    fig, axes = plt.subplots(1, len(groups), figsize=(2.0 * len(groups) + 0.6, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (cid, label) in zip(axes, groups):
        g = df[df["config_id"] == cid]
        for _, row in g.iterrows():
            ax.plot([0, 1], [row["last"], row["best"]], color=C_GRAY, linewidth=0.7, alpha=0.8)
        for x, col, color in ((0, "last", C_ALT), (1, "best", C_REF)):
            lo, hi = boot_ci(g[col])
            m = g[col].mean()
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=color, markersize=5, capsize=3, zorder=3)
        ax.set_xticks([0, 1], ["último", "melhor"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f"{label} ({len(g)} seeds)", fontsize=8)
        ax.axhline(mf.SOLVED_THRESHOLD, color=mf.TEXT_SECONDARY, linewidth=0.7, linestyle="--")
        ax.grid(axis="x", visible=False)
        out[cid] = {
            "label": label,
            "n": int(len(g)),
            "last": {"mean": float(g["last"].mean()), "ci": boot_ci(g["last"]), "std": float(g["last"].std(ddof=1)),
                     "success": float(g["last_succ"].mean())},
            "best": {"mean": float(g["best"].mean()), "ci": boot_ci(g["best"]), "std": float(g["best"].std(ddof=1)),
                     "success": float(g["best_succ"].mean())},
            "best_step_median": float(g["best_step"].median()),
        }
    axes[0].set_ylabel("retorno (100 episódios novos)")
    fig.tight_layout()
    fig.savefig(OUT / "fig-checkpoint.pdf")
    plt.close(fig)
    return out


def epsilon_ablation(runs: pd.DataFrame) -> dict:
    """Ablação de ε na configuração final, nas mesmas seeds (comparação pareada)."""
    base = runs[runs["config_id"] == FINAL].set_index("seed")["mean_return"]
    out = {}
    for cid, hp in round_ids("round_15_epsilon"):
        g = runs[runs["config_id"] == cid]
        if not len(g):
            continue
        diff = g.set_index("seed")["mean_return"] - base.reindex(g["seed"]).to_numpy()
        lo, hi = boot_ci(diff.to_numpy())
        out[cid] = {"epsilon_end": hp["epsilon_end"], "epsilon_decay_steps": hp["epsilon_decay_steps"],
                    **stats(g), "diff": float(diff.mean()), "diff_ci": [lo, hi], "wins": int((diff > 0).sum())}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs(RESULTS_DIR)
    fresh = runs[runs["seed"] >= FRESH]

    numbers: dict = {"axes": {}, "steps": {}, "stability": {}}
    search = runs[runs["seed"].isin([0, 1, 2])]
    for name, _, fmt in AXES:
        numbers["axes"][name] = {fmt(hp): stats(search[search["config_id"] == cid]) for cid, hp in sweep_ids(name)}
    for cid, hp in [(REF, {"total_timesteps": 300000})] + round_ids("round_09_steps"):
        g = fresh[fresh["config_id"] == cid]
        if len(g):
            numbers["steps"][str(hp["total_timesteps"])] = stats(g)
    best_stab, best_mean = None, fresh[fresh["config_id"] == REF]["mean_return"].mean()
    for cid, hp in round_ids("round_09_stability") + round_ids("round_09_winner_stable"):
        g = fresh[fresh["config_id"] == cid]
        if len(g):
            numbers["stability"][cid] = {"hp": {k: hp[k] for k in ("max_grad_norm", "learning_rate_end", "batch_size", "hidden_sizes", "train_freq")}, **stats(g)}
            if cid in dict(round_ids("round_09_stability")) and len(g) >= 20 and g["mean_return"].mean() > best_mean:
                best_stab, best_mean = cid, g["mean_return"].mean()

    fig_sweeps(runs)
    numbers["selection"] = fig_selection(runs)
    numbers["final"] = fig_final(runs)
    numbers["stab_1m_10_29"] = stats(fresh[(fresh["config_id"] == STAB_1M) & (fresh["seed"] < 30)])
    for cid in sorted(runs[runs["algo"] == "sb3"]["config_id"].unique()):
        g = runs[runs["config_id"] == cid]
        numbers[f"sb3_{int(g['total_timesteps'].iloc[0])}"] = stats(g)
    numbers["sb3_300k"] = stats(runs[runs["config_id"] == SB3_300K])
    numbers["final_300k"] = stats(fresh[fresh["config_id"] == "dqn-78d3c8"])
    w = fresh[fresh["config_id"] == "dqn-3bc15f"]
    numbers["wide_stab"] = stats(w)
    numbers["epsilon_1m"] = epsilon_ablation(runs)
    numbers["checkpoint"] = fig_checkpoint()
    numbers["n_runs"] = int(len(runs))
    (OUT / "numeros.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False))
    print(f"figuras e números em {OUT}")


if __name__ == "__main__":
    main()
