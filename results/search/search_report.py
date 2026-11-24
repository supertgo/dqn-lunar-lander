"""Gera o relatório HTML da busca de hiperparâmetros (todas as rodadas) a partir de results/runs/.

Uso: uv run python search_report.py <saida.html>
"""

import base64
import html
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path("/home/supertgo/programas/ufmg/rl/dqn-lunar-lander")
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from aggregate import load_runs, summarize  # noqa: E402
from make_figures import (  # noqa: E402
    GRID_COLOR,
    MARKERS,
    SERIES_COLORS,
    SOLVED_THRESHOLD,
    TEXT_SECONDARY,
    format_steps,
    load_curves,
    mean_curve,
)
from matplotlib.ticker import FuncFormatter  # noqa: E402

from dqn.config import load_hyperparameters, load_sweep  # noqa: E402

RESULTS = REPO / "results" / "runs"
ROUNDS_FILE = REPO / "results" / "search" / "rounds.json"
BASELINE = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(load_hyperparameters()).items()}
REF_GRAY = "#8a8f98"
plt.rcParams["savefig.dpi"] = 120


# ---------------------------------------------------------------- dados
def diff_label(hp: dict) -> str:
    """O que a configuração muda em relação à referência, ex.: 'batch 64 · lr 0.001'."""
    short = {
        "learning_rate": "lr", "batch_size": "batch", "target_update_interval": "alvo",
        "buffer_size": "buffer", "epsilon_decay_steps": "ε decai", "epsilon_end": "ε final",
        "epsilon_start": "ε inicial", "train_freq": "train_freq", "hidden_sizes": "rede",
        "gamma": "γ", "learning_starts": "início", "total_timesteps": "passos",
        "max_grad_norm": "clip", "learning_rate_end": "lr fim",
    }
    parts = []
    for k, base in BASELINE.items():
        v = hp.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        v = list(v) if isinstance(v, (tuple, list)) else v
        if v != base:
            if k == "hidden_sizes":
                v = "×".join(str(x) for x in v)
            elif isinstance(v, float) and v.is_integer() and k not in ("learning_rate", "gamma"):
                v = int(v)
            if k == "total_timesteps":
                v = f"{int(v) / 1e6:g}M" if int(v) >= 1_000_000 else f"{int(v) // 1000}k"
            parts.append(f"{short.get(k, k)} {v}")
    return " · ".join(parts) if parts else "referência"


def load_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = load_runs(RESULTS)
    configs = summarize(runs)
    configs["score"] = configs["mean_return"] - configs["mean_return_std"].fillna(0)
    hp_by_config = {}
    for run in sorted(RESULTS.iterdir()):
        cfg_file = run / "config.json"
        if cfg_file.is_file():
            c = json.loads(cfg_file.read_text())
            hp_by_config[c["config_id"]] = c["hyperparameters"]
    configs["label"] = [
        "SB3 (padrão)" if algo == "sb3" else diff_label(hp_by_config.get(cid, {}))
        for cid, algo in zip(configs["config_id"], configs["algo"])
    ]
    return runs, configs


def round_config_ids(rnd: dict) -> list[str]:
    ids: list[str] = []
    for path in rnd["sweeps"]:
        for cfg in load_sweep(REPO / path).run_configs():
            if cfg.config_id not in ids:
                ids.append(cfg.config_id)
    return ids


# ---------------------------------------------------------------- figuras
def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def round_figure(configs: pd.DataFrame, ids: list[str], sb3: pd.Series | None, title: str) -> str:
    """Esq.: curvas das 5 melhores da rodada (+ referência em cinza). Dir.: todas as configs, ordenadas."""
    rows = configs[configs["config_id"].isin(ids)].sort_values("score", ascending=False)
    if rows.empty:
        return ""
    top = [cid for cid in rows["config_id"] if rows.set_index("config_id").loc[cid, "label"] != "referência"][:5]
    ref = rows[rows["label"] == "referência"]["config_id"].tolist()
    curves = load_curves(RESULTS, top + ref + ([sb3["config_id"]] if sb3 is not None else []))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, max(3.6, 0.32 * len(rows) + 1.2)), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for i, cid in enumerate(top):
        one = curves[curves["config_id"] == cid]
        if one.empty:
            continue
        m = mean_curve(one)
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax1.fill_between(m["step"], m["mean"] - m["std"], m["mean"] + m["std"], color=color, alpha=0.13, lw=0)
        ax1.plot(m["step"], m["mean"], color=color, lw=1.6, marker=MARKERS[i], markersize=4,
                 markevery=20, label=rows.set_index("config_id").loc[cid, "label"])
    for cid, name, color, style in ([(c, "referência", REF_GRAY, "-") for c in ref]
                                    + ([(sb3["config_id"], "SB3 (padrão)", "#0b0b0b", ":")] if sb3 is not None else [])):
        one = curves[curves["config_id"] == cid]
        if not one.empty:
            m = mean_curve(one)
            ax1.plot(m["step"], m["mean"], color=color, lw=1.4, ls=style, label=name)
    ax1.axhline(SOLVED_THRESHOLD, color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax1.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax1.set_xlabel("passos de treino")
    ax1.set_ylabel("retorno médio (avaliação periódica)")
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.set_title("curvas de aprendizado (até 5 melhores)", fontsize=9, loc="left", color=TEXT_SECONDARY)

    y = np.arange(len(rows))[::-1]
    colors = [REF_GRAY if lab == "referência" else SERIES_COLORS[0] for lab in rows["label"]]
    for yi, mean, std, color in zip(y, rows["mean_return"], rows["mean_return_std"].fillna(0), colors):
        ax2.errorbar(mean, yi, xerr=std, fmt="o", color=color, markersize=5, elinewidth=1.2, capsize=2.5)
    ax2.set_yticks(y, [f"{lab}  (n={n})" for lab, n in zip(rows["label"], rows["n_seeds"])], fontsize=7.5)
    ax2.axvline(SOLVED_THRESHOLD, color=TEXT_SECONDARY, lw=0.8, ls="--")
    if sb3 is not None:
        ax2.axvline(sb3["mean_return"], color="#0b0b0b", lw=1, ls=":")
        ax2.text(sb3["mean_return"], len(rows) - 0.4, " SB3", fontsize=7.5, va="bottom")
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("retorno na avaliação final (média ± desvio entre seeds)")
    ax2.set_title("todas as configurações (ordenadas por score)", fontsize=9, loc="left", color=TEXT_SECONDARY)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return png(fig)


def progress_figure(history: list[tuple[int, float, str]]) -> str:
    if not history:
        return ""
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    xs = [h[0] for h in history]
    ys = [h[1] for h in history]
    ax.plot(xs, ys, color=SERIES_COLORS[0], lw=1.8, marker="o", markersize=5)
    ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel("rodada")
    ax.set_ylabel("melhor score")
    ax.grid(color=GRID_COLOR)
    fig.tight_layout()
    return png(fig)


# ---------------------------------------------------------------- html
def fmt(v, digits=1):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def table(rows: pd.DataFrame, best_id: str | None, first_round: dict) -> str:
    out = ['<div class="tablewrap"><table><thead><tr><th>configuração (diferença da referência)</th>'
           '<th class="num">retorno</th><th class="num">desvio</th><th class="num">score</th>'
           '<th class="num">sucesso</th><th class="num">seeds</th><th class="num">rodada</th></tr></thead><tbody>']
    for _, r in rows.iterrows():
        cls = ' class="best"' if r["config_id"] == best_id else (' class="sb3"' if r["algo"] == "sb3" else "")
        solved = "chip ok" if r["mean_return"] >= SOLVED_THRESHOLD else "chip"
        out.append(
            f'<tr{cls}><td><span class="cfg">{html.escape(r["label"])}</span>'
            f'<span class="cid">{r["config_id"]}</span></td>'
            f'<td class="num"><span class="{solved}">{fmt(r["mean_return"])}</span></td>'
            f'<td class="num">±{fmt(r["mean_return_std"])}</td><td class="num strong">{fmt(r["score"])}</td>'
            f'<td class="num">{r["success_rate"]:.0%}</td><td class="num">{int(r["n_seeds"])}</td>'
            f'<td class="num">{first_round.get(r["config_id"], "–")}</td></tr>'
        )
    out.append("</tbody></table></div>")
    return "".join(out)


CSS = """
:root{--bg:#f6f7f9;--panel:#ffffff;--ink:#14171c;--muted:#5b6270;--line:#e0e3e8;--accent:#2a78d6;
--ok:#0a7d4f;--ok-bg:#e3f4ec;--warn-bg:#fdf1dc;--best:#eaf2fc;--plate:#ffffff;
--display:"Space Grotesk",system-ui,sans-serif;--body:"Inter",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111317;--panel:#181b21;--ink:#e9ebef;
--muted:#9aa1ad;--line:#2a2f38;--accent:#5b9ef0;--ok:#4fd39a;--ok-bg:#16352a;--warn-bg:#3a2f18;--best:#1a2a40;--plate:#f4f5f7;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#111317;--panel:#181b21;--ink:#e9ebef;--muted:#9aa1ad;--line:#2a2f38;--accent:#5b9ef0;
--ok:#4fd39a;--ok-bg:#16352a;--warn-bg:#3a2f18;--best:#1a2a40;--plate:#f4f5f7;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--body);line-height:1.55}
.wrap{max-width:1080px;margin:0 auto;padding-block:32px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;line-height:1.2}
h1{font-size:2rem}h2{font-size:1.35rem}h3{font-size:1.05rem}
p{margin:0;max-width:72ch}.muted{color:var(--muted)}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header{display:grid;gap:10px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:grid;gap:4px;min-width:0}
.stat b{font-family:var(--mono);font-size:1.45rem;font-variant-numeric:tabular-nums}
.stat span{font-size:.8rem;color:var(--muted)}
.stat.lead{border-color:var(--accent)}
section{display:grid;gap:14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;display:grid;gap:12px;min-width:0}
.rule{font-size:.9rem;display:grid;gap:6px}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th{font-family:var(--mono);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);text-align:left;
padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
td.num,th.num{text-align:right;font-family:var(--mono);white-space:nowrap}
td.strong{font-weight:600}
tr.best td{background:var(--best)}
tr.sb3 td{background:var(--warn-bg)}
.cfg{display:block}.cid{display:block;font-family:var(--mono);font-size:.7rem;color:var(--muted)}
.chip{display:inline-block;padding:1px 7px;border-radius:999px}
.chip.ok{background:var(--ok-bg);color:var(--ok)}
.round{border-top:1px solid var(--line);padding-top:22px;display:grid;gap:12px}
.round-head{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline}
.status{font-family:var(--mono);font-size:.7rem;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.status.running{border-color:var(--accent);color:var(--accent)}
.status.done{border-color:var(--ok);color:var(--ok)}
figure{margin:0;background:var(--plate);border-radius:10px;padding:10px;overflow-x:auto}
figure img{display:block;max-width:100%;height:auto}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
footer{font-size:.8rem;color:var(--muted)}
"""


def build() -> str:
    rounds = json.loads(ROUNDS_FILE.read_text())
    runs, configs = load_table()
    dqn = configs[configs["algo"] == "dqn"].copy()
    sb3_rows = configs[configs["algo"] == "sb3"]
    sb3 = sb3_rows.iloc[0] if not sb3_rows.empty else None

    first_round: dict[str, int] = {}
    history: list[tuple[int, float, str]] = []
    for rnd in rounds:
        ids = round_config_ids(rnd)
        for cid in ids:
            first_round.setdefault(cid, rnd["round"])
        if rnd["status"] == "done":
            seen = dqn[dqn["config_id"].isin(first_round)]
            if not seen.empty:
                b = seen.sort_values("score", ascending=False).iloc[0]
                history.append((rnd["round"], float(b["score"]), b["config_id"]))

    best = dqn.sort_values("score", ascending=False).iloc[0]
    done_rounds = [r for r in rounds if r["status"] == "done"]
    running = [r for r in rounds if r["status"] != "done"]
    no_gain = 0
    for prev, cur in zip(history, history[1:]):
        no_gain = no_gain + 1 if cur[1] < prev[1] + 5 else 0

    numbers = json.loads((REPO / "report" / "figures" / "numeros.json").read_text())
    fin = numbers["final"]["dqn-39b3e6"]
    ctrl = numbers["final"]["dqn-0bde89"]
    sb3_1m, sb3_2m = numbers["sb3_1000000"], numbers["sb3_2000000"]
    a = runs[runs["config_id"] == "dqn-39b3e6"].set_index("seed")["mean_return"]
    c = runs[(runs["config_id"] == "dqn-0bde89") & (runs["seed"] >= 30)].set_index("seed")["mean_return"]
    d = (a - c).dropna().to_numpy()
    boot = np.random.default_rng(0).choice(d, size=(10000, len(d))).mean(axis=1)
    paired = (float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int((d > 0).sum()))
    png_bytes = subprocess.run(["pdftoppm", "-png", "-r", "130", "-singlefile", str(REPO / "report" / "figures" / "fig-final.pdf")],
                               capture_output=True, check=True).stdout
    final_png = "data:image/png;base64," + base64.b64encode(png_bytes).decode()

    parts = [
        '<title>Diário da Busca DQN</title>',
        '<meta name="claude:viewport-fit-sides" content="cover">',
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&'
        'family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap">',
        f"<style>{CSS}</style>",
        '<main class="wrap">',
        "<header>",
        '<span class="eyebrow">TP DQN · Lunar Lander · busca de hiperparâmetros</span>',
        "<h1>Diário da Busca DQN</h1>",
        f'<p class="muted">Atualizado em {datetime.now():%d/%m/%Y %H:%M} · {len(runs)} runs concluídas · '
        f"{len(done_rounds)} rodada(s) concluída(s)"
        + (f" · rodada {running[0]['round']} em andamento" if running else "")
        + "</p></header>",
        '<div class="stats">',
        f'<div class="stat lead"><span>configuração final · 1M passos</span><b>{fmt(fin["mean"])}</b>'
        f'<span>IC 95% [{fmt(fin["ci"][0], 0)}, {fmt(fin["ci"][1], 0)}] · {fin["n"]} seeds que nunca participaram da escolha (30–49)</span></div>',
        f'<div class="stat"><span>pousos bem-sucedidos (≥ 200)</span><b>{fin["success"]:.0%}</b>'
        f'<span>pior seed: {fmt(fin["min"])}</span></div>',
        f'<div class="stat"><span>SB3 padrão · 1M passos</span><b>{fmt(sb3_1m["mean"])}</b>'
        f'<span>{sb3_1m["success"]:.0%} de pousos · com 2M: {fmt(sb3_2m["mean"])}</span></div>',
        f'<div class="stat"><span>ganho sobre a final anterior</span><b>+{fmt(paired[0])}</b>'
        f'<span>mesmas seeds, IC 95% [{fmt(paired[1])}, {fmt(paired[2])}]</span></div>',
        "</div>",
        '<section class="panel rule"><h3>Resultado final</h3>'
        "<p><b>Configuração final:</b> a referência (lr 5e-4, γ 0.99, batch 128, buffer 100k, rede 256×256) com "
        "<b>lr inicial 1e-3 decaindo até 5e-5</b>, <b>rede alvo a cada 2000 passos</b>, <b>clipping do gradiente em 10</b> "
        "e <b>1 milhão de passos</b>.</p>"
        f'<figure><img alt="curvas de aprendizado da configuração final, do controle e do SB3" src="{final_png}"></figure>'
        "<p>Ela foi medida nas seeds 30–49, junto com a final anterior (lr 5e-4, alvo 1000) nas mesmas seeds, como controle: "
        f"{fmt(fin['mean'])} contra {fmt(ctrl['mean'])}. Comparando seed a seed, a nova vence em {paired[3]} de {fin['n']}.</p></section>",
        '<section class="panel rule"><h3>Como a busca funciona</h3>'
        "<p><b>Score</b> = retorno médio na avaliação final (100 episódios, seeds fixas) menos o desvio entre as seeds "
        "de treino. Ele premia configurações que resolvem o pouso de forma consistente, e não só em uma seed de sorte.</p>"
        "<p>As rodadas 0 a 10 usaram 300 mil passos: cada uma <b>confirmava</b> as líderes com mais seeds e <b>explorava</b> "
        "vizinhanças em torno delas. A lição principal foi o <b>viés de seleção</b>: configurações escolhidas nas seeds 0–9 "
        "caíam nas seeds novas. Por isso as rodadas 11 a 14, já com 1 milhão de passos, separam as seeds: 10–19 para escolher, "
        "20–29 para confirmar e 30–49 para o teste final, com controle.</p></section>",
    ]
    prog = progress_figure(history)
    if prog:
        parts.append(f'<section><h2>Evolução do melhor score</h2><figure><img alt="melhor score por rodada" src="{prog}"></figure></section>')

    lb = pd.concat([dqn.sort_values("score", ascending=False).head(15), sb3_rows])
    parts.append("<section><h2>Classificação geral</h2>"
                 '<p class="muted">As 15 melhores configurações de todas as rodadas, mais o SB3. '
                 "Retorno em verde quando a média passa do limiar de pouso resolvido (200).</p>"
                 + table(lb, best["config_id"], first_round) + "</section>")

    parts.append("<section><h2>Rodadas</h2>")
    for rnd in sorted(rounds, key=lambda r: -r["round"]):
        ids = round_config_ids(rnd)
        rows = configs[configs["config_id"].isin(ids)].sort_values("score", ascending=False)
        status = "done" if rnd["status"] == "done" else "running"
        parts.append(f'<article class="round"><div class="round-head"><h3>Rodada {rnd["round"]} · {html.escape(rnd["title"])}</h3>'
                     f'<span class="status {status}">{"concluída" if status == "done" else "em andamento"}</span>'
                     f'<span class="muted">{len(rows)} de {len(ids)} configurações com resultado</span></div>')
        parts.append(f"<p><b>Proposta.</b> {html.escape(rnd['rationale'])}</p>")
        if rnd.get("conclusion"):
            parts.append(f"<p><b>Resultado.</b> {html.escape(rnd['conclusion'])}</p>")
        if not rows.empty:
            parts.append(table(rows, best["config_id"], first_round))
            if rnd["round"] == 0:
                for path in rnd["sweeps"]:
                    sw_ids = []
                    for cfg in load_sweep(REPO / path).run_configs():
                        if cfg.config_id not in sw_ids:
                            sw_ids.append(cfg.config_id)
                    img = round_figure(configs, sw_ids, sb3, f"sweep {Path(path).stem}")
                    if img:
                        parts.append(f'<figure><img alt="sweep {Path(path).stem}" src="{img}"></figure>')
            else:
                img = round_figure(configs, ids, sb3, f"rodada {rnd['round']}")
                if img:
                    parts.append(f'<figure><img alt="rodada {rnd["round"]}" src="{img}"></figure>')
        parts.append("</article>")
    parts.append("</section>")
    parts.append('<footer>Gerado a partir de results/runs/ e results/search/rounds.json na branch experiments. '
                 "Score e desvio usam o desvio amostral entre seeds de treino; o retorno de cada seed é a média "
                 "de 100 episódios de avaliação gulosa.</footer></main>")
    return "\n".join(parts)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.2f} MB")
