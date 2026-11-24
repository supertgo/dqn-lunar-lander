"""Fila de experimentos em torno da configuração final (1M passos), etapa após etapa.

Roda na raiz do projeto:  uv run --extra sb3 python -u queue_experiments.py [--dry-run]
Pode ser reiniciado a qualquer momento: runs concluídas são puladas e as escolhas de
cada etapa dependem só dos resultados das etapas anteriores, então se repetem iguais.

Etapas
  11 vizinhança  — um fator por vez em torno da final, seeds 10–19 (seleção)
  12 confirmação — até 3 melhores da etapa 11 nas seeds 20–29 (estimativa sem viés)
  13 combinação  — junta as mudanças confirmadas, seeds 30–49, com a final como controle
                    nas mesmas seeds; se só uma foi confirmada, ela contra o controle
  14 orçamento   — final com 2M passos (seeds 10–29) e SB3 padrão com 2M (seeds 0–4)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from dqn.config import RESULTS_DIR, dqn_run_config

ROOT = Path.cwd()
ROUNDS = ROOT / "configs" / "sweeps" / "rounds"
SUMMARY = ROOT / "queue_summary.json"
DRY = "--dry-run" in sys.argv

FINAL = {"max_grad_norm": 10.0, "learning_rate_end": 0.00005, "total_timesteps": 1_000_000}
S_SEL, S_CONF, S_COMBO, S_2M = range(10, 20), range(20, 30), range(30, 50), range(10, 30)

NEIGHBOURHOOD = {  # nome curto -> mudança em relação à final
    "lr 1e-3": {"learning_rate": 0.001},
    "lr fim 1e-5": {"learning_rate_end": 0.00001},
    "alvo 500": {"target_update_interval": 500},
    "alvo 2000": {"target_update_interval": 2000},
    "batch 256": {"batch_size": 256},
    "buffer 300k": {"buffer_size": 300000},
    "ε decai 100k": {"epsilon_decay_steps": 100000},
    "γ 0.995": {"gamma": 0.995},
    "train_freq 2": {"train_freq": 2},
    "clip 1": {"max_grad_norm": 1.0},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def workers(default: int = 25) -> int:
    """90% dos núcleos se a máquina estiver livre; metade se houver outro usuário pesado."""
    out = subprocess.run(["ps", "-eo", "user:30,pcpu", "--no-headers"], capture_output=True, text=True).stdout
    me = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
    others = sum(float(l.split()[1]) for l in out.splitlines() if l.split() and l.split()[0] != me)
    w = default if others < 200 else 12
    log(f"CPU de outros usuários: {others:.0f}% -> {w} workers")
    return w


def write_sweep(name: str, seeds, variants: list[dict], comment: str) -> Path:
    path = ROUNDS / f"{name}.yaml"
    body = yaml.safe_dump(
        {"name": name, "seeds": list(seeds), "variants": variants}, sort_keys=False, default_flow_style=None
    )
    path.write_text("".join(f"# {line}\n" for line in comment.splitlines()) + body, encoding="utf-8")
    log(f"sweep {path.name}: {len(variants)} variante(s) x {len(list(seeds))} seeds")
    return path


def run(sweeps: list[Path], sb3: tuple[int, list[int]] | None = None, n_workers: int = 25) -> None:
    if DRY:
        log(f"(dry-run) rodaria {[p.name for p in sweeps]} sb3={sb3}")
        return
    procs = [subprocess.Popen([sys.executable, "-u", "scripts/run_sweep.py", *map(str, sweeps), "--workers", str(n_workers)])]
    if sb3:
        steps, seeds = sb3
        procs += [
            subprocess.Popen([sys.executable, "-u", "scripts/run_sb3.py", "--total-timesteps", str(steps), "--seeds", str(s)])
            for s in seeds
        ]
    codes = [p.wait() for p in procs]
    if any(codes):
        log(f"ERRO: códigos de saída {codes}; rode de novo para retomar")
        sys.exit(1)


def returns(variant: dict, seeds) -> np.ndarray:
    vals = []
    for s in seeds:
        f = RESULTS_DIR / dqn_run_config(s, {**FINAL, **variant}).run_id / "summary.json"
        if f.is_file():
            vals.append(json.loads(f.read_text())["mean_return"])
    return np.array(vals)


def describe(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {"n": int(len(x)), "mean": float(x.mean()), "std": sd, "score": float(x.mean() - sd), "min": float(x.min())}


def merge(changes: list[dict]) -> dict | None:
    out: dict = {}
    for c in changes:
        if set(c) & set(out):
            return None  # mudanças no mesmo hiperparâmetro não se combinam
        out.update(c)
    return out


def main() -> None:
    summary: dict = {"final": {}, "stages": {}}
    w = workers()

    # ---- etapa 11: vizinhança
    variants = [{**FINAL, **c} for c in NEIGHBOURHOOD.values()]
    p11 = write_sweep("round_11_neighbourhood", S_SEL, variants,
                      "Etapa 11 — um fator por vez em torno da final (1M passos), seeds 10–19 para seleção.")
    run([p11], n_workers=w)
    base_sel = describe(returns({}, S_SEL))
    stage = {name: describe(returns(c, S_SEL)) for name, c in NEIGHBOURHOOD.items()}
    summary["final"]["seeds_10_19"] = base_sel
    summary["stages"]["11"] = stage
    ranked = sorted((n for n in stage if stage[n].get("n")), key=lambda n: stage[n]["score"], reverse=True)
    better = [n for n in ranked if base_sel.get("n") and stage[n]["mean"] > base_sel["mean"]][:3]
    chosen = better or ranked[:2]
    log(f"final nas seeds 10–19: {base_sel}")
    for n in ranked:
        log(f"  {n:14s} {stage[n]}")
    log(f"para confirmar: {chosen} ({'melhores que a final' if better else 'nenhuma superou; confirmando as 2 de maior score'})")

    # ---- etapa 12: confirmação em seeds novas
    if chosen:
        p12 = write_sweep("round_12_confirm", S_CONF, [{**FINAL, **NEIGHBOURHOOD[n]} for n in chosen],
                          f"Etapa 12 — confirmação nas seeds 20–29 de: {', '.join(chosen)}.")
        w = workers()
        run([p12], n_workers=w)
    else:
        log("etapa 12 pulada: etapa 11 sem resultados")
    base_all = describe(returns({}, list(S_SEL) + list(S_CONF)))
    conf = {n: {"seeds_20_29": describe(returns(NEIGHBOURHOOD[n], S_CONF)),
                "seeds_10_29": describe(returns(NEIGHBOURHOOD[n], list(S_SEL) + list(S_CONF)))} for n in chosen}
    summary["final"]["seeds_10_29"] = base_all
    summary["stages"]["12"] = conf
    confirmed = sorted((n for n in chosen if conf[n]["seeds_20_29"].get("n")
                        and conf[n]["seeds_10_29"]["mean"] > base_all.get("mean", float("inf"))),
                       key=lambda n: conf[n]["seeds_10_29"]["mean"], reverse=True)
    log(f"final nas seeds 10–29: {base_all}")
    for n in chosen:
        log(f"  {n:14s} {conf[n]}")
    log(f"confirmadas (média em 20 seeds acima da final): {confirmed}")

    # ---- etapa 13: combinação (ou a confirmada sozinha) contra a final nas seeds 30–49
    combos: dict[str, dict] = {}
    if len(confirmed) >= 2:
        for k in (2, 3):
            if len(confirmed) >= k:
                m = merge([NEIGHBOURHOOD[n] for n in confirmed[:k]])
                if m is not None:
                    combos[" + ".join(confirmed[:k])] = m
    if len(confirmed) == 1 or (len(confirmed) >= 2 and not combos):
        combos[confirmed[0]] = NEIGHBOURHOOD[confirmed[0]]
    if combos:
        p13 = write_sweep("round_13_combo", S_COMBO, [dict(FINAL)] + [{**FINAL, **c} for c in combos.values()],
                          "Etapa 13 — combinação das mudanças confirmadas, seeds 30–49, com a final como controle.")
        w = workers()
        run([p13], n_workers=w)
        summary["stages"]["13"] = {"controle (final)": describe(returns({}, S_COMBO)),
                                   **{n: describe(returns(c, S_COMBO)) for n, c in combos.items()}}
        for n, v in summary["stages"]["13"].items():
            log(f"  {n:30s} {v}")
    else:
        log("etapa 13 pulada: nenhuma mudança confirmada")

    # ---- etapa 14: orçamento de 2M passos
    p14 = write_sweep("round_14_2M", S_2M, [{**FINAL, "total_timesteps": 2_000_000}],
                      "Etapa 14 — final com 2M passos (lr decai ao longo dos 2M), seeds 10–29; SB3 com 2M em paralelo.")
    w = workers(default=22)  # + 5 processos do SB3
    run([p14], sb3=(2_000_000, [0, 1, 2, 3, 4]), n_workers=w)
    summary["stages"]["14"] = {"final 2M": describe(returns({"total_timesteps": 2_000_000}, S_2M))}
    log(f"  final 2M {summary['stages']['14']['final 2M']}")

    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"FILA CONCLUÍDA — resumo em {SUMMARY.name}")


if __name__ == "__main__":
    main()
