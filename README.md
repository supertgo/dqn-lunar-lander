# DQN no Lunar Lander

Trabalho prático de Aprendizado por Reforço (UFMG, 2026/2): implementação do Deep Q-Network
do zero (NumPy + PyTorch) no ambiente Lunar Lander do Gymnasium, busca de hiperparâmetros e
comparação com o DQN do Stable Baselines3. A especificação está em [`docs/RL_TP_Grad.pdf`](docs/RL_TP_Grad.pdf).

## Pacotes e dependências

### Requisitos

- Python 3.11 ou 3.12
- [`uv`](https://docs.astral.sh/uv/) para criar o ambiente e instalar as dependências travadas
- `make` para utilizar os comandos automatizados do projeto
- Linux, Windows ou macOS

As dependências do projeto são declaradas em [`pyproject.toml`](pyproject.toml) e travadas em
[`uv.lock`](uv.lock). Para instalar o ambiente padrão, execute:

```bash
make install
```

### Dependências principais

| Pacote | Uso no projeto |
|---|---|
| `gymnasium[box2d]` | Criação e execução do ambiente LunarLander |
| `torch` | Rede neural, otimização e treinamento do DQN |
| `numpy` | Manipulação de estados, ações e dados numéricos |
| `pandas` | Agregação e análise dos resultados das execuções |
| `matplotlib` | Geração dos gráficos e figuras do relatório |
| `pyyaml` | Leitura dos arquivos de configuração em YAML |

### Dependências opcionais e de desenvolvimento

| Pacote | Instalação/uso |
|---|---|
| `stable-baselines3` | Dependência opcional do baseline DQN em `scripts/run_sb3.py`; instale com `uv sync --locked --extra sb3` |
| `pytest` | Execução dos testes automatizados |
| `ruff` | Verificação de estilo e lint |

Os pacotes opcionais de desenvolvimento podem ser instalados junto com o ambiente principal com:

```bash
uv sync --locked --extra sb3
```

## Arquitetura

O trabalho é feito por três pessoas em máquinas diferentes. A arquitetura existe para garantir três coisas:

1. **Ambiente idêntico em qualquer máquina.** Dependências travadas com `uv` (`uv.lock` no git).
2. **Runs distribuídas sem colisão.** Cada run tem um id determinístico, derivado da sua
   configuração, então a mesma run feita em PCs diferentes cai no mesmo diretório.
3. **Uma fonte única de resultados.** As métricas (CSV/JSON leves) vão para o git, e todas as
   figuras do relatório são geradas por script a partir delas.

## Estrutura

```
dqn-lunar-lander/
├── pyproject.toml / uv.lock     # dependências
├── .python-version
├── configs/
│   ├── hyperparameters.yaml     # hiperparâmetros de referência (baseline da busca)
│   ├── evaluation.yaml          # protocolo de avaliação — fixo para DQN e SB3
│   └── sweeps/                  # um arquivo por eixo da busca
│       ├── learning_rate.yaml  gamma.yaml  buffer_size.yaml
│       ├── batch_size.yaml  target_update.yaml  epsilon.yaml
│       ├── final.yaml           # configuração final (1M passos, seeds 30–49)
│       ├── smoke.yaml           # teste rápido de ponta a ponta
│       ├── rounds/              # rodadas 1 a 14 da busca (a rodada 0 são os sweeps de um fator acima)
│       └── extra/               # análise extra do melhor checkpoint
├── src/dqn/                     # pacote do DQN — nunca importa SB3/RLlib/Keras-RL
│   ├── network.py               # rede Q
│   ├── replay_buffer.py         # replay buffer
│   ├── exploration.py           # schedule de ε
│   ├── agent.py                 # DQNAgent: act, update, sync_target, save
│   ├── train.py                 # loop de treino + avaliação periódica + logging
│   ├── evaluate.py              # avaliação gulosa de qualquer policy_fn(obs) -> ação
│   ├── config.py                # leitura dos YAMLs, overrides, run_id/config_id
│   └── run_logger.py            # escreve a run no formato padrão
├── scripts/
│   ├── run_sweep.py             # expande sweep × seeds e roda em paralelo
│   ├── run_sb3.py               # baseline SB3 (único lugar que importa SB3)
│   ├── aggregate.py             # junta results/runs/ em tabelas
│   ├── make_figures.py          # uma figura por sweep de um fator
│   ├── report_figures.py        # figuras e números do relatório (report/figures/)
│   └── eval_checkpoints.py      # análise extra: último modelo vs. melhor checkpoint
├── results/
│   ├── runs/<run_id>/           # todas as runs da busca, da final e do SB3 (só arquivos leves)
│   ├── extra/best_checkpoint/   # runs da análise do melhor checkpoint
│   └── search/                  # diário da busca: rounds.json, logs e resumo da fila final
├── tests/                       # testes dos componentes + regra de imports
├── report/                      # relatório LaTeX (main.tex, referencias.bib, figures/)
└── docs/                        # especificação do trabalho
```

## Componentes

### Núcleo do DQN (`src/dqn/`)

| Elemento da spec | Arquivo |
|---|---|
| Rede neural para estimar os Q-values | `network.py` |
| Rede alvo e mecanismo de atualização | `agent.py` (`sync_target`) |
| Replay buffer | `replay_buffer.py` |
| Política ε-greedy com decaimento | `exploration.py` + `agent.act` |
| Treinamento | `train.py` + `agent.update` |
| Avaliação | `evaluate.py` |

### Configuração (`configs/` + `config.py`)

Uma run é definida por **(algoritmo, hiperparâmetros, protocolo de avaliação, seed)**.

- `hyperparameters.yaml` guarda a configuração de referência.
- `evaluation.yaml` guarda o protocolo de avaliação. Ele é separado dos hiperparâmetros porque
  não faz parte da busca e precisa ser o mesmo para todas as runs.
- Cada sweep lista `seeds` e `variants`. Cada variant sobrescreve parte do
  `hyperparameters.yaml`, e todas as variants rodam com todas as seeds.

Identificação:

- `config_id = <algo>-<hash de tudo menos a seed>` agrupa as seeds de uma mesma configuração.
- `run_id = <config_id>_s<seed>` dá nome ao diretório em `results/runs/`.

Como o id depende só do conteúdo, uma configuração repetida entre sweeps roda uma vez só,
sweeps interrompidos podem ser retomados e resultados de protocolos diferentes não se misturam.

### Formato de saída de uma run (`run_logger.py`)

Este formato é o contrato entre o treino, a agregação e as figuras.

```
results/runs/<run_id>/
├── config.json    configuração completa + run_id/config_id
├── meta.json      hostname, commit git, versões, horário
├── train.csv      métricas por episódio de treino
├── eval.csv       métricas por avaliação periódica
├── summary.json   avaliação final; sua existência marca a run como concluída
└── model.pt       pesos finais — fora do git
```

### Avaliação (`evaluate.py`)

`evaluate` recebe uma função `obs -> ação` e roda os episódios definidos em `evaluation.yaml`,
com seeds fixas. **O mesmo código avalia o nosso DQN e o SB3**, então os dois enfrentam os mesmos
episódios e são medidos pelos mesmos critérios.

### Comparação com o Stable Baselines3 (`scripts/run_sb3.py`)

Treina o DQN do SB3 e escreve no mesmo formato de `results/runs/`. O SB3 é dependência
opcional e só este script o importa. `tests/test_no_forbidden_imports.py` garante que `src/dqn/`
não importa bibliotecas proibidas pela spec.

### Agregação e figuras (`scripts/aggregate.py`, `scripts/make_figures.py`)

- `aggregate.py` lê as runs concluídas e gera tabelas em `results/aggregated/`, que fica fora do git.
- `make_figures.py` lê cada sweep de `configs/sweeps/`, encontra as runs pelo `run_id` e gera as
  figuras em `report/figures/`.

## Executar

Com `uv` instalado, use os comandos do `Makefile`:

```bash
make install                 # instala as dependências de uv.lock
make run                     # teste rápido ponta a ponta (sweep smoke)
make sweeps                  # executa os sweeps de hiperparâmetros e o final
make plots                   # gera as figuras em report/figures/
make seminar-plots           # gera as figuras e números do relatório em report/figures/
make all WORKERS=4           # executa os sweeps em até 4 processos e gera as figuras
```

As runs concluídas são ignoradas automaticamente, então é possível repetir `make sweeps` para retomar execuções interrompidas. `make plots` usa as runs já presentes em `results/runs/` e avisa quais sweeps ainda não têm resultados; use `make all` para executar os experimentos antes de gerar as figuras. `make seminar-plots` executa `scripts/report_figures.py`, que lê as runs versionadas em `results/` e as rodadas em `configs/sweeps/rounds/`.

## Reproduzir os resultados do relatório

Todas as runs usadas no relatório estão em `results/` (sem os pesos dos modelos), então as
figuras e tabelas podem ser regeneradas sem treinar nada: `make seminar-plots`.

Para treinar de novo:

```bash
# configuração final: 20 seeds (30–49), 1M passos, ~20–30 min por run em 1 núcleo
uv run --locked python scripts/run_sweep.py configs/sweeps/final.yaml --workers 4
# SB3 padrão com o mesmo orçamento
uv run --locked --extra sb3 python scripts/run_sb3.py --total-timesteps 1000000 --seeds 0 1 2 3 4
# uma rodada específica da busca, por exemplo o estudo do número de passos
uv run --locked python scripts/run_sweep.py configs/sweeps/rounds/round_09_steps.yaml --workers 4
# análise extra do melhor checkpoint
uv run --locked python scripts/run_sweep.py configs/sweeps/extra/*.yaml --workers 4
uv run --locked --extra sb3 python scripts/run_sb3.py --seeds 0 1 2 3 4 --results-dir results/extra/best_checkpoint
uv run --locked --extra sb3 python scripts/eval_checkpoints.py results/extra/best_checkpoint
```

Os ids são determinísticos, então uma run refeita cai no mesmo diretório e dá o mesmo
resultado. A análise extra refez 35 runs e reproduziu exatamente os números originais.