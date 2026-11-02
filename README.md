# DQN no Lunar Lander

Trabalho prático de Aprendizado por Reforço (UFMG, 2026/2): implementação do Deep Q-Network
do zero (NumPy + PyTorch) no ambiente Lunar Lander do Gymnasium, busca de hiperparâmetros e
comparação com o DQN do Stable Baselines3. A especificação está em [`docs/RL_TP_Grad.pdf`](docs/RL_TP_Grad.pdf).

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
│       ├── final.yaml           # melhor configuração (comparação com SB3)
│       └── smoke.yaml           # teste rápido de ponta a ponta
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
│   └── make_figures.py          # gera as figuras do relatório
├── results/runs/<run_id>/       # resultados versionados (só arquivos leves)
├── tests/                       # testes dos componentes + regra de imports
├── report/                      # relatório LaTeX (main.tex + figures/)
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
