.DEFAULT_GOAL := help

UV ?= uv
WORKERS ?= 1
SWEEPS := $(filter-out configs/sweeps/smoke.yaml,$(wildcard configs/sweeps/*.yaml))
SEMINAR_SWEEPS := \
	configs/sweeps/rounds/round_09_steps.yaml \
	configs/sweeps/rounds/round_09_stability.yaml \
	configs/sweeps/rounds/round_09_winner_stable.yaml

.PHONY: help install run sweeps plots seminar-plots all entrega

help:
	@echo "Comandos disponíveis:"
	@echo "  make install          instala as dependências travadas"
	@echo "  make run              executa o sweep smoke (teste ponta a ponta)"
	@echo "  make sweeps           executa os sweeps, exceto o smoke"
	@echo "  make plots            gera as figuras em report/figures/"
	@echo "  make seminar-plots    gera as figuras do relatório em report/figures/"
	@echo "  make all              executa os sweeps e gera as figuras"
	@echo "  make sweeps WORKERS=4 executa até 4 runs em paralelo"
	@echo "  make entrega          gera dqn-lunar-lander.zip com o que está commitado (exige report/relatorio.pdf)"

install:
	$(UV) sync --locked

run:
	$(UV) run --locked python scripts/run_sweep.py configs/sweeps/smoke.yaml --workers $(WORKERS)

sweeps:
	$(UV) run --locked python scripts/run_sweep.py $(SWEEPS) --workers $(WORKERS)

plots:
	$(UV) run --locked python scripts/make_figures.py

seminar-plots:
	@missing=0; \
	for config in $(SEMINAR_SWEEPS); do \
		if [ ! -f "$$config" ]; then \
			echo "ERRO: configuração necessária ausente: $$config" >&2; \
			missing=1; \
		fi; \
	done; \
	if [ "$$missing" -ne 0 ]; then \
		echo "Restaure esses arquivos locais em configs/sweeps/rounds/ e tente novamente." >&2; \
		exit 1; \
	fi
	$(UV) run --locked python scripts/report_figures.py

all: sweeps plots

entrega:
	@git ls-files --error-unmatch report/relatorio.pdf >/dev/null 2>&1 || \
		{ echo "ERRO: compile report/main.tex, salve como report/relatorio.pdf e faça commit." >&2; exit 1; }
	git archive --format=zip --prefix=dqn-lunar-lander/ -o dqn-lunar-lander.zip HEAD
	@echo "dqn-lunar-lander.zip gerado a partir do commit $$(git rev-parse --short HEAD)"
