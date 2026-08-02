.PHONY: help install-dev install-all install-docs format lint test test-slow smoke \
        docs docs-serve \
        week-1 week-2 week-3 week-4 week-5 week-6 week-7 week-8 week-9 week-10 week-11 week-12 week-13 \
        test-week-1 test-week-2 test-week-3 test-week-4 test-week-5 test-week-6 \
        test-week-7 test-week-8 test-week-9 test-week-10 test-week-11 test-week-12 test-week-13 \
        reproduce-2 reproduce-3 reproduce-4 reproduce-5 reproduce-6 reproduce-7 \
        reproduce-8 reproduce-9 reproduce-10 reproduce-11 reproduce-12 reproduce-13 \
        fetch-data portfolio-build clean \
        docker-dev docker-dev-shell docker-dev-test

help:
	@echo "Core targets:"
	@echo "  install-dev       install editable package + dev deps"
	@echo "  install-all       install every optional-dependency group"
	@echo "  format            ruff format + fix"
	@echo "  lint              ruff + mypy"
	@echo "  test              pytest (all)"
	@echo "  smoke             quick import + env check"
	@echo ""
	@echo "Weekly targets (installs the right dep groups and points you at the module):"
	@echo "  week-N            N in 1..13"
	@echo "  test-week-N       pytest just week N"
	@echo ""
	@echo "Portfolio artifacts:"
	@echo "  reproduce-N       run artifact N's documented reproduction command (N in 2..13)"
	@echo "                    runtimes vary from seconds to hours — see the honesty"
	@echo "                    table in README.md before starting a long one"
	@echo ""
	@echo "Utilities:"
	@echo "  fetch-data        download datasets referenced by the curriculum"
	@echo "  portfolio-build   render PORTFOLIO.md + each artifact README to reports/portfolio/"
	@echo "  clean             remove caches"

install-dev:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

install-all:
	python -m pip install -U pip
	python -m pip install -e ".[all]"

install-docs:
	python -m pip install -U pip
	python -m pip install -e ".[docs]"

# --- Docs site (mkdocs-material) ----------------------------------------------
# `make docs` builds the static site into `./site/` after staging markdown
# into `./docs/`. `make docs-serve` runs the live-reload dev server.
docs:
	python scripts/build_docs.py --no-build
	mkdocs build --strict

docs-serve:
	python scripts/build_docs.py --serve

format:
	ruff format .
	ruff check . --fix

lint:
	ruff check .
	mypy src

test:
	pytest

test-slow:
	pytest --run-slow -q

smoke:
	python -c "import mlcourse; print('mlcourse', mlcourse.__version__ if hasattr(mlcourse,'__version__') else 'ok')"

# --- Weekly targets ------------------------------------------------------------
# Each week installs the minimal dep groups it needs and points you at the module.
# Weeks 1-4 run on pure NumPy / scikit-learn (no torch). Weeks 5+ pull in torch.

week-1: install-dev
	@echo "Week 1: Math foundations. See modules/01_math_foundations/README.md"

week-2: install-dev
	@echo "Week 2: Statistical learning & NumPy linear models. See modules/02_stat_learning/README.md"

week-3: install-dev
	python -m pip install -e ".[ops]"
	@echo "Week 3: Classical supervised (XGBoost/LightGBM). See modules/03_classical_supervised/README.md"

week-4: install-dev
	@echo "Week 4: Classical unsupervised + PCA stat-arb. See modules/04_classical_unsupervised/README.md"

week-5: install-dev
	python -m pip install -e ".[dl]"
	@echo "Week 5: Autograd from scratch. See modules/05_nn_from_scratch/README.md"

week-6: install-dev
	python -m pip install -e ".[dl,ops]"
	@echo "Week 6: PyTorch + reproducibility stack. See modules/06_pytorch_trainer/README.md"

week-7: install-dev
	python -m pip install -e ".[dl,ops]"
	@echo "Week 7: CNNs & vision. See modules/07_cnns_vision/README.md"

week-8: install-dev
	python -m pip install -e ".[dl,ops]"
	@echo "Week 8: Transformers from scratch. See modules/08_transformers/README.md"

week-9: install-dev
	python -m pip install -e ".[dl,llm,ops]"
	@echo "Week 9: LLMs, SFT, DPO. See modules/09_llms_dpo/README.md"

week-10: install-dev
	python -m pip install -e ".[dl,diffusion,ops]"
	@echo "Week 10: Diffusion & multimodal. See modules/10_diffusion_multimodal/README.md"

week-11: install-dev
	python -m pip install -e ".[dl,rl,ops]"
	@echo "Week 11: RL & agents. See modules/11_rl_agents/README.md"

week-12: install-dev
	python -m pip install -e ".[dl,sciml,ops]"
	@echo "Week 12: Applied tracks & capstone. See modules/12_applied_capstone/README.md"

week-13: install-dev
	python -m pip install -e ".[devsurface,ops]"
	@echo "Week 13: LLMs as a development surface. See modules/13_llms_dev_surface/README.md"

# --- Per-week problem-set test targets -----------------------------------------
test-week-1:
	pytest tests/week_01

test-week-2:
	pytest tests/week_02

test-week-3:
	pytest tests/week_03

test-week-4:
	pytest tests/week_04

test-week-5:
	pytest tests/week_05

test-week-6:
	pytest tests/week_06

test-week-7:
	pytest tests/week_07

test-week-8:
	pytest tests/week_08

test-week-9:
	pytest tests/week_09

test-week-10:
	pytest tests/week_10

test-week-11:
	pytest tests/week_11

test-week-12:
	pytest tests/week_12

test-week-13:
	pytest tests/week_13

# --- Portfolio reproduction ----------------------------------------------------
# One target per artifact, running the command documented under that artifact's
# README "Reproduce" heading. Most are seconds-to-minutes; W7/W8/W9/W10/W11 train
# real models. W8 and W9 need a corpus download / HF auth for the full run, so
# those targets run the documented smoke path and print the full command.

reproduce-2:
	python portfolio/02_numpy_linreg/demo.py

reproduce-3:
	python portfolio/03_tabular_benchmark/benchmark.py

reproduce-4:
	python portfolio/04_pca_statarb/demo.py

reproduce-5:
	python portfolio/05_micrograd/demo.py

reproduce-6:
	python portfolio/06_trainer/demo.py

reproduce-7:
	python portfolio/07_vision_classifier/demo.py

reproduce-8:
	@echo ">> Smoke config. Full run needs a TinyStories corpus — see portfolio/08_tinygpt/README.md"
	python portfolio/08_tinygpt/train.py --max-iters 200 --n-layer 2 --d-model 128

reproduce-9:
	@echo ">> Smoke config. Full run is ~2-4 h on MPS — see portfolio/09_dpo_tinyllama/README.md"
	python portfolio/09_dpo_tinyllama/dpo_train.py --quick

reproduce-10:
	python portfolio/10_ddpm/train.py
	python portfolio/10_ddpm/ablate.py

reproduce-11:
	python portfolio/11_rl_agent/train_ppo.py

reproduce-12:
	@echo ">> Track B (offline, seconds). Track A: python portfolio/12_capstone/demo_pinn.py"
	python portfolio/12_capstone/demo_statarb.py

reproduce-13:
	python portfolio/13_dev_surface/demo.py

# --- Utilities -----------------------------------------------------------------
fetch-data:
	python -m mlcourse.data.fetch

portfolio-build:
	python -m mlcourse.utils.portfolio_build

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +

# --- Dev Docker image ---------------------------------------------------------
# See `docker/Dockerfile.dev` and `docker-compose.dev.yml` for details.
docker-dev:
	docker compose -f docker-compose.dev.yml build dev

docker-dev-shell: docker-dev
	docker compose -f docker-compose.dev.yml run --rm dev bash

docker-dev-test: docker-dev
	docker compose -f docker-compose.dev.yml run --rm dev pytest --run-slow -q
