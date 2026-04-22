.PHONY: help install-dev install-all format lint test smoke \
        week-1 week-2 week-3 week-4 week-5 week-6 week-7 week-8 week-9 week-10 week-11 week-12 \
        test-week-1 test-week-2 test-week-3 test-week-4 test-week-5 test-week-6 \
        test-week-7 test-week-8 test-week-9 test-week-10 test-week-11 test-week-12 \
        fetch-data portfolio-build clean

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
	@echo "  week-N            N in 1..12"
	@echo "  test-week-N       pytest just week N"
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

format:
	ruff format .
	ruff check . --fix

lint:
	ruff check .
	mypy src

test:
	pytest

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
