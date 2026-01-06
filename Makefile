.PHONY: help install-dev format lint test

help:
	@echo "Targets:"
	@echo "  install-dev  - install editable package + dev deps"
	@echo "  format       - format with ruff"
	@echo "  lint         - lint with ruff + typecheck with mypy"
	@echo "  test         - run pytest"

install-dev:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

format:
	ruff format .
	ruff check . --fix

lint:
	ruff check .
	mypy src

test:
	pytest
