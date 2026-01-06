# Contributing

Thanks for helping improve this repo.

## Development setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
make format
make lint
make test
```

## Style
- Formatting + linting: Ruff
- Tests: pytest
- Prefer putting logic in `src/` rather than notebooks
