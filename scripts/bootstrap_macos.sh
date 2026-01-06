#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. On macOS, install via Homebrew: brew install python@3.12"
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name mlcourse --display-name "mlcourse (.venv)"

echo "Done. Next: source .venv/bin/activate && jupyter lab"
