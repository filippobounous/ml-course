#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name mlcourse --display-name "mlcourse (.venv)"

echo "Done. Run: source .venv/bin/activate && jupyter lab"
