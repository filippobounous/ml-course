#!/usr/bin/env bash
# Bootstrap a local macOS (or Linux) dev environment for the ML/AI course.
#
# What this does:
#   1. Verifies Python >= 3.11
#   2. Creates a .venv and installs the project in editable mode with dev deps
#   3. Registers the Jupyter kernel
#   4. Runs an MPS sanity check (Apple Silicon) if torch is available
#   5. Optionally installs MLX (Apple-native) on Darwin
#   6. Optionally runs `huggingface-cli login` for gated models / Spaces
#   7. Primes the HuggingFace dataset cache directory
#
# Usage:
#   bash scripts/bootstrap_macos.sh               # base setup
#   INSTALL_ALL=1 bash scripts/bootstrap_macos.sh # also installs all optional deps
#   INSTALL_DL=1  bash scripts/bootstrap_macos.sh # also installs the 'dl' group
#   HF_LOGIN=1    bash scripts/bootstrap_macos.sh # also runs huggingface-cli login

set -euo pipefail

# --- 0. Preflight --------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. On macOS: brew install python@3.12" >&2
  exit 1
fi

PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  echo "Python >= 3.11 required (found $PY_MAJOR.$PY_MINOR). Install via: brew install python@3.12" >&2
  exit 1
fi

# --- 1. venv + editable install ------------------------------------------------
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

# Optional groups
if [ "${INSTALL_ALL:-0}" = "1" ]; then
  python -m pip install -e ".[all]"
elif [ "${INSTALL_DL:-0}" = "1" ]; then
  python -m pip install -e ".[dl,ops]"
fi

# --- 2. Jupyter kernel ---------------------------------------------------------
python -m ipykernel install --user --name mlcourse --display-name "mlcourse (.venv)"

# --- 3. MPS sanity check (Apple Silicon) ---------------------------------------
python - <<'PY'
try:
    import torch
    print(f"torch {torch.__version__}")
    if torch.backends.mps.is_available():
        x = torch.randn(4, 4, device="mps")
        y = x @ x.T
        print(f"MPS OK: matmul {tuple(y.shape)} on {y.device}")
    elif torch.cuda.is_available():
        print("CUDA available (not expected on M-series, but fine).")
    else:
        print("Neither MPS nor CUDA available. CPU-only mode.")
except ImportError:
    print("torch not installed yet. Run with INSTALL_DL=1 to install the DL group.")
PY

# --- 4. MLX (Apple-native) -----------------------------------------------------
if [ "$(uname -s)" = "Darwin" ] && [ "${INSTALL_ALL:-0}" = "1" ]; then
  python -m pip install --upgrade mlx mlx-lm || echo "mlx install skipped (non-arm64?)"
fi

# --- 5. HuggingFace login (optional) -------------------------------------------
if [ "${HF_LOGIN:-0}" = "1" ]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login || true
  else
    echo "huggingface-cli not installed (install with INSTALL_ALL=1 or .[llm]) — skipping login."
  fi
fi

# --- 6. Prime dataset cache dir ------------------------------------------------
mkdir -p data/raw data/interim data/processed models reports

cat <<'MSG'

Bootstrap complete.

Next steps:
  source .venv/bin/activate
  jupyter lab
  # or progress through weeks:
  make week-1
  make week-2
  # ...

Tips:
  * Re-run with INSTALL_ALL=1 once you reach week 9+ to pull all optional deps.
  * Re-run with HF_LOGIN=1 before week 9 to push your model card / Gradio Space.

MSG
