#!/usr/bin/env bash
set -euo pipefail
cd /work

# In case dependencies changed since build, sync quickly on container start.
python -m pip install -e ".[dev]" >/dev/null

exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
