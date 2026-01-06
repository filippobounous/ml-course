FROM python:3.12-slim

# System deps (git is useful; tini helps signal handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tini \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Python deps
RUN python -m pip install -U pip
# Install Jupyter + the repo's declared deps
COPY pyproject.toml README.md /work/
COPY src/ /work/src/
RUN python -m pip install -e ".[dev]"

ENTRYPOINT ["/usr/bin/tini", "--"]
