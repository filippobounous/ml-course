# ML Course Repo

This repo is a clean starting point for the course: reproducible environment, linting/testing, and a Jupyter option
(local or via Docker).

## Option A — Local on macOS (recommended)

### 0) Ensure you have a recent Python
- If you use Homebrew:
  ```bash
  brew install python@3.12
  ```
  Then use `python3` below.

### 1) Create the venv
```bash
python3 -m venv .venv
```

### 2) Activate it
```bash
source .venv/bin/activate
```

### 3) Upgrade pip + install the project (editable) with dev tools
```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### 4) Register the kernel for Jupyter
```bash
python -m ipykernel install --user --name mlcourse --display-name "mlcourse (.venv)"
```

### 5) Run JupyterLab
```bash
jupyter lab
```

## Option B — Jupyter as a Docker service

Prereqs: Docker Desktop + Docker Compose.

```bash
docker compose up --build jupyter
```

Then open:
- http://localhost:8888 (token is set in `docker-compose.yml`)

Stop:
```bash
docker compose down
```

## VS Code (macOS)

- Install recommended extensions (VS Code will prompt; see `.vscode/extensions.json`)
- `Cmd+Shift+P` → **Python: Select Interpreter** → choose `.venv/bin/python`
- In a notebook: **Select Kernel** → `mlcourse (.venv)`

## Common commands

```bash
make format
make lint
make test
```

## Repo layout

- `src/mlcourse/` — Python package (your reusable code)
- `notebooks/` — notebooks (keep them light; push logic into `src/`)
- `data/` — raw/interim/processed data (not committed except small samples)
- `models/` — saved checkpoints
- `reports/` — figures and write-ups
- `tests/` — pytest tests

## GitHub / first push

```bash
git init
git add -A
git commit -m "Week 0 scaffolding"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

- CI runs via GitHub Actions: `.github/workflows/ci.yml`
- Dependabot is enabled via `.github/dependabot.yml`
