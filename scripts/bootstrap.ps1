python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name mlcourse --display-name "mlcourse (.venv)"
Write-Host "Done. Run: jupyter lab"
