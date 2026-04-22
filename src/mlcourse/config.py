"""Project paths and environment flags used across modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    models: Path
    reports: Path
    modules: Path
    portfolio: Path
    capstone: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> Paths:
        return Paths(
            repo_root=repo_root,
            data_raw=repo_root / "data" / "raw",
            data_interim=repo_root / "data" / "interim",
            data_processed=repo_root / "data" / "processed",
            models=repo_root / "models",
            reports=repo_root / "reports",
            modules=repo_root / "modules",
            portfolio=repo_root / "portfolio",
            capstone=repo_root / "capstone",
        )


def repo_root() -> Path:
    """Return the repo root, preferring MLCOURSE_ROOT if set, else walking upwards."""
    env = os.environ.get("MLCOURSE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


def default_paths() -> Paths:
    return Paths.from_repo_root(repo_root())
