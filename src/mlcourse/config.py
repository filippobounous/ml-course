from __future__ import annotations

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

    @staticmethod
    def from_repo_root(repo_root: Path) -> "Paths":
        return Paths(
            repo_root=repo_root,
            data_raw=repo_root / "data" / "raw",
            data_interim=repo_root / "data" / "interim",
            data_processed=repo_root / "data" / "processed",
            models=repo_root / "models",
            reports=repo_root / "reports",
        )
