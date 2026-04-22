"""Build a static portfolio index from PORTFOLIO.md + each `portfolio/*/README.md`.

Phase A: aggregates artifact READMEs into reports/portfolio/index.md as a
single browsable file. A richer HTML renderer can be added later.
"""

from __future__ import annotations

from pathlib import Path

from mlcourse.config import default_paths


def _collect_artifact_readmes(portfolio_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if not portfolio_dir.is_dir():
        return items
    for child in sorted(portfolio_dir.iterdir()):
        if not child.is_dir():
            continue
        readme = child / "README.md"
        if readme.is_file():
            items.append((child.name, readme))
    return items


def build() -> Path:
    paths = default_paths()
    out_dir = paths.reports / "portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)

    header = (paths.repo_root / "PORTFOLIO.md").read_text(encoding="utf-8")
    artifacts = _collect_artifact_readmes(paths.portfolio)

    sections = ["# Portfolio Index (generated)", "", header, "", "---", ""]
    for name, readme in artifacts:
        sections.append(f"## {name}")
        sections.append("")
        sections.append(readme.read_text(encoding="utf-8"))
        sections.append("")

    out_path = out_dir / "index.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    return out_path


def main() -> None:
    out = build()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
