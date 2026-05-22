"""Stage repo markdown into ``docs/`` (gitignored) and build mkdocs.

mkdocs refuses ``docs_dir: .`` (the docs dir must be a sibling of
``mkdocs.yml``), so we symlink the source-of-truth markdown into
``docs/`` before each build. This keeps the writing surface single-source
— authors edit the README in its natural place, not a doc-only fork —
while still satisfying mkdocs.

Run as:

    python scripts/build_docs.py [--serve] [--no-build]

Flags:
  --serve    : after staging, run ``mkdocs serve`` on http://127.0.0.1:8000
  --no-build : stage symlinks only; don't invoke mkdocs (useful from CI
               when you want to run ``mkdocs build --strict`` separately)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# Source -> destination (relative to DOCS).
PAGES: list[tuple[Path, Path]] = []


def _add(src: str, dst: str) -> None:
    PAGES.append((ROOT / src, DOCS / dst))


# Top-level.
_add("README.md", "README.md")
_add("STUDY_GUIDE.md", "STUDY_GUIDE.md")
_add("SYLLABUS.md", "SYLLABUS.md")
_add("PORTFOLIO.md", "PORTFOLIO.md")
_add("gallery.md", "gallery.md")
# Cross-referenced by other pages; not in the main nav but must be present
# for strict-mode link resolution. Listed in mkdocs.yml under `not_in_nav`.
_add("TODO.md", "TODO.md")
_add("PR_PLAN.md", "PR_PLAN.md")
_add("CONTRIBUTING.md", "CONTRIBUTING.md")

# Per-week module pages.
WEEKS = [
    ("01_math_foundations", "01"),
    ("02_stat_learning", "02"),
    ("03_classical_supervised", "03"),
    ("04_classical_unsupervised", "04"),
    ("05_nn_from_scratch", "05"),
    ("06_pytorch_trainer", "06"),
    ("07_cnns_vision", "07"),
    ("08_transformers", "08"),
    ("09_llms_dpo", "09"),
    ("10_diffusion_multimodal", "10"),
    ("11_rl_agents", "11"),
    ("12_applied_capstone", "12"),
    ("13_llms_dev_surface", "13"),
]
for module_dir, week in WEEKS:
    _add(
        f"modules/{module_dir}/notebooks/lecture_notes.md",
        f"modules/{week}/lecture_notes.md",
    )
    _add(f"modules/{module_dir}/problems/README.md", f"modules/{week}/problems.md")
    if week not in ("13",):  # W13 has no solutions_theory yet.
        _add(
            f"modules/{module_dir}/problems/solutions_theory.md",
            f"modules/{week}/solutions_theory.md",
        )
    # Worked examples only exist for W1–W12 (W13 doesn't have them).
    if week != "13":
        _add(
            f"modules/{module_dir}/notebooks/worked_examples.md",
            f"modules/{week}/worked_examples.md",
        )

# Portfolio pages.
PORTFOLIOS = [
    ("02_numpy_linreg", "02"),
    ("03_tabular_benchmark", "03"),
    ("04_pca_statarb", "04"),
    ("05_micrograd", "05"),
    ("06_trainer", "06"),
    ("07_vision_classifier", "07"),
    ("08_tinygpt", "08"),
    ("09_dpo_tinyllama", "09"),
    ("10_ddpm", "10"),
    ("11_rl_agent", "11"),
    ("12_capstone", "12"),
    ("13_dev_surface", "13"),
]
_add("portfolio/README.md", "portfolio/index.md")
_add("portfolio/model_card_template.md", "portfolio/model_card_template.md")
# Each artifact gets a subdir so the source's relative cross-links
# (`model_card.md`, `../model_card_template.md`) keep working in
# the staged tree.
for portfolio_dir, week in PORTFOLIOS:
    _add(f"portfolio/{portfolio_dir}/README.md", f"portfolio/{week}/index.md")

# Model cards live alongside artifacts that ship a trained model.
MODEL_CARDS = [
    ("05_micrograd", "05"),
    ("07_vision_classifier", "07"),
    ("08_tinygpt", "08"),
    ("09_dpo_tinyllama", "09"),
    ("10_ddpm", "10"),
    ("11_rl_agent", "11"),
    ("12_capstone", "12"),
]
for portfolio_dir, week in MODEL_CARDS:
    _add(
        f"portfolio/{portfolio_dir}/model_card.md",
        f"portfolio/{week}/model_card.md",
    )

# Reference.
_add("src/mlcourse/configs/README.md", "reference/hydra_configs.md")


def stage() -> None:
    """(Re)create ``docs/`` populated with symlinks (or copies on platforms
    that can't symlink, but Ubuntu / macOS support symlinks natively)."""
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir()
    missing: list[Path] = []
    for src, dst in PAGES:
        if not src.exists():
            missing.append(src)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)
    print(f"staged {len(PAGES) - len(missing)} pages into {DOCS}")
    if missing:
        print(f"  warning: {len(missing)} source files missing — nav may be incomplete:")
        for path in missing:
            print(f"    - {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="run mkdocs serve after staging")
    parser.add_argument("--no-build", action="store_true", help="stage only; don't invoke mkdocs")
    args = parser.parse_args()

    stage()
    if args.no_build:
        return 0
    cmd = ["mkdocs", "serve"] if args.serve else ["mkdocs", "build", "--strict"]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
