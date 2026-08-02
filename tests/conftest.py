"""Top-level pytest config.

Two things live here: the `--run-slow` opt-in, and the switch that decides
*whose* code the week's problem-set tests grade.

## Which implementation gets graded

Default (`MLCOURSE_SOLUTIONS` unset, or `mine`): the tests grade **your**
work — `modules/NN_*/problems/starter.py`. On a fresh clone every function
there raises `NotImplementedError`, so the per-week tests fail until you
implement them. That is the intended starting state: a green test is a
signal you earned, not one the repository hands you.

`MLCOURSE_SOLUTIONS=reference`: grade the committed reference implementation
in `modules/NN_*/problems/_reference/solutions.py` instead. Use it to confirm
that a failing test is your bug and not the course's, and to read a working
version once you have your own. CI runs this mode so the repository itself
stays verified.

    pytest tests/week_01                              # grade your starter
    MLCOURSE_SOLUTIONS=reference pytest tests/week_01  # grade the reference
    make test-reference                                # the whole suite, reference mode

Portfolio artifacts under `portfolio/` are still shipped fully implemented and
are loaded directly regardless of this setting — see TODO.md.

## Slow tests

Tests marked `@pytest.mark.slow` (typically things that train a real model
end-to-end) are skipped by default so the unit-test gate stays fast. Opt in:

    pytest --run-slow -q          # include slow tests
    pytest -m slow --run-slow -q  # run only slow tests
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS_MODE_ENV = "MLCOURSE_SOLUTIONS"
_VALID_MODES = ("mine", "reference")


def solutions_mode() -> str:
    """Return `"mine"` or `"reference"`; raise on a typo rather than guessing."""
    mode = os.environ.get(SOLUTIONS_MODE_ENV, "mine").strip().lower()
    if mode not in _VALID_MODES:
        raise pytest.UsageError(
            f"{SOLUTIONS_MODE_ENV}={mode!r} is not valid; "
            f"expected one of {', '.join(_VALID_MODES)}."
        )
    return mode


def load_path(path: Path, name: str) -> ModuleType:
    """Import a module from an explicit file path.

    Registered in `sys.modules` so that `@dataclass` under
    `from __future__ import annotations` can resolve its forward references.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"Could not load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def problems_path(module_dir: str) -> Path:
    """Path to the problem-set implementation currently under test."""
    base = REPO_ROOT / "modules" / module_dir / "problems"
    if solutions_mode() == "reference":
        return base / "_reference" / "solutions.py"
    return base / "starter.py"


@pytest.fixture(scope="session")
def load_module():
    """Factory: load any module by repo-relative path (used for portfolio code)."""

    def _load(rel_path: str, name: str) -> ModuleType:
        return load_path(REPO_ROOT / rel_path, name)

    return _load


@pytest.fixture(scope="session")
def load_problems():
    """Factory: load a week's problem-set implementation (starter or reference)."""

    def _load(module_dir: str, name: str) -> ModuleType:
        path = problems_path(module_dir)
        if not path.exists():
            raise pytest.UsageError(
                f"{path} is missing. In `mine` mode the tests grade "
                f"modules/{module_dir}/problems/starter.py — restore it from git, or run "
                f"with {SOLUTIONS_MODE_ENV}=reference."
            )
        return load_path(path, name)

    return _load


def pytest_report_header(config: pytest.Config) -> str:
    mode = solutions_mode()
    if mode == "reference":
        return f"{SOLUTIONS_MODE_ENV}=reference — grading the committed reference solutions"
    return (
        f"{SOLUTIONS_MODE_ENV}=mine — grading your starter.py files "
        f"(NotImplementedError means 'not written yet'; "
        f"use {SOLUTIONS_MODE_ENV}=reference to check the course itself)"
    )


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    """Explain a red run in `mine` mode — the project default is `-q`, which
    hides `pytest_report_header`, so the hint has to go here."""
    if solutions_mode() != "mine" or not terminalreporter.stats.get("failed"):
        return
    terminalreporter.write_sep("-", "grading your work")
    terminalreporter.write_line(
        "These tests grade modules/NN_*/problems/starter.py. A NotImplementedError "
        "means that problem is still unwritten — implement it and re-run."
    )
    terminalreporter.write_line(
        f"To check the course itself rather than your work: "
        f"{SOLUTIONS_MODE_ENV}=reference pytest  (or `make test-reference`)."
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Include tests marked @pytest.mark.slow.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
