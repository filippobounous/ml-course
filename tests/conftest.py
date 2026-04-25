"""Top-level pytest config — adds the `--run-slow` opt-in for integration tests.

Tests marked `@pytest.mark.slow` (typically things that train a real model
end-to-end) are skipped by default so the unit-test gate stays fast. Opt in with:

    pytest --run-slow -q          # include slow tests
    pytest -m slow --run-slow -q  # run only slow tests
"""

from __future__ import annotations

import pytest


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
