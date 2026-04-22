"""Smoke tests for the Phase A scaffold.

These verify that the core package and its submodules import cleanly and that
the scaffolded paths resolve. They do NOT exercise any ML functionality — that
is covered by the per-week test directories (`tests/week_NN/`) populated as the
course progresses.
"""

from __future__ import annotations


def test_package_imports() -> None:
    import mlcourse

    assert mlcourse.__version__


def test_subpackages_import() -> None:
    import mlcourse.autograd
    import mlcourse.configs
    import mlcourse.data
    import mlcourse.models
    import mlcourse.utils

    for mod in (
        mlcourse.autograd,
        mlcourse.configs,
        mlcourse.data,
        mlcourse.models,
        mlcourse.utils,
    ):
        assert mod is not None


def test_paths_resolve() -> None:
    from mlcourse.config import default_paths

    paths = default_paths()
    assert (paths.repo_root / "pyproject.toml").is_file()
    # Curriculum directories exist in the Phase A scaffold.
    assert paths.modules.is_dir()
    assert paths.portfolio.is_dir()


def test_seed_everything_is_deterministic() -> None:
    import numpy as np

    from mlcourse.utils import seed_everything

    seed_everything(42)
    a = np.random.rand(4)
    seed_everything(42)
    b = np.random.rand(4)
    assert np.array_equal(a, b)


def test_detect_device_returns_valid() -> None:
    from mlcourse.utils import detect_device

    assert detect_device() in {"cpu", "mps", "cuda"}
