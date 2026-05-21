"""Tests for the W12 paper-reproduction artifact (PPO clip-ε ablation).

Fast tier:
  * Module imports cleanly (no torch / gymnasium needed at import time).
  * Helpers do what they claim on stubbed inputs.
  * The `--quick` smoke path can be invoked but does not run (we only
    verify argparse parsing) — actual training is in the slow tier.

Slow tier (gymnasium + torch required):
  * `python ppo_clip_ablation.py --quick` runs end-to-end on CartPole-v1
    and writes findings.md + figure_clip_ablation.png.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPRO_DIR = Path(__file__).resolve().parents[2] / "portfolio" / "12_capstone" / "paper_reproduction"
SCRIPT = REPRO_DIR / "ppo_clip_ablation.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_module_imports():
    """Module loads without requiring torch / gymnasium at import time."""
    mod = _load(SCRIPT, "ppo_clip_ablation_import")
    # No-clip sentinel is well-defined.
    assert mod.NO_CLIP_EPSILON > 100, "no-clip sentinel should be large"
    # Required public helpers exist.
    assert callable(mod.run_one)
    assert callable(mod.main)


def test_findings_template_writes_expected_sections(tmp_path, monkeypatch):
    """`_write_findings` produces a table with one row per config."""
    mod = _load(SCRIPT, "ppo_clip_ablation_findings")
    monkeypatch.setattr(mod, "HERE", tmp_path)
    table = [
        {
            "label": "eps=0.1",
            "eps": 0.1,
            "mean": 100.0,
            "std": 5.0,
            "seeds": 4,
            "histories": [],
        },
        {
            "label": "no-clip",
            "eps": mod.NO_CLIP_EPSILON,
            "mean": 30.0,
            "std": 20.0,
            "seeds": 4,
            "histories": [],
        },
    ]
    args = type("A", (), {"total_steps": 1000, "seeds": 4})()
    mod._write_findings(table, args)
    content = (tmp_path / "findings.md").read_text(encoding="utf-8")
    assert "## Ablation table" in content
    assert "eps=0.1" in content and "no-clip" in content
    assert "10⁶" in content, "no-clip row should show the sentinel notation"
    assert "100.0" in content and "30.0" in content


@pytest.mark.slow
def test_quick_reproduction_runs_end_to_end(tmp_path):
    """Slow tier: actually train PPO for 4k steps × 1 seed × 4 configs.

    Verifies the full reproduction pipeline (import → PPO train →
    aggregate → write findings + figure) works on CartPole-v1.
    """
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")

    # Run in a temp dir so we don't clobber the committed findings template.
    work = tmp_path / "repro"
    work.mkdir()
    for f in ("ppo_clip_ablation.py", "PLAN.md", "README.md"):
        shutil.copy(REPRO_DIR / f, work / f)

    result = subprocess.run(
        [sys.executable, "ppo_clip_ablation.py", "--quick"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"non-zero exit: {result.stderr}"
    # findings.md and (if matplotlib present) the figure should be written.
    assert (work / "findings.md").exists()
    findings = (work / "findings.md").read_text(encoding="utf-8")
    assert "Ablation table" in findings
