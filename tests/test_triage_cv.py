"""Tests for Task 3.1: Systematic Leave-One-Attack-Event-Out Validation & Frozen Trust Config."""
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_trust_frozen_config_exists_and_valid():
    """Verifies that trust_frozen_config.json exists, was selected on validation, and is valid."""
    cfg_path = ROOT / "trust_frozen_config.json"
    assert cfg_path.exists(), "trust_frozen_config.json must exist prior to test set triage evaluation"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "metadata" in cfg
    assert "trust_weights_3term" in cfg
    assert "ati_bands" in cfg
    assert "policy_cutoffs" in cfg
    assert "validation_metrics" in cfg

    # Verify weights sum to 1.0
    w = cfg["trust_weights_3term"]
    assert pytest.approx(w["w_pcs"] + w["w_uqs"] + w["w_ers"], 1e-4) == 1.0

    # Verify validation performance preserves all attacks
    vm = cfg["validation_metrics"]
    assert vm["attacks_preserved"] == "5/5"
    assert vm["action_fa_events"] <= 5


def test_triage_cv_reports_exist():
    """Verifies that the validation cross-validation reports exist."""
    res_dir = ROOT / "results" / "v3"
    assert (res_dir / "triage_cv_report.json").exists()
    assert (res_dir / "triage_cv_report.csv").exists()
    assert (res_dir / "triage_cv_report.md").exists()


def test_engine_docstring_cites_triage_cv():
    """Verifies that UnifiedTrustEngine docstring directly cites experiments/run_triage_cv.py."""
    from models.trust.unified_trust_engine import UnifiedTrustEngine
    doc = UnifiedTrustEngine.__doc__
    assert doc is not None
    assert "experiments/run_triage_cv.py" in doc
    assert "trust_frozen_config.json" in doc
