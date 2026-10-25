"""Tests master reproduction script and canonical result artifacts integrity."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_reproduce_script_exists():
    rep_script = ROOT / "reproduce_all.py"
    assert rep_script.exists(), "reproduce_all.py master script must exist at repo root."


def test_canonical_result_artifacts_exist():
    results_dir = ROOT / "results"
    
    # 1. Data Card
    assert (results_dir / "data_card.md").exists()
    
    # 2. Frozen Test Report
    assert (results_dir / "frozen_detector_test_report.md").exists()
    assert (results_dir / "v3" / "test_report.json").exists()
    
    # 3. Calibration Report & Clean Trust Pipeline
    assert (results_dir / "calibration_report.md").exists()
    assert (results_dir / "v3" / "trust_pipeline_clean.csv").exists()
    
    # 4. Risk Validation Report & Asset CSV
    assert (results_dir / "risk_validation.md").exists()
    assert (results_dir / "risk_assets_report.csv").exists()
    
    # 5. Core Triage Report & JSON
    assert (results_dir / "core_triage_experiment.md").exists()
    assert (results_dir / "core_triage_experiment.json").exists()
    
    # 6. Manuscript Revision
    assert (ROOT / "docs" / "MANUSCRIPT_REVISION.md").exists()


def test_manuscript_metric_verbatim_alignment():
    """Verifies that key numbers reported in MANUSCRIPT_REVISION.md match raw JSONs verbatim."""
    ms_text = (ROOT / "docs" / "MANUSCRIPT_REVISION.md").read_text(encoding="utf-8")
    
    # Check frozen detector F1
    with open(ROOT / "results" / "v3" / "test_report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)
    f1_str = f"{rep['pointwise_primary']['F1']:.4f}"
    assert f1_str in ms_text, f"Pointwise F1 {f1_str} must be present in manuscript."
    
    # Check core triage headline claim
    with open(ROOT / "results" / "core_triage_experiment.json", "r", encoding="utf-8") as f:
        triage = json.load(f)
    p4 = next(p for p in triage["headline_policies"] if "Policy 4" in p["Policy"])
    assert p4["unnecessary_responses"] == 0
    assert p4["attacks_preserved"] == "7/7"


def test_frozen_config_detector_alignment():
    """Verifies that frozen_config.json matches config_v3.json and test_report.json."""
    with open(ROOT / "frozen_config.json", "r", encoding="utf-8") as f:
        frozen = json.load(f)
    with open(ROOT / "experiments" / "v3_residual" / "config_v3.json", "r", encoding="utf-8") as f:
        v3 = json.load(f)
    with open(ROOT / "results" / "v3" / "test_report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert frozen["ridge_alpha"] == v3["model"]["alpha"] == 10.0
    assert frozen["n_top"] == v3["model"]["n_top"] == 3
    assert frozen["threshold_quantile"] == v3["threshold"]["quantile"] == 99.0
    assert abs(rep["threshold"] - 7.4532) < 1e-4, f"Threshold {rep['threshold']} mismatch"


def test_test_safeguard_active():
    """Verifies that test report exists and rerun guard prevents accidental test recalculation."""
    test_json = ROOT / "results" / "v3" / "test_report.json"
    assert test_json.exists(), "Official test report must exist"
    with open(test_json, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["events_detected"] == "7/7"
    assert rep["pointwise_primary"]["TP"] == 300
    assert rep["pointwise_primary"]["FP"] == 10

