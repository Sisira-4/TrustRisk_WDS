"""Unit tests for Authoritative Configuration Architecture and Consistency (Task A2).

Verifies that:
1. config/trustrisk.yaml is the single authoritative source of truth.
2. All derived view configs (frozen_config.json, config_v3.json) are synchronized.
3. Every report JSON contains the authoritative config SHA-256 hash.
4. results/run_manifest.json is generated and tracks all input hashes and parameters.
5. No script defines conflicting thresholds, weights, or hyperparameters.
"""
from pathlib import Path
import json
import re
import pytest
import yaml

from config.loader import load_config, get_config_hash, ROOT, generate_run_manifest

def test_authoritative_config_exists_and_valid():
    """Validates structure and completeness of config/trustrisk.yaml."""
    cfg_path = ROOT / "config" / "trustrisk.yaml"
    assert cfg_path.exists(), "config/trustrisk.yaml must exist as the single authoritative config"

    cfg = load_config(cfg_path)
    assert cfg.get("metadata", {}).get("authoritative") is True

    # Check top-level sections
    required_sections = ["metadata", "seeds", "data", "detector", "trust", "risk", "costs", "paths"]
    for sec in required_sections:
        assert sec in cfg, f"Missing required configuration section: {sec}"

    # Detector parameters
    det = cfg["detector"]
    assert det["ridge_alpha"] == 10.0
    assert det["n_top"] == 3
    assert det["threshold_quantile"] == 99.0
    assert det["persistence"]["k"] == 3
    assert det["persistence"]["n"] == 6
    assert det["persistence"]["causal"] is True
    assert det["checkpoint_sha256"] == "5758d9eb92ceb0cbd01cd361bdc350ca4f07d013ea401dd1e494caa3e030d8e2"

    # Trust parameters
    trust = cfg["trust"]
    assert trust["calibration"]["method"] in ("platt", "temperature_scaling", "isotonic")
    w = trust["weights"]
    assert pytest.approx(w["w1"] + w["w2"] + w["w3"] + w["w4"], 1e-4) == 1.0
    assert pytest.approx(w["w_pcs_3term"] + w["w_uqs_3term"] + w["w_ers_3term"], 1e-4) == 1.0
    assert trust["ati_bands"]["action_threshold"] == 0.80
    assert trust["ati_bands"]["review_threshold"] == 0.50
    assert trust["hrs"]["rolling_window_hours"] == 168

    # Risk parameters
    risk = cfg["risk"]
    tdcri_w = risk["tdcri_weights"]
    assert pytest.approx(tdcri_w["w_mdci"] + tdcri_w["w_ass"] + tdcri_w["w_iss"] + tdcri_w["w_cri"] + tdcri_w["w_ars"] + tdcri_w["w_ts"], 1e-4) == 1.0
    assert risk["tppi"]["mitigation_budget_dollars"] == 50000.0

    # Costs
    costs = cfg["costs"]
    assert "operational_triage" in costs
    op_costs = costs["operational_triage"]
    for c_key in ["actuator_damage", "downtime", "contaminated_volume", "false_dispatch", "inspection", "unmitigated_attack_hourly_rate"]:
        assert c_key in op_costs, f"Missing operational cost key: {c_key}"


def test_derived_views_match_master_config():
    """Verifies that frozen_config.json and config_v3.json are identical to authoritative config."""
    cfg = load_config()
    cfg_hash = get_config_hash()

    # 1. frozen_config.json
    frozen_path = ROOT / "frozen_config.json"
    assert frozen_path.exists(), "frozen_config.json view must exist"
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen = json.load(f)

    assert frozen["ridge_alpha"] == cfg["detector"]["ridge_alpha"]
    assert frozen["n_top"] == cfg["detector"]["n_top"]
    assert frozen["threshold_quantile"] == cfg["detector"]["threshold_quantile"]
    assert frozen["persistence"] == cfg["detector"]["persistence"]
    assert frozen["checkpoint_sha256"] == cfg["detector"]["checkpoint_sha256"]
    assert frozen["authoritative_config_sha256"] == cfg_hash

    # 2. experiments/v3_residual/config_v3.json
    v3_path = ROOT / "experiments" / "v3_residual" / "config_v3.json"
    assert v3_path.exists(), "config_v3.json view must exist"
    with open(v3_path, "r", encoding="utf-8") as f:
        v3 = json.load(f)

    assert v3["model"]["alpha"] == cfg["detector"]["ridge_alpha"]
    assert v3["model"]["n_top"] == cfg["detector"]["n_top"]
    assert v3["threshold"]["quantile"] == cfg["detector"]["threshold_quantile"]
    assert v3["persistence"] == cfg["detector"]["persistence"]
    assert v3["authoritative_config_sha256"] == cfg_hash


def test_run_manifest_consistency():
    """Verifies results/run_manifest.json contains accurate commit, config hash, and raw data hashes."""
    manifest_path = ROOT / "results" / "run_manifest.json"
    if not manifest_path.exists():
        generate_run_manifest(manifest_path)
    assert manifest_path.exists(), "results/run_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        mf = json.load(f)

    assert mf["config_sha256"] == get_config_hash()
    assert "git_commit" in mf and len(mf["git_commit"]) > 0
    assert "data_hashes" in mf
    assert "package_versions" in mf
    assert "detector_checkpoint_sha256" in mf


def test_reports_contain_config_hash():
    """Verifies all JSON reports include config_sha256 matching authoritative config."""
    cfg_hash = get_config_hash()
    report_paths = [
        ROOT / "results" / "v3" / "test_report.json",
        ROOT / "results" / "v3" / "block_bootstrap_report.json",
        ROOT / "results" / "baseline_comparison.json",
        ROOT / "results" / "core_triage_experiment.json",
    ]

    for p in report_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "config_sha256" in data, f"Report {p.name} is missing config_sha256"
            assert data["config_sha256"] == cfg_hash, (
                f"Report {p.name} config_sha256 {data['config_sha256']} != authoritative {cfg_hash}"
            )


def test_code_constants_match_config():
    """Verifies that Python engine classes load defaults matching authoritative config."""
    from models.trust.unified_trust_engine import UnifiedTrustEngine
    cfg = load_config()

    ute = UnifiedTrustEngine()
    w_cfg = cfg["trust"]["weights"]
    assert pytest.approx(ute.w_pcs) == w_cfg["w_pcs_3term"]
    assert pytest.approx(ute.w_uqs) == w_cfg["w_uqs_3term"]
    assert pytest.approx(ute.w_ers) == w_cfg["w_ers_3term"]

    # Verify ATI band assignment boundaries
    bands = ute.assign_ati_bands([0.85, 0.79, 0.50, 0.49])
    assert list(bands) == ["ACTION", "REVIEW", "REVIEW", "MONITOR"]
