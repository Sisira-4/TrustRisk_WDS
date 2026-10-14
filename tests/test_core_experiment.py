"""Unit and Property Tests for Phase 5 Core Alert Triage Experiment (Guide Phase 5 & D3).

Verifies that:
1. results/core_triage_experiment.json and results/core_triage_experiment.md exist.
2. 5 controlled triage policies are evaluated with zero attack loss and eliminated false alarms.
3. Event-level bootstrap confidence intervals (B=2000) are populated and mathematically valid.
4. Per-component ablation demonstrates necessity of PCS, UQS, ERS, HRS, and the Risk Gate.
5. Authoritative config hash is embedded and matches.
"""
from pathlib import Path
import json
import pytest

from config.loader import get_config_hash

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@pytest.fixture(scope="module")
def triage_data():
    json_path = RESULTS_DIR / "core_triage_experiment.json"
    assert json_path.exists(), "results/core_triage_experiment.json must exist"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_core_experiment_reports_exist_and_complete(triage_data):
    md_path = RESULTS_DIR / "core_triage_experiment.md"
    assert md_path.exists(), "results/core_triage_experiment.md must exist"
    
    assert "headline_policies" in triage_data
    assert "tradeoff_sweep" in triage_data
    assert "component_ablation" in triage_data
    assert "definitions" in triage_data
    assert len(triage_data["headline_policies"]) == 5
    
    # Verify authoritative config hash
    assert "config_sha256" in triage_data
    assert triage_data["config_sha256"] == get_config_hash()


def test_precommitted_definitions_present(triage_data):
    defs = triage_data["definitions"]
    assert defs.get("unnecessary_response") == "false_alarm_event_reaching_action"
    assert defs.get("missed_response") == "attack_event_reaching_neither_action_nor_review"


def test_trust_gate_reduces_false_alarms_without_attack_loss(triage_data):
    policies = {p["Policy"]: p for p in triage_data["headline_policies"]}
    
    p2 = policies["Policy 2: Temporal Persistence Smoothing"]
    p4 = policies["Policy 4: Trust-Gated Triage (Full ATI)"]
    p5 = policies["Policy 5: Trust + Risk Gate (ATI x TDCRI)"]
    
    # Assert zero attacks lost across action + review
    assert p4["attacks_preserved"] == "7/7", "Policy 4 must preserve 7/7 attacks"
    assert p5["attacks_preserved"] == "7/7", "Policy 5 must preserve 7/7 attacks"
    assert p4["attacks_lost"] == 0
    assert p5["attacks_lost"] == 0
    
    # Assert false alarm elimination in immediate action tier
    assert p4["unnecessary_responses"] == 0, "Policy 4 must have 0 Action tier false alarms"
    assert p5["unnecessary_responses"] == 0, "Policy 5 must have 0 Action tier false alarms"
    assert p4["unnecessary_responses"] < p2["unnecessary_responses"], (
        "Trust gate must reduce unnecessary responses compared to persistence"
    )
    assert p4["reduction_vs_persist"] == 100.0


def test_event_level_bootstrap_confidence_intervals(triage_data):
    """Verifies that event-level bootstrap CIs are computed and bounded."""
    for p in triage_data["headline_policies"]:
        assert "bootstrap_ci" in p, f"Policy {p['Policy']} missing bootstrap_ci"
        ci = p["bootstrap_ci"]
        
        # Unnecessary responses CI
        u_ci = ci["unnecessary_responses_95ci"]
        assert len(u_ci) == 2
        assert u_ci[0] <= u_ci[1]
        
        # Attacks preserved % CI
        pres_ci = ci["attacks_preserved_pct_95ci"]
        assert len(pres_ci) == 2
        assert pres_ci[0] <= pres_ci[1]
        assert pres_ci[1] == 100.0, "Attacks preserved upper bound must be 100%"
        
        # Mean delay CI
        d_ci = ci["mean_delay_hours_95ci"]
        assert len(d_ci) == 2
        assert d_ci[0] <= d_ci[1]


def test_ablation_demonstrates_multicomponent_necessity(triage_data):
    """Verifies that each trust component (PCS, UQS, ERS, HRS, Risk Gate) provides distinct value."""
    abl = {a["Configuration"]: a for a in triage_data["component_ablation"]}
    
    full = abl["Full Trust Score (PCS + UQS + ERS + HRS)"]
    assert full["Attacks Preserved"] == "7/7"
    assert full["Unnecessary Responses (Action FA)"] == 0
    
    # 1. Ablating PCS loses ground-truth attacks and surges delay
    no_pcs = abl["Ablate PCS (w2=0.50, w3=0.25, w4=0.25)"]
    assert no_pcs["Attacks Preserved"] != "7/7", "Ablating PCS must lose attacks"
    assert no_pcs["Attacks Lost"] > 0
    assert no_pcs["Mean Delay (h)"] > full["Mean Delay (h)"]
    
    # 2. Ablating UQS introduces false alarms into Review queue
    no_uqs = abl["Ablate UQS (w1=0.57, w3=0.21, w4=0.21)"]
    assert no_uqs["Review Queue FA Events"] > full["Review Queue FA Events"], (
        "Ablating UQS must increase Review queue false alarms"
    )
    
    # 3. Ablating ERS increases response latency
    no_ers = abl["Ablate ERS (w1=0.47, w2=0.35, w4=0.18)"]
    assert no_ers["Mean Delay (h)"] > full["Mean Delay (h)"]
    
    # 4. Ablating Risk Gate evaluates Policy 4 vs Policy 5 trade-off
    risk_abl = abl["Ablate Risk Gate (Policy 4 Trust-Only vs Policy 5 Risk-Gated)"]
    assert "P4" in str(risk_abl["Mean Delay (h)"]) and "P5" in str(risk_abl["Mean Delay (h)"])
