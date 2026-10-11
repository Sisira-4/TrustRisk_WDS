import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_core_experiment_reports_exist():
    json_path = ROOT / "results" / "core_triage_experiment.json"
    md_path = ROOT / "results" / "core_triage_experiment.md"
    assert json_path.exists(), "results/core_triage_experiment.json must exist"
    assert md_path.exists(), "results/core_triage_experiment.md must exist"
    
    with open(json_path, "r") as f:
        data = json.load(f)
    assert "headline_policies" in data
    assert "tradeoff_sweep" in data
    assert "component_ablation" in data
    assert len(data["headline_policies"]) == 5


def test_trust_gate_reduces_false_alarms_without_attack_loss():
    json_path = ROOT / "results" / "core_triage_experiment.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    
    policies = {p["Policy"]: p for p in data["headline_policies"]}
    
    p2 = policies["Policy 2: Temporal Persistence Smoothing"]
    p4 = policies["Policy 4: Trust-Gated Triage (Full ATI)"]
    p5 = policies["Policy 5: Trust + Risk Gate (ATI x TDCRI)"]
    
    # Assert zero attacks lost across action + review
    assert p4["attacks_preserved"] == "7/7", "Policy 4 must preserve 7/7 attacks"
    assert p5["attacks_preserved"] == "7/7", "Policy 5 must preserve 7/7 attacks"
    assert p4["attacks_lost"] == 0
    assert p5["attacks_lost"] == 0
    
    # Assert false alarm reduction in immediate action tier
    assert p4["unnecessary_responses"] < p2["unnecessary_responses"], \
        "Trust gate must reduce unnecessary responses compared to persistence"
    assert p4["reduction_vs_persist"] > 0.0


def test_ablation_demonstrates_multicomponent_necessity():
    json_path = ROOT / "results" / "core_triage_experiment.json"
    with open(json_path, "r") as f:
        data = json.load(f)
        
    abl = {a["Configuration"]: a for a in data["component_ablation"]}
    full = abl["Full Trust Score (PCS + UQS + ERS)"]
    no_pcs = abl["Ablate PCS (UQS + ERS only: 0.60(1-U) + 0.40 ERS)"]
    
    # Ablating PCS increases false alarms in Action tier
    assert no_pcs["Unnecessary Responses (Action FA)"] > full["Unnecessary Responses (Action FA)"], \
        "Ablating PCS must degrade false alarm suppression"

