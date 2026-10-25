"""Unit and Property Tests for Identical-Protocol Baselines and Publication Figures."""
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def test_baseline_json_structure_and_completeness():
    base_file = RESULTS_DIR / "baseline_comparison.json"
    assert base_file.exists(), "results/baseline_comparison.json must exist"
    
    with open(base_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    expected_models = [
        "Physical_GNN",
        "LSTM_Autoencoder",
        "LightGBM_Tabular_Residual",
        "MultiSensor_Ridge_Residual_Ours",
        "TrustRisk_WDS_Policy4_TrustGate",
        "TrustRisk_WDS_Policy5_TrustRiskGate"
    ]
    for model in expected_models:
        assert model in data, f"Model {model} must be present in baseline comparison"
        m = data[model]
        assert "pointwise_f1" in m
        assert "pa_f1" in m
        assert "events_detected" in m
        assert 0.0 <= m["pointwise_f1"] <= 1.0
        assert 0.0 <= m["pa_f1"] <= 1.0


def test_pointwise_vs_pa_metric_inflation_invariant():
    base_file = RESULTS_DIR / "baseline_comparison.json"
    with open(base_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for name, m in data.items():
        if not isinstance(m, dict):
            continue
        pw_f1 = m["pointwise_f1"]
        pa_f1 = m["pa_f1"]
        # Mathematical invariant: Point-adjustment cannot reduce F1
        assert pa_f1 >= pw_f1 - 1e-4, f"Model {name} PA F1 ({pa_f1}) cannot be lower than pointwise F1 ({pw_f1})"


def test_trust_gate_false_alarm_elimination_invariant():
    base_file = RESULTS_DIR / "baseline_comparison.json"
    with open(base_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    p4 = data["TrustRisk_WDS_Policy4_TrustGate"]
    p5 = data["TrustRisk_WDS_Policy5_TrustRiskGate"]
    assert p4["action_fa_events"] == 0, "Policy 4 must have 0 Action tier false alarm events"
    assert p5["action_fa_events"] == 0, "Policy 5 must have 0 Action tier false alarm events"


def test_all_publication_figures_exist_and_non_empty():
    expected_figs = [
        "fig1_data_card_masking.png",
        "fig2_calibration_reliability.png",
        "fig3_uqs_ers_dynamics.png",
        "fig4_hydraulic_risk_scatter.png",
        "fig5_triage_policy_tradeoff.png"
    ]
    for fig_name in expected_figs:
        fig_path = FIG_DIR / fig_name
        assert fig_path.exists(), f"Figure {fig_name} must exist in {FIG_DIR}"
        assert fig_path.stat().st_size > 50000, f"Figure {fig_name} must be a valid non-empty image (>50KB)"
