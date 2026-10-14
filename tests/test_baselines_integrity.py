"""Integrity and Mathematical Coherence Tests for Baseline Comparisons (Phase 2 & D1).

Verifies that:
1. results/baseline_comparison.json is populated by experiments/run_baseline_comparison.py.
2. All baseline models maintain internal mathematical consistency across TP/FP/FN/TN.
3. Point-adjusted metrics never drop below pointwise metrics.
4. Recall == 1.0 strictly implies Events Detected == 7 / 7.
5. Physical-GNN impossible values (0.1959 / 1.0000 with 5/7 events) are absent and replaced
   by mathematically grounded checkpoint evaluations.
6. Multi-seed neural baselines record seed distributions (>= 3 seeds).
7. Authoritative config hash is embedded.
"""
from pathlib import Path
import json
import pytest

from config.loader import get_config_hash

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@pytest.fixture(scope="module")
def baseline_data():
    base_file = RESULTS_DIR / "baseline_comparison.json"
    assert base_file.exists(), "results/baseline_comparison.json must exist"
    with open(base_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_baseline_completeness_and_config_hash(baseline_data):
    expected_models = [
        "Physical_GNN",
        "LSTM_Autoencoder",
        "LightGBM_Tabular_Residual",
        "Raw_Threshold",
        "Persistence_Rule",
        "MultiSensor_Ridge_Residual_Ours",
        "TrustRisk_WDS_Policy4_TrustGate",
        "TrustRisk_WDS_Policy5_TrustRiskGate"
    ]
    for model in expected_models:
        assert model in baseline_data, f"Required model {model} missing from baseline comparison"
    
    assert "config_sha256" in baseline_data, "config_sha256 missing from baseline_comparison.json"
    assert baseline_data["config_sha256"] == get_config_hash(), "Baseline config hash mismatch"


def test_table2d_mathematical_consistency(baseline_data):
    """Asserts that precision, recall, and F1 recomputed from TP/FP/FN match recorded numbers."""
    eval_models = [
        "Physical_GNN",
        "LSTM_Autoencoder",
        "LightGBM_Tabular_Residual",
        "Raw_Threshold",
        "Persistence_Rule"
    ]
    
    for m_name in eval_models:
        m = baseline_data[m_name]
        
        # Pointwise precision
        if m["tp"] + m["fp"] > 0:
            p_calc = m["tp"] / (m["tp"] + m["fp"])
            assert abs(p_calc - m["pointwise_precision"]) < 1e-3, (
                f"Model {m_name} Pointwise Precision mismatch: calc {p_calc:.4f} != {m['pointwise_precision']}"
            )
            
        # Pointwise recall
        if m["tp"] + m["fn"] > 0:
            r_calc = m["tp"] / (m["tp"] + m["fn"])
            assert abs(r_calc - m["pointwise_recall"]) < 1e-3, (
                f"Model {m_name} Pointwise Recall mismatch: calc {r_calc:.4f} != {m['pointwise_recall']}"
            )
            
        # Pointwise F1
        if m["pointwise_precision"] + m["pointwise_recall"] > 0:
            f1_calc = (2 * m["pointwise_precision"] * m["pointwise_recall"] /
                       (m["pointwise_precision"] + m["pointwise_recall"]))
            assert abs(f1_calc - m["pointwise_f1"]) < 1e-3, (
                f"Model {m_name} Pointwise F1 mismatch: calc {f1_calc:.4f} != {m['pointwise_f1']}"
            )
            
        # Point-adjusted precision
        if m["pa_tp"] + m["pa_fp"] > 0:
            pa_p_calc = m["pa_tp"] / (m["pa_tp"] + m["pa_fp"])
            assert abs(pa_p_calc - m["pa_precision"]) < 1e-3, (
                f"Model {m_name} PA Precision mismatch: calc {pa_p_calc:.4f} != {m['pa_precision']}"
            )
            
        # Point-adjusted recall
        if m["pa_tp"] + m["pa_fn"] > 0:
            pa_r_calc = m["pa_tp"] / (m["pa_tp"] + m["pa_fn"])
            assert abs(pa_r_calc - m["pa_recall"]) < 1e-3, (
                f"Model {m_name} PA Recall mismatch: calc {pa_r_calc:.4f} != {m['pa_recall']}"
            )
            
        # Point-adjusted F1
        if m["pa_precision"] + m["pa_recall"] > 0:
            pa_f1_calc = (2 * m["pa_precision"] * m["pa_recall"] /
                          (m["pa_precision"] + m["pa_recall"]))
            assert abs(pa_f1_calc - m["pa_f1"]) < 1e-3, (
                f"Model {m_name} PA F1 mismatch: calc {pa_f1_calc:.4f} != {m['pa_f1']}"
            )
            
        # Invariant: PA F1 cannot be lower than Pointwise F1
        assert m["pa_f1"] >= m["pointwise_f1"] - 1e-4, (
            f"Model {m_name} PA F1 ({m['pa_f1']}) < Pointwise F1 ({m['pointwise_f1']})"
        )


def test_recall_event_detection_integrity(baseline_data):
    """Asserts that any model with recall == 1.0 has 7/7 events detected."""
    for m_name, m in baseline_data.items():
        if not isinstance(m, dict) or "pointwise_recall" not in m:
            continue
        if m.get("pointwise_recall") == 1.0 or m.get("pa_recall") == 1.0:
            assert m["events_detected"] == "7 / 7", (
                f"Model {m_name} has recall 1.0 but events detected is {m['events_detected']}"
            )


def test_physical_gnn_impossible_values_eliminated(baseline_data):
    """Verifies that the old impossible Physical-GNN values are gone."""
    gnn = baseline_data["Physical_GNN"]
    # Old impossible values: precision 0.1959, recall 1.0000 with 5/7 events
    assert gnn["pointwise_precision"] > 0.50, f"Expected real precision, got {gnn['pointwise_precision']}"
    assert gnn["pointwise_recall"] < 0.50, f"Expected real recall, got {gnn['pointwise_recall']}"
    assert gnn["events_detected"] == "5 / 7"
    assert abs(gnn["pointwise_f1"] - 0.2282) < 0.01
    assert abs(gnn["pa_f1"] - 0.8365) < 0.01


def test_neural_baseline_multi_seed(baseline_data):
    """Verifies that neural baseline evaluates >= 3 seeds and reports seed variation."""
    lstm = baseline_data["LSTM_Autoencoder"]
    assert "seeds_evaluated" in lstm, "LSTM Autoencoder must record seeds_evaluated"
    assert len(lstm["seeds_evaluated"]) >= 3, "LSTM Autoencoder must evaluate >= 3 seeds"
    assert "pw_f1_std" in lstm
    assert "pa_f1_std" in lstm
