import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.trust.policy5_router import (
    load_risk_and_mapping,
    get_sensor_physical_risk,
    compute_alert_physical_risk,
    evaluate_policy_5_decision
)


def test_dashboard_default_mode_is_operational_no_oracle():
    # 1. Static file verification
    app_path = ROOT / "dashboard" / "app.py"
    content = app_path.read_text(encoding="utf-8")
    assert "Operational SCADA View" in content
    assert "Offline Evaluation / Forensic Mode" in content
    assert "index=0" in content
    assert 'if dashboard_mode == "Offline Evaluation / Forensic Mode":' in content
    assert "hidden in operational scada mode" in content.lower()

    # 2. Live rendered UI verification via Streamlit AppTest (MANDATORY RUNTIME CHECK)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(app_path))
    at.run()
    assert len(at.exception) == 0, f"Dashboard threw unhandled exceptions: {at.exception}"
    assert len(at.sidebar.radio) > 0
    assert at.sidebar.radio[0].value == "Operational SCADA View"
    
    # In Operational mode, ensure no ground-truth oracle labels are rendered
    all_rendered_text = " ".join([m.value for m in at.markdown] + [c.label for c in at.metric])
    assert "y_true" not in all_rendered_text
    assert "ATTACK" not in all_rendered_text
    assert "ground truth" not in all_rendered_text.lower()
    
    # In Forensic mode, verify ground-truth warning is rendered
    at.sidebar.radio[0].set_value("Offline Evaluation / Forensic Mode")
    at.run()
    assert len(at.exception) == 0
    forensic_text = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning])
    assert "ground-truth" in forensic_text.lower() or "oracle" in forensic_text.lower()


def test_shared_policy5_routing_equivalence():
    assert evaluate_policy_5_decision(1, 0.85, 0.68) == "ACTION"
    assert evaluate_policy_5_decision(1, 0.76, 0.60) == "ACTION"
    assert evaluate_policy_5_decision(1, 0.81, 0.36) == "ACTION"
    assert evaluate_policy_5_decision(1, 0.72, 0.60) == "REVIEW"
    assert evaluate_policy_5_decision(1, 0.79, 0.30) == "REVIEW"
    assert evaluate_policy_5_decision(1, 0.55, 0.20) == "REVIEW"
    assert evaluate_policy_5_decision(1, 0.40, 0.90) == "MONITOR"
    assert evaluate_policy_5_decision(0, 0.95, 0.95) == "MONITOR"


def test_sensor_to_asset_mapping_and_fallbacks():
    s2a, a2r = load_risk_and_mapping()
    asset, risk = get_sensor_physical_risk("L_T1", s2a, a2r)
    assert asset == "T1"
    assert risk > 0.40
    asset2, risk2 = get_sensor_physical_risk("F_PU1", s2a, a2r)
    assert asset2 == "PU1"
    assert risk2 > 0.40
    u_asset, u_rsk = get_sensor_physical_risk("INVALID_SENSOR_XY", s2a, a2r)
    assert u_asset == "UNMAPPED"
    assert u_rsk == 0.20
    n_asset, nonrsk = get_sensor_physical_risk(None, s2a, a2r)
    assert n_asset == "UNKNOWN"
    assert nonrsk == 0.20
    b_asset, b_rsk = compute_alert_physical_risk(["INVALID", "F_PU1", "L_T1"], s2a, a2r)
    assert b_asset in ("PU1", "T1")
    assert b_rsk >= 0.40


def test_block_bootstrap_report_integrity():
    report_file = ROOT / "results" / "v3" / "block_bootstrap_report.json"
    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["pointwise_f1_nominal"] == 0.8368
    assert rep["pointwise_recall_nominal"] == 0.7371
    assert rep["pointwise_precision_nominal"] == 0.9677
    blocks = rep["moving_block_bootstrap"]
    assert "12h_blocks" in blocks
    assert "24h_blocks" in blocks
    assert "48h_blocks" in blocks
    ci12 = blocks["12h_blocks"]["ci_95"]
    ci48 = blocks["48h_blocks"]["ci_95"]
    assert (ci48[1] - ci48[0]) > (ci12[1] - ci12[0])
