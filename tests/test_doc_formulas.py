"""Tests verifying formula and weight consistency between docs/MANUSCRIPT_REVISION.md and config/trustrisk.yaml."""
import re
from pathlib import Path
import pytest
from config.loader import load_config, get_config_hash

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_PATH = ROOT / "docs" / "MANUSCRIPT_REVISION.md"


@pytest.fixture
def manuscript_text():
    assert MANUSCRIPT_PATH.exists(), f"Missing {MANUSCRIPT_PATH}"
    return MANUSCRIPT_PATH.read_text(encoding="utf-8")


@pytest.fixture
def config():
    return load_config()


def test_config_hash_in_manuscript(manuscript_text):
    """Assert that the config hash in MANUSCRIPT_REVISION matches current authoritative config."""
    expected_hash = get_config_hash()
    assert expected_hash in manuscript_text, (
        f"Expected config hash {expected_hash} not found in {MANUSCRIPT_PATH}"
    )


def test_trust_score_weights_match_config(manuscript_text, config):
    """Verify 4-term and 3-term Trust Score weights match config/trustrisk.yaml."""
    tw = config["trust"]["weights"]

    # 4-term regex: w1 PCS + w2 (1 - UQS) + w3 ERS + w4 HRS
    m4 = re.search(
        r"\\text\{TS\}_t\s*=\s*([0-9.]+)\s*\\cdot\s*\\text\{PCS\}_t\s*\+\s*([0-9.]+)\s*\\cdot\s*\(1\s*-\s*\\text\{UQS\}_t\)\s*\+\s*([0-9.]+)\s*\\cdot\s*\\text\{ERS\}_t\s*\+\s*([0-9.]+)\s*\\cdot\s*\\text\{HRS\}_t",
        manuscript_text,
    )
    assert m4 is not None, "4-term Trust Score formula not found in manuscript"
    w1, w2, w3, w4 = [float(x) for x in m4.groups()]
    assert pytest.approx(w1) == tw["w1"]
    assert pytest.approx(w2) == tw["w2"]
    assert pytest.approx(w3) == tw["w3"]
    assert pytest.approx(w4) == tw["w4"]

    # 3-term regex (in cold-start context)
    m3 = re.search(
        r"cold-start.*?\\text\{TS\}_t\s*=\s*([0-9.]+)\s*\\cdot\s*\\text\{PCS\}_t\s*\+\s*([0-9.]+)\s*\\cdot\s*\(1\s*-\s*\\text\{UQS\}_t\)\s*\+\s*([0-9.]+)\s*\\cdot\s*\\text\{ERS\}_t",
        manuscript_text,
    )
    assert m3 is not None, "3-term Trust Score formula not found in manuscript"
    w_pcs, w_uqs, w_ers = [float(x) for x in m3.groups()]
    assert pytest.approx(w_pcs) == tw["w_pcs_3term"]
    assert pytest.approx(w_uqs) == tw["w_uqs_3term"]
    assert pytest.approx(w_ers) == tw["w_ers_3term"]


def test_tdcri_weights_match_config(manuscript_text, config):
    """Verify TDCRI formula weights match config/trustrisk.yaml."""
    rw = config["risk"]["tdcri_weights"]
    m = re.search(
        r"\\text\{TDCRI\}\s*=\s*([0-9.]+)\s*\\cdot\s*\\text\{MDCI\}\s*\+\s*([0-9.]+)\s*\\cdot\s*\\text\{ASS\}\s*\+\s*([0-9.]+)\s*\\cdot\s*\\text\{ISS\}\s*\+\s*([0-9.]+)\s*\\cdot\s*\\text\{CRI\}\s*\+\s*([0-9.]+)\s*\\cdot\s*\(1\s*-\s*\\text\{ARS\}\)\s*\+\s*([0-9.]+)\s*\\cdot\s*\\text\{TS\}",
        manuscript_text,
    )
    assert m is not None, "TDCRI formula not found in manuscript"
    w_mdci, w_ass, w_iss, w_cri, w_ars, w_ts = [float(x) for x in m.groups()]
    assert pytest.approx(w_mdci) == rw["w_mdci"]
    assert pytest.approx(w_ass) == rw["w_ass"]
    assert pytest.approx(w_iss) == rw["w_iss"]
    assert pytest.approx(w_cri) == rw["w_cri"]
    assert pytest.approx(w_ars) == rw["w_ars"]
    assert pytest.approx(w_ts) == rw["w_ts"]


def test_mdci_ass_ars_component_weights(manuscript_text, config):
    """Verify sub-component weights (beta, gamma, v) match config."""
    # MDCI
    mw = config["risk"]["mdci_weights"]
    m_beta = re.search(
        r"\\beta_1=([0-9.]+).*?\\beta_2=([0-9.]+).*?\\beta_3=([0-9.]+).*?\\beta_4=([0-9.]+)",
        manuscript_text,
    )
    assert m_beta is not None, "MDCI beta weights not found in manuscript"
    b1, b2, b3, b4 = [float(x) for x in m_beta.groups()]
    assert pytest.approx(b1) == mw["beta_1"]
    assert pytest.approx(b2) == mw["beta_2"]
    assert pytest.approx(b3) == mw["beta_3"]
    assert pytest.approx(b4) == mw["beta_4"]

    # ASS
    aw = config["risk"]["ass_weights"]
    m_gamma = re.search(
        r"\\gamma_1=([0-9.]+).*?\\gamma_2=([0-9.]+).*?\\gamma_3=([0-9.]+).*?\\gamma_4=([0-9.]+).*?\\gamma_5=([0-9.]+)",
        manuscript_text,
    )
    assert m_gamma is not None, "ASS gamma weights not found in manuscript"
    g1, g2, g3, g4, g5 = [float(x) for x in m_gamma.groups()]
    assert pytest.approx(g1) == aw["gamma_1"]
    assert pytest.approx(g2) == aw["gamma_2"]
    assert pytest.approx(g3) == aw["gamma_3"]
    assert pytest.approx(g4) == aw["gamma_4"]
    assert pytest.approx(g5) == aw["gamma_5"]

    # ARS
    rw = config["risk"]["ars_weights"]
    m_v = re.search(
        r"v_1=([0-9.]+).*?v_2=([0-9.]+).*?v_3=([0-9.]+).*?v_4=([0-9.]+)",
        manuscript_text,
    )
    assert m_v is not None, "ARS v weights not found in manuscript"
    v1, v2, v3, v4 = [float(x) for x in m_v.groups()]
    assert pytest.approx(v1) == rw["v_1"]
    assert pytest.approx(v2) == rw["v_2"]
    assert pytest.approx(v3) == rw["v_3"]
    assert pytest.approx(v4) == rw["v_4"]
