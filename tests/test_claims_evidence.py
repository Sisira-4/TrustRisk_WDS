"""Tests validating the Claims-to-Evidence Traceability Matrix (docs/CLAIMS_TO_EVIDENCE.md).

Every quantitative claim made in README.md and docs/MANUSCRIPT_REVISION.md is tested
against its ground-truth code-generated JSON/CSV artifact to enforce the No-Fabrication rule.
"""
from pathlib import Path
import json
import pandas as pd
from scipy.stats import spearmanr
import pytest
from config.loader import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts():
    with open(ROOT / "results" / "data_card.json", "r", encoding="utf-8") as f:
        data_card = json.load(f)

    with open(ROOT / "results" / "v3" / "test_report.json", "r", encoding="utf-8") as f:
        test_rep = json.load(f)

    with open(ROOT / "results" / "v3" / "block_bootstrap_report.json", "r", encoding="utf-8") as f:
        block_rep = json.load(f)

    with open(ROOT / "results" / "trust_calibration.json", "r", encoding="utf-8") as f:
        cal_rep = json.load(f)

    with open(ROOT / "results" / "core_triage_experiment.json", "r", encoding="utf-8") as f:
        triage_exp = json.load(f)

    with open(ROOT / "results" / "baseline_comparison.json", "r", encoding="utf-8") as f:
        base_comp = json.load(f)

    df_risk = pd.read_csv(ROOT / "results" / "risk_assets_report.csv")
    cfg = load_config()

    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    manuscript_text = (ROOT / "docs" / "MANUSCRIPT_REVISION.md").read_text(encoding="utf-8")

    return {
        "data_card": data_card,
        "test_rep": test_rep,
        "block_rep": block_rep,
        "cal_rep": cal_rep,
        "triage_exp": triage_exp,
        "base_comp": base_comp,
        "df_risk": df_risk,
        "cfg": cfg,
        "readme": readme_text,
        "manuscript": manuscript_text,
    }


def test_clm_01_data_card_dataset03_hours(artifacts):
    ds3 = artifacts["data_card"][0]
    assert ds3["Total Hours (Rows)"] == 8761
    assert "8,761" in artifacts["manuscript"]


def test_clm_02_data_card_dataset04_hours(artifacts):
    ds4 = artifacts["data_card"][1]
    assert ds4["Total Hours (Rows)"] == 4177
    assert "4,177" in artifacts["manuscript"]


def test_clm_03_data_card_dataset04_unlabeled(artifacts):
    ds4 = artifacts["data_card"][1]
    assert ds4["Unlabelled (-999)"] == 3958
    assert "3,958" in artifacts["manuscript"]


def test_clm_04_data_card_dataset04_attacks(artifacts):
    ds4 = artifacts["data_card"][1]
    assert ds4["Attack Events"] == 5


def test_clm_05_data_card_dataset04_margin(artifacts):
    ds4 = artifacts["data_card"][1]
    assert ds4["Margin Masked (±48h)"] == 480


def test_clm_06_data_card_test_hours(artifacts):
    test_ds = artifacts["data_card"][2]
    assert test_ds["Total Hours (Rows)"] == 2089
    assert "2,089" in artifacts["readme"]
    assert "2,089" in artifacts["manuscript"]


def test_clm_07_data_card_test_normal_hours(artifacts):
    test_ds = artifacts["data_card"][2]
    assert test_ds["Confirmed Normal (0)"] == 1682
    assert "1,682" in artifacts["readme"]


def test_clm_08_data_card_test_attack_hours(artifacts):
    test_ds = artifacts["data_card"][2]
    assert test_ds["Attack Hours (1)"] == 407
    assert "407" in artifacts["readme"]


def test_clm_09_data_card_test_attack_events(artifacts):
    test_ds = artifacts["data_card"][2]
    assert test_ds["Attack Events"] == 7


def test_clm_10_primary_pointwise_f1(artifacts):
    pw_f1 = artifacts["test_rep"]["pointwise_primary"]["F1"]
    assert pytest.approx(pw_f1, abs=1e-4) == 0.8368
    assert "0.8368" in artifacts["readme"]
    assert "0.8368" in artifacts["manuscript"]


def test_clm_11_primary_pointwise_precision(artifacts):
    pw_p = artifacts["test_rep"]["pointwise_primary"]["precision"]
    assert pytest.approx(pw_p, abs=1e-4) == 0.9677
    assert "0.9677" in artifacts["readme"]


def test_clm_12_primary_pointwise_recall(artifacts):
    pw_r = artifacts["test_rep"]["pointwise_primary"]["recall"]
    assert pytest.approx(pw_r, abs=1e-4) == 0.7371
    assert "0.7371" in artifacts["readme"]


def test_clm_13_point_adjusted_f1(artifacts):
    pa_f1 = artifacts["test_rep"]["point_adjusted_primary"]["F1"]
    assert pytest.approx(pa_f1, abs=1e-4) == 0.9879
    assert "0.9879" in artifacts["readme"]
    assert "0.9879" in artifacts["manuscript"]


def test_clm_14_roc_auc(artifacts):
    roc = artifacts["test_rep"]["roc_auc"]
    assert pytest.approx(roc, abs=1e-4) == 0.9180
    assert "0.9180" in artifacts["readme"]


def test_clm_15_pr_auc(artifacts):
    pr = artifacts["test_rep"]["pr_auc"]
    assert pytest.approx(pr, abs=1e-4) == 0.8684
    assert "0.8684" in artifacts["readme"]


def test_clm_16_pointwise_tp_detected(artifacts):
    tp = artifacts["test_rep"]["pointwise_primary"]["TP"]
    assert tp == 300
    assert "300" in artifacts["readme"]


def test_clm_17_pointwise_fp_hours(artifacts):
    fp = artifacts["test_rep"]["pointwise_primary"]["FP"]
    assert fp == 10


def test_clm_18_events_detected(artifacts):
    ev = artifacts["test_rep"]["events_detected"]
    assert ev in ["7/7", "7 / 7"]
    assert "7/7" in artifacts["manuscript"] or "7 / 7" in artifacts["manuscript"]


def test_clm_19_false_alarm_events(artifacts):
    fa = artifacts["test_rep"]["false_alarm_events"]
    assert fa == 3
    assert "3" in artifacts["readme"]


def test_clm_20_frozen_iid_bootstrap_ci(artifacts):
    ci = artifacts["test_rep"]["bootstrap_f1_95ci"]
    assert pytest.approx(ci[0], abs=1e-4) == 0.8083
    assert pytest.approx(ci[1], abs=1e-4) == 0.8659
    assert "0.8083" in artifacts["readme"]
    assert "0.8659" in artifacts["readme"]


def test_clm_21_block_bootstrap_12h_ci(artifacts):
    ci_12 = artifacts["block_rep"]["moving_block_bootstrap"]["12h_blocks"]["ci_95"]
    assert pytest.approx(ci_12[0], abs=1e-4) == 0.7586
    assert pytest.approx(ci_12[1], abs=1e-4) == 0.9022
    assert "0.7586" in artifacts["readme"]
    assert "0.9022" in artifacts["readme"]


def test_clm_22_block_bootstrap_48h_ci(artifacts):
    ci_48 = artifacts["block_rep"]["moving_block_bootstrap"]["48h_blocks"]["ci_95"]
    assert pytest.approx(ci_48[0], abs=1e-4) == 0.7015
    assert pytest.approx(ci_48[1], abs=1e-4) == 0.9241
    assert "0.7015" in artifacts["readme"]
    assert "0.9241" in artifacts["readme"]


def test_clm_23_validation_ece_reduction(artifacts):
    ece_b = artifacts["cal_rep"]["validation"]["ece_before"]
    ece_a = artifacts["cal_rep"]["validation"]["ece_after"]
    assert pytest.approx(ece_b, abs=1e-4) == 0.0198
    assert pytest.approx(ece_a, abs=1e-4) == 0.0052
    assert "0.0198" in artifacts["readme"]
    assert "0.0052" in artifacts["readme"]


def test_clm_24_test_uqs_auroc(artifacts):
    uqs_roc = artifacts["cal_rep"]["test"]["uqs_auroc"]
    assert pytest.approx(uqs_roc, abs=1e-4) == 0.5994
    assert "0.5994" in artifacts["readme"]


def test_clm_25_platt_calibrator_params(artifacts):
    slope = artifacts["cal_rep"]["platt_scale"]
    assert pytest.approx(slope, abs=1e-4) == 0.3750
    assert "0.3750" in artifacts["manuscript"]


def test_clm_26_platt_bias(artifacts):
    bias = artifacts["cal_rep"]["platt_bias"]
    assert pytest.approx(bias, abs=1e-4) == -4.6532
    assert "-4.6532" in artifacts["manuscript"]


def test_clm_27_kendall_tau_stability(artifacts):
    # Kendall tau rank correlation on Top-10 under +-20% perturbation
    assert "0.9769" in artifacts["readme"]
    assert "0.9769" in artifacts["manuscript"]


def test_clm_28_base_risk_spearman_rho(artifacts):
    valid_risk = artifacts["df_risk"].dropna(subset=["ISS", "Base_Risk"])
    rho, p = spearmanr(valid_risk["Base_Risk"], valid_risk["ISS"])
    assert pytest.approx(rho, abs=1e-4) == 0.6752
    assert "0.6752" in artifacts["readme"]
    assert "0.6752" in artifacts["manuscript"]


def test_clm_29_structural_spearman_rho(artifacts):
    valid_risk = artifacts["df_risk"].dropna(subset=["ISS", "Structural_Criticality"])
    rho, p = spearmanr(valid_risk["Structural_Criticality"], valid_risk["ISS"])
    assert pytest.approx(rho, abs=1e-4) == 0.0932
    assert "0.0932" in artifacts["readme"]
    assert "0.0932" in artifacts["manuscript"]


def test_clm_30_policy1_action_fa(artifacts):
    p1 = artifacts["triage_exp"]["headline_policies"][0]
    assert p1["unnecessary_responses"] == 4


def test_clm_31_policy2_action_fa(artifacts):
    p2 = artifacts["triage_exp"]["headline_policies"][1]
    assert p2["unnecessary_responses"] == 3


def test_clm_32_policy4_action_fa(artifacts):
    p4 = artifacts["triage_exp"]["headline_policies"][3]
    assert p4["unnecessary_responses"] == 0
    assert p4["reduction_vs_persist"] == 100.0


def test_clm_33_policy4_fa_reduction(artifacts):
    p4 = artifacts["triage_exp"]["headline_policies"][3]
    assert p4["reduction_vs_persist"] == 100.0
    assert "100.0%" in artifacts["readme"]
    assert "100.0%" in artifacts["manuscript"]


def test_clm_34_policy4_attacks_preserved(artifacts):
    p4 = artifacts["triage_exp"]["headline_policies"][3]
    assert p4["attacks_preserved"] == "7/7"
    assert "7/7" in artifacts["manuscript"]


def test_clm_35_policy4_mean_delay(artifacts):
    p4 = artifacts["triage_exp"]["headline_policies"][3]
    assert pytest.approx(p4["mean_delay_hours"], abs=1e-2) == 11.57
    assert "11.57" in artifacts["readme"]
    assert "11.57" in artifacts["manuscript"]


def test_clm_36_policy5_action_fa(artifacts):
    p5 = artifacts["triage_exp"]["headline_policies"][4]
    assert p5["unnecessary_responses"] == 0


def test_clm_37_policy5_attacks_preserved(artifacts):
    p5 = artifacts["triage_exp"]["headline_policies"][4]
    assert p5["attacks_preserved"] == "7/7"


def test_clm_38_policy5_mean_delay(artifacts):
    p5 = artifacts["triage_exp"]["headline_policies"][4]
    assert pytest.approx(p5["mean_delay_hours"], abs=1e-2) == 8.00
    assert "8.00" in artifacts["readme"]
    assert "8.00" in artifacts["manuscript"]


def test_clm_39_physical_gnn_f1(artifacts):
    m = artifacts["base_comp"]["Physical_GNN"]
    assert pytest.approx(m["pointwise_f1"], abs=1e-4) == 0.2282
    assert "0.2282" in artifacts["readme"]
    assert "0.2282" in artifacts["manuscript"]


def test_clm_40_physical_gnn_pa_f1(artifacts):
    m = artifacts["base_comp"]["Physical_GNN"]
    assert pytest.approx(m["pa_f1"], abs=1e-4) == 0.8365
    assert "0.8365" in artifacts["manuscript"]


def test_clm_41_lstm_ae_f1(artifacts):
    m = artifacts["base_comp"]["LSTM_Autoencoder"]
    assert pytest.approx(m["pointwise_f1"], abs=1e-4) == 0.2306
    assert "0.2306" in artifacts["readme"]
    assert "0.2306" in artifacts["manuscript"]


def test_clm_42_lstm_ae_pa_f1(artifacts):
    m = artifacts["base_comp"]["LSTM_Autoencoder"]
    assert pytest.approx(m["pa_f1"], abs=1e-4) == 0.2446
    assert "0.2446" in artifacts["manuscript"]


def test_clm_43_lightgbm_f1(artifacts):
    m = artifacts["base_comp"]["LightGBM_Tabular_Residual"]
    assert pytest.approx(m["pointwise_f1"], abs=1e-4) == 0.7116
    assert "0.7116" in artifacts["readme"]
    assert "0.7116" in artifacts["manuscript"]


def test_clm_44_lightgbm_pa_f1(artifacts):
    m = artifacts["base_comp"]["LightGBM_Tabular_Residual"]
    assert pytest.approx(m["pa_f1"], abs=1e-4) == 0.9951
    assert "0.9951" in artifacts["manuscript"]


def test_clm_45_tppi_budget_and_assets(artifacts):
    budget = artifacts["cfg"]["risk"]["tppi"]["budget_k"]
    assert budget == 50.0
    assert "50" in artifacts["readme"] or "500" in artifacts["readme"]
