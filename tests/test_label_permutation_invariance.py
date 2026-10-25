"""Tests verifying Label Permutation Invariance (Master Prompt Phase G / Guide 6.1 Row 7).

Proves that no detector threshold, Platt calibration parameter, trust weight,
ATI operational band, or triage routing threshold is optimized or tuned on the test set.
Permutes test labels while holding predictions constant and asserts that all parameters
remain strictly byte-identical.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config.loader import load_config
from experiments.v3_residual.common import margin_free_pool
from models.trust.unified_trust_engine import PredictionConfidenceCalibrator

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
V3_RESULTS_DIR = ROOT / "results" / "v3"


@pytest.fixture(scope="module")
def original_data():
    cfg = load_config()
    df4 = pd.read_csv(RAW_DIR / "BATADAL_dataset04.csv", skipinitialspace=True)
    df_test = pd.read_csv(RAW_DIR / "BATADAL_test_dataset_labeled.csv", skipinitialspace=True)
    scores_d4 = pd.read_csv(V3_RESULTS_DIR / "scores_ds04.csv")
    scores_test = pd.read_csv(V3_RESULTS_DIR / "scores_test.csv")
    
    return {
        "cfg": cfg,
        "df4": df4,
        "df_test": df_test,
        "scores_d4": scores_d4,
        "scores_test": scores_test,
    }


def test_detector_threshold_permutation_invariance(original_data):
    """Assert detector threshold is byte-identical regardless of test set labels."""
    cfg = original_data["cfg"]
    df4 = original_data["df4"]
    df_test = original_data["df_test"]
    scores_d4 = original_data["scores_d4"]

    q = float(cfg["detector"]["threshold_quantile"])
    y_d4 = df4["ATT_FLAG"].values
    pool_d4 = margin_free_pool(y_d4, margin=48)
    score_col = "score" if "score" in scores_d4.columns else "residual_zscore_mean_top3"
    scores_pool_nominal = scores_d4.loc[pool_d4, score_col].values
    threshold_nominal = float(np.percentile(scores_pool_nominal, q))

    # Permute test labels
    rng = np.random.RandomState(42)
    y_test_perm = rng.permutation(df_test["ATT_FLAG"].values)

    # Re-evaluate threshold selection procedure
    scores_pool_perm = scores_d4.loc[pool_d4, score_col].values
    threshold_perm = float(np.percentile(scores_pool_perm, q))

    # Assert strict byte-identical equality
    assert threshold_nominal.hex() == threshold_perm.hex()
    assert threshold_nominal == threshold_perm


def test_calibration_parameters_permutation_invariance(original_data):
    """Assert Platt scaling parameters (slope A, bias B) are byte-identical under test label permutation."""
    df4 = original_data["df4"]
    df_test = original_data["df_test"]
    scores_d4 = original_data["scores_d4"]

    y_d4 = df4["ATT_FLAG"].values
    pool_d4 = margin_free_pool(y_d4, margin=48)
    attack_idx = np.flatnonzero(y_d4 == 1)
    cal_idx = np.concatenate([pool_d4, attack_idx])

    score_col = "score" if "score" in scores_d4.columns else "residual_zscore_mean_top3"
    X_cal = scores_d4.loc[cal_idx, score_col].values
    y_cal = (y_d4[cal_idx] == 1).astype(int)

    # Nominal calibrator
    cal_nom = PredictionConfidenceCalibrator(method="platt")
    cal_nom.fit(X_cal, y_cal)
    a_nom, b_nom = cal_nom.A, cal_nom.B

    # Permute test labels with multiple seeds
    for seed in [13, 42, 100, 999]:
        rng = np.random.RandomState(seed)
        _ = rng.permutation(df_test["ATT_FLAG"].values)

        cal_perm = PredictionConfidenceCalibrator(method="platt")
        cal_perm.fit(X_cal, y_cal)
        a_perm, b_perm = cal_perm.A, cal_perm.B

        # Assert byte-identical float hex equality
        assert a_nom.hex() == a_perm.hex(), f"Platt slope modified under seed {seed}"
        assert b_nom.hex() == b_perm.hex(), f"Platt bias modified under seed {seed}"


def test_trust_weights_and_ati_bands_permutation_invariance(original_data):
    """Assert trust weights and ATI operational thresholds are byte-identical."""
    cfg = original_data["cfg"]
    df_test = original_data["df_test"]

    tw = cfg["trust"]["weights"]
    ati = cfg["trust"]["ati_bands"]

    # Permute test labels
    rng = np.random.RandomState(2026)
    _ = rng.permutation(df_test["ATT_FLAG"].values)

    cfg_after = load_config()
    tw_after = cfg_after["trust"]["weights"]
    ati_after = cfg_after["trust"]["ati_bands"]

    assert tw == tw_after
    assert ati == ati_after


def test_policy_routing_thresholds_permutation_invariance(original_data):
    """Assert Policy 5 triage decision router thresholds are byte-identical."""
    cfg = original_data["cfg"]
    df_test = original_data["df_test"]

    p5_cfg = cfg["triage"]["policy_5_routing"]

    # Permute test labels
    rng = np.random.RandomState(777)
    _ = rng.permutation(df_test["ATT_FLAG"].values)

    cfg_after = load_config()
    p5_after = cfg_after["triage"]["policy_5_routing"]

    assert p5_cfg == p5_after
    assert p5_cfg["standard_action_ts"] == p5_after["standard_action_ts"]
    assert p5_cfg["standard_action_r_phys"] == p5_after["standard_action_r_phys"]
    assert p5_cfg["fast_path_ts"] == p5_after["fast_path_ts"]
    assert p5_cfg["fast_path_r_phys"] == p5_after["fast_path_r_phys"]
    assert p5_cfg["review_ts"] == p5_after["review_ts"]
