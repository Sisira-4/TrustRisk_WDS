"""Tests for Hydraulic Simulation Convergence and Error Tracking (Phase B.1).

Verifies:
1. Every asset marked valid in the simulation had a converged WNTR run (Converged == True, Error_Code == 0).
2. The invalid assets tracking file (results/risk_invalid_assets.csv) is generated and formatted properly.
3. No silent fallbacks: any unconverged or errored runs are excluded from valid rankings.
"""
from pathlib import Path
import pytest
import pandas as pd

from risk.hydraulic_simulation import HydraulicRiskEngine, INP_PATH, RESULTS_DIR


@pytest.fixture(scope="module")
def simulated_engine():
    engine = HydraulicRiskEngine(inp_path=INP_PATH)
    df = engine.simulate_outages()
    return engine, df


def test_hydraulic_convergence_all_valid(simulated_engine):
    """Verifies that every asset in the output table actually achieved numerical convergence."""
    _, df = simulated_engine

    assert "Converged" in df.columns, "Output dataframe must explicitly record convergence"
    assert "Error_Code" in df.columns, "Output dataframe must record error code"
    assert "Iterations" in df.columns, "Output dataframe must record iteration count"

    # All assets included in final analysis must have converged
    assert df["Converged"].all(), "Every asset marked valid must have Converged == True"
    assert (df["Error_Code"] == 0).all(), "Every asset marked valid must have Error_Code == 0"
    assert (df["Iterations"] > 0).all(), "Every converged asset must have positive iterations"


def test_invalid_assets_log_exists():
    """Verifies that results/risk_invalid_assets.csv exists and has expected schema."""
    invalid_csv = RESULTS_DIR / "risk_invalid_assets.csv"
    assert invalid_csv.exists(), "risk_invalid_assets.csv must be generated"

    df_inv = pd.read_csv(invalid_csv)
    expected_cols = ["Asset_ID", "Type", "Error_Trace", "Converged", "Error_Code"]
    for col in expected_cols:
        assert col in df_inv.columns, f"Missing column {col} in risk_invalid_assets.csv"
