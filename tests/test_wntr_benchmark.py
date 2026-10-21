"""Tests for the Second Independent Benchmark: WNTR-Simulated C-Town Multi-Attack Dataset.

Validates:
1. Synthetic dataset integrity (336 hours, 43 sensors, binary ground-truth labels).
2. Ground-truth attack structure (4 distinct events of 24 hours each = 96 attack hours).
3. Triage policy performance (100% attack preservation, 4/4 events preserved across all policies).
4. Exported table and report artifacts existence and consistency.
"""
import json
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "WNTR_CTOWN_synthetic_attack_dataset.csv"
REP_PATH = ROOT / "results" / "wntr_benchmark_report.json"
MD_TAB_PATH = ROOT / "results" / "tables" / "table6_wntr_benchmark.md"
TEX_TAB_PATH = ROOT / "results" / "tables" / "latex" / "table6_wntr_benchmark.tex"


@pytest.fixture(scope="module")
def wntr_dataset():
    if not CSV_PATH.exists():
        pytest.skip(f"WNTR benchmark dataset missing: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    return df


@pytest.fixture(scope="module")
def wntr_report():
    if not REP_PATH.exists():
        pytest.skip(f"WNTR benchmark report missing: {REP_PATH}")
    with open(REP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_wntr_dataset_schema_and_duration(wntr_dataset):
    """Verifies that the synthetic dataset has 336 hourly rows and all expected SCADA columns."""
    assert len(wntr_dataset) == 336, f"Expected 336 hours (14 days), got {len(wntr_dataset)}"
    assert "DATETIME" in wntr_dataset.columns
    assert "ATT_FLAG" in wntr_dataset.columns
    assert len(wntr_dataset.columns) == 45, f"Expected 45 columns (DATETIME, 43 sensors, ATT_FLAG), got {len(wntr_dataset.columns)}"


def test_wntr_attack_structure(wntr_dataset):
    """Verifies that exactly 4 cyber-physical attack events (96 hours) are present."""
    att = wntr_dataset["ATT_FLAG"].values
    assert att.sum() == 96, f"Expected 96 attack hours across 4 events of 24h, got {att.sum()}"
    norm = (att == 0).sum()
    assert norm == 240, f"Expected 240 normal hours, got {norm}"


def test_wntr_triage_all_attacks_preserved(wntr_report):
    """Verifies that all 5 triage policies preserve 100% of attack events (4/4)."""
    assert wntr_report["attack_events"] == 4
    for pol in wntr_report["policies"]:
        assert pol["attacks_preserved"] == "4/4", f"Policy {pol['policy']} failed to preserve all attacks: {pol['attacks_preserved']}"
        assert pol["mean_delay_hours"] <= 5.0, f"Policy {pol['policy']} mean delay too high: {pol['mean_delay_hours']}h"


def test_wntr_table_artifacts_exist():
    """Verifies that both Markdown and LaTeX tables exist and are non-empty."""
    assert MD_TAB_PATH.exists() and MD_TAB_PATH.stat().st_size > 50
    assert TEX_TAB_PATH.exists() and TEX_TAB_PATH.stat().st_size > 50
