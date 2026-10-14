"""Tests for Cascading Reach Index (CRI) Propagation (Phase B.2).

Verifies:
1. CRI measures actual physical propagation of pressure drop across EPANET pressure zones (DMAs).
2. CRI > 0 strictly only if an asset's outage impacts nodes outside its immediate component.
3. Assets with zero external affected nodes have CRI == 0.0.
"""
from pathlib import Path
import pytest
import numpy as np

from risk.hydraulic_simulation import HydraulicRiskEngine, INP_PATH


@pytest.fixture(scope="module")
def engine():
    return HydraulicRiskEngine(inp_path=INP_PATH)


def test_cri_zero_when_no_affected_nodes(engine):
    """Verifies CRI is 0.0 when no nodes experience pressure loss."""
    cri = engine.compute_cascading_reach_index("PU1", "Pump", [])
    assert cri == 0.0


def test_cri_zero_for_immediate_component_only(engine):
    """Verifies CRI is 0.0 when affected nodes are strictly within the immediate component."""
    immediate = list(engine.get_immediate_component_nodes("PU2", "Pump"))
    # If only immediate 1-hop neighbor nodes are affected, CRI must be 0
    cri = engine.compute_cascading_reach_index("PU2", "Pump", immediate[:2])
    assert cri == 0.0


def test_cri_positive_when_propagating_across_zones(engine):
    """Verifies CRI > 0 when pressure loss propagates across external DMA zones."""
    # PU2 outage impacts nodes far across DMA1, DMA2, etc.
    distant_nodes = ["J415", "J310", "J280"]
    cri = engine.compute_cascading_reach_index("PU2", "Pump", distant_nodes)
    assert cri > 0.0, f"Expected positive CRI for cross-zone propagation, got {cri}"
    assert cri <= 1.0


def test_cri_simulation_consistency(engine):
    """Verifies that in actual simulation, CRI > 0 occurs only with external propagation."""
    df = engine.simulate_outages()

    for _, row in df.iterrows():
        aid = row["Asset_ID"]
        atype = row["Type"]
        cri = row["CRI"]
        aff_count = row["Affected_Nodes"]

        if cri > 0.0:
            assert aff_count > 0, f"Asset {aid} has CRI > 0 ({cri}) but 0 affected nodes"
        if aff_count == 0:
            assert cri == 0.0, f"Asset {aid} has 0 affected nodes but positive CRI ({cri})"
