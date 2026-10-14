"""Shared Alert Triage Policy Definitions and Cyber-Physical Risk Calculations.

This module provides the canonical, shared implementation of Policy 5
(Trust + Physical Risk Gated Triage) and sensor-to-asset mapping used identically
across experiments and the operational SCADA dashboard.
"""
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_PATH = ROOT / "network" / "mappings" / "batadal_epanet_mapping.csv"
DEFAULT_RISK_PATH = ROOT / "results" / "risk_assets_report.csv"


def load_risk_and_mapping(
    mapping_path: Optional[Path] = None,
    risk_path: Optional[Path] = None
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Loads mapping from telemetry variables to EPANET elements and Base_Risk values."""
    m_path = mapping_path or DEFAULT_MAPPING_PATH
    r_path = risk_path or DEFAULT_RISK_PATH
    
    s2a: Dict[str, str] = {}
    if m_path.exists():
        m_df = pd.read_csv(m_path)
        s2a = dict(zip(m_df["variable"], m_df["element"]))
        
    a2r: Dict[str, float] = {}
    if r_path.exists():
        r_df = pd.read_csv(r_path)
        a2r = dict(zip(r_df["Asset_ID"], r_df["Base_Risk"]))
        
    return s2a, a2r


def get_sensor_physical_risk(
    sensor_name: Optional[str],
    s2a: Dict[str, str],
    a2r: Dict[str, float],
    default_risk: float = 0.20
) -> Tuple[str, float]:
    """Resolves an anomalous sensor to its mapped physical asset ID and Base_Risk."""
    if not sensor_name or not isinstance(sensor_name, str):
        return "UNKNOWN", default_risk
        
    sensor_clean = sensor_name.strip()
    asset = s2a.get(sensor_clean)
    if not asset:
        return "UNMAPPED", default_risk
        
    risk = a2r.get(asset, default_risk)
    return asset, float(risk)


def compute_alert_physical_risk(
    top_sensors: list,
    s2a: Dict[str, str],
    a2r: Dict[str, float],
    default_risk: float = 0.20
) -> Tuple[str, float]:
    """Computes peak physical consequence across the top anomalous sensors for an alert."""
    best_asset = "UNMAPPED"
    max_risk = default_risk
    
    for sensor in top_sensors:
        asset, risk = get_sensor_physical_risk(sensor, s2a, a2r, default_risk)
        if risk > max_risk or best_asset in ("UNKNOWN", "UNMAPPED"):
            max_risk = risk
            best_asset = asset
            
    return best_asset, max_risk


from config.loader import load_config


def evaluate_policy_5_decision(
    alarm_persist: int,
    ts: float,
    r_phys: float,
    config: Optional[Dict] = None
) -> str:
    """Evaluates the Policy 5 triage decision for a single telemetry timestep.
    
    Thresholds are loaded strictly from authoritative config (config/trustrisk.yaml).
    """
    cfg = config or load_config()
    p5_cfg = cfg.get("triage", {}).get("policy_5_routing", {})
    std_ts = float(p5_cfg.get("standard_action_ts", 0.80))
    std_r = float(p5_cfg.get("standard_action_r_phys", 0.35))
    fast_ts = float(p5_cfg.get("fast_path_ts", 0.75))
    fast_r = float(p5_cfg.get("fast_path_r_phys", 0.55))
    rev_ts = float(p5_cfg.get("review_ts", 0.50))

    if alarm_persist == 1:
        is_action = (
            ((ts >= std_ts) and (r_phys >= std_r)) or
            ((ts >= fast_ts) and (r_phys >= fast_r))
        )
        if is_action:
            return "ACTION"
        if ts >= rev_ts:
            return "REVIEW"
        return "MONITOR"
    return "MONITOR"


def evaluate_policy_4_decision(
    alarm_persist: int,
    ts: float,
    config: Optional[Dict] = None
) -> str:
    """Evaluates Policy 4 (Trust Gate Only / ATI Bands) using thresholds from config."""
    cfg = config or load_config()
    bands = cfg.get("trust", {}).get("ati_bands", {})
    action_th = float(bands.get("action_threshold", 0.80))
    review_th = float(bands.get("review_threshold", 0.50))

    if alarm_persist == 1:
        if ts >= action_th:
            return "ACTION"
        if ts >= review_th:
            return "REVIEW"
        return "MONITOR"
    return "MONITOR"

