"""Attack Susceptibility Score (ASS) Module (Phase B.4).

Implements the 5-factor cyber-physical attack susceptibility evaluation:
1. Accessibility (network-facing SCADA/PLC interface vs. isolated)
2. Control Capability (direct actuator state control vs. passive sensor)
3. Protocol Vulnerability (unauthenticated industrial fieldbus like Modbus/TCP)
4. Detection Difficulty (stealth potential / residual masking capability)
5. Physical Isolation (remote perimeter facility vs. central secure station)

All weights are read from config/trustrisk.yaml.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any
import numpy as np
import pandas as pd

from config.loader import load_config


class AttackSusceptibilityEvaluator:
    """Evaluates 5-factor Attack Susceptibility Score (ASS) across WDS components."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.ass_cfg = self.config.get("risk", {}).get("ass_weights", {})
        
        self.gamma_1 = float(self.ass_cfg.get("gamma_1", 0.25)) # Accessibility
        self.gamma_2 = float(self.ass_cfg.get("gamma_2", 0.20)) # Control Capability
        self.gamma_3 = float(self.ass_cfg.get("gamma_3", 0.20)) # Protocol Vulnerability
        self.gamma_4 = float(self.ass_cfg.get("gamma_4", 0.20)) # Detection Difficulty
        self.gamma_5 = float(self.ass_cfg.get("gamma_5", 0.15)) # Physical Isolation

    def compute_accessibility(self, asset_id: str, asset_type: str) -> float:
        """Accessibility: Network-facing SCADA/PLC connectivity."""
        access_map = {
            "Pump": 0.95,      # Network-facing PLC with remote speed/status command interface
            "Valve": 0.85,     # Remote telemetry unit (RTU) / motorized solenoid valve
            "Tank": 0.80,      # Remote level telemetry transmitter link
            "Junction": 0.50,  # Field pressure transmitter tap
        }
        return access_map.get(asset_type, 0.20)

    def compute_control_capability(self, asset_id: str, asset_type: str) -> float:
        """Control Capability: Direct physical actuation power."""
        control_map = {
            "Pump": 1.00,      # Dynamic flow generation / high-head energy injection
            "Valve": 0.85,     # Flow throttling / boundary isolation
            "Tank": 0.60,      # Storage vessel capacity buffer (passive constraint)
            "Junction": 0.15,  # Passive demand node (zero actuation capability)
        }
        return control_map.get(asset_type, 0.10)

    def compute_protocol_vulnerability(self, asset_id: str, asset_type: str) -> float:
        """Protocol Vulnerability: Modbus/TCP or unauthenticated industrial fieldbus."""
        proto_map = {
            "Pump": 0.85,      # Unauthenticated Modbus/TCP command injection vulnerability
            "Valve": 0.85,     # Modbus coil write commands vulnerable to spoofing
            "Tank": 0.80,      # Unencrypted analog holding register telemetry
            "Junction": 0.65,  # Modbus sensor telemetry broadcast
        }
        return proto_map.get(asset_type, 0.30)

    def compute_detection_difficulty(self, asset_id: str, asset_type: str) -> float:
        """Detection Difficulty: Stealth potential and subtle bias manipulation."""
        detect_map = {
            "Valve": 0.85,     # Subtle throttling mimics normal friction / pressure head losses
            "Pump": 0.80,      # Standby switching or small frequency shifts blend into demand peaks
            "Tank": 0.70,      # Slow drift level alteration takes hours to exceed threshold
            "Junction": 0.60,  # Pressure sensor scaling / bias injection
        }
        return detect_map.get(asset_type, 0.40)

    def compute_physical_isolation(self, asset_id: str, asset_type: str) -> float:
        """Physical Isolation: Remote unmanned installation vs. hardened facility."""
        isolation_map = {
            "Tank": 0.85,      # Remote elevated reservoirs with sparse physical patrol
            "Valve": 0.80,     # Roadside valve vaults / perimeter distribution chambers
            "Pump": 0.70,      # Booster pump stations in suburban utility sheds
            "Junction": 0.35,  # Buried piping / street-level manholes
        }
        return isolation_map.get(asset_type, 0.30)

    def evaluate_asset(self, asset_id: str, asset_type: str) -> Dict[str, float]:
        """Calculates 5 susceptibility factors and unified ASS."""
        f_access = self.compute_accessibility(asset_id, asset_type)
        f_control = self.compute_control_capability(asset_id, asset_type)
        f_proto = self.compute_protocol_vulnerability(asset_id, asset_type)
        f_detect = self.compute_detection_difficulty(asset_id, asset_type)
        f_iso = self.compute_physical_isolation(asset_id, asset_type)

        ass = (
            self.gamma_1 * f_access +
            self.gamma_2 * f_control +
            self.gamma_3 * f_proto +
            self.gamma_4 * f_detect +
            self.gamma_5 * f_iso
        )

        return {
            "Asset_ID": asset_id,
            "Type": asset_type,
            "Accessibility": f_access,
            "Control_Capability": f_control,
            "Protocol_Vulnerability": f_proto,
            "Detection_Difficulty": f_detect,
            "Physical_Isolation": f_iso,
            "ASS": float(ass)
        }

    def evaluate_dataframe(self, df_assets: pd.DataFrame) -> pd.DataFrame:
        """Evaluates ASS across a dataframe of assets."""
        records = []
        for _, row in df_assets.iterrows():
            rec = self.evaluate_asset(row["Asset_ID"], row["Type"])
            records.append(rec)
        return pd.DataFrame(records)
