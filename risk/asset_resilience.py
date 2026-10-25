"""Asset Resilience Score (ARS) Module (Phase B.5).

Implements the 4-factor cyber-physical asset resilience evaluation:
1. Redundancy (parallel paths, standby pumps, or redundant links in EPANET graph)
2. Recovery Time (feasibility of rapid manual isolation / bypass switching)
3. Alternative Supply Paths (graph connectivity and non-bridge routing margin)
4. Operational Flexibility (hydraulic storage buffer capacity, e.g. tanks vs. pure actuators)

All weights are read directly from config/trustrisk.yaml.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any
import numpy as np
import pandas as pd
import networkx as nx
import wntr

from config.loader import load_config


class AssetResilienceEvaluator:
    """Evaluates 4-factor Asset Resilience Score (ARS) across WDS components."""

    def __init__(self, wn: wntr.network.WaterNetworkModel, config: Dict[str, Any] = None):
        self.wn = wn
        self.config = config or load_config()
        self.ars_cfg = self.config.get("risk", {}).get("ars_weights", {})
        
        self.v_1 = float(self.ars_cfg.get("v_1", 0.30)) # Redundancy
        self.v_2 = float(self.ars_cfg.get("v_2", 0.25)) # Recovery Time
        self.v_3 = float(self.ars_cfg.get("v_3", 0.25)) # Alternative Paths
        self.v_4 = float(self.ars_cfg.get("v_4", 0.20)) # Operational Flexibility
        
        self.G = self.wn.to_graph().to_undirected()
        self._identify_bridges()

    def _identify_bridges(self):
        """Identifies bridge edges and articulation points in the network graph."""
        self.bridges = set(nx.bridges(self.G))
        self.articulation_points = set(nx.articulation_points(self.G))

    def compute_redundancy(self, asset_id: str, asset_type: str) -> float:
        """Redundancy: Standby operational units or parallel pipelines."""
        if asset_type == "Pump":
            # CTOWN pump stations operate in twin pairs: PU1/PU2, PU4/PU5, PU6/PU7, PU8/PU9, PU10/PU11
            twin_pairs = {
                "PU1": "PU2", "PU2": "PU1",
                "PU4": "PU5", "PU5": "PU4",
                "PU6": "PU7", "PU7": "PU6",
                "PU8": "PU9", "PU9": "PU8",
                "PU10": "PU11", "PU11": "PU10"
            }
            return 0.85 if asset_id in twin_pairs else 0.30
        elif asset_type == "Tank":
            # 7 interconnected storage tanks provide network-wide storage buffer
            return 0.65
        elif asset_type == "Valve":
            # Control valves in CTOWN often lack bypass pipes
            return 0.25
        elif asset_type == "Junction":
            deg = self.G.degree(asset_id) if asset_id in self.G else 2
            return min(1.0, deg / 4.0)
        return 0.30

    def compute_recovery_time_score(self, asset_type: str) -> float:
        """Recovery Time Score: Rapid isolation and bypass capability (higher = faster recovery)."""
        recovery_time_map = {
            "Junction": 0.85,  # Rapid local isolation valve shutoff (1-2 hours)
            "Valve": 0.75,     # Manual bypass handwheel operation (2-4 hours)
            "Pump": 0.50,      # Standby pump switchover or electrical lockout (4-8 hours)
            "Tank": 0.25,      # Emptying, structural inspection, water quality testing (24-48 hours)
        }
        return recovery_time_map.get(asset_type, 0.40)

    def compute_alternative_paths(self, asset_id: str, asset_type: str) -> float:
        """Alternative Paths: Graph connectivity margin if this component is removed."""
        if asset_type in ("Pump", "Valve"):
            try:
                link = self.wn.get_link(asset_id)
                u, v = link.start_node_name, link.end_node_name
                edge = (u, v) if (u, v) in self.bridges else (v, u)
                if edge in self.bridges:
                    return 0.15  # Critical bridge link; outage partitions network
                # Check path count
                paths = list(nx.all_simple_paths(self.G, u, v, cutoff=5))
                return min(1.0, len(paths) / 3.0)
            except Exception:
                return 0.40
        else:
            if asset_id in self.articulation_points:
                return 0.15  # Articulation node; failure disconnects graph
            deg = self.G.degree(asset_id) if asset_id in self.G else 2
            return min(1.0, max(0.2, (deg - 1) / 3.0))

    def compute_operational_flexibility(self, asset_id: str, asset_type: str) -> float:
        """Operational Flexibility: Physical hydraulic buffer capacity."""
        if asset_type == "Tank":
            return 0.95  # Bulk water storage acts as hydraulic shock absorber
        elif asset_type == "Junction":
            return 0.50  # Meshed grid manifold provides localized inertia
        elif asset_type == "Valve":
            return 0.25  # Throttling flexibility without storage
        elif asset_type == "Pump":
            return 0.15  # Dynamic machine with zero static buffer capacity
        return 0.20

    def evaluate_asset(self, asset_id: str, asset_type: str) -> Dict[str, float]:
        """Calculates 4 resilience factors and unified ARS."""
        r = self.compute_redundancy(asset_id, asset_type)
        t = self.compute_recovery_time_score(asset_type)
        a = self.compute_alternative_paths(asset_id, asset_type)
        f = self.compute_operational_flexibility(asset_id, asset_type)

        ars = self.v_1 * r + self.v_2 * t + self.v_3 * a + self.v_4 * f

        return {
            "Asset_ID": asset_id,
            "Type": asset_type,
            "Redundancy": float(r),
            "Recovery_Time_Score": float(t),
            "Alternative_Paths": float(a),
            "Operational_Flexibility": float(f),
            "ARS": float(ars)
        }

    def evaluate_dataframe(self, df_assets: pd.DataFrame) -> pd.DataFrame:
        """Evaluates ARS across a dataframe of assets."""
        records = []
        for _, row in df_assets.iterrows():
            rec = self.evaluate_asset(row["Asset_ID"], row["Type"])
            records.append(rec)
        return pd.DataFrame(records)
