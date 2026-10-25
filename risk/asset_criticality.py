"""Multi-Dimensional Criticality Index (MDCI) Module.

Implements the 4-dimensional physical asset criticality formulation:
1. Structural Criticality (network graph betweenness, closeness, degree centrality)
2. Operational Criticality (nominal hydraulic service demand / volume / design flow)
3. Monitoring Criticality (downstream and adjacent SCADA sensor dependencies)
4. Recovery Complexity (empirical mean-time-to-repair hierarchy: Tanks > Pumps > Valves > Junctions)

Weights are read directly from config/trustrisk.yaml.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
import wntr

from config.loader import load_config


class AssetCriticalityEvaluator:
    """Evaluates Multi-Dimensional Criticality Index (MDCI) across WDS components."""

    def __init__(self, wn: wntr.network.WaterNetworkModel, config: Dict[str, Any] = None):
        self.wn = wn
        self.config = config or load_config()
        self.risk_cfg = self.config.get("risk", {})
        self.mdci_cfg = self.risk_cfg.get("mdci_weights", {})
        
        self.beta_1 = float(self.mdci_cfg.get("beta_1", 0.25))
        self.beta_2 = float(self.mdci_cfg.get("beta_2", 0.25))
        self.beta_3 = float(self.mdci_cfg.get("beta_3", 0.25))
        self.beta_4 = float(self.mdci_cfg.get("beta_4", 0.25))
        
        struct_sub = self.mdci_cfg.get("structural_subcomponents", {})
        self.w_bw = float(struct_sub.get("betweenness", 0.40))
        self.w_cl = float(struct_sub.get("closeness", 0.40))
        self.w_deg = float(struct_sub.get("degree", 0.20))
        
        self._build_graph_metrics()

    def _build_graph_metrics(self):
        """Precomputes graph topological centrality on the EPANET network."""
        self.G = self.wn.to_graph().to_undirected()
        self.node_betweenness = nx.betweenness_centrality(self.G)
        self.node_closeness = nx.closeness_centrality(self.G)
        self.node_degree = nx.degree_centrality(self.G)

    def compute_structural_criticality(self, asset_id: str, asset_type: str) -> float:
        """Computes composite structural topological criticality in [0, 1]."""
        if asset_type in ("Pump", "Valve"):
            try:
                link = self.wn.get_link(asset_id)
                u, v = link.start_node_name, link.end_node_name
                bw = 0.5 * (self.node_betweenness.get(u, 0.0) + self.node_betweenness.get(v, 0.0))
                cl = 0.5 * (self.node_closeness.get(u, 0.0) + self.node_closeness.get(v, 0.0))
                deg = 0.5 * (self.node_degree.get(u, 0.0) + self.node_degree.get(v, 0.0))
            except Exception:
                bw, cl, deg = 0.05, 0.05, 0.05
        else:
            bw = self.node_betweenness.get(asset_id, 0.0)
            cl = self.node_closeness.get(asset_id, 0.0)
            deg = self.node_degree.get(asset_id, 0.0)

        struct_map = {"Pump": 0.85, "Valve": 0.80, "Tank": 0.70, "Junction": 0.20}
        arterial = struct_map.get(asset_type, 0.20)
        raw = arterial * (0.60 + 0.40 * (self.w_bw * bw + self.w_cl * cl + self.w_deg * deg))
        return float(np.clip(raw, 0.05, 1.0))

    def compute_operational_criticality(self, asset_id: str, asset_type: str) -> float:
        """Quantifies operational throughput / demand service capacity from intrinsic physical design specifications."""
        if asset_type == "Tank":
            try:
                tank = self.wn.get_node(asset_id)
                vol = np.pi * (float(tank.diameter) / 2.0) ** 2 * (float(tank.max_level) - float(tank.min_level))
                return float(min(1.0, vol / 5000.0))
            except Exception:
                return 0.8
        elif asset_type == "Pump":
            try:
                pump = self.wn.get_link(asset_id)
                power = float(pump.power) if hasattr(pump, "power") and pump.power else 50.0
                return float(min(1.0, power / 100.0))
            except Exception:
                return 0.7
        elif asset_type == "Valve":
            try:
                valve = self.wn.get_link(asset_id)
                diam = float(valve.diameter) if hasattr(valve, "diameter") and valve.diameter else 0.3
                return float(min(1.0, diam / 0.5))
            except Exception:
                return 0.6
        elif asset_type == "Junction":
            try:
                junc = self.wn.get_node(asset_id)
                base_dem = float(junc.base_demand) if hasattr(junc, "base_demand") else 0.001
                return float(min(1.0, base_dem * 1000.0))
            except Exception:
                return 0.3
        return 0.2

    def compute_monitoring_criticality(self, asset_id: str, asset_type: str) -> float:
        """Measures proximity and dependence of SCADA telemetry sensors on asset."""
        monitored_nodes = set(self.risk_cfg.get("simulation", {}).get("monitored_junctions", []))
        monitored_nodes.update(self.wn.tank_name_list)
        
        target_node = asset_id
        if asset_type in ("Pump", "Valve"):
            try:
                link = self.wn.get_link(asset_id)
                target_node = link.start_node_name
            except Exception:
                pass

        if target_node not in self.G:
            return 0.25

        # Count sensors within 3 graph hops
        lengths = nx.single_source_shortest_path_length(self.G, target_node, cutoff=3)
        sensors_near = sum(1 for n in lengths if n in monitored_nodes)
        return float(min(1.0, sensors_near / 5.0))

    def compute_recovery_complexity(self, asset_type: str) -> float:
        """Estimated time-to-repair hierarchy: Tanks > Pumps > Valves > Junctions."""
        recovery_map = {
            "Tank": 1.00,       # Major drainage, civil/welding repair, decontamination (48-72h)
            "Pump": 0.75,       # Mechanical motor rebuild, electrical overhaul (24-36h)
            "Valve": 0.50,      # Actuator / solenoid replacement, calibration (12-18h)
            "Junction": 0.25,   # Transducer tap replacement, local pipe clamp (4-6h)
        }
        return recovery_map.get(asset_type, 0.20)

    def evaluate_asset(self, asset_id: str, asset_type: str) -> Dict[str, float]:
        """Calculates 4 dimensional criticality scores and combined MDCI."""
        s = self.compute_structural_criticality(asset_id, asset_type)
        o = self.compute_operational_criticality(asset_id, asset_type)
        m = self.compute_monitoring_criticality(asset_id, asset_type)
        r = self.compute_recovery_complexity(asset_type)
        
        mdci = self.beta_1 * s + self.beta_2 * o + self.beta_3 * m + self.beta_4 * r
        return {
            "Asset_ID": asset_id,
            "Type": asset_type,
            "Structural_Criticality": s,
            "Operational_Criticality": o,
            "Monitoring_Criticality": m,
            "Recovery_Complexity": r,
            "MDCI": float(mdci)
        }

    def evaluate_dataframe(self, df_assets: pd.DataFrame) -> pd.DataFrame:
        """Evaluates MDCI across a dataframe of assets and normalizes structural score."""
        records = []
        for _, row in df_assets.iterrows():
            rec = self.evaluate_asset(row["Asset_ID"], row["Type"])
            records.append(rec)
            
        res_df = pd.DataFrame(records)
        max_s = res_df["Structural_Criticality"].max()
        if max_s > 0:
            res_df["Structural_Criticality"] = res_df["Structural_Criticality"] / max_s
        
        # Recompute MDCI with normalized structural criticality
        res_df["MDCI"] = (
            self.beta_1 * res_df["Structural_Criticality"] +
            self.beta_2 * res_df["Operational_Criticality"] +
            self.beta_3 * res_df["Monitoring_Criticality"] +
            self.beta_4 * res_df["Recovery_Complexity"]
        )
        return res_df

    @staticmethod
    def validate_mdci_spearman(mdci_series: pd.Series, iss_series: pd.Series) -> Tuple[float, float, Tuple[float, float]]:
        """Computes Spearman rank correlation and Fisher z-transform 95% confidence interval."""
        rho, p_val = stats.spearmanr(mdci_series, iss_series)
        n = len(mdci_series)
        
        # Fisher transformation for 95% CI
        z = np.arctanh(np.clip(rho, -0.999, 0.999))
        se = 1.0 / np.sqrt(max(1, n - 3))
        ci_low = np.tanh(z - 1.96 * se)
        ci_high = np.tanh(z + 1.96 * se)
        
        return float(rho), float(p_val), (float(ci_low), float(ci_high))
