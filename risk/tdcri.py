"""Threat-Driven Cyber-Physical Risk Index (TDCRI) Module (Phase B.6).

Implements the canonical proposal formula:
TDCRI = w_mdci * MDCI + w_ass * ASS + w_iss * ISS + w_cri * CRI + w_ars * (1 - ARS) + w_ts * TS

Default weights (from config/trustrisk.yaml):
- w_mdci: 0.25 (Multi-Dimensional Criticality Index)
- w_ass:  0.20 (Attack Susceptibility Score)
- w_iss:  0.20 (Hydraulic Impact Severity Score)
- w_cri:  0.15 (Cascading Reach Index)
- w_ars:  0.10 (Asset Resilience penalty: 1 - ARS)
- w_ts:   0.10 (Operational SCADA Trust Score)

Also implements +/-20% weight perturbation sensitivity testing with Kendall's tau.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import scipy.stats as stats

from config.loader import load_config


class TDCRICalculator:
    """Calculates unified TDCRI and validates ranking sensitivity."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.w_cfg = self.config.get("risk", {}).get("tdcri_weights", {})
        
        self.w_mdci = float(self.w_cfg.get("w_mdci", 0.25))
        self.w_ass = float(self.w_cfg.get("w_ass", 0.20))
        self.w_iss = float(self.w_cfg.get("w_iss", 0.20))
        self.w_cri = float(self.w_cfg.get("w_cri", 0.15))
        self.w_ars = float(self.w_cfg.get("w_ars", 0.10))
        self.w_ts = float(self.w_cfg.get("w_ts", 0.10))

    def compute_tdcri(self, df: pd.DataFrame, trust_score: float = 0.90) -> pd.Series:
        """
        Computes TDCRI for each asset row using the canonical formula.
        If TS column is present in df, it uses it; otherwise uses trust_score argument.
        """
        ts_val = df["TS"] if "TS" in df.columns else trust_score
        ars_val = df["ARS"] if "ARS" in df.columns else 0.50
        
        tdcri = (
            self.w_mdci * df["MDCI"] +
            self.w_ass * df["ASS"] +
            self.w_iss * df["ISS"] +
            self.w_cri * df["CRI"] +
            self.w_ars * (1.0 - ars_val) +
            self.w_ts * ts_val
        )
        return tdcri

    def test_ranking_sensitivity_kendall(
        self,
        df: pd.DataFrame,
        top_k: int = 5,
        n_trials: int = 1000,
        perturbation_range: float = 0.20,
        seed: int = 42
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Tests ranking stability under random perturbations on all weights.
        Draws n_trials random weight vectors within +/- perturbation_range of nominal,
        re-normalises to sum to 1.0, recomputes the full ranking over all assets,
        and computes Kendall's tau between the nominal ranking and perturbed ranking.
        Also tracks top-k retention frequency for each asset.
        
        Returns:
            (mean_tau, min_tau, details_dict)
        """
        rng = np.random.RandomState(seed)
        base_scores = self.compute_tdcri(df)
        base_ranking = stats.rankdata(-base_scores.values, method="average")
        
        # Nominal top-k assets
        sorted_indices = np.argsort(-base_scores.values)
        nominal_top_k = df.iloc[sorted_indices[:top_k]]["Asset_ID"].tolist()
        
        base_weights = np.array([
            self.w_mdci, self.w_ass, self.w_iss,
            self.w_cri, self.w_ars, self.w_ts
        ])
        
        taus = []
        top_k_counts = {asset_id: 0 for asset_id in df["Asset_ID"]}
        
        ars_val = df["ARS"].values if "ARS" in df.columns else np.full(len(df), 0.50)
        ts_val = df["TS"].values if "TS" in df.columns else np.full(len(df), 0.90)
        mdci_val = df["MDCI"].values
        ass_val = df["ASS"].values
        iss_val = df["ISS"].values
        cri_val = df["CRI"].values
        
        for _ in range(n_trials):
            # Perturb each weight independently by [-perturbation_range, +perturbation_range]
            delta = rng.uniform(-perturbation_range, perturbation_range, size=len(base_weights))
            pert_w = base_weights * (1.0 + delta)
            pert_w = pert_w / pert_w.sum()  # renormalize to sum to 1.0
            
            pert_score = (
                pert_w[0] * mdci_val +
                pert_w[1] * ass_val +
                pert_w[2] * iss_val +
                pert_w[3] * cri_val +
                pert_w[4] * (1.0 - ars_val) +
                pert_w[5] * ts_val
            )
            
            pert_ranking = stats.rankdata(-pert_score, method="average")
            tau, _ = stats.kendalltau(base_ranking, pert_ranking)
            taus.append(float(tau) if not np.isnan(tau) else 1.0)
            
            # Record top-k occurrences
            pert_sorted = np.argsort(-pert_score)
            for idx in pert_sorted[:top_k]:
                top_k_counts[df.iloc[idx]["Asset_ID"]] += 1
                
        taus_arr = np.array(taus)
        mean_tau = float(np.mean(taus_arr))
        std_tau = float(np.std(taus_arr))
        p5_tau = float(np.percentile(taus_arr, 5))
        p95_tau = float(np.percentile(taus_arr, 95))
        min_tau = float(np.min(taus_arr))
        max_tau = float(np.max(taus_arr))
        
        top_k_freq = {k: v / float(n_trials) for k, v in top_k_counts.items() if v > 0}
        
        stats_dict = {
            "mean_tau": mean_tau,
            "std_tau": std_tau,
            "p5_tau": p5_tau,
            "p95_tau": p95_tau,
            "min_tau": min_tau,
            "max_tau": max_tau,
            "taus": taus,
            "n_trials": n_trials,
            "top_k": top_k,
            "nominal_top_k": nominal_top_k,
            "top_k_frequencies": top_k_freq
        }
        return mean_tau, min_tau, stats_dict
