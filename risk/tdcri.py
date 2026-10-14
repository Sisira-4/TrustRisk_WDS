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
        top_k: int = 10,
        n_trials: int = 50,
        perturbation_range: float = 0.20,
        seed: int = 42
    ) -> Tuple[float, float, List[float]]:
        """
        Tests ranking stability under +/-20% perturbations on all weights.
        Returns (mean_tau, min_tau, list_of_taus).
        """
        rng = np.random.RandomState(seed)
        base_scores = self.compute_tdcri(df)
        base_ranking = list(df.assign(score=base_scores).sort_values(by="score", ascending=False)["Asset_ID"].head(top_k))
        
        base_weights = np.array([
            self.w_mdci, self.w_ass, self.w_iss,
            self.w_cri, self.w_ars, self.w_ts
        ])
        
        taus = []
        for _ in range(n_trials):
            # Perturb each weight independently by [-perturbation_range, +perturbation_range]
            delta = rng.uniform(-perturbation_range, perturbation_range, size=len(base_weights))
            pert_w = base_weights * (1.0 + delta)
            pert_w = pert_w / pert_w.sum() # renormalize
            
            ars_val = df["ARS"] if "ARS" in df.columns else 0.50
            ts_val = df["TS"] if "TS" in df.columns else 0.90
            
            pert_score = (
                pert_w[0] * df["MDCI"] +
                pert_w[1] * df["ASS"] +
                pert_w[2] * df["ISS"] +
                pert_w[3] * df["CRI"] +
                pert_w[4] * (1.0 - ars_val) +
                pert_w[5] * ts_val
            )
            
            pert_ranking = list(df.assign(score=pert_score).sort_values(by="score", ascending=False)["Asset_ID"].head(top_k))
            
            # Rank positions for Kendall tau
            rank_base = [base_ranking.index(x) if x in base_ranking else top_k + 5 for x in base_ranking]
            rank_pert = [pert_ranking.index(x) if x in pert_ranking else top_k + 5 for x in base_ranking]
            
            tau, _ = stats.kendalltau(rank_base, rank_pert)
            if not np.isnan(tau):
                taus.append(float(tau))
            else:
                taus.append(1.0)
                
        mean_tau = float(np.mean(taus))
        min_tau = float(np.min(taus))
        return mean_tau, min_tau, taus
