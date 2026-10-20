"""Threat-Prioritized Protection Investment (TPPI) 0/1 Knapsack Module (Phase B.7).

Formulates the municipal cyber-physical security investment allocation problem as a 0/1 Knapsack:
    Maximize   sum_{i=1}^N (TDCRI_i * Benefit_i * TS) * x_i
    Subject to sum_{i=1}^N Cost_i * x_i <= Budget_k
               x_i in {0, 1}

Solves using Pyomo with the HiGHS integer programming solver (highspy).
Includes an exact Dynamic Programming (DP) integer knapsack implementation for cross-validation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from config.loader import load_config

# Documented Unit Cost & Protection Benefit Table
COST_BENEFIT_TABLE = {
    "Pump": {"unit_cost_k": 120, "protection_benefit": 0.85, "description": "Secure PLC, authenticated actuator & physical enclave"},
    "Valve": {"unit_cost_k": 80, "protection_benefit": 0.80, "description": "Remote telemetry encryption & dual-redundant solenoid"},
    "Tank": {"unit_cost_k": 100, "protection_benefit": 0.90, "description": "Multi-sensor level probe & physical intrusion fencing"},
    "Junction": {"unit_cost_k": 40, "protection_benefit": 0.70, "description": "Pressure transducer cryptographic signing tap"}
}


class TPPISolver:
    """Solves TPPI 0/1 Knapsack Optimization using Pyomo/HiGHS with DP cross-validation."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.tppi_cfg = self.config.get("risk", {}).get("tppi", {})
        self.default_budget_k = float(self.tppi_cfg.get("budget_k", 500.0))

    def prepare_assets(self, df: pd.DataFrame, trust_score: float = 0.90) -> pd.DataFrame:
        """Enriches asset dataframe with unit costs, protection benefits, and protection values."""
        df_opt = df.copy()
        
        # Populate Unit Cost and Protection Benefit if missing
        if "Unit_Cost_k" not in df_opt.columns:
            df_opt["Unit_Cost_k"] = df_opt["Type"].map(lambda t: COST_BENEFIT_TABLE.get(t, {}).get("unit_cost_k", 40))
        if "Protection_Benefit" not in df_opt.columns:
            df_opt["Protection_Benefit"] = df_opt["Type"].map(lambda t: COST_BENEFIT_TABLE.get(t, {}).get("protection_benefit", 0.70))

        risk_col = "TDCRI" if "TDCRI" in df_opt.columns else ("Base_Risk" if "Base_Risk" in df_opt.columns else "ISS")
        df_opt["Protection_Value"] = df_opt[risk_col] * df_opt["Protection_Benefit"] * trust_score
        df_opt["TPPI_Priority_Density"] = df_opt["Protection_Value"] / df_opt["Unit_Cost_k"]
        
        # Sort deterministically: priority density descending, protection value descending, asset ID ascending
        df_opt = df_opt.sort_values(
            by=["TPPI_Priority_Density", "Protection_Value", "Asset_ID"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        return df_opt

    def find_tied_groups(self, df: pd.DataFrame, tol: float = 1e-4, trust_score: float = 0.90) -> List[Dict[str, Any]]:
        """
        Identifies and reports groups of assets with near-identical priority density scores.
        Used for reporting tie clusters (e.g. monitored junctions with identical unit cost and priority).
        """
        df_items = self.prepare_assets(df, trust_score=trust_score)
        densities = df_items["TPPI_Priority_Density"].values
        n = len(df_items)
        
        tied_groups = []
        visited = set()
        for i in range(n):
            if i in visited:
                continue
            group_idx = [i]
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if abs(densities[i] - densities[j]) <= tol and df_items.loc[i, "Unit_Cost_k"] == df_items.loc[j, "Unit_Cost_k"]:
                    group_idx.append(j)
            if len(group_idx) > 1:
                visited.update(group_idx)
                tied_sub = df_items.iloc[group_idx]
                tied_groups.append({
                    "type": tied_sub["Type"].iloc[0],
                    "asset_ids": tied_sub["Asset_ID"].tolist(),
                    "unit_cost_k": int(tied_sub["Unit_Cost_k"].iloc[0]),
                    "mean_priority_density": float(tied_sub["TPPI_Priority_Density"].mean()),
                    "count": len(group_idx)
                })
        return tied_groups

    def solve_pyomo(
        self,
        df: pd.DataFrame,
        budget_k: float = None,
        trust_score: float = 0.90,
        solver_name: str = "appsi_highs"
    ) -> Tuple[pd.DataFrame, float, float, str]:
        """
        Solves 0/1 Knapsack optimization using Pyomo with HiGHS solver.
        Uses deterministic secondary preference to break ties among identical assets.
        Returns: (selected_df, total_cost, total_benefit, solver_status)
        """
        budget = float(budget_k if budget_k is not None else self.default_budget_k)
        df_items = self.prepare_assets(df, trust_score=trust_score).reset_index(drop=True)
        
        model = pyo.ConcreteModel(name="TPPI_Knapsack")
        n = len(df_items)
        model.I = pyo.RangeSet(0, n - 1)
        model.x = pyo.Var(model.I, within=pyo.Binary)

        # Objective: Maximize sum_{i} (Protection_Value[i] + eps * (n - i)) * x[i]
        # Infinitesimal perturbation eps = 1e-8 preserves exact values while guaranteeing deterministic tie-breaking
        def obj_rule(m):
            return sum((float(df_items.loc[i, "Protection_Value"]) + 1e-8 * (n - i)) * m.x[i] for i in m.I)
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

        # Budget Constraint: sum_{i} Unit_Cost_k[i] * x[i] <= budget
        def budget_rule(m):
            return sum(float(df_items.loc[i, "Unit_Cost_k"]) * m.x[i] for i in m.I) <= budget
        model.budget_con = pyo.Constraint(rule=budget_rule)

        # Solve with HiGHS
        solver = None
        for s_candidate in [solver_name, "appsi_highs", "highs"]:
            try:
                s = pyo.SolverFactory(s_candidate)
                if s.available():
                    solver = s
                    break
            except Exception:
                continue

        if solver is None:
            raise RuntimeError("HiGHS solver not available in Pyomo environment.")

        res = solver.solve(model)
        status_str = str(res.solver.status)

        selected_indices = [i for i in model.I if pyo.value(model.x[i]) > 0.5]
        selected_df = df_items.iloc[selected_indices].copy()
        
        total_cost = float(selected_df["Unit_Cost_k"].sum())
        total_value = float(selected_df["Protection_Value"].sum())
        
        return selected_df, total_cost, total_value, status_str

    def solve_multi_budget(
        self,
        df: pd.DataFrame,
        budgets: List[int] = None,
        trust_score: float = 0.90
    ) -> Dict[int, Dict[str, Any]]:
        """
        Solves TPPI 0/1 Knapsack across multiple municipal budget tiers (e.g., $200k, $500k, $1,000k).
        Returns a dictionary mapping budget_k -> outcome details.
        """
        if budgets is None:
            budgets = [200, 500, 1000]
        multi_results = {}
        for b in budgets:
            sel_df, cost, val, status = self.solve_pyomo(df, budget_k=b, trust_score=trust_score)
            multi_results[b] = {
                "budget_k": b,
                "spent_k": int(cost),
                "total_value": float(val),
                "asset_count": len(sel_df),
                "selected_assets": sel_df["Asset_ID"].tolist(),
                "selected_df": sel_df,
                "solver_status": status
            }
        return multi_results

    def solve_dynamic_programming(
        self,
        df: pd.DataFrame,
        budget_k: float = None,
        trust_score: float = 0.90
    ) -> Tuple[pd.DataFrame, int, float]:
        """
        Solves integer 0/1 knapsack using exact Dynamic Programming table.
        Used for mathematical cross-validation against Pyomo/HiGHS.
        """
        budget = int(budget_k if budget_k is not None else self.default_budget_k)
        df_items = self.prepare_assets(df, trust_score=trust_score).reset_index(drop=True)
        
        costs = df_items["Unit_Cost_k"].values.astype(int)
        # Scale values by 10,000 for precision integer DP
        values = (df_items["Protection_Value"].values * 10000).astype(int)
        n = len(df_items)

        K = np.zeros((n + 1, budget + 1), dtype=int)
        for i in range(1, n + 1):
            w = costs[i - 1]
            v = values[i - 1]
            for b in range(budget + 1):
                if w <= b:
                    K[i, b] = max(v + K[i - 1, b - w], K[i - 1, b])
                else:
                    K[i, b] = K[i - 1, b]

        # Backtrack solution
        res_budget = budget
        selected_idx = []
        for i in range(n, 0, -1):
            if K[i, res_budget] != K[i - 1, res_budget]:
                selected_idx.append(i - 1)
                res_budget -= costs[i - 1]

        selected_df = df_items.iloc[selected_idx].copy()
        total_cost = int(selected_df["Unit_Cost_k"].sum())
        total_value = float(selected_df["Protection_Value"].sum())
        return selected_df, total_cost, total_value
