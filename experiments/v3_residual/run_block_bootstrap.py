"""Moving-Block Bootstrap Sensitivity Analysis for Temporal Dependence.

Evaluates the stability and uncertainty of Pointwise F1 on the official BATADAL
test evaluation stream across varying block lengths L in {12, 24, 48} hours.
Preserves the frozen detector and single-shot evaluation protocol; reports
temporal autocorrelation sensitivity alongside the standard i.i.d. bootstrap.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCORES_PATH = ROOT / "results" / "v3" / "trust_pipeline_clean.csv"
OUT_JSON = ROOT / "results" / "v3" / "block_bootstrap_report.json"


def moving_block_bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_length: int,
    n_bootstraps: int = 2000,
    seed: int = 42
):
    """Performs circular moving-block bootstrap resampling to account for temporal dependence."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block_length))
    boot_f1 = []
    
    for _ in range(n_bootstraps):
        start_indices = rng.integers(0, n, size=nblocks)
        sampled_indices = []
        for s in start_indices:
            block_idx = (s + np.arange(block_length)) % n
            sampled_indices.extend(block_idx)
            
        sampled_indices = np.array(sampled_indices[:n])
        b_y = y_true[sampled_indices]
        b_p = y_pred[sampled_indices]
        
        p, r, f1, _ = precision_recall_fscore_support(b_y, b_p, average="binary", zero_division=0)
        boot_f1.append(float(f1))
        
    ci_lower = round(float(np.percentile(boot_f1, 2.5)), 4)
    ci_upper = round(float(np.percentile(boot_f1, 97.5)), 4)
    std_err = round(float(np.std(boot_f1)), 4)
    median_f1 = round(float(np.median(boot_f1)), 4)
    
    return {
        "block_length_hours": block_length,
        "n_blocks": nblocks,
        "n_bootstraps": n_bootstraps,
        "f1_median": median_f1,
        "f1_std_err": std_err,
        "ci_95": [ci_lower, ci_upper]
    }


def main():
    print(">> Running Temporal-Dependence Moving-Block Bootstrap Sensitivity Analysis...")
    df = pd.read_csv(SCORES_PATH)
    y_true = df["y_true"].values.astype(int)
    y_pred = df["alarm_persist"].values.astype(int)
    
    p, r, f1_pointwise, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    iid_res = moving_block_bootstrap_f1(y_true, y_pred, block_length=1, n_bootstraps=2000, seed=42)
    block_12 = moving_block_bootstrap_f1(y_true, y_pred, block_length=12, n_bootstraps=2000, seed=42)
    block_24 = moving_block_bootstrap_f1(y_true, y_pred, block_length=24, n_bootstraps=2000, seed=42)
    block_48 = moving_block_bootstrap_f1(y_true, y_pred, block_length=48, n_bootstraps=2000, seed=42)
    
    report = {
        "experiment": "temporal_dependence_moving_block_bootstrap",
        "detector": "v3_residual_ridge",
        "seed": 42,
        "total_evaluation_hours": len(y_true),
        "attack_hours": int(y_true.sum()),
        "pointwise_f1_nominal": round(float(f1_pointwise), 4),
        "pointwise_recall_nominal": round(float(r), 4),
        "pointwise_precision_nominal": round(float(p), 4),
        "iid_bootstrap_95ci": iid_res["ci_95"],
        "moving_block_bootstrap": {
            "12h_blocks": block_12,
            "24h_blocks": block_24,
            "48h_blocks": block_48
        },
        "methodological_safeguard": "Reported as sensitivity analysis alongside i.i.d. bootstrap; does not replace frozen test protocol."
    }
    
    try:
        from config.loader import get_config_hash
        report["config_sha256"] = get_config_hash()
    except Exception:
        pass

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Saved block bootstrap report to {OUT_JSON}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
