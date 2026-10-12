"""Fit on dataset04 (margin-free -999 rows) and export per-hour scores and per-sensor residual z-scores
for ANY input CSV (BATADAL layout). Downstream risk/trust layers read this file. Usage:
    python experiments/v3_residual/export_scores.py --input <csv> --output results/v3/scores.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, HERE, load_series, sensor_columns, margin_free_pool, persist, ResidualDetector

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT)); ap.add_argument("--config", default=str(HERE / "config_v3.json"))
ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
a = ap.parse_args()
cfg = json.loads(Path(a.config).read_text()); root = Path(a.root)
d4 = load_series(root / cfg["train"]["source"]); names = sensor_columns(d4)
pool = margin_free_pool(d4["ATT_FLAG"].values, cfg["train"]["margin_hours"])
det = ResidualDetector(cfg["model"]["alpha"], cfg["model"]["n_top"]).fit(d4[names].values.astype(float), pool, names)
thr = det.threshold(cfg["threshold"]["quantile"])
d = load_series(a.input)
A, s = det.score(d[names].values.astype(float))
raw = (s > thr).astype(int)
out = pd.DataFrame({"DATETIME": d.get("DATETIME", pd.RangeIndex(len(d))), "score": s, "threshold": thr,
                    "alarm_raw": raw, "alarm_persist": persist(raw, cfg["persistence"]["k"], cfg["persistence"]["n"])})
top = np.argsort(-A, axis=1)[:, :3]
for r in range(3):
    out[f"top{r+1}_sensor"] = [det.names[j] for j in top[:, r]]
for j, n in enumerate(det.names):
    out[f"z_{n}"] = A[:, j]
Path(a.output).parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.output, index=False)
print(f"wrote {a.output}: {len(out)} rows, alarms(persisted)={int(out.alarm_persist.sum())}")
