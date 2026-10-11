"""ONE-SHOT evaluation on the 2017 labelled test set with the settings frozen in config_v3.json.

- Refuses to run without --confirm-test-use.
- Refuses to run again if results/v3/test_report.json already exists.
- Fits ONLY on dataset04 rows; the threshold is a percentile of TRAINING scores.
Nothing here may be re-tuned after looking at the numbers.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, HERE, load_series, sensor_columns, runs, margin_free_pool, persist,
                    ResidualDetector, confusion, prf)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT))
ap.add_argument("--config", default=str(HERE / "config_v3.json"))
ap.add_argument("--confirm-test-use", action="store_true")
a = ap.parse_args()
if not a.confirm_test_use:
    sys.exit("Refusing: pass --confirm-test-use to acknowledge this uses the test set once.")
root = Path(a.root)
out = root / "results" / "v3"; out.mkdir(parents=True, exist_ok=True)
report = out / "test_report.json"
if report.exists():
    sys.exit(f"Refusing: {report} already exists. The test set has been used; do not re-run or retune.")
cfg = json.loads(Path(a.config).read_text())

d4 = load_series(root / cfg["train"]["source"])
names = sensor_columns(d4)
X4, lab4 = d4[names].values.astype(float), d4["ATT_FLAG"].values
pool = margin_free_pool(lab4, cfg["train"]["margin_hours"])
det = ResidualDetector(cfg["model"]["alpha"], cfg["model"]["n_top"]).fit(X4, pool, names)
thr = det.threshold(cfg["threshold"]["quantile"])

te = load_series(root / cfg["test_file"])
y = (te["ATT_FLAG"].values == 1).astype(int)
_, s = det.score(te[names].values.astype(float))
raw = (s > thr).astype(int)
per = persist(raw, cfg["persistence"]["k"], cfg["persistence"]["n"])
primary = per if cfg["persistence"]["enabled"] else raw


def summarize(p):
    tp, fn, fp, tn = confusion(y, p)
    P, R, F = prf(tp, fn, fp, tn)
    return dict(F1=round(F, 4), precision=round(P, 4), recall=round(R, 4), TP=tp, FN=fn, FP=fp, TN=tn)


def point_adjust(labels, preds):
    adjusted = preds.copy()
    for s, e in runs(labels == 1):
        if np.any(preds[s:e+1] == 1):
            adjusted[s:e+1] = 1
    return adjusted


# Point-adjusted summary
pa_primary = point_adjust(y, primary)
pa_raw = point_adjust(y, raw)

# Bootstrap 95% confidence intervals on pointwise primary
rng = np.random.default_rng(42)
n_boot = 1000
boot_f1 = []
n_pts = len(y)
for _ in range(n_boot):
    b_idx = rng.integers(0, n_pts, n_pts)
    b_tp, b_fn, b_fp, b_tn = confusion(y[b_idx], primary[b_idx])
    _, _, b_f = prf(b_tp, b_fn, b_fp, b_tn)
    boot_f1.append(b_f)
boot_ci = [round(float(np.percentile(boot_f1, 2.5)), 4), round(float(np.percentile(boot_f1, 97.5)), 4)]

events = []
detected_events_count = 0
for i, (es, ee) in enumerate(runs(y == 1), 1):
    seg = primary[es:ee + 1]
    is_det = bool(seg.any())
    if is_det:
        detected_events_count += 1
    events.append(dict(event=i, start=es, end=ee, hours=ee - es + 1, hours_detected=int(seg.sum()),
                       detected=is_det, recall=round(float(seg.mean()), 3),
                       delay_h=int(np.argmax(seg)) if is_det else None))

fp_mask = (primary == 1) & (y == 0)
prev = float(y.mean())
ceiling = max(f1_score(y, (s > t).astype(int)) for t in np.percentile(s, np.arange(50, 99.9, 0.5)))

res = dict(
    config=cfg["name"],
    rows=len(y),
    attack_hours=int(y.sum()),
    prevalence=round(prev, 4),
    threshold=round(thr, 4),
    pointwise_primary=summarize(primary),
    point_adjusted_primary=summarize(pa_primary),
    pointwise_raw=summarize(raw),
    point_adjusted_raw=summarize(pa_raw),
    bootstrap_f1_95ci=boot_ci,
    roc_auc=round(roc_auc_score(y, s), 4),
    pr_auc=round(average_precision_score(y, s), 4),
    events_detected=f"{detected_events_count}/7",
    false_alarm_events=len(runs(fp_mask)),
    always_alarm_F1=round(2 * prev / (1 + prev), 4),
    per_event=events,
    DIAGNOSTIC_ONLY_oracle_threshold_F1=round(float(ceiling), 4),
    note="Frozen single-shot evaluation on official BATADAL test set. Pointwise and PA metrics reported side-by-side."
)

report.write_text(json.dumps(res, indent=2))

# Generate Markdown Report
md_report = f"""# Frozen Residual Detector: Single-Shot Test Evaluation Report

**Model Configuration**: `{cfg['name']}`  
**Threshold Source**: Training margin-free pool (q={cfg['threshold']['quantile']}, threshold = {thr:.4f})  
**Evaluation Protocol**: Single-shot evaluation on official BATADAL test set (`BATADAL_test_dataset_labeled.csv`).

---

## 1. Primary Test Metrics (Pointwise & Point-Adjusted)

| Metric | Pointwise (Primary) | Point-Adjusted (PA) | Raw (No Persistence) | Raw PA |
| :--- | :---: | :---: | :---: | :---: |
| **F1 Score** | **{res['pointwise_primary']['F1']}** (95% CI: [{boot_ci[0]}, {boot_ci[1]}]) | **{res['point_adjusted_primary']['F1']}** | {res['pointwise_raw']['F1']} | {res['point_adjusted_raw']['F1']} |
| **Precision** | {res['pointwise_primary']['precision']} | {res['point_adjusted_primary']['precision']} | {res['pointwise_raw']['precision']} | {res['point_adjusted_raw']['precision']} |
| **Recall** | {res['pointwise_primary']['recall']} | {res['point_adjusted_primary']['recall']} | {res['pointwise_raw']['recall']} | {res['point_adjusted_raw']['recall']} |
| **True Positive Hours** | {res['pointwise_primary']['TP']} / 407 | {res['point_adjusted_primary']['TP']} / 407 | {res['pointwise_raw']['TP']} / 407 | {res['point_adjusted_raw']['TP']} / 407 |
| **False Positive Hours** | {res['pointwise_primary']['FP']} / 1682 | {res['point_adjusted_primary']['FP']} / 1682 | {res['pointwise_raw']['FP']} / 1682 | {res['point_adjusted_raw']['FP']} / 1682 |

- **ROC-AUC**: {res['roc_auc']}
- **PR-AUC**: {res['pr_auc']}
- **Attack Events Detected**: **{res['events_detected']}**
- **False Alarm Events**: **{res['false_alarm_events']}**

---

## 2. Per-Attack Timeline Breakdown (Events 1 to 7)

| Event | Start Index | End Index | Duration (h) | Detected? | Delay (h) | Hours Detected | Event Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
for ev in events:
    det_str = "YES" if ev['detected'] else "**NO**"
    delay_str = str(ev['delay_h']) if ev['delay_h'] is not None else "-"
    md_report += f"| {ev['event']} | {ev['start']} | {ev['end']} | {ev['hours']} | {det_str} | {delay_str} | {ev['hours_detected']} | {ev['recall']:.3f} |\n"

md_report += f"""
---

## 3. Negative Findings & Evasion Analysis
- Stealthy or low-magnitude attack behavior (e.g. short duration or deceptive hydraulic actions) are documented above.
- Diagnostic oracle threshold ceiling (test-swept, not achievable in practice): F1 = {res['DIAGNOSTIC_ONLY_oracle_threshold_F1']}.
"""

md_out = root / "results" / "frozen_detector_test_report.md"
md_out.write_text(md_report, encoding="utf-8")

print(json.dumps(res, indent=2))
print(f"\nWritten markdown report to {md_out}")

