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


events = []
for i, (es, ee) in enumerate(runs(y == 1), 1):
    seg = primary[es:ee + 1]
    events.append(dict(event=i, start=es, end=ee, hours=ee - es + 1, hours_detected=int(seg.sum()),
                       recall=round(float(seg.mean()), 3), delay_h=int(np.argmax(seg)) if seg.any() else None))
fp_mask = (primary == 1) & (y == 0)
prev = float(y.mean())
ceiling = max(f1_score(y, (s > t).astype(int)) for t in np.percentile(s, np.arange(50, 99.9, 0.5)))
res = dict(config=cfg["name"], rows=len(y), attack_hours=int(y.sum()), prevalence=round(prev, 4),
           threshold=round(thr, 4), primary=summarize(primary), raw_no_persistence=summarize(raw),
           roc_auc=round(roc_auc_score(y, s), 4), pr_auc=round(average_precision_score(y, s), 4),
           always_alarm_F1=round(2 * prev / (1 + prev), 4), false_alarm_events=len(runs(fp_mask)),
           per_event=events,
           DIAGNOSTIC_ONLY_oracle_threshold_F1=round(float(ceiling), 4),
           note="Pointwise metrics, no point-adjustment. Oracle F1 picks a threshold using test labels: a ceiling, not a result.")
report.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
