"""Leave-one-attack-event-out cross-validation on dataset04 (test set NOT touched).

Fold k holds out labelled attack k (+/-48h margin) for recall and one contiguous block of
margin-free -999 rows (+/-24h gap) for false alarms. The detector and every threshold are fitted
on the remaining rows only. Writes results/v3/cv_report.{csv,json,md}.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, load_series, sensor_columns, runs, margin_free_pool, persist,
                    ResidualDetector, confusion, prf, projected_f1)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT))
ap.add_argument("--out", default=None)
ap.add_argument("--quantiles", type=float, nargs="+", default=[90, 95, 97, 99, 99.5])
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--prevalence", type=float, default=0.195)
a = ap.parse_args()
root = Path(a.root)
out = Path(a.out) if a.out else root / "results" / "v3"
out.mkdir(parents=True, exist_ok=True)

d = load_series(root / "data" / "raw" / "BATADAL_dataset04.csv")
names = sensor_columns(d)
X, lab = d[names].values.astype(float), d["ATT_FLAG"].values
T = len(d)
events = runs(lab == 1)
pool = margin_free_pool(lab, 48)
blocks = np.array_split(pool, len(events))
assert len(events) == len(blocks)
print(f"events={events}\npool rows={len(pool)} blocks={[len(b) for b in blocks]}")

cfgs = [(q, pers) for q in a.quantiles for pers in (False, True)]
acc = {c: dict(TP=0, FN=0, FP=0, TN=0, fa_events=0, hours=0, delays=[], cover=[]) for c in cfgs}
idx = np.arange(T)
for k, (es, ee) in enumerate(events):
    vb = blocks[k]
    gap = (idx >= vb.min() - 24) & (idx <= vb.max() + 24)
    held = np.zeros(T, bool); held[max(0, es - 48): ee + 49] = True
    train_idx = np.array([i for i in pool if not gap[i] and not held[i]])
    # sanity: nothing evaluated is in training
    assert not set(train_idx) & set(vb) and not set(train_idx) & set(range(es, ee + 1))
    det = ResidualDetector(a.alpha).fit(X, train_idx, names)
    _, s = det.score(X)
    for (q, pers) in cfgs:
        al = (s > det.threshold(q)).astype(int)
        if pers:
            al = persist(al, 3, 6)
        ev, nb = al[es:ee + 1], al[vb]
        tp, fn, _, _ = confusion(np.ones(len(ev)), ev)
        _, _, fp, tn = confusion(np.zeros(len(nb)), nb)
        r = acc[(q, pers)]
        r["TP"] += tp; r["FN"] += fn; r["FP"] += fp; r["TN"] += tn
        r["fa_events"] += len(runs(nb == 1)); r["hours"] += len(nb)
        r["delays"].append(int(np.argmax(ev)) if ev.any() else None)
        r["cover"].append(round(float(ev.mean()), 3))
    print(f"fold {k}: train rows={len(train_idx)}", flush=True)

rows = []
for (q, pers), r in acc.items():
    tp, fn, fp, tn = r["TP"], r["FN"], r["FP"], r["TN"]
    P, R, F = prf(tp, fn, fp, tn)
    fpr = fp / (fp + tn)
    rows.append(dict(quantile=q, persistence_3of6=pers, TP=tp, FN=fn, FP=fp, TN=tn,
                     TPR=round(R, 3), FPR_per_hr=round(fpr, 4),
                     FA_events_per_week=round(r["fa_events"] / (r["hours"] / 168), 2),
                     precision_cv=round(P, 3), F1_pointwise_cv=round(F, 3),
                     projected_F1_at_prevalence=round(projected_f1(R, fpr, a.prevalence), 3),
                     per_event_delay_h=r["delays"], per_event_coverage=r["cover"]))
df = pd.DataFrame(rows)
df.to_csv(out / "cv_report.csv", index=False)
(out / "cv_report.json").write_text(json.dumps(rows, indent=2))
note = (f"configurations compared: {len(cfgs)} (selecting the best row of this table is optimistic).\n"
        "FPR is measured on -999 rows, which are UNLABELLED, so it is an upper bound if they hide attacks.\n"
        "Thresholds come from each fold's training scores. Pointwise metrics only; no point-adjustment.\n")
(out / "cv_report.md").write_text(df.drop(columns=["per_event_coverage"]).to_markdown(index=False) + "\n\n" + note
                                  if hasattr(df, "to_markdown") else df.to_string() + "\n\n" + note)
print(df.drop(columns=["per_event_coverage"]).to_string(index=False)); print(note)
