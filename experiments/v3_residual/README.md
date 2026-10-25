# V3 residual detector (branch: feature/residual-detector)

Per-sensor ridge regressors predict each SCADA signal from the other sensors and the previous hour,
trained only on normal-ish rows. The anomaly score is the mean of the 3 largest residual z-scores.
Pointwise metrics only (no point-adjustment).

| File | Purpose |
|---|---|
| `common.py` | detector, causal persistence, pool selection, metrics. Never references the test file |
| `config_v3.json` | frozen settings (DS04-only training, ridge alpha 1, q99 threshold, causal 3-of-6 persistence) |
| `run_cv.py` | leave-one-attack-event-out CV on dataset04 -> `results/v3/cv_report.{csv,json,md}` |
| `run_test_once.py` | ONE-SHOT test evaluation. Needs `--confirm-test-use`; refuses to run twice |
| `export_scores.py` | per-hour score + per-sensor z-scores for any CSV, for the risk/trust layers |
| `../../tests/test_v3_guard.py` | guards: only `run_test_once.py` may read the test file; persistence is causal; thresholds use training scores only |

```bash
pip install numpy pandas scikit-learn pytest tabulate
python -m pytest tests/test_v3_guard.py -q
python experiments/v3_residual/run_cv.py
python experiments/v3_residual/export_scores.py --input data/raw/BATADAL_dataset04.csv --output results/v3/scores_ds04.csv
# only when you decide to spend the test set (once):
python experiments/v3_residual/run_test_once.py --confirm-test-use
```

## Protocol facts
* dataset04 labels are only `1` or `-999`. `-999` means UNLABELLED, not normal. Training rows are -999 rows more than 48 h from any labelled attack. False-alarm rates measured on -999 rows are therefore upper bounds.
* DS03 (2014) differs systematically from DS04 (2016) and the 2017 test period, so it is not used for training.
* Thresholds are percentiles of TRAINING scores. Persistence is causal (trailing window).

## Reference numbers (reproduced by the package)
* CV, q99 + persistence: TPR 0.74, FPR 0.023/h, projected F1 0.81 at 19.5% prevalence (10 configurations compared, so optimistic).
* One-shot test, frozen config: F1 0.843, precision 0.968, recall 0.747, ROC-AUC 0.917 (always-alarm F1 = 0.326). Strong on 4 of 7 attacks (>= 93% of hours), weak on attacks 1, 2 and 7.
