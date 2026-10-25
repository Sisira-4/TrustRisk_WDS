# Frozen Residual Detector: Single-Shot Test Evaluation Report

**Model Configuration**: `v3_residual_ridge`  
**Threshold Source**: Training margin-free pool (q=99.0, threshold = 7.4532)  
**Evaluation Protocol**: Single-shot evaluation on official BATADAL test set (`BATADAL_test_dataset_labeled.csv`).

---

## 1. Primary Test Metrics (Pointwise & Point-Adjusted)

| Metric | Pointwise (Primary) | Point-Adjusted (PA) | Raw (No Persistence) | Raw PA |
| :--- | :---: | :---: | :---: | :---: |
| **F1 Score** | **0.8368** (95% CI: [0.8083, 0.8659]) | **0.9879** | 0.8115 | 0.9927 |
| **Precision** | 0.9677 | 0.976 | 0.9792 | 0.9855 |
| **Recall** | 0.7371 | 1.0 | 0.6929 | 1.0 |
| **True Positive Hours** | 300 / 407 | 407 / 407 | 282 / 407 | 407 / 407 |
| **False Positive Hours** | 10 / 1682 | 10 / 1682 | 6 / 1682 | 6 / 1682 |

- **ROC-AUC**: 0.918
- **PR-AUC**: 0.8684
- **Attack Events Detected**: **7/7**
- **False Alarm Events**: **3**

---

## 2. Per-Attack Timeline Breakdown (Events 1 to 7)

| Event | Start Index | End Index | Duration (h) | Detected? | Delay (h) | Hours Detected | Event Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 297 | 366 | 70 | YES | 7 | 44 | 0.629 |
| 2 | 632 | 696 | 65 | YES | 11 | 20 | 0.308 |
| 3 | 867 | 897 | 31 | YES | 2 | 29 | 0.935 |
| 4 | 937 | 967 | 31 | YES | 2 | 29 | 0.935 |
| 5 | 1229 | 1328 | 100 | YES | 1 | 98 | 0.980 |
| 6 | 1574 | 1653 | 80 | YES | 1 | 71 | 0.887 |
| 7 | 1940 | 1969 | 30 | YES | 10 | 9 | 0.300 |

---

## 3. Negative Findings & Evasion Analysis
- Stealthy or low-magnitude attack behavior (e.g. short duration or deceptive hydraulic actions) are documented above.
- Diagnostic oracle threshold ceiling (test-swept, not achievable in practice): F1 = 0.8377.
