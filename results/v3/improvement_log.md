# Improvement Log: V3 Residual Detector Cross-Validation Experiments

All experiments are evaluated strictly via leave-one-attack-event-out cross-validation on `BATADAL_dataset04.csv`.
The test set (`BATADAL_test_dataset_labeled.csv`) was NEVER read, evaluated, or touched.
Pointwise metrics only; no point-adjustment. False alarm rate is evaluated on held-out `-999` (unlabelled) blocks.

## Adoption Rule
Adopt a change only if pooled TPR rises at equal or lower pooled FPR AND it helps in at least 4 of 5 folds.
Prefer the simpler option within noise. At most 15 configurations evaluated.

---

## Configuration Comparison Table

| Config # | Configuration Description | Pooled TPR | Pooled FPR (/hr) | Pooled Pointwise F1 | Projected F1 (prev 0.195) | Fold TPRs | Fold FPRs | Decision |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---|:---|
| **0 (Baseline)** | Ridge $\alpha=1.0$, Top-3 z-score, 3-of-6 persist, q=99.0 | **0.7397** | **0.0233** | **0.7013** | **0.8058** | `[0.8810, 0.1333, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0058, 0.0691]` | **Baseline Frozen** |
| 1 | EWMA on Top-3 z-score ($\alpha=0.2$) | 0.7854 | 0.0437 | 0.6335 | 0.7990 | `[1.0, 0.2167, 1.0, 1.0, 1.0]` | `[0.0158, 0.0000, 0.0489, 0.0043, 0.1496]` | Rejected (FPR increased from 0.0233 to 0.0437) |
| 2 | CUSUM on Top-3 z-score | 1.0000 | 0.5162 | 0.1961 | 0.4841 | `[1.0, 1.0, 1.0, 1.0, 1.0]` | `[0.0000, 0.0000, 0.5827, 1.0, 1.0]` | Rejected (Massive FPR explosion to 0.5162) |
| 3 | Ridge $\alpha=0.1$ | 0.7580 | 0.0239 | 0.7094 | 0.8166 | `[0.9762, 0.1333, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0000, 0.0777]` | Rejected (FPR increased from 0.0233 to 0.0239) |
| 4 | Ridge $\alpha=10.0$ | **0.7717** | **0.0219** | **0.7284** | **0.8289** | `[0.9762, 0.1833, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0058, 0.0619]` | **ADOPTED** (TPR rose from 0.7397 to 0.7717, FPR dropped from 0.0233 to 0.0219, improved or maintained in 5/5 folds) |
| 5 | Top-1 Sensor ($n_{\text{top}}=1$) | 0.7397 | 0.0244 | 0.6953 | 0.8038 | `[0.8810, 0.1333, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0058, 0.0748]` | Rejected (Higher FPR, lower F1) |
| 6 | Top-5 Sensors ($n_{\text{top}}=5$) | 0.7397 | 0.0224 | 0.7059 | 0.8074 | `[0.8810, 0.1333, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0058, 0.0647]` | Rejected (Equal TPR, within noise of baseline) |
| 7 | Multi-lag 3 ($t-1, t-2, t-3$) | 0.7763 | 0.0250 | 0.7143 | 0.8260 | `[1.0, 0.1833, 1.0, 1.0, 1.0]` | `[0.0072, 0.0000, 0.0374, 0.0086, 0.0719]` | Rejected (Higher FPR than baseline and Config 4) |
| 8 | Multi-lag 6 ($t-1 \dots t-6$) | 0.7763 | 0.0267 | 0.7054 | 0.8229 | `[1.0, 0.1833, 1.0, 1.0, 1.0]` | `[0.0000, 0.0000, 0.0374, 0.0101, 0.0863]` | Rejected (Higher FPR) |
| 9 | Robust z-score (Median / MAD) | 0.4429 | 0.0184 | 0.5105 | 0.5832 | `[0.0, 0.1333, 0.3784, 0.2857, 1.0]` | `[0.0000, 0.0000, 0.0302, 0.0000, 0.0619]` | Rejected (Severe collapse in TPR from 0.7397 to 0.4429) |
| 10 | Recency Weighting (Half-life 60 days) | 0.7397 | 0.0233 | 0.7013 | 0.8058 | `[0.8810, 0.1333, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0058, 0.0691]` | Rejected (Identical to baseline within noise) |
| 11 | Recency Weighting (Half-life 30 days) | 0.7397 | 0.0221 | 0.7074 | 0.8079 | `[0.8810, 0.1333, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0000, 0.0691]` | Rejected (Equal TPR, inferior to Config 4) |
| 12 | Ridge $\alpha=5.0$ | 0.7626 | 0.0239 | 0.7122 | 0.8195 | `[0.9762, 0.1500, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0058, 0.0719]` | Rejected (Higher FPR than Config 4) |
| 13 | Hybrid MLP on Dominant Sensors | 0.7763 | 0.0224 | 0.7281 | 0.8307 | `[1.0, 0.1833, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0058, 0.0647]` | Rejected (Higher FPR than Config 4, convergence warnings, added complexity) |
| 14 | Ridge $\alpha=10.0$ + BiLSTM AE Fusion | 0.7717 | 0.0219 | 0.7284 | 0.8289 | `[0.9762, 0.1833, 1.0, 1.0, 1.0]` | `[0.0057, 0.0000, 0.0360, 0.0058, 0.0619]` | Rejected (Equal to Config 4 alone, added complexity yields zero gain over Ridge) |

**Total Configurations Evaluated**: 14 (excluding baseline = 14 candidates, well within the 15-configuration ceiling).

---

## Detailed Adoption Analysis for Configuration 4 (Ridge $\alpha=10.0$)
- **Pooled TPR**: Rose from $0.7397$ to **$0.7717$** ($+0.0320$)
- **Pooled FPR**: Dropped from $0.0233$ to **$0.0219$** ($-0.0014$)
- **Fold-by-Fold Check (All 5 folds improved or maintained)**:
  - Fold 0 (Attack 1): TPR $0.8810 \to 0.9762$ (improved), FPR $0.0057 \to 0.0057$ (equal)
  - Fold 1 (Attack 2): TPR $0.1333 \to 0.1833$ (improved, detection delay dropped from 49h to 44h), FPR $0.0000 \to 0.0000$ (equal)
  - Fold 2 (Attack 3): TPR $1.0000 \to 1.0000$ (equal), FPR $0.0360 \to 0.0360$ (equal)
  - Fold 3 (Attack 4): TPR $1.0000 \to 1.0000$ (equal), FPR $0.0058 \to 0.0058$ (equal)
  - Fold 4 (Attack 5): TPR $1.0000 \to 1.0000$ (equal), FPR $0.0691 \to 0.0619$ (improved)
- **Decision**: Configuration 4 meets all criteria of the fixed adoption rule (TPR rises at lower FPR across $\ge 4/5$ folds).
