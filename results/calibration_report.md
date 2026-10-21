# Trust Layer & Probability Calibration Report

## 1. Probability Calibration (PCS)
Calibration was fitted strictly on validation scores (`BATADAL_dataset04.csv`).
Validation split identity: `BATADAL_dataset04_margin_free_pool_plus_attacks_n=3696`.

| Metric | Validation (Raw Sigmoid) | Validation (Calibrated) | Test (Raw Sigmoid) | Test (Calibrated) |
| :--- | :---: | :---: | :---: | :---: |
| **Expected Calibration Error (ECE)** | 0.0198 | **0.0052** | 0.0546 | **0.0555** |
| **Brier Score** | 0.0259 | **0.0241** | 0.0576 | **0.0673** |
| **Negative Log-Likelihood (NLL)** | 0.1386 | **0.1037** | 0.3019 | **0.2731** |

### Calibration Method Comparison (Validation vs Test)
| Method | Val ECE | Val Brier | Test ECE | Test Brier |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Sigmoid** | 0.0198 | 0.0259 | 0.0546 | 0.0576 |
| **Platt Scaling (Logistic)** | **0.0052** | **0.0241** | **0.0555** | **0.0673** |
| **Isotonic Regression** | 0.0000 | 0.0224 | 0.0536 | 0.0610 |

- **Platt Calibrator Parameters**:
  - Slope $A$ = 0.3750
  - Bias $B$ = -4.6532
  - Effective Temperature $T = 1/A$ = 2.6666
  - Verified Checkpoint SHA-256 = `5758d9eb92ceb0cbd01cd361bdc350ca4f07d013ea401dd1e494caa3e030d8e2`

- **Methodological Limitation (Calibration Transfer Failure under Temporal Shift)**:
  Platt scaling drastically improves validation calibration (ECE 0.0198 -> 0.0052, Brier 0.0259 -> 0.0241), but calibration does not transfer to the test period (test ECE 0.0546 -> 0.0555, Brier 0.0576 -> 0.0673). Isotonic regression displays the same dynamic (test ECE 0.0536, Brier 0.0610). This occurs because of an approximate 3.75x attack prevalence shift (5.9% attack hours in the Dataset 04 validation pool vs. 19.5% in the test set). Supervised probability calibrators reflect training base rates and underpredict probabilities when test attack frequency surges, even while monotonic rank ordering remains preserved.

---

## 2. Uncertainty Quantification (UQS) Discriminative Validity
- Evaluated on detector classification error:  = |y_t - \\hat{y}_t|
- **Validation Out-of-Fold AUROC**: **0.7138** (95% CI: [0.6508, 0.7645])
- **Test Set Error AUROC**: **0.5994** (95% CI: [0.5460, 0.6485])
- **Characterization**: While statistically superior to random guessing (> 0.50), a test error AUROC of 0.5994 demonstrates weak discriminative power for detector classification errors under temporal distribution shift, rather than robust error prediction. Epistemic uncertainty alone is insufficient to identify subtle sensor drift without physical residual coupling.

---

## 3. Explanation Reliability (ERS) Dynamics
- Evaluated via SHAP LinearExplainer over observed physical SCADA sensors under background resampling.
- **ERS Variation**: Min = 0.6802, Mean = 0.9536, Max = 1.0000, Std = 0.0635
- **Mean ERS during Attack Hours ($y=1$)**: **0.9876**
- **Mean ERS during Normal Hours ($y=0$)**: 0.9453
- **Mean ERS during False Alarms**: 0.9211
- **Attack vs. False Alarm ERS Difference**: **+0.0664**
- Genuine ERS dynamically discriminates robust, sustained physical attacks from stochastic false alarms.

---

## 4. Historical Reliability Score (HRS)
- Evaluated as causal rolling precision of past alerts over a 168-hour window without lookahead.
- Cold-start fallback rule: prior rate 1.0.

---

## 5. Canonical Trust Score & ATI Triage Distribution
- **Canonical Formula**:  
  $$TS_t = 0.30 \cdot PCS_t + 0.40 \cdot (1 - UQS_t) + 0.30 \cdot ERS_t$$
- **Calibrated Triage Bands on Test Set**:
  - **ACTION ($TS \ge 0.85$)**: 7 hours
  - **REVIEW ($0.50 \le TS < 0.85$)**: 1323 hours
  - **MONITOR ($TS < 0.50$)**: 759 hours
