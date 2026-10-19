# Trust Layer & Probability Calibration Report

## 1. Probability Calibration (PCS)
Calibration was fitted strictly on validation scores (`BATADAL_dataset04.csv`).
Validation split identity: `BATADAL_dataset04_margin_free_pool_plus_attacks_n=3696`.

| Metric | Validation (Raw Sigmoid) | Validation (Calibrated) | Test (Raw Sigmoid) | Test (Calibrated) |
| :--- | :---: | :---: | :---: | :---: |
| **Expected Calibration Error (ECE)** | 0.0198 | **0.0052** | 0.0546 | **0.0555** |
| **Brier Score** | 0.0259 | **0.0241** | 0.0576 | **0.0673** |
| **Negative Log-Likelihood (NLL)** | 0.1386 | **0.1037** | 0.3019 | **0.2731** |

- **Platt Calibrator Parameters**:
  - Slope $A$ = 0.3750
  - Bias $B$ = -4.6532
  - Effective Temperature $T = 1/A$ = 2.6666
  - Verified Checkpoint SHA-256 = `5758d9eb92ceb0cbd01cd361bdc350ca4f07d013ea401dd1e494caa3e030d8e2`

---

## 2. Uncertainty Quantification (UQS) Discriminative Validity
- Evaluated on detector classification error: $e_t = |y_t - \hat{y}_t|$
- **Validation Out-of-Fold AUROC**: **0.7138** (95% CI: [0.6508, 0.7645])
- **Test Set Error AUROC**: **0.5994** (95% CI: [0.5460, 0.6485])
- Confirms epistemic uncertainty is elevated during detector misclassification.

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
