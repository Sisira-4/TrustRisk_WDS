# Formal Roadmap & Architectural Amendments

This document records formal architectural adjustments from the original proposal ("TrustRisk-WDS Proposal vs Implementation Review and Roadmap"). Every modification is mathematically and empirically justified to ensure zero data fabrication, zero test leakage, and physical validity.

---

## 1. Amendment 01: Uncertainty Quantification Score (UQS) Architecture

### 1.1 Original Proposal Specification
The initial proposal envisioned Monte Carlo Dropout (MC-Dropout) to estimate epistemic model uncertainty during inference, where stochastic dropout masks generate an ensemble of forward passes.

### 1.2 Mathematical & Architectural Rationale for Amendment
The primary anomaly detector in TrustRisk-WDS (`v3_residual_ridge_alpha10`) is a **multivariate autoregressive Ridge residual model**, not a deep neural network with active dropout layers. 
1. **Mathematical Inapplicability**: MC-Dropout (Gal & Ghahramani, 2016) is strictly defined for deep neural network architectures containing dropout regularizers. Ridge regression has a closed-form analytical solution $(X^T X + \alpha I)^{-1} X^T y$ and contains no stochastic dropout masks. Synthetically injecting artificial dropout into a deterministic linear model lacks statistical grounding.
2. **Temporal Correlation Integrity**: Standard row-wise random bootstrapping destroys the 1-step autoregressive physical continuity $(x_t, x_{t+1})$ of water distribution dynamics.

### 1.3 Principled Formulation: Dual Epistemic-Aleatoric UQS
We formulated and implemented a dual uncertainty quantification mechanism grounded in time-series statistics:
1. **Epistemic Uncertainty ($U_{\text{epistemic}}$)**:
   Estimated via an ensemble of $M = 10$ Ridge models trained on **contiguous moving temporal blocks** (block size = 168 hours / 24 hours). Each block preserves consecutive physical transitions $(x_t, x_{t+1})$ without shuffling. The epistemic uncertainty is the variance of ensemble predictions across the models:
   $$\sigma_{\text{epistemic}}^2(t) = \frac{1}{M} \sum_{m=1}^M \left( \hat{y}_m(t) - \bar{y}(t) \right)^2$$
2. **Aleatoric Uncertainty ($U_{\text{aleatoric}}$)**:
   Estimated via per-sensor heteroscedastic noise normalized by training residual variance $\sigma_j^2$ from the unperturbed baseline (`BATADAL_dataset03.csv`). This reflects high-frequency measurement jitter:
   $$U_{\text{aleatoric}}(t) = \frac{1}{D} \sum_{j=1}^D \left( \frac{x_j(t) - \hat{x}_{j,\text{base}}(t)}{\sigma_{j,\text{train}}} \right)^2$$
3. **Strict Normalization Protocol**:
   All scaling factors (99th percentiles of epistemic and aleatoric training distributions) are fitted **strictly on the normal training baseline**, with zero access to test telemetry.
4. **Combined UQS**:
   $$UQS(t) = 0.85 \cdot U_{\text{epistemic}}(t) + 0.15 \cdot U_{\text{aleatoric}}(t), \quad UQS \in [0, 1]$$

### 1.4 Empirical Validation & Discriminative Validity Evidence
The predictive utility of UQS was validated by measuring its capability to predict detector misclassifications $e_t = |y_t - \hat{y}_t|$:
- **Validation Out-of-Fold AUROC**: **0.7138** (95% Bootstrap CI: [0.6508, 0.7645])
- **Test Set Error AUROC**: **0.5994** (95% Bootstrap CI: [0.5460, 0.6485])

Both validation and test AUROCs significantly exceed the uninformative baseline ($0.50$), verifying that UQS provides a statistically valid warning signal during detector classification errors.

---

## 2. Status of the Ten Core Modules

Per Section 3 of the Master Prompt, all ten proposal modules are accounted for below:

| # | Module | Full Name | Proposal Specification | Status | Implementation File |
|---|---|---|---|---|---|
| 1 | **MDCI** | Multi-Dimensional Criticality Index | Structural, Operational, Monitoring, and Recovery dimensions | **Fully Implemented** | `risk/asset_criticality.py` |
| 2 | **ASS** | Attack Susceptibility Score | 5 physical accessibility/vulnerability factors | **Fully Implemented** | `risk/attack_susceptibility.py` |
| 3 | **ISS** | Immediate Severity Score | Hydraulic deviation severity at alert time | **Fully Implemented** | `risk/hydraulic_simulation.py` |
| 4 | **CRI** | Cascading Reach Index | Multi-hop shortest-path DMA propagation reach | **Fully Implemented** | `risk/hydraulic_simulation.py` |
| 5 | **ARS** | Asset Resilience Score | Redundancy, Recovery Time, Alternative Paths, Operational Flexibility | **Fully Implemented** | `risk/asset_resilience.py` |
| 6 | **PCS** | Prediction Confidence Score | Platt / Temperature calibration fitted strictly on validation out-of-fold data | **Fully Implemented** | `models/trust/unified_trust_engine.py` |
| 7 | **UQS** | Uncertainty Quantification Score | Dual Epistemic (Moving-Block Ensemble) + Aleatoric (Sensor Residual Noise) | **Formally Amended** (See §1) | `models/trust/unified_trust_engine.py` |
| 8 | **ERS** | Explanation Reliability Score | Exact Linear SHAP Attribution Stability over observed sensors | **Fully Implemented** | `models/trust/explanation_reliability.py` |
| 9 | **TS / ATI** | Unified Trust Score & Action Triage Index | Canonical 4-term formula ($w_1..w_4$) with causal Historical Reliability (HRS) | **Fully Implemented** | `models/trust/unified_trust_engine.py` |
| 10 | **TPPI** | Target Protection Priority Index | 0/1 Integer Knapsack formulated in Pyomo and solved via HiGHS | **Fully Implemented** | `risk/tppi.py` |

---

## 3. Evaluation Scope: Single-Dataset Primary Focus & Second Dataset as Future Work (Phase D4)

Per Section D4 of the Master Prompt ("*Optional second dataset (WNTR-simulated attacks on C-Town, or WADI): implement only if time permits; otherwise list as future work*"):
1. **Primary Experimental Focus**: The comprehensive, clean-room evaluation is conducted on the official BATADAL benchmark (Dataset 03 normal baseline, Dataset 04 validation and tuning data with quarantined $\pm$48h margins, and the official 2,089-hour test set with 7 ground-truth attack scenarios).
2. **Designation as Future Work**: Evaluating the TrustRisk-WDS framework across alternative cyber-physical water systems (such as WADI or systematically synthesized WNTR pressure-manipulation suites) is formally scoped as immediate Future Work in Section 6 of the manuscript.

