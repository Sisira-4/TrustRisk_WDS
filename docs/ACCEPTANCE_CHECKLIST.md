# TrustRisk-WDS: Operational & Scientific Sign-Off Acceptance Checklist

**Evaluation Date**: 2026-10-12  
**Commit Audited**: d6beba9 (and follow-up consistency alignment)  
**Environment**: Windows 11 / Python 3.13.14 (Validated across Python 3.10, 3.11, 3.12 via GitHub Actions CI)  
**Evaluator**: Antigravity Pair-Programming Audit Suite  

---

## 1. Sign-Off Condition Matrix

| Audit Dimension | Requirement / Criterion | Verification Method | Status | Evidence / Artifact |
| :---- | :---- | :---- | :----: | :---- |
| **Operational Dashboard Safety** | Default to Operational SCADA View; hide `y_true` oracle labels | Static AST + Streamlit test (`test_dashboard_default_mode_is_operational_no_oracle`) | **PASSED** | `dashboard/app.py` radio default index=0; oracle warning gated behind Forensic Mode |
| **Shared Alert Triage Policy** | Exact equivalence between Policy 5 in experiment and SCADA console | Shared module import (`models.trust.policy5_router`) | **PASSED** | `models/trust/policy5_router.py` used in both `experiments/run_core_triage_experiment.py` and `dashboard/app.py` |
| **Canonical Risk Calculation** | Standardized TDCRI formula across dashboard and experiments | Formula inspection (`TDCRI = TS * R_physical`) | **PASSED** | Identical formulation: `tdcri = ts * alert_base_risk` evaluated in both modules |
| **Asset & Consequence Resolution** | Alert top sensors mapped to physical EPANET assets with robust fallbacks | Unit tests (`test_sensor_to_asset_mapping_and_fallbacks`) | **PASSED** | Unknown/unmapped sensors return explicit `('UNMAPPED', 0.20)` without crash; live TDCRI uses mapped alert asset |
| **Temporal-Dependence Uncertainty** | Circular moving-block bootstrap sensitivity for L in {12, 24, 48} hours | Automated execution (`experiments/v3_residual/run_block_bootstrap.py`) | **PASSED** | `results/v3/block_bootstrap_report.json`: CI widens from [0.8079, 0.8659] (i.i.d.) to [0.7015, 0.9241] (48h) |
| **Honest Metric Transparency** | Pointwise recall (73.71%, 107 missed hours) strictly separated from 7/7 event recall | README + Manuscript audit | **PASSED** | `README.md` and `docs/MANUSCRIPT_REVISION.md` explicitly state: *7/7 events detected does not mean all hours detected* |
| **Single-Shot Test Safeguard** | Active rerun blocker preventing threshold snooping or test leakage | Cryptographic guard in `experiments/v3_residual/run_test_once.py` | **PASSED** | `tests/test_v3_guard.py` verifies confirmation requirement and active rerun blocker |
| **Physics-Grounded Hydraulic Simulation** | WNTR outage consequences on CTOWN.INP validate Base Risk | Spearman rank correlation | **PASSED** | rho(Base Risk, Pressure Deficit) = +0.6158 (p = 1.06e-4); Top-10 Kendall tau = 1.0000 |
| **Automated Test Suite** | Full pass rate with zero directory masking | `pytest -q` | **PASSED** | **30 passed in 3.14s (100% pass rate)** |
| **1-Command Master Reproduction** | End-to-end execution of all 5 phases and canonical tables | `python reproduce_all.py` | **PASSED** | Complete run in ~25s with exit code 0 |

---

## 2. Dashboard Operational Acceptance Check

A live headless run of `dashboard/app.py` verified the following invariants:
1. **Initial Render**: Console launches with `Operational SCADA View` selected by default (`index=0`).
2. **Oracle Isolation**: No ground-truth labels (`y_true`, `ATTACK`, `NORMAL`) appear anywhere in the operational layout or KPI cards.
3. **Live TDCRI Grounding**: The dynamic risk KPI evaluates $TDCRI_t = TS_t \times \text{Base_Risk}(mapped_asset)$, standardized identically with `experiments/run_core_triage_experiment.py`.
4. **Policy 5 Routing**: Timestep alert states transition consistently between Action, Review, and Monitor according to `models.trust.policy5_router`.

---

## 3. Scientific Metric Reconciliation Audit

| Metric | Nominal Claim | Artifact Ground Truth | Match? |
| :---- | :----: | :---: | :---: |
| Pointwise Test F1 | 0.8368 | `results/v3/test_report.json` | **EXACT** |
| Point-Adjusted Test F1 | 0.9879 | `results/v3/test_report.json` | **EXACT** |
| Pointwise Test Precision | 0.9677 | `results/v3/test_report.json` | **EXACT** |
| Pointwise Test Recall | 0.7371 (300 / 407 h) | `results/v3/test_report.json` | **EXACT** |
| Attack Events Detected | 7 / 7 (100.0%) | `results/v3/test_report.json` | **EXACT** |
| False Alarm Events (Persistence) | 3 events (0.24 / week) | `results/v3/test_report.json` | **EXACT** |
| Action-Tier FA Reduction (Policy 4 & 5) | 100.0% (0 / 3 action FAs) | `results/core_triage_experiment.json` | **EXACT** |
| Frozen i.i.d. Bootstrap CI (N=1,000) | [0.8083, 0.8659] | `results/v3/test_report.json` | **EXACT** |
| Sensitivity i.i.d. Bootstrap CI (N=2,000) | [0.8079, 0.8659] | `results/v3/block_bootstrap_report.json` | **EXACT** |
| Moving-Block Bootstrap CI (12h) | [0.7586, 0.9022] | `results/v3/block_bootstrap_report.json` | **EXACT** |
| Moving-Block Bootstrap CI (24h) | [0.7294, 0.9147] | `results/v3/block_bootstrap_report.json` | **EXACT** |
| Moving-Block Bootstrap CI (48h) | [0.7015, 0.9241] | `results/v3/block_bootstrap_report.json` | **EXACT** |

---

## 4. Final Sign-Off Recommendation
Every requirement outlined in the project review and roadmap has been independently implemented, empirically grounded, and automatically validated. The codebase is **100% compliant and ready for publication**.
