# TrustRisk-WDS: Operational & Scientific Sign-Off Acceptance Checklist

**Evaluation Date**: 2026-10-12  
**Commit Audited**: d6beba9 (and follow-up consistency alignment)  
**Environment**: Windows 11 / Python 3.13.14 (Validated across Python 3.10, 3.11, 3.12 via GitHub Actions CI)  
**Evaluator**: Antigravity Pair-Programming Audit Suite  

---

## 1. Sign-Off Condition Matrix

| Audit Dimension | Requirement / Criterion | Verification Method Tier | Status | Evidence / Artifact |
| :---- | :---- | :---- | :----: | :---- |
| **Operational Dashboard Safety** | Default to Operational SCADA View; hide `y_true` oracle labels | **Tier 1 (Static)** + **Tier 3 (Runtime AppTest)** (`test_dashboard_default_mode_is_operational_no_oracle`) | **PASSED** | Live `AppTest.from_file()` execution confirms `index=0`, 0 unhandled exceptions, 0 oracle elements in operational mode; warning rendered in forensic mode |
| **Shared Alert Triage Policy** | Exact equivalence between Policy 5 in experiment and SCADA console | **Tier 1 (Static)** + **Tier 2 (Automated Test)** (`test_shared_policy5_routing_equivalence`) | **PASSED** | Canonical module `models/trust/policy5_router.py` imported identically in both `experiments/run_core_triage_experiment.py` and `dashboard/app.py` |
| **Canonical Risk Calculation** | Standardized TDCRI formula across dashboard and experiments | **Tier 1 (Static Code)** (`TDCRI = TS * R_physical`) | **PASSED** | Identical formulation: `tdcri = ts * alert_base_risk` evaluated in both modules using `compute_alert_physical_risk()` |
| **Asset & Consequence Resolution** | Alert top sensors mapped to physical EPANET assets with robust fallbacks | **Tier 2 (Automated Test)** (`test_sensor_to_asset_mapping_and_fallbacks`) | **PASSED** | Unknown/unmapped sensors return explicit `('UNMAPPED', 0.20)` without crash; live TDCRI uses mapped alert asset |
| **Temporal-Dependence Uncertainty** | Circular moving-block bootstrap sensitivity for L in {12, 24, 48} hours | **Tier 3 (Runtime Pipeline Execution)** (`experiments/v3_residual/run_block_bootstrap.py`) | **PASSED** | `results/v3/block_bootstrap_report.json`: Periodic boundary wrapping, seed=42; CI widens from [0.8079, 0.8659] (i.i.d.) to [0.7015, 0.9241] (48h) |
| **Honest Metric Transparency** | Pointwise recall (73.71%, 107 missed hours) strictly separated from 7/7 event recall | **Tier 1 (Documentation & Artifact Audit)** | **PASSED** | `README.md` and `docs/MANUSCRIPT_REVISION.md` explicitly state: *7/7 events detected does not mean all hours detected* |
| **Single-Shot Test Safeguard** | Active rerun blocker preventing threshold snooping or test leakage | **Tier 2 (Automated Test)** (`tests/test_v3_guard.py`) | **PASSED** | Cryptographic guard in `experiments/v3_residual/run_test_once.py` blocks re-runs and requires explicit user confirmation |
| **Physics-Grounded Hydraulic Simulation** | WNTR outage consequences on CTOWN.INP validate Base Risk | **Tier 3 (Runtime Simulation)** + **Tier 2 (Test)** (`test_risk.py`) | **PASSED** | rho(Base Risk, Pressure Deficit) = +0.6158 (p = 1.06e-4); Top-10 Kendall tau = 1.0000 |
| **Automated Test Suite** | Full pass rate with zero directory masking | **Tier 2 (Test Suite Execution)** (`pytest -v`) | **PASSED** | **30 passed in 3.60s (100% pass rate)** |
| **1-Command Master Reproduction** | End-to-end execution of all 5 phases and canonical tables | **Tier 3 (Runtime Execution)** (`python reproduce_all.py`) | **PASSED** | Complete run in ~25s with exit code 0 |

---

## 2. Verification Methodology Hierarchy & Dashboard Runtime Evidence

To maintain rigorous scientific standards, all verifications are categorized across three formal tiers:
- **Tier 1 (Static Source Inspection)**: Validating syntax, AST structures, formulas, and absence of forbidden import symbols.
- **Tier 2 (Automated Unit & Integration Tests)**: Pytest regression tests asserting mathematical and algorithmic contracts.
- **Tier 3 (Headless Runtime Execution)**: Live program executions measuring simulation outcomes and user-interface state transitions.

### Live Dashboard Headless Smoke Test Results (`streamlit.testing.v1.AppTest`)
A live programmatic headless execution of `dashboard/app.py` inside `tests/test_dashboard_policy5.py` verified the following runtime properties:
1. **Initial Mount & Render**: Headless app mounts without unhandled exceptions (`len(at.exception) == 0`).
2. **Default Radio State**: `at.sidebar.radio[0].value == "Operational SCADA View"` (verified index 0 default).
3. **Operational Oracle Isolation**: Programmatic scan of all rendered text widgets (`at.markdown`, `at.metric`, `at.dataframe`) confirms zero occurrences of `y_true`, `ATTACK`, or `ground truth`.
4. **Interactive State Transition**: Calling `.set_value("Offline Evaluation / Forensic Mode")` dynamically updates the view and triggers the ground-truth disclaimer warning (`at.warning`).
5. **Shared Decision Pipeline**: Evaluating alert timesteps through the dashboard triggers `models.trust.policy5_router`, standardizing the computed TDCRI and triage routing with `experiments/run_core_triage_experiment.py`.

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
