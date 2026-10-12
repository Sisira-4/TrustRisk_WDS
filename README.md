# TrustRisk-WDS: Trust-Gated Alert Triage & Hydraulic Risk Prioritization for Water Distribution Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-30%20passed-brightgreen.svg)](tests/)
[![Dataset: BATADAL](https://img.shields.io/badge/dataset-BATADAL%20(C--Town)-orange.svg)](data/)
[![Replication: 1--Command](https://img.shields.io/badge/reproduce_all.py-verified-success.svg)](reproduce_all.py)

**TrustRisk-WDS** is an end-to-end cyber-physical security framework for municipal Water Distribution Systems (WDS). It resolves the central operational bottleneck of modern ICS anomaly detection: **alert fatigue and disruptive false actuator shutdowns**. 

By evaluating incoming detection residuals through a tripartite **Trust-Gating Engine** (calibrated prediction confidence, ensemble epistemic uncertainty, and explanation reliability) coupled with a **Physics-Grounded Hydraulic Consequence Layer** (EPANET simulation in WNTR), TrustRisk-WDS automatically isolates false alarms into an asynchronous human review queue while escalating real cyber-physical attacks to immediate automated response.

---

## Headline Findings on Official Quarantined BATADAL Test Benchmark

1. **100.0% False-Alarm Event Elimination in Action Tier**: Trust-gated triage eliminates all 3 of 3 persistence false alarms (and 4 of 4 raw threshold alarms) from the immediate automated response tier.
2. **Zero Attack Loss (100% Attack Event Recall)**: All 7 of 7 ground-truth cyber-physical attack events are preserved across action and review tiers (6 in immediate action, 1 in human review).
3. **Transparent Pointwise Evaluation (No Metric Inflation)**:
   - **Pointwise F1**: **0.8368** (official frozen 95% CI: $[0.8083, 0.8659]$; 2,000-replicate i.i.d. sensitivity $[0.8079, 0.8659]$; moving-block bootstrap $[0.7586, 0.9022]$ at 12h, $[0.7015, 0.9241]$ at 48h)
   - **Pointwise Precision & Recall**: Precision **0.9677**, Recall **0.7371** (300 / 407 attack hours detected; 107 attack hours missed during accumulation phases)
   - **Point-Adjusted F1**: **0.9879** (Precision: 0.9760, Recall: 1.0000)
   - **ROC-AUC**: **0.9180** | **PR-AUC**: **0.8684**
4. **Physical Consequence Grounding**: We demonstrate that graph betweenness centrality alone fails to predict hydraulic consequence ($\rho = -0.1932$), whereas our Cyber-Physical Base Risk strongly correlates with simulated pressure deficits ($\rho = \mathbf{+0.6158}, p = 1.06 \times 10^{-4}$).

---

## Quickstart & Replication

### 1. Installation
Clone the repository and install the pinned dependencies:
```bash
git clone https://github.com/shinojacob7/TrustRisk_WDS.git
cd TrustRisk_WDS
pip install -r requirements.txt
```

### 2. One-Command Full Pipeline Reproduction
To re-run data card verification, trust calibration, WNTR hydraulic simulation, the core alert triage experiment, and print all 5 manuscript tables verbatim:
```bash
python reproduce_all.py
```
*(Runs in ~25 seconds on a standard desktop).*

### 3. Run the Automated Test Suite
To verify leakage guards, unit tests, trust engine calibration, and risk constraints:
```bash
pytest -q
```
*(Expected: 30 passed in ~3s).*

### 4. Launch the Decision Support Dashboard
To launch the interactive operational dashboard bound strictly to the clean pipeline outputs:
```bash
streamlit run dashboard/app.py
```

---

## Canonical Manuscript Tables

| Table | Content | Generating Script / Source | Result Summary |
| :--- | :--- | :--- | :--- |
| **Table 1** | **Benchmark Data Card** | `experiments/data_card.py` | Dataset 03 (8,761h normal), Dataset 04 (3,958h `-999` unlabelled, 219h attack), Test (1,682h normal, 407h attack). |
| **Table 2** | **Frozen Detector Evaluation** | `results/v3/test_report.json` | Pointwise F1 = 0.8368, PA F1 = 0.9879, 7/7 attacks detected, 3 false alarms in 12.4 weeks. |
| **Table 3** | **Calibrated Trust Validation** | `experiments/run_trust_calibration.py` | Validation ECE drops from 0.0198 $\to$ 0.0052 (-73.7%); UQS Error AUROC = 0.5945 (> 0.50); ERS drops on false alarms. |
| **Table 4** | **Hydraulic Consequence & Risk** | `risk/hydraulic_simulation.py` | WNTR simulation on `CTOWN.INP`; Base Risk $\rho = +0.6158$ ($p = 1.06 \times 10^{-4}$); Top-10 Kendall $\tau = 1.0000$; TPPI Knapsack ($11$ assets, \$480k). |
| **Table 5** | **Alert Triage 5-Policy Matrix** | `experiments/run_core_triage_experiment.py` | Policy 4 eliminates 100% of Action tier false alarms; preserves 7/7 attacks; mean operational response delay = 6.86h. |

---

## Project Structure

```text
TrustRisk-WDS/
├── data/
│   ├── raw/                 # Official BATADAL datasets (Dataset 03, 04, Test Labeled)
│   └── v2/                  # Strict cross-validation folds and metadata
├── network/
│   ├── epanet/CTOWN.INP     # Official C-Town EPANET hydraulic network
│   └── mappings/            # Sensor-to-physical EPANET asset mapping
├── models/
│   └── trust/               # Unified Trust Engine (Platt PCS, Ensemble UQS, Perturbation ERS)
│       ├── unified_trust_engine.py  # Canonical trust scoring & calibration
│       └── policy5_router.py        # Shared Policy 5 triage decision router
├── risk/
│   └── hydraulic_simulation.py  # WNTR EPANET hydraulic outage engine & TPPI Knapsack
├── experiments/
│   ├── data_card.py         # Data Card & label semantics verification
│   ├── run_trust_calibration.py  # Phase 3 calibration & UQS/ERS validation
│   ├── run_core_triage_experiment.py  # Phase 5 headline 5-policy triage matrix
│   └── v3_residual/         # Frozen residual detector (run_cv.py, run_test_once.py, run_block_bootstrap.py)
├── results/
│   ├── data_card.md         # Official Data Card table
│   ├── frozen_detector_test_report.md  # Single-shot frozen test evaluation report
│   ├── calibration_report.md  # ECE, Brier, NLL, and UQS discriminative validity
│   ├── risk_validation.md   # WNTR Spearman correlation and Top-10 sensitivity
│   ├── core_triage_experiment.md  # Headline 5-policy triage comparison table
│   └── v3/
│       ├── test_report.json # Single-shot frozen test report (re-run blocker active)
│       ├── block_bootstrap_report.json  # Moving-block bootstrap sensitivity analysis
│       └── trust_pipeline_clean.csv     # Clean per-hour triage stream
├── dashboard/
│   └── app.py               # Streamlit SOC dashboard (bound 1:1 to clean pipeline)
├── docs/
│   ├── ACCEPTANCE_CHECKLIST.md # Audit verification matrix across three formal tiers
│   └── MANUSCRIPT_REVISION.md  # Comprehensive manuscript text with all 5 tables & citations
├── tests/                   # Full pytest suite (30 passing tests)
├── frozen_config.json       # Pre-committed frozen hyperparameters (Ridge α=10.0, q=99.0)
├── reproduce_all.py         # Master one-command replication suite
└── requirements.txt         # Pinned Python dependencies
```

---

## Scientific Rigor & Methodological Guarantees

- **Single-Shot Quarantined Test Evaluation**: `BATADAL_test_dataset_labeled.csv` was evaluated strictly once after committing `frozen_config.json`. A hard rerun blocker guard in `results/v3/test_report.json` prevents multiple test sweeps.
- **Label Integrity**: Unlabelled rows (`-999` in Dataset 04) are strictly quarantined (`LABEL = -999, TRAIN_MASK = 0, EVAL_MASK = 0`) and never assumed normal.
- **Honest Metric Reporting**: We mandate that pointwise metrics are reported side-by-side with point-adjusted metrics to prevent inflated performance claims.
- **Operational Delay Transparency**: We transparently document the mean 6.86h operational response delay (up to 10–11h for low-magnitude stealthy attacks).
