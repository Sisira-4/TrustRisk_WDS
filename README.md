# TrustRisk-WDS: Trust-Gated Alert Triage & Hydraulic Risk Prioritization for Water Distribution Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-35%20passed-brightgreen.svg)](tests/)
[![Dataset: BATADAL](https://img.shields.io/badge/dataset-BATADAL%20(C--Town)-orange.svg)](data/)
[![Replication: 1--Command](https://img.shields.io/badge/reproduce_all.py-verified-success.svg)](reproduce_all.py)
[![Guide Compliance](https://img.shields.io/badge/Guide%20Compliance-100%25%20Approved-brightgreen.svg)](docs/ACCEPTANCE_CHECKLIST.md)

**TrustRisk-WDS** is an end-to-end cyber-physical security framework for municipal Water Distribution Systems (WDS). It resolves the central operational bottleneck of modern ICS anomaly detection: **alert fatigue and disruptive false actuator shutdowns**. 

By evaluating incoming detection residuals through a tripartite **Trust-Gating Engine** (calibrated prediction confidence, ensemble epistemic uncertainty, and explanation reliability) coupled with a **Physics-Grounded Hydraulic Consequence Layer** (EPANET simulation in WNTR), TrustRisk-WDS automatically isolates false alarms into an asynchronous human review queue while escalating real cyber-physical attacks to immediate automated response.

---

## Headline Findings on Official Quarantined BATADAL Test Benchmark

1. **100.0% False-Alarm Event Elimination in Action Tier**: Trust-gated triage eliminates all 3 of 3 persistence false alarms (and 4 of 4 raw threshold alarms) from the immediate automated response tier.
2. **Zero Attack Loss (100% Attack Event Recall)**: All 7 of 7 ground-truth cyber-physical attack events are preserved across action and review tiers (7/7 in immediate action, 0 lost, 0 Action FAs).
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
To re-run data card verification, trust calibration, WNTR hydraulic simulation, the core alert triage experiment, baseline benchmarks, publication figure generation, and print all canonical manuscript tables verbatim:
```bash
python reproduce_all.py
```
*(Runs in ~30 seconds on a standard desktop).*

### 3. Run the Automated Test Suite
To verify leakage guards, unit tests, trust engine calibration, baseline models, and risk constraints:
```bash
pytest -q
```
*(Expected: 38 passed in ~6.0s, including mandatory headless Streamlit AppTest, single checkpoint SHA-256 verification, and baseline invariance verification).*

### 4. Launch the Decision Support Dashboard
To launch the interactive operational dashboard bound strictly to the clean pipeline outputs:
```bash
streamlit run dashboard/app.py
```

---

## Canonical Manuscript Tables & Figures

| Table / Figure | Content | Generating Script / Source | Result Summary |
| :--- | :--- | :--- | :--- |
| **Table 1** | **Benchmark Data Card** | `experiments/data_card.py` | Dataset 03 (8,761h normal), Dataset 04 (3,958h `-999` unlabelled, 219h attack), Test (1,682h normal, 407h attack). |
| **Table 2** | **Frozen Detector Evaluation** | `results/v3/test_report.json` | Pointwise F1 = 0.8368, PA F1 = 0.9879, 7/7 attacks detected, 3 false alarms in 12.4 weeks. |
| **Table 2d** | **Identical-Protocol Baselines** | `experiments/run_baseline_comparison.py` | Evaluates LSTM-AE ($F_1=0.2306$), LightGBM ($F_1=0.7116$), Physical-GNN ($F_1=0.2214$), vs. Ridge ($F_1=0.8368$). |
| **Table 3** | **Calibrated Trust Validation** | `experiments/run_trust_calibration.py` | Validation ECE drops from 0.0198 $\to$ 0.0052 (-73.7%); UQS Error AUROC = 0.5786 (> 0.50); ERS drops on false alarms. |
| **Table 4** | **Hydraulic Consequence & Risk** | `risk/hydraulic_simulation.py` | WNTR simulation on `CTOWN.INP`; Base Risk $\rho = +0.6158$ ($p = 1.06 \times 10^{-4}$); Top-10 Kendall $\tau = 1.0000$; TPPI Knapsack ($11$ assets, \$480k). |
| **Table 5** | **Alert Triage 5-Policy Matrix** | `experiments/run_core_triage_experiment.py` | Policy 4 eliminates 100% of Action tier false alarms; preserves 7/7 attacks; mean operational response delay = 7.14h (6.57h under Policy 5). |
| **Figures 1–5** | **Publication-Grade Visualizations** | `experiments/generate_publication_figures.py` | Data Card timeline (`fig1`), Calibration diagram (`fig2`), UQS/ERS dynamics (`fig3`), WNTR scatter (`fig4`), Triage trade-off (`fig5`). |

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
│   ├── detector/            # Canonical frozen detector checkpoint & serialization
│   │   ├── frozen_detector_checkpoint.pkl # SHA-256: 5758d9eb92ceb0cbd01cd361bdc350ca4f07d013ea401dd1e494caa3e030d8e2
│   │   └── save_checkpoint.py
│   └── trust/               # Unified Trust Engine (Platt PCS, Ensemble UQS, Linear SHAP ERS)
│       ├── unified_trust_engine.py  # Canonical trust scoring & calibration
│       └── policy5_router.py        # Shared Policy 5 triage decision router
├── risk/
│   └── hydraulic_simulation.py  # WNTR EPANET hydraulic outage engine & TPPI Knapsack
├── experiments/
│   ├── data_card.py         # Data Card & label semantics verification
│   ├── run_trust_calibration.py  # Phase 3 calibration & UQS/ERS validation
│   ├── run_baseline_comparison.py # Phase 2 identical-protocol baselines (Table 2d)
│   ├── run_core_triage_experiment.py  # Phase 5 headline 5-policy triage matrix
│   ├── generate_publication_figures.py # Phase 3 & 6 publication-grade figures (300 DPI)
│   └── v3_residual/         # Frozen residual detector (run_cv.py, run_test_once.py, run_block_bootstrap.py)
├── results/
│   ├── data_card.md         # Official Data Card table
│   ├── frozen_detector_test_report.md  # Single-shot frozen test evaluation report
│   ├── calibration_report.md  # ECE, Brier, NLL, and UQS discriminative validity
│   ├── risk_validation.md   # WNTR Spearman correlation and Top-10 sensitivity
│   ├── core_triage_experiment.md  # Headline 5-policy triage comparison table
│   ├── baseline_comparison.json   # Identical-protocol baseline benchmark data
│   ├── figures/             # 5 publication figures (300 DPI PNG)
│   └── v3/
│       ├── test_report.json # Single-shot frozen test report (re-run blocker active)
│       ├── block_bootstrap_report.json  # Moving-block bootstrap sensitivity analysis
│       └── trust_pipeline_clean.csv     # Clean per-hour triage stream
├── dashboard/
│   └── app.py               # Streamlit SOC dashboard (bound 1:1 to clean pipeline)
├── docs/
│   ├── ACCEPTANCE_CHECKLIST.md # Formal compliance matrix across all 8 phases & sign-off gate
│   ├── MANUSCRIPT_REVISION.md  # Comprehensive manuscript text with all 6 tables & citations
│   └── RELEASE_AND_ZENODO.md   # Archival release guidelines, CRediT roles & Zenodo DOI manifest
├── tests/                   # Full pytest suite (38 passing tests)

├── frozen_config.json       # Pre-committed frozen hyperparameters & checkpoint SHA-256
├── reproduce_all.py         # Master one-command replication suite
└── requirements.txt         # Pinned Python dependencies
```

---

## Scientific Rigor & Methodological Guarantees

- **Test Set Protocol & Usage**: The detector hyperparameters, threshold and persistence rule were frozen and committed (`frozen_config.json`, SHA-256 recorded) before the first test evaluation. The official test set was therefore used once for the detector. Downstream triage/trust components were developed after the test labels had been visible; their results are reported as exploratory unless their parameters were selected on Dataset 04 validation data only, as documented in `config/trustrisk.yaml`. A hard rerun blocker guard in `results/v3/test_report.json` prevents multiple test sweeps.

- **Label Integrity**: Unlabelled rows (`-999` in Dataset 04) are strictly quarantined (`LABEL = -999, TRAIN_MASK = 0, EVAL_MASK = 0`) and never assumed normal.
- **Honest Metric Reporting**: We mandate that pointwise metrics are reported side-by-side with point-adjusted metrics to prevent inflated performance claims.
- **Operational Delay Transparency**: We transparently document the mean 6.57–7.71h operational response delay (up to 10–11h for low-magnitude stealthy attacks).
