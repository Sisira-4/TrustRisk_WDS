# TrustRisk-WDS: Archival Release, CRediT Roles & Zenodo Deposit Plan

This document establishes the official open-science archival protocol, CRediT author statement, target SCIE venue review, and cryptographic manifest for the **TrustRisk-WDS** benchmark repository in compliance with **Phase 7** of the Project Guide.

---

## 1. Archival Release Metadata

- **Project Title**: TrustRisk-WDS: A Trust-Aware Cyber-Physical Risk Intelligence and Critical Asset Protection Framework for Smart Water Distribution Systems
- **Release Version**: `v1.0.0-gold`
- **Permanent Identifier**: Zenodo Reserved DOI `10.5281/zenodo.10148922` *(To be finalized on GitHub release webhook)*
- **Repository URI**: `https://github.com/shinojacob7/TrustRisk_WDS`
- **Software License**: MIT License
- **Academic Classification**: SCIE Computer Science / Cyber-Physical Systems / Water Resources Engineering

---

## 2. CRediT Author Statement & Contributions

In accordance with the *Contributor Roles Taxonomy (CRediT)*:

| Author | Affiliation | CRediT Roles |
| :--- | :--- | :--- |
| **Aardhra S Nair** | Govt. Engineering College Idukki | Conceptualization, Methodology, Software (Residual Anomaly Detectors), Data Curation, Formal Analysis |
| **Ashna Santhosh** | Govt. Engineering College Idukki | Methodology, Software (Unified Trust Engine, Probability Calibration, UQS), Validation, Writing – Review & Editing |
| **Shino Jacob C X** | Govt. Engineering College Idukki | Software (WNTR Hydraulic Modeling, SCADA Console Dashboard, CI Pipeline), Investigation, Data Visualization, Writing – Original Draft |
| **Sisira K S** | Govt. Engineering College Idukki | Software (Risk Index Formulations, Knapsack Optimization), Formal Analysis, Validation, Data Curation |
| **Dr. Sunil K S** | Govt. Engineering College Idukki | Supervision, Project Administration, Funding Acquisition, Conceptualization, Methodology, Writing – Review & Editing |

---

## 3. Candidate SCIE Venues Comparative Review (Phase 7)

In strict adherence to Phase 7 guidance (*"SCIE-indexed, scope covering CPS or water-infrastructure security, no mandatory APC; verify each candidate's current scope and policy on its own site"*):

| Target Venue | Publisher | Indexing / Metric | Scope Alignment | Mandatory APC? | Verdict & Submission Strategy |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **IEEE Transactions on Information Forensics and Security (TIFS)** | IEEE | SCIE (IF: 6.8, Core A*) | Cyber-physical security, ICS intrusion detection, trust modeling, false alarm suppression. | **No** (Optional Gold Open Access; free subscription publishing) | **Primary Target**: Highest prestige; directly values the trust-gating false alarm routing to review and formal statistical grounding. |
| **IEEE Internet of Things Journal (IoT-J)** | IEEE | SCIE (IF: 8.2, Core A) | Smart municipal water grids, SCADA telemetry, edge intrusion detection, risk intelligence. | **No** (Hybrid model; free subscription option available) | **Co-Primary Alternative**: Extremely high visibility across CPS and smart utility communities. |
| **Journal of Water Resources Planning and Management** | ASCE | SCIE (IF: 3.4) | Municipal water network security, EPANET hydraulic modeling, BATADAL benchmark community. | **No** (Standard ASCE publishing agreement; no mandatory APC) | **Strong Domain Alternative**: The exact host venue of the original BATADAL competition; reviewers intimately understand C-Town hydraulics. |
| **Computers & Security** | Elsevier | SCIE (IF: 4.8) | Industrial control system (ICS) security, alert triage in SOCs, operator alarm fatigue. | **No** (Hybrid subscription model; no fees for authors) | **Fallback Target**: Excellent fit for the alert fatigue operational triage paradigm. |

---

## 4. Cryptographic SHA-256 Frozen Artifact Manifest

To guarantee absolute experimental reproducibility and tamper-evident provenance across independent clones:

| File / Artifact Path | Canonical Description | SHA-256 Cryptographic Hash |
| :--- | :--- | :--- |
| [`frozen_config.json`](../frozen_config.json) | Frozen Detector Hyperparameter Specification | `3d5733ccc4f90b193d890250db7370631c5bb0f7a75f13d2d2e7c83e46528a6e` |
| [`results/v3/test_report.json`](../results/v3/test_report.json) | Official Quarantined Single-Shot Evaluation | `473b62481cba954c8462e0b123730a6ac0b2534fc05f3c7134c1331080f7f54e` |
| [`results/core_triage_experiment.json`](../results/core_triage_experiment.json) | 5-Policy Alert Triage Matrix Outcomes | `a7576ad404ea71d1e7693241e0f4e5466b4658e67ca068cb102a8e05960f4111` |
| [`results/baseline_comparison.json`](../results/baseline_comparison.json) | Identical-Protocol Baseline Evaluation | `6bed987263adb04e1f2a2823882566ea981c3535f24e5a27342ad400a95cb0f5` |
| [`results/risk_assets_report.csv`](../results/risk_assets_report.csv) | WNTR Extended-Period Hydraulic Simulation Results | `9f432e6bb372864ccd09f854bccb42edcaf56afb1c775e70b62235db5bbfc5bd` |
| [`network/epanet/CTOWN.INP`](../network/epanet/CTOWN.INP) | Benchmark C-Town Network Geometry | `9198b0bb43f551b29bfb550e206ff95f1df220118fc0068c7f130cec78d58d9e` |

---

## 5. Zenodo Archival Protocol & Reproduction Checklist

### Pre-Deposit Verification Checklist
1. [x] Working tree clean; zero temporary or untracked cache files.
2. [x] Single-command master reproduction suite (`python reproduce_all.py`) completes in < 30 seconds with exit code 0.
3. [x] Full test suite (`pytest -v`) passes 100% across all modules.
4. [x] All 5 publication figures in `results/figures/` generated at 300 DPI.
5. [x] Cryptographic hashes match the table above exactly.

### Release Execution Procedure
1. Create a lightweight local tag: `git tag -a v1.0.0-gold -m "TrustRisk-WDS Release v1.0.0-gold"`.
2. When directed by project guide, push tags to GitHub (`git push origin v1.0.0-gold`).
3. GitHub Actions triggers automated Zenodo webhook release archive creation.
4. Update `README.md` badge with confirmed Zenodo DOI badge: `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10148922.svg)](https://doi.org/10.5281/zenodo.10148922)`.
