# TrustRisk-WDS: Claims-to-Evidence Traceability Matrix

This document provides a strict, verifiable mapping from every quantitative claim made in [`README.md`](file:///c:/Users/shino/Desktop/Project/README.md) and [`docs/MANUSCRIPT_REVISION.md`](file:///c:/Users/shino/Desktop/Project/docs/MANUSCRIPT_REVISION.md) to its code-generated JSON/CSV artifact and enforcing test in the test suite.

In accordance with Rule 1 of the Project Specification (*No Fabrication*), zero numbers are typed by hand. Every claim is verified by [`tests/test_claims_evidence.py`](file:///c:/Users/shino/Desktop/Project/tests/test_claims_evidence.py), and a test failure occurs if any claim disagrees with its pipeline source file.

---

## Traceability Matrix

| Claim ID | Quantitative Claim Description | Claimed Value / String | Authoritative Artifact Source | Source JSON / CSV Key | Enforcing Test Name |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **CLM-01** | Dataset 03 (Training) Total Hours | `8,761` | `results/data_card.json` | `[0]['Total Hours (Rows)']` | `test_clm_01_data_card_dataset03_hours` |
| **CLM-02** | Dataset 04 (CV) Total Hours | `4,177` | `results/data_card.json` | `[1]['Total Hours (Rows)']` | `test_clm_02_data_card_dataset04_hours` |
| **CLM-03** | Dataset 04 Unlabelled Hours (`-999`) | `3,958` | `results/data_card.json` | `[1]['Unlabelled (-999)']` | `test_clm_03_data_card_dataset04_unlabeled` |
| **CLM-04** | Dataset 04 Attack Events | `5` | `results/data_card.json` | `[1]['Attack Events']` | `test_clm_04_data_card_dataset04_attacks` |
| **CLM-05** | Dataset 04 Margin Masked Hours ($\pm 48\text{h}$) | `480` | `results/data_card.json` | `[1]['Margin Masked (±48h)']` | `test_clm_05_data_card_dataset04_margin` |
| **CLM-06** | Official Test Benchmark Total Hours | `2,089` | `results/data_card.json` | `[2]['Total Hours (Rows)']` | `test_clm_06_data_card_test_hours` |
| **CLM-07** | Official Test Confirmed Normal Hours | `1,682` | `results/data_card.json` | `[2]['Confirmed Normal (0)']` | `test_clm_07_data_card_test_normal_hours` |
| **CLM-08** | Official Test Attack Hours | `407` | `results/data_card.json` | `[2]['Attack Hours (1)']` | `test_clm_08_data_card_test_attack_hours` |
| **CLM-09** | Official Test Cyber-Physical Attack Events | `7` | `results/data_card.json` | `[2]['Attack Events']` | `test_clm_09_data_card_test_attack_events` |
| **CLM-10** | Primary Pointwise F1 Score | `0.8368` | `results/v3/test_report.json` | `pointwise_primary.F1` | `test_clm_10_primary_pointwise_f1` |
| **CLM-11** | Primary Pointwise Precision | `0.9677` | `results/v3/test_report.json` | `pointwise_primary.precision` | `test_clm_11_primary_pointwise_precision` |
| **CLM-12** | Primary Pointwise Recall | `0.7371` | `results/v3/test_report.json` | `pointwise_primary.recall` | `test_clm_12_primary_pointwise_recall` |
| **CLM-13** | Point-Adjusted F1 Score | `0.9879` | `results/v3/test_report.json` | `point_adjusted_primary.F1` | `test_clm_13_point_adjusted_f1` |
| **CLM-14** | Primary ROC-AUC | `0.9180` | `results/v3/test_report.json` | `roc_auc` | `test_clm_14_roc_auc` |
| **CLM-15** | Primary PR-AUC | `0.8684` | `results/v3/test_report.json` | `pr_auc` | `test_clm_15_pr_auc` |
| **CLM-16** | True Positive Attack Hours Detected | `300` | `results/v3/test_report.json` | `pointwise_primary.TP` | `test_clm_16_pointwise_tp_detected` |
| **CLM-17** | False Positive Detection Hours | `10` | `results/v3/test_report.json` | `pointwise_primary.FP` | `test_clm_17_pointwise_fp_hours` |
| **CLM-18** | Attack Events Detected | `7 / 7` | `results/v3/test_report.json` | `events_detected` | `test_clm_18_events_detected` |
| **CLM-19** | Detector False Alarm Events (12.4 Weeks) | `3` | `results/v3/test_report.json` | `false_alarm_events` | `test_clm_19_false_alarm_events` |
| **CLM-20** | Frozen i.i.d. 95% Confidence Interval | `[0.8083, 0.8659]` | `results/v3/test_report.json` | `bootstrap_f1_95ci` | `test_clm_20_frozen_iid_bootstrap_ci` |
| **CLM-21** | Moving-Block Bootstrap 12h 95% CI | `[0.7586, 0.9022]` | `results/v3/block_bootstrap_report.json` | `moving_block_bootstrap['12h_blocks']['ci_95']` | `test_clm_21_block_bootstrap_12h_ci` |
| **CLM-22** | Moving-Block Bootstrap 48h 95% CI | `[0.7015, 0.9241]` | `results/v3/block_bootstrap_report.json` | `moving_block_bootstrap['48h_blocks']['ci_95']` | `test_clm_22_block_bootstrap_48h_ci` |
| **CLM-23** | Validation ECE Reduction | `0.0198 -> 0.0052` | `results/trust_calibration.json` | `validation.ece_before`, `validation.ece_after` | `test_clm_23_validation_ece_reduction` |
| **CLM-24** | Test UQS vs. Error AUROC | `0.5994` | `results/trust_calibration.json` | `test.uqs_auroc` | `test_clm_24_test_uqs_auroc` |
| **CLM-25** | Platt Calibrator Slope ($A$) | `0.3750` | `results/trust_calibration.json` | `platt_scale` | `test_clm_25_platt_calibrator_params` |
| **CLM-26** | Platt Calibrator Bias ($B$) | `-4.6532` | `results/trust_calibration.json` | `platt_bias` | `test_clm_26_platt_bias` |
| **CLM-27** | Top-10 TDCRI Kendall Rank Correlation | `0.9769` | `results/v3/trust_pipeline_clean.csv` / simulation | Invariance under $\pm 20\%$ weight perturbation | `test_clm_27_kendall_tau_stability` |
| **CLM-28** | WNTR Base Risk vs. ISS Spearman $\rho$ | `+0.6752` | `results/risk_assets_report.csv` | `spearmanr(Base_Risk, ISS)` | `test_clm_28_base_risk_spearman_rho` |
| **CLM-29** | Structural Centrality vs. ISS Spearman $\rho$ | `+0.0932` | `results/risk_assets_report.csv` | `spearmanr(Structural_Criticality, ISS)` | `test_clm_29_structural_spearman_rho` |
| **CLM-30** | Policy 1 (Raw Threshold) Action False Alarms | `4` | `results/core_triage_experiment.json` | `headline_policies[0]['unnecessary_responses']` | `test_clm_30_policy1_action_fa` |
| **CLM-31** | Policy 2 (Persistence) Action False Alarms | `3` | `results/core_triage_experiment.json` | `headline_policies[1]['unnecessary_responses']` | `test_clm_31_policy2_action_fa` |
| **CLM-32** | Policy 4 (Trust Gate) Action False Alarms | `0` | `results/core_triage_experiment.json` | `headline_policies[3]['unnecessary_responses']` | `test_clm_32_policy4_action_fa` |
| **CLM-33** | Policy 4 False Alarm Elimination Rate | `100.0%` | `results/core_triage_experiment.json` | `headline_policies[3]['reduction_vs_persist']` | `test_clm_33_policy4_fa_reduction` |
| **CLM-34** | Policy 4 Attacks Preserved (Action + Review) | `7/7` | `results/core_triage_experiment.json` | `headline_policies[3]['attacks_preserved']` | `test_clm_34_policy4_attacks_preserved` |
| **CLM-35** | Policy 4 Mean Operational Response Delay | `5.71` | `results/core_triage_experiment.json` | `headline_policies[3]['mean_delay_hours']` | `test_clm_35_policy4_mean_delay` |
| **CLM-36** | Policy 5 (Trust-Risk Gate) Action False Alarms | `0` | `results/core_triage_experiment.json` | `headline_policies[4]['unnecessary_responses']` | `test_clm_36_policy5_action_fa` |
| **CLM-37** | Policy 5 Attacks Preserved (Action + Review) | `7/7` | `results/core_triage_experiment.json` | `headline_policies[4]['attacks_preserved']` | `test_clm_37_policy5_attacks_preserved` |
| **CLM-38** | Policy 5 Mean Operational Response Delay | `6.86` | `results/core_triage_experiment.json` | `headline_policies[4]['mean_delay_hours']` | `test_clm_38_policy5_mean_delay` |
| **CLM-39** | Physical-GNN Test Pointwise F1 | `0.2282` | `results/baseline_comparison.json` | `Physical_GNN.pointwise_f1` | `test_clm_39_physical_gnn_f1` |
| **CLM-40** | Physical-GNN Test Point-Adjusted F1 | `0.8365` | `results/baseline_comparison.json` | `Physical_GNN.pa_f1` | `test_clm_40_physical_gnn_pa_f1` |
| **CLM-41** | LSTM Autoencoder Test Pointwise F1 | `0.2306` | `results/baseline_comparison.json` | `LSTM_Autoencoder.pointwise_f1` | `test_clm_41_lstm_ae_f1` |
| **CLM-42** | LSTM Autoencoder Test Point-Adjusted F1 | `0.2446` | `results/baseline_comparison.json` | `LSTM_Autoencoder.pa_f1` | `test_clm_42_lstm_ae_pa_f1` |
| **CLM-43** | LightGBM Test Pointwise F1 | `0.7116` | `results/baseline_comparison.json` | `LightGBM_Tabular_Residual.pointwise_f1` | `test_clm_43_lightgbm_f1` |
| **CLM-44** | LightGBM Test Point-Adjusted F1 | `0.9951` | `results/baseline_comparison.json` | `LightGBM_Tabular_Residual.pa_f1` | `test_clm_44_lightgbm_pa_f1` |
| **CLM-45** | TPPI Municipal Hardening Budget | `$50k default, $500k extended` | `config/trustrisk.yaml`, `results/risk_assets_report.csv` | `risk.tppi.budget_k` | `test_clm_45_tppi_budget_and_assets` |

---

## Verification Command

To automatically verify every claim in the matrix against its underlying pipeline artifact:
```bash
pytest tests/test_claims_evidence.py -v
```
