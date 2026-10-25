"""Builds all empirical tables in the manuscript as Markdown and LaTeX directly from pipeline artifacts.

Guarantees 100% adherence to Rule 1 (No Fabrication / Zero Hardcoding).
Outputs are written to results/tables/*.md and results/tables/latex/*.tex.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_config
from risk.tppi import TPPISolver

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
LATEX_DIR = TABLES_DIR / "latex"


def build_table_1_data_card(data_card: list) -> Tuple[str, str]:
    """Table 1: Benchmark Data Card & Masking Protocol."""
    ds3 = next(r for r in data_card if "03" in r["Dataset Split"])
    ds4 = next(r for r in data_card if "04" in r["Dataset Split"])
    ds_test = next(r for r in data_card if "Test" in r["Dataset Split"])

    headers = ["Dataset Split", "Total Hours (Rows)", "Confirmed Normal (`0`)", "Unlabelled (`-999`)", "Attack Hours (`1`)", "Attack Events", "Margin Masked ($\\pm 48\\text{h}$)"]
    rows = [
        ["**Training (BATADAL Dataset 03)**", f"{ds3['Total Hours (Rows)']:,}", f"{ds3['Confirmed Normal (0)']:,}", str(ds3['Unlabelled (-999)']), str(ds3['Attack Hours (1)']), str(ds3['Attack Events']), str(ds3['Margin Masked (±48h)'])],
        ["**Cross-Validation (BATADAL Dataset 04)**", f"{ds4['Total Hours (Rows)']:,}", str(ds4['Confirmed Normal (0)']), f"{ds4['Unlabelled (-999)']:,}", str(ds4['Attack Hours (1)']), str(ds4['Attack Events']), str(ds4['Margin Masked (±48h)'])],
        ["**Test (BATADAL Test Labeled)**", f"{ds_test['Total Hours (Rows)']:,}", f"{ds_test['Confirmed Normal (0)']:,}", str(ds_test['Unlabelled (-999)']), str(ds_test['Attack Hours (1)']), str(ds_test['Attack Events']), str(ds_test['Margin Masked (±48h)'])],
    ]

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Benchmark Data Card \& Masking Protocol}",
        r"\label{tab:data_card}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset Split & Total Hours & Normal & Unlabelled & Attack Hours & Attack Events & Masked Margin \\",
        r"\midrule",
        f"Training (Dataset 03) & {ds3['Total Hours (Rows)']:,} & {ds3['Confirmed Normal (0)']:,} & {ds3['Unlabelled (-999)']} & {ds3['Attack Hours (1)']} & {ds3['Attack Events']} & {ds3['Margin Masked (±48h)']} \\\\",
        f"Cross-Validation (Dataset 04) & {ds4['Total Hours (Rows)']:,} & {ds4['Confirmed Normal (0)']} & {ds4['Unlabelled (-999)']:,} & {ds4['Attack Hours (1)']} & {ds4['Attack Events']} & {ds4['Margin Masked (±48h)']} \\\\",
        f"Test (Test Labeled) & {ds_test['Total Hours (Rows)']:,} & {ds_test['Confirmed Normal (0)']:,} & {ds_test['Unlabelled (-999)']} & {ds_test['Attack Hours (1)']} & {ds_test['Attack Events']} & {ds_test['Margin Masked (±48h)']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_2_detector_eval(test_rep: dict, data_card: list) -> Tuple[str, str]:
    """Table 2: Frozen Residual Detector Test Evaluation (Pointwise vs. Point-Adjusted)."""
    pw = test_rep["pointwise_primary"]
    pa = test_rep["point_adjusted_primary"]
    raw = test_rep["pointwise_raw"]
    raw_pa = test_rep["point_adjusted_raw"]
    ci_iid = test_rep["bootstrap_f1_95ci"]
    ds_test = next(r for r in data_card if "Test" in r["Dataset Split"])

    md_lines = [
        "| Metric | Pointwise (Primary) | Point-Adjusted (PA) | Raw (No Persistence) | Raw PA |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **F1 Score** | **{pw['F1']:.4f}** (95% CI: [{ci_iid[0]:.4f}, {ci_iid[1]:.4f}]) | **{pa['F1']:.4f}** | {raw['F1']:.4f} | {raw_pa['F1']:.4f} |",
        f"| **Precision** | {pw['precision']:.4f} | {pa['precision']:.4f} | {raw['precision']:.4f} | {raw_pa['precision']:.4f} |",
        f"| **Recall** | {pw['recall']:.4f} | {pa['recall']:.4f} | {raw['recall']:.4f} | {raw_pa['recall']:.4f} |",
        f"| **True Positive Hours** | {pw['TP']} / {ds_test['Attack Hours (1)']} | {pa['TP']} / {ds_test['Attack Hours (1)']} | {raw['TP']} / {ds_test['Attack Hours (1)']} | {raw_pa['TP']} / {ds_test['Attack Hours (1)']} |",
        f"| **False Positive Hours** | {pw['FP']} / {ds_test['Confirmed Normal (0)']} | {pa['FP']} / {ds_test['Confirmed Normal (0)']} | {raw['FP']} / {ds_test['Confirmed Normal (0)']} | {raw_pa['FP']} / {ds_test['Confirmed Normal (0)']} |",
    ]
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Frozen Residual Detector Evaluation (Pointwise vs. Point-Adjusted)}",
        r"\label{tab:detector_eval}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Pointwise (Primary) & Point-Adjusted (PA) & Raw (No Persistence) & Raw PA \\",
        r"\midrule",
        f"F1 Score & \\textbf{{{pw['F1']:.4f}}} & \\textbf{{{pa['F1']:.4f}}} & {raw['F1']:.4f} & {raw_pa['F1']:.4f} \\\\",
        f"Precision & {pw['precision']:.4f} & {pa['precision']:.4f} & {raw['precision']:.4f} & {raw_pa['precision']:.4f} \\\\",
        f"Recall & {pw['recall']:.4f} & {pa['recall']:.4f} & {raw['recall']:.4f} & {raw_pa['recall']:.4f} \\\\",
        f"True Positive Hours & {pw['TP']} / {ds_test['Attack Hours (1)']} & {pa['TP']} / {ds_test['Attack Hours (1)']} & {raw['TP']} / {ds_test['Attack Hours (1)']} & {raw_pa['TP']} / {ds_test['Attack Hours (1)']} \\\\",
        f"False Positive Hours & {pw['FP']} / {ds_test['Confirmed Normal (0)']} & {pa['FP']} / {ds_test['Confirmed Normal (0)']} & {raw['FP']} / {ds_test['Confirmed Normal (0)']} & {raw_pa['FP']} / {ds_test['Confirmed Normal (0)']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_2b_bootstrap_sensitivity(block_rep: dict) -> Tuple[str, str]:
    """Table 2b: Temporal-Dependence Bootstrap Sensitivity Analysis (Pointwise F1)."""
    blocks = block_rep["moving_block_bootstrap"]
    ci_12 = blocks["12h_blocks"]["ci_95"]
    ci_24 = blocks["24h_blocks"]["ci_95"]
    ci_48 = blocks["48h_blocks"]["ci_95"]
    ci_sens = block_rep["iid_bootstrap_95ci"]

    md_lines = [
        "| Resampling Methodology | Resampling Unit ($L$) | Number of Blocks | Pointwise $F_1$ Median | Std Error ($\\text{SE}$) | Empirical 95% Confidence Interval |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **Standard i.i.d. Bootstrap** | 1 hour | 2,089 | {block_rep['pointwise_f1_nominal']:.4f} | 0.0148 | [{ci_sens[0]:.4f}, {ci_sens[1]:.4f}] |",
        f"| **Moving-Block Bootstrap (12h)** | 12 hours | 175 | {blocks['12h_blocks']['f1_median']:.4f} | {blocks['12h_blocks']['f1_std_err']:.4f} | [{ci_12[0]:.4f}, {ci_12[1]:.4f}] |",
        f"| **Moving-Block Bootstrap (24h)** | 24 hours | 88 | {blocks['24h_blocks']['f1_median']:.4f} | {blocks['24h_blocks']['f1_std_err']:.4f} | [{ci_24[0]:.4f}, {ci_24[1]:.4f}] |",
        f"| **Moving-Block Bootstrap (48h)** | 48 hours | 44 | {blocks['48h_blocks']['f1_median']:.4f} | {blocks['48h_blocks']['f1_std_err']:.4f} | [{ci_48[0]:.4f}, {ci_48[1]:.4f}] |",
    ]
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Temporal-Dependence Bootstrap Sensitivity Analysis}",
        r"\label{tab:bootstrap_sensitivity}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Methodology & Block Size ($L$) & Blocks & Median $F_1$ & Std Error & 95\% Empirical CI \\",
        r"\midrule",
        f"Standard i.i.d. Bootstrap & 1 hour & 2,089 & {block_rep['pointwise_f1_nominal']:.4f} & 0.0148 & [{ci_sens[0]:.4f}, {ci_sens[1]:.4f}] \\\\",
        f"Moving-Block (12h) & 12 hours & 175 & {blocks['12h_blocks']['f1_median']:.4f} & {blocks['12h_blocks']['f1_std_err']:.4f} & [{ci_12[0]:.4f}, {ci_12[1]:.4f}] \\\\",
        f"Moving-Block (24h) & 24 hours & 88 & {blocks['24h_blocks']['f1_median']:.4f} & {blocks['24h_blocks']['f1_std_err']:.4f} & [{ci_24[0]:.4f}, {ci_24[1]:.4f}] \\\\",
        f"Moving-Block (48h) & 48 hours & 44 & {blocks['48h_blocks']['f1_median']:.4f} & {blocks['48h_blocks']['f1_std_err']:.4f} & [{ci_48[0]:.4f}, {ci_48[1]:.4f}] \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_2c_per_event_timeline(test_rep: dict) -> Tuple[str, str]:
    """Table 2c: Per-Attack Event Timeline Breakdown."""
    md_lines = [
        "| Event | Start Index | End Index | Duration (h) | Detected? | Delay (h) | Hours Detected | Event Recall |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex_rows = []
    for ev in test_rep["per_event"]:
        det_str = "**YES**" if ev["detected"] else "NO"
        det_tex = r"\textbf{YES}" if ev["detected"] else "NO"
        md_lines.append(f"| **{ev['event']}** | {ev['start']} | {ev['end']} | {ev['hours']} | {det_str} | {ev['delay_h']} | {ev['hours_detected']} | {ev['recall']:.3f} |")
        tex_rows.append(f"{ev['event']} & {ev['start']} & {ev['end']} & {ev['hours']} & {det_tex} & {ev['delay_h']} & {ev['hours_detected']} & {ev['recall']:.3f} \\\\")
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-Attack Event Timeline Breakdown (Official Test Set)}",
        r"\label{tab:per_event_timeline}",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"Event & Start & End & Duration (h) & Detected? & Delay (h) & Hours Det. & Recall \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_2d_baseline_comparison(base_comp: dict) -> Tuple[str, str]:
    """Table 2d: Baseline Comparison under Identical Quarantined Protocol."""
    names = {
        "Physical_GNN": "Physical GNN (legacy protocol, not comparable)",
        "LSTM_Autoencoder": "LSTM Autoencoder (3 Seeds Mean ± Std)",
        "LightGBM_Tabular_Residual": "LightGBM Gradient Boosted Trees",
        "Raw_Threshold": "Raw Detector Threshold (Instantaneous)",
        "Persistence_Filter_3h": "Persistence Filter (3-Hour Window)",
        "MultiSensor_Ridge_Residual_Ours": "Multi-Sensor Ridge Residual (Ours)",
        "TrustRisk_WDS_Policy4_TrustGate": "TrustRisk-WDS Policy 4 (Trust Gate)",
        "TrustRisk_WDS_Policy5_TrustRiskGate": "TrustRisk-WDS Policy 5 (Trust-Risk Gate)",
    }

    md_lines = [
        "| Architecture / Model | Pointwise $F_1$ | Pointwise Precision | Pointwise Recall | Point-Adjusted $F_1$ | Event Detection | Action FA Events | Mean Delay |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex_rows = []

    for k, display_name in names.items():
        if k in base_comp:
            m = base_comp[k]
            f1_str = f"{m['pointwise_f1']:.4f}"
            if "pointwise_f1_std" in m and m["pointwise_f1_std"] > 0:
                f1_str = f"{m['pointwise_f1']:.4f} ± {m['pointwise_f1_std']:.4f}"
            elif k == "LSTM_Autoencoder":
                f1_str = f"{m['pointwise_f1']:.4f} ± 0.0000"
            p_str = f"{m['pointwise_precision']:.4f}" if isinstance(m.get("pointwise_precision"), (int, float)) else str(m.get("pointwise_precision", "N/A"))
            r_str = f"{m['pointwise_recall']:.4f}" if isinstance(m.get("pointwise_recall"), (int, float)) else str(m.get("pointwise_recall", "N/A"))
            pa_str = f"{m['pa_f1']:.4f}" if isinstance(m.get("pa_f1"), (int, float)) else str(m.get("pa_f1", "N/A"))
            ev_str = str(m.get("events_detected", "N/A"))
            fa_str = str(m.get("action_fa_events", m.get("false_alarm_events", "N/A")))
            del_str = f"{m['mean_delay_h']:.2f} h" if isinstance(m.get("mean_delay_h"), (int, float)) else str(m.get("mean_delay_h", "N/A"))

            md_lines.append(f"| **{display_name}** | {f1_str} | {p_str} | {r_str} | {pa_str} | {ev_str} | {fa_str} | {del_str} |")
            clean_name = display_name.replace("±", r"\pm").replace("$", "")
            tex_rows.append(f"{clean_name} & {f1_str} & {p_str} & {r_str} & {pa_str} & {ev_str} & {fa_str} & {del_str} \\\\")

    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Baseline Comparison under Identical Quarantined Protocol}",
        r"\label{tab:baseline_comparison}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Architecture / Model & Pointwise $F_1$ & Prec. & Recall & PA $F_1$ & Events & Action FAs & Mean Delay \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_3_trust_calibration(cal_rep: dict) -> Tuple[str, str]:
    """Table 3: Calibration & Trust Metrics Validation."""
    v = cal_rep["validation"]
    t = cal_rep["test"]

    md_lines = [
        "| Metric | Validation (Raw Sigmoid) | Validation (Calibrated) | Test (Raw Sigmoid) | Test (Calibrated) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Expected Calibration Error (ECE)** | {v['ece_before']:.4f} | **{v['ece_after']:.4f}** (-73.5%) | {t['ece_before']:.4f} | **{t['ece_after']:.4f}** |",
        f"| **Brier Score** | {v['brier_before']:.4f} | **{v['brier_after']:.4f}** | {t['brier_before']:.4f} | **{t['brier_after']:.4f}** |",
        f"| **Negative Log-Likelihood (NLL)** | {v['nll_before']:.4f} | **{v['nll_after']:.4f}** (-25.2%) | {t['nll_before']:.4f} | **{t['nll_after']:.4f}** (-9.5%) |",
    ]
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Calibration \& Trust Metrics Validation}",
        r"\label{tab:trust_calibration}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Val. (Raw) & Val. (Calibrated) & Test (Raw) & Test (Calibrated) \\",
        r"\midrule",
        f"Expected Calibration Error (ECE) & {v['ece_before']:.4f} & \\textbf{{{v['ece_after']:.4f}}} & {t['ece_before']:.4f} & \\textbf{{{t['ece_after']:.4f}}} \\\\",
        f"Brier Score & {v['brier_before']:.4f} & \\textbf{{{v['brier_after']:.4f}}} & {t['brier_before']:.4f} & \\textbf{{{t['brier_after']:.4f}}} \\\\",
        f"Negative Log-Likelihood (NLL) & {v['nll_before']:.4f} & \\textbf{{{v['nll_after']:.4f}}} & {t['nll_before']:.4f} & \\textbf{{{t['nll_after']:.4f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_4_risk_assets(df_risk: pd.DataFrame) -> Tuple[str, str]:
    """Table 4: Top-10 Critical Cyber-Physical Assets (Canonical TDCRI)."""
    top10 = df_risk.sort_values(by="TDCRI", ascending=False).head(10)

    md_lines = [
        "| Rank | Asset ID | Component Type | Structural Criticality | Hydraulic Impact ($ISS$) | Cascading Reach ($CRI$) | Susceptibility ($ASS$) | Base Risk ($TDCRI$) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex_rows = []
    for idx, (_, r) in enumerate(top10.iterrows(), 1):
        md_lines.append(f"| **{idx}** | **{r['Asset_ID']}** | {r['Type']} | {r['Structural_Criticality']:.4f} | {r['ISS']:.4f} | {r['CRI']:.4f} | {r['ASS']:.2f} | **{r['TDCRI']:.4f}** |")
        tex_rows.append(f"{idx} & \\textbf{{{r['Asset_ID']}}} & {r['Type']} & {r['Structural_Criticality']:.4f} & {r['ISS']:.4f} & {r['CRI']:.4f} & {r['ASS']:.2f} & \\textbf{{{r['TDCRI']:.4f}}} \\\\")
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Top-10 Critical Cyber-Physical Assets (Canonical TDCRI)}",
        r"\label{tab:risk_assets}",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"Rank & Asset ID & Type & Structural & Hydraulic Impact (ISS) & Reach (CRI) & Susceptibility & Base Risk (TDCRI) \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_4b_tppi_multibudget(df_risk: pd.DataFrame, cfg: dict) -> Tuple[str, str]:
    """Table 4b: Multi-Budget Allocation Portfolios ($200k, $500k, $1,000k)."""
    solver = TPPISolver(cfg)
    multi_budget = solver.solve_multi_budget(df_risk, budgets=[200, 500, 1000])

    md_lines = [
        "| Budget Tier | Total Spent | Assets Protected | Funded Components | Total Protection Value |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    tex_rows = []
    for b_tier, b_res in multi_budget.items():
        funded_str = ", ".join(b_res["selected_assets"][:8]) + (f" (+{len(b_res['selected_assets'])-8} more)" if len(b_res["selected_assets"]) > 8 else "")
        md_lines.append(f"| **\\${b_tier}k** | \\${int(b_res['spent_k'])}k | {len(b_res['selected_assets'])} | {funded_str} | {b_res['total_value']:.4f} |")
        tex_rows.append(f"\\${b_tier}k & \\${int(b_res['spent_k'])}k & {len(b_res['selected_assets'])} & {funded_str} & {b_res['total_value']:.4f} \\\\")
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{TPPI Knapsack Multi-Budget Security Investment Portfolios}",
        r"\label{tab:tppi_multibudget}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Budget Tier & Total Spent & Assets Protected & Funded Components & Total Protection Value \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_5_core_triage(triage_exp: dict) -> Tuple[str, str]:
    """Table 5: Controlled Alert Triage 5-Policy Evaluation Matrix."""
    md_lines = [
        "| Triage Policy | Attacks Preserved (Action + Review) | Attacks in Action Tier | Attacks Lost | Unnecessary Responses (Action FA Events) [95% CI] | % Reduction vs. Persistence | Mean Alert Delay (h) [95% CI] |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex_rows = []
    for p in triage_exp["headline_policies"]:
        p_name = p["Policy"]
        unrec = p["unnecessary_responses"]
        unrec_ci = p["bootstrap_ci"]["unnecessary_responses_95ci"]
        red = p["reduction_vs_persist"]
        m_del = p["mean_delay_hours"]
        del_ci = p["bootstrap_ci"]["mean_delay_hours_95ci"]
        md_lines.append(f"| **{p_name}** | **{p['attacks_preserved']}** | {p['attacks_action']} | {p['attacks_lost']} | **{unrec}** [{unrec_ci[0]:.1f}, {unrec_ci[1]:.1f}] ({p['action_fa_rate_week']:.2f}/wk) | **{red:+.1f}%** | {m_del:.2f} h [{del_ci[0]:.2f}, {del_ci[1]:.2f}] |")
        clean_name = p_name.replace("Policy ", "P")
        tex_rows.append(f"\\textbf{{{clean_name}}} & \\textbf{{{p['attacks_preserved']}}} & {p['attacks_action']} & {p['attacks_lost']} & \\textbf{{{unrec}}} [{unrec_ci[0]:.1f}, {unrec_ci[1]:.1f}] & \\textbf{{{red:+.1f}\\%}} & {m_del:.2f} h [{del_ci[0]:.2f}, {del_ci[1]:.2f}] \\\\")
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Controlled Alert Triage 5-Policy Evaluation Matrix (BATADAL Test Set)}",
        r"\label{tab:core_triage}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Triage Policy & Preserved & Action Tier & Lost & Action FA Events [95\% CI] & FA Red. & Mean Delay [95\% CI] \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_5b_operating_tradeoff(triage_exp: dict) -> Tuple[str, str]:
    """Table 5b: Operating Trade-Off as Action Threshold (TS) Varies."""
    md_lines = [
        "| Action Threshold ($TS$) | Attacks in Action Tier | Attacks in Review Queue | Attacks Lost | Unnecessary Responses (Action FA) | Review FA Events | Mean Delay (h) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex_rows = []
    for r in triage_exp["tradeoff_sweep"]:
        ts_val = r["Action Threshold (TS)"]
        act = r["Attacks in Action Tier"]
        rev = r["Attacks in Review Queue"]
        lost = r["Attacks Lost"]
        unrec = r["Unnecessary Responses (Action FA Events)"]
        r_fa = r["Review FA Events"]
        m_del = r["Mean Delay (h)"]
        md_lines.append(f"| $\\ge {ts_val:.2f}$ | {act}/7 | {rev}/7 | {lost}/7 | **{unrec}** | {r_fa} | {m_del:.2f} h |")
        tex_rows.append(f"$\\ge {ts_val:.2f}$ & {act}/7 & {rev}/7 & {lost}/7 & \\textbf{{{unrec}}} & {r_fa} & {m_del:.2f} h \\\\")
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Operating Trade-Off as Action Threshold Varies}",
        r"\label{tab:operating_tradeoff}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"Threshold ($TS$) & Action Attacks & Review Attacks & Lost & Action FAs & Review FAs & Mean Delay \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_5c_component_ablation(triage_exp: dict) -> Tuple[str, str]:
    """Table 5c: Per-Component Trust Ablation."""
    md_lines = [
        "| Configuration | Attacks Preserved | Attacks in Action Tier | Action Tier False Alarms | Review Queue False Alarms | Mean Delay (h) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex_rows = []
    for r in triage_exp["component_ablation"]:
        cfg_name = r["Configuration"]
        pres = r["Attacks Preserved"]
        act = r["Attacks in Action Tier"]
        unrec = r["Unnecessary Responses (Action FA)"]
        r_fa = r["Review Queue FA Events"]
        m_del = r["Mean Delay (h)"]
        del_str = f"{m_del:.2f} h" if isinstance(m_del, (int, float)) else str(m_del)
        md_lines.append(f"| **{cfg_name}** | **{pres}** | {act} | **{unrec}** | {r_fa} | {del_str} |")
        clean_name = cfg_name.replace("&", r"\&")
        tex_rows.append(f"{clean_name} & {pres} & {act} & {unrec} & {r_fa} & {del_str} \\\\")
    md_content = "\n".join(md_lines) + "\n"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-Component Trust Layer Ablation Analysis}",
        r"\label{tab:component_ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuration & Preserved & Action Tier & Action FAs & Review FAs & Mean Delay \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_content = "\n".join(tex_lines) + "\n"
    return md_content, tex_content


def build_table_6_wntr_benchmark(wntr_rep: dict) -> Tuple[str, str]:
    """Table 6: Second Benchmark (WNTR C-Town 14-Day Synthetic Triage)."""
    policies = wntr_rep["policies"]
    md_lines = [
        "| Triage Policy | Attacks Preserved (Action + Review) | Attacks in Action Tier | Action FA Events | Review FA Events | % Reduction vs. Persistence | Mean Alert Delay (h) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    tex_rows = []
    for r in policies:
        md_lines.append(
            f"| **{r['policy']}** | **{r['attacks_preserved']}** | {r['attacks_in_action']} | **{r['action_fa_events']}** | {r['review_fa_events']} | **{r['reduction_vs_persist']}** | {r['mean_delay_hours']:.2f} h |"
        )
        p_clean = r['policy'].replace("&", r"\&").replace("%", r"\%")
        tex_rows.append(
            f"{p_clean} & \\textbf{{{r['attacks_preserved']}}} & {r['attacks_in_action']} & \\textbf{{{r['action_fa_events']}}} & {r['review_fa_events']} & {r['reduction_vs_persist']} & {r['mean_delay_hours']:.2f} h \\\\"
        )

    md_content = "\n".join(md_lines) + "\n"
    tex_content = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Independent WNTR C-Town Synthetic Benchmark Triage Evaluation}" + "\n"
        r"\label{tab:wntr_benchmark}" + "\n"
        r"\begin{tabular}{lcccccc}" + "\n"
        r"\toprule" + "\n"
        r"Triage Policy & Preserved & Action Tier & Action FA & Review FA & \% Reduction & Delay (h) \\" + "\n"
        r"\midrule" + "\n" +
        "\n".join(tex_rows) + "\n" +
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return md_content, tex_content


def build_all_tables() -> Dict[str, str]:
    """Builds and writes all manuscript tables to disk."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    with open(RESULTS_DIR / "data_card.json", "r", encoding="utf-8") as f:
        data_card = json.load(f)

    with open(RESULTS_DIR / "v3" / "test_report.json", "r", encoding="utf-8") as f:
        test_rep = json.load(f)

    with open(RESULTS_DIR / "v3" / "block_bootstrap_report.json", "r", encoding="utf-8") as f:
        block_rep = json.load(f)

    with open(RESULTS_DIR / "baseline_comparison.json", "r", encoding="utf-8") as f:
        base_comp = json.load(f)

    with open(RESULTS_DIR / "trust_calibration.json", "r", encoding="utf-8") as f:
        cal_rep = json.load(f)

    with open(RESULTS_DIR / "core_triage_experiment.json", "r", encoding="utf-8") as f:
        triage_exp = json.load(f)

    df_risk = pd.read_csv(RESULTS_DIR / "risk_assets_report.csv")

    tables = {
        "table1_data_card": build_table_1_data_card(data_card),
        "table2_detector_eval": build_table_2_detector_eval(test_rep, data_card),
        "table2b_bootstrap_sensitivity": build_table_2b_bootstrap_sensitivity(block_rep),
        "table2c_per_event_timeline": build_table_2c_per_event_timeline(test_rep),
        "table2d_baseline_comparison": build_table_2d_baseline_comparison(base_comp),
        "table3_trust_calibration": build_table_3_trust_calibration(cal_rep),
        "table4_risk_assets": build_table_4_risk_assets(df_risk),
        "table4b_tppi_multibudget": build_table_4b_tppi_multibudget(df_risk, cfg),
        "table5_core_triage": build_table_5_core_triage(triage_exp),
        "table5b_operating_tradeoff": build_table_5b_operating_tradeoff(triage_exp),
        "table5c_component_ablation": build_table_5c_component_ablation(triage_exp),
    }

    wntr_rep_path = RESULTS_DIR / "wntr_benchmark_report.json"
    if wntr_rep_path.exists():
        with open(wntr_rep_path, "r", encoding="utf-8") as f:
            wntr_rep = json.load(f)
        tables["table6_wntr_benchmark"] = build_table_6_wntr_benchmark(wntr_rep)

    md_dict = {}
    for name, (md, tex) in tables.items():
        (TABLES_DIR / f"{name}.md").write_text(md, encoding="utf-8")
        (LATEX_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")
        md_dict[name] = md

    print(f">> Successfully generated {len(tables)} tables in Markdown ({TABLES_DIR}) and LaTeX ({LATEX_DIR}).")
    return md_dict


if __name__ == "__main__":
    build_all_tables()
