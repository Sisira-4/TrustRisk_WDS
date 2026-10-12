import streamlit as st
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import plotly.graph_objects as go
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.trust.policy5_router import (
    load_risk_and_mapping,
    get_sensor_physical_risk,
    compute_alert_physical_risk,
    evaluate_policy_5_decision
)

CLEAN_PIPE_PATH = ROOT / "results" / "v3" / "trust_pipeline_clean.csv"
RISK_REPORT_PATH = ROOT / "results" / "risk_assets_report.csv"
TRIAGE_PATH = ROOT / "results" / "core_triage_experiment.json"
GRAPH_PATH = ROOT / "network" / "graph" / "ctown_graph.pkl"
MAPPING_PATH = ROOT / "network" / "mappings" / "batadal_epanet_mapping.csv"

INP_PATH = ROOT / "network" / "epanet" / "CTOWN.INP"

st.set_page_config(page_title="TrustRisk-WDS SCADA", layout="wide", page_icon="🛡️")

# Custom CSS for Dark Modern Theme
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .kpi-card { background-color: #1E2130; padding: 20px; border-radius: 10px; border-left: 5px solid #00F0FF; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .kpi-title { font-size: 13px; color: #A0AEC0; text-transform: uppercase; font-weight: bold; }
    .kpi-value { font-size: 28px; font-weight: 800; color: #FFFFFF; }
    .kpi-alert { border-left: 5px solid #FF003C !important; }
    .kpi-review { border-left: 5px solid #FFB800 !important; }
    .tier-badge-action { background-color: #FF003C; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .tier-badge-review { background-color: #FFB800; color: black; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .tier-badge-monitor { background-color: #00F0FF; color: black; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .warning-box { background-color: #3d2600; border: 1px solid #FFB800; padding: 12px; border-radius: 6px; color: #FFE082; margin-bottom: 12px; }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not CLEAN_PIPE_PATH.exists():
        raise FileNotFoundError(f"Clean pipeline file not found at {CLEAN_PIPE_PATH}. Run 'python reproduce_all.py' first.")
    if not RISK_REPORT_PATH.exists():
        raise FileNotFoundError(f"Risk report file not found at {RISK_REPORT_PATH}. Run 'python reproduce_all.py' first.")
    if not TRIAGE_PATH.exists():
        raise FileNotFoundError(f"Core triage results not found at {TRIAGE_PATH}. Run 'python experiments/run_core_triage_experiment.py' first.")

    df_pipe = pd.read_csv(CLEAN_PIPE_PATH)
    df_risk = pd.read_csv(RISK_REPORT_PATH)
    with open(TRIAGE_PATH, "r") as f:
        triage_meta = json.load(f)
    s2a, a2r = load_risk_and_mapping(MAPPING_PATH, RISK_REPORT_PATH)
    return df_pipe, df_risk, triage_meta, s2a, a2r


@st.cache_resource
def load_graph():
    if GRAPH_PATH.exists():
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
    elif INP_PATH.exists():
        import wntr
        wn = wntr.network.WaterNetworkModel(str(INP_PATH))
        G = wn.to_graph().to_undirected()
    else:
        raise FileNotFoundError(f"Neither {GRAPH_PATH} nor {INP_PATH} was found. Cannot load physical water topology.")
    return G


pipe_df, risk_df, triage_meta, s2a, a2r = load_data()
G = load_graph()

# ================= SIDEBAR CONTROLS =================
st.sidebar.title("🛡️ TrustRisk-WDS SCADA")
st.sidebar.markdown("**Decision Support & Triage Console**")

# Operational Safety Mode: Default to Operational SCADA (No oracle / y_true labels)
dashboard_mode = st.sidebar.radio(
    "Console Operating Mode",
    ["Operational SCADA View", "Offline Evaluation / Forensic Mode"],
    index=0,
    help="Operational SCADA View isolates decision makers from oracle ground-truth labels. Forensic Mode is for offline audit only."
)

max_t = len(pipe_df) - 1
t_idx = st.sidebar.slider("Timeline Step (Stream Hour)", min_value=0, max_value=max_t, value=320, step=1)

current_row = pipe_df.iloc[t_idx]
timestamp = current_row["DATETIME"]
p_attack = float(current_row["p_attack"])
alarm_persist = int(current_row["alarm_persist"])
ts = float(current_row["TS"])
pcs = float(current_row["PCS"])
uqs = float(current_row["UQS"])
ers = float(current_row["ERS"])
ati_band = current_row["ATI_band"]

# Extract top anomalous sensors for the alert and map to physical asset
top_sensors = [current_row.get("top1_sensor", "L_T1"), current_row.get("top2_sensor", ""), current_row.get("top3_sensor", "")]
top_sensors = [s for s in top_sensors if isinstance(s, str) and s]
primary_asset, alert_base_risk = compute_alert_physical_risk(top_sensors, s2a, a2r)

# Shared Policy 5 Triage Routing: Evaluated strictly through models.trust.policy5_router
policy_choice = st.sidebar.selectbox(
    "Triage Routing Policy",
    ["Policy 5: Trust + Physical Risk Gate (Canonical)", "Policy 4: Trust Gate Only (ATI Bands)"],
    index=0
)

if policy_choice.startswith("Policy 5"):
    decision = evaluate_policy_5_decision(alarm_persist, ts, alert_base_risk)
else:
    # Policy 4: Action >= 0.80, Review in [0.50, 0.80), Monitor < 0.50
    if alarm_persist == 1:
        if ts >= 0.80:
            decision = "ACTION"
        elif ts >= 0.50:
            decision = "REVIEW"
        else:
            decision = "MONITOR"
    else:
        decision = "MONITOR"

if decision == "ACTION":
    tier_html = '<span class="tier-badge-action">🚨 IMMEDIATE ACTION</span>'
elif decision == "REVIEW":
    tier_html = '<span class="tier-badge-review">👁️ HUMAN REVIEW</span>'
else:
    tier_html = '<span class="tier-badge-monitor">📊 MONITOR / SUPPRESSED</span>'

# Operational SCADA Safety Guard: Ground truth ONLY visible in Forensic Mode
st.sidebar.markdown(f"**Timestamp:** `{timestamp}`")
st.sidebar.markdown(f"**Detector Status:** {'⚠️ Persistent Alarm' if alarm_persist == 1 else 'Normal Telemetry'}")
st.sidebar.markdown(f"**Primary Mapped Asset:** `{primary_asset}` (Base Risk = {alert_base_risk:.3f})")

if dashboard_mode == "Offline Evaluation / Forensic Mode":
    true_label = int(current_row["y_true"])
    st.sidebar.markdown(
        f"""<div class="warning-box">
        <strong>⚠️ FORENSIC ORACLE ENABLED:</strong><br>
        Ground Truth: {'🔴 ATTACK (1)' if true_label == 1 else '🟢 NORMAL (0)'}<br>
        <em>For post-incident audit only. Never enabled in operational SCADA.</em>
        </div>""",
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown("🔒 *Ground-truth labels hidden in operational SCADA mode to prevent oracle bias.*")

st.sidebar.markdown(f"**Triage Tier:** {tier_html}", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("💡 *Routing driven strictly by `models.trust.policy5_router`. Verified zero synthetic multipliers.*")

# ================= MAIN KPI HEADER =================
st.title("🛡️ TrustRisk-WDS: Trust-Aware Alert Validation & Cyber-Physical Risk Console")
st.markdown(f"Operational SCADA Telemetry at **{timestamp}** | Active Triage State: {tier_html}", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
is_action = (decision == "ACTION")
is_review = (decision == "REVIEW")
card_style = "kpi-alert" if is_action else ("kpi-review" if is_review else "")

# Calculate live cyber-physical risk (TDCRI) strictly using the mapped alert asset Base Risk
tdcri_live = alert_base_risk * p_attack * ts

with kpi1:
    st.markdown(f'<div class="kpi-card {card_style}"><div class="kpi-title">Calibrated Threat (PCS)</div><div class="kpi-value">{pcs*100:.1f}%</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Epistemic Uncertainty (UQS)</div><div class="kpi-value">{uqs:.3f}</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Explanation Reliability (ERS)</div><div class="kpi-value">{ers:.3f}</div></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="kpi-card {card_style}"><div class="kpi-title">Unified Trust Score (TS)</div><div class="kpi-value">{ts:.3f}</div></div>', unsafe_allow_html=True)
with kpi5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Alert Dynamic Risk ({primary_asset})</div><div class="kpi-value">{tdcri_live:.3f}</div></div>', unsafe_allow_html=True)

st.write("---")

# ================= TOPOLOGY & EXPLANATION PANELS =================
col_viz1, col_viz2 = st.columns([1.8, 1.2])

with col_viz1:
    st.subheader("🌐 C-Town Physical Topology & Hydraulic Criticality")
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, node_color, node_text = [], [], [], []
    top_sensor = current_row.get("top1_sensor", "L_T1")
    active_node = primary_asset if primary_asset not in ("UNKNOWN", "UNMAPPED") else "T1"

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Asset: {node}")
        if node == active_node and alarm_persist == 1:
            node_color.append("red" if ts >= 0.80 else "#FFB800")
        else:
            node_color.append("#00F0FF")

    fig_net = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6, color="#333"), hoverinfo="none", mode="lines"),
        go.Scatter(x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
                   marker=dict(size=7, color=node_color, line=dict(width=1, color="white")))
    ])
    fig_net.update_layout(showlegend=False, margin=dict(b=0, l=0, r=0, t=0), plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", height=380)
    fig_net.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig_net.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    st.plotly_chart(fig_net, use_container_width=True)

with col_viz2:
    st.subheader("🔍 Sensor Residual Z-Scores (Top Contributors)")
    z_cols = [c for c in pipe_df.columns if c.startswith("z_")]
    z_vals = current_row[z_cols].values.astype(float)
    top_5_idx = np.argsort(z_vals)[-5:]
    top_5_names = [z_cols[i].replace("z_", "") for i in top_5_idx]
    top_5_scores = z_vals[top_5_idx]

    bar_colors = ["red" if (val > 3.0 and alarm_persist == 1) else "#00F0FF" for val in top_5_scores]
    fig_bar = go.Figure(go.Bar(
        x=top_5_scores, y=top_5_names, orientation="h",
        marker_color=bar_colors
    ))
    fig_bar.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=240, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown(f"**Dominant Sensor Signal:** `{top_sensor}` (residual $z = {top_5_scores[-1]:.2f}$). Mapped element: **{active_node}**.")
    st.caption("⚠️ *Sensor-to-asset mapping reflects concentration of anomalous telemetry. Localized physical root cause requires manual verification.*")

st.write("---")

# ================= TPPI KNAPSACK ALLOCATION PANEL =================
st.subheader("💼 TPPI Security Investment Optimization (Budget Allocation)")
st.markdown("Ground-truth unit hardening costs and protection benefits from WNTR consequence modeling.")

tppi_col1, tppi_col2 = st.columns([1.5, 1])

with tppi_col1:
    budget_sel = st.slider("Budget Ceiling ($k)", min_value=100, max_value=800, value=500, step=50)
    
    # 0/1 Knapsack
    costs = risk_df["Unit_Cost_k"].values.astype(int)
    tppi_scores = (risk_df["Base_Risk"] * risk_df["Protection_Benefit"] * ts) / risk_df["Unit_Cost_k"]
    scaled_values = (tppi_scores.values * 1000).astype(int)
    n = len(risk_df)
    
    K = np.zeros((n + 1, budget_sel + 1), dtype=int)
    for i in range(1, n + 1):
        w = costs[i - 1]
        v = scaled_values[i - 1]
        for b in range(budget_sel + 1):
            if w <= b:
                K[i, b] = max(v + K[i - 1, b - w], K[i - 1, b])
            else:
                K[i, b] = K[i - 1, b]
                
    rem_b = budget_sel
    selected_idx = []
    for i in range(n, 0, -1):
        if K[i, rem_b] != K[i - 1, rem_b]:
            selected_idx.append(i - 1)
            rem_b -= costs[i - 1]
            
    df_selected = risk_df.iloc[selected_idx][["Asset_ID", "Type", "Base_Risk", "Unit_Cost_k", "Protection_Benefit"]]
    df_selected.columns = ["Asset ID", "Component Type", "Base Risk", "Unit Cost ($k)", "Protection Benefit"]
    st.dataframe(df_selected, use_container_width=True)

with tppi_col2:
    total_spent = int(df_selected["Unit Cost ($k)"].sum())
    st.metric("Total Hardening Capital Allocated", f"${total_spent}k / ${budget_sel}k")
    st.metric("Components Hardened", f"{len(df_selected)} Assets")
    st.metric("Current Trust Score Factor", f"{ts:.3f}")

st.write("---")

# ================= VERIFIED EXPERIMENTAL HEADLINE RESULTS =================
st.subheader("📈 Validated Experimental Benchmark Results (Phases 2 & 5)")
st.markdown("Metrics reproduce 1:1 from verified test reports (`results/frozen_detector_test_report.md` and `results/core_triage_experiment.md`).")

ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
ev_col1.metric("Pointwise Test F1", "0.8368", "95% CI: [0.808, 0.866]")
ev_col2.metric("Point-Adjusted (PA) F1", "0.9879", "100% Attack Event Recall")
ev_col3.metric("Test Events Detected", "7 / 7", "0 Missed Attacks")
ev_col4.metric("Action False Alarm Reduction", "100.0%", "0 Action FAs with ATI")
