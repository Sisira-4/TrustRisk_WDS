# Legacy Exploratory Prototypes & Superseded Checkpoints

This directory houses exploratory research artifacts, historical model checkpoints, and experimental sweep scripts developed during the initial architectural phases of the **TrustRisk-WDS** project.

---

## 1. Context & Motivation

During early project design, a spatiotemporal Graph Neural Network coupled with Gated Recurrent Units (**Physical-GNN-GRU**) was investigated for anomaly detection over the C-Town SCADA sensor topology.

However, comprehensive empirical benchmarking under strict leakage-free protocols demonstrated that:
1. **Multi-Sensor Ridge Residual Outperformance**: A simpler, physics-informed multi-sensor Ridge residual detector achieved vastly superior anomaly detection performance (**Pointwise $F_1 = 0.8368$**, Precision = 0.9677, Recall = 0.7371) compared to the Physical-GNN (**Pointwise $F_1 = 0.2282$**, Precision = 0.7333, Recall = 0.1351).
2. **True Framework Contribution**: The core scientific contribution of TrustRisk-WDS lies in the downstream **calibrated trust triage layer** and the **physics-grounded hydraulic risk layer (WNTR)** that eliminates operational alarm fatigue, rather than deep neural network architectures for time-series forecasting.

To maintain clarity and show a single authoritative pipeline in the primary repository, all superseded GNN models, checkpoints, and preliminary hyperparameter sweep scripts were relocated to this `legacy/` folder.

---

## 2. Directory Structure

```text
legacy/
├── models/
│   └── gnn_gru/                     # Superseded PyTorch Geometric GNN-GRU code
│       ├── physical_dataset.py       # Graph dataset builder for C-Town topology
│       ├── physical_gnn_gru_model_C.pth # Trained Physical-GNN checkpoint (evaluated in Table 2d baseline)
│       ├── sensor_dataset.py         # Sensor-level graph dataset
│       ├── train_gnn_gru_physical.py # Spatiotemporal GNN-GRU training loop
│       └── ...                       # Additional variant checkpoints & tuning scripts
└── experiments/
    └── experimental/                # Historical preliminary sweep & audit scripts
        ├── gnn_gru_experiments/     # Preliminary window sweeps (W12, W24, W36, W48)
        ├── exp_A_class_weight.py    # Class weighting trials
        ├── exp_E1_masked_mean.py    # Masked mean aggregation trials
        └── phase_6K_temporal_sweep.py # Temporal window evaluations
```

---

## 3. Baseline Auditability

The canonical checkpoint `legacy/models/gnn_gru/physical_gnn_gru_model_C.pth` is maintained and evaluated by `experiments/run_baseline_comparison.py` to ensure that Table 2d in the publication honestly documents the Physical-GNN baseline comparison directly from its live model weights.
