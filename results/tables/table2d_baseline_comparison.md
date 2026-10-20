| Architecture / Model | Pointwise $F_1$ | Pointwise Precision | Pointwise Recall | Point-Adjusted $F_1$ | Event Detection | Action FA Events | Mean Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Physical GNN (legacy protocol, not comparable)** | 0.2282 | 0.7333 | 0.1351 | 0.8365 | 5 / 7 | 5 | 31.29 h |
| **LSTM Autoencoder (3 Seeds Mean ± Std)** | 0.2306 ± 0.0000 | 0.6042 | 0.1425 | 0.2446 | 2 / 7 | 3 | 49.86 h |
| **LightGBM Gradient Boosted Trees** | 0.7116 | 0.9827 | 0.5577 | 0.9951 | 7 / 7 | 2 | 5.57 h |
| **Raw Detector Threshold (Instantaneous)** | 0.8115 | 0.9792 | 0.6929 | 0.9927 | 7 / 7 | 4 | 1.57 h |
| **Multi-Sensor Ridge Residual (Ours)** | 0.8368 | 0.9677 | 0.7371 | 0.9879 | 7 / 7 | 3 | 4.86 h |
| **TrustRisk-WDS Policy 4 (Trust Gate)** | 0.8368 | 0.9677 | 0.7371 | 0.9879 | 7 / 7 | 0 | 5.71 h |
| **TrustRisk-WDS Policy 5 (Trust-Risk Gate)** | 0.8368 | 0.9677 | 0.7371 | 0.9879 | 7 / 7 | 0 | 6.86 h |
