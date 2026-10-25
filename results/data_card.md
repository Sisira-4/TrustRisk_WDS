# TrustRisk-WDS Benchmark Data Card

Summary of dataset splits, label semantics, and masking protocol:

| Dataset Split                         |   Total Hours (Rows) |   Confirmed Normal (0) |   Unlabelled (-999) |   Attack Hours (1) |   Attack Events |   Margin Masked (±48h) |
|:--------------------------------------|---------------------:|-----------------------:|--------------------:|-------------------:|----------------:|-----------------------:|
| Training (BATADAL Dataset 03)         |                 8761 |                   8761 |                   0 |                  0 |               0 |                      0 |
| Cross-Validation (BATADAL Dataset 04) |                 4177 |                      0 |                3958 |                219 |               5 |                    480 |
| Test (BATADAL Test Labeled)           |                 2089 |                   1682 |                   0 |                407 |               7 |                      0 |

### Label Semantics & Masking Policy:
1. **Provisional Normal Training Pool**: Unlabelled rows (`-999`) in Dataset 04 situated > 48h from any attack are used as a provisional normal training pool for residual regression fitting.
2. **Evaluation Semantics (Never Confirmed Negatives)**: `-999` rows are strictly treated as **unlabelled** in all evaluation protocols. Any alarms occurring during `-999` periods are termed *'unexplained alarms'* (or estimated false alarms), acknowledging unlabelled operational disturbances.
3. **Transition Masking (±48h Buffer)**: A strict 48-hour buffer before and after each labelled attack in Dataset 04 is masked (`MARGIN_MASK = 1`) to eliminate transitional hydraulic recovery periods from both training and evaluation.
