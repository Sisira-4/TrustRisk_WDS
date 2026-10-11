# TrustRisk-WDS Benchmark Data Card

Summary of dataset splits, label semantics, and masking protocol:

| Dataset Split                         |   Total Hours (Rows) |   Confirmed Normal (0) |   Unlabelled (-999) |   Attack Hours (1) |   Attack Events |   Margin Masked (±48h) |
|:--------------------------------------|---------------------:|-----------------------:|--------------------:|-------------------:|----------------:|-----------------------:|
| Training (BATADAL Dataset 03)         |                 8761 |                   8761 |                   0 |                  0 |               0 |                      0 |
| Cross-Validation (BATADAL Dataset 04) |                 4177 |                      0 |                3958 |                219 |               5 |                    480 |
| Test (BATADAL Test Labeled)           |                 2089 |                   1682 |                   0 |                407 |               7 |                      0 |

### Label Semantics & Masking Rules:
- `0`: Confirmed normal operational period.
- `-999`: **Unlabelled** data (NOT confirmed normal). In cross-validation, alarms in -999 are termed 'unexplained alarms'.
- `1`: Ground-truth cyber-physical attack.
- `±48h Margins`: Temporal buffer surrounding each attack event in Dataset 04 to prevent transitional contamination.
