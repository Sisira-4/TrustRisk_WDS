# Dataset 04 Triage Cross-Validation Report (Task 3.1)

**Protocol**: Leave-One-Attack-Event-Out cross-validation across all 5 labelled attack events on Dataset 04.  
**Search Objective**: Pre-committed objective rule:
$$\text{Attacks Preserved} = 5/5 \implies \min(\text{Action FA Events}) \implies \min(\text{Mean Alert Delay})$$

---

## 1. Selected Pareto-Optimal Parameters
- **Trust Score Weights (3-term)**: $w_{\text{pcs}} = 0.30$, $w_{\text{uqs}} = 0.40$, $w_{\text{ers}} = 0.30$
- **Action Cutoff**: $TS \ge 0.85$ (Review queue: $0.50 \le TS < 0.85$)
- **Policy 3 PCS Cutoff**: $\text{PCS} \ge 0.85$
- **Validation Score**: Attacks Preserved: **5/5**, Action FA Events: **4**, Mean Delay: **14.6h**

---

## 2. Policy 3 Confidence-Only Cutoff CV Comparison
| policy   |   pcs_cutoff | attacks_preserved   |   attacks_lost |   action_fa_events |   mean_delay_hours |
|:---------|-------------:|:--------------------|---------------:|-------------------:|-------------------:|
| Policy 3 |         0.7  | 5/5                 |              0 |                 13 |                9.4 |
| Policy 3 |         0.75 | 5/5                 |              0 |                 13 |                9.4 |
| Policy 3 |         0.8  | 5/5                 |              0 |                 13 |                9.4 |
| Policy 3 |         0.85 | 5/5                 |              0 |                 13 |                9.4 |
| Policy 3 |         0.9  | 5/5                 |              0 |                 15 |                9.4 |

---

## 3. Top-10 Pareto Candidates (Policy 4 Trust-Gated Triage)
|   w_pcs |   w_uqs |   w_ers |   action_cutoff |   review_cutoff | attacks_preserved   |   attacks_lost |   action_fa_events |   review_fa_events |   mean_delay_hours |
|--------:|--------:|--------:|----------------:|----------------:|:--------------------|---------------:|-------------------:|-------------------:|-------------------:|
|    0.3  |    0.4  |    0.3  |            0.85 |             0.5 | 5/5                 |              0 |                  4 |                 16 |               14.6 |
|    0.35 |    0.4  |    0.25 |            0.85 |             0.5 | 5/5                 |              0 |                  4 |                 15 |               15.2 |
|    0.35 |    0.35 |    0.3  |            0.85 |             0.5 | 5/5                 |              0 |                  5 |                 15 |               13.4 |
|    0.3  |    0.35 |    0.35 |            0.85 |             0.5 | 5/5                 |              0 |                  5 |                 17 |               13.4 |
|    0.5  |    0.35 |    0.15 |            0.85 |             0.5 | 5/5                 |              0 |                  5 |                 11 |               14   |
|    0.45 |    0.35 |    0.2  |            0.85 |             0.5 | 5/5                 |              0 |                  5 |                 11 |               14   |
|    0.4  |    0.35 |    0.25 |            0.85 |             0.5 | 5/5                 |              0 |                  5 |                 13 |               14   |
|    0.55 |    0.4  |    0.05 |            0.85 |             0.5 | 5/5                 |              0 |                  5 |                  8 |               15   |
|    0.5  |    0.4  |    0.1  |            0.85 |             0.5 | 5/5                 |              0 |                  5 |                  8 |               15   |
|    0.45 |    0.4  |    0.15 |            0.85 |             0.5 | 5/5                 |              0 |                  5 |                  9 |               15   |
