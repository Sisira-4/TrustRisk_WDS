"""Task 3.1: Systematic Leave-One-Attack-Event-Out Cross-Validation on Dataset 04.

PURPOSE:
Empirically selects Pareto-optimal Trust Score weights (w_pcs, w_uqs, w_ers), Action Triage Index
(ATI) action cutoff, and Policy 3 PCS cutoff on BATADAL_dataset04 validation folds alone,
without any exposure to the official test dataset.

PRE-COMMITTED SEARCH SPACE:
- w_pcs in {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
- w_uqs in {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}
- w_ers = round(1.0 - w_pcs - w_uqs, 4) >= 0.0
- Action Cutoff in {0.70, 0.75, 0.80, 0.85}
- Policy 3 PCS Cutoff in {0.70, 0.75, 0.80, 0.85, 0.90}

PRE-COMMITTED SELECTION RULE:
1. Zero Lost Attacks: Attacks Preserved == 5/5 across all held-out attack folds (Action + Review).
2. Minimal Operational False Alarms: Minimize false-alarm events in the Action tier on held-out -999 normal blocks.
3. Minimal Response Latency: Minimize mean alert delay (hours) across preserved attack events.
4. Deterministic Tie-Breaker: Prefer higher w_pcs (calibrated probability signal), then higher Action cutoff (safety margin).

Outputs:
- results/v3/triage_cv_report.json
- results/v3/triage_cv_report.csv
- results/v3/triage_cv_report.md
- trust_frozen_config.json
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3_residual.common import (
    load_series,
    sensor_columns,
    runs,
    margin_free_pool,
    persist,
    ResidualDetector,
)
from models.trust.unified_trust_engine import (
    PredictionConfidenceCalibrator,
    EpistemicUncertaintyQuantifier,
    ExplanationReliabilityEstimator,
)

RESULTS_V3 = ROOT / "results" / "v3"
RESULTS_V3.mkdir(parents=True, exist_ok=True)


def run_triage_cv(alpha=10.0, n_top=3, q=99.0, k_persist=3, n_persist=6):
    """Executes leave-one-attack-event-out CV across the 5 Dataset 04 folds."""
    d4_path = ROOT / "data" / "raw" / "BATADAL_dataset04.csv"
    d4 = load_series(d4_path)
    names = sensor_columns(d4)
    X = d4[names].values.astype(float)
    lab = d4["ATT_FLAG"].values
    T = len(d4)

    events = runs(lab == 1)
    pool = margin_free_pool(lab, margin=48)
    blocks = np.array_split(pool, len(events))
    assert len(events) == len(blocks) == 5, f"Expected 5 folds, got {len(events)}"

    idx = np.arange(T)
    folds_data = []

    print(f">> Starting Dataset 04 5-Fold Cross-Validation (Total hours: {T}, Attacks: {len(events)})...")

    for k in range(len(events)):
        es, ee = events[k]
        vb = blocks[k]
        gap = (idx >= vb.min() - 24) & (idx <= vb.max() + 24)
        held = np.zeros(T, bool)
        held[max(0, es - 48): ee + 49] = True
        train_idx = np.array([i for i in pool if not gap[i] and not held[i]])

        # 1. Fit detector strictly on train_idx
        det = ResidualDetector(alpha, n_top).fit(X, train_idx, names)
        thr = det.threshold(q)
        _, s = det.score(X)
        raw_al = (s > thr).astype(int)
        al = persist(raw_al, k_persist, n_persist)

        # 2. Fit Platt Calibrator strictly on non-held-out training data
        train_att_idx = [i for i in np.flatnonzero(lab == 1) if not (es <= i <= ee)]
        train_val_idx = np.concatenate([train_att_idx, train_idx])
        y_train_val = np.concatenate([
            np.ones(len(train_att_idx), dtype=int),
            np.zeros(len(train_idx), dtype=int)
        ])
        cal = PredictionConfidenceCalibrator().fit(s[train_val_idx], y_train_val)
        pcs = cal.predict_prob(s)

        # 3. Fit UQS ensemble strictly on X[train_idx]
        uq = EpistemicUncertaintyQuantifier(n_estimators=10, block_size=24).fit(X[train_idx])
        uqs = uq.compute_uqs(X)

        # 4. Compute ERS on z-matrix
        Z = ((X - det.mu) / det.sd)[:, det.keep]
        ers = ExplanationReliabilityEstimator().compute_ers(Z)

        folds_data.append({
            "fold": k,
            "es": es,
            "ee": ee,
            "vb": vb,
            "al_ev": al[es : ee + 1],
            "pcs_ev": pcs[es : ee + 1],
            "uqs_ev": uqs[es : ee + 1],
            "ers_ev": ers[es : ee + 1],
            "al_nb": al[vb],
            "pcs_nb": pcs[vb],
            "uqs_nb": uqs[vb],
            "ers_nb": ers[vb],
        })
        print(f"   Fold {k+1}/5 computed: held-out attack ({es}..{ee}), held-out normal block ({len(vb)}h)")

    # -------------------------------------------------------------
    # Grid Evaluation: Policy 3 PCS Cutoff Search
    # -------------------------------------------------------------
    print("\n>> Evaluating Policy 3 (Confidence-Only Gating) PCS Cutoffs...")
    p3_cuts = [0.70, 0.75, 0.80, 0.85, 0.90]
    p3_results = []
    for p3_c in p3_cuts:
        lost = 0
        fa_act = 0
        delays = []
        for fd in folds_data:
            act_ev = (fd["al_ev"] == 1) & (fd["pcs_ev"] >= p3_c)
            rev_ev = (fd["al_ev"] == 1) & (fd["pcs_ev"] < p3_c)
            if act_ev.any():
                delays.append(int(np.argmax(act_ev)))
            elif rev_ev.any():
                delays.append(int(np.argmax(rev_ev)))
            else:
                lost += 1
                delays.append(len(fd["al_ev"]))

            act_nb = (fd["al_nb"] == 1) & (fd["pcs_nb"] >= p3_c)
            fa_act += len(runs(act_nb))

        p3_results.append({
            "policy": "Policy 3",
            "pcs_cutoff": p3_c,
            "attacks_preserved": f"{5 - lost}/5",
            "attacks_lost": lost,
            "action_fa_events": fa_act,
            "mean_delay_hours": round(float(np.mean(delays)), 2),
        })
    df_p3 = pd.DataFrame(p3_results)
    print(df_p3.to_string(index=False))

    # Choose optimal Policy 3 cutoff: lost==0 -> min fa_act -> min delay -> max cutoff
    valid_p3 = df_p3[df_p3["attacks_lost"] == 0]
    best_p3 = valid_p3.sort_values(
        by=["action_fa_events", "mean_delay_hours", "pcs_cutoff"],
        ascending=[True, True, False]
    ).iloc[0]
    chosen_p3_cutoff = float(best_p3["pcs_cutoff"])
    print(f">> Chosen Policy 3 PCS Cutoff on Validation: {chosen_p3_cutoff:.2f}")

    # -------------------------------------------------------------
    # Grid Evaluation: Policy 4 Trust Score Weights & Cutoffs
    # -------------------------------------------------------------
    print("\n>> Evaluating Policy 4 (Trust-Gated Triage) Grid Space...")
    w_pcs_vals = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    w_uqs_vals = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    cuts = [0.70, 0.75, 0.80, 0.85]

    grid_results = []
    for wp in w_pcs_vals:
        for wu in w_uqs_vals:
            we = round(1.0 - wp - wu, 4)
            if we < 0.0:
                continue
            for c in cuts:
                lost = 0
                fa_act = 0
                fa_rev = 0
                delays = []
                for fd in folds_data:
                    # Attack evaluation
                    ts_ev = wp * fd["pcs_ev"] + wu * (1.0 - fd["uqs_ev"]) + we * fd["ers_ev"]
                    act_ev = (fd["al_ev"] == 1) & (ts_ev >= c)
                    rev_ev = (fd["al_ev"] == 1) & (ts_ev >= 0.50) & (ts_ev < c)
                    if act_ev.any():
                        delays.append(int(np.argmax(act_ev)))
                    elif rev_ev.any():
                        delays.append(int(np.argmax(rev_ev)))
                    else:
                        lost += 1
                        delays.append(len(ts_ev))

                    # Normal block evaluation
                    ts_nb = wp * fd["pcs_nb"] + wu * (1.0 - fd["uqs_nb"]) + we * fd["ers_nb"]
                    act_nb = (fd["al_nb"] == 1) & (ts_nb >= c)
                    rev_nb = (fd["al_nb"] == 1) & (ts_nb >= 0.50) & (ts_nb < c)
                    fa_act += len(runs(act_nb))
                    fa_rev += len(runs(rev_nb))

                grid_results.append({
                    "w_pcs": wp,
                    "w_uqs": wu,
                    "w_ers": we,
                    "action_cutoff": c,
                    "review_cutoff": 0.50,
                    "attacks_preserved": f"{5 - lost}/5",
                    "attacks_lost": lost,
                    "action_fa_events": fa_act,
                    "review_fa_events": fa_rev,
                    "mean_delay_hours": round(float(np.mean(delays)), 2),
                })

    df_grid = pd.DataFrame(grid_results)
    print(f">> Total grid points evaluated: {len(df_grid)}")

    # Pre-committed selection rule:
    # 1. attacks_lost == 0
    # 2. min action_fa_events
    # 3. min mean_delay_hours
    # 4. Tie-breakers: max w_pcs, max action_cutoff, min w_uqs
    valid_grid = df_grid[df_grid["attacks_lost"] == 0]
    assert len(valid_grid) > 0, "FATAL: No configuration preserved all 5 validation attacks!"

    sorted_candidates = valid_grid.sort_values(
        by=["action_fa_events", "mean_delay_hours", "w_pcs", "action_cutoff"],
        ascending=[True, True, False, False]
    )

    chosen = sorted_candidates.iloc[0]
    chosen_w_pcs = float(chosen["w_pcs"])
    chosen_w_uqs = float(chosen["w_uqs"])
    chosen_w_ers = float(chosen["w_ers"])
    chosen_cutoff = float(chosen["action_cutoff"])

    print("\n" + "=" * 70)
    print(">> OPTIMAL VALIDATION CONFIGURATION SELECTED VIA PRE-COMMITTED RULE:")
    print("=" * 70)
    print(f"w_pcs: {chosen_w_pcs:.2f} | w_uqs: {chosen_w_uqs:.2f} | w_ers: {chosen_w_ers:.2f}")
    print(f"Action Cutoff: {chosen_cutoff:.2f} (Review Cutoff: 0.50)")
    print(f"Validation Performance: Attacks Preserved {chosen['attacks_preserved']}, "
          f"Action FA Events: {chosen['action_fa_events']}, Mean Delay: {chosen['mean_delay_hours']}h")
    print("=" * 70)

    # -------------------------------------------------------------
    # Serialize Outputs
    # -------------------------------------------------------------
    report_dict = {
        "metadata": {
            "title": "Dataset 04 Triage Cross-Validation Grid Search",
            "dataset": "BATADAL_dataset04 (Leave-One-Attack-Event-Out, 5 Folds)",
            "timestamp": datetime.now().isoformat(),
            "selection_rule": "Attacks Preserved == 5/5 -> min Action FA events -> min Mean Delay",
        },
        "chosen_configuration": {
            "w_pcs": chosen_w_pcs,
            "w_uqs": chosen_w_uqs,
            "w_ers": chosen_w_ers,
            "action_cutoff": chosen_cutoff,
            "review_cutoff": 0.50,
            "policy3_pcs_cutoff": chosen_p3_cutoff,
            "validation_metrics": {
                "attacks_preserved": chosen["attacks_preserved"],
                "action_fa_events": int(chosen["action_fa_events"]),
                "review_fa_events": int(chosen["review_fa_events"]),
                "mean_delay_hours": float(chosen["mean_delay_hours"]),
            }
        },
        "top_10_candidates": sorted_candidates.head(10).to_dict(orient="records"),
        "policy3_results": p3_results,
    }

    # 1. JSON
    json_path = RESULTS_V3 / "triage_cv_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # 2. CSV
    csv_path = RESULTS_V3 / "triage_cv_report.csv"
    df_grid.to_csv(csv_path, index=False)

    # 3. Markdown
    md_path = RESULTS_V3 / "triage_cv_report.md"
    top_md = sorted_candidates.head(10).to_markdown(index=False)
    p3_md = df_p3.to_markdown(index=False)
    md_text = f"""# Dataset 04 Triage Cross-Validation Report (Task 3.1)

**Protocol**: Leave-One-Attack-Event-Out cross-validation across all 5 labelled attack events on Dataset 04.  
**Search Objective**: Pre-committed objective rule:
$$\\text{{Attacks Preserved}} = 5/5 \\implies \\min(\\text{{Action FA Events}}) \\implies \\min(\\text{{Mean Alert Delay}})$$

---

## 1. Selected Pareto-Optimal Parameters
- **Trust Score Weights (3-term)**: $w_{{\\text{{pcs}}}} = {chosen_w_pcs:.2f}$, $w_{{\\text{{uqs}}}} = {chosen_w_uqs:.2f}$, $w_{{\\text{{ers}}}} = {chosen_w_ers:.2f}$
- **Action Cutoff**: $TS \\ge {chosen_cutoff:.2f}$ (Review queue: $0.50 \\le TS < {chosen_cutoff:.2f}$)
- **Policy 3 PCS Cutoff**: $\\text{{PCS}} \\ge {chosen_p3_cutoff:.2f}$
- **Validation Score**: Attacks Preserved: **{chosen['attacks_preserved']}**, Action FA Events: **{chosen['action_fa_events']}**, Mean Delay: **{chosen['mean_delay_hours']}h**

---

## 2. Policy 3 Confidence-Only Cutoff CV Comparison
{p3_md}

---

## 3. Top-10 Pareto Candidates (Policy 4 Trust-Gated Triage)
{top_md}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # 4. Freeze to trust_frozen_config.json
    frozen_cfg_path = ROOT / "trust_frozen_config.json"
    frozen_cfg = {
        "metadata": {
            "description": "Frozen Trust Score weights, ATI action cutoff, and Policy 3 PCS cutoff",
            "validation_source": "BATADAL_dataset04 leave-one-attack-event-out CV (5 folds)",
            "selection_rule": "Attacks Preserved == 5/5 -> min Action FA events -> min Mean Delay",
            "authoritative_script": "experiments/run_triage_cv.py",
            "frozen_timestamp": datetime.now().isoformat(),
        },
        "trust_weights_3term": {
            "w_pcs": chosen_w_pcs,
            "w_uqs": chosen_w_uqs,
            "w_ers": chosen_w_ers,
        },
        "ati_bands": {
            "action_threshold": chosen_cutoff,
            "review_threshold": 0.50,
            "monitor_threshold": 0.00,
        },
        "policy_cutoffs": {
            "policy3_pcs_cutoff": chosen_p3_cutoff,
        },
        "validation_metrics": {
            "attacks_preserved": chosen["attacks_preserved"],
            "action_fa_events": int(chosen["action_fa_events"]),
            "review_fa_events": int(chosen["review_fa_events"]),
            "mean_delay_hours": float(chosen["mean_delay_hours"]),
        }
    }
    with open(frozen_cfg_path, "w", encoding="utf-8") as f:
        json.dump(frozen_cfg, f, indent=2)

    print(f"\n>> Results saved to:\n   - {json_path}\n   - {csv_path}\n   - {md_path}\n   - {frozen_cfg_path}")
    return frozen_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dataset 04 Triage CV grid search.")
    args = parser.parse_args()
    run_triage_cv()
