"""Generates the official Data Card table for the TrustRisk-WDS benchmark.

Tracks row counts, normal hours, unlabelled (-999) hours, attack hours, attack events,
and margin-masked hours across all splits.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def runs(mask):
    idx = np.flatnonzero(np.asarray(mask))
    if idx.size == 0:
        return []
    return [(int(g[0]), int(g[-1])) for g in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)]


def analyze_split(csv_path, name, has_margin_calc=False, margin=48):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    flag_col = "ATT_FLAG" if "ATT_FLAG" in df.columns else "ATTACK_FLAG"
    labels = df[flag_col].values
    total_rows = len(df)
    
    normal_0 = int((labels == 0).sum())
    unlabelled_999 = int((labels == -999).sum())
    attack_1 = int((labels == 1).sum())
    
    attack_events = runs(labels == 1)
    n_events = len(attack_events)
    
    margin_masked = 0
    if has_margin_calc and n_events > 0:
        near = np.zeros(total_rows, bool)
        for s, e in attack_events:
            near[max(0, s - margin): min(total_rows, e + margin + 1)] = True
        margin_masked = int(near.sum() - attack_1)

    return {
        "Dataset Split": name,
        "Total Hours (Rows)": total_rows,
        "Confirmed Normal (0)": normal_0,
        "Unlabelled (-999)": unlabelled_999,
        "Attack Hours (1)": attack_1,
        "Attack Events": n_events,
        "Margin Masked (±48h)": margin_masked
    }


def main():
    records = []
    
    # 1. Dataset 03 (Training baseline)
    p3 = RAW_DIR / "BATADAL_dataset03.csv"
    if p3.exists():
        records.append(analyze_split(p3, "Training (BATADAL Dataset 03)"))
        
    # 2. Dataset 04 (Cross-validation / Attack dynamics)
    p4 = RAW_DIR / "BATADAL_dataset04.csv"
    if p4.exists():
        records.append(analyze_split(p4, "Cross-Validation (BATADAL Dataset 04)", has_margin_calc=True, margin=48))
        
    # 3. Test Dataset (Quarantined Test Set)
    p_test = RAW_DIR / "BATADAL_test_dataset_labeled.csv"
    if p_test.exists():
        records.append(analyze_split(p_test, "Test (BATADAL Test Labeled)"))

    card_df = pd.DataFrame(records)
    md_content = "# TrustRisk-WDS Benchmark Data Card\n\n"
    md_content += "Summary of dataset splits, label semantics, and masking protocol:\n\n"
    md_content += card_df.to_markdown(index=False)
    md_content += "\n\n### Label Semantics & Masking Policy:\n"
    md_content += "1. **Provisional Normal Training Pool**: Unlabelled rows (`-999`) in Dataset 04 situated > 48h from any attack are used as a provisional normal training pool for residual regression fitting.\n"
    md_content += "2. **Evaluation Semantics (Never Confirmed Negatives)**: `-999` rows are strictly treated as **unlabelled** in all evaluation protocols. Any alarms occurring during `-999` periods are termed *'unexplained alarms'* (or estimated false alarms), acknowledging unlabelled operational disturbances.\n"
    md_content += "3. **Transition Masking (±48h Buffer)**: A strict 48-hour buffer before and after each labelled attack in Dataset 04 is masked (`MARGIN_MASK = 1`) to eliminate transitional hydraulic recovery periods from both training and evaluation.\n"

    out_file = OUT_DIR / "data_card.md"
    out_file.write_text(md_content, encoding="utf-8")
    print(f"Generated data card at {out_file}:\n")
    print(card_df.to_string(index=False))


if __name__ == "__main__":
    main()
