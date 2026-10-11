"""Clean, leakage-free dataset preprocessor enforcing strict -999 unlabelled segregation.

Protocol:
1. Dataset 03: Confirmed normal baseline (LABEL = 0, EVAL_MASK = 1).
2. Dataset 04: Attacks (LABEL = 1, EVAL_MASK = 1).
   Unlabelled rows (ATT_FLAG == -999) are PRESERVED as -999 and explicitly masked out:
   - TRAIN_MASK = 0 (excluded from training loss)
   - EVAL_MASK = 0 (excluded from validation evaluation)
3. ±48h Transition Margin: Transition buffer surrounding each attack event in Dataset 04
   is strictly masked to prevent transitional hydraulic recovery from contaminating normal statistics.
4. Test set is strictly quarantined for the single-shot frozen evaluation.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
V2_DIR = ROOT / 'data' / 'v2'
V2_DIR.mkdir(parents=True, exist_ok=True)


def build_clean_datasets():
    print("Building clean cross-validation dataset splits with strict -999 segregation...")
    
    # 1. Load Dataset 03 and 04
    p3 = RAW_DIR / "BATADAL_dataset03.csv"
    p4 = RAW_DIR / "BATADAL_dataset04.csv"
    
    df3 = pd.read_csv(p3, skipinitialspace=True)
    df4 = pd.read_csv(p4, skipinitialspace=True)
    
    df3.columns = df3.columns.str.strip()
    df4.columns = df4.columns.str.strip()
    
    df3['DATETIME'] = pd.to_datetime(df3['DATETIME'], format='%d/%m/%y %H')
    df4['DATETIME'] = pd.to_datetime(df4['DATETIME'], format='%d/%m/%y %H')
    
    # 2. Identify 5 attack events in Dataset 04
    attack_indices = df4[df4['ATT_FLAG'] == 1].index.to_numpy()
    event_splits = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    
    # Split dataset03 into 5 contiguous blocks for held-out normal evaluation
    block_size = len(df3) // 5
    df3_blocks = [np.arange(i * block_size, (i + 1) * block_size if i < 4 else len(df3)) for i in range(5)]
    
    margin = 48
    
    for fold, ev_idx in enumerate(event_splits):
        val_event_start = ev_idx[0]
        val_event_end = ev_idx[-1]
        
        event_with_margin_indices = np.arange(
            max(0, val_event_start - margin),
            min(len(df4), val_event_end + margin + 1)
        )
        
        # Validation Normal (from dataset03)
        val_normal_idx = df3_blocks[fold]
        val_normal_df = df3.iloc[val_normal_idx].copy()
        val_normal_df['LABEL'] = 0
        val_normal_df['EVAL_MASK'] = 1
        
        # Validation Attack (from dataset04, ONLY scoring ATT_FLAG == 1)
        val_attack_df = df4.iloc[event_with_margin_indices].copy()
        # Strictly preserve -999 as -999, never replace with 0
        val_attack_df['LABEL'] = np.where(
            val_attack_df['ATT_FLAG'] == 1, 1,
            np.where(val_attack_df['ATT_FLAG'] == -999, -999, 0)
        )
        # Exclude -999 from evaluation score completely
        val_attack_df['EVAL_MASK'] = (val_attack_df['ATT_FLAG'] == 1).astype(int)
        
        val_df = pd.concat([val_normal_df, val_attack_df], ignore_index=True)
        
        # Training Normal (dataset03 minus held-out block)
        train_normal_idx = np.setdiff1d(np.arange(len(df3)), val_normal_idx)
        train_normal_df = df3.iloc[train_normal_idx].copy()
        train_normal_df['LABEL'] = 0
        train_normal_df['TRAIN_MASK'] = 1
        
        # Training Attack/Masked (dataset04 minus event and margin)
        train_df4_idx = np.setdiff1d(np.arange(len(df4)), event_with_margin_indices)
        train_df4 = df4.iloc[train_df4_idx].copy()
        train_df4['LABEL'] = np.where(
            train_df4['ATT_FLAG'] == 1, 1,
            np.where(train_df4['ATT_FLAG'] == -999, -999, 0)
        )
        # Mask out the -999 unlabelled rows from training loss
        train_df4['TRAIN_MASK'] = (train_df4['ATT_FLAG'] != -999).astype(int)
        
        train_df = pd.concat([train_normal_df, train_df4], ignore_index=True)
        
        # Save fold CSVs
        train_df.to_csv(V2_DIR / f'fold_{fold}_train.csv', index=False)
        val_df.to_csv(V2_DIR / f'fold_{fold}_val.csv', index=False)
        
    print(f">> Generated {len(event_splits)} clean folds in {V2_DIR} with -999 unlabelled masking.")


if __name__ == "__main__":
    build_clean_datasets()
