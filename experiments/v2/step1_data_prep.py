import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
raw_dir = os.path.join(base_dir, 'data', 'raw')
v2_data_dir = os.path.join(base_dir, 'data', 'v2')
os.makedirs(v2_data_dir, exist_ok=True)

df3 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset03.csv'), skipinitialspace=True)
df4 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset04.csv'), skipinitialspace=True)

df3['DATETIME'] = pd.to_datetime(df3['DATETIME'], format='%d/%m/%y %H')
df4['DATETIME'] = pd.to_datetime(df4['DATETIME'], format='%d/%m/%y %H')

# Identify 5 events
attack_indices = df4[df4['ATT_FLAG'] == 1].index.to_numpy()
event_splits = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)

# Split dataset03 into 5 contiguous blocks for held-out normal evaluation
block_size = len(df3) // 5
df3_blocks = [np.arange(i * block_size, (i + 1) * block_size if i < 4 else len(df3)) for i in range(5)]

table_data = []

for fold, ev_idx in enumerate(event_splits):
    val_event_start = ev_idx[0]
    val_event_end = ev_idx[-1]
    
    # Margin = 48 hours
    margin = 48
    event_with_margin_indices = np.arange(max(0, val_event_start - margin), min(len(df4), val_event_end + margin + 1))
    
    # Validation Normal (from dataset03)
    val_normal_idx = df3_blocks[fold]
    val_normal_df = df3.iloc[val_normal_idx].copy()
    val_normal_df['LABEL'] = 0
    val_normal_df['EVAL_MASK'] = 1  # 1 means we score this row
    
    # Validation Attack (from dataset04, ONLY scoring the ATT_FLAG == 1 rows)
    val_attack_df = df4.iloc[event_with_margin_indices].copy()
    val_attack_df['LABEL'] = (val_attack_df['ATT_FLAG'] == 1).astype(int)
    # Exclude -999 from scoring completely
    val_attack_df['EVAL_MASK'] = (val_attack_df['ATT_FLAG'] == 1).astype(int) 
    
    val_df = pd.concat([val_normal_df, val_attack_df], ignore_index=True)
    
    # Training Normal (dataset03 minus the held out block)
    train_normal_idx = np.setdiff1d(np.arange(len(df3)), val_normal_idx)
    train_normal_df = df3.iloc[train_normal_idx].copy()
    train_normal_df['LABEL'] = 0
    train_normal_df['TRAIN_MASK'] = 1  # Full normal training data
    
    # Training Attack/Masked (dataset04 minus the event and its margin)
    train_df4_idx = np.setdiff1d(np.arange(len(df4)), event_with_margin_indices)
    train_df4 = df4.iloc[train_df4_idx].copy()
    train_df4['LABEL'] = (train_df4['ATT_FLAG'] == 1).astype(int)
    # Mask out the -999 unlabeled rows from training loss!
    train_df4['TRAIN_MASK'] = (train_df4['ATT_FLAG'] != -999).astype(int)
    
    train_df = pd.concat([train_normal_df, train_df4], ignore_index=True)
    
    # Save folds
    train_df.to_csv(os.path.join(v2_data_dir, f'fold_{fold}_train.csv'), index=False)
    val_df.to_csv(os.path.join(v2_data_dir, f'fold_{fold}_val.csv'), index=False)
    
    # Count stats
    train_masked_999 = len(train_df4[train_df4['ATT_FLAG'] == -999])
    margin_excluded = len(event_with_margin_indices)
    scored_positives = len(val_attack_df[val_attack_df['ATT_FLAG'] == 1])
    scored_normals = len(val_normal_df)
    
    table_data.append({
        'Fold': fold,
        'Train Rows': len(train_df),
        'Train Masked (-999)': train_masked_999,
        'Margin-Excluded (val context)': margin_excluded,
        'Scored Positives (Val)': scored_positives,
        'Scored Normals (Val)': scored_normals
    })

print("\n--- V2 CV Folds Summary ---")
df_summary = pd.DataFrame(table_data)
print(df_summary.to_markdown(index=False))
