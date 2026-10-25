import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
raw_dir = os.path.join(base_dir, 'data', 'raw')
v2_dir = os.path.join(base_dir, 'experiments', 'v2')

sensors = [
    'L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
    'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3',
    'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6',
    'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9',
    'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2',
    'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
    'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422'
]

def build_features(X_raw, lags=1):
    T = len(X_raw)
    Y = X_raw[lags:]
    lag_feats = [X_raw[lags-l : T-l] for l in range(1, lags + 1)]
    lag_matrix = np.hstack(lag_feats) if lags > 0 else np.empty((T, 0))
    return Y, lag_matrix, X_raw[lags:]

def run_audit():
    df4 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset04.csv'), skipinitialspace=True)
    
    # Identify labeled events
    attack_idx = df4[df4['ATT_FLAG'] == 1].index.values
    events = np.split(attack_idx, np.where(np.diff(attack_idx) != 1)[0] + 1)
    
    margin = 48
    all_attack_and_margin = set()
    for ev in events:
        for idx in range(max(0, ev[0] - margin), min(len(df4), ev[-1] + margin + 1)):
            all_attack_and_margin.add(idx)
            
    normal_idx_ds4 = np.array(sorted(set(range(len(df4))) - all_attack_and_margin))
    ds4_blocks = np.array_split(normal_idx_ds4, 5)
    
    print("=== TASK 3: LEAKAGE AUDIT & ALARM HOURS ===")
    alarm_hours_per_fold = []
    
    for fold in range(5):
        val_norm = ds4_blocks[fold]
        train_norm = np.setdiff1d(normal_idx_ds4, val_norm)
        
        print(f"Fold {fold}: Train Range={train_norm[0]}-{train_norm[-1]}, Val Range={val_norm[0]}-{val_norm[-1]}")
        
        X_tr = df4.loc[train_norm, sensors].values
        Y_tr, lag_tr, X_curr_tr = build_features(X_tr, 1)
        
        models, scalers = [], []
        train_res = np.zeros_like(Y_tr)
        
        for i in range(len(sensors)):
            X_tr_feat = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr_feat)
            mod = Ridge(alpha=1.0)
            mod.fit(X_tr_sc, Y_tr[:, i])
            models.append(mod)
            scalers.append(sc)
            train_res[:, i] = Y_tr[:, i] - mod.predict(X_tr_sc)
            
        res_mu, res_std = np.mean(train_res, axis=0), np.std(train_res, axis=0) + 1e-8
        train_z = np.max(np.abs(train_res - res_mu)/res_std, axis=1) # The reference mentions top-3 but some say top-1. I will check top-3.
        
        train_top3 = np.mean(np.sort(np.abs(train_res - res_mu)/res_std, axis=1)[:, -3:], axis=1)
        
        thr = np.percentile(train_top3, 99.9)
        
        # Eval
        val_eval_idx = np.concatenate([[val_norm[0]-1], val_norm])
        X_val = df4.loc[val_eval_idx, sensors].values
        Y_val, lag_val, X_curr_val = build_features(X_val, 1)
        val_res = np.zeros_like(Y_val)
        
        for i in range(len(sensors)):
            X_val_feat = np.hstack([lag_val, np.delete(X_curr_val, i, axis=1)])
            val_res[:, i] = Y_val[:, i] - models[i].predict(scalers[i].transform(X_val_feat))
            
        val_z = np.mean(np.sort(np.abs(val_res - res_mu)/res_std, axis=1)[:, -3:], axis=1)
        alarm_hrs = int(np.sum(val_z > thr))
        alarm_hours_per_fold.append(alarm_hrs)
        print(f"  -> In-Sample q99.9 Thr = {thr:.2f}, Val Alarm Hours = {alarm_hrs}")

    print(f"\nReported Top-3 Per-Fold Alarm Hours: {alarm_hours_per_fold}")
    
    # Save a minimal frozen_config to git
    config = {
        "training_set": "DS04-999-only (all blocks combined for final model)",
        "threshold_procedure": "Inner-CV targeting <=1 FA-event/week on hold-out folds",
        "regime": "DS04-999-only",
        "score_variant": "top3_mean_z",
        "lags": 1,
        "quantile": 99.9,
        "persistence": "3/6",
        "cusum_params": {"k": "optimized_per_fold", "h": "optimized_per_fold"},
        "ewma_params": {"span": 12},
        "masked_clusters": []
    }
    with open(os.path.join(v2_dir, 'frozen_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

run_audit()
