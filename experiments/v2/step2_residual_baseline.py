import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import json

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
v2_data_dir = os.path.join(base_dir, 'data', 'v2')

def point_adjust(labels, preds):
    adjusted = preds.copy()
    attack_indices = np.where(labels == 1)[0]
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1) if len(attack_indices) > 0 else []
    for event in events:
        if np.any(preds[event] == 1):
            adjusted[event] = 1
    return adjusted

sensors = [
    'L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
    'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3',
    'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6',
    'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9',
    'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2',
    'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
    'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422'
]

def build_features(X_raw, lags):
    T = len(X_raw)
    Y = X_raw[lags:]
    lag_feats = []
    for l in range(1, lags + 1):
        lag_feats.append(X_raw[lags-l : T-l])
    if lags > 0:
        lag_matrix = np.hstack(lag_feats)
    else:
        lag_matrix = np.empty((T, 0))
    return Y, lag_matrix, X_raw[lags:]

def run_experiment(exp_name, quantile=99.9, lags=1, robust=False, persistence=False, model_type='ridge'):
    print(f"\nRunning Experiment: {exp_name}")
    fold_train_scores = []
    fold_val_data = []
    
    # 1. Train and collect scores
    for fold in range(5):
        train_df = pd.read_csv(os.path.join(v2_data_dir, f'fold_{fold}_train.csv'))
        val_df = pd.read_csv(os.path.join(v2_data_dir, f'fold_{fold}_val.csv'))
        
        train_fit_df = train_df[(train_df['TRAIN_MASK'] == 1) & (train_df['LABEL'] == 0)].copy().reset_index(drop=True)
        
        X_train_raw = train_fit_df[sensors].values
        X_val_raw = val_df[sensors].values
        
        Y_tr, lag_tr, X_curr_tr = build_features(X_train_raw, lags)
        Y_va, lag_va, X_curr_va = build_features(X_val_raw, lags)
        
        val_labels = val_df['LABEL'].values[lags:]
        val_mask = val_df['EVAL_MASK'].values[lags:]
        
        val_residuals = np.zeros_like(Y_va)
        train_residuals = np.zeros_like(Y_tr)
        
        for i, s in enumerate(sensors):
            X_tr = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
            y_tr = Y_tr[:, i]
            
            X_va = np.hstack([lag_va, np.delete(X_curr_va, i, axis=1)])
            y_va = Y_va[:, i]
            
            scaler_x = StandardScaler()
            X_tr_sc = scaler_x.fit_transform(X_tr)
            X_va_sc = scaler_x.transform(X_va)
            
            if model_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif model_type == 'lightgbm':
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=1, verbose=-1)
            
            model.fit(X_tr_sc, y_tr)
            
            train_residuals[:, i] = y_tr - model.predict(X_tr_sc)
            val_residuals[:, i] = y_va - model.predict(X_va_sc)
            
        if robust:
            res_mu = np.median(train_residuals, axis=0)
            res_mad = np.median(np.abs(train_residuals - res_mu), axis=0) + 1e-8
            res_std = res_mad * 1.4826
        else:
            res_mu = np.mean(train_residuals, axis=0)
            res_std = np.std(train_residuals, axis=0) + 1e-8
        
        train_z = np.abs(train_residuals - res_mu) / res_std
        val_z = np.abs(val_residuals - res_mu) / res_std
        
        train_scores = np.max(train_z, axis=1)
        val_scores = np.max(val_z, axis=1)
        
        fold_train_scores.append(train_scores)
        fold_val_data.append({
            'scores': val_scores,
            'labels': val_labels,
            'mask': val_mask,
            'val_z': val_z
        })

    # 2. Determine global threshold strictly from train
    pooled_train_scores = np.concatenate(fold_train_scores)
    global_thr = np.percentile(pooled_train_scores, quantile)
    
    # 3. Evaluate Out-Of-Fold
    all_eval_labels = []
    all_eval_preds = []
    all_eval_pa = []
    
    per_fold_metrics = []
    
    for fold, vdata in enumerate(fold_val_data):
        scores = vdata['scores']
        labels = vdata['labels']
        mask = vdata['mask']
        
        preds = (scores > global_thr).astype(int)
        
        if persistence:
            # Alarm if >=3 of 6 consecutive hours are over threshold
            new_preds = np.zeros_like(preds)
            for i in range(5, len(preds)):
                if np.sum(preds[i-5:i+1]) >= 3:
                    new_preds[i] = 1
            preds = new_preds
            
        pa_preds = point_adjust(labels, preds)
        
        eval_idx = np.where(mask == 1)[0]
        eval_labels = labels[eval_idx]
        eval_preds = preds[eval_idx]
        eval_pa = pa_preds[eval_idx]
        
        all_eval_labels.extend(eval_labels)
        all_eval_preds.extend(eval_preds)
        all_eval_pa.extend(eval_pa)
        
        # Event metrics
        attack_indices = np.where(eval_labels == 1)[0]
        events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1) if len(attack_indices) > 0 else []
        
        detected = False
        delay = -1
        for ev in events:
            if len(ev) > 0:
                if np.any(eval_preds[ev] == 1):
                    detected = True
                    delay = int(np.argmax(eval_preds[ev] == 1))
                    break
                    
        normal_idx = np.where(eval_labels == 0)[0]
        fa_count = np.sum(eval_preds[normal_idx] == 1)
        
        per_fold_metrics.append({
            'fold': fold,
            'f1': f1_score(eval_labels, eval_preds, zero_division=0),
            'pa_f1': f1_score(eval_labels, eval_pa, zero_division=0),
            'detected': detected,
            'delay': delay,
            'fa_count': fa_count,
            'normal_hours': len(normal_idx)
        })
        
        if fold == 1 and exp_name == "Base":
            # Diagnose Fold 1
            val_z = vdata['val_z'][eval_idx]
            fa_locs = np.where((eval_labels == 0) & (eval_preds == 1))[0]
            if len(fa_locs) > 0:
                trigger_sensors = np.argmax(val_z[fa_locs], axis=1)
                unique, counts = np.unique(trigger_sensors, return_counts=True)
                print(f"Fold 1 FA Sensor Counts:")
                for u, c in zip(unique, counts):
                    print(f"  {sensors[u]}: {c}")

    all_eval_labels = np.array(all_eval_labels)
    all_eval_preds = np.array(all_eval_preds)
    all_eval_pa = np.array(all_eval_pa)
    
    prec = precision_score(all_eval_labels, all_eval_preds, zero_division=0)
    rec = recall_score(all_eval_labels, all_eval_preds, zero_division=0)
    f1 = f1_score(all_eval_labels, all_eval_preds, zero_division=0)
    pa_f1 = f1_score(all_eval_labels, all_eval_pa, zero_division=0)
    
    total_normal = len(np.where(all_eval_labels == 0)[0])
    total_fa = np.sum((all_eval_labels == 0) & (all_eval_preds == 1))
    tpr = np.sum((all_eval_labels == 1) & (all_eval_preds == 1)) / len(np.where(all_eval_labels == 1)[0]) if len(np.where(all_eval_labels == 1)[0])>0 else 0
    fpr_per_hour = total_fa / total_normal if total_normal > 0 else 0
    
    print(f"Global Threshold (99.9th perc train): {global_thr:.4f}")
    print(f"Pooled OOF: Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, PA-F1={pa_f1:.4f}")
    print(f"TPR: {tpr:.4f}, FPR per normal hour: {fpr_per_hour:.6f}")
    for m in per_fold_metrics:
        print(f"  Fold {m['fold']}: F1={m['f1']:.4f} PA={m['pa_f1']:.4f} Det={m['detected']} Dly={m['delay']} FA={m['fa_count']}")
        
    return f1

run_experiment("Base", lags=1, robust=False, persistence=False, model_type='ridge')
run_experiment("Lags=3", lags=3, robust=False, persistence=False, model_type='ridge')
run_experiment("Robust Z (Median/MAD)", lags=1, robust=True, persistence=False, model_type='ridge')
run_experiment("Persistence (>=3 of 6h)", lags=1, robust=False, persistence=True, model_type='ridge')
run_experiment("LightGBM", lags=1, robust=False, persistence=False, model_type='lightgbm')
