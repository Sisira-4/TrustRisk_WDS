import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
raw_dir = os.path.join(base_dir, 'data', 'raw')

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
    lag_feats = []
    for l in range(1, lags + 1):
        lag_feats.append(X_raw[lags-l : T-l])
    lag_matrix = np.hstack(lag_feats) if lags > 0 else np.empty((T, 0))
    return Y, lag_matrix, X_raw[lags:]

def run_analysis():
    df4 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset04.csv'), skipinitialspace=True)
    
    # 1. CLUSTER CLAIMS (Margin Check)
    attack_idx = df4[df4['ATT_FLAG'] == 1].index.values
    events = np.split(attack_idx, np.where(np.diff(attack_idx) != 1)[0] + 1)
    margin = 48
    
    def check_dist(idx):
        return np.min(np.abs(attack_idx - idx))
        
    print("=== TASK 1: MARGIN CHECK ===")
    for cl_name, cl_idx in [("1772", 1772), ("2887", 2887), ("3736", 3736)]:
        dist = check_dist(cl_idx)
        print(f"Cluster {cl_name} distance to nearest attack: {dist} hours. In margin? {dist <= 48}")

    # Baseline DS04-only
    print("\n=== TASK 3 & 4: DS04-ONLY DELAY & SENSORS ===")
    normal_ds4_idx = df4[df4['ATT_FLAG'] == -999].index.values
    
    X_train = df4.loc[normal_ds4_idx, sensors].values
    Y_tr, lag_tr, X_curr_tr = build_features(X_train, 1)
    
    models, scalers = [], []
    train_res = np.zeros_like(Y_tr)
    for i in range(len(sensors)):
        X_tr = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
        y_tr = Y_tr[:, i]
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        mod = Ridge(alpha=1.0)
        mod.fit(X_tr_sc, y_tr)
        models.append(mod)
        scalers.append(sc)
        train_res[:, i] = y_tr - mod.predict(X_tr_sc)
        
    res_mu = np.mean(train_res, axis=0)
    res_std = np.std(train_res, axis=0) + 1e-8
    
    train_z = np.abs(train_res - res_mu) / res_std
    train_scores = np.max(train_z, axis=1)
    thr = np.percentile(train_scores, 99.9)
    print(f"DS04-Only Training Threshold (99.9th): {thr:.2f}")

    # Evaluate unexplained alarms
    def get_top_sensor(idxs):
        # We need sequential features for eval
        eval_idx = np.concatenate([[idxs[0]-1], idxs])
        X_eval = df4.loc[eval_idx, sensors].values
        Y_ev, lag_ev, X_curr_ev = build_features(X_eval, 1)
        res = np.zeros_like(Y_ev)
        for i in range(len(sensors)):
            X_feat = np.hstack([lag_ev, np.delete(X_curr_ev, i, axis=1)])
            X_sc = scalers[i].transform(X_feat)
            res[:, i] = Y_ev[:, i] - models[i].predict(X_sc)
        z = np.abs(res - res_mu) / res_std
        peak_hr = np.argmax(np.max(z, axis=1))
        top_sens = sensors[np.argmax(z[peak_hr])]
        return np.max(z), top_sens

    print("\nUnexplained Alarms Sensors (DS04-Only Model):")
    for start, end in [(2026, 2041), (3963, 3971), (3987, 3995), (4011, 4019)]:
        idxs = np.arange(start, end)
        mz, sens = get_top_sensor(idxs)
        print(f"  {start}-{end-1}: Peak Z={mz:.2f} driven by {sens}")

    # Task 3: Recompute 60h event delay (Event 2)
    ev2_idx = events[1]
    ev2_eval_idx = np.concatenate([[ev2_idx[0]-1], ev2_idx])
    X_ev2 = df4.loc[ev2_eval_idx, sensors].values
    Y_ev2, lag_ev2, X_curr_ev2 = build_features(X_ev2, 1)
    ev2_res = np.zeros_like(Y_ev2)
    for i in range(len(sensors)):
        X_feat = np.hstack([lag_ev2, np.delete(X_curr_ev2, i, axis=1)])
        X_sc = scalers[i].transform(X_feat)
        ev2_res[:, i] = Y_ev2[:, i] - models[i].predict(X_sc)
        
    ev2_z = np.abs(ev2_res - res_mu) / res_std
    ev2_max_z = np.max(ev2_z, axis=1)
    
    # Delay metrics
    base_alarm = (ev2_max_z > thr).astype(int)
    delay_base = np.argmax(base_alarm) if np.any(base_alarm) else -1
    
    s_preds = pd.Series(base_alarm)
    pers_alarm = (s_preds.rolling(6, min_periods=1).sum() >= 3).astype(int).values
    delay_pers = np.argmax(pers_alarm) if np.any(pers_alarm) else -1
    
    print(f"\nDelay Recomputation for 60-Hour Event (Threshold {thr:.2f}):")
    print(f"  No Persistence Delay: {delay_base}h")
    print(f"  Persistence (3/6) Delay: {delay_pers}h")
    
    # Task 5: Recency Check
    print("\n=== TASK 5: RECENCY CHECK ===")
    train_idx_recent = normal_ds4_idx[normal_ds4_idx <= 3900]
    test_idx_recent = normal_ds4_idx[normal_ds4_idx > 3900]
    
    # Just check the previously computed train_z for FA rate before and after 3900
    # Because train_z is computed on normal_ds4_idx
    z_before = train_scores[normal_ds4_idx[1:] <= 3900]
    z_after = train_scores[normal_ds4_idx[1:] > 3900]
    
    ar_before = np.mean(z_before > thr)
    ar_after = np.mean(z_after > thr)
    print(f"  FA Rate (Idx <= 3900): {ar_before:.5f}")
    print(f"  FA Rate (Idx >  3900): {ar_after:.5f}")

run_analysis()
