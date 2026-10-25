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

def run_empirical_tables():
    df3 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset03.csv'), skipinitialspace=True)
    df4 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset04.csv'), skipinitialspace=True)
    
    attack_idx = df4[df4['ATT_FLAG'] == 1].index.values
    events = np.split(attack_idx, np.where(np.diff(attack_idx) != 1)[0] + 1)
    
    # Exclude margins
    margin = 48
    all_attack_and_margin = set()
    for ev in events:
        for idx in range(max(0, ev[0] - margin), min(len(df4), ev[-1] + margin + 1)):
            all_attack_and_margin.add(idx)
    normal_idx_ds4 = np.array(sorted(set(range(len(df4))) - all_attack_and_margin))
    
    # =========================================================================
    # TASK 2: DELAY TABLE (60h event = Event 2, Fold 1)
    # =========================================================================
    ds4_blocks = np.array_split(normal_idx_ds4, 5)
    val_norm_idx = ds4_blocks[1]
    train_norm_idx = np.setdiff1d(normal_idx_ds4, val_norm_idx)
    ev2_eval_idx = np.concatenate([[events[1][0]-1], events[1]])
    
    X_train = df4.loc[train_norm_idx, sensors].values
    Y_tr, lag_tr, X_curr_tr = build_features(X_train, 1)
    
    models, scalers = [], []
    train_res = np.zeros_like(Y_tr)
    for i in range(len(sensors)):
        X_tr = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        mod = Ridge(alpha=1.0)
        mod.fit(X_tr_sc, Y_tr[:, i])
        models.append(mod)
        scalers.append(sc)
        train_res[:, i] = Y_tr[:, i] - mod.predict(X_tr_sc)
        
    res_mu, res_std = np.mean(train_res, axis=0), np.std(train_res, axis=0) + 1e-8
    train_top3 = np.mean(np.sort(np.abs(train_res - res_mu)/res_std, axis=1)[:, -3:], axis=1)
    
    def evaluate_block(eval_idxs):
        X_ev = df4.loc[eval_idxs, sensors].values
        Y_ev, lag_ev, X_curr_ev = build_features(X_ev, 1)
        res = np.zeros_like(Y_ev)
        for i in range(len(sensors)):
            X_ev_sc = scalers[i].transform(np.hstack([lag_ev, np.delete(X_curr_ev, i, axis=1)]))
            res[:, i] = Y_ev[:, i] - models[i].predict(X_ev_sc)
        return np.mean(np.sort(np.abs(res - res_mu)/res_std, axis=1)[:, -3:], axis=1)

    ev2_scores = evaluate_block(ev2_eval_idx)
    val_scores = evaluate_block(np.concatenate([[val_norm_idx[0]-1], val_norm_idx]))
    val_weeks = len(val_scores) / 168.0
    
    def eval_delay_fa(thr, method, params):
        if method == 'base':
            alarms = (val_scores > thr).astype(int)
            ev2_al = (ev2_scores > thr).astype(int)
        elif method == 'pers':
            alarms = (pd.Series((val_scores > thr).astype(int)).rolling(6, min_periods=1).sum() >= 3).astype(int).values
            ev2_al = (pd.Series((ev2_scores > thr).astype(int)).rolling(6, min_periods=1).sum() >= 3).astype(int).values
        elif method == 'ewma':
            alpha, h = params
            alarms = (pd.Series(val_scores).ewm(alpha=alpha).mean().values > h).astype(int)
            ev2_al = (pd.Series(ev2_scores).ewm(alpha=alpha).mean().values > h).astype(int)
        elif method == 'cusum':
            k, h = params
            def cusum(s):
                c = np.zeros_like(s)
                for i in range(1, len(s)): c[i] = max(0, c[i-1] + s[i] - k)
                return c
            alarms = (cusum(val_scores) > h).astype(int)
            ev2_al = (cusum(ev2_scores) > h).astype(int)
            
        del_ = np.argmax(ev2_al) if np.any(ev2_al) else -1
        clusters = np.split(np.where(alarms == 1)[0], np.where(np.diff(np.where(alarms == 1)[0]) != 1)[0] + 1)
        fa_evts = len([c for c in clusters if len(c) > 0])
        return f"{del_}h / {fa_evts/val_weeks:.1f}"

    print("=== TASK 2: DELAY TABLE (60h Event, DS04-only, Top-3 Z) ===")
    print(f"{'Quantile':<10} | {'Base (No Pers)':<20} | {'Pers (3/6)':<20} | {'EWMA (h=Thr)':<20} | {'CUSUM (k=mean, h=20)':<20}")
    for q in [99.0, 99.5, 99.9]:
        thr = np.percentile(train_top3, q)
        b = eval_delay_fa(thr, 'base', None)
        p = eval_delay_fa(thr, 'pers', None)
        # EWMA tuned inside (proxy for now, alpha=0.1, h=thr*0.8)
        e = eval_delay_fa(thr, 'ewma', (0.1, thr * 0.8))
        # CUSUM tuned inside (proxy k=mean(train), h=thr*2)
        c = eval_delay_fa(thr, 'cusum', (np.mean(train_top3), thr * 2))
        print(f"{q:<10} | {b:<20} | {p:<20} | {e:<20} | {c:<20}")

    # =========================================================================
    # TASK 3: RECENCY OUT-OF-SAMPLE
    # =========================================================================
    print("\n=== TASK 3: RECENCY WEIGHTING (Test on >= 3924, 253 rows) ===")
    tr_rec_idx = np.arange(3923) # 0 to 3922
    val_rec_idx = np.arange(3923, 3924 + 253) # 253 rows
    
    def run_recency(hl):
        X_tr = df4.loc[tr_rec_idx, sensors].values
        Y_tr, lag_tr, X_curr_tr = build_features(X_tr, 1)
        
        T_max = len(Y_tr)
        if hl is None:
            weights = np.ones(T_max)
        else:
            weights = np.exp(-np.log(2) * (T_max - np.arange(T_max)) / hl)
            
        models, scalers = [], []
        for i in range(len(sensors)):
            X_tr_feat = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr_feat)
            mod = Ridge(alpha=1.0)
            mod.fit(X_tr_sc, Y_tr[:, i], sample_weight=weights)
            models.append(mod)
            scalers.append(sc)
            
        # Get thresholds from train
        train_res = np.zeros_like(Y_tr)
        for i in range(len(sensors)):
            X_tr_feat = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
            train_res[:, i] = Y_tr[:, i] - models[i].predict(scalers[i].transform(X_tr_feat))
        res_mu, res_std = np.mean(train_res, axis=0), np.std(train_res, axis=0) + 1e-8
        train_z = np.max(np.abs(train_res - res_mu)/res_std, axis=1)
        
        # Eval on >= 3924
        X_ev = df4.loc[val_rec_idx, sensors].values
        Y_ev, lag_ev, X_curr_ev = build_features(X_ev, 1)
        val_res = np.zeros_like(Y_ev)
        for i in range(len(sensors)):
            X_ev_feat = np.hstack([lag_ev, np.delete(X_curr_ev, i, axis=1)])
            val_res[:, i] = Y_ev[:, i] - models[i].predict(scalers[i].transform(X_ev_feat))
        val_z = np.max(np.abs(val_res - res_mu)/res_std, axis=1)
        
        q99, q995, q999 = np.percentile(train_z, 99), np.percentile(train_z, 99.5), np.percentile(train_z, 99.9)
        return np.sum(val_z > q99), np.sum(val_z > q995), np.sum(val_z > q999)

    u99, u995, u999 = run_recency(None)
    h30_99, h30_995, h30_999 = run_recency(720) # 30 days
    h60_99, h60_995, h60_999 = run_recency(1440) # 60 days
    
    print(f"{'Weighting':<20} | {'q=99':<10} | {'q=99.5':<10} | {'q=99.9':<10}")
    print(f"{'None':<20} | {u99:<10} | {u995:<10} | {u999:<10}")
    print(f"{'HL=30d':<20} | {h30_99:<10} | {h30_995:<10} | {h30_999:<10}")
    print(f"{'HL=60d':<20} | {h60_99:<10} | {h60_995:<10} | {h60_999:<10}")

    # =========================================================================
    # TASK 4: SENSOR ATTRIBUTION
    # =========================================================================
    print("\n=== TASK 4: SENSOR ATTRIBUTION ===")
    
    def get_attr(train_df):
        X_tr = train_df[sensors].values
        Y_tr, lag_tr, X_curr_tr = build_features(X_tr, 1)
        models, scalers = [], []
        train_res = np.zeros_like(Y_tr)
        for i in range(len(sensors)):
            X_tr_feat = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
            sc = StandardScaler()
            mod = Ridge()
            mod.fit(sc.fit_transform(X_tr_feat), Y_tr[:, i])
            models.append(mod)
            scalers.append(sc)
            train_res[:, i] = Y_tr[:, i] - mod.predict(sc.transform(X_tr_feat))
        res_mu, res_std = np.mean(train_res, axis=0), np.std(train_res, axis=0) + 1e-8
        
        attr = {}
        for start, end in [(2026, 2041), (3963, 3971), (3987, 3995), (4011, 4019)]:
            eval_idx = np.arange(start-1, end)
            X_ev = df4.loc[eval_idx, sensors].values
            Y_ev, lag_ev, X_curr_ev = build_features(X_ev, 1)
            res = np.zeros_like(Y_ev)
            for i in range(len(sensors)):
                X_ev_feat = np.hstack([lag_ev, np.delete(X_curr_ev, i, axis=1)])
                res[:, i] = Y_ev[:, i] - models[i].predict(scalers[i].transform(X_ev_feat))
            z = np.max(np.abs(res - res_mu)/res_std, axis=0) # max per sensor over the block
            top5 = [sensors[idx] for idx in np.argsort(z)[-5:][::-1]]
            attr[f"{start}-{end-1}"] = top5
        return attr

    attr_ds4 = get_attr(df4.loc[normal_idx_ds4])
    attr_ds34 = get_attr(pd.concat([df3, df4.loc[normal_idx_ds4]]))
    
    for cl in attr_ds4.keys():
        print(f"Cluster {cl}:")
        print(f"  DS04-Only: {', '.join(attr_ds4[cl])}")
        print(f"  DS03+DS04: {', '.join(attr_ds34[cl])}")

run_empirical_tables()
