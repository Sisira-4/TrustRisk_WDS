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

def cusum(series, threshold=0.5, drift=0.1):
    c = np.zeros_like(series)
    for i in range(1, len(series)):
        c[i] = max(0, c[i-1] + series[i] - drift)
    return c

print("=== EVENT 1 HOURLY TRACE (60 HOURS) ===")
# Load
df3 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset03.csv'), skipinitialspace=True)
df4 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset04.csv'), skipinitialspace=True)

# Train baseline on DS03 + DS04(-999) to establish stats
X_train = np.vstack([df3[sensors].values, df4[df4['ATT_FLAG'] == -999][sensors].values])
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

# Evaluate on Event 1
attack_idx = df4[df4['ATT_FLAG'] == 1].index.values
events = np.split(attack_idx, np.where(np.diff(attack_idx) != 1)[0] + 1)
ev1_idx = events[0]
# Add 1 lag row before event 1
ev1_eval_idx = np.concatenate([[ev1_idx[0]-1], ev1_idx])
X_ev1 = df4.iloc[ev1_eval_idx][sensors].values
Y_ev1, lag_ev1, X_curr_ev1 = build_features(X_ev1, 1)

ev1_res = np.zeros_like(Y_ev1)
for i in range(len(sensors)):
    X_ev_feat = np.hstack([lag_ev1, np.delete(X_curr_ev1, i, axis=1)])
    X_ev_sc = scalers[i].transform(X_ev_feat)
    ev1_res[:, i] = Y_ev1[:, i] - models[i].predict(X_ev_sc)

ev1_z = np.abs(ev1_res - res_mu) / res_std
ev1_max_z = np.max(ev1_z, axis=1)
ev1_top3_mean_z = np.mean(np.sort(ev1_z, axis=1)[:, -3:], axis=1)

# Compute EWMA and CUSUM on max_z
ev1_ewma = pd.Series(ev1_max_z).ewm(span=12).mean().values
ev1_cusum = cusum(ev1_max_z, drift=np.percentile(np.max(np.abs(train_res - res_mu)/res_std, axis=1), 90))

print(f"{'Hr':>3} | {'Max Z':>7} | {'Top3 Z':>7} | {'EWMA':>7} | {'CUSUM':>7} | {'Top 5 Sensors'}")
print("-" * 65)
for hr in range(len(ev1_max_z)):
    z_scores = ev1_z[hr]
    top_5_idx = np.argsort(z_scores)[-5:][::-1]
    top_5_str = ", ".join([f"{sensors[idx]}({z_scores[idx]:.1f})" for idx in top_5_idx])
    
    print(f"{hr:>3} | {ev1_max_z[hr]:>7.2f} | {ev1_top3_mean_z[hr]:>7.2f} | {ev1_ewma[hr]:>7.2f} | {ev1_cusum[hr]:>7.2f} | {top_5_str}")


print("\n=== FPR FLOOR DIAGNOSTIC (DS04-999) ===")
# Predict on entire DS04 -999 to find clusters
X_ds4 = df4[sensors].values
Y_ds4, lag_ds4, X_curr_ds4 = build_features(X_ds4, 1)
ds4_res = np.zeros_like(Y_ds4)
for i in range(len(sensors)):
    X_ds4_feat = np.hstack([lag_ds4, np.delete(X_curr_ds4, i, axis=1)])
    X_ds4_sc = scalers[i].transform(X_ds4_feat)
    ds4_res[:, i] = Y_ds4[:, i] - models[i].predict(X_ds4_sc)
    
ds4_z = np.abs(ds4_res - res_mu) / res_std
ds4_max_z = np.max(ds4_z, axis=1)

q99 = np.percentile(np.max(np.abs(train_res - res_mu)/res_std, axis=1), 99)
print(f"99th Percentile Threshold (on train): {q99:.2f}")

alarms = (ds4_max_z > q99).astype(int)
# Only consider -999 rows
flags = df4['ATT_FLAG'].values[1:]
alarms[flags != -999] = 0

# Find contiguous clusters
clusters = np.split(np.where(alarms == 1)[0], np.where(np.diff(np.where(alarms == 1)[0]) != 1)[0] + 1)
if len(clusters[0]) == 0: clusters = []

print(f"Found {len(clusters)} alarm clusters in DS04 normal blocks.")
for idx, cl in enumerate(clusters):
    if len(cl) >= 4:
        start_idx = cl[0]
        end_idx = cl[-1]
        peak_hr = start_idx + np.argmax(ds4_max_z[cl])
        peak_z = ds4_max_z[peak_hr]
        top_sens = sensors[np.argmax(ds4_z[peak_hr])]
        print(f"  Cluster {idx+1}: {len(cl):>2} consecutive hrs (Indices {start_idx}-{end_idx}) | Peak Z = {peak_z:.2f} driven by {top_sens} -> POTENTIAL UNLABELED ATTACK")
