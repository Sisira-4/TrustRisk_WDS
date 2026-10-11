import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.train_gnn_gru_physical import PhysicalGNN_GRU, get_probs, point_adjust
from models.gnn_gru.zone_ids_clean import ZoneAdapter, GlobalZoneModel, create_sequences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_metrics_at_thresh(labels, probs, thresh=0.5):
    preds = (probs >= thresh).astype(int)
    cm = confusion_matrix(labels, preds)
    if cm.shape == (1, 1):
        if labels[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # Point-Adjusted metrics
    pa_preds = point_adjust(labels, preds)
    pa_cm = confusion_matrix(labels, pa_preds)
    pa_tn, pa_fp, pa_fn, pa_tp = pa_cm.ravel() if pa_cm.shape == (2, 2) else (tn, fp, fn, tp)
    pa_prec = precision_score(labels, pa_preds, zero_division=0)
    pa_rec = recall_score(labels, pa_preds, zero_division=0)
    pa_f1 = f1_score(labels, pa_preds, zero_division=0)

    try:
        roc = roc_auc_score(labels, probs)
    except:
        roc = 0.5

    return {
        "threshold": float(thresh),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pa_f1": float(pa_f1),
        "pa_precision": float(pa_prec),
        "pa_recall": float(pa_rec),
        "roc_auc": float(roc),
        "cm": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "pa_cm": {
            "TN": int(pa_tn),
            "FP": int(pa_fp),
            "FN": int(pa_fn),
            "TP": int(pa_tp)
        }
    }


def find_optimal_eval(labels, probs):
    best_f1 = -1
    best_res = None
    for t in np.arange(0.02, 0.98, 0.02):
        res = eval_metrics_at_thresh(labels, probs, t)
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_res = res
    return best_res


def evaluate_previous_system(df, split_name="Test"):
    W = 12
    ds = WDSPhysicalGraphDataset(df, base_dir, window_size=W, use_physical_attributes=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    edge_index = ds.edge_index

    model = PhysicalGNN_GRU(
        node_dim=ds.F_node, edge_dim=ds.F_edge, 
        gnn_hidden=64, gru_hidden=128
    ).to(device)

    save_path = os.path.join(base_dir, 'models', 'gnn_gru', 'physical_gnn_gru_model.pth')
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    labels, probs = get_probs(model, loader, device, edge_index)
    return labels, probs


def evaluate_updated_zone_system(df, train_df, seq_len=11):
    zones = {
        0: ['F_PU4', 'F_PU5', 'F_PU6', 'F_PU7', 'L_T2', 'L_T3', 'L_T4', 'P_J256', 'P_J289', 'P_J300', 'P_J415', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7'],
        1: ['F_PU1', 'F_PU2', 'F_PU3', 'F_PU8', 'F_V2', 'L_T1', 'P_J14', 'P_J269', 'P_J280', 'P_J302', 'P_J422', 'S_PU1', 'S_PU2', 'S_PU3', 'S_PU8', 'S_V2'],
        2: ['F_PU10', 'F_PU11', 'L_T6', 'L_T7', 'P_J307', 'P_J317', 'S_PU10', 'S_PU11'],
        3: ['F_PU9', 'L_T5', 'P_J306', 'S_PU9']
    }

    from sklearn.preprocessing import StandardScaler
    target_col = 'ATT_FLAG' if 'ATT_FLAG' in df.columns else 'ATTACK_FLAG'
    labels = (df[target_col] != 0).astype(int).values[seq_len:]

    zone_tensors = {}
    for z, cols in zones.items():
        scaler = StandardScaler()
        scaler.fit(train_df[cols].values)
        scaled_eval = scaler.transform(df[cols].values)
        seqs = create_sequences(scaled_eval, seq_len)
        zone_tensors[z] = torch.tensor(seqs, dtype=torch.float32)

    adapters = {}
    for z in zones.keys():
        m = ZoneAdapter(input_dim=len(zones[z])).to(device)
        model_path = os.path.join(base_dir, 'models', 'gnn_gru', f'clean_adapter_{z}.pth')
        m.load_state_dict(torch.load(model_path, map_location=device))
        m.eval()
        adapters[z] = m

    global_model = GlobalZoneModel(num_zones=len(zones)).to(device)
    global_model.load_state_dict(torch.load(os.path.join(base_dir, 'models', 'gnn_gru', 'clean_global_zone.pth'), map_location=device))
    global_model.eval()

    n = len(labels)
    global_probs = []
    zone_probs = {z: [] for z in zones}

    with torch.no_grad():
        for i in range(0, n, 128):
            embs = []
            for z in zones.keys():
                xb = zone_tensors[z][i:i + 128].to(device)
                logits, emb_seq = adapters[z](xb)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                zone_probs[z].extend(probs)
                embs.append(emb_seq)
            xg = torch.stack(embs, dim=2)
            gp = torch.sigmoid(global_model(xg).squeeze(-1)).cpu().numpy()
            global_probs.extend(gp)

    return labels, np.array(global_probs)


def main():
    val_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'val_processed.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'test_processed.csv'))
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'train.csv'))

    results = {}

    print("=" * 60)
    print("EVALUATING PREVIOUS SYSTEM (Monolithic Physical GNN-GRU)")
    print("=" * 60)
    val_labels_prev, val_probs_prev = evaluate_previous_system(val_df, "Validation")
    test_labels_prev, test_probs_prev = evaluate_previous_system(test_df, "Test")

    prev_val_def = eval_metrics_at_thresh(val_labels_prev, val_probs_prev, thresh=0.5)
    prev_val_opt = find_optimal_eval(val_labels_prev, val_probs_prev)
    prev_test_def = eval_metrics_at_thresh(test_labels_prev, test_probs_prev, thresh=0.5)
    prev_test_opt = find_optimal_eval(test_labels_prev, test_probs_prev)

    print("=" * 60)
    print("EVALUATING UPDATED SYSTEM (Hierarchical Zone IDS)")
    print("=" * 60)
    val_labels_upd, val_probs_upd = evaluate_updated_zone_system(val_df, train_df)
    test_labels_upd, test_probs_upd = evaluate_updated_zone_system(test_df, train_df)

    upd_val_def = eval_metrics_at_thresh(val_labels_upd, val_probs_upd, thresh=0.5)
    upd_val_opt = find_optimal_eval(val_labels_upd, val_probs_upd)
    upd_test_def = eval_metrics_at_thresh(test_labels_upd, test_probs_upd, thresh=0.5)
    upd_test_opt = find_optimal_eval(test_labels_upd, test_probs_upd)

    summary = {
        "Validation": {
            "Previous_System": {
                "default_thresh_0.5": prev_val_def,
                "optimal_thresh": prev_val_opt
            },
            "Updated_System": {
                "default_thresh_0.5": upd_val_def,
                "optimal_thresh": upd_val_opt
            }
        },
        "Test": {
            "Previous_System": {
                "default_thresh_0.5": prev_test_def,
                "optimal_thresh": prev_test_opt
            },
            "Updated_System": {
                "default_thresh_0.5": upd_test_def,
                "optimal_thresh": upd_test_opt
            }
        }
    }

    out_file = os.path.join(base_dir, 'results', 'system_comparison_val_test.json')
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=4)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
