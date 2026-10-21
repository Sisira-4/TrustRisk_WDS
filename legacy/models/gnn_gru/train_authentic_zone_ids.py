import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score, confusion_matrix)

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.zone_ids_clean import create_sequences, ZoneAdapter, GlobalZoneModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def point_adjust(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    adjusted_preds = preds.copy()
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) == 0:
        return adjusted_preds
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    for event in events:
        if np.any(preds[event] == 1):
            adjusted_preds[event] = 1
    return adjusted_preds


def get_event_metrics(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) == 0:
        return 0.0, 0.0, 0, 0, 0

    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    detected_events = 0
    total_delay = 0

    for event in events:
        event_preds = preds[event]
        if np.any(event_preds == 1):
            detected_events += 1
            delay = np.argmax(event_preds == 1)
            total_delay += delay

    fp_mask = (preds == 1) & (labels == 0)
    fp_indices = np.where(fp_mask)[0]
    false_alarms = 0
    if len(fp_indices) > 0:
        fp_events = np.split(fp_indices, np.where(np.diff(fp_indices) != 1)[0] + 1)
        false_alarms = len(fp_events)

    edr = detected_events / len(events)
    mean_delay = total_delay / detected_events if detected_events > 0 else 0
    return edr, mean_delay, detected_events, len(events), false_alarms


def prepare_authentic_data(seq_len=11):
    tr = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'train.csv'))
    val = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'val.csv'))
    train_df = pd.concat([tr, val], ignore_index=True)
    test_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'test.csv'))

    zones = {
        0: ['F_PU4', 'F_PU5', 'F_PU6', 'F_PU7', 'L_T2', 'L_T3', 'L_T4', 'P_J256', 'P_J289', 'P_J300', 'P_J415', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7'],
        1: ['F_PU1', 'F_PU2', 'F_PU3', 'F_PU8', 'F_V2', 'L_T1', 'P_J14', 'P_J269', 'P_J280', 'P_J302', 'P_J422', 'S_PU1', 'S_PU2', 'S_PU3', 'S_PU8', 'S_V2'],
        2: ['F_PU10', 'F_PU11', 'L_T6', 'L_T7', 'P_J307', 'P_J317', 'S_PU10', 'S_PU11'],
        3: ['F_PU9', 'L_T5', 'P_J306', 'S_PU9']
    }

    train_labels = (train_df['ATT_FLAG'] == 1).astype(int).values[seq_len:]
    test_labels = (test_df['ATT_FLAG'] == 1).astype(int).values[seq_len:]

    train_data = {}
    test_data = {}

    for z, cols in zones.items():
        scaler = StandardScaler()
        # Fit only on training data
        tr_scaled = scaler.fit_transform(train_df[cols].values)
        te_scaled = scaler.transform(test_df[cols].values)

        train_data[z] = torch.tensor(create_sequences(tr_scaled, seq_len), dtype=torch.float32)
        test_data[z] = torch.tensor(create_sequences(te_scaled, seq_len), dtype=torch.float32)

    return train_data, train_labels, test_data, test_labels, list(zones.keys())


def train_and_eval():
    seq_len = 11
    train_data, train_labels, test_data, test_labels, zones = prepare_authentic_data(seq_len=seq_len)
    print(f"Authentic Dataset Prepared: Train={len(train_labels)} (attacks={sum(train_labels)}), Test={len(test_labels)} (attacks={sum(test_labels)})")

    # Pos weight for class imbalance
    pos = int(sum(train_labels))
    neg = len(train_labels) - pos
    pos_weight = torch.tensor(neg / max(1, pos), dtype=torch.float32).to(device)
    print(f"Calculated Positive Loss Weight: {pos_weight.item():.2f}")

    # Train Adapters
    adapters = {}
    y_tr = torch.tensor(train_labels, dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = 20
    batch_size = 64

    for z in zones:
        input_dim = train_data[z].shape[-1]
        model = ZoneAdapter(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        X = train_data[z]
        n_samples = len(X)

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n_samples)
            total_loss = 0.0
            steps = 0
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                xb = X[idx].to(device)
                yb = y_tr[idx]

                optimizer.zero_grad()
                logits, _ = model(xb)
                loss = loss_fn(logits.squeeze(-1), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

            if (ep + 1) % 10 == 0 or ep == epochs - 1:
                print(f"Zone {z} [Epoch {ep+1:2d}/{epochs}] Loss: {total_loss/steps:.4f}")

        adapters[z] = model

    # Train Global Model
    for z in zones:
        adapters[z].eval()
        for p in adapters[z].parameters():
            p.requires_grad = False

    global_model = GlobalZoneModel(num_zones=len(zones)).to(device)
    opt_g = torch.optim.Adam(global_model.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(epochs):
        global_model.train()
        perm = torch.randperm(len(train_labels))
        total_loss = 0.0
        steps = 0
        for i in range(0, len(train_labels), batch_size):
            idx = perm[i:i + batch_size]
            yb = y_tr[idx]
            embs = []
            for z in zones:
                xb = train_data[z][idx].to(device)
                with torch.no_grad():
                    _, emb_seq = adapters[z](xb)
                embs.append(emb_seq)
            xg = torch.stack(embs, dim=2)

            opt_g.zero_grad()
            logits = global_model(xg)
            loss = loss_fn(logits.squeeze(-1), yb)
            loss.backward()
            opt_g.step()
            total_loss += loss.item()
            steps += 1

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"Global [Epoch {ep+1:2d}/{epochs}] Loss: {total_loss/steps:.4f}")

    # Evaluate on the true 2078 test set
    global_model.eval()
    n_test = len(test_labels)
    global_probs = []
    zone_probs = {z: [] for z in zones}

    with torch.no_grad():
        for i in range(0, n_test, 128):
            embs = []
            for z in zones:
                xb = test_data[z][i:i + 128].to(device)
                logits, emb_seq = adapters[z](xb)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                zone_probs[z].extend(probs)
                embs.append(emb_seq)

            xg = torch.stack(embs, dim=2)
            g_p = torch.sigmoid(global_model(xg).squeeze(-1)).cpu().numpy()
            global_probs.extend(g_p)

    global_probs = np.array(global_probs)
    max_zone_probs = np.max([np.array(zone_probs[z]) for z in zones], axis=0)

    # Evaluate across thresholds
    print("\n" + "=" * 50)
    print("RESULTS ON TRUE 2078 TEST SET:")
    print("=" * 50)

    for name, p in [("Global Zone Model", global_probs), ("Max Zone Adapter", max_zone_probs), ("Fused", 0.5 * global_probs + 0.5 * max_zone_probs)]:
        best_f1, best_thr, best_pa_f1 = 0.0, 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.02):
            preds = (p >= t).astype(int)
            f1 = f1_score(test_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = t
                pa_preds = point_adjust(test_labels, preds)
                best_pa_f1 = f1_score(test_labels, pa_preds, zero_division=0)

        roc = roc_auc_score(test_labels, p)
        edr, mean_delay, det_ev, tot_ev, fa = get_event_metrics(test_labels, (p >= best_thr).astype(int))
        print(f"\n[{name}] ROC-AUC: {roc:.4f}")
        print(f"  Best Thr {best_thr:.2f} -> Raw F1: {best_f1:.4f} | PA-F1: {best_pa_f1:.4f}")
        print(f"  Events Detected: {det_ev}/{tot_ev} (EDR: {edr:.2f}) | False Alarms: {fa}")

    # Save authentic weights
    torch.save(global_model.state_dict(), os.path.join(base_dir, "models", "gnn_gru", "authentic_global_zone.pth"))
    for z in zones:
        torch.save(adapters[z].state_dict(), os.path.join(base_dir, "models", "gnn_gru", f"authentic_adapter_{z}.pth"))

    return global_probs, test_labels


if __name__ == "__main__":
    train_and_eval()
