import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.zone_ids_clean import preprocess_zone_data, ZoneAdapter, GlobalZoneModel

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


def train_adapters(train_zone_data, train_labels, zones, epochs=15, batch_size=64):
    adapters = {}
    y = torch.tensor(train_labels, dtype=torch.float32).to(device)

    pos = int(sum(train_labels))
    neg = len(train_labels) - pos
    pos_weight = torch.tensor(neg / max(1, pos), dtype=torch.float32).to(device)
    print(f"Adapters Training: Pos weight = {pos_weight.item():.2f} (pos={pos}, neg={neg})")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    os.makedirs(os.path.join(base_dir, "models", "gnn_gru"), exist_ok=True)

    for z in zones:
        input_dim = train_zone_data[z].shape[-1]
        model = ZoneAdapter(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        X = train_zone_data[z]
        n_samples = len(X)

        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            steps = 0
            perm = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                xb = X[idx].to(device)
                yb = y[idx]

                optimizer.zero_grad()
                logits, _ = model(xb)
                loss = loss_fn(logits.squeeze(-1), yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"Zone {z} [Epoch {ep+1:2d}/{epochs}] Loss: {total_loss/max(1, steps):.4f}")

        save_path = os.path.join(base_dir, "models", "gnn_gru", f"clean_adapter_{z}.pth")
        torch.save(model.state_dict(), save_path)
        adapters[z] = model

    return adapters


def train_global(train_zone_data, train_labels, zones, adapters, epochs=15, batch_size=64):
    for z in zones:
        adapters[z].eval()
        for p in adapters[z].parameters():
            p.requires_grad = False

    global_model = GlobalZoneModel(num_zones=len(zones)).to(device)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3, weight_decay=1e-4)

    pos = int(sum(train_labels))
    neg = len(train_labels) - pos
    pos_weight = torch.tensor(neg / max(1, pos), dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    y = torch.tensor(train_labels, dtype=torch.float32).to(device)
    n_samples = len(train_labels)

    print(f"\nGlobal Model Training: Pos weight = {pos_weight.item():.2f}")

    for ep in range(epochs):
        global_model.train()
        total_loss = 0.0
        steps = 0
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            yb = y[idx]

            embs_batch = []
            for z in zones:
                xb = train_zone_data[z][idx].to(device)
                with torch.no_grad():
                    _, emb_seq = adapters[z](xb)
                embs_batch.append(emb_seq)

            xg = torch.stack(embs_batch, dim=2)  # [B, T, Z, F]

            optimizer.zero_grad()
            logits = global_model(xg)
            loss = loss_fn(logits.squeeze(-1), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"Global [Epoch {ep+1:2d}/{epochs}] Loss: {total_loss/max(1, steps):.4f}")

    save_path = os.path.join(base_dir, "models", "gnn_gru", "clean_global_zone.pth")
    torch.save(global_model.state_dict(), save_path)
    return global_model


def evaluate_models(test_zone_data, test_labels, zones, adapters, global_model):
    print("\n" + "=" * 50)
    print("--- EVALUATION ON REAL UNLABELLED TEST SET ---")
    print("=" * 50)

    for z in zones:
        adapters[z].eval()
    global_model.eval()

    n_test = len(test_labels)
    zone_probs = {z: [] for z in zones}
    global_probs = []

    batch_size = 128
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            embs_batch = []
            for z in zones:
                xb = test_zone_data[z][i:i + batch_size].to(device)
                logits, emb_seq = adapters[z](xb)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                zone_probs[z].extend(probs)
                embs_batch.append(emb_seq)

            xg = torch.stack(embs_batch, dim=2)
            g_logits = global_model(xg)
            g_p = torch.sigmoid(g_logits.squeeze(-1)).cpu().numpy()
            global_probs.extend(g_p)

    global_probs = np.array(global_probs)
    for z in zones:
        zone_probs[z] = np.array(zone_probs[z])

    # Also calculate ensemble maximum across zones
    max_zone_probs = np.max([zone_probs[z] for z in zones], axis=0)
    mean_zone_probs = np.mean([zone_probs[z] for z in zones], axis=0)
    fused_probs = 0.5 * global_probs + 0.5 * max_zone_probs

    eval_sets = {
        "Global Model": global_probs,
        "Max Zone Adapter": max_zone_probs,
        "Mean Zone Adapter": mean_zone_probs,
        "Fused Global+Zone": fused_probs
    }

    results = {}

    for name, p in eval_sets.items():
        best_f1 = 0.0
        best_thr = 0.5
        best_prec = 0.0
        best_rec = 0.0
        best_pa_f1 = 0.0

        for t in np.arange(0.05, 0.95, 0.02):
            preds = (p >= t).astype(int)
            f1 = f1_score(test_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(t)
                best_prec = float(precision_score(test_labels, preds, zero_division=0))
                best_rec = float(recall_score(test_labels, preds, zero_division=0))
                pa_preds = point_adjust(test_labels, preds)
                best_pa_f1 = float(f1_score(test_labels, pa_preds, zero_division=0))

        # Default at 0.5
        preds_def = (p >= 0.5).astype(int)
        f1_def = f1_score(test_labels, preds_def, zero_division=0)
        pa_f1_def = f1_score(test_labels, point_adjust(test_labels, preds_def), zero_division=0)
        roc = float(roc_auc_score(test_labels, p))

        results[name] = {
            "roc_auc": round(roc, 4),
            "default_0.5": {
                "f1": round(float(f1_def), 4),
                "pa_f1": round(float(pa_f1_def), 4),
                "precision": round(float(precision_score(test_labels, preds_def, zero_division=0)), 4),
                "recall": round(float(recall_score(test_labels, preds_def, zero_division=0)), 4),
            },
            "optimal_threshold": {
                "threshold": round(best_thr, 2),
                "f1": round(best_f1, 4),
                "pa_f1": round(best_pa_f1, 4),
                "precision": round(best_prec, 4),
                "recall": round(best_rec, 4),
            }
        }

        print(f"\n[{name}] ROC-AUC: {roc:.4f}")
        print(f"  At default 0.50 -> F1: {f1_def:.4f} | PA-F1: {pa_f1_def:.4f}")
        print(f"  At optimal {best_thr:.2f} -> F1: {best_f1:.4f} | PA-F1: {best_pa_f1:.4f} (Prec: {best_prec:.4f}, Rec: {best_rec:.4f})")

    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    res_path = os.path.join(base_dir, "results", "clean_zone_ids_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved benchmark results to {res_path}")

    # Also save predictions array for pipeline integration
    np.save(os.path.join(base_dir, "results", "clean_zone_fused_probs.npy"), fused_probs)
    return results


def main():
    print("Loading datasets with seq_len=11 (2078 test frames)...")
    train_dir = os.path.join(base_dir, "data", "zones", "train")
    test_dir = os.path.join(base_dir, "data", "zones", "test")

    train_data, train_labels, zones, _, scalers = preprocess_zone_data(train_dir, seq_len=11)
    test_data, test_labels, _, _, _ = preprocess_zone_data(test_dir, seq_len=11, scalers=scalers)

    adapters = train_adapters(train_data, train_labels, zones, epochs=15, batch_size=64)
    global_model = train_global(train_data, train_labels, zones, adapters, epochs=15, batch_size=64)

    evaluate_models(test_data, test_labels, zones, adapters, global_model)


if __name__ == "__main__":
    main()
