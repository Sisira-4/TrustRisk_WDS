import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score, confusion_matrix)

base_dir = os.path.dirname(os.path.abspath(__file__))
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


def main():
    print("--- INTEGRATING CLEAN LEAK-FREE ZONE IDS INTO TRUST PIPELINE ---")

    train_dir = os.path.join(base_dir, "data", "zones", "train")
    test_dir = os.path.join(base_dir, "data", "zones", "test")

    # Load data using train scalers
    _, _, zones, _, scalers = preprocess_zone_data(train_dir, seq_len=11)
    test_data, test_labels, _, _, _ = preprocess_zone_data(test_dir, seq_len=11, scalers=scalers)

    # Load trained models
    adapters = {}
    for z in zones:
        m = ZoneAdapter(input_dim=test_data[z].shape[-1]).to(device)
        m.load_state_dict(torch.load(os.path.join(base_dir, "models", "gnn_gru", f"clean_adapter_{z}.pth"), map_location=device))
        m.eval()
        adapters[z] = m

    global_model = GlobalZoneModel(num_zones=len(zones)).to(device)
    global_model.load_state_dict(torch.load(os.path.join(base_dir, "models", "gnn_gru", "clean_global_zone.pth"), map_location=device))
    global_model.eval()

    # Compute inference probabilities
    n_test = len(test_labels)
    batch_size = 128
    global_probs = []
    zone_probs = {z: [] for z in zones}

    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            embs_batch = []
            for z in zones:
                xb = test_data[z][i:i + batch_size].to(device)
                logits, emb_seq = adapters[z](xb)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                zone_probs[z].extend(probs)
                embs_batch.append(emb_seq)

            xg = torch.stack(embs_batch, dim=2)
            g_logits = global_model(xg)
            g_p = torch.sigmoid(g_logits.squeeze(-1)).cpu().numpy()
            global_probs.extend(g_p)

    p_clean = np.array(global_probs)
    print(f"Computed clean probabilities: min={p_clean.min():.4f}, max={p_clean.max():.4f}, mean={p_clean.mean():.4f}")

    # Update trust_pipeline_virgin.csv
    virgin_path = os.path.join(base_dir, "experiments", "final", "trust_pipeline_virgin.csv")
    df_v = pd.read_csv(virgin_path)
    df_v["p_attack"] = p_clean
    df_v["p_attack_gnn"] = p_clean
    df_v.to_csv(virgin_path, index=False)
    print(f"Updated {virgin_path}")

    # Update trust_pipeline.csv
    pipeline_path = os.path.join(base_dir, "experiments", "final", "trust_pipeline.csv")
    df_p = pd.read_csv(pipeline_path)
    df_p["p_attack"] = p_clean
    df_p["p_attack_gnn"] = p_clean
    df_p["p_fused"] = p_clean
    df_p["C"] = np.maximum(p_clean, 1 - p_clean)
    df_p["Trust_Score"] = df_p["C"] * (1 - df_p["U_norm"]) * df_p["ECP"]
    df_p.to_csv(pipeline_path, index=False)
    print(f"Updated {pipeline_path}")

    # Calculate metrics with a clean threshold of 0.50
    thresh = 0.50
    preds = (p_clean >= thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
    p_prec = precision_score(test_labels, preds, zero_division=0)
    p_rec = recall_score(test_labels, preds, zero_division=0)
    p_f1 = f1_score(test_labels, preds, zero_division=0)
    acc = accuracy_score(test_labels, preds)
    roc = roc_auc_score(test_labels, p_clean)
    pr_auc = average_precision_score(test_labels, p_clean)

    pa_preds = point_adjust(test_labels, preds)
    pa_tn, pa_fp, pa_fn, pa_tp = confusion_matrix(test_labels, pa_preds).ravel()
    pa_prec = precision_score(test_labels, pa_preds, zero_division=0)
    pa_rec = recall_score(test_labels, pa_preds, zero_division=0)
    pa_f1 = f1_score(test_labels, pa_preds, zero_division=0)

    edr, mean_delay, det_ev, tot_ev, fa = get_event_metrics(test_labels, preds)

    print("\n" + "=" * 50)
    print("NEW PIPELINE PERFORMANCE (Zero Hack, Leak-Free Zone Model):")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"Pointwise Precision: {p_prec:.4f} | Recall: {p_rec:.4f} | F1: {p_f1:.4f}")
    print(f"PA Precision:        {pa_prec:.4f} | PA Recall: {pa_rec:.4f} | PA F1: {pa_f1:.4f}")
    print(f"Event Detection Rate: {edr:.4f} ({det_ev}/{tot_ev})")
    print(f"False Alarm Events:   {fa}")
    print(f"Mean Detection Delay: {mean_delay:.2f} frames")
    print("=" * 50)

    # Save to final_metrics.json
    final_metrics = {
        "Accuracy_PA": round(acc, 4),
        "F1_Score_PA": round(pa_f1, 4),
        "Precision_PA": round(pa_prec, 4),
        "Recall_PA": round(pa_rec, 4),
        "Pointwise_F1": round(p_f1, 4),
        "Pointwise_Precision": round(p_prec, 4),
        "Pointwise_Recall": round(p_rec, 4),
        "ROC_AUC": round(roc, 4),
        "PR_AUC": round(pr_auc, 4),
        "Event_Detection_Rate": round(edr, 4),
        "Detected_Events": int(det_ev),
        "Total_Events": int(tot_ev),
        "Mean_Detection_Delay": round(mean_delay, 2),
        "False_Alarm_Events": int(fa),
        "Confusion_Matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ],
        "Threshold": thresh
    }

    metrics_path = os.path.join(base_dir, "experiments", "final", "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
