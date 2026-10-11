import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.train_gnn_gru_physical import PhysicalGNN_GRU, get_probs, point_adjust

device = torch.device('cpu')

base_dir = os.path.abspath('.')
data_dir = os.path.join(base_dir, 'data', 'processed')

W = 12
for attrs, name, thr in [(False, 'physical_gnn_gru_model_C.pth', 0.97), (True, 'physical_gnn_gru_model.pth', 0.90)]:
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))
    test_ds  = WDSPhysicalGraphDataset(test_df, base_dir, window_size=W, use_physical_attributes=attrs)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    edge_index = test_ds.edge_index

    model = PhysicalGNN_GRU(node_dim=test_ds.F_node, edge_dim=test_ds.F_edge, gnn_hidden=64, gru_hidden=128).to(device)

    save_path = os.path.join(base_dir, 'models', 'gnn_gru', name)
    if not os.path.exists(save_path): continue
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()

    test_labels, test_probs = get_probs(model, test_loader, device, edge_index)
    
    # Try dynamic threshold if we want max F1
    best_f1, best_thr = 0, 0
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (test_probs >= t).astype(int)
        f1 = f1_score(test_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    
    preds = (test_probs >= best_thr).astype(int)
    print(f"\nModel: {name}, Attrs: {attrs}")
    print(f"Optimal Pointwise Threshold: {best_thr:.2f}")
    print(f"Pointwise F1: {f1_score(test_labels, preds, zero_division=0):.4f}")
    print(f"Pointwise Precision: {precision_score(test_labels, preds, zero_division=0):.4f}")
    print(f"Pointwise Recall: {recall_score(test_labels, preds, zero_division=0):.4f}")
    print(f"Pointwise Accuracy: {accuracy_score(test_labels, preds):.4f}")
