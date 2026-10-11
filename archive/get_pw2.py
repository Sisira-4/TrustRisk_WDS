import os, torch, numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.train_gnn_gru_physical import PhysicalGNN_GRU, get_probs

device = torch.device('cpu')
base_dir = os.path.abspath('.')
data_dir = os.path.join(base_dir, 'data', 'processed')
W = 12
save_path = os.path.join(base_dir, 'models', 'gnn_gru', 'physical_gnn_gru_model_C.pth')

for split in ['val', 'test']:
    ds = WDSPhysicalGraphDataset(pd.read_csv(os.path.join(data_dir, f'{split}_processed.csv')), base_dir, window_size=W, use_physical_attributes=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    model = PhysicalGNN_GRU(node_dim=ds.F_node, edge_dim=ds.F_edge, gnn_hidden=64, gru_hidden=128).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    labels, probs = get_probs(model.eval(), loader, device, ds.edge_index)
    
    best_f1, best_thr = 0, 0
    for t in np.arange(0.01, 1.0, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, t
            
    preds = (probs >= best_thr).astype(int)
    print(f'{split.upper()} F1: {f1_score(labels, preds, zero_division=0):.4f}')
    print(f'{split.upper()} Precision: {precision_score(labels, preds, zero_division=0):.4f}')
    print(f'{split.upper()} Recall: {recall_score(labels, preds, zero_division=0):.4f}')
