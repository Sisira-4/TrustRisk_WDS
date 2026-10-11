import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_sequences(data, seq_len):
    """Creates sliding window sequences of shape [N - seq_len, seq_len, num_features]."""
    seqs = []
    for i in range(len(data) - seq_len):
        seqs.append(data[i:i + seq_len])
    return np.array(seqs, dtype=np.float32)


def preprocess_zone_data(data_dir, seq_len=10, scalers=None):
    """
    Loads zone datasets with STRICT label separation to prevent data leakage.
    - Excludes 'ATTACK_FLAG', 'ATT_FLAG', and 'DATETIME' from feature columns.
    - Fits StandardScaler only if scalers is None (e.g. during training), otherwise transforms using provided scalers.
    """
    zone_data = {}
    labels = None
    feature_cols_map = {}
    is_fit = scalers is None
    if is_fit:
        scalers = {}

    zones = [0, 1, 2, 3]

    for z in zones:
        csv_path = os.path.join(data_dir, f"zone{z}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Zone CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)

        # Extract labels from the first zone processed
        if labels is None:
            if "ATTACK_FLAG" in df.columns:
                labels = (df["ATTACK_FLAG"] != 0).astype(int).values
            elif "ATT_FLAG" in df.columns:
                labels = (df["ATT_FLAG"] != 0).astype(int).values
            else:
                raise ValueError("No attack flag column found in dataset!")

        # Strictly exclude target labels and non-feature columns
        exclude_cols = {"ATTACK_FLAG", "ATT_FLAG", "DATETIME", "Timestamp"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        feature_cols_map[z] = feature_cols

        data = df[feature_cols].select_dtypes(include=[np.number]).values

        if is_fit:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            scalers[z] = scaler
        else:
            data = scalers[z].transform(data)

        seqs = create_sequences(data, seq_len)
        zone_data[z] = torch.tensor(seqs, dtype=torch.float32)

    # Align labels with sliding sequence offset
    labels = labels[seq_len:]

    return zone_data, labels, zones, feature_cols_map, scalers


def verify_clean_shapes(data_dir="data/zones/train", seq_len=10):
    """Sanity check to guarantee feature dimensions are free of label leakage."""
    zone_data, labels, zones, feature_cols_map, _ = preprocess_zone_data(data_dir, seq_len=seq_len)
    expected_dims = {0: 15, 1: 16, 2: 8, 3: 4}
    print("--- VERIFYING CLEAN SHAPES (NO LEAKAGE) ---")
    for z in zones:
        f_dim = zone_data[z].shape[-1]
        print(f"Zone {z}: shape {zone_data[z].shape} (features: {f_dim})")
        assert f_dim == expected_dims[z], f"Mismatch for Zone {z}: expected {expected_dims[z]}, got {f_dim}"
        assert "ATTACK_FLAG" not in feature_cols_map[z]
        assert "ATT_FLAG" not in feature_cols_map[z]
    print("All zones verified: Zero label leakage confirmed!\n")
    return True


class ZoneAdapter(nn.Module):
    """Local zone temporal encoder producing local predictions and dynamic embeddings."""
    def __init__(self, input_dim, hidden_dim=128, embed_dim=32, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_embed = nn.Linear(hidden_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.gru(x)               # [B, T, hidden_dim]
        out = self.dropout(out)
        emb_seq = self.fc_embed(out)       # [B, T, embed_dim]
        logits = self.classifier(emb_seq[:, -1])  # Classify using last time step
        return logits, emb_seq


class SimpleZoneGNN(nn.Module):
    """GNN layer over inter-zone graph representation."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


class GlobalZoneModel(nn.Module):
    """Hierarchical model fusing zone temporal embeddings across the network topology."""
    def __init__(self, embed_dim=32, hidden_dim=64, num_zones=4, dropout=0.2):
        super().__init__()
        self.gnn = SimpleZoneGNN(embed_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim * num_zones, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, Z, F]
        B, T, Z, F = x.shape
        out_seq = []
        for t in range(T):
            xt = x[:, t]              # [B, Z, F]
            xt = self.gnn(xt)         # [B, Z, H]
            out_seq.append(xt)

        x = torch.stack(out_seq, dim=1)   # [B, T, Z, H]
        x = x.view(B, T, -1)              # Flatten zones across channels: [B, T, Z*H]
        out, _ = self.gru(x)              # [B, T, H]
        out = self.dropout(out[:, -1])    # Last timestep
        return self.fc(out)
