"""Fits and serializes the frozen multi-sensor Ridge residual detector.

Produces models/detector/frozen_detector_checkpoint.pkl and calculates its SHA-256 hash.
Downstream trust layers and dashboard load this checkpoint and verify its hash.
"""
from pathlib import Path
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DETECTOR_DIR = ROOT / "models" / "detector"
DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = DETECTOR_DIR / "frozen_detector_checkpoint.pkl"


def load_series(path):
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = d.columns.str.strip()
    return d


def sensor_columns(df):
    return [c for c in df.columns if c not in ("ATT_FLAG", "ATTACK_FLAG", "DATETIME")]


def runs(mask):
    idx = np.flatnonzero(np.asarray(mask))
    if idx.size == 0:
        return []
    return [(int(g[0]), int(g[-1])) for g in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)]


def margin_free_pool(labels, margin=48):
    labels = np.asarray(labels)
    near = np.zeros(len(labels), bool)
    for s, e in runs(labels == 1):
        near[max(0, s - margin) : e + margin + 1] = True
    pool = np.flatnonzero((labels == -999) & ~near)
    return pool[pool > 0]


def save_frozen_checkpoint():
    print("=" * 70)
    print("   SERIALIZING FROZEN DETECTOR CHECKPOINT (PHASE 2)")
    print("=" * 70)

    cfg_path = ROOT / "frozen_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    d4_path = ROOT / cfg.get("train_source", "data/raw/BATADAL_dataset04.csv")
    d4 = load_series(d4_path)
    sensor_names = sensor_columns(d4)

    pool = margin_free_pool(d4["ATT_FLAG"].values, margin=cfg.get("train_margin_hours", 48))
    X_pool = d4.loc[pool, sensor_names].values.astype(float)

    # Normalization parameters across all sensors
    mu_raw = X_pool.mean(axis=0)
    sd_raw = X_pool.std(axis=0)
    keep = sd_raw > 1e-6
    mu, sd = mu_raw[keep], sd_raw[keep]
    names = [n for n, k in zip(sensor_names, keep) if k]

    Z_pool = ((X_pool[:, keep] - mu) / sd)
    Z_lag_pool = np.vstack([Z_pool[:1], Z_pool[:-1]])

    import sys
    sys.path.insert(0, str(ROOT / "experiments" / "v3_residual"))
    from common import ResidualDetector

    alpha = cfg.get("ridge_alpha", 10.0)
    n_top = cfg.get("n_top", 3)

    det = ResidualDetector(alpha=alpha, n_top=n_top).fit(
        d4[sensor_names].values.astype(float),
        pool,
        names=sensor_names
    )
    threshold = det.threshold(cfg.get("threshold_quantile", 99.0))

    checkpoint_payload = {
        "model_name": cfg.get("name", "v3_residual_ridge_alpha10"),
        "sensor_names": det.names,
        "mu": det.mu.tolist(),
        "sd": det.sd.tolist(),
        "keep": det.keep.tolist(),
        "r_mu": det.r_mu.tolist(),
        "r_sd": det.r_sd.tolist(),
        "models": [{"coef": m.coef_.tolist(), "intercept": float(m.intercept_)} for m in det.models],
        "alpha": float(alpha),
        "n_top": int(n_top),
        "threshold": threshold,
        "frozen_timestamp": cfg.get("frozen_at", "2026-10-11T15:39:00+05:30")
    }

    # Serialize payload
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(checkpoint_payload, f)

    # Compute SHA-256 hash
    with open(CHECKPOINT_PATH, "rb") as f:
        chk_hash = hashlib.sha256(f.read()).hexdigest()

    print(f">> Saved checkpoint to {CHECKPOINT_PATH}")
    print(f">> Checkpoint SHA-256: {chk_hash}")
    print(f">> Threshold: {threshold:.4f} across {len(names)} sensors")

    # Update frozen_config.json with checkpoint path and hash
    cfg["checkpoint_file"] = "models/detector/frozen_detector_checkpoint.pkl"
    cfg["checkpoint_sha256"] = chk_hash

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f">> Updated {cfg_path} with checkpoint hash.")

    return chk_hash


if __name__ == "__main__":
    save_frozen_checkpoint()
