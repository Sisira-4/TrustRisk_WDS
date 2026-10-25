"""Tests Phase C7: Cryptographic checkpoint identity verification and tamper rejection."""
import sys
import tempfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trust.unified_trust_engine import verify_checkpoint_hash, load_checkpoint
from config.loader import load_config


def test_checkpoint_hash_matches_authoritative_config():
    """Verify C7: Checkpoint SHA-256 matches frozen_config.json and trustrisk.yaml exactly."""
    cfg = load_config()
    expected_hash = cfg["detector"]["checkpoint_sha256"]
    
    actual_hash = verify_checkpoint_hash()
    assert actual_hash == expected_hash, f"Checkpoint hash mismatch: {actual_hash} != {expected_hash}"


def test_checkpoint_loader_loads_valid_structure():
    """Verify C7: Checkpoint loader deserializes all expected model components."""
    chk = load_checkpoint()
    assert isinstance(chk, dict)
    required_keys = ["model_name", "sensor_names", "mu", "sd", "keep", "r_mu", "r_sd", "models", "alpha", "n_top", "threshold"]
    for k in required_keys:
        assert k in chk, f"Missing key '{k}' in verified detector checkpoint"


def test_checkpoint_loader_rejects_corrupted_checkpoint(monkeypatch):
    """Verify C7: Loader strictly rejects tampered/corrupted checkpoint."""
    import json
    orig_load = json.load

    def fake_json_load(fp):
        data = orig_load(fp)
        if isinstance(data, dict) and "checkpoint_sha256" in data:
            data = data.copy()
            data["checkpoint_sha256"] = "0" * 64
        return data

    monkeypatch.setattr(json, "load", fake_json_load)

    with pytest.raises(ValueError, match="Checkpoint hash mismatch"):
        verify_checkpoint_hash()

