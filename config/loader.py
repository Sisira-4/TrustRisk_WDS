"""Authoritative Configuration Loader & Manifest Generator for TrustRisk-WDS.

Provides single-point access to config/trustrisk.yaml, calculates its SHA-256 hash,
synchronizes legacy/view JSON configs, and enforces global consistency across all modules.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "trustrisk.yaml"


def compute_sha256(file_path: Path) -> str:
    """Computes the hex SHA-256 digest of a given file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {file_path}")
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_config_hash() -> str:
    """Returns the cryptographic SHA-256 hash of config/trustrisk.yaml."""
    return compute_sha256(CONFIG_PATH)


def load_config(config_path: Path = None) -> dict:
    """Loads the authoritative YAML configuration and embeds the SHA-256 hash."""
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Authoritative configuration not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_sha256"] = compute_sha256(path)
    return cfg


def sync_views(cfg: dict = None) -> None:
    """Synchronizes frozen_config.json and experiments/v3_residual/config_v3.json as derived views."""
    if cfg is None:
        cfg = load_config()

    det = cfg["detector"]
    data = cfg["data"]

    # 1. Synchronize frozen_config.json
    frozen_dict = {
        "name": det["name"],
        "detector_type": "MultiSensorResidualDetector",
        "ridge_alpha": det["ridge_alpha"],
        "lags": det["lags"],
        "n_top": det["n_top"],
        "score_aggregation": det["score_aggregation"],
        "threshold_quantile": det["threshold_quantile"],
        "threshold_source": det["threshold_source"],
        "persistence": det["persistence"],
        "train_source": data["train_cross_validation"],
        "train_margin_hours": data["margin_hours"],
        "test_file": data["test_labeled"],
        "cv_performance": {
            "validation_protocol": data["cv_protocol"],
            "pooled_tpr": det["cv_performance"]["pooled_tpr"],
            "pooled_fpr_per_hour": det["cv_performance"]["pooled_fpr_per_hour"],
            "pooled_pointwise_f1": det["cv_performance"]["pooled_pointwise_f1"],
            "projected_f1": det["cv_performance"]["projected_f1"]
        },
        "frozen_at": det["frozen_at"],
        "status": det["status"],
        "checkpoint_file": det["checkpoint_file"],
        "checkpoint_sha256": det["checkpoint_sha256"],
        "authoritative_config_sha256": cfg["_config_sha256"]
    }
    frozen_path = ROOT / "frozen_config.json"
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen_dict, f, indent=2)

    # 2. Synchronize experiments/v3_residual/config_v3.json
    v3_dict = {
        "name": "v3_residual_ridge",
        "train": {
            "source": data["train_cross_validation"],
            "rows": "ATT_FLAG == -999 unlabelled provisional training pool (> margin_hours from labelled attacks; never treated as confirmed normal)",
            "margin_hours": data["margin_hours"]
        },
        "model": {
            "type": det["model_type"],
            "alpha": det["ridge_alpha"],
            "lags": det["lags"],
            "score": det["score_aggregation"],
            "n_top": det["n_top"]
        },
        "threshold": {
            "quantile": det["threshold_quantile"],
            "source": det["threshold_source"]
        },
        "persistence": det["persistence"],
        "test_file": data["test_labeled"],
        "notes": f"Adopted Configuration (Ridge alpha={det['ridge_alpha']}, Top-{det['n_top']}, Persistence {det['persistence']['k']}-of-{det['persistence']['n']}, q={det['threshold_quantile']}). Settings fixed in config/trustrisk.yaml before test evaluation.",
        "authoritative_config_sha256": cfg["_config_sha256"]
    }
    v3_path = ROOT / "experiments" / "v3_residual" / "config_v3.json"
    v3_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v3_path, "w", encoding="utf-8") as f:
        json.dump(v3_dict, f, indent=2)


def get_git_commit_hash() -> str:
    """Attempts to retrieve the current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def generate_run_manifest(output_path: Path = None) -> dict:
    """Generates results/run_manifest.json with hashes, package versions, and environment metadata."""
    cfg = load_config()
    out = output_path or (ROOT / "results" / "run_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Collect package versions
    packages = {}
    for pkg in ["numpy", "scipy", "pandas", "sklearn", "wntr", "yaml", "torch", "lightgbm", "matplotlib"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg] = "not_installed"

    # Compute raw data hashes
    data_hashes = {}
    for key, rel_path in cfg["data"].items():
        if isinstance(rel_path, str) and (rel_path.endswith(".csv") or rel_path.endswith(".INP")):
            fp = ROOT / rel_path
            if fp.exists():
                data_hashes[key] = {
                    "path": rel_path,
                    "sha256": compute_sha256(fp),
                    "size_bytes": fp.stat().st_size
                }

    manifest = {
        "manifest_version": "1.0",
        "git_commit": get_git_commit_hash(),
        "config_sha256": cfg["_config_sha256"],
        "config_path": "config/trustrisk.yaml",
        "seeds": cfg["seeds"],
        "data_hashes": data_hashes,
        "detector_checkpoint_sha256": cfg["detector"]["checkpoint_sha256"],
        "package_versions": packages,
        "output_paths": cfg["paths"]
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def attach_config_hash(report_dict: dict) -> dict:
    """Injects authoritative config_sha256 into a results dictionary before serialization."""
    report_dict["config_sha256"] = get_config_hash()
    return report_dict
