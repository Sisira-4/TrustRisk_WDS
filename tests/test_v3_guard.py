import sys, re
from pathlib import Path
import numpy as np
V3 = Path(__file__).resolve().parents[1] / "experiments" / "v3_residual"
sys.path.insert(0, str(V3))
from common import persist, ResidualDetector, margin_free_pool

# built by concatenation so this file does not trip its own scan
FORBIDDEN = ["BATADAL_" + "test", "test_" + "dataset", "splits/" + "test", "test_" + "processed", "test" + ".csv"]


def test_only_run_test_once_reads_the_test_file():
    for f in V3.glob("*.py"):
        if f.name == "run_test_once.py":
            continue
        txt = f.read_text().lower()
        for bad in FORBIDDEN:
            assert bad.lower() not in txt, f"{f.name} references the test set ({bad})"


def test_run_test_once_requires_confirmation_and_blocks_reruns():
    txt = (V3 / "run_test_once.py").read_text()
    assert "--confirm-test-use" in txt and "already exists" in txt


def test_persistence_is_causal():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 200)
    b = a.copy(); b[120:] = 1 - b[120:]              # change only the FUTURE
    assert (persist(a)[:120] == persist(b)[:120]).all()


def test_threshold_comes_from_training_scores_only():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 6)); X[:, 1] += 0.8 * X[:, 0]
    tr = np.arange(0, 400)
    det = ResidualDetector().fit(X, tr)
    thr = det.threshold(99)
    X2 = X.copy(); X2[400:] += 50                     # corrupt rows outside training
    det2 = ResidualDetector().fit(X2, tr)
    assert abs(det2.threshold(99) - thr) < 1e-9


def test_pool_excludes_margin_and_labelled_rows():
    lab = np.full(500, -999); lab[200:210] = 1
    pool = margin_free_pool(lab, 48)
    assert not ((pool >= 152) & (pool <= 257)).any() and 0 not in pool


def test_pipeline_modules_do_not_import_test_set():
    """Verify that integration and risk modules do not load test datasets directly."""
    for module_dir in ["integration", "risk"]:
        dir_path = Path(__file__).resolve().parents[1] / module_dir
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.py"):
            txt = f.read_text(encoding="utf-8").lower()
            for bad in ["batadal_test_dataset", "test_dataset_labeled"]:
                assert bad not in txt, f"{f.name} in {module_dir} references test set ({bad})"

