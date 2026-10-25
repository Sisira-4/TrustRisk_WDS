import os
import glob
import re

def test_no_test_set_leakage():
    # Strict regex pattern for any mention of the test sets
    v2_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments', 'v2')
    py_files = glob.glob(os.path.join(v2_dir, '**', '*.py'), recursive=True)
    
    suspicious_patterns = [
        r'test_dataset',
        r'BATADAL_test',
        r'test\.csv',
        r'test_processed',
        r'splits[\\/]test',
        r'true_label'
    ]
    
    allowed_files = ['step5_final_evaluation.py']
    
    for fpath in py_files:
        basename = os.path.basename(fpath)
        if basename in allowed_files:
            continue
            
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
            for pattern in suspicious_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                assert not match, f"Leakage detected in {basename}: found pattern '{pattern}'"

def test_scaler_fitting_inside_folds():
    # Attempt to catch generic scaler.fit() that might be reading raw data outside a fold loop
    # We enforce that fit() is used on fold data. (This is a heuristic structural check)
    v2_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments', 'v2')
    py_files = glob.glob(os.path.join(v2_dir, '**', '*.py'), recursive=True)
    
    for fpath in py_files:
        basename = os.path.basename(fpath)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
            # If standardscaler or similar is used, ensure it is fit on train_df
            if 'StandardScaler' in content:
                # We expect to see something like scaler.fit(train_df...)
                assert re.search(r'\.fit(_transform)?\(.*(train|_tr)', content, re.IGNORECASE) or 'step1' in basename, \
                    f"Scaler fitted improperly or without explicit 'train' reference in {basename}."

def test_negative_999_never_mapped_to_normal_in_folds():
    """Verify that in all CV folds, rows with ATT_FLAG == -999 are not marked as normal (0)."""
    import pandas as pd
    root = os.path.dirname(os.path.dirname(__file__))
    v2_data = os.path.join(root, 'data', 'v2')
    if not os.path.exists(v2_data):
        return
        
    for fname in os.listdir(v2_data):
        if fname.startswith('fold_') and fname.endswith('_val.csv'):
            fpath = os.path.join(v2_data, fname)
            df = pd.read_csv(fpath)
            if 'ATT_FLAG' in df.columns and 'LABEL' in df.columns:
                unlabelled = df[df['ATT_FLAG'] == -999]
                assert not (unlabelled['LABEL'] == 0).any(), f"Found -999 rows cast to 0 in {fname}"

def test_data_card_consistency():
    """Verify that results/data_card.md exists and accurately reflects the raw data."""
    root = os.path.dirname(os.path.dirname(__file__))
    card_path = os.path.join(root, 'results', 'data_card.md')
    assert os.path.exists(card_path), "results/data_card.md must exist (run experiments/data_card.py)"
    with open(card_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "8761" in content, "Dataset 03 row count missing"
    assert "4177" in content, "Dataset 04 row count missing"
    assert "3958" in content, "Dataset 04 unlabelled (-999) count missing"
    assert "2089" in content, "Test set row count missing"
    assert "-999" in content and "Unlabelled" in content


def test_preprocessing_never_maps_negative_999_to_zero():
    """Verify that preprocessing/build_dataset.py never maps -999 to 0."""
    root = os.path.dirname(os.path.dirname(__file__))
    preproc_file = os.path.join(root, 'preprocessing', 'build_dataset.py')
    with open(preproc_file, 'r', encoding='utf-8') as f:
        txt = f.read()
    assert ".replace(-999, 0)" not in txt, "build_dataset.py must never replace -999 with 0"
    assert "TRAIN_MASK" in txt and "EVAL_MASK" in txt, "build_dataset.py must enforce TRAIN_MASK and EVAL_MASK"


