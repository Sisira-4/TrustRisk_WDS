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
