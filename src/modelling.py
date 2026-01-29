"""Model training utilities for Phase 3 PD modeling.

This module implements reproducible, simple, and interpretable logistic
regression training and tuning functions. It favors explainability over
complexity and includes class-weight handling and L1/L2 regularization.

When executed as a script (e.g. `python src/modeling.py`) ensure the
project root is on `sys.path` so absolute imports like `from src.utils`
work correctly.
"""
from pathlib import Path
import sys
# Add project root to sys.path when run directly so `from src import ...` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Tuple, Optional, Dict, Any, cast

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, brier_score_loss
from sklearn.calibration import calibration_curve

import argparse

# Prefer package imports, but allow running as a script where `src` may not be
# importable (e.g., no package __init__). Fall back to loading modules from
# the local `src/` files so `python src/modeling.py` still works.
try:
    from src.utils import set_seed, save_model
    from src.target_definition import ensure_default_target
    from src import evaluation as eval_utils
except Exception:
    import importlib.util
    _ROOT_SRC = Path(__file__).resolve().parents[0]

    def _load_local(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module {module_name} from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    _utils = _load_local("utils", _ROOT_SRC / "utils.py")
    set_seed = _utils.set_seed
    save_model = _utils.save_model

    _target = _load_local("target_definition", _ROOT_SRC / "target_definition.py")
    ensure_default_target = _target.ensure_default_target

    eval_utils = _load_local("evaluation", _ROOT_SRC / "evaluation.py")


def load_features(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, index_col=0)


def time_based_split(df: pd.DataFrame, time_col: str = "issue_d", train_end: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: all rows with time_col <= train_end go to train.

    If `train_end` is None, split by 80/20 time quantile by `time_col`.
    Does not shuffle data.
    """
    df = df.copy()
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in dataframe")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if train_end is None:
        cutoff = df[time_col].quantile(0.8)
    else:
        cutoff = pd.to_datetime(train_end)
    train = df[df[time_col] <= cutoff].copy()
    val = df[df[time_col] > cutoff].copy()
    return train, val


def recover_time_column(df: pd.DataFrame, clean_path: Path, time_col: str = "issue_d") -> pd.DataFrame:
    """Attempt to recover a datetime `time_col` from the cleaned CSV.

    Strategies (in order):
    1. If shapes equal, attach by positional alignment.
    2. If indexes match exactly, attach by index alignment.
    3. If a common identifier column exists between the two frames, merge on it.
    4. If 'Unnamed: 0' exists in the cleaned file and matches the feature index, align on that.

    The function returns the dataframe (mutated copy) with `time_col` added if found,
    otherwise returns the original dataframe unchanged.
    """
    clean_path = Path(clean_path)
    if not clean_path.exists():
        return df

    try:
        df_clean = pd.read_csv(clean_path)
    except Exception:
        return df

    if time_col not in df_clean.columns:
        return df

    df = df.copy()

    # 1) positional alignment when rows equal
    try:
        if df_clean.shape[0] == df.shape[0]:
            df[time_col] = pd.to_datetime(df_clean[time_col].values, errors="coerce")
            print(f"Added '{time_col}' from {clean_path} by positional alignment")
            return df
    except Exception:
        pass

    # 2) index equality
    try:
        if list(df_clean.index) == list(df.index):
            df[time_col] = pd.to_datetime(df_clean[time_col].values, errors="coerce")
            print(f"Added '{time_col}' from {clean_path} by index alignment")
            return df
    except Exception:
        pass

    # 3) merge on common identifier column
    try:
        left = df.reset_index()
        common = set(left.columns).intersection(set(df_clean.columns))
        # prefer 'Unnamed: 0' or obvious id-like columns
        preferred = ["Unnamed: 0", "id", "loan_id"]
        key = None
        for p in preferred:
            if p in common:
                key = p
                break
        if key is None and common:
            key = sorted(common)[0]

        if key is not None:
            merged = left.merge(df_clean[[key, time_col]], how="left", on=key)
            if time_col in merged.columns and merged[time_col].notna().any():
                # restore index
                merged = merged.set_index(left.columns[0])
                df[time_col] = pd.to_datetime(merged[time_col].values, errors="coerce")
                print(f"Added '{time_col}' from {clean_path} by merging on '{key}'")
                return df
    except Exception:
        pass

    return df


def train_baseline_logistic(X: pd.DataFrame, y: pd.Series, C: float = 1.0, penalty: Any = "l2", class_weight: Optional[Dict] = None, solver: Any = "saga", seed: int = 42) -> Pipeline:
    """Train a simple logistic regression inside a pipeline with scaler.

    Returns fitted sklearn Pipeline.
    """
    set_seed(seed)
    # default to balanced class weights when caller does not provide one
    cw = class_weight if class_weight is not None else "balanced"
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty=penalty, class_weight=cw, solver=solver, max_iter=1000, random_state=seed))
    ])
    pipe.fit(X, y)
    return pipe


def tune_logistic(X: pd.DataFrame, y: pd.Series, param_grid: Optional[dict] = None, cv: int = 5, seed: int = 42) -> GridSearchCV:
    """Perform grid search over logistic hyperparameters.

    Returns the fitted GridSearchCV object with best_estimator_ present.
    """
    set_seed(seed)
    if param_grid is None:
        param_grid = {
            "clf__penalty": ["l2"],
            "clf__C": [0.01, 0.1, 1.0, 10.0]
        }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", solver="saga", max_iter=2000, random_state=seed))
    ])
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=1)
    grid.fit(X, y)
    return grid


def stratified_subsample(X: pd.DataFrame, y: pd.Series, frac: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Return a stratified subsample of (X, y) preserving class proportions.

    frac: fraction of rows to sample (e.g., 0.1-0.2)
    """
    if not 0 < frac <= 1.0:
        raise ValueError("frac must be in (0, 1]")
    # If dataset is small, return the original
    n = X.shape[0]
    subsample_n = int(max(2, round(n * frac)))
    if subsample_n >= n:
        return X.copy(), y.copy()

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=subsample_n,
        stratify=y,
        random_state=seed,
    )
    return X_sub, y_sub


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute AUC, KS, precision, recall (0.5 threshold), Brier, and calibration table."""
    out: Dict[str, Any] = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_scores))
    except Exception:
        out["auc"] = None

    try:
        out["ks"] = float(eval_utils.ks_statistic(y_true, y_scores))
    except Exception:
        out["ks"] = None

    try:
        y_pred = (y_scores >= 0.5).astype(int)
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        out["precision"] = out["recall"] = None

    try:
        out["brier"] = float(brier_score_loss(y_true, y_scores))
    except Exception:
        out["brier"] = None

    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_scores, n_bins=n_bins)
        out["calibration"] = {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()}
    except Exception:
        out["calibration"] = None

    return out


def save_pipeline(model, path: Path):
    save_model(model, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train baseline and tuned logistic PD models')
    parser.add_argument('--features', type=str, default='data/processed/accepted_fe.csv', help='Path to engineered features CSV')
    parser.add_argument('--time-col', type=str, default='issue_d', help='Datetime column used for time-based split')
    parser.add_argument('--train-end', type=str, default=None, help='Cutoff date (inclusive) for training set, e.g. 2016-12-31')
    parser.add_argument('--baseline-out', type=str, default='models/baseline_logistic.pkl', help='Path to save baseline model')
    parser.add_argument('--tuned-out', type=str, default='models/tuned_logistic.pkl', help='Path to save tuned model')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    set_seed(args.seed)

    feats_path = Path(args.features)
    if not feats_path.exists():
        raise SystemExit(f'Feature file not found: {feats_path}')

    print('Loading features from', feats_path)
    df = load_features(feats_path)

    # Ensure we have binary `default` target. If features file doesn't include
    # the target (common when features and labels were saved separately), try
    # to locate a labels file and merge it in.
    if 'default' not in df.columns:
        # common labels filenames to try
        candidates = [
            feats_path.parent / 'accepted_labels.csv',
            Path('data') / 'processed' / 'accepted_labels.csv',
        ]
        labels = None
        for p in candidates:
            if p.exists():
                try:
                    labels = pd.read_csv(p, index_col=0)
                    print('Loaded labels from', p)
                    break
                except Exception:
                    continue

        if labels is not None:
            # Find a sensible column in labels (prefer 'target')
            if 'target' in labels.columns:
                label_col = 'target'
            else:
                label_col = labels.columns[0]
            # join labels to features by index
            df = df.join(labels[label_col])
            # normalize column name to `default` if it's not already
            if label_col != 'default':
                df = df.rename(columns={label_col: 'default'})
            print('Merged labels into features; using column `default`.')
        else:
            # As a last resort, if `loan_status` is present try to construct target
            if 'loan_status' in df.columns:
                df = ensure_default_target(df, target_col='default')
            else:
                raise SystemExit(
                    "No 'default' column found in features and no labels file located. "
                    "Run Phase 2 to create 'data/processed/accepted_fe.csv' or provide labels at 'accepted_labels.csv'."
                )

        # If the requested time column is missing from the features, attempt to
        # retrieve it from the original cleaned dataset `accepted_clean.csv` and
        # align it to the features. This handles cases where features were saved
        # without metadata columns such as issue dates.
        if args.time_col not in df.columns:
            clean_path = Path('data') / 'processed' / 'accepted_clean.csv'
            df = recover_time_column(df, clean_path, time_col=args.time_col)
            if args.time_col not in df.columns or df[args.time_col].isna().all():
                print(f"Could not recover '{args.time_col}' from {clean_path}; it remains missing or all-NA")

    # Time-based split
    print('Creating time-based train/validation split using', args.time_col)
    train, val = time_based_split(df, time_col=args.time_col, train_end=args.train_end)
    print('Train rows:', train.shape[0], 'Validation rows:', val.shape[0])

    # Prepare X/y
    y_train = train['default']
    y_val = val['default']
    X_train = train.drop(columns=['default', args.time_col], errors='ignore')
    X_val = val.drop(columns=['default', args.time_col], errors='ignore')

    # Train baseline
    print('Training baseline logistic (L2, class-balanced)')
    baseline = train_baseline_logistic(X_train, y_train, C=1.0, penalty='l2', seed=args.seed)
    save_model(baseline, Path(args.baseline_out))
    print('Saved baseline ->', args.baseline_out)

    # Evaluate baseline
    try:
        if hasattr(baseline, 'predict_proba'):
            val_scores = cast(Any, baseline).predict_proba(X_val)[:, 1]
            metrics_val = eval_utils.basic_metrics(np.asarray(y_val), np.asarray(val_scores))
            print('Baseline validation metrics:', metrics_val)
        else:
            print('Baseline model has no predict_proba; skipping validation scoring')
    except Exception:
        print('Warning: could not score validation set for baseline (check features).')

    # Lightweight tuning on a stratified subsample to reduce compute burden
    print('Creating stratified subsample for tuning (10-20% of training set)')
    subsample_frac = 0.15
    X_sub, y_sub = stratified_subsample(X_train, y_train, frac=subsample_frac, seed=args.seed)
    print(f'Subsample size: {X_sub.shape[0]} (original train {X_train.shape[0]})')

    print('Tuning logistic (grid search) on subsample')
    param_grid = {"clf__penalty": ["l2"], "clf__C": [0.1, 1.0, 10.0]}
    grid = tune_logistic(X_sub, y_sub, param_grid=param_grid, cv=min(3, 3), seed=args.seed)
    best_C = grid.best_params_.get('clf__C', 1.0)
    print('Best C from subsample tuning:', best_C)

    # Refit on full training set using best C
    print('Refitting logistic on full training set with best C')
    tuned = train_baseline_logistic(X_train, y_train, C=best_C, penalty='l2', seed=args.seed)
    save_model(tuned, Path(args.tuned_out))
    print('Saved tuned ->', args.tuned_out)

    # Evaluate both models on held-out validation set
    print('Evaluating baseline and tuned models on validation set')
    results: Dict[str, Any] = {}

    try:
        if hasattr(baseline, 'predict_proba'):
            val_scores = cast(Any, baseline).predict_proba(X_val)[:, 1]
            results['baseline'] = compute_metrics(np.asarray(y_val), np.asarray(val_scores))
            print('Baseline validation metrics:', results['baseline'])
        else:
            print('Baseline model has no predict_proba; skipping baseline scoring')
    except Exception:
        print('Warning: could not score validation set for baseline (check features).')

    try:
        if hasattr(tuned, 'predict_proba'):
            val_scores_t = cast(Any, tuned).predict_proba(X_val)[:, 1]
            results['tuned'] = compute_metrics(np.asarray(y_val), np.asarray(val_scores_t))
            print('Tuned validation metrics:', results['tuned'])
        else:
            print('Tuned model has no predict_proba; skipping tuned scoring')
    except Exception:
        print('Warning: could not score validation set for tuned model.')

    # Model selection decision
    baseline_auc = results.get('baseline', {}).get('auc')
    tuned_auc = results.get('tuned', {}).get('auc')
    selection_note = ''
    if baseline_auc is None or tuned_auc is None:
        selection_note = 'Insufficient metrics to perform model selection.'
    else:
        # If tuned AUC improvement is < 0.005 prefer baseline for stability
        delta = tuned_auc - baseline_auc
        if delta < 0.005:
            selection_note = (f'Tuned model AUC improvement is marginal (delta={delta:.4f}). '
                              'Prefer baseline for stability and interpretability.')
            selected = 'baseline'
        else:
            selection_note = (f'Tuned model improves AUC by {delta:.4f}; select tuned model.')
            selected = 'tuned'

    # Write report
    report_path = Path('reports') / 'model_performance.md'
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as fh:
            fh.write('# Model Performance Report\n\n')
            fh.write('## Summary\n')
            fh.write(f"- Training rows: {X_train.shape[0]}\n")
            fh.write(f"- Validation rows: {X_val.shape[0]}\n")
            fh.write('\n')
            fh.write('## Subsampling and Tuning Rationale\n')
            fh.write('- Subsampling: stratified subsample used to reduce compute burden during hyperparameter search.\n')
            fh.write('- Fraction: 15% of training set used (configurable).\n')
            fh.write('- Tuning: limited grid on `C` (0.1, 1.0, 10.0), L2 penalty, class_weight="balanced", 3-fold CV.\n')
            fh.write('- Reason: dataset is large; full CV across millions of rows is computationally prohibitive. Subsampling preserves target rate and yields stable hyperparameter estimates while saving compute.\n\n')

            fh.write('## Tuning Results\n')
            fh.write(f'- Best C (from subsample): {best_C}\n')
            fh.write('\n')

            fh.write('## Validation Metrics\n')
            for name in ['baseline', 'tuned']:
                m = results.get(name)
                fh.write(f'### {name.capitalize()}\n')
                if m is None:
                    fh.write('- No metrics available.\n')
                    continue
                fh.write(f"- AUC: {m.get('auc')}\n")
                fh.write(f"- KS: {m.get('ks')}\n")
                fh.write(f"- Precision (0.5): {m.get('precision')}\n")
                fh.write(f"- Recall (0.5): {m.get('recall')}\n")
                fh.write(f"- Brier score: {m.get('brier')}\n")
                fh.write('\n')

            fh.write('## Model Selection Decision\n')
            fh.write(f'- {selection_note}\n')
            fh.write('\n')
            fh.write('## Reproducibility\n')
            fh.write('- Random seed used: {0}\n'.format(args.seed))
            fh.write('- Subsample fraction: {0}\n'.format(subsample_frac))
            fh.write('- Parameter grid: C in {0}\n'.format([0.1, 1.0, 10.0]))
    except Exception as e:
        print('Warning: could not write report:', e)

    print('Done.')


if __name__ == '__main__':
    main()
