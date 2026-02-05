"""Calibration diagnostics and optional recalibration for PD model predictions."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# Ensure project root is in sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.evaluation import plot_calibration
    from src.utils import load_model, save_model
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
    
    _eval = _load_local("evaluation", _ROOT_SRC / "evaluation.py")
    plot_calibration = _eval.plot_calibration
    _utils = _load_local("utils", _ROOT_SRC / "utils.py")
    load_model = _utils.load_model
    save_model = _utils.save_model


def parse_args():
    p = argparse.ArgumentParser(description="Calibration diagnostics for scored validation set")
    p.add_argument("--scored", type=Path, default=Path("data/processed/scored_validation.csv"))
    p.add_argument("--model", type=Path, default=Path("models/baseline_logistic.pkl"))
    p.add_argument("--apply", action="store_true", help="If set, fit Platt (sigmoid) calibration on a holdout of the validation set and save recalibrated probs")
    p.add_argument("--out", type=Path, default=Path("data/processed/scored_validation_recalibrated.csv"))
    p.add_argument("--cal-plot", type=Path, default=Path("reports/calibration_plot.png"))
    return p.parse_args()


def main(argv=None):
    """Run calibration diagnostics and optionally fit Platt scaling.

    Args:
        argv: Optional list of CLI args (for testing); if None, parsed from sys.argv.
    """
    args = parse_args() if argv is None else parse_args()

    if not args.scored.exists():
        print(f"Scored validation set not found: {args.scored}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.scored, index_col=0)
    if "pd_hat" not in df.columns or "default" not in df.columns:
        print("Scored file must contain 'pd_hat and 'default' columns", file=sys.stderr)
        sys.exit(3)
    
    y = df["default"].values
    p = df["pd_hat"].values

    # Brier score
    from sklearn.metrics import brier_score_loss

    brier = float(brier_score_loss(y, p))
    print(f"Brier score: {brier:.6f}")

    # Calibration plot
    plot_calibration(y, p, n_bins=10, out_path=args.cal_plot)
    print(f"Calibration plot saved to {args.cal_plot}")

    # Optional recalibration with Platt scaling
    if args.apply:
        # We will re-fit calibration using sigmoid (Platt) scaling on a holdout of the validation set
        from sklearn.model_selection import train_test_split

        X_dummy = p.reshape(-1, 1)  # CalibratedClassifierCV expects 2D input
        X_cal, X_rest, y_cal, y_rest = train_test_split(X_dummy, y, test_size=0.8, random_state=42, stratify=y)
        
        # Use CalibratedClassifierCV with base estimator as a lightweight LogisticRegression
        base = LogisticRegression(solver="lbfgs")
        # Train base on calibration input as features and labels
        try:
            base.fit(X_cal, y_cal)
            calibrator = CalibratedClassifierCV(base_estimator=base, cv="prefit", method="sigmoid")
            calibrator.fit(X_cal, y_cal)
            p_recal = calibrator.predict_proba(X_dummy)[:,1]
            df2 = df.copy()
            df2["pd_hat_recal"] = p_recal
            args.out.parent.mkdir(parents=True, exist_ok=True)
            df2.to_csv(args.out)
            print(f"Saved recalibrates score file to {args.out}")
        except Exception as e:
            print("Calibration failed:", e, file=sys.stderr)
            sys.exit(4)

if __name__ == "__main__":
    main()
