"""Scoring utilities: load model, score validation set, save scored CSV."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd
import numpy as np

# Ensure project root is on sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils import load_model
    from src.modelling import load_features, time_based_split
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
    load_model = _utils.load_model

    _modeling = _load_local("modeling", _ROOT_SRC / "modelling.py")
    load_features = _modeling.load_features
    time_based_split = _modeling.time_based_split


def parse_args():
    """Parse CLI arguments for the scoring script."""
    p = argparse.ArgumentParser(description="Score validation set with a trained PD model")
    p.add_argument("--model", type=Path, default=Path("models/baseline_logistic.pkl"))
    p.add_argument("--features", type=Path, default=Path("data/processed/accepted_fe.csv"))
    p.add_argument("--time-col", type=str, default="issue_d")
    p.add_argument("--out", type=Path, default=Path("data/processed/scored_validation.csv"))
    return p.parse_args()


def main(argv=None):
    """Run scoring: load features, score validation partition, and persist scored CSV.

    Args:
        argv: Optional list of CLI args (for testing); if None, parsed from sys.argv.
    """
    args = parse_args() if argv is None else parse_args()

    if not args.model.exists():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        sys.exit(2)

    if not args.features.exists():
        print(f"Features file not found: {args.features}", file=sys.stderr)
        sys.exit(2)

    print("Loading features from", args.features)
    df = load_features(args.features)

    print("Creating time-based split using", args.time_col)
    train, val = time_based_split(df, time_col=args.time_col)
    print("Validation rows:", val.shape[0])

    print("Loading model from", args.model)
    model = load_model(args.model)

    if not hasattr(model, "predict_proba"):
        print("Model has no predict_proba method", file=sys.stderr)
        sys.exit(3)

    X_val = val.drop(columns=["default", args.time_col], errors="ignore")

    print("Scoring validation set")
    try:
        probs = model.predict_proba(X_val)[:, 1]
    except Exception as e:
        print("Error during predict_proba:", e, file=sys.stderr)
        sys.exit(4)

    val_out = val.copy()
    val_out["pd_hat"] = probs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    val_out.to_csv(args.out)
    print("Saved scored validation to", args.out)


if __name__ == "__main__":
    main()
