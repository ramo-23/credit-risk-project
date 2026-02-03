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
    fro, src.utils import load_model, save_model
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