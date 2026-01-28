"""Utility helpers for modeling pipeline.

Keep small, auditable helpers here (IO, seeding, model persistence).
"""
from pathlib import Path
import joblib
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set deterministic seeds for numpy and random modules used in modeling."""
    np.random.seed(seed)
    random.seed(seed)

def save_model(obj, path: Path):
    """
    Persist a Python object to disk using joblib.

    Args:
        obj: Python object to serialize (e.g., sklearn Pipeline)
        path: Destination file path where the object will be saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_model(path: Path):
    """
    Load a Python object from disk using joblib.

    Args:
        path: Path to the joblib file

    Returns:
        Deserialized Python object
    """
    return joblib.load(path)