"""Target definition utilities for credit_risk_project.

Provide a clear, reproducible mapping for the binary `default` target used
in Phase 3 modeling. Functions here should be simple, documented, and easy
to review by model validation.
"""
from pathlib import Path
import pandas as pd


def ensure_default_target(df: pd.DataFrame, target_col: str = "default") -> pd.DataFrame:
    """Ensure a binary `default` column exists in the DataFrame.

    If `target_col` already exists and is numeric/binary, it is returned as-is.
    If not present, attempt to infer from common loan status columns (e.g. `loan_status`).

    Args:
        df: Input DataFrame
        target_col: Desired target column name

    Returns:
        DataFrame with guaranteed `target_col` present (0/1 integers)
    """
    df = df.copy()
    if target_col in df.columns:
        # coerce to integer binary if possible
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
        return df
    
    # Infer from common alternative columns
    if "loan_status" in df.columns:
        mapping = {k: (1 if "charg" in str(k).lower() or "default" in str(k).lower() else 0) for k in df["loan_status"].unique()}
        df[target_col] = df["loan_status"].map(mapping).fillna(0).astype(int)
        return df

    raise KeyError("Could not find or construct a binary target column. Provide 'default' or 'loan_status'.")
