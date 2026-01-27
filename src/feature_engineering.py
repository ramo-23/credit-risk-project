""" Development of feature engineering functions and classes.

This module contains functions and classes for performing feature engineering tasks
such as encoding categorical variables, scaling numerical features, and creating new features
based on existing data. These tools help prepare datasets for machine learning models
by transforming raw data into a more suitable format.

The goals of this module include:
- Reproducible, stateless transformations
- Simple, well-documented helpers that are easy to test
- Conservative handling of missing data to avoid data leakage
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from typing import Optional, Union, List
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import joblib


SEED = 42


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV data into a DataFrame.

    Accepts either a string path or a pathlib.Path and returns the loaded
    DataFrame.
    """
    path = Path(path)
    df = pd.read_csv(path)
    return df


def map_target(df:pd.DataFrame, target_col: str = 'loan_status') -> pd.Series:
    """Map raw `loan_status` strings to a binary target.

    The mapping is intentionally conservative: statuses that contain
    substrings like 'charg' or 'default' (case-insensitive) are treated as
    positives (1). All other statuses map to 0. This keeps the mapping simple
    and robust to small variations in textual labels.

    Args:
        df: DataFrame containing the target column.
        target_col: Column name containing loan status labels.

    Returns:
        A pandas Series of 0/1 encoded targets aligned with `df` index.
    """
    # Conservative mapping: treat charged/default-like statuses as positive (1)
    mapping = {
        k: (
            1
            if (isinstance(k,str) and ("charg" in k.lower() or "default" in k.lower()))
            else 0
        )
        for k in df[target_col].unique()
    }
    return df[target_col].map(mapping)


def parse_emp_length(x):
    """Parse employment length text into a numeric year value.

    Handles common formats such as:
      - '10+ years' -> 10
      - '< 1 year'  -> 0
      - 'n/a' or empty -> NaN

    Args:
        x: raw employment length value (string-like)

    Returns:
        int  or NaN representing years of employment
    """
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in ["n/a", "nan", "None", ""]:
        return np.nan
    # Handle '10+ years' case
    if "+" in x:
        return int(x.replace("+ years", "").replace("+ year", "").strip())
    # Handle '< 1 year' case
    if "<" in x:
        return 0
    try:
        return int(x.split()[0])
    except Exception:
        return np.nan
    

def credit_history_years(df: pd.DataFrame, issue_col: str = "issue_d", earliest_col: str = "earliest_cr_line"):
    """Compute approximate credit history length in years.

    Calculates the difference between `issue_col` and `earliest_col` as
    a fractional number of years. Invalid or unparsable dates are coerced to
    NaT and produce NaN in the returned Series.

    Args:
        df: DataFrame containing date columns.
        issue_col: Column name for loan issue date.
        earliest_col: Column name for earliest credit line date.

    Returns:
        pandas Series with credit history length in years (float)
    """
    # Parse dates flexibly
    issue = pd.to_datetime(df[issue_col], errors='coerce')
    earliest = pd.to_datetime(df[earliest_col], errors='coerce')
    years = (issue - earliest).dt.days / 365.25
    return years


def kfold_target_encode(train_series, target, n_splits=5, seed=SEED):
    """Perform K-fold target encoding using out-of-fold means.

    This function computes target means for categories using only training
    folds and applies them to validation folds. It avoids leaking target
    information by not using the validation fold when computing its own
    encodings. Missing or unseen categories are filled with the global mean.

    Args:
        train_series: Series of categorical values to encode (aligned with target)
        target: Series of binary or numeric target values
        n_splits: Number of K-Folds to use for out-of-fold encoding
        seed: Random seed for fold shuffling

    Returns:
        Series of float encodings aligned with `train_series` index
    """
    # Returns an encoded series aligned with train_series index using out-of-fold means
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = pd.Series(index=train_series.index, dtype=float)
    for tr_idx, val_idx in folds.split(train_series):
        tr, val = train_series.iloc[tr_idx], train_series.iloc[val_idx]
        tr_t = target.iloc[tr_idx]
        means = tr_t.groupby(tr).mean()
        result.iloc[val_idx] = val.map(means)
    # global mean for any unseen or NaN
    global_mean = target.mean()
    result = result.fillna(global_mean)
    return result


def frequency_encode(series: pd.Series) -> pd.Series:
    """Encode categories by their frequency counts.

    Frequency encoding is a simple deterministic alternative to target
    encoding for high-cardinality features. Nulls are treated as a category.

    Args:
        series: Categorical Series to encode

    Returns:
        Float Series corresponding to category frequencies
    """
    freq = series.value_counts(dropna=False)
    return series.map(freq).astype(float)


def build_features(df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
    """Construct modeling features from a cleaned DataFrame.

    This function performs a sequence of common feature engineering steps:
      - Parse and coerce numeric-like text fields (e.g., interest rates)
      - Construct ratio features such as LTI (loan-to-income)
      - Bin employment length and interest rates for low-cardinality treatment
      - Create interaction features and frequency encodings for high-cardinality
        categorical variables
      - Optionally perform target-based encoding for a column if `target` is
        supplied (uses K-fold to avoid leakage)

    The function is defensive and will create placeholder NaNs for missing
    input columns rather than failing, which makes it safe to run on
    different dataset snapshots.

    Args:
        df: Cleaned DataFrame (preferably output of `clean_accepted`)
        target: Optional target Series aligned with `df` used for K-fold
                target encoding

    Returns:
        DataFrame with additional engineered features (original columns preserved)
    """
    df = df.copy()

    # Numeric parsing / cleaning
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%","", regex=False)
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors='coerce')
    
    # Debt-to-Income ratio: ensure that the column exists
    if "dti" not in df.columns and "annual_inc" in df.columns:
        df["dti"] = np.nan  # Placeholder if not present
    df["DTI"] = df["dti"] if "dti" in df.columns else np.nan

    # Loan-to-Income ratio
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["LTI"] = df["loan_amnt"] / df["annual_inc"].replace({0: np.nan})
    else:
        df["LTI"] = np.nan

    # Employment length parsing
    if "emp_length" in df.columns:
        df["emp_length_num"] = df["emp_length"].apply(parse_emp_length)
        bins = [-0.1, 2, 5, 9, 100]
        labels = ["0-2", "3-5", "6-9", "10+"]
        df["emp_length_bin"] = pd.cut(df["emp_length_num"].fillna(-1), bins=bins, labels=labels, include_lowest=True)
        df["emp_length_bin"] = df["emp_length_bin"].cat.add_categories("missing").fillna("missing")

     # Credit history length (years)
    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        df["credit_history_years"] = credit_history_years(df, issue_col="issue_d", earliest_col="earliest_cr_line")
    else:
        df["credit_history_years"] = np.nan

    # Interest rate buckets for low-cardinality one-hot encoding
    if "int_rate" in df.columns:
        bins = [0, 8, 12, 16, 20, 100]
        labels = ["<8", "8-12", "12-16", "16-20", ">=20"]
        df["int_rate_bin"] = pd.cut(df["int_rate"].fillna(-1), bins=bins, labels=labels, include_lowest=True)
        df["int_rate_bin"] = df["int_rate_bin"].cat.add_categories("missing").fillna("missing")

    # Loan term numeric and a simple boolean flag for long terms
    if "term" in df.columns:
        df["term_months"] = df["term"].astype(str).str.extract(r'(\d+)').astype(float)
        df["term_long"] = (df["term_months"] > 36).astype(float)
    else:
        df["term_months"] = np.nan
        df["term_long"] = np.nan
    
    # Interaction features: loan amount * interest rate (use median for missing int_rate)
    if "loan_amnt" in df.columns and "int_rate" in df.columns:
        df["loan_x_int"] = df["loan_amnt"] * df["int_rate"].fillna(df["int_rate"].median())
    else:
        df["loan_x_int"] = np.nan
    
    # Frequency encoding for high-cardinality categorical features
    high_card_candidates = [c for c in ["emp_title", "emp_length", "addr_state"] if c in df.columns]
    for c in high_card_candidates:
        df[f"{c}_freq_enc"] = frequency_encode(df[c].fillna("missing"))
    
    # One-hot encoding for low-cardinality categorical features
    low_card = [c for c in ["home_ownership", "purpose", "int_rate_bucket", "emp_length_bin"] if c in df.columns]
    df = pd.get_dummies(df, columns=low_card, dummy_na=True, drop_first=True)

    # Create a simple K-fold target encoding for 'emp_title' if target is provided
    if target is not None and "emp_title" in df.columns:
        df["emp_title_target_te"] = kfold_target_encode(df["emp_title"].fillna("missing"), target)
    
    # Drop identifiers and post-issue leakage columns if present
    drop_cols = [c for c in ["id", "member_id", "url", "desc", "title", "last_pymnt_d", "next_pymnt_d", "collection_recovery_fee"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def handle_missing_and_outliers(df: pd.DataFrame, clip_pct=(0.01, 0.99)) -> pd.DataFrame:
    """Impute missing values and reduce extreme outliers.

    Steps performed:
      - Add binary missing indicators for columns with >5% missingness
      - Impute numeric columns with their median
      - Clip numeric columns to the specified percentile range (default 1st-99th)
      - Add log1p-transformed versions of common skewed financial fields

    Args:
        df: DataFrame with features to clean
        clip_pct: Tuple of lower and upper quantile bounds for clipping

    Returns:
        DataFrame with imputed and clipped numeric features and new transformed columns
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Handle missing indicators
    for c in df.columns:
        miss_rate = df[c].isna().mean()
        if miss_rate > 0.05:
            df[f"{c}_missing"] = df[c].isna().astype(int)
    
    # Numeric impute median 
    for c in num_cols:
        median = df[c].median()
        df[c] = df[c].fillna(median)
    
    # Clip extreme values
    lower_q, upper_q = clip_pct
    for c in num_cols:
        low = df[c].quantile(lower_q)
        high = df[c].quantile(upper_q)
        df[c] = df[c].clip(lower=low, upper=high)

    # Transform skewed financial fields
    for c in ["annual_inc", "loan_amnt", "revol_bal", "installment"]:
        if c in df.columns:
            df[f"{c}_log1p"] = np.log1p(df[c].clip(lower=0))

    return df
            

def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: list):
    """Fit a StandardScaler on `train_df` and apply to both train and test.

    Returning the fitted scaler enables later use during inference.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[numeric_cols] = scaler.transform(train_df[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_scaled, test_scaled, scaler


def run_feature_engineering(
    input_path: Union[str, Path] = "data/processed/accepted_clean.csv",
    output_features: Union[str, Path] = "data/processed/accepted_features.csv",
    output_labels: Union[str, Path] = "data/processed/accepted_labels.csv",
    artifcats_dir: Union[str, Path] = "artifacts",
    output_merged: Union[str, Path] = "data/processed/accepted_fe.csv",
    test_size: float = 0.2,
    seed: int = SEED
):
    input_path = Path(input_path)
    out_features = Path(output_features)
    out_labels = Path(output_labels)
    artifacts_dir = Path(artifcats_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")
    df = load_data(input_path)

    print("Mapping target variable...")
    y = map_target(df, target_col='loan_status')

    print("Building features...")
    df_features = build_features(df, target=y)

    print("Handling missing values and outliers...")
    df_features = handle_missing_and_outliers(df_features)

    # Keep only numeric columns for modeling
    X = df_features.select_dtypes(include=[np.number])

    # Train-test split
    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Scaling {len(numeric_cols)} numeric features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numeric_cols)

    # Save full transfored dataset
    X_full = pd.concat([X_train_scaled, X_test_scaled]).sort_index()
    y_full = pd.concat([y_train, y_test]).sort_index()

    out_features.parent.mkdir(parents=True, exist_ok=True)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    X_full.to_csv(out_features, index=True)
    y_full.to_csv(out_labels, index=True, header=['target'])

    # Additionally save a merged features+target file for downstream use
    try:
        merged_path = Path(output_merged)
        merged_df = X_full.copy()
        # ensure the target column in named default for consistency (Third Step)
        merged_df["default"] = y_full
        # Reattach 'issue_d' if present in original df for temporal splits
        # downstream scripts that expect the origination date can use it
        try:
            if "issue_d" in df.columns:
                merged_df["issue_d"] = pd.to_datetime(df.reindex(X_full.index)["issue_d"], errors="coerce").values
        except Exception:
            # non-fatal: if reattachment fails, continue without issue_d
            pass
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(merged_path, index=True)
        print(f"Saved merged features+target -> {merged_path}")
    except Exception:
        print("Warning: failed to save merged features+target file")

    # Save artifacts: scaler and numeric columns
    artifcats = {"scaler": scaler, "numeric_cols": numeric_cols}
    joblib.dump(artifcats, artifacts_dir / "feature_artifacts.joblib")

    # Also save train/test splits for reproducibility
    X_train_scaled.to_csv(artifacts_dir / "X_train.csv", index=True)
    X_test_scaled.to_csv(artifacts_dir / "X_test.csv", index=True)
    y_train.to_csv(artifacts_dir / "y_train.csv", index=True, header=['target'])
    y_test.to_csv(artifacts_dir / "y_test.csv", index=True, header=['target'])

    print(f"Saved features -> {out_features}")
    print(f"Saved labels -> {out_labels}")
    print(f"Saved artifacts -> {artifacts_dir / 'feature_artifacts.joblib'}")

    return {
        "X_path": str(out_features),
        "y_path": str(out_labels),
        "artifacts": str(artifacts_dir / "feature_artifacts.joblib"),
    }


def _cli():
    parser = argparse.ArgumentParser(description="Feature engineering for credit risk dataset")
    parser.add_argument("--input", default="data/processed/accepted_clean.csv")
    parser.add_argument("--out-features", default="data/processed/accepted_features.csv")
    parser.add_argument("--out-labels", default="data/processed/accepted_labels.csv")
    parser.add_argument("--out-merged", default="data/processed/accepted_fe.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_feature_engineering(args.input, args.out_features, args.out_labels, args.artifacts, args.out_merged, args.test_size, args.seed)


if __name__ == "__main__":
    _cli()


# -------------------------
# Phase 2 helpers & runner
# -------------------------


def define_default_target(df: pd.DataFrame, source_col: str = "loan_status", target_col: str = "default") -> pd.DataFrame:
    """Define a conservative binary default target and exclude ambiguous cases.

    Mapping logic:
      - 1: statuses containing 'charg' or 'default' (charged off / default)
      - 0: statuses containing 'fully paid' or 'paid'
      - ambiguous statuses (e.g., 'Current', 'Late') are set to NaN and removed

    Returns the filtered DataFrame with `target_col` present as int.
    """
    df = df.copy()
    s = df[source_col].astype(str).str.lower()
    pos_mask = s.str.contains("charg") | s.str.contains("default")
    neg_mask = s.str.contains("fully paid") | s.str.contains("fullypaid") | s.str.contains("paid")

    df[target_col] = np.nan
    df.loc[pos_mask, target_col] = 1
    df.loc[neg_mask, target_col] = 0

    before = df.shape[0]
    df = df[~df[target_col].isna()].copy()
    after = df.shape[0]
    dropped = before - after
    print(f"Defined target '{target_col}'. Kept {after} rows; dropped {dropped} ambiguous rows.")
    df[target_col] = df[target_col].astype(int)
    return df


def remove_leakage_columns(df: pd.DataFrame, report_path: Union[str, Path] = "reports/leakage_checks.md") -> pd.DataFrame:
    """Drop common post-origination or leakage-prone columns and document them.

    The function drops columns if present and writes a short report listing
    which columns were removed for reproducibility and model validation.
    """
    df = df.copy()
    leak_cols = [
        # Payment history / post-origination
        "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
        # Recovery / collection
        "collection_recovery_fee", "recoveries",
        # Aggregates and post-payment totals
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
        # Principal remaining
        "out_prncp", "out_prncp_inv",
        # Any other fields that reflect post-origination performance
        "last_pymnt_d", "next_pymnt_d"
    ]
    removed = [c for c in leak_cols if c in df.columns]
    if removed:
        df = df.drop(columns=removed)

    # Write simple leakage report
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Leakage Checks", "", "Dropped columns:", ""]
    if removed:
        lines += [f"- {c}" for c in removed]
    else:
        lines.append("- (none found)")
    rp.write_text("\n".join(lines))
    print(f"Leakage report written to {rp} (dropped {len(removed)} columns)")
    return df


def summarize_features(df: pd.DataFrame, target_col: str = "default", out_path: Union[str, Path] = "reports/feature_summary.md") -> None:
    """Write a short feature summary: rows, class balance, feature counts, top missingness."""
    rp = Path(out_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = df.shape
    if target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
    else:
        counts = {}
    numeric_count = df.select_dtypes(include=[np.number]).shape[1]
    missing = df.isna().mean().sort_values(ascending=False).head(10)

    lines = [
        "# Feature Summary",
        "",
        f"Rows: {rows}",
        f"Columns: {cols}",
        f"Numeric features: {numeric_count}",
        "",
        "## Target distribution",
        "",
    ]
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Top missingness", ""]
    for idx, val in missing.items():
        lines.append(f"- {idx}: {val:.3f}")

    rp.write_text("\n".join(lines))
    print(f"Feature summary written to {rp}")


def run_phase2(input_path: Union[str, Path] = "data/processed/accepted_clean.csv", output_path: Union[str, Path] = "data/processed/accepted_fe.csv", reports_dir: Union[str, Path] = "reports") -> dict:
    """Run Phase 2: target definition, leakage removal, feature engineering, save final dataset.

    Returns dict with paths to saved files for reproducibility.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned data from {input_path}")
    df = load_data(input_path)

    # 1. Define conservative binary target and drop ambiguous records
    df = define_default_target(df, source_col="loan_status", target_col="default")

    # 2. Remove leakage-prone columns and write report
    df = remove_leakage_columns(df, report_path=reports_dir / "leakage_checks.md")

    # 3. Build features (uses earlier helpers)
    print("Building features...")
    y = df["default"]
    df_feat = build_features(df, target=y)

    # 4. Handle missing / outliers
    df_feat = handle_missing_and_outliers(df_feat)

    # 5. Keep numeric and one-hot encoded columns only (model-ready)
    X = df_feat.select_dtypes(include=[np.number])

    # 6. Save final feature dataset with target
    # Reattach target aligned to X's index
    final = X.copy()
    final["default"] = y.loc[final.index]
    # If the input `df` contains `issue_d`, reattach it so Phase 2 outputs
    # preserve origination metadata for downstream monitoring or scoring.
    try:
        if "issue_d" in df.columns:
            final["issue_d"] = pd.to_datetime(df.reindex(final.index)["issue_d"], errors="coerce").values
    except Exception:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=True)
    print(f"Saved model-ready features to {output_path}")

    # 7. Summarize features and write file
    summarize_features(final, target_col="default", out_path=reports_dir / "feature_summary.md")

    return {"features": str(output_path), "feature_summary": str(reports_dir / "feature_summary.md"), "leakage_report": str(reports_dir / "leakage_checks.md")}


def _cli_phase2():
    parser = argparse.ArgumentParser(description="Phase 2 feature engineering + target definition")
    parser.add_argument("--input", default="data/processed/accepted_clean.csv")
    parser.add_argument("--out", default="data/processed/accepted_fe.csv")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    run_phase2(args.input, args.out, args.reports)

