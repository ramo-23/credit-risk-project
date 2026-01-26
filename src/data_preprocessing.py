"""
Preprocessing module for data cleaning and transformation.

This module provides reproducible data cleaning functions for the Lending Club loan dataset.
All transformations are deterministic and thus documented to ensure consistent preprocessing across different runs.
across different execution environments.

Its primary functions are:
    clean_accepted: Orchestrates the complete data cleaning process for accepted loans.
    It also loads the raw data from a CSV file, applies a series of cleaning steps, and returns the cleaned DataFrame.

The design principles followed in this module include:
    - Reproducibility: All cleaning steps are deterministic and documented.
    - Modularity: Each cleaning step is encapsulated in its own function for clarity and reusability.
    - Transparency: Each function includes detailed docstrings explaining its purpose, parameters, and return values.  
"""

from pathlib import Path
import pandas as pd
import numpy as np


# Module-level constants
DEFAULT_DROP_THRESHOLD = 0.6 # Threshold for dropping columns with missing values

def _snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all column names in the DataFrame to snake_case.

    What was done:
        - Replaced spaces with underscores.
        - Converted all characters to lowercase.
        - Converted '%' to '_pct'.
        - Removed special characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame with potentially inconsistent column names.
    
    Returns:
        pd.DataFrame: DataFrame with standardized snake_case column names.
    """

    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('%', '_pct', regex=False)
        .str.replace('[^0-9a-zA-Z_]', '', regex=True)
    )
    return df


def _parse_percent_series(s: pd.Series) -> pd.Series:
    """
    Convert percentage strings to numeric values.

    Handles common percentage formats (e.g., '13.5%', '1,234.56%') by removing
    percentage signs and commas, then converting to float.

    Args:
        s: Series containing percentage values as strings.
    
    Returns:
        Series with numeric perccentage values.
    """
    return (
        s.astype(str)
        .str.replace('%','', regex = False)
        .str.replace(',','', regex = False)
        .replace({'nan': np.nan, 'None': np.nan})
        .astype(float)
    )


def _extract_term(term_series: pd.Series) -> pd.Series:
    """
    Extract numeric loan term from formatted strings (e.g., '36 months' -> 36).

    Converts loan term strings to integer values representing the number of months.

    Args:
        term_series: Series containing loan term strings.
    
    Returns:
        Series with numeric loan term values.
    """
    return(
        term_series.astype(str)
        .str.extract(r'([0-9]+)', expand=True)
        .astype(float)
        .iloc[:,0]
    )


def _emp_length_to_int(s: pd.Series) -> pd.Series:
    """
    Convert employment length strings to integer values.

    What was done:
        - 'Replaced '< 1 year' with 0.
        - 'Replaced '10+ years' with 10.
        - 'Replaced 'n/a' with NaN.
        - 'Replaced 5 years' with 5
        - 'Replaced 1 year' with 1.

    Args:
        s: Series containing employment length strings.
    
    Returns:
        Series with numeric employment length values.
    """
    s = s.astype(str).str.strip()
    s = s.replace({'< 1 year': '0', '1 year': '1', 'n/a': np.nan, '': np.nan})
    s = s.str.replace(' years', '', regex=False).str.replace(' year', '', regex=False)
    s = s.str.replace('+', '', regex=False) # Handle '10+ years'
    return pd.to_numeric(s, errors='coerce')


def _drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are irrelevant for modeling or introduce data leakage.
    
    Dropped column categories:
        - Identifiers: id, member_id, url (not predictive)
        - Post-funding metrics: payment amounts, recovery fees (target leakage)
        - Administrative: policy_code, application_type (constant or non-informative)
        - Free text: desc, title (require separate NLP processing)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with irrelevant columns removed
    """

    df = df.copy()
    # Define columns to drop
    drop_cols = [
        # Identifiers (non-predictive)
        'id', 'member_id', 'url',

        # Administrative (non-informative)
        'policy_code', 'application_type',

        # Free text fields (reuqire special handling)
        'desc', 'title',

        # Post-funding metrics (target leakage)
        'out_prncp', 'out_prncp_inv', 'collection_recovery_fee', 'recoveries',
        'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'total_paymnt',
        'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int'
    ]

    # Onlu drop columns that exist in the DataFrame
    existing_cols = [col for col in drop_cols if col in df.columns]
    return df.drop(columns=existing_cols)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types for analysis.

    Transformation:
        - Convert date columns to datetime.
        - Categorical conversion: Extract numeric components from formatted strings.
        - Convert percentage strings to floats.
        - Feature engineering: Create new features from existing data.

    Args:
        df: Input DataFrame with mixed data types.

    Returns:
        DataFrame with properly typed columns and derived features.
        """
    df = df.copy()

    # Convert date columns to datetime
    # Lending Club data consists typicall of the 'Mon-YYYY' format
    date_columns = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d']
    for date_col in date_columns:
        if date_col in df.columns:
            date_str = df[date_col].astype(str).str.strip()

            # Attempt to parse dates
            parsed_dates = pd.to_datetime(date_str, format='%b-%Y', errors='coerce')

            # Fallback to flexible parsing if standard format fails
            if parsed_dates.isna().sum() > (len(parsed_dates) * 0.1):  # If more than 10% fail to parse
                fallback_dates = pd.to_datetime(date_str, errors='coerce', infer_datetime_format=True)
                parsed_dates = parsed_dates.fillna(fallback_dates)
            
            df[date_col] = parsed_dates
    
    # Convert percentage columns to numeric
    percentage_columns = ['int_rate', 'revol_util']
    for pct_col in percentage_columns:
        if pct_col in df.columns:
            df[pct_col] = _parse_percent_series(df[pct_col])

    # Extract numeric term in months
    if 'term' in df.columns:
        df['term_months'] = _extract_term(df['term'])

    # Convert employment length to integer
    if 'emp_length' in df.columns:
        df['emp_length_int'] = _emp_length_to_int(df['emp_length'])
    
    # Calculate mean FICO score from range
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_mean'] = (
            pd.to_numeric(df['fico_range_low'], errors='coerce') + 
            pd.to_numeric(df['fico_range_high'], errors='coerce')
            ) / 2.0
        
    # Ensure core numeric columns are properly typed
    numeric_columns = [
        'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
        'annual_inc', 'dti', 'revol_bal'
    ]
    for num_col in numeric_columns:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

    return df


def _handle_missing(df: pd.DataFrame, drop_threshold: float = DEFAULT_DROP_THRESHOLD) -> pd.DataFrame:
    """
    Implement comprehensive missing data strategy.

    Steps:
        - Drop columns exceeding missingness threshold (set to a default of 60%).
        - Drop rows with missing target variables or essential features.
        - Impute remaining numeric columns with median values.
        - Impute remaining categorical columns with mode values.

    Args:
        df: Input DataFrame with missing values.
        drop_threshold: Proportion of missing values allowed per column

    Returns:
        Dataframe with missing values handled according to the defined strategy.
    """
    df = df.copy()

    # Remove columns with excessive missingness
    missing_fraction = df.isna().mean()
    high_missing_cols = missing_fraction[missing_fraction > drop_threshold].index.tolist()
    if high_missing_cols:
        df = df.drop(columns=high_missing_cols)
    
    # Remove rows with missing critical values
    # Drop rows where target variable is missing
    if 'loan_status' in df.columns:
        df = df[~df['loan_status'].isna()].copy()

    # Drop rows where essential features are missing
    essential_features = [col for col in ['loan_amnt', 'int_rate', 'annual_inc'] if col in df.columns]
    if essential_features:
        df = df.dropna(subset=essential_features)
    
    # Numeric imputation with missing indicators
    # Build all missing indicator flags first to avoid fragmentation warnings
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    missing_indicators = {}

    for col in numeric_cols:
        if df[col].isna().any():
            # Create missing indicator flag
            missing_indicators[f'{col}_missing'] = df[col].isna()

            # Impute with median
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    
    # Add all missing indicator columns at once for efficiency
    if missing_indicators:
        indicators_df = pd.DataFrame(missing_indicators, index=df.index)
        df = pd.concat([df, indicators_df], axis=1)

    # Categoorical imputation with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna('Unknown')

    return df
    

def clean_accepted(
        in_path: str | Path, 
        out_path: str | Path,
        drop_threshold: float = DEFAULT_DROP_THRESHOLD
) -> pd.DataFrame:
    """
    Execute the complete data cleaning pipeline for accepted loans.

    The stages involved in the pipeline are as follows:
        1. Load raw data from CSV (supports gzip compression)
        2. Standardize column names to snake_case.
        3. Drop irrelevant columns that do not contribute to modeling.
        4. Coerce columns to appropriate data types.
        5. Handle missing values using a defined strategy.
        6. Reorder columns to place the target variable first.
        7. Persist the cleaned DataFrame to a CSV file.
    
    Args:
        in_path: Path to the raw CSV file containing accepted loans data.
        out_path: Path to save the cleaned CSV file.
        drop_threshold: Proportion of missing values allowed per column before dropping.
    
    Returns:
        Cleaned DataFrame ready for analysis or modeling.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    # Load raw data
    df = pd.read_csv(in_path, compression='infer', low_memory=False)

    # Execute cleaning pipeline
    df = _snake_case_columns(df)
    df = _drop_irrelevant_columns(df)
    df = _coerce_dtypes(df)
    df = _handle_missing(df, drop_threshold=drop_threshold)

    # Reorder columns to place target variable first
    cols = df.columns.tolist()
    if 'loan_status' in cols:
        cols = ['loan_status'] + [c for c in cols if c!= 'loan_status']
        df = df[cols]
    
    # Ensure that the output directory exists and save the cleaned data
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df

if __name__ == "__main__":
    """
    Command-line interface for data cleaning pipeline.
    
    Usage:
        python data_preprocessing.py --in data/raw/file.csv.gz --out data/processed/clean.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean Lending Club dataset with reproducible preprocessing steps.'
    )
    parser.add_argument(
        '--in', 
        dest='infile', 
        default='data/raw/accepted_2007_to_2018Q4.csv.gz',
        help='Path to raw input CSV file (supports gzip compression)'
    )
    parser.add_argument(
        '--out', 
        dest='outfile', 
        default='data/processed/accepted_clean.csv',
        help='Path for cleaned output CSV file'
    )
    parser.add_argument(
        '--drop-thresh', 
        dest='drop_thresh', 
        type=float, 
        default=DEFAULT_DROP_THRESHOLD,
        help=f'Missingness threshold for dropping columns (default: {DEFAULT_DROP_THRESHOLD})'
    )
    args = parser.parse_args()

    print('Initiating data cleaning pipeline (this may take several minutes)...')
    df_clean = clean_accepted(args.infile, args.outfile, drop_threshold=args.drop_thresh)
    print(f'Pipeline complete. Cleaned dataset shape: {df_clean.shape}')
    print(f'Output saved to: {args.outfile}')
