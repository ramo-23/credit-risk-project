"""Model validation script: computes OOT performance, PSI, drift, segment-level checks, and writes reports."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.evaluation import ks_statistic
except ImportError:
    # Local import fallback
    import importlib.util
    _ROOT_SRC = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "evaluation", 
        str(_ROOT_SRC / "evaluation.py")
    )
    if spec and spec.loader:
        eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod)
        ks_statistic = eval_mod.ks_statistic
    else:
        raise ImportError("Could not load evaluation module")


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    auc: Optional[float] = None
    ks: Optional[float] = None
    brier: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            'auc': self.auc,
            'ks': self.ks,
            'brier': self.brier
        }


@dataclass
class ValidationConfig:
    """Configuration for validation run."""
    scored_path: Path
    train_path: Path
    time_col: str = 'issue_d'
    out_dir: Path = Path('reports')
    key_features: List[str] = None
    
    def __post_init__(self):
        if self.key_features is None:
            self.key_features = ['loan_amnt', 'annual_inc', 'dti', 'fico_mean']


def compute_basic_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> ValidationMetrics:
    """
    Compute basic classification metrics.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        
    Returns:
        ValidationMetrics object with computed metrics
    """
    metrics = ValidationMetrics()
    
    try:
        metrics.auc = float(roc_auc_score(y_true, y_scores))
        logger.info(f"Computed AUC: {metrics.auc:.4f}")
    except Exception as e:
        logger.warning(f"Failed to compute AUC: {e}")
    
    try:
        metrics.ks = float(ks_statistic(y_true, y_scores))
        logger.info(f"Computed KS: {metrics.ks:.4f}")
    except Exception as e:
        logger.warning(f"Failed to compute KS: {e}")
    
    try:
        metrics.brier = float(brier_score_loss(y_true, y_scores))
        logger.info(f"Computed Brier Score: {metrics.brier:.4f}")
    except Exception as e:
        logger.warning(f"Failed to compute Brier Score: {e}")
    
    return metrics


def compute_psi(
    expected: np.ndarray, 
    actual: np.ndarray, 
    buckets: int = 10,
    epsilon: float = 1e-8
) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.
    
    Args:
        expected: Reference/training distribution
        actual: Current/validation distribution
        buckets: Number of quantile buckets
        epsilon: Small value to avoid log(0)
        
    Returns:
        PSI value (higher indicates more drift)
    """
    try:
        # Create quantile bins based on expected distribution
        quantiles = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
        bins = np.unique(quantiles)
        
        if len(bins) <= 1:
            logger.warning("Insufficient unique bins for PSI calculation")
            return float('nan')
        
        # Compute distributions
        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)
        
        exp_perc = exp_counts / (exp_counts.sum() + epsilon)
        act_perc = act_counts / (act_counts.sum() + epsilon)
        
        # Avoid zeros in log calculation
        exp_perc = np.maximum(exp_perc, epsilon)
        act_perc = np.maximum(act_perc, epsilon)
        
        # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
        psi_val = np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc))
        
        return float(psi_val)
    
    except Exception as e:
        logger.error(f"PSI calculation failed: {e}")
        return float('nan')


def compute_temporal_summary(
    df: pd.DataFrame, 
    time_col: str = 'issue_d', 
    freq: str = 'Q'
) -> pd.DataFrame:
    """
    Compute temporal summary statistics.
    
    Args:
        df: DataFrame with predictions and outcomes
        time_col: Column containing timestamps
        freq: Pandas frequency string ('Q' for quarterly, 'M' for monthly)
        
    Returns:
        DataFrame with temporal aggregations
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df['period'] = df[time_col].dt.to_period(freq).astype(str)
    
    grouped = df.groupby('period').agg(
        avg_pd=('pd_hat', 'mean'),
        obs_rate=('default', 'mean'),
        n=('pd_hat', 'size')
    )
    
    return grouped.reset_index()


def compute_segment_performance(
    df: pd.DataFrame, 
    segment_col: str
) -> pd.DataFrame:
    """
    Compute performance metrics by segment.
    
    Args:
        df: DataFrame with predictions and outcomes
        segment_col: Column to segment by
        
    Returns:
        DataFrame with segment-level metrics
    """
    rows = []
    
    for segment_value, sub_df in df.groupby(segment_col):
        y_true = sub_df['default'].values
        y_pred = sub_df['pd_hat'].values
        
        metrics = compute_basic_metrics(y_true, y_pred)
        
        rows.append({
            'segment': segment_value,
            'n': len(sub_df),
            'obs_rate': float(y_true.mean()),
            'avg_pd': float(y_pred.mean()),
            'auc': metrics.auc,
            'ks': metrics.ks,
            'brier': metrics.brier
        })
    
    return pd.DataFrame(rows).sort_values('segment')


def load_data(path: Path, name: str) -> Optional[pd.DataFrame]:
    """
    Safely load CSV data.
    
    Args:
        path: Path to CSV file
        name: Descriptive name for logging
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        if not path.exists():
            logger.warning(f"{name} file not found: {path}")
            return None
        
        df = pd.read_csv(path, index_col=0)
        logger.info(f"Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load {name} from {path}: {e}")
        return None


def write_validation_report(
    out_dir: Path,
    metrics: ValidationMetrics,
    psi_pd: float,
    feature_psi: Dict[str, float]
) -> None:
    """Write model validation report."""
    report_path = out_dir / 'model_validation.md'
    
    with report_path.open('w', encoding='utf-8') as fh:
        fh.write("# Model Validation Report\n\n")
        fh.write("## Out-of-Time Performance Metrics\n\n")
        
        auc_str = f"{metrics.auc:.4f}" if metrics.auc is not None else "N/A"
        ks_str = f"{metrics.ks:.4f}" if metrics.ks is not None else "N/A"
        brier_str = f"{metrics.brier:.4f}" if metrics.brier is not None else "N/A"
        
        fh.write(f"- **AUC**: {auc_str}\n")
        fh.write(f"- **KS Statistic**: {ks_str}\n")
        fh.write(f"- **Brier Score**: {brier_str}\n\n")
        
        fh.write("## Population Stability Index (PSI)\n\n")
        psi_str = f"{psi_pd:.4f}" if not np.isnan(psi_pd) else "N/A"
        fh.write(f"- **PD Score PSI**: {psi_str}\n\n")
        
        if feature_psi:
            fh.write("### Feature-level PSI\n\n")
            for feature, psi_val in feature_psi.items():
                psi_str = f"{psi_val:.4f}" if not np.isnan(psi_val) else "N/A"
                fh.write(f"- **{feature}**: {psi_str}\n")
    
    logger.info(f"Validation report written to {report_path}")


def write_stability_report(
    out_dir: Path,
    feature_psi: Dict[str, float],
    temporal_summary: pd.DataFrame
) -> None:
    """Write stability analysis report."""
    report_path = out_dir / 'stability_analysis.md'
    
    with report_path.open('w', encoding='utf-8') as fh:
        fh.write("# Stability Analysis Report\n\n")
        fh.write("## Feature PSI Values\n\n")
        fh.write("PSI Interpretation:\n")
        fh.write("- < 0.1: No significant change\n")
        fh.write("- 0.1 - 0.25: Moderate change\n")
        fh.write("- > 0.25: Significant change\n\n")
        
        for feature, psi_val in feature_psi.items():
            psi_str = f"{psi_val:.4f}" if not np.isnan(psi_val) else "N/A"
            fh.write(f"- **{feature}**: {psi_str}\n")
        
        fh.write("\n## Temporal Summary\n\n")
        fh.write(temporal_summary.to_markdown(index=False))
    
    logger.info(f"Stability report written to {report_path}")


def write_assumptions_report(out_dir: Path) -> None:
    """Write assumptions and limitations report."""
    report_path = out_dir / 'assumptions_and_limits.md'
    
    with report_path.open('w', encoding='utf-8') as fh:
        fh.write("# Model Assumptions and Limitations\n\n")
        fh.write("## Key Assumptions\n\n")
        fh.write("- **Data Representativeness**: Validation data represents recent origination periods\n")
        fh.write("- **Economic Stationarity**: Model assumes stable macroeconomic environment\n")
        fh.write("- **Feature Stability**: Input features maintain similar distributions over time\n\n")
        
        fh.write("## Known Limitations\n\n")
        fh.write("- **Use Case**: Model designed for risk segmentation only\n")
        fh.write("- **Stress Scenarios**: Extreme economic conditions not covered in validation\n")
        fh.write("- **Pricing**: Do not use for pricing without additional analysis\n")
    
    logger.info(f"Assumptions report written to {report_path}")


def write_approval_recommendation(
    out_dir: Path,
    metrics: ValidationMetrics,
    psi_pd: float
) -> None:
    """Write approval recommendation report."""
    report_path = out_dir / 'approval_recommendation.md'
    
    # Simple decision logic
    issues = []
    
    if metrics.auc and metrics.auc < 0.65:
        issues.append("AUC below acceptable threshold (0.65)")
    
    if not np.isnan(psi_pd) and psi_pd > 0.25:
        issues.append("Significant population drift detected (PSI > 0.25)")
    
    if metrics.brier and metrics.brier > 0.20:
        issues.append("Calibration concerns (high Brier score)")
    
    decision = "Approve with conditions" if not issues else "Conditional approval - address concerns"
    
    with report_path.open('w', encoding='utf-8') as fh:
        fh.write("# Model Approval Recommendation\n\n")
        fh.write(f"## Recommendation: **{decision}**\n\n")
        
        if issues:
            fh.write("### Identified Issues\n\n")
            for issue in issues:
                fh.write(f"- {issue}\n")
            fh.write("\n")
        
        fh.write("### Conditions and Monitoring Requirements\n\n")
        fh.write("- Implement monthly PSI monitoring for PD scores and key features\n")
        fh.write("- Monitor calibration metrics and trigger recalibration if Brier > 0.20\n")
        fh.write("- Review model performance quarterly with segment-level analysis\n")
        fh.write("- Establish retraining triggers if AUC drops below 0.65\n")
    
    logger.info(f"Approval recommendation written to {report_path}")


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive model validation checks'
    )
    parser.add_argument(
        '--scored',
        type=Path,
        default=Path('data/processed/scored_validation.csv'),
        help='Path to scored validation data'
    )
    parser.add_argument(
        '--train',
        type=Path,
        default=Path('data/processed/accepted_fe.csv'),
        help='Path to training data'
    )
    parser.add_argument(
        '--time-col',
        type=str,
        default='issue_d',
        help='Column name for temporal analysis'
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('reports'),
        help='Output directory for reports'
    )
    
    args = parser.parse_args(argv)
    
    # Load validation data
    df_val = load_data(args.scored, "Validation data")
    if df_val is None:
        logger.error(f"Cannot proceed without validation data")
        return 2
    
    # Validate required columns
    required_cols = ['default', 'pd_hat']
    missing_cols = [col for col in required_cols if col not in df_val.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return 2
    
    # Compute overall metrics
    logger.info("Computing overall validation metrics...")
    metrics = compute_basic_metrics(
        df_val['default'].values,
        df_val['pd_hat'].values
    )
    
    # Compute PSI
    logger.info("Computing PSI metrics...")
    psi_pd = float('nan')
    feature_psi = {}
    
    df_train = load_data(args.train, "Training data")
    if df_train is not None and 'pd_hat' in df_train.columns:
        psi_pd = compute_psi(
            df_train['pd_hat'].values,
            df_val['pd_hat'].values
        )
        logger.info(f"PD Score PSI: {psi_pd:.4f}")
        
        # Feature-level PSI
        config = ValidationConfig(
            scored_path=args.scored,
            train_path=args.train
        )
        
        for feature in config.key_features:
            if feature in df_train.columns and feature in df_val.columns:
                try:
                    feature_psi[feature] = compute_psi(
                        df_train[feature].values.astype(float),
                        df_val[feature].values.astype(float)
                    )
                    logger.info(f"Feature {feature} PSI: {feature_psi[feature]:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to compute PSI for {feature}: {e}")
                    feature_psi[feature] = float('nan')
    
    # Temporal analysis
    logger.info("Computing temporal summary...")
    temporal_df = compute_temporal_summary(df_val, time_col=args.time_col)
    
    # Save temporal summary
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    temporal_path = processed_dir / 'by_period.csv'
    temporal_df.to_csv(temporal_path, index=False)
    logger.info(f"Temporal summary saved to {temporal_path}")
    
    # Segment-level analysis
    segment_analyses = {}
    
    if 'pd_band' in df_val.columns:
        logger.info("Computing PD band segment performance...")
        segment_analyses['pd_band'] = compute_segment_performance(df_val, 'pd_band')
    
    if 'term_months' in df_val.columns:
        logger.info("Computing term segment analysis...")
        segment_analyses['term'] = compute_segment_performance(df_val, 'term_months')
    
    grade_col = 'grade' if 'grade' in df_val.columns else (
        'sub_grade' if 'sub_grade' in df_val.columns else None
    )
    if grade_col:
        logger.info(f"Computing {grade_col} segment analysis...")
        segment_analyses['grade'] = compute_segment_performance(df_val, grade_col)
    
    # Write all reports
    logger.info("Generating validation reports...")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    write_validation_report(args.out_dir, metrics, psi_pd, feature_psi)
    write_stability_report(args.out_dir, feature_psi, temporal_df)
    write_assumptions_report(args.out_dir)
    write_approval_recommendation(args.out_dir, metrics, psi_pd)
    
    logger.info(f"All validation reports generated in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())