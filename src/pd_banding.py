"""Create PD bands and summaries for business use and documentation."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from datetime import datetime

import pandas as pd
import numpy as np

# Ensure project root is in sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Create PD bands and summaries for business use and documentation")
    p.add_argument("--scored", type=Path, default=Path("data/processed/scored_validation.csv"))
    p.add_argument("--n-bands", type=int, default=10, help="Number of PD bands to create")
    p.add_argument("--out-md", type=Path, default=Path("reports/pd_bands.md"))
    p.add_argument("--out-csv", type=Path, default=Path("reports/pd_bands_summary.csv"))
    return p.parse_args()


def create_markdown_table(summary: pd.DataFrame) -> str:
    """Generate a formatted markdown table from the summary dataframe."""
    lines = []
    
    # Header
    lines.append("| Band | PD Range | Observed Default Rate | Population Share | Count |")
    lines.append("|------|----------|----------------------|------------------|-------|")
    
    # Data rows
    for _, row in summary.iterrows():
        pd_range = f"[{row['pd_min']:.4f}, {row['pd_max']:.4f}]"
        default_rate = f"{row['observed_default_rate']:.2%}"
        pop_share = f"{row['population_share']:.2%}"
        count = f"{int(row['count']):,}"
        
        lines.append(f"| {int(row['band'])} | {pd_range} | {default_rate} | {pop_share} | {count} |")
    
    return "\n".join(lines)


def calculate_metrics(df: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """Calculate additional performance metrics."""
    return {
        "total_observations": len(df),
        "overall_default_rate": df["default"].mean(),
        "pd_range_min": df["pd_hat"].min(),
        "pd_range_max": df["pd_hat"].max(),
        "bands_created": len(summary),
        "avg_band_size": summary["count"].mean(),
        "calibration_gap": abs(summary["observed_default_rate"] - 
                              summary[["pd_min", "pd_max"]].mean(axis=1)).mean()
    }


def generate_risk_labels(n_bands: int) -> dict:
    """Generate risk labels based on number of bands."""
    if n_bands <= 5:
        labels = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
    elif n_bands <= 10:
        labels = ["Minimal Risk", "Very Low Risk", "Low Risk", "Low-Medium Risk", 
                 "Medium Risk", "Medium-High Risk", "High Risk", "Very High Risk", 
                 "Critical Risk", "Extreme Risk"]
    else:
        # Generic labels for many bands
        labels = [f"Band {i}" for i in range(n_bands)]
    
    return {i: labels[min(i, len(labels)-1)] for i in range(n_bands)}


def generate_action_recommendations(summary: pd.DataFrame, risk_labels: dict) -> str:
    """Generate dynamic action recommendations based on observed default rates."""
    lines = []
    
    for _, row in summary.iterrows():
        band = int(row['band'])
        risk_label = risk_labels.get(band, f"Band {band}")
        default_rate = row['observed_default_rate']
        
        # Dynamic recommendations based on default rate
        if default_rate < 0.01:
            action = "**Auto-approve** - Minimal risk, standard terms"
        elif default_rate < 0.05:
            action = "**Auto-approve** with monitoring - Low risk, consider favorable terms"
        elif default_rate < 0.10:
            action = "**Conditional approval** - Manual review for high-value cases"
        elif default_rate < 0.20:
            action = "**Manual review required** - Enhanced due diligence, risk-based pricing"
        elif default_rate < 0.30:
            action = "**Selective approval** - Decline unless strong mitigating factors"
        else:
            action = "**Auto-decline** - Unacceptable risk level"
        
        lines.append(f"- **{risk_label}** (Band {band}): {action}")
    
    return "\n".join(lines)


def create_enhanced_markdown(args, summary: pd.DataFrame, df: pd.DataFrame, 
                            risk_labels: dict, metrics: dict) -> str:
    """Create comprehensive markdown report."""
    report = []
    
    # Title and metadata
    report.append("# PD Banding Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Source Data:** `{args.scored}`")
    report.append(f"**Number of Bands:** {args.n_bands}\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(f"- **Total Observations:** {metrics['total_observations']:,}")
    report.append(f"- **Overall Default Rate:** {metrics['overall_default_rate']:.2%}")
    report.append(f"- **PD Score Range:** [{metrics['pd_range_min']:.4f}, {metrics['pd_range_max']:.4f}]")
    report.append(f"- **Bands Created:** {metrics['bands_created']}")
    report.append(f"- **Average Band Size:** {metrics['avg_band_size']:.0f} observations")
    report.append(f"- **Mean Calibration Gap:** {metrics['calibration_gap']:.4f}\n")
    
    # Band Performance Table
    report.append("## Band Performance Summary\n")
    report.append(create_markdown_table(summary))
    report.append("")
    
    # Detailed Band Descriptions
    report.append("## Detailed Band Analysis\n")
    for _, row in summary.iterrows():
        band = int(row['band'])
        risk_label = risk_labels.get(band, f"Band {band}")
        report.append(f"### {risk_label} (Band {band})\n")
        report.append(f"- **PD Range:** [{row['pd_min']:.6f}, {row['pd_max']:.6f}]")
        report.append(f"- **Observed Default Rate:** {row['observed_default_rate']:.4%}")
        report.append(f"- **Population Share:** {row['population_share']:.2%}")
        report.append(f"- **Count:** {int(row['count']):,} observations")
        
        # Add calibration assessment
        mid_pd = (row['pd_min'] + row['pd_max']) / 2
        calibration_diff = row['observed_default_rate'] - mid_pd
        if abs(calibration_diff) < 0.02:
            calibration = "Well calibrated ✓"
        elif calibration_diff > 0:
            calibration = f"Under-predicting risk (observed {abs(calibration_diff):.2%} higher)"
        else:
            calibration = f"Over-predicting risk (observed {abs(calibration_diff):.2%} lower)"
        report.append(f"- **Calibration:** {calibration}\n")
    
    # Action Recommendations
    report.append("## Recommended Actions by Band\n")
    report.append(generate_action_recommendations(summary, risk_labels))
    report.append("")
    
    # Implementation Notes
    report.append("## Implementation Guidelines\n")
    report.append("### Model Monitoring")
    report.append("- Review band performance monthly to detect distribution shifts")
    report.append("- Monitor actual vs predicted default rates for calibration drift")
    report.append("- Recalibrate bands quarterly or when performance degrades\n")
    
    report.append("### Business Integration")
    report.append("- Map bands to existing credit policies and approval workflows")
    report.append("- Set override authorities for borderline cases")
    report.append("- Document exceptions and track override performance")
    report.append("- Integrate with pricing strategy based on risk-adjusted returns\n")
    
    report.append("### Regulatory Compliance")
    report.append("- Ensure consistent application across protected classes")
    report.append("- Document business justification for band thresholds")
    report.append("- Maintain audit trail of band definitions and changes")
    report.append("- Prepare adverse action notices for declined applications\n")
    
    # Appendix
    report.append("## Appendix\n")
    report.append("### Methodology")
    report.append(f"- Bands created using quantile-based segmentation (n={args.n_bands})")
    report.append("- Observed default rates calculated from validation dataset")
    report.append("- Population shares represent proportion of total observations")
    report.append("- Risk labels assigned based on observed default rate thresholds\n")
    
    return "\n".join(report)


def main(argv=None):
    args = parse_args() if argv is None else parse_args()

    if not args.scored.exists():
        print(f"Scored validation set not found: {args.scored}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.scored, index_col=0)
    if "pd_hat" not in df.columns or "default" not in df.columns:
        print(f"Scored validation set must contain 'pd_hat' and 'default' columns.", file=sys.stderr)
        sys.exit(3)
    
    n = args.n_bands
    # Create bands by quantiles of predicted PD
    df = df.copy()
    df['pd_band'] = pd.qcut(df['pd_hat'], q=n, labels=False, duplicates='drop')

    summary_rows = []
    for b in sorted(df["pd_band"].unique()):
        sub = df[df["pd_band"] == b]
        pd_min = float(sub["pd_hat"].min())
        pd_max = float(sub["pd_hat"].max())
        obs_rate = float(sub["default"].mean())
        pop_share = float(len(sub)) / len(df)
        summary_rows.append({
            "band": int(b),
            "pd_min": pd_min,
            "pd_max": pd_max,
            "observed_default_rate": obs_rate,
            "population_share": pop_share,
            "count": len(sub)
        })

    summary = pd.DataFrame(summary_rows).sort_values("band")
    
    # Calculate additional metrics
    metrics = calculate_metrics(df, summary)
    risk_labels = generate_risk_labels(n)
    
    # Save CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)

    # Write enhanced markdown report
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    markdown_content = create_enhanced_markdown(args, summary, df, risk_labels, metrics)
    
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(markdown_content)

    print(f"✓ Wrote PD band summary to {args.out_md} and {args.out_csv}")
    print(f"✓ Created {len(summary)} bands covering {len(df):,} observations")
    print(f"✓ Overall default rate: {metrics['overall_default_rate']:.2%}")


if __name__ == "__main__":    
    main()