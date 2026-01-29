"""Evaluation metrics and reporting for PD models.

Include AUC, KS, basic precision/recall, calibration helpers, and a
simple markdown report writer for regulatory review.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt


def ks_statistic(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return float(np.max(np.abs(tpr - fpr)))


def basic_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (y_scores >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_scores)
    ks = ks_statistic(y_true, y_scores)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    return {"auc": float(auc), "ks": float(ks), "precision": float(prec), "recall": float(rec)}


def plot_calibration(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10, out_path: Path = None):
    # calibration curve as grouped average predicted vs observed
    df = pd.DataFrame({"y": y_true, "p": y_scores})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin").agg({"p": "mean", "y": "mean"}).reset_index()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(grouped["p"], grouped["y"], marker="o", label="Observed vs Predicted")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean Predicted PD")
    ax.set_ylabel("Observed Default Rate")
    ax.set_title("Calibration Plot")
    ax.legend()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics_train: dict, metrics_val: dict, out_path: Path, notes: str = ""):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Performance Report",
        "",
        "## Train Metrics",
        "",
        str(metrics_train),
        "",
        "## Validation Metrics",
        "",
        str(metrics_val),
        "",
        "## Notes",
        "",
        notes,
    ]
    out_path.write_text("\n".join(lines))
