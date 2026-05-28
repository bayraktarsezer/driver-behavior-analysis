"""
Model evaluation utilities.

For both sklearn and PyTorch models:
    - Metric computation (accuracy, F1, confusion matrix)
    - JSON persistence
    - Visualisation (confusion matrix, ROC, feature importance)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('fivethirtyeight')
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

LABEL_NAMES = ["normal", "aggressive", "economic"]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: list[str] = LABEL_NAMES,
) -> dict:
    """
    Compute classification metrics and return them as a dictionary.

    Return keys:
        accuracy, macro_f1, weighted_f1, per_class_f1,
        confusion_matrix, roc_auc_macro (if y_proba provided)
    """
    metrics: dict = {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "macro_f1":     float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1":  float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            label_names[i]: float(s)
            for i, s in enumerate(
                f1_score(y_true, y_pred, average=None, zero_division=0)
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0
        ),
    }

    if y_proba is not None:
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        try:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["roc_auc_macro"] = None

    return metrics


def save_metrics(
    metrics: dict,
    path: str | Path,
    model_name: str = "model",
) -> None:
    """Append metrics to metrics.json without overwriting existing entries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                pass

    existing[model_name] = {
        k: v for k, v in metrics.items() if k != "report"
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved: {path}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] = LABEL_NAMES,
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
    normalize: bool = True,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    idx = np.argsort(importances)[::-1][:top_n]
    names  = [feature_names[i] for i in idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: Optional[list[float]] = None,
    val_accs: Optional[list[float]] = None,
    title: str = "Training History",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    n_plots = 2 if train_accs is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    if train_accs is not None:
        ax2 = axes[1]
        ax2.plot(train_accs, label="Train Accuracy")
        ax2.plot(val_accs,   label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def make_comparison_table(
    all_results: dict[str, dict],
) -> "pd.DataFrame":
    """
    Convert LODO results for multiple models into a comparison DataFrame.

    Parameters
    ----------
    all_results : {model_name: {"fold_metrics": [...], "mean_accuracy": ..., ...}}
        Output of run_lodo_evaluation() / compare_models().

    Returns
    -------
    pd.DataFrame  — rows = models, columns = Accuracy/F1 mean±std + per-class F1
    """
    import pandas as pd

    rows = []
    for name, res in all_results.items():
        fold_accs = [m["accuracy"] for m in res["fold_metrics"]]
        fold_f1s  = [m["macro_f1"] for m in res["fold_metrics"]]

        per_class: dict[str, list[float]] = {}
        for m in res["fold_metrics"]:
            for cls, val in m.get("per_class_f1", {}).items():
                per_class.setdefault(cls, []).append(val)

        row: dict = {
            "Model":    name,
            "Acc mean": float(np.mean(fold_accs)),
            "Acc std":  float(np.std(fold_accs)),
            "F1 mean":  float(np.mean(fold_f1s)),
            "F1 std":   float(np.std(fold_f1s)),
        }
        for cls, vals in per_class.items():
            row[f"F1_{cls}"] = float(np.mean(vals))

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("F1 mean", ascending=False).reset_index(drop=True)
    return df


def plot_model_comparison(
    comparison_df: "pd.DataFrame",
    title: str = "Model Comparison (LODO)",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Bar chart of model comparison results.
    Error bars show ± 1 std across folds.
    """
    import pandas as pd

    models    = comparison_df["Model"].tolist()
    acc_means = comparison_df["Acc mean"].tolist()
    acc_stds  = comparison_df["Acc std"].tolist()
    f1_means  = comparison_df["F1 mean"].tolist()
    f1_stds   = comparison_df["F1 std"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 5))
    bars1 = ax.bar(x - width / 2, acc_means, width, yerr=acc_stds,
                   label="Accuracy", color="#4C72B0", capsize=4)
    bars2 = ax.bar(x + width / 2, f1_means, width, yerr=f1_stds,
                   label="Macro F1", color="#DD8452", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_class_distribution(
    labels: np.ndarray,
    label_names: list[str] = LABEL_NAMES,
    title: str = "Class Distribution",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    unique, counts = np.unique(labels, return_counts=True)
    names  = [label_names[u] for u in unique]
    colors = ["#4C72B0", "#DD8452", "#55A868"][:len(unique)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Sample Count")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
