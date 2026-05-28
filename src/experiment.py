"""
Experiment runner.

Main functions:
    run_lodo_evaluation()  — full LODO CV for a single model (per-fold scaling)
    compare_models()       — evaluate multiple models and save results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from evaluate import compute_metrics, make_comparison_table, plot_confusion_matrix, plot_model_comparison, save_metrics
from features import leave_one_driver_out_splits, select_top_features
from models import save_model


# ---------------------------------------------------------------------------
# Single-model LODO evaluation
# ---------------------------------------------------------------------------

def run_lodo_evaluation(
    df: pd.DataFrame,
    model_builder: Callable,
    model_name: str,
    feature_cols: list[str],
    label_col: str = "label",
    driver_col: str = "driver",
    top_k_features: Optional[int] = None,
    models_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Run Leave-One-Driver-Out cross-validation for a single model.

    Per fold:
        1. NaN → impute with training column means
        2. Optional: feature selection on training data only
        3. StandardScaler — fit on training fold only (no data leakage)
        4. Train model → evaluate on test fold
        5. Save fold model (if models_dir is provided)

    Parameters
    ----------
    df            : Processed feature table (meta + feature columns)
    model_builder : Callable that returns a new sklearn-compatible model
    model_name    : Name used for logging and saving
    feature_cols  : List of feature column names to use
    top_k_features: None → use all features; int → select top-k per fold
    models_dir    : Directory to save trained fold models
    verbose       : Print per-fold progress

    Returns
    -------
    {
        "model_name":    str,
        "n_folds":       int,
        "fold_metrics":  list[dict],   # compute_metrics() output per fold
        "mean_accuracy": float,
        "std_accuracy":  float,
        "mean_f1":       float,
        "std_f1":        float,
    }
    """
    splits = leave_one_driver_out_splits(df, driver_col)
    fold_metrics: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        test_driver = df.iloc[test_idx[0]][driver_col]

        X_train_raw = df.iloc[train_idx][feature_cols].values.astype(float)
        X_test_raw  = df.iloc[test_idx][feature_cols].values.astype(float)
        y_train     = df.iloc[train_idx][label_col].values.astype(int)
        y_test      = df.iloc[test_idx][label_col].values.astype(int)

        # Impute NaN using training column means
        col_means = np.nanmean(X_train_raw, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for arr in (X_train_raw, X_test_raw):
            nan_idx = np.where(np.isnan(arr))
            arr[nan_idx] = np.take(col_means, nan_idx[1])

        used_feature_names = feature_cols

        # Feature selection (training data only)
        if top_k_features is not None:
            X_train_raw, selected_names = select_top_features(
                X_train_raw, y_train, feature_cols, k=top_k_features
            )
            mask = np.array([fn in set(selected_names) for fn in feature_cols])
            X_test_raw = X_test_raw[:, mask]
            used_feature_names = selected_names

        # Scale (fit on training fold only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        # Train
        model = model_builder()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        m = compute_metrics(y_test, y_pred, y_proba)
        m["fold"]        = fold_i
        m["test_driver"] = test_driver
        fold_metrics.append(m)

        # Save fold model
        if models_dir is not None:
            models_dir = Path(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            save_model(
                model,
                models_dir / f"{model_name}_fold{fold_i}_test{test_driver}.joblib",
            )

        if verbose:
            print(
                f"  Fold {fold_i + 1}/6 | Test: {test_driver} | "
                f"Acc={m['accuracy']:.3f}  Macro-F1={m['macro_f1']:.3f}"
            )

    accs = [m["accuracy"] for m in fold_metrics]
    f1s  = [m["macro_f1"] for m in fold_metrics]

    result = {
        "model_name":    model_name,
        "n_folds":       len(splits),
        "fold_metrics":  fold_metrics,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "mean_f1":       float(np.mean(f1s)),
        "std_f1":        float(np.std(f1s)),
    }

    if verbose:
        print(
            f"  >> {model_name}: "
            f"Acc={result['mean_accuracy']:.3f}+/-{result['std_accuracy']:.3f}  "
            f"F1={result['mean_f1']:.3f}+/-{result['std_f1']:.3f}"
        )

    return result


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(
    df: pd.DataFrame,
    model_builders: dict[str, Callable],
    feature_cols: list[str],
    label_col: str = "label",
    driver_col: str = "driver",
    top_k_features: Optional[int] = None,
    results_dir: Optional[Path] = None,
    verbose: bool = True,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """
    Evaluate multiple models with LODO and save results.

    Parameters
    ----------
    model_builders : {"ModelName": builder_function, ...}
    results_dir    : Metrics, figures, and models are saved here

    Returns
    -------
    (all_results, comparison_df)
        all_results   : {model_name: run_lodo_evaluation() output}
        comparison_df : make_comparison_table() output (DataFrame)
    """
    if results_dir is not None:
        results_dir = Path(results_dir)
        fig_dir     = results_dir / "figures"
        metrics_dir = results_dir / "metrics"
        models_dir  = results_dir / "models"
        fig_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
    else:
        fig_dir = metrics_dir = models_dir = None

    all_results: dict[str, dict] = {}

    for name, builder in model_builders.items():
        if verbose:
            print(f"\n[{name}]")
        res = run_lodo_evaluation(
            df=df,
            model_builder=builder,
            model_name=name,
            feature_cols=feature_cols,
            top_k_features=top_k_features,
            models_dir=models_dir,
            verbose=verbose,
        )
        all_results[name] = res

        # Append fold metrics to JSON
        if metrics_dir is not None:
            metrics_path = metrics_dir / "metrics.json"
            _append_json(metrics_path, name, res)

    # Build comparison table
    comp_df = make_comparison_table(all_results)

    if verbose:
        print("\n=== LODO Comparison Table ===")
        print(comp_df.to_string(index=False))

    # Save figures
    if fig_dir is not None:
        plot_model_comparison(
            comp_df,
            save_path=fig_dir / "model_comparison.png",
        )

        # Confusion matrix for best model (last fold)
        best_name = comp_df.iloc[0]["Model"]
        best_res  = all_results[best_name]
        last_fold = best_res["fold_metrics"][-1]
        _save_best_cm(
            last_fold=last_fold,
            model_name=best_name,
            fig_dir=fig_dir,
        )

    return all_results, comp_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_json(path: Path, model_name: str, result: dict) -> None:
    existing: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                pass

    serializable = {
        "mean_accuracy": result["mean_accuracy"],
        "std_accuracy":  result["std_accuracy"],
        "mean_f1":       result["mean_f1"],
        "std_f1":        result["std_f1"],
        "n_folds":       result["n_folds"],
        "fold_details":  [
            {k: v for k, v in m.items() if k != "report"}
            for m in result["fold_metrics"]
        ],
    }
    existing[model_name] = serializable

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def _save_best_cm(last_fold: dict, model_name: str, fig_dir: Path) -> None:
    import numpy as np
    from evaluate import LABEL_NAMES

    cm_raw = np.array(last_fold["confusion_matrix"])
    n      = cm_raw.shape[0]

    # Reconstruct y_true / y_pred from confusion matrix for visualisation
    y_true_vis: list[int] = []
    y_pred_vis: list[int] = []
    for i in range(n):
        for j in range(n):
            count = int(cm_raw[i, j])
            y_true_vis.extend([i] * count)
            y_pred_vis.extend([j] * count)

    plot_confusion_matrix(
        np.array(y_true_vis),
        np.array(y_pred_vis),
        label_names=LABEL_NAMES[:n],
        title=f"{model_name} — Confusion Matrix (Last Fold)",
        save_path=fig_dir / f"cm_{model_name.lower().replace(' ', '_')}.png",
    )
