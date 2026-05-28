"""
Feature engineering helpers.

This module provides higher-level utilities on top of the low-level
functions in preprocessing.py:
    - Feature name grouping and selection
    - Feature importance formatting
    - 3-D tensor construction for LSTM
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

# Mapping from feature name substrings to group names
FEATURE_GROUPS: dict[str, list[str]] = {
    "stat_basic":    ["_mean", "_std", "_min", "_max", "_range"],
    "stat_shape":    ["_skew", "_kurt", "_iqr", "_p25", "_p75", "_rms"],
    "fft":           ["_fft_"],
    "jerk":          ["jerk_"],
    "accel_derived": ["accel_mag_", "lateral_g_", "braking_"],
    "speed":         ["speed_"],
    "orientation":   ["roll_", "pitch_", "yaw_"],
}


def get_feature_columns(
    df: pd.DataFrame,
    meta_cols: Optional[list[str]] = None,
) -> list[str]:
    """Return all non-metadata feature column names."""
    if meta_cols is None:
        meta_cols = ["session_id", "driver", "behavior", "label", "road_type", "window_idx"]
    return [c for c in df.columns if c not in meta_cols]


def group_features(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Assign feature names to predefined groups.
    A feature may match at most one group (first match wins).
    """
    grouped: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}
    grouped["other"] = []

    for feat in feature_names:
        assigned = False
        for group, patterns in FEATURE_GROUPS.items():
            if any(p in feat for p in patterns):
                grouped[group].append(feat)
                assigned = True
                break
        if not assigned:
            grouped["other"].append(feat)

    return {k: v for k, v in grouped.items() if v}


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    k: int = 30,
    method: str = "f_classif",
) -> tuple[np.ndarray, list[str]]:
    """
    Select the top-k features using a univariate statistical test.

    Parameters
    ----------
    method : "f_classif" | "mutual_info"
    """
    scorer = f_classif if method == "f_classif" else mutual_info_classif
    selector = SelectKBest(scorer, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected_names = [name for name, sel in zip(feature_names, mask) if sel]
    return X_selected, selected_names


def feature_importance_df(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of feature name + importance score sorted descending.
    Compatible with scikit-learn model .feature_importances_.
    """
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n)
    return df


# ---------------------------------------------------------------------------
# Sequence tensor construction for LSTM
# ---------------------------------------------------------------------------

def build_sequence_dataset(
    trips: list,            # list[TripData]
    feature_cols: list[str],
    window_s: float = 5.0,
    overlap_s: float = 2.0,
    pad_length: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an LSTM input tensor from raw time-series windows.

    Each window contains unscaled raw signal values;
    shape: (n_windows, timesteps, n_features).

    Parameters
    ----------
    pad_length : If None, uses the longest window length.
                 If set, all windows are zero-padded to this length.

    Returns
    -------
    X : np.ndarray  shape (N, T, F)
    y : np.ndarray  shape (N,)
    """
    from preprocessing import fill_missing, segment_trip

    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for trip in trips:
        df = trip.merged if not trip.merged.empty else trip.accel
        df = fill_missing(df)

        segments = segment_trip(df, window_s=window_s, overlap_s=overlap_s)
        for seg in segments:
            available = [c for c in feature_cols if c in seg.columns]
            arr = seg[available].values.astype(float)
            # Replace NaN with zero
            arr = np.where(np.isnan(arr), 0.0, arr)
            sequences.append(arr)
            labels.append(trip.meta.label)

    if not sequences:
        return np.empty((0, 0, 0)), np.empty((0,), dtype=int)

    # Pad to equal length
    T = pad_length if pad_length is not None else max(s.shape[0] for s in sequences)
    F = sequences[0].shape[1]

    X = np.zeros((len(sequences), T, F), dtype=np.float32)
    for i, seq in enumerate(sequences):
        t = min(seq.shape[0], T)
        X[i, :t, :] = seq[:t]

    return X, np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Cross-validation helper (driver-based split)
# ---------------------------------------------------------------------------

def leave_one_driver_out_splits(
    df: pd.DataFrame,
    driver_col: str = "driver",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate Leave-One-Driver-Out (LODO) cross-validation indices.

    In each fold one driver is held out for testing and the rest are used
    for training. This strategy measures generalisation across unseen drivers.

    Returns: [(train_idx, test_idx), ...] list of index arrays
    """
    drivers = df[driver_col].unique()
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for test_driver in sorted(drivers):
        test_mask  = df[driver_col] == test_driver
        train_mask = ~test_mask
        splits.append((
            np.where(train_mask)[0],
            np.where(test_mask)[0],
        ))

    return splits


def stratified_driver_split(
    df: pd.DataFrame,
    test_driver: str,
    feature_cols: list[str],
    label_col: str = "label",
    driver_col: str = "driver",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data so one specific driver is the test set and the rest are training.

    Returns: (X_train, X_test, y_train, y_test)
    """
    test_mask  = df[driver_col] == test_driver
    train_mask = ~test_mask

    X_train = df.loc[train_mask, feature_cols].values.astype(float)
    X_test  = df.loc[test_mask,  feature_cols].values.astype(float)
    y_train = df.loc[train_mask, label_col].values.astype(int)
    y_test  = df.loc[test_mask,  label_col].values.astype(int)

    return X_train, X_test, y_train, y_test
