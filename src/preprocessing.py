"""
Preprocessing pipeline.

Steps:
    1. Missing value imputation
    2. Sliding window segmentation (5 s window, 2 s overlap)
    3. Statistical feature extraction
    4. Frequency-domain features (FFT)
    5. Derived physics-based features (jerk, lateral G-force)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Missing value imputation
# ---------------------------------------------------------------------------

def fill_missing(
    df: pd.DataFrame,
    method: str = "ffill",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply forward-fill then backward-fill for time-series data.

    Parameters
    ----------
    df     : Input DataFrame
    method : "ffill" | "bfill" | "interpolate"
    limit  : Maximum number of consecutive NaNs to fill
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if method == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(
            method="linear", limit=limit, limit_direction="both"
        )
    else:
        df[numeric_cols] = (
            df[numeric_cols]
            .ffill(limit=limit)
            .bfill(limit=limit)
        )
    return df


# ---------------------------------------------------------------------------
# 2. Sliding window segmentation
# ---------------------------------------------------------------------------

def sliding_window_indices(
    timestamps: np.ndarray,
    window_s: float = 5.0,
    overlap_s: float = 2.0,
) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs for sliding windows over timestamps.

    Stride calculation:
        stride = window_s - overlap_s   →   stride = 3 s, window = 5 s

    Returns
    -------
    [(start_idx, end_idx), ...]  — half-open intervals [start, end)
    """
    if len(timestamps) == 0:
        return []

    stride_s = window_s - overlap_s
    t_start = timestamps[0]
    t_end = timestamps[-1]

    windows: list[tuple[int, int]] = []
    win_start_t = t_start

    while win_start_t + window_s <= t_end + 1e-6:
        win_end_t = win_start_t + window_s
        mask = (timestamps >= win_start_t) & (timestamps < win_end_t)
        indices = np.where(mask)[0]
        if len(indices) >= 2:
            windows.append((int(indices[0]), int(indices[-1]) + 1))
        win_start_t += stride_s

    return windows


def segment_trip(
    df: pd.DataFrame,
    window_s: float = 5.0,
    overlap_s: float = 2.0,
    timestamp_col: str = "timestamp",
) -> list[pd.DataFrame]:
    """
    Split a single trip into overlapping sliding windows.

    Returns: list of DataFrames, one per window.
    """
    if df.empty or timestamp_col not in df.columns:
        return []

    ts = df[timestamp_col].values
    indices = sliding_window_indices(ts, window_s=window_s, overlap_s=overlap_s)
    return [df.iloc[s:e].reset_index(drop=True) for s, e in indices]


# ---------------------------------------------------------------------------
# 3. Statistical feature extraction
# ---------------------------------------------------------------------------

def statistical_features(
    segment: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Compute statistical features for a single window segment.

    Features extracted per column:
        mean, std, min, max, range, rms, skewness, kurtosis, iqr, p25, p75
    """
    from scipy.stats import kurtosis, skew

    features: dict[str, float] = {}

    for col in feature_cols:
        if col not in segment.columns:
            continue
        vals = segment[col].dropna().values.astype(float)

        if len(vals) == 0:
            for suffix in ("mean", "std", "min", "max", "range",
                           "rms", "skew", "kurt", "iqr", "p25", "p75"):
                features[f"{col}_{suffix}"] = np.nan
            continue

        p25, p75 = np.percentile(vals, [25, 75])
        features[f"{col}_mean"]  = float(np.mean(vals))
        features[f"{col}_std"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        features[f"{col}_min"]   = float(np.min(vals))
        features[f"{col}_max"]   = float(np.max(vals))
        features[f"{col}_range"] = float(np.max(vals) - np.min(vals))
        features[f"{col}_rms"]   = float(np.sqrt(np.mean(vals ** 2)))
        features[f"{col}_skew"]  = float(skew(vals)) if len(vals) > 2 else 0.0
        features[f"{col}_kurt"]  = float(kurtosis(vals)) if len(vals) > 3 else 0.0
        features[f"{col}_iqr"]   = float(p75 - p25)
        features[f"{col}_p25"]   = float(p25)
        features[f"{col}_p75"]   = float(p75)

    return features


# ---------------------------------------------------------------------------
# 4. FFT features
# ---------------------------------------------------------------------------

def fft_features(
    segment: pd.DataFrame,
    feature_cols: list[str],
    fs: float = 10.0,
    n_coeffs: int = 5,
) -> dict[str, float]:
    """
    Fast Fourier Transform based frequency-domain features.

    Per column:
        - First n_coeffs FFT magnitude coefficients
        - Dominant frequency (Hz)
        - Spectral entropy
        - Total power

    Parameters
    ----------
    fs       : Sampling frequency (Hz). Default 10 for accelerometer.
    n_coeffs : Number of FFT coefficients to retain.
    """
    features: dict[str, float] = {}

    for col in feature_cols:
        if col not in segment.columns:
            continue
        vals = segment[col].dropna().values.astype(float)
        n = len(vals)

        if n < 4:
            for k in range(n_coeffs):
                features[f"{col}_fft_{k}"] = np.nan
            features[f"{col}_fft_dom_freq"]    = np.nan
            features[f"{col}_fft_spectral_ent"] = np.nan
            features[f"{col}_fft_total_power"]  = np.nan
            continue

        # Hann window reduces spectral leakage
        windowed = vals * np.hanning(n)
        fft_vals  = np.abs(np.fft.rfft(windowed))
        freqs     = np.fft.rfftfreq(n, d=1.0 / fs)

        # First n coefficients (including DC)
        for k in range(n_coeffs):
            features[f"{col}_fft_{k}"] = float(fft_vals[k]) if k < len(fft_vals) else 0.0

        # Dominant frequency (skip DC bin)
        dom_idx = int(np.argmax(fft_vals[1:]) + 1)
        features[f"{col}_fft_dom_freq"] = float(freqs[dom_idx]) if dom_idx < len(freqs) else 0.0

        # Spectral entropy
        psd = fft_vals ** 2
        psd_norm = psd / (psd.sum() + 1e-12)
        features[f"{col}_fft_spectral_ent"] = float(
            -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        )

        # Total power
        features[f"{col}_fft_total_power"] = float(np.sum(psd))

    return features


# ---------------------------------------------------------------------------
# 5. Derived physics-based features
# ---------------------------------------------------------------------------

def derived_features(
    segment: pd.DataFrame,
    fs: float = 10.0,
) -> dict[str, float]:
    """
    Compute physics-based derived features.

    Features:
        jerk_x/y/z  : Time derivative of acceleration (G/s)
        lateral_g   : Lateral G-force magnitude statistics
        accel_mag   : 3D acceleration magnitude = sqrt(x² + y² + z²)
        braking_*   : Negative longitudinal acceleration (braking) statistics
    """
    features: dict[str, float] = {}
    dt = 1.0 / fs

    for axis, col in [("x", "acc_x_kf"), ("y", "acc_y_kf"), ("z", "acc_z_kf")]:
        if col not in segment.columns:
            continue
        vals = segment[col].ffill().bfill().values.astype(float)

        # Jerk = d(acceleration)/dt
        jerk = np.diff(vals) / dt if len(vals) > 1 else np.array([0.0])
        features[f"jerk_{axis}_mean"] = float(np.mean(np.abs(jerk)))
        features[f"jerk_{axis}_max"]  = float(np.max(np.abs(jerk)))
        features[f"jerk_{axis}_std"]  = float(np.std(jerk, ddof=1)) if len(jerk) > 1 else 0.0

    # 3D acceleration magnitude
    if all(c in segment.columns for c in ("acc_x_kf", "acc_y_kf", "acc_z_kf")):
        ax = segment["acc_x_kf"].ffill().bfill().values.astype(float)
        ay = segment["acc_y_kf"].ffill().bfill().values.astype(float)
        az = segment["acc_z_kf"].ffill().bfill().values.astype(float)

        mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        features["accel_mag_mean"] = float(np.mean(mag))
        features["accel_mag_max"]  = float(np.max(mag))
        features["accel_mag_std"]  = float(np.std(mag, ddof=1)) if len(mag) > 1 else 0.0

        # Lateral G-force (perpendicular to travel direction)
        lat_g = np.abs(ay)
        features["lateral_g_mean"] = float(np.mean(lat_g))
        features["lateral_g_max"]  = float(np.max(lat_g))
        features["lateral_g_rms"]  = float(np.sqrt(np.mean(lat_g ** 2)))

        # Braking events (negative longitudinal acceleration)
        braking = ax[ax < -0.05]  # 0.05 G threshold
        features["braking_count"] = float(len(braking))
        features["braking_mean"]  = float(np.mean(np.abs(braking))) if len(braking) > 0 else 0.0
        features["braking_max"]   = float(np.max(np.abs(braking))) if len(braking) > 0 else 0.0

    # Speed statistics (from GPS data in merged DataFrame)
    if "speed_kmh" in segment.columns:
        spd = segment["speed_kmh"].dropna().values.astype(float)
        if len(spd) > 0:
            features["speed_mean"]  = float(np.mean(spd))
            features["speed_std"]   = float(np.std(spd, ddof=1)) if len(spd) > 1 else 0.0
            features["speed_max"]   = float(np.max(spd))
            features["speed_range"] = float(np.max(spd) - np.min(spd))

    return features


# ---------------------------------------------------------------------------
# 6. Combined feature extraction
# ---------------------------------------------------------------------------

# Base accelerometer columns (raw + Kalman-filtered)
DEFAULT_STAT_COLS: list[str] = [
    "acc_x", "acc_y", "acc_z",
    "acc_x_kf", "acc_y_kf", "acc_z_kf",
    "roll", "pitch", "yaw",
]

DEFAULT_FFT_COLS: list[str] = [
    "acc_x_kf", "acc_y_kf", "acc_z_kf",
]


def extract_features_from_segment(
    segment: pd.DataFrame,
    stat_cols: list[str] = DEFAULT_STAT_COLS,
    fft_cols: list[str] = DEFAULT_FFT_COLS,
    fs: float = 10.0,
    fft_n_coeffs: int = 5,
) -> dict[str, float]:
    """
    Combine all feature groups for a single window segment.

    Returns a flat feature dict ready to be fed into an ML model.
    """
    feats: dict[str, float] = {}
    feats.update(statistical_features(segment, stat_cols))
    feats.update(fft_features(segment, fft_cols, fs=fs, n_coeffs=fft_n_coeffs))
    feats.update(derived_features(segment, fs=fs))
    return feats


# ---------------------------------------------------------------------------
# 7. Full trip processing pipeline
# ---------------------------------------------------------------------------

def process_trips(
    trips: list,             # list[TripData] — Any to avoid circular import
    window_s: float = 5.0,
    overlap_s: float = 2.0,
    fs: float = 10.0,
    stat_cols: list[str] = DEFAULT_STAT_COLS,
    fft_cols: list[str] = DEFAULT_FFT_COLS,
    fft_n_coeffs: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on all trips and return a flat feature table.

    Returned DataFrame columns:
        session_id, driver, behavior, label, road_type, window_idx, <features...>
    """
    from tqdm import tqdm

    all_rows: list[dict] = []
    trip_iter = tqdm(trips, desc="Processing trips") if verbose else trips

    for trip in trip_iter:
        # Use merged DataFrame if available; fall back to raw accelerometer
        df = trip.merged if not trip.merged.empty else trip.accel
        df = fill_missing(df)

        segments = segment_trip(df, window_s=window_s, overlap_s=overlap_s)

        for win_idx, seg in enumerate(segments):
            feats = extract_features_from_segment(
                seg,
                stat_cols=stat_cols,
                fft_cols=fft_cols,
                fs=fs,
                fft_n_coeffs=fft_n_coeffs,
            )
            feats.update({
                "session_id": trip.meta.session_id,
                "driver":     trip.meta.driver_id,
                "behavior":   trip.meta.behavior,
                "label":      trip.meta.label,
                "road_type":  trip.meta.road_type,
                "window_idx": win_idx,
            })
            all_rows.append(feats)

    if not all_rows:
        return pd.DataFrame()

    df_features = pd.DataFrame(all_rows)

    # Place metadata columns first
    meta_cols = ["session_id", "driver", "behavior", "label", "road_type", "window_idx"]
    feat_cols = [c for c in df_features.columns if c not in meta_cols]
    return df_features[meta_cols + feat_cols]


# ---------------------------------------------------------------------------
# 8. Scaling utilities
# ---------------------------------------------------------------------------

def fit_scaler(
    X_train: np.ndarray,
) -> StandardScaler:
    """Fit a StandardScaler on training data and return it."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def get_feature_matrix(
    df_features: pd.DataFrame,
    meta_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Return the raw (unscaled) feature matrix and labels.

    Use this in cross-validation loops: fit the scaler on the training
    fold only to prevent data leakage.

    Returns: (X_raw, y, feature_col_names)
    """
    if meta_cols is None:
        meta_cols = ["session_id", "driver", "behavior", "label", "road_type", "window_idx"]

    y = df_features["label"].values.astype(int)
    feat_cols = [c for c in df_features.columns if c not in meta_cols]
    X = df_features[feat_cols].values.astype(float)

    # NaN → column mean
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    return X, y, feat_cols


def scale_features(
    df_features: pd.DataFrame,
    meta_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale the feature matrix using the entire dataset.

    WARNING: This function fits on the full dataset — do not use inside
    cross-validation loops. For CV, use get_feature_matrix() and fit a
    StandardScaler on each training fold separately.

    Returns: (X_scaled, y, scaler)
    """
    import warnings
    warnings.warn(
        "scale_features() fits on the entire dataset. "
        "For cross-validation, use get_feature_matrix() and fit the scaler "
        "on the training fold only.",
        UserWarning,
        stacklevel=2,
    )

    if meta_cols is None:
        meta_cols = ["session_id", "driver", "behavior", "label", "road_type", "window_idx"]

    y = df_features["label"].values.astype(int)
    feat_cols = [c for c in df_features.columns if c not in meta_cols]
    X = df_features[feat_cols].values.astype(float)

    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
