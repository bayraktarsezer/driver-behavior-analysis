"""
UAH-DRIVESET-v1 data loader.

Folder name format:
    YYYYMMDDhhmmss-{dist}km-D{n}-{BEHAVIOR}[-{N}]-{ROAD_TYPE}
    Example: 20151110175712-16km-D1-NORMAL1-SECONDARY

File formats:
    RAW_ACCELEROMETERS.txt  ~10 Hz, space-separated, 11 columns
    RAW_GPS.txt             ~1  Hz, space-separated, 12 columns

Label mapping (project target: normal / aggressive / economic):
    NORMAL / NORMAL1 / NORMAL2  → "normal"
    AGGRESSIVE                  → "aggressive"
    DROWSY                      → "economic"   (smooth, efficient driving)
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

ACCEL_COLS: list[str] = [
    "timestamp",
    "activation",       # 1 if speed > 50 km/h
    "acc_x",            # X acceleration raw (Gs) – longitudinal
    "acc_y",            # Y acceleration raw (Gs) – lateral
    "acc_z",            # Z acceleration raw (Gs) – vertical
    "acc_x_kf",         # X acceleration Kalman-filtered (Gs)
    "acc_y_kf",         # Y acceleration Kalman-filtered (Gs)
    "acc_z_kf",         # Z acceleration Kalman-filtered (Gs)
    "roll",             # Roll angle (degrees)
    "pitch",            # Pitch angle (degrees)
    "yaw",              # Yaw angle (degrees)
]

GPS_COLS: list[str] = [
    "timestamp",
    "speed_kmh",        # GPS speed (km/h)
    "latitude",
    "longitude",
    "altitude",
    "vertical_acc",
    "horizontal_acc",
    "course",           # Heading (degrees)
    "diff_course",      # Course change
    "pos_state",        # Internal state
    "lanex_state",      # Internal
    "lanex_history",    # Internal
]

# Label → integer encoding
LABEL_MAP: dict[str, int] = {
    "normal":     0,
    "aggressive": 1,
    "economic":   2,
}
INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Raw dataset behavior string → project label
_RAW_BEHAVIOR_MAP: dict[str, str] = {
    "NORMAL":     "normal",
    "NORMAL1":    "normal",
    "NORMAL2":    "normal",
    "AGGRESSIVE": "aggressive",
    "DROWSY":     "economic",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TripMetadata:
    driver_id: str          # "D1" … "D6"
    behavior: str           # "normal" | "aggressive" | "economic"
    road_type: str          # "MOTORWAY" | "SECONDARY"
    distance_km: Optional[float]
    session_id: str         # Folder name (unique identifier)
    label: int              # LABEL_MAP[behavior]


@dataclass
class TripData:
    meta: TripMetadata
    accel: pd.DataFrame     # RAW_ACCELEROMETERS
    gps: pd.DataFrame       # RAW_GPS
    merged: pd.DataFrame = field(default_factory=pd.DataFrame)  # Merged


# ---------------------------------------------------------------------------
# Session folder name parser
# ---------------------------------------------------------------------------

_SESSION_PATTERN = re.compile(
    r"""
    (?P<datetime>\d{14})        # YYYYMMDDhhmmss
    -
    (?:(?P<distance>[\d.]+)km-)? # optional distance
    (?P<driver>D\d+)             # D1 … D6
    -
    (?P<behavior>NORMAL\d?|AGGRESSIVE|DROWSY)
    -
    (?P<road>MOTORWAY|SECONDARY)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_session_name(folder_name: str) -> Optional[TripMetadata]:
    """Parse a folder name into TripMetadata; returns None if no match."""
    m = _SESSION_PATTERN.search(folder_name)
    if m is None:
        return None

    raw_behavior = m.group("behavior").upper()
    behavior = _RAW_BEHAVIOR_MAP.get(raw_behavior)
    if behavior is None:
        return None

    dist_str = m.group("distance")
    distance = float(dist_str) if dist_str else None

    return TripMetadata(
        driver_id=m.group("driver").upper(),
        behavior=behavior,
        road_type=m.group("road").upper(),
        distance_km=distance,
        session_id=folder_name,
        label=LABEL_MAP[behavior],
    )


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_txt(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a space-separated txt file; missing columns are filled with NaN."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        comment="#",
        engine="python",
        on_bad_lines="skip",
    )
    n_actual = df.shape[1]
    n_expected = len(columns)

    if n_actual <= n_expected:
        df.columns = columns[:n_actual]
        for missing_col in columns[n_actual:]:
            df[missing_col] = np.nan
    else:
        df = df.iloc[:, :n_expected]
        df.columns = columns

    return df


def _read_txt_from_zip(
    zf: zipfile.ZipFile,
    zip_path: str,
    columns: list[str],
) -> pd.DataFrame:
    """Read a txt file from inside a zip archive."""
    with zf.open(zip_path) as fh:
        df = pd.read_csv(
            fh,
            sep=r"\s+",
            header=None,
            comment="#",
            engine="python",
            on_bad_lines="skip",
        )
    n_actual = df.shape[1]
    n_expected = len(columns)

    if n_actual <= n_expected:
        df.columns = columns[:n_actual]
        for missing_col in columns[n_actual:]:
            df[missing_col] = np.nan
    else:
        df = df.iloc[:, :n_expected]
        df.columns = columns

    return df


# ---------------------------------------------------------------------------
# Merge accelerometer + GPS
# ---------------------------------------------------------------------------

def merge_accel_gps(accel: pd.DataFrame, gps: pd.DataFrame) -> pd.DataFrame:
    """
    Merge accelerometer (~10 Hz) and GPS (~1 Hz) data by timestamp.
    GPS columns are aligned to accelerometer time via nearest-neighbour merge.
    """
    if accel.empty or gps.empty:
        return accel.copy()

    gps_sorted = gps.sort_values("timestamp")
    accel_sorted = accel.sort_values("timestamp").copy()

    # Align GPS columns to accelerometer timestamps
    gps_cols_to_merge = [c for c in GPS_COLS[1:] if c in gps.columns]
    merged = pd.merge_asof(
        accel_sorted,
        gps_sorted[["timestamp"] + gps_cols_to_merge],
        on="timestamp",
        direction="nearest",
        suffixes=("", "_gps"),
    )
    return merged


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class UAHDriveSetLoader:
    """
    Loads the UAH-DRIVESET-v1 dataset.

    Supports reading from both an extracted directory and a zip archive.

    Parameters
    ----------
    data_path : Path
        - Directory: data/raw/UAH-DRIVESET-v1/
        - Zip      : data/raw/UAH-DRIVESET-v1.zip
    drivers : list[str] | None
        Drivers to load, e.g. ["D1", "D2"]. None loads all.
    behaviors : list[str] | None
        Behaviors to load: ["normal", "aggressive", "economic"].
    road_types : list[str] | None
        "MOTORWAY" and/or "SECONDARY".
    merge : bool
        If True, accelerometer and GPS data are merged.
    """

    def __init__(
        self,
        data_path: str | Path,
        drivers: Optional[list[str]] = None,
        behaviors: Optional[list[str]] = None,
        road_types: Optional[list[str]] = None,
        merge: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.drivers = [d.upper() for d in drivers] if drivers else None
        self.behaviors = [b.lower() for b in behaviors] if behaviors else None
        self.road_types = [r.upper() for r in road_types] if road_types else None
        self.merge = merge

        self._is_zip = self.data_path.suffix == ".zip"
        self._trips: list[TripData] = []
        self._loaded = False

    # ------------------------------------------------------------------

    def load(self, verbose: bool = True) -> "UAHDriveSetLoader":
        """Load all matching trips and return self (for chaining)."""
        if self._is_zip:
            self._load_from_zip(verbose)
        else:
            self._load_from_dir(verbose)
        self._loaded = True
        return self

    # ------------------------------------------------------------------

    def _should_include(self, meta: TripMetadata) -> bool:
        if self.drivers and meta.driver_id not in self.drivers:
            return False
        if self.behaviors and meta.behavior not in self.behaviors:
            return False
        if self.road_types and meta.road_type not in self.road_types:
            return False
        return True

    # ------------------------------------------------------------------

    def _load_from_dir(self, verbose: bool) -> None:
        root = self.data_path
        # Enter UAH-DRIVESET-v1/ sub-directory if present
        candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
        driver_dirs: list[Path] = []
        for cand in candidates:
            for sub in cand.iterdir() if cand.is_dir() else []:
                if re.match(r"D\d+$", sub.name, re.IGNORECASE) and sub.is_dir():
                    driver_dirs.append(sub)

        if not driver_dirs:
            raise FileNotFoundError(f"No driver folders found under: {root}")

        for driver_dir in sorted(driver_dirs):
            for session_dir in sorted(driver_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                meta = parse_session_name(session_dir.name)
                if meta is None or not self._should_include(meta):
                    continue

                accel_path = session_dir / "RAW_ACCELEROMETERS.txt"
                gps_path = session_dir / "RAW_GPS.txt"

                if not accel_path.exists() or not gps_path.exists():
                    if verbose:
                        print(f"  [SKIP] Missing file: {session_dir.name}")
                    continue

                accel = _read_txt(accel_path, ACCEL_COLS)
                gps = _read_txt(gps_path, GPS_COLS)

                trip = TripData(
                    meta=meta,
                    accel=accel,
                    gps=gps,
                    merged=merge_accel_gps(accel, gps) if self.merge else pd.DataFrame(),
                )
                self._trips.append(trip)

                if verbose:
                    print(
                        f"  [LOADED] {meta.driver_id} | {meta.behavior:10s} | "
                        f"{meta.road_type:10s} | {len(accel):5d} rows"
                    )

    # ------------------------------------------------------------------

    def _load_from_zip(self, verbose: bool) -> None:
        with zipfile.ZipFile(self.data_path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            session_map: dict[str, dict[str, str]] = {}

            for name in names:
                parts = name.split("/")
                if len(parts) < 3:
                    continue
                driver_part = parts[1]
                if not re.match(r"D\d+$", driver_part, re.IGNORECASE):
                    continue
                session_part = parts[2]
                filename = parts[-1]
                key = f"{parts[0]}/{driver_part}/{session_part}"
                session_map.setdefault(key, {})[filename] = name

            for session_key in sorted(session_map):
                session_name = session_key.split("/")[-1]
                meta = parse_session_name(session_name)
                if meta is None or not self._should_include(meta):
                    continue

                files = session_map[session_key]
                accel_zip = files.get("RAW_ACCELEROMETERS.txt")
                gps_zip = files.get("RAW_GPS.txt")

                if not accel_zip or not gps_zip:
                    if verbose:
                        print(f"  [SKIP] Missing file: {session_name}")
                    continue

                accel = _read_txt_from_zip(zf, accel_zip, ACCEL_COLS)
                gps = _read_txt_from_zip(zf, gps_zip, GPS_COLS)

                trip = TripData(
                    meta=meta,
                    accel=accel,
                    gps=gps,
                    merged=merge_accel_gps(accel, gps) if self.merge else pd.DataFrame(),
                )
                self._trips.append(trip)

                if verbose:
                    print(
                        f"  [LOADED] {meta.driver_id} | {meta.behavior:10s} | "
                        f"{meta.road_type:10s} | {len(accel):5d} rows"
                    )

    # ------------------------------------------------------------------
    # Access interface
    # ------------------------------------------------------------------

    @property
    def trips(self) -> list[TripData]:
        if not self._loaded:
            raise RuntimeError("Call .load() first.")
        return self._trips

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with metadata for every trip."""
        rows = []
        for t in self.trips:
            m = t.meta
            rows.append({
                "session_id":   m.session_id,
                "driver":       m.driver_id,
                "behavior":     m.behavior,
                "label":        m.label,
                "road_type":    m.road_type,
                "distance_km":  m.distance_km,
                "accel_rows":   len(t.accel),
                "gps_rows":     len(t.gps),
                "duration_s":   t.accel["timestamp"].max() - t.accel["timestamp"].min()
                                if not t.accel.empty else 0,
            })
        return pd.DataFrame(rows)

    def get_by_driver(self, driver_id: str) -> list[TripData]:
        return [t for t in self.trips if t.meta.driver_id == driver_id.upper()]

    def get_by_behavior(self, behavior: str) -> list[TripData]:
        return [t for t in self.trips if t.meta.behavior == behavior.lower()]

    def __len__(self) -> int:
        return len(self._trips)

    def __repr__(self) -> str:
        status = f"{len(self._trips)} trips loaded" if self._loaded else "not yet loaded"
        return f"UAHDriveSetLoader({self.data_path.name}, {status})"
