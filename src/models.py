"""
Model definitions.

Classical ML models (sklearn-based) and deep learning models (PyTorch LSTM)
are defined here.

If PyTorch is not installed, classical ML models still work; attempting to
instantiate LSTM/CNN-LSTM classes raises ImportError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Classical ML models
# ---------------------------------------------------------------------------

def build_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )


def build_gradient_boosting(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


def build_svm(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    class_weight: str = "balanced",
    random_state: int = 42,
) -> SVC:
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weight,
        random_state=random_state,
        probability=True,
    )


def build_xgboost(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> "XGBClassifier":
    if not _XGB_AVAILABLE:
        raise ImportError(
            "XGBoost not found. Install with: pip install xgboost"
        )
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric="mlogloss",
        verbosity=0,
        use_label_encoder=False,
    )


def build_lightgbm(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = -1,
    num_leaves: int = 63,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> "LGBMClassifier":
    if not _LGB_AVAILABLE:
        raise ImportError(
            "LightGBM not found. Install with: pip install lightgbm"
        )
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        class_weight=class_weight,
        random_state=random_state,
        verbosity=-1,
    )


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> None:
    """Save an sklearn / XGBoost / LightGBM model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    """Load a previously saved model."""
    return joblib.load(Path(path))


# ---------------------------------------------------------------------------
# Helper: PyTorch availability check
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch not found. To use LSTM models:\n"
            "  pip install torch torchvision\n"
            "Classical ML models (RF / XGBoost / LightGBM) do not require torch."
        )


# ---------------------------------------------------------------------------
# LSTM model (PyTorch)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class DriverBehaviorLSTM(torch.nn.Module):
        """
        Two-layer LSTM classifier.

        Architecture:
            LSTM(hidden_size, num_layers=2, bidirectional=False)
            → LayerNorm → Dropout → Linear(hidden_size, n_classes)

        Input shape : (batch, timesteps, n_features)
        Output shape: (batch, n_classes)  — raw logits
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            n_classes: int = 3,
            dropout: float = 0.3,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()
            self.hidden_size   = hidden_size
            self.num_layers    = num_layers
            self.bidirectional = bidirectional
            self.n_directions  = 2 if bidirectional else 1

            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.norm    = torch.nn.LayerNorm(hidden_size * self.n_directions)
            self.dropout = torch.nn.Dropout(dropout)
            self.fc      = torch.nn.Linear(hidden_size * self.n_directions, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)      # (B, T, H*D)
            out    = out[:, -1, :]     # Last time step: (B, H*D)
            out    = self.norm(out)
            out    = self.dropout(out)
            return self.fc(out)

    class DriverBehaviorCNN_LSTM(torch.nn.Module):
        """
        CNN + LSTM hybrid model.

        CNN layers extract local temporal patterns;
        LSTM captures long-range dependencies.
        """

        def __init__(
            self,
            input_size: int,
            cnn_channels: int = 64,
            lstm_hidden: int = 128,
            n_classes: int = 3,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(cnn_channels),
                torch.nn.ReLU(),
                torch.nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(cnn_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            self.lstm = torch.nn.LSTM(
                input_size=cnn_channels,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(lstm_hidden, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x.permute(0, 2, 1)    # (B, T, F) → (B, F, T)
            x = self.cnn(x)            # (B, C, T)
            x = x.permute(0, 2, 1)    # (B, T, C)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

else:
    # Stub classes: importable without torch, but raise on instantiation.

    class DriverBehaviorLSTM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

    class DriverBehaviorCNN_LSTM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()
