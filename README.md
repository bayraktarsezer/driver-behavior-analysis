# Driver Behavior Analysis

**Classifying driving style** (normal / aggressive / economic) from smartphone accelerometer and GPS data using classical ML — with a rigorous **Leave-One-Driver-Out (LODO)** cross-validation protocol.

---

## Results

### LODO Cross-Validation (6 drivers, no data leakage)

| Model | Accuracy | Macro F1 | F1 Normal | F1 Aggressive | F1 Economic |
|---|---|---|---|---|---|
| **Random Forest** | **0.619 ± 0.042** | **0.604 ± 0.049** | 0.660 | 0.650 | 0.502 |
| XGBoost | 0.608 ± 0.036 | 0.602 ± 0.042 | 0.622 | 0.648 | 0.537 |
| LightGBM | 0.602 ± 0.041 | 0.598 ± 0.046 | 0.613 | 0.643 | 0.537 |

> **LSTM / CNN-LSTM** is implemented in `notebooks/04_lstm.ipynb` but not yet benchmarked — results will be added once training is complete.

> LODO CV: each fold holds out one driver completely — the model never sees any windows from that driver during training. This is the hardest and most realistic evaluation for driver behaviour classification.

### Model Comparison

![Model Comparison](results/figures/model_comparison.png)

### Confusion Matrix (Random Forest — best model)

![Confusion Matrix](results/figures/cm_randomforest.png)

### Top Feature Importances

![RF Feature Importance](results/figures/feature_importance_randomforest.png)

### SHAP Explainability — Aggressive Driving

![SHAP Beeswarm Aggressive](results/figures/shap_beeswarm_aggressive.png)

### SHAP Feature Impact by Class

![SHAP Bar per Class](results/figures/shap_bar_per_class.png)

---

## Dataset

**UAH-DriveSet v1** — publicly available dataset from the University of Alcalá de Henares.

| Property | Value |
|---|---|
| Drivers | 6 |
| Trips | 40 |
| Sensor | Smartphone (accelerometer ~10 Hz, GPS ~1 Hz) |
| Road types | Motorway, Secondary |
| Classes | Normal (0), Aggressive (1), Economic (2) |
| Windows | 10,434 (5 s window, 2 s overlap) |

> **Class label note:** The UAH-DriveSet uses the label `DROWSY`; this project maps it to *economic* driving to reflect its signal characteristics (smooth, low-intensity behaviour) rather than a literal drowsiness interpretation.
| Features | 145 (statistical + FFT + physics-derived) |

---

## Feature Engineering

From each 5-second sliding window, 145 features are extracted across 3 groups:

**Statistical (per signal channel)**
`mean`, `std`, `min`, `max`, `range`, `rms`, `skewness`, `kurtosis`, `iqr`, `p25`, `p75`

**Frequency domain (FFT)**
Top-5 FFT coefficients, dominant frequency, spectral entropy, total power

**Physics-derived**
Jerk (rate of acceleration change), acceleration magnitude, lateral G-force, braking index, GPS speed statistics

---

## Tech Stack

- **Python 3.13** · NumPy · Pandas · scikit-learn
- **XGBoost 3.2** · **LightGBM 4.6**
- **PyTorch** (LSTM / CNN-LSTM)
- **Matplotlib** · Seaborn · SHAP

---

## Project Structure

```
driver-behavior-analysis/
├── data/
│   ├── raw/UAH-DRIVESET-v1/      # original dataset (D1–D6)
│   └── processed/
│       └── features_5s_2s.parquet  # extracted feature table
├── src/
│   ├── data_loader.py    # trip loading + accel/GPS merge
│   ├── preprocessing.py  # missing value imputation, sliding window
│   ├── features.py       # statistical, FFT, physics feature extraction
│   ├── models.py         # RF, XGBoost, LightGBM, LSTM, CNN-LSTM
│   ├── evaluate.py       # metrics, confusion matrix, visualisation
│   └── experiment.py     # LODO CV pipeline (no data leakage)
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory data analysis
│   ├── 02_preprocessing.ipynb  # feature extraction walkthrough
│   ├── 03_classical_ml.ipynb   # RF / XGBoost / LightGBM LODO
│   ├── 04_lstm.ipynb           # LSTM / CNN-LSTM training
│   └── 05_shap.ipynb           # SHAP explainability
├── results/
│   ├── figures/   # all plots (model comparison, CM, feature importance)
│   ├── metrics/   # metrics.json (all model results)
│   └── models/    # saved joblib model files per fold
├── run_experiment.py   # one-shot script: train all models + save results
└── requirements.txt
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing (generates data/processed/features_5s_2s.parquet)
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb

# 3. Train all classical ML models + generate figures
python run_experiment.py

# 4. (Optional) LSTM training
jupyter nbconvert --to notebook --execute notebooks/04_lstm.ipynb

# 5. (Optional) SHAP explainability
jupyter nbconvert --to notebook --execute notebooks/05_shap.ipynb
```

Or open any notebook interactively in JupyterLab / VS Code.

---

## Key Design Decisions

**Leave-One-Driver-Out CV** — Standard k-fold would leak driver identity into the test set (different windows of the same driver in train and test). LODO ensures the model is evaluated on a completely unseen driver, reflecting real-world deployment.

**Per-fold StandardScaler** — The scaler is fit on the training fold only and applied to the test fold. Fitting on the full dataset before splitting is a common source of data leakage.

**Window-level classification** — Each 5-second window is classified independently. Trip-level prediction can be obtained by majority vote or probability averaging across all windows of a trip.
