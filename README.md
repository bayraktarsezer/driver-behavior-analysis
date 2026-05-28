# Driver Behavior Analysis

**3-class driving style classification** (normal / aggressive / economic) from smartphone inertial and GPS signals, evaluated with a **Leave-One-Driver-Out (LODO)** cross-validation protocol designed to prevent subject-level data leakage.

---

## Results at a Glance

| Model | Accuracy | Macro F1 | F1 Normal | F1 Aggressive | F1 Economic |
|---|---|---|---|---|---|
| **Random Forest** | **0.619 ± 0.042** | **0.604 ± 0.049** | 0.660 | 0.650 | 0.502 |
| XGBoost | 0.608 ± 0.036 | 0.602 ± 0.042 | 0.622 | 0.648 | 0.537 |
| LightGBM | 0.602 ± 0.041 | 0.598 ± 0.046 | 0.613 | 0.643 | 0.537 |

> LSTM / CNN-LSTM architectures are implemented in `notebooks/04_lstm.ipynb` but not yet included in the final comparison — see [Future Work](#future-work).

Performance varies substantially across held-out drivers (RF accuracy range: **0.555 – 0.670**), which reflects the core difficulty of cross-subject generalization.

![Model Comparison](results/figures/model_comparison.png)

![Per-Driver Accuracy](results/figures/fold_performance.png)

---

## Why This Problem Is Hard

### Random splits are misleading

A naïve 80/20 random split of the 10,434 windows places different windows from the *same driver* in both training and test sets. The model then learns individual driving fingerprints rather than transferable behavioral patterns. Accuracies reported under random splits are systematically inflated and do not reflect real-world deployment performance.

### Subject-independent evaluation

LODO cross-validation holds out **one complete driver** (all trips, all windows) per fold. The model is trained on 5 drivers and evaluated on a 6th it has never observed — which mirrors the actual deployment scenario where a model must classify unknown individuals.

### Driver generalization gap

Individual driving styles vary considerably across subjects. Driver D3 and D5 are the hardest test cases (RF accuracy: 0.555 and 0.581), while D2 and D4 yield much higher results (0.668 and 0.670). This 12 percentage-point gap between easiest and hardest driver is the fundamental challenge in behavioral recognition — the model must learn what "aggressive" means in a subject-independent sense.

### Class imbalance and why accuracy is insufficient

| Class | Windows | Share |
|---|---|---|
| Normal | 4,418 | 42.3% |
| Economic | 3,348 | 32.1% |
| Aggressive | 2,668 | 25.6% |

A degenerate classifier that predicts "normal" for every window achieves **42.3% accuracy** — higher than a poor but non-trivial model — while being entirely uninformative. **Macro-averaged F1** is therefore used as the primary metric: it averages precision–recall across all three classes with equal weight regardless of class size, and directly penalises a model that ignores any single class.

### Sensor and data limitations

The dataset uses a consumer smartphone mounted in the vehicle. Sensor noise, mounting position variability, and GPS dropouts (urban canyons, tunnels) introduce signal artifacts. Kalman-filtered accelerometer channels (`acc_*_kf`) partially mitigate measurement noise, but inter-driver mounting differences remain an uncontrolled confound.

---

## Dataset

**UAH-DriveSet v1** — publicly available sensor dataset from the University of Alcalá de Henares, Spain.

| Property | Value |
|---|---|
| Drivers | 6 |
| Trips | 40 |
| Sensor | Smartphone (accelerometer ~10 Hz, GPS ~1 Hz) |
| Road types | Motorway, Secondary |
| Classes | Normal (0), Aggressive (1), Economic (2) |
| Windows | 10,434 (5 s window, 2 s overlap) |
| Features | 145 (statistical + FFT + physics-derived) |

> **Class label note:** The UAH-DriveSet uses the label `DROWSY`; this project maps it to *economic* driving to reflect the signal profile — smooth, low-intensity behaviour consistent with fuel-efficient driving rather than literal drowsiness.

Download the dataset and place it at `data/raw/UAH-DRIVESET-v1/` before running preprocessing.

---

## Feature Engineering

From each 5-second sliding window, 145 features are extracted across three groups:

**Statistical** (per signal channel) — `mean`, `std`, `min`, `max`, `range`, `rms`, `skewness`, `kurtosis`, `iqr`, `p25`, `p75`

**Frequency domain (FFT)** — top-5 coefficients, dominant frequency, spectral entropy, total power

**Physics-derived** — jerk (rate of acceleration change), acceleration magnitude, lateral G-force, braking index, GPS speed statistics

---

## Methodology

### LODO cross-validation pipeline

For each of 6 folds:
1. Assign all windows of one driver to the test set; remaining 5 drivers form the training set
2. Impute NaNs using **training-fold column means only** — no test data touches this step
3. Fit `StandardScaler` on the training fold only; apply the learned transform to the test fold
4. Train the classifier; evaluate on the completely unseen driver

Fitting the scaler on the full dataset before splitting is a common, subtle form of data leakage that inflates reported results. The per-fold fit ensures the reported numbers are genuinely pessimistic estimates of unseen-driver performance.

---

## Key Findings

### What actually distinguishes driving styles (SHAP analysis)

SHAP values computed on the held-out D6 fold (Random Forest) reveal the features that drive individual predictions:

| Rank | Feature | Type | Behavioral interpretation |
|---|---|---|---|
| 1 | `speed_max` | GPS | Economic drivers show the lowest peak speeds; aggressive drivers the highest |
| 2 | `speed_mean` | GPS | Average speed within the window is the single strongest class discriminator |
| 3–9 | `yaw_*` (7 features) | IMU | Yaw rate distribution captures steering aggression — sharp manoeuvres vs. smooth curves |
| 10–15 | `acc_z_kf_*` | IMU (KF) | Vertical axis variability reflects braking intensity and vehicle response to road surface |

**Class-level findings:**

- **Aggressive driving** is most strongly identified by high yaw rate variance — rapid steering corrections and sharp lateral movements. The cluster of 7 yaw features in the top 15 confirms that steering dynamics, not raw speed alone, define aggressive style.
- **Normal driving** occupies the middle ground: neither the smoothest nor the most erratic signal profile. GPS speed features contribute most, reflecting mid-range, consistent travel speeds.
- **Economic driving** is the hardest class to separate (lowest F1 across almost all folds). Its signal profile overlaps substantially with normal driving at moderate speeds. Driver D5 illustrates this most sharply: economic-class F1 of 0.345 indicates that for this particular driver, the model cannot reliably distinguish economic from normal behaviour within a 5-second window.

The dominance of yaw-axis features over lateral acceleration (`acc_y`) suggests that **steering dynamics** are more behaviorally informative than direct lateral G-force — yaw captures both the speed and sharpness of directional changes, while lateral G can saturate or be confounded by road curvature.

![SHAP Beeswarm — Aggressive](results/figures/shap_beeswarm_aggressive.png)

![SHAP Feature Impact by Class](results/figures/shap_bar_per_class.png)

---

## Additional Figures

### Confusion matrices

![RF Confusion Matrix](results/figures/cm_randomforest.png)

### Top feature importances

![RF Feature Importance](results/figures/feature_importance_randomforest.png)

---

## Tech Stack

- **Python 3.13** · NumPy · Pandas · SciPy · scikit-learn
- **XGBoost 3.2** · **LightGBM 4.6**
- **PyTorch** (LSTM / CNN-LSTM — future work)
- **Matplotlib** · Seaborn · SHAP

---

## Project Structure

```
driver-behavior-analysis/
├── data/
│   ├── raw/UAH-DRIVESET-v1/        # original dataset (D1–D6, not tracked)
│   └── processed/
│       └── features_5s_2s.parquet  # extracted feature table (not tracked)
├── src/
│   ├── data_loader.py    # trip loading + accel/GPS merge
│   ├── preprocessing.py  # sliding window, imputation
│   ├── features.py       # statistical, FFT, physics feature extraction
│   ├── models.py         # RF, XGBoost, LightGBM, LSTM, CNN-LSTM builders
│   ├── evaluate.py       # metrics, visualisation
│   └── experiment.py     # LODO CV pipeline (no data leakage)
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory data analysis
│   ├── 02_preprocessing.ipynb  # feature extraction walkthrough
│   ├── 03_classical_ml.ipynb   # RF / XGBoost / LightGBM LODO
│   ├── 04_lstm.ipynb           # LSTM / CNN-LSTM (work in progress)
│   └── 05_shap.ipynb           # SHAP explainability
├── results/
│   ├── figures/   # all plots
│   ├── metrics/   # metrics.json
│   └── models/    # joblib model files per fold (not tracked)
├── run_experiment.py   # train all classical ML models + save results
├── run_shap.py         # generate all SHAP figures
├── restyle_figures.py  # regenerate all figures with consistent style
└── requirements.txt
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place UAH-DriveSet v1 at data/raw/UAH-DRIVESET-v1/

# 3. Run preprocessing  (generates data/processed/features_5s_2s.parquet)
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb

# 4. Train all classical ML models + generate figures
python run_experiment.py

# 5. SHAP explainability figures
python run_shap.py

# 6. (Optional) LSTM training
jupyter nbconvert --to notebook --execute notebooks/04_lstm.ipynb
```

Or open any notebook interactively in JupyterLab / VS Code.

---

## Limitations

- **Small N:** 6 drivers is a limited subject pool; cross-driver generalization claims should be interpreted cautiously.
- **Single dataset:** All recordings come from one study (UAH, Spain, 2015). Road conditions, cultural driving norms, and sensor mounting may limit external validity.
- **Window-level labels:** Labels are assigned per trip, not per window. Transitional behaviour (a momentary aggressive manoeuvre within an otherwise normal trip) is not captured.
- **No temporal context:** Classical ML treats each 5-second window independently, discarding sequential dependencies across consecutive windows.

---

## Future Work

- **LSTM / CNN-LSTM benchmark** — sequence models that process windows as temporal series are already implemented; benchmarking against the classical ML baseline is the immediate next step.
- **Trip-level aggregation** — majority voting or probability averaging across all windows of a trip, evaluated against trip-level ground truth.
- **Larger subject pool** — more drivers would strengthen cross-subject generalization claims.
- **Multimodal fusion** — combining accelerometer, gyroscope, and GPS with learned feature weighting.
