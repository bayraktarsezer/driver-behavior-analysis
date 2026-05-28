"""
Generate SHAP figures for the best model (RandomForest, fold 5, test D6).
Run from project root: python run_shap.py
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import shap
from pathlib import Path

from models import load_model
from features import get_feature_columns
from sklearn.preprocessing import StandardScaler

PROC_DIR    = Path('data/processed')
MODELS_DIR  = Path('results/models')
FIG_DIR     = Path('results/figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)
LABEL_NAMES = ['normal', 'aggressive', 'drowsy']

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df        = pd.read_parquet(PROC_DIR / 'features_5s_2s.parquet')
feat_cols = get_feature_columns(df)

fold_idx    = 5
test_driver = 'D6'

train_idx  = df[df['driver'] != test_driver].index
test_idx   = df[df['driver'] == test_driver].index
X_train    = df.loc[train_idx, feat_cols].values.astype(float)
X_test     = df.loc[test_idx,  feat_cols].values.astype(float)
y_test     = df.loc[test_idx,  'label'].values.astype(int)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model_path = MODELS_DIR / f'RandomForest_fold{fold_idx}_test{test_driver}.joblib'
model      = load_model(model_path)
print(f"Model: {model_path.name}")

# ── SHAP values ────────────────────────────────────────────────────────────
print("Computing SHAP values (this may take ~1 minute)...")
explainer  = shap.TreeExplainer(model)

rng        = np.random.default_rng(42)
sample_idx = rng.choice(len(X_test_sc), size=min(500, len(X_test_sc)), replace=False)
X_sample   = X_test_sc[sample_idx]
y_sample   = y_test[sample_idx]
feat_df    = pd.DataFrame(X_sample, columns=feat_cols)

shap_values_raw = explainer.shap_values(feat_df)

# SHAP >=0.41 may return (n_samples, n_features, n_classes) instead of a list
if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
else:
    shap_values = shap_values_raw

# expected_value: array of shape (n_classes,) or list
ev = explainer.expected_value
if not hasattr(ev, '__len__'):
    ev = [ev] * len(LABEL_NAMES)

print(f"SHAP values computed. Shape per class: {shap_values[0].shape}")

# ── Beeswarm plots ─────────────────────────────────────────────────────────
print("Generating beeswarm plots...")
for cls_idx, cls_name in enumerate(LABEL_NAMES):
    shap.summary_plot(
        shap_values[cls_idx], feat_df,
        max_display=20,
        show=False,
        plot_type='dot',
    )
    plt.title(f'SHAP Beeswarm — {cls_name.capitalize()} driving', fontsize=13)
    plt.tight_layout()
    save_path = FIG_DIR / f'shap_beeswarm_{cls_name}.png'
    plt.gcf().savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {save_path}")

# ── Bar summary ────────────────────────────────────────────────────────────
print("Generating bar summary...")
colors = {'normal': '#4C72B0', 'aggressive': '#DD8452', 'drowsy': '#55A868'}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for cls_idx, (cls_name, ax) in enumerate(zip(LABEL_NAMES, axes)):
    mean_abs = np.abs(shap_values[cls_idx]).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:15]
    ax.barh(
        [feat_cols[i] for i in top_idx][::-1],
        mean_abs[top_idx][::-1],
        color=colors[cls_name]
    )
    ax.set_title(f'{cls_name.capitalize()} — Top 15 Features', fontsize=11)
    ax.set_xlabel('Mean |SHAP value|')

plt.suptitle('Feature Impact by Driving Class', fontsize=13, y=1.02)
plt.tight_layout()
bar_path = FIG_DIR / 'shap_bar_per_class.png'
fig.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {bar_path}")

# ── Waterfall (single window) ──────────────────────────────────────────────
print("Generating waterfall plot...")
aggressive_mask = (y_sample == 1)
window_idx      = np.where(aggressive_mask)[0][0] if aggressive_mask.sum() > 0 else 0

pred_class = model.predict(X_sample[[window_idx]])[0]
print(f"  True: {LABEL_NAMES[y_sample[window_idx]]}  |  Predicted: {LABEL_NAMES[pred_class]}")

exp = shap.Explanation(
    values       = shap_values[pred_class][window_idx],
    base_values  = ev[pred_class],
    data         = X_sample[window_idx],
    feature_names= feat_cols,
)
shap.plots.waterfall(exp, max_display=15, show=False)
plt.title(f'Waterfall — predicted: {LABEL_NAMES[pred_class]}', fontsize=11)
plt.tight_layout()
wf_path = FIG_DIR / 'shap_waterfall_example.png'
plt.gcf().savefig(wf_path, dpi=150, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {wf_path}")

print("\nAll SHAP figures saved to results/figures/")
