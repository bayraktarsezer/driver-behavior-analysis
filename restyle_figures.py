"""
Re-generate all result figures using the fivethirtyeight matplotlib style.
Also adds missing confusion matrices for XGBoost and LightGBM.
Run from project root:  python restyle_figures.py
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from features import get_feature_columns
from models   import load_model

plt.style.use('fivethirtyeight')

# ── paths ──────────────────────────────────────────────────────────────────
PROC_DIR     = Path('data/processed')
MODELS_DIR   = Path('results/models')
METRICS_PATH = Path('results/metrics/metrics.json')
FIG_DIR      = Path('results/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ['normal', 'aggressive', 'economic']
SHAP_LABELS = ['normal', 'aggressive', 'drowsy']

# ── load ──────────────────────────────────────────────────────────────────
print("Loading assets...")
with open(METRICS_PATH) as f:
    metrics = json.load(f)

df        = pd.read_parquet(PROC_DIR / 'features_5s_2s.parquet')
feat_cols = get_feature_columns(df)
df_clean  = df.dropna(subset=feat_cols).reset_index(drop=True)
COLORS    = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(f"  {len(df_clean):,} windows  |  {len(feat_cols)} features\n")

_count = 0

def _save(fig, name):
    global _count
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close('all')
    _count += 1
    print(f"  [{_count:02d}] {name}")

# ── 1. Class distribution ──────────────────────────────────────────────────
labels         = df_clean['label'].values
unique, counts = np.unique(labels, return_counts=True)
names          = [LABEL_NAMES[u] for u in unique]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(names, counts, color=COLORS[:len(unique)], edgecolor='white', linewidth=1)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f'{c:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Sample Count')
ax.set_title('Class Distribution  (All Windows)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
_save(fig, 'class_distribution.png')

# ── 2. Model comparison ────────────────────────────────────────────────────
model_names = list(metrics.keys())
acc_m = [metrics[m]['mean_accuracy'] for m in model_names]
acc_s = [metrics[m]['std_accuracy']  for m in model_names]
f1_m  = [metrics[m]['mean_f1']       for m in model_names]
f1_s  = [metrics[m]['std_f1']        for m in model_names]
x = np.arange(len(model_names))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
b1 = ax.bar(x - w / 2, acc_m, w, yerr=acc_s, label='Accuracy',
            color=COLORS[0], capsize=5, error_kw={'elinewidth': 1.5})
b2 = ax.bar(x + w / 2, f1_m,  w, yerr=f1_s,  label='Macro F1',
            color=COLORS[1], capsize=5, error_kw={'elinewidth': 1.5})
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — LODO CV (6 Folds)')
ax.legend(loc='upper right', fontsize=11)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.013,
            f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
_save(fig, 'model_comparison.png')

# ── 3–5. Confusion matrices (all 3 models, aggregated across folds) ─────────
print("Confusion matrices:")
for mname in model_names:
    folds  = metrics[mname]['fold_details']
    cm_agg = sum(np.array(fd['confusion_matrix']) for fd in folds)
    cm_n   = cm_agg.astype(float) / cm_agg.sum(axis=1, keepdims=True)
    cm_n   = np.nan_to_num(cm_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                ax=ax, linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': 13, 'fontweight': 'bold'})
    ax.grid(False)
    ax.set_xlabel('Predicted', fontweight='bold', labelpad=10)
    ax.set_ylabel('True', fontweight='bold', labelpad=10)
    ax.set_title(f'{mname} — Normalized Confusion Matrix\n(Aggregated, 6 LODO Folds)')
    plt.tight_layout()
    _save(fig, f'cm_{mname.lower()}.png')

# ── 6–8. Feature importances ─────────────────────────────────────────────────
print("Feature importances:")
for ci, mname in enumerate(model_names):
    folds     = metrics[mname]['fold_details']
    last_fold = len(folds) - 1
    test_drv  = folds[last_fold]['test_driver']
    mpath     = MODELS_DIR / f'{mname}_fold{last_fold}_test{test_drv}.joblib'
    if not mpath.exists():
        print(f"  [SKIP] model not found: {mpath.name}")
        continue
    model = load_model(mpath)
    if not hasattr(model, 'feature_importances_'):
        print(f"  [SKIP] no feature_importances_: {mname}")
        continue

    top_n       = 25
    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1][:top_n]
    top_names   = [feat_cols[j] for j in idx]
    top_vals    = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    ax.barh(top_names[::-1], top_vals[::-1], color=COLORS[ci % len(COLORS)])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'{mname} — Top {top_n} Feature Importances\n'
                 f'(Fold {last_fold}, Test Driver {test_drv})')
    plt.tight_layout()
    _save(fig, f'feature_importance_{mname.lower()}.png')

# ── 9–13. SHAP figures ────────────────────────────────────────────────────
print("SHAP figures (this may take ~1–2 minutes)...")
fold_idx    = 5
test_driver = 'D6'

X_train = df[df['driver'] != test_driver][feat_cols].values.astype(float)
X_test  = df[df['driver'] == test_driver][feat_cols].values.astype(float)
y_test  = df[df['driver'] == test_driver]['label'].values.astype(int)

scaler    = StandardScaler().fit(X_train)
X_test_sc = scaler.transform(X_test)

rf_path = MODELS_DIR / f'RandomForest_fold{fold_idx}_test{test_driver}.joblib'
model   = load_model(rf_path)
print(f"  Model: {rf_path.name}")

rng        = np.random.default_rng(42)
sample_idx = rng.choice(len(X_test_sc), size=min(500, len(X_test_sc)), replace=False)
X_sample   = X_test_sc[sample_idx]
y_sample   = y_test[sample_idx]
feat_df    = pd.DataFrame(X_sample, columns=feat_cols)

explainer = shap.TreeExplainer(model)
shap_raw  = explainer.shap_values(feat_df)
if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
    shap_values = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
else:
    shap_values = shap_raw

ev = explainer.expected_value
if not hasattr(ev, '__len__'):
    ev = [ev] * len(SHAP_LABELS)
print(f"  SHAP values per class: {shap_values[0].shape}")

# beeswarm
for cls_idx, cls_name in enumerate(SHAP_LABELS):
    shap.summary_plot(shap_values[cls_idx], feat_df,
                      max_display=20, show=False, plot_type='dot')
    plt.title(f'SHAP Beeswarm — {cls_name.capitalize()} Driving', fontsize=13)
    plt.tight_layout()
    _save(plt.gcf(), f'shap_beeswarm_{cls_name}.png')

# bar summary
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for cls_idx, (cls_name, ax) in enumerate(zip(SHAP_LABELS, axes)):
    mean_abs = np.abs(shap_values[cls_idx]).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:15]
    ax.barh([feat_cols[i] for i in top_idx][::-1],
            mean_abs[top_idx][::-1],
            color=COLORS[cls_idx % len(COLORS)])
    ax.set_title(f'{cls_name.capitalize()} — Top 15', fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean |SHAP value|')
plt.suptitle('Feature Impact by Driving Class (SHAP)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
_save(fig, 'shap_bar_per_class.png')

# waterfall
agg_mask = (y_sample == 1)
win_idx  = int(np.where(agg_mask)[0][0]) if agg_mask.sum() > 0 else 0
pred_cls = int(model.predict(X_sample[[win_idx]])[0])
exp = shap.Explanation(
    values        = shap_values[pred_cls][win_idx],
    base_values   = ev[pred_cls],
    data          = X_sample[win_idx],
    feature_names = feat_cols,
)
shap.plots.waterfall(exp, max_display=15, show=False)
plt.title(f'SHAP Waterfall — Predicted: {SHAP_LABELS[pred_cls].capitalize()}',
          fontsize=11)
plt.tight_layout()
_save(plt.gcf(), 'shap_waterfall_example.png')

print(f"\nDone — {_count} figures saved to {FIG_DIR}/")
