"""
Run LODO experiment for all classical ML models and generate all result figures.
Run from project root: python run_experiment.py
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pathlib import Path

from models import build_random_forest, build_xgboost, build_lightgbm, load_model
from features import get_feature_columns
from evaluate import plot_feature_importance, plot_class_distribution, plot_confusion_matrix
from experiment import compare_models

PROC_DIR    = Path('data/processed')
RESULTS_DIR = Path('results')
FIG_DIR     = RESULTS_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'metrics').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'models').mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading feature table...")
df = pd.read_parquet(PROC_DIR / 'features_5s_2s.parquet')
feat_cols = get_feature_columns(df)
df_clean  = df.dropna(subset=feat_cols).reset_index(drop=True)
print(f"  {len(df_clean)} windows × {len(feat_cols)} features")
print(f"  Drivers : {sorted(df_clean['driver'].unique())}")
print(f"  Classes : {dict(pd.Series(df_clean['label']).value_counts().sort_index())}\n")

# ── Class distribution figure ──────────────────────────────────────────────
plot_class_distribution(
    df_clean['label'].values,
    title='Class Distribution (all windows)',
    save_path=FIG_DIR / 'class_distribution.png',
)
plt.close('all')

# ── LODO experiment ────────────────────────────────────────────────────────
model_builders = {
    'RandomForest': build_random_forest,
    'XGBoost'     : build_xgboost,
    'LightGBM'    : build_lightgbm,
}

all_results, comp_df = compare_models(
    df=df_clean,
    model_builders=model_builders,
    feature_cols=feat_cols,
    results_dir=RESULTS_DIR,
    verbose=True,
)

# ── Print comparison table ─────────────────────────────────────────────────
print("\n=== Final Results ===")
display_cols = ['Model', 'Acc mean', 'Acc std', 'F1 mean', 'F1 std']
per_class_cols = [c for c in comp_df.columns if c.startswith('F1_')]
print(comp_df[display_cols + per_class_cols].to_string(index=False, float_format='{:.3f}'.format))

# ── Feature importance for each tree model ─────────────────────────────────
models_dir = RESULTS_DIR / 'models'
for mname in ['RandomForest', 'XGBoost', 'LightGBM']:
    if mname not in all_results:
        continue
    res        = all_results[mname]
    last_fold  = res['n_folds'] - 1
    last_driver= res['fold_metrics'][last_fold]['test_driver']
    mpath      = models_dir / f'{mname}_fold{last_fold}_test{last_driver}.joblib'
    if not mpath.exists():
        print(f"  [SKIP] {mpath.name} not found")
        continue
    model = load_model(mpath)
    if not hasattr(model, 'feature_importances_'):
        continue
    plot_feature_importance(
        feat_cols,
        model.feature_importances_,
        top_n=25,
        title=f'{mname} — Top 25 Features (Last Fold)',
        save_path=FIG_DIR / f'feature_importance_{mname.lower()}.png',
    )
    plt.close('all')
    print(f"  Feature importance saved: {mname}")

print("\nDone. All figures in results/figures/, metrics in results/metrics/metrics.json")
