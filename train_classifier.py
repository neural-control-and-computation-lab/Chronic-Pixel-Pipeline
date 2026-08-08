"""
Train a custom binary noise/neural classifier on hand-labeled units.

Inputs:
  E:/labeling_pool/manifest.csv  — id, source, sort_dir, survey, bank, unit_id
  E:/labeling_pool/labels.csv    — id, label (neural|noise|skip), timestamp

Workflow:
  1. Load labels (drop 'skip')
  2. For each labeled unit, load analyzer_cache and pull the feature row
     matching what UnitRefine expects (REQUIRED_QM + REQUIRED_TM in
     auto_classify_units.py).
  3. Build X (n_units, n_features), y (n_units,) where y=1 means neural.
  4. GroupKFold CV by `source` so we don't leak across surveys / wireless.
  5. Train final RandomForestClassifier on all data, save model + feature
     order to E:/classifier_model/.

Usage:
  python train_classifier.py
  python train_classifier.py --min-labels 100
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reuse the feature spec from auto_classify_units
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_classify_units import (REQUIRED_QM, REQUIRED_TM, ensure_extensions,
                                  _build_feature_df)

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
POOL_DIR = DATA_ROOT / 'labeling_pool'
SORT_ROOT = DATA_ROOT / 'sorting_output'
MODEL_DIR = DATA_ROOT / 'classifier_model'


def load_training_data(min_labels=50):
    """Build (X, y, groups) from manifest + labels."""
    import spikeinterface.full as si

    manifest = pd.read_csv(POOL_DIR / 'manifest.csv')
    labels = pd.read_csv(POOL_DIR / 'labels.csv')
    if 'id' in labels.columns:
        labels['id'] = labels['id'].astype(int)

    # Restrict to neural/noise (drop skip)
    labels = labels[labels['label'].isin(['neural', 'noise'])].copy()
    if len(labels) < min_labels:
        raise RuntimeError(
            f'Only {len(labels)} labeled units (need >= {min_labels}). '
            f'Continue labeling in label_units_web.py.')

    df = manifest.merge(labels[['id', 'label']], on='id', how='inner')
    logger.info('Loaded %d labeled units (%d neural, %d noise)',
                len(df), (df['label'] == 'neural').sum(),
                (df['label'] == 'noise').sum())
    logger.info('Sources: %s',
                df.groupby('source').size().to_dict())

    # Group rows by sort_dir + bank so we load each analyzer once
    feats = []
    keep_idx = []
    for (source, bank), grp in df.groupby(['source', 'bank']):
        # Look up sort_dir
        if source.startswith('wl_'):
            wireless_name = source.replace('wl_', '') + '_wireless'
            sort_dir = SORT_ROOT / wireless_name
        else:
            sort_dir = SORT_ROOT / f"survey_{grp['survey'].iloc[0]}"

        cache_dir = sort_dir / f'bank{int(bank):02d}_ks4' / 'analyzer_cache'
        if not cache_dir.exists():
            logger.warning('Missing analyzer_cache: %s', cache_dir)
            continue
        try:
            analyzer = si.load_sorting_analyzer(cache_dir)
            for ext in analyzer.get_saved_extension_names():
                try:
                    analyzer.load_extension(ext)
                except Exception:
                    pass
        except Exception as e:
            logger.warning('Failed to load %s: %s', cache_dir, e)
            continue

        try:
            ensure_extensions(analyzer)
        except Exception as e:
            logger.warning('ensure_extensions failed for %s: %s', cache_dir, e)

        feat_df = _build_feature_df(analyzer)
        for _, r in grp.iterrows():
            uid = int(r['unit_id'])
            if uid not in feat_df.index:
                logger.warning('uid %d not in feat_df for %s bank%d',
                                uid, source, bank)
                continue
            feats.append(feat_df.loc[uid].to_dict())
            keep_idx.append(r.name)

    X = pd.DataFrame(feats).reset_index(drop=True)
    y = (df.loc[keep_idx, 'label'].values == 'neural').astype(int)
    groups = df.loc[keep_idx, 'source'].values

    logger.info('Feature matrix: %s, label balance: %d neural / %d noise',
                X.shape, y.sum(), len(y) - y.sum())
    return X, y, groups, df.loc[keep_idx].reset_index(drop=True)


def build_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, class_weight='balanced',
            random_state=42, n_jobs=-1)),
    ])


def cross_validate(X, y, groups):
    """GroupKFold CV by source (no leakage between surveys/wireless)."""
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        logger.warning('Only %d groups — skipping CV', n_groups)
        return None, None
    cv = GroupKFold(n_splits=n_splits)
    pipe = build_pipeline()
    y_pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=1)
    y_proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups,
                                 method='predict_proba', n_jobs=1)[:, 1]
    return y_pred, y_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-labels', type=int, default=50,
                         help='Minimum number of labels needed to train')
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info('Loading training data...')
    X, y, groups, meta = load_training_data(min_labels=args.min_labels)
    feature_order = list(X.columns)

    # Cross-validate
    logger.info('Cross-validating (GroupKFold by source)...')
    y_pred, y_proba = cross_validate(X, y, groups)
    if y_pred is not None:
        logger.info('\n%s', classification_report(
            y, y_pred, target_names=['noise', 'neural']))
        cm = confusion_matrix(y, y_pred)
        logger.info('Confusion matrix (rows=true, cols=pred):\n%s', cm)
        try:
            auc = roc_auc_score(y, y_proba)
            logger.info('ROC-AUC: %.3f', auc)
        except Exception:
            pass

        # Plot CV diagnostics
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                              fontsize=14, color='black')
        axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['noise', 'neural'])
        axes[0].set_yticklabels(['noise', 'neural'])
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
        axes[0].set_title('CV confusion (GroupKFold by source)')

        axes[1].hist(y_proba[y == 0], bins=30, alpha=0.6, color='red',
                     label='noise (true)')
        axes[1].hist(y_proba[y == 1], bins=30, alpha=0.6, color='green',
                     label='neural (true)')
        axes[1].set_xlabel('P(neural) [CV]')
        axes[1].set_ylabel('count'); axes[1].legend()
        axes[1].set_title('Out-of-fold probability distribution')
        fig.tight_layout()
        fig.savefig(MODEL_DIR / 'cv_diagnostics.png', dpi=120)
        logger.info('Saved %s', MODEL_DIR / 'cv_diagnostics.png')

        # Save CV predictions per labeled unit for inspection
        meta_out = meta.copy()
        meta_out['cv_pred'] = y_pred
        meta_out['cv_p_neural'] = y_proba
        meta_out.to_csv(MODEL_DIR / 'cv_predictions.csv', index=False)

    # Train final model on all data
    logger.info('Fitting final model on all %d labeled units...', len(y))
    pipe = build_pipeline()
    pipe.fit(X, y)

    joblib.dump(pipe, MODEL_DIR / 'classifier.joblib')
    with open(MODEL_DIR / 'feature_order.json', 'w') as f:
        json.dump(feature_order, f, indent=2)
    logger.info('Saved %s', MODEL_DIR / 'classifier.joblib')

    # Feature importance — SimpleImputer drops all-NaN columns, so the
    # surviving feature names live on imputer.feature_names_in_ minus any
    # dropped during fit. Use the imputer's get_feature_names_out as the
    # source of truth.
    rf = pipe.named_steps['rf']
    imputer = pipe.named_steps['imputer']
    try:
        kept = list(imputer.get_feature_names_out())
    except Exception:
        kept = feature_order
    if len(kept) != len(rf.feature_importances_):
        # Fallback: just label columns by index
        kept = [f'feat_{i}' for i in range(len(rf.feature_importances_))]
    imp = pd.DataFrame({'feature': kept,
                        'importance': rf.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    imp.to_csv(MODEL_DIR / 'feature_importance.csv', index=False)
    logger.info('Top 10 features:\n%s', imp.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
