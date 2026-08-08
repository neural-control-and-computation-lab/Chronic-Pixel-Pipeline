"""
Automated unit classification using SpikeInterface's pretrained models.

For each bank of a survey:
  1. Load existing analyzer cache
  2. Compute missing extensions (principal_components, expanded quality metrics)
  3. Run noise_neural classifier — predicts noise vs neural (real)
  4. Run sua_mua classifier — predicts single-unit vs multi-unit
  5. Save labels.csv with predictions and probabilities

Models from HuggingFace:
  SpikeInterface/UnitRefine_noise_neural_classifier
  SpikeInterface/UnitRefine_sua_mua_classifier

Usage:
  python auto_classify_units.py 20260421
  python auto_classify_units.py 20260421 --bank 0 1 2
  python auto_classify_units.py all
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import argparse
import logging
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress sklearn version mismatch warnings
import sklearn
sklearn.set_config(print_changed_only=True)

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'

# All features the pretrained UnitRefine models expect (37 metrics)
REQUIRED_QM = [
    'num_spikes', 'firing_rate', 'presence_ratio', 'snr',
    'isi_violations_ratio', 'isi_violations_count',
    'rp_contamination', 'rp_violations',
    'sliding_rp_violation',
    'amplitude_cutoff', 'amplitude_median',
    'amplitude_cv_median', 'amplitude_cv_range',
    'drift_ptp', 'drift_std', 'drift_mad',
    'firing_range',
    'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
    # PCA-based
    'd_prime', 'isolation_distance', 'l_ratio', 'silhouette',
    'nn_hit_rate', 'nn_miss_rate',
]

REQUIRED_TM = [
    'peak_to_valley', 'peak_trough_ratio', 'half_width',
    'repolarization_slope', 'recovery_slope',
    'num_positive_peaks', 'num_negative_peaks',
    'velocity_above', 'velocity_below',
    'exp_decay', 'spread',
]


def ensure_extensions(analyzer):
    """Compute any missing extensions needed for the classifier."""
    import spikeinterface.full as si

    qm_ext = analyzer.get_extension('quality_metrics')
    qm_df = qm_ext.get_data() if qm_ext is not None else pd.DataFrame()

    missing_qm = [m for m in REQUIRED_QM if m not in qm_df.columns]
    if missing_qm:
        # Determine which metric_names to compute (some metrics produce multiple cols)
        metric_to_compute = set()
        col_to_metric = {
            'sliding_rp_violation': 'sliding_rp_violation',
            'amplitude_cv_median': 'amplitude_cv',
            'amplitude_cv_range': 'amplitude_cv',
            'sync_spike_2': 'synchrony',
            'sync_spike_4': 'synchrony',
            'sync_spike_8': 'synchrony',
            'd_prime': 'd_prime',
            'isolation_distance': 'isolation_distance',
            'l_ratio': 'l_ratio',
            'silhouette': 'silhouette',
            'nn_hit_rate': 'nearest_neighbor',
            'nn_miss_rate': 'nearest_neighbor',
        }
        for col in missing_qm:
            if col in col_to_metric:
                metric_to_compute.add(col_to_metric[col])
            else:
                logger.warning("  Don't know how to compute missing metric: %s", col)

        # Need principal_components for PCA-based metrics
        pca_metrics = {'d_prime', 'isolation_distance', 'l_ratio',
                       'silhouette', 'nearest_neighbor'}
        if metric_to_compute & pca_metrics:
            if 'principal_components' not in analyzer.get_loaded_extension_names():
                logger.info("  Computing principal_components...")
                t0 = time.time()
                analyzer.compute('principal_components',
                                  n_components=3, mode='by_channel_local')
                logger.info("    PCA done in %.1fs", time.time() - t0)

        # Compute the missing quality metrics
        # Get existing metric list and merge
        existing_metrics = set()
        col_metric_map = {
            'snr': 'snr', 'firing_rate': 'firing_rate',
            'num_spikes': 'num_spikes', 'presence_ratio': 'presence_ratio',
            'isi_violations_ratio': 'isi_violation', 'isi_violations_count': 'isi_violation',
            'rp_violations': 'rp_violation', 'rp_contamination': 'rp_violation',
            'amplitude_cutoff': 'amplitude_cutoff',
            'amplitude_median': 'amplitude_median',
            'drift_ptp': 'drift', 'drift_std': 'drift', 'drift_mad': 'drift',
            'firing_range': 'firing_range',
        }
        for col in qm_df.columns:
            if col in col_metric_map:
                existing_metrics.add(col_metric_map[col])
        all_metrics = list(existing_metrics | metric_to_compute)
        logger.info("  Computing quality metrics: %s", sorted(all_metrics))
        t0 = time.time()
        analyzer.compute('quality_metrics', metric_names=all_metrics)
        logger.info("    Quality metrics done in %.1fs", time.time() - t0)


def _patch_sklearn_compat(model):
    """Patch sklearn version mismatch attributes (1.4.2 -> 1.8.0)."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    if hasattr(model, 'steps'):
        for name, step in model.steps:
            if isinstance(step, SimpleImputer):
                if not hasattr(step, '_fill_dtype'):
                    step._fill_dtype = np.float64
                if not hasattr(step, '_fit_dtype'):
                    step._fit_dtype = np.float64
    return model


def _load_unitrefine_model(repo_id):
    """Download and load a UnitRefine model from HuggingFace, with sklearn patches.

    Returns (model, label_conv, feature_order).
    """
    import json
    from huggingface_hub import hf_hub_download
    import skops.io as sio

    model_path = hf_hub_download(repo_id=repo_id, filename='best_model.skops')
    info_path = hf_hub_download(repo_id=repo_id, filename='model_info.json')
    with open(info_path) as f:
        info = json.load(f)

    trusted = sio.get_untrusted_types(file=model_path)
    model = sio.load(model_path, trusted=trusted)
    model = _patch_sklearn_compat(model)

    label_conv = info.get('label_conversion', {})
    # Handle both formats: {'0': 'sua'} (int->str) and {'sua': 1} (str->int)
    converted = {}
    for k, v in label_conv.items():
        try:
            converted[int(k)] = v
        except (ValueError, TypeError):
            converted[int(v)] = k  # reversed format
    label_conv = converted

    # Get exact feature order from first step
    feature_order = None
    if hasattr(model, 'steps'):
        for name, step in model.steps:
            if hasattr(step, 'feature_names_in_'):
                feature_order = list(step.feature_names_in_)
                break

    return model, label_conv, feature_order


def _build_feature_df(analyzer, feature_order=None):
    """Build the feature DataFrame the UnitRefine model expects.

    If feature_order is provided, columns are returned in that exact order.
    """
    qm = analyzer.get_extension('quality_metrics').get_data()
    tm = analyzer.get_extension('template_metrics').get_data()
    df = pd.concat([qm, tm], axis=1)
    # Ensure all expected columns are present (NaN if missing)
    cols = feature_order if feature_order is not None else (REQUIRED_QM + REQUIRED_TM)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols]


def classify_bank(cache_dir, output_path):
    """Run both classifiers on one bank's analyzer cache."""
    import spikeinterface.full as si

    if not (cache_dir / 'spikeinterface_info.json').exists():
        logger.warning("  No analyzer cache, skipping")
        return None

    # Load
    t0 = time.time()
    analyzer = si.load_sorting_analyzer(cache_dir)
    n_units = len(analyzer.unit_ids)
    logger.info("  Loaded analyzer: %d units (in %.1fs)",
                 n_units, time.time() - t0)

    if n_units == 0:
        return None

    # Compute missing extensions
    ensure_extensions(analyzer)

    df = pd.DataFrame(index=analyzer.unit_ids)
    df.index.name = 'unit_id'

    logger.info("  Running noise/neural classifier...")
    t0 = time.time()
    try:
        model, label_conv, feat_order = _load_unitrefine_model(
            'SpikeInterface/UnitRefine_noise_neural_classifier')
        X = _build_feature_df(analyzer, feature_order=feat_order)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        df['noise_neural_label'] = [label_conv[p] for p in preds]
        df['noise_neural_prob'] = probs.max(axis=1)
        logger.info("    Done in %.1fs (features: %d)",
                     time.time() - t0, X.shape[1])
    except Exception as e:
        logger.error("  noise/neural classifier failed: %s", e)
        import traceback; traceback.print_exc()

    logger.info("  Running sua/mua classifier...")
    t0 = time.time()
    try:
        model, label_conv, feat_order = _load_unitrefine_model(
            'SpikeInterface/UnitRefine_sua_mua_classifier')
        X = _build_feature_df(analyzer, feature_order=feat_order)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        df['sua_mua_label'] = [label_conv[p] for p in preds]
        df['sua_mua_prob'] = probs.max(axis=1)
        logger.info("    Done in %.1fs", time.time() - t0)
    except Exception as e:
        logger.error("  sua/mua classifier failed: %s", e)
        import traceback; traceback.print_exc()

    # Add KS4 label for comparison
    ks_label_path = cache_dir.parent / 'sorter_output' / 'cluster_KSLabel.tsv'
    if ks_label_path.exists():
        ks_df = pd.read_csv(ks_label_path, sep='\t')
        ks_map = dict(zip(ks_df['cluster_id'], ks_df['KSLabel']))
        df['ks4_label'] = [ks_map.get(uid, 'unknown') for uid in df.index]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info("  Saved: %s", output_path)

    # Summary
    if 'noise_neural_label' in df.columns:
        n_neural = (df['noise_neural_label'] == 'neural').sum()
        logger.info("    noise/neural: %d neural, %d noise",
                     n_neural, n_units - n_neural)
    if 'sua_mua_label' in df.columns:
        n_sua = (df['sua_mua_label'] == 'sua').sum()
        logger.info("    sua/mua: %d sua, %d mua", n_sua, n_units - n_sua)
    if 'ks4_label' in df.columns:
        n_ks4_good = (df['ks4_label'] == 'good').sum()
        logger.info("    KS4: %d good, %d mua", n_ks4_good, n_units - n_ks4_good)

    return df


def process_survey(survey_key, banks=None, force=False):
    """Process all banks for one survey or wireless dataset.

    Accepts either:
      - 'YYYYMMDD' (looks under SORT_ROOT/survey_YYYYMMDD)
      - direct directory name (e.g. 'April21_wireless') under SORT_ROOT
    """
    sort_dir = SORT_ROOT / f'survey_{survey_key}'
    if not sort_dir.exists():
        # Try as a direct directory name (wireless datasets)
        alt = SORT_ROOT / survey_key
        if alt.exists():
            sort_dir = alt
        else:
            logger.warning("Not found: %s or %s", sort_dir, alt)
            return

    if banks is None:
        banks = list(range(7))

    results = []
    for bank_id in banks:
        cache_dir = sort_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
        output_path = cache_dir / 'extensions' / 'auto_curation' / 'labels.csv'

        logger.info("=" * 60)
        logger.info("Survey %s, Bank %d", survey_key, bank_id)
        logger.info("=" * 60)

        if not force and output_path.exists():
            logger.info("  Already classified, skipping (use --force to redo)")
            df = pd.read_csv(output_path, index_col='unit_id')
        else:
            df = classify_bank(cache_dir, output_path)

        if df is not None:
            df = df.copy()
            df['bank'] = bank_id
            df['survey'] = survey_key
            results.append(df)

    if results:
        combined = pd.concat(results)
        out = sort_dir / 'auto_curation_labels.csv'
        combined.to_csv(out)
        logger.info("\nSaved combined: %s (%d rows)", out, len(combined))


def main():
    parser = argparse.ArgumentParser(
        description='Automated unit classification with SpikeInterface UnitRefine')
    parser.add_argument('survey', help='Survey date key (e.g. 20260421) or "all"')
    parser.add_argument('--bank', type=int, nargs='+', default=None)
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if labels.csv exists')
    args = parser.parse_args()

    if args.survey == 'all':
        # Find all surveys
        surveys = sorted(d.name.replace('survey_', '')
                          for d in SORT_ROOT.glob('survey_*'))
        logger.info("Processing %d surveys: %s", len(surveys), surveys)
        for s in surveys:
            process_survey(s, banks=args.bank, force=args.force)
    else:
        process_survey(args.survey, banks=args.bank, force=args.force)


if __name__ == '__main__':
    main()
