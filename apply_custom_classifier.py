"""
Apply the locally-trained classifier (E:/classifier_model/) to all units
across surveys and wireless datasets.

Writes:
  <sort_dir>/bank<bb>_ks4/analyzer_cache/extensions/auto_curation/labels_custom.csv
  <sort_dir>/auto_curation_labels_custom.csv

Columns: unit_id, custom_label, custom_p_neural, ks4_label
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

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_classify_units import ensure_extensions, _build_feature_df

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
MODEL_DIR = DATA_ROOT / 'classifier_model'


def load_model():
    pipe = joblib.load(MODEL_DIR / 'classifier.joblib')
    feat_order = json.loads((MODEL_DIR / 'feature_order.json').read_text())
    return pipe, feat_order


def classify_bank(cache_dir, output_path, pipe, feat_order):
    import spikeinterface.full as si
    if not (cache_dir / 'spikeinterface_info.json').exists():
        return None
    analyzer = si.load_sorting_analyzer(cache_dir)
    if len(analyzer.unit_ids) == 0:
        return None
    ensure_extensions(analyzer)
    X = _build_feature_df(analyzer, feature_order=feat_order)
    p_neural = pipe.predict_proba(X)[:, 1]
    pred = (p_neural >= 0.5).astype(int)

    df = pd.DataFrame({
        'unit_id': analyzer.unit_ids,
        'custom_label': np.where(pred == 1, 'neural', 'noise'),
        'custom_p_neural': p_neural,
    }).set_index('unit_id')

    ks_path = cache_dir.parent / 'sorter_output' / 'cluster_KSLabel.tsv'
    if ks_path.exists():
        ks = pd.read_csv(ks_path, sep='\t')
        ks_map = dict(zip(ks['cluster_id'], ks['KSLabel']))
        df['ks4_label'] = [ks_map.get(uid, 'unknown') for uid in df.index]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    n_neural = (df['custom_label'] == 'neural').sum()
    logger.info('  %d / %d neural (%.0f%%)', n_neural, len(df),
                100 * n_neural / max(1, len(df)))
    return df


def process_dataset(sort_dir, pipe, feat_order, banks=None):
    if banks is None:
        # Banks 0-6 only — upper banks 7-11 are mostly boundary noise and
        # are ignored in downstream analyses. Override with --banks 0..11
        # to include all.
        banks = list(range(7))
    results = []
    for b in banks:
        cache_dir = sort_dir / f'bank{b:02d}_ks4' / 'analyzer_cache'
        if not cache_dir.exists():
            continue
        out_path = (cache_dir / 'extensions' / 'auto_curation'
                     / 'labels_custom.csv')
        logger.info('%s bank%d', sort_dir.name, b)
        df = classify_bank(cache_dir, out_path, pipe, feat_order)
        if df is not None:
            df = df.copy()
            df['bank'] = b
            df['survey'] = sort_dir.name.replace('survey_', '')
            results.append(df)
    if results:
        combined = pd.concat(results)
        out = sort_dir / 'auto_curation_labels_custom.csv'
        combined.to_csv(out)
        logger.info('Saved %s (%d rows)', out, len(combined))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', nargs='*', default=['all'],
        help='survey keys or wireless dir names; "all" runs all surveys')
    parser.add_argument('--include-wireless', action='store_true',
        help='When targets="all", also process wireless datasets')
    parser.add_argument('--banks', type=int, nargs='+', default=None,
        help='Bank IDs to process (default: 0-6).')
    args = parser.parse_args()
    pipe, feat_order = load_model()

    targets = args.targets
    if targets == ['all']:
        targets = sorted(d.name.replace('survey_', '')
                          for d in SORT_ROOT.glob('survey_*'))
        if args.include_wireless:
            for w in ['April21_wireless', 'March31_wireless', 'March23_wireless']:
                if (SORT_ROOT / w).exists():
                    targets.append(w)

    for t in targets:
        sort_dir = SORT_ROOT / f'survey_{t}'
        if not sort_dir.exists():
            sort_dir = SORT_ROOT / t
        if not sort_dir.exists():
            logger.warning('Not found: %s', t)
            continue
        process_dataset(sort_dir, pipe, feat_order, banks=args.banks)


if __name__ == '__main__':
    main()
