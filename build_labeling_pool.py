"""
Build a labeling pool: sample units stratified by classifier probability,
copying their existing summary PNGs to a labeling folder with a manifest CSV.

Strategy:
  - Stratified sampling across probability bins (low/medium/high P(neural))
  - Mix of surveys (wired) and wireless data
  - Spread across banks
  - Total ~250 units

Output:
  E:/labeling_pool/<id>.png           — copies of unit summary figures
  E:/labeling_pool/manifest.csv       — id, source, survey, bank, unit_id, p_neural, ks4_label
  E:/labeling_pool/labels.csv         — created/updated by web app

Usage:
  python build_labeling_pool.py
  python build_labeling_pool.py --n 300 --reset
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
FIG_ROOT = DATA_ROOT / 'figures' / 'sorting'
POOL_DIR = DATA_ROOT / 'labeling_pool'

# Lazy import of plot_custom_unit_summary
_plot_fn = None
_si = None
_analyzer_cache = {}


def _get_plot_fn():
    global _plot_fn, _si
    if _plot_fn is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_single_neuron_plots import plot_custom_unit_summary
        import spikeinterface.full as si
        _plot_fn = plot_custom_unit_summary
        _si = si
    return _plot_fn, _si


def _get_analyzer(sort_dir, bank_id):
    key = (str(sort_dir), bank_id)
    if key in _analyzer_cache:
        return _analyzer_cache[key]
    _, si = _get_plot_fn()
    cache_dir = sort_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
    if not cache_dir.exists():
        _analyzer_cache[key] = None
        return None
    try:
        analyzer = si.load_sorting_analyzer(cache_dir)
        for ext_name in analyzer.get_saved_extension_names():
            try:
                analyzer.load_extension(ext_name)
            except Exception:
                pass
        _analyzer_cache[key] = analyzer
        return analyzer
    except Exception as e:
        print(f"  Failed to load analyzer for {cache_dir}: {e}")
        _analyzer_cache[key] = None
        return None


def generate_figure(row, dst_path):
    """Generate a unit summary figure on the fly."""
    plot_fn, _ = _get_plot_fn()
    sort_dir = Path(row['sort_dir'])
    bank = int(row['bank'])
    uid = int(row['unit_id'])
    analyzer = _get_analyzer(sort_dir, bank)
    if analyzer is None:
        return False
    if uid not in list(analyzer.sorting.unit_ids):
        return False
    try:
        fig = plt.figure(figsize=(16, 9))
        plot_fn(analyzer, uid, fig)
        p_neural = float(row['p_neural'])
        fig.suptitle(
            f"{row['source']}  bank{bank:02d}  uid={uid}  P(neural)={p_neural:.2f}  KS4={row['ks4_label']}",
            fontsize=12, fontweight='bold', y=1.02)
        fig.savefig(dst_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return True
    except Exception as e:
        plt.close('all')
        print(f"  Failed to plot {row['source']} bank{bank} u{uid}: {e}")
        return False


def list_data_sources(include_wireless=False):
    """Return list of (key, sort_dir, fig_dir) tuples."""
    sources = []
    for d in sorted(SORT_ROOT.glob('survey_*')):
        key = d.name.replace('survey_', 'svy_')
        sources.append((key, d, FIG_ROOT / d.name / 'units'))
    if include_wireless:
        for name in ['April21_wireless', 'March31_wireless', 'March23_wireless']:
            d = SORT_ROOT / name
            if d.exists():
                sources.append((f'wl_{name.replace("_wireless","")}', d, FIG_ROOT / name / 'units'))
    return sources


def collect_units(include_wireless=False):
    """Collect all units with classifier probabilities and figure paths."""
    rows = []
    for key, sort_dir, fig_dir in list_data_sources(include_wireless=include_wireless):
        labels_csv = sort_dir / 'auto_curation_labels.csv'
        if not labels_csv.exists():
            continue
        df = pd.read_csv(labels_csv)
        if 'noise_neural_label' not in df.columns:
            continue
        df['p_neural'] = df.apply(
            lambda r: r['noise_neural_prob'] if r['noise_neural_label'] == 'neural'
            else 1 - r['noise_neural_prob'], axis=1)
        df['source'] = key
        df['sort_dir'] = str(sort_dir)
        df['fig_dir'] = str(fig_dir)
        rows.append(df[['source', 'sort_dir', 'fig_dir', 'survey', 'bank',
                        'unit_id', 'noise_neural_label', 'noise_neural_prob',
                        'p_neural', 'ks4_label']])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


_GLOBAL_IDX_CACHE = {}


def _build_global_idx_map(sort_dir):
    """For a survey sort dir, build (bank, unit_id) -> global_unit_idx.

    Mirrors logic in run_single_neuron_plots.py: global_unit_idx is a running
    counter incremented for every uid_idx in sorting.unit_ids across banks.
    """
    key = str(sort_dir)
    if key in _GLOBAL_IDX_CACHE:
        return _GLOBAL_IDX_CACHE[key]
    mapping = {}
    g = 0
    for bank_id in range(12):
        ks_tsv = sort_dir / f'bank{bank_id:02d}_ks4' / 'sorter_output' / 'cluster_KSLabel.tsv'
        if not ks_tsv.exists():
            continue
        ks_df = pd.read_csv(ks_tsv, sep='\t')
        for uid in ks_df['cluster_id']:
            mapping[(bank_id, int(uid))] = g
            g += 1
    _GLOBAL_IDX_CACHE[key] = mapping
    return mapping


def find_figure_path(row):
    """Return path to existing PNG for a unit, or None."""
    fig_dir = Path(row['fig_dir'])
    if not fig_dir.exists():
        return None
    bank = int(row['bank'])
    uid = int(row['unit_id'])
    src = row['source']

    # Wireless: figure name has uXXX directly
    if src.startswith('wl_'):
        patterns = [
            f'*bank{bank:02d}_u{uid}.png',
            f'*bank{bank:02d}_u{uid:03d}.png',
            f'*bank{bank:02d}_u{uid:04d}.png',
        ]
        for p in patterns:
            matches = list(fig_dir.glob(p))
            if matches:
                return matches[0]
        return None

    # Survey: filename = unit_{global_idx:04d}_bank{bb:02d}.png
    sort_dir = Path(row['sort_dir'])
    gmap = _build_global_idx_map(sort_dir)
    g = gmap.get((bank, uid))
    if g is None:
        return None
    candidate = fig_dir / f'unit_{g:04d}_bank{bank:02d}.png'
    if candidate.exists():
        return candidate
    return None


def stratified_sample(df, n_per_bin):
    """Stratified sample by P(neural) bin and source type (survey/wireless).

    If only one source type is present, all bins go to that type.
    """
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]
    df['source_type'] = df['source'].apply(
        lambda s: 'wireless' if s.startswith('wl_') else 'survey')
    df['p_bin'] = pd.cut(df['p_neural'], bins=[b[0] for b in bins] + [1.01],
                         labels=[f'{a:.1f}-{b:.1f}' for a, b in bins])

    samples = []
    rng = np.random.default_rng(42)
    source_types = sorted(df['source_type'].unique())
    for st in source_types:
        for p_bin in df['p_bin'].cat.categories:
            sub = df[(df['source_type'] == st) & (df['p_bin'] == p_bin)]
            if len(sub) == 0:
                continue
            n = min(n_per_bin, len(sub))
            samples.append(sub.sample(n=n, random_state=rng.integers(1e9)))
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=250)
    parser.add_argument('--reset', action='store_true',
                        help='Clear pool dir before populating')
    parser.add_argument('--include-wireless', action='store_true',
                        help='Include wireless datasets (default: surveys only)')
    args = parser.parse_args()

    if args.reset and POOL_DIR.exists():
        shutil.rmtree(POOL_DIR)
    POOL_DIR.mkdir(parents=True, exist_ok=True)

    print('Collecting all units with classifier labels...')
    df = collect_units(include_wireless=args.include_wireless)
    print(f'  {len(df)} total units, {df.source.nunique()} sources')
    if not args.include_wireless:
        print('  (wireless excluded — pass --include-wireless to add them)')

    # Pick n per bin to roughly target args.n total
    bins = 5
    types = 2 if args.include_wireless else 1
    n_per_bin = max(1, args.n // (bins * types))
    print(f'\nStratified sampling: ~{n_per_bin} per bin ({bins} P-bins x {types} source types)')
    sample = stratified_sample(df, n_per_bin)
    print(f'  Sampled {len(sample)} units')

    print('\nSource distribution:')
    print(sample.groupby(['source_type', 'p_bin'], observed=False).size().unstack(fill_value=0))

    # Resolve figure paths
    print('\nResolving figure paths (existing on disk)...')
    sample = sample.reset_index(drop=True)
    sample['fig_path'] = sample.apply(find_figure_path, axis=1)
    n_existing = sample['fig_path'].notna().sum()
    n_missing = sample['fig_path'].isna().sum()
    print(f'  {n_existing} existing, {n_missing} need to be generated')

    # Sort sample so order is stable and meaningful (group by source/bank to
    # exploit analyzer cache when generating figures)
    sample = sample.sort_values(['source', 'bank', 'unit_id']).reset_index(drop=True)

    print(f'\nProducing {len(sample)} figures in {POOL_DIR}...')
    kept_rows = []
    for i, r in sample.iterrows():
        new_id = len(kept_rows)
        dst = POOL_DIR / f"unit_{new_id:04d}.png"
        if r['fig_path'] is not None and not (isinstance(r['fig_path'], float) and pd.isna(r['fig_path'])):
            shutil.copy2(Path(r['fig_path']), dst)
            kept_rows.append(i)
        else:
            ok = generate_figure(r, dst)
            if ok:
                kept_rows.append(i)
        if (i + 1) % 10 == 0:
            print(f'  Progress: {i+1}/{len(sample)}  (kept {len(kept_rows)})')

    sample = sample.loc[kept_rows].reset_index(drop=True)
    sample['id'] = range(len(sample))

    # Save manifest
    manifest = sample[['id', 'source', 'survey', 'bank', 'unit_id',
                        'noise_neural_label', 'noise_neural_prob', 'p_neural',
                        'ks4_label']].copy()
    manifest.to_csv(POOL_DIR / 'manifest.csv', index=False)
    print(f'\nManifest: {POOL_DIR / "manifest.csv"}')
    print(f'Pool ready: {POOL_DIR}')

    # Initialize empty labels.csv if not present
    labels_path = POOL_DIR / 'labels.csv'
    if not labels_path.exists():
        pd.DataFrame(columns=['id', 'label', 'timestamp']).to_csv(labels_path, index=False)
        print(f'Labels: {labels_path} (empty, will be populated by labeler)')


if __name__ == '__main__':
    main()
