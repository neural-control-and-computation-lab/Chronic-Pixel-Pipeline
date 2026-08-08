"""
Regenerate all tracking figures from already-saved outputs:

  E:/sorting_output/tracking/tracking_summary.csv          (pairwise matches)
  E:/sorting_output/tracking/bank{NN}/prob_matrix.npy
  E:/sorting_output/survey_*/auto_curation_labels_custom.csv

Does NOT re-run UnitMatch — pulls neuron identity from the saved match list
by union-find over (session, bank, unit) nodes.

Re-fires:
  - trackability_curve.png       (new clean design)
  - good_units_over_time.png     (via track_neurons helper)
  - tracking_summary.png         (via track_neurons helper)

Usage:
  python regen_tracking_plots.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_neurons import (SURVEYS, plot_trackability_curve,
                            get_classifier_good_unit_ids)

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
OUTPUT_DIR = SORT_ROOT / 'tracking'
FIG_DIR = DATA_ROOT / 'figures' / 'sorting' / 'tracking'


class DSU:
    """Disjoint-set union (union-find)."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        while self.parent.setdefault(x, x) != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_um_results_from_summary():
    """Reconstruct an all_um_results-shaped dict from tracking_summary.csv
    + per-survey custom labels."""
    summary_path = OUTPUT_DIR / 'tracking_summary.csv'
    matches = pd.read_csv(summary_path)
    # session_* columns come back as int — re-cast to string to match
    # date_key format ('20260226').
    matches['session_1'] = matches['session_1'].astype(str)
    matches['session_2'] = matches['session_2'].astype(str)
    logger.info('Loaded %d pairwise matches', len(matches))

    date_keys = sorted(SURVEYS.keys())

    all_um_results = {}
    for bank_id in sorted(matches['bank'].unique()):
        bank_matches = matches[matches['bank'] == bank_id]

        # Build list of (session, unit) -> "neuron candidate" for every good
        # unit in every session for this bank
        nodes = []   # ordered list of (session, unit)
        for dk in date_keys:
            sort_dir = SURVEYS[dk]['sort_dir']
            try:
                good = get_classifier_good_unit_ids(sort_dir, bank_id)
            except Exception as e:
                logger.warning('  %s bank%d: %s', dk, bank_id, e)
                continue
            for u in good:
                nodes.append((dk, int(u)))

        if not nodes:
            continue
        node_idx = {n: i for i, n in enumerate(nodes)}

        # Union pairs from match list
        dsu = DSU()
        for _, r in bank_matches.iterrows():
            a = (str(r['session_1']), int(r['unit_1']))
            b = (str(r['session_2']), int(r['unit_2']))
            # It's possible matches reference units that aren't in our
            # "good" list (e.g. classifier filter changed since matching).
            # Treat them as nodes anyway so spans still compute.
            if a not in node_idx:
                node_idx[a] = len(nodes); nodes.append(a)
            if b not in node_idx:
                node_idx[b] = len(nodes); nodes.append(b)
            dsu.union(a, b)

        # Each connected component = one neuron with a UID; assign integer UIDs
        roots = {}
        uids = np.empty(len(nodes), dtype=int)
        session_id = np.empty(len(nodes), dtype=int)
        active_dates = date_keys
        date_to_session = {dk: i for i, dk in enumerate(date_keys)}
        next_uid = 0
        for i, (dk, u) in enumerate(nodes):
            r = dsu.find((dk, u))
            if r not in roots:
                roots[r] = next_uid
                next_uid += 1
            uids[i] = roots[r]
            session_id[i] = date_to_session[dk]

        all_um_results[bank_id] = {
            'UIDs': uids,
            'session_id': session_id,
            'active_dates': active_dates,
            'prob_matrix': None,
        }
        logger.info('  Bank %d: %d candidate units, %d unique neurons '
                     '(%d in ≥2 sessions)',
                     bank_id, len(nodes), next_uid,
                     sum(1 for c in pd.Series(uids).value_counts()
                          if c > 1))
    return all_um_results, date_keys


SURGERY_DATE = datetime(2026, 2, 19)


def plot_good_units_over_time(all_um_results, date_keys, fig_dir):
    """Stacked-area plot of good units per bank over days-since-surgery."""
    parsed_dates = [datetime.strptime(dk, '%Y%m%d') for dk in date_keys]
    days_since = np.array([(d - SURGERY_DATE).days for d in parsed_dates])
    bank_ids = sorted(all_um_results.keys())

    counts = pd.DataFrame(0, index=date_keys, columns=bank_ids)
    for bank_id, um in all_um_results.items():
        for dk_idx, dk in enumerate(um['active_dates']):
            mask = um['session_id'] == dk_idx
            counts.loc[dk, bank_id] = int(mask.sum())

    total = counts.sum(axis=1).values
    ymax = total.max() * 1.18

    # Stacked area: banks in numeric order (bank 0 on bottom)
    bank_order = sorted(bank_ids)
    stack = np.vstack([counts[b].values for b in bank_order])

    fig, ax = plt.subplots(figsize=(12.5, 5.6))

    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(bank_order)))
    ax.stackplot(days_since, stack,
                  labels=[f'Bank {b}' for b in bank_order],
                  colors=cmap, edgecolor='white', linewidth=0.6,
                  alpha=0.92)

    # Total line and per-survey markers
    ax.plot(days_since, total, color='#222222', lw=1.6, alpha=0.9,
             zorder=4)
    ax.scatter(days_since, total, s=22, color='#222222', zorder=5)

    for x, t in zip(days_since, total):
        ax.text(x, t + ymax * 0.022, str(int(t)),
                ha='center', va='bottom', fontsize=8, color='#222222')

    # Surgery marker at day 0
    ax.axvline(0, color='#C0504D', linestyle='--', lw=1.4, alpha=0.9,
                zorder=6)
    ax.annotate('Surgery', xy=(0, ymax * 0.95),
                 xytext=(4, 0), textcoords='offset points',
                 fontsize=10, color='#C0504D', fontweight='bold',
                 va='top', ha='left')

    ax.set_xlim(-3, days_since.max() + 3)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Days since surgery')
    ax.set_ylabel('Classifier-good units (P ≥ 0.5)')
    ax.set_title(f'Good units per survey across {len(date_keys)} sessions '
                 f'(banks 0–{bank_ids[-1] if bank_ids else 6})',
                 fontsize=12, fontweight='bold')
    ax.legend(ncol=len(bank_order), fontsize=8, loc='upper right',
               frameon=False)
    ax.grid(True, alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = fig_dir / 'good_units_over_time.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(out.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', out)


def regenerate_drift_scatter_plots():
    """Re-run drift_scatter_per_survey.py for both the all-units and the
    neural-only variants. Run as subprocesses so the (heavy) data-loading
    code in that script doesn't import its globals twice into this process.
    """
    import subprocess
    repo = Path(__file__).resolve().parent
    script = repo / 'drift_scatter_per_survey.py'
    for label, args in [('all units', []),
                         ('neural-only', ['--neural-only'])]:
        logger.info('Regenerating drift_scatter_per_survey (%s)...', label)
        try:
            subprocess.run([sys.executable, str(script), *args], check=True,
                            env={**os.environ})
        except subprocess.CalledProcessError as e:
            logger.warning('drift_scatter_per_survey (%s) failed: %s',
                           label, e)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info('Rebuilding tracking results from saved CSV...')
    all_um_results, date_keys = build_um_results_from_summary()

    logger.info('\nRegenerating trackability_curve.png...')
    plot_trackability_curve(all_um_results, date_keys, FIG_DIR)

    logger.info('\nRegenerating good_units_over_time.png...')
    plot_good_units_over_time(all_um_results, date_keys, FIG_DIR)

    logger.info('\nRegenerating drift_scatter_per_survey plots...')
    regenerate_drift_scatter_plots()

    logger.info('\nSyncing tracking figures to Dropbox...')
    try:
        from figure_sync import sync_to_dropbox
        sync_to_dropbox('tracking')
    except Exception as e:
        logger.warning('Dropbox sync failed: %s', e)

    logger.info('Done.')


if __name__ == '__main__':
    main()
