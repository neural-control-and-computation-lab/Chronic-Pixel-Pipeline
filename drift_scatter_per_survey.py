"""
Build a drift-scatter raster: one column per survey, depth (µm) on Y, time
on X. Each dot is a detected spike, alpha-scaled by amplitude.

Reads from each survey's KS4 output:
  sorter_output/spike_times.npy       (samples)
  sorter_output/spike_clusters.npy
  sorter_output/amplitudes.npy
  sorter_output/channel_positions.npy
  sorter_output/spike_templates.npy   (or via cluster_KSLabel.tsv)

Output:
  E:/figures/sorting/tracking/drift_scatter_per_survey.png
  E:/figures/sorting/tracking/drift_scatter_per_survey.svg

Banks are stacked vertically (bank 0 on bottom).
Surveys are arranged horizontally with vertical separators.

Usage:
  python drift_scatter_per_survey.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import logging
import sys
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
from track_neurons import SURVEYS

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
FIG_DIR = DATA_ROOT / 'figures' / 'sorting' / 'tracking'

N_BANKS = 7
BANK_HEIGHT_UM = 3840  # full NP1 bank height
FS = 30000.0


def load_spike_data(sort_dir, bank_id, neural_only=False, p_thresh=0.5):
    """Load spike times (s), depths (µm), amplitudes (a.u.) for a bank.

    If ``neural_only``, restrict to clusters labeled neural (custom_p_neural
    >= p_thresh) in ``auto_curation_labels_custom.csv``.
    """
    bank_dir = sort_dir / f'bank{bank_id:02d}_ks4' / 'sorter_output'
    if not bank_dir.exists():
        return None
    spike_times_p = bank_dir / 'spike_times.npy'
    if not spike_times_p.exists():
        return None
    spike_times = np.load(spike_times_p).flatten().astype(np.int64)
    if len(spike_times) == 0:
        return None
    spike_clusters = np.load(bank_dir / 'spike_clusters.npy').flatten()
    amplitudes = np.load(bank_dir / 'amplitudes.npy').flatten()
    spike_templates = np.load(bank_dir / 'spike_templates.npy').flatten()
    chan_positions = np.load(bank_dir / 'channel_positions.npy')
    templates = np.load(bank_dir / 'templates.npy')

    if neural_only:
        labels_path = (sort_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
                        / 'extensions' / 'auto_curation' / 'labels_custom.csv')
        if not labels_path.exists():
            return None
        labels = pd.read_csv(labels_path)
        neural_ids = set(
            labels.loc[labels['custom_p_neural'] >= p_thresh,
                       'unit_id'].astype(int).tolist())
        if not neural_ids:
            return None
        keep = np.isin(spike_clusters, list(neural_ids))
        if not keep.any():
            return None
        spike_times = spike_times[keep]
        spike_clusters = spike_clusters[keep]
        amplitudes = amplitudes[keep]
        spike_templates = spike_templates[keep]

    # Spike depth = depth of the template's peak channel
    template_ptp = np.ptp(templates, axis=1)            # (n_templates, n_chan)
    peak_chan_per_template = np.argmax(template_ptp, axis=1)
    spike_depth = chan_positions[
        peak_chan_per_template[spike_templates], 1]
    times_s = spike_times / FS
    return {
        'times_s': times_s,
        'depth_um': spike_depth.astype(float),
        'amp': np.abs(amplitudes).astype(float),
        'duration_s': float(times_s.max()) if len(times_s) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural-only', action='store_true',
                        help='Only plot spikes from clusters classified neural '
                              '(custom_p_neural >= --p-thresh)')
    parser.add_argument('--p-thresh', type=float, default=0.5)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    survey_keys = sorted(SURVEYS.keys())

    suffix = '_neural' if args.neural_only else ''

    # 1) Load all per-survey, per-bank spike data + survey durations
    logger.info('Loading spike data across %d surveys × %d banks (neural_only=%s)...',
                len(survey_keys), N_BANKS, args.neural_only)
    survey_data = {}
    survey_durations = []
    for sk in survey_keys:
        sort_dir = SURVEYS[sk]['sort_dir']
        per_bank = {}
        max_dur = 0.0
        for b in range(N_BANKS):
            d = load_spike_data(sort_dir, b,
                                  neural_only=args.neural_only,
                                  p_thresh=args.p_thresh)
            if d is not None:
                per_bank[b] = d
                max_dur = max(max_dur, d['duration_s'])
        if per_bank:
            survey_data[sk] = per_bank
            survey_durations.append(max_dur)
            logger.info('  %s: %d banks, max %.1fs', sk, len(per_bank), max_dur)
        else:
            logger.warning('  %s: no banks loaded', sk)

    if not survey_data:
        logger.error('No data loaded.')
        return

    # 2) Layout: each survey gets a column of width = its duration; banks
    #    stacked vertically; depth axis spans all banks.
    used_surveys = [sk for sk in survey_keys if sk in survey_data]
    durations = [max(survey_data[sk][b]['duration_s']
                      for b in survey_data[sk])
                  for sk in used_surveys]
    cum_offsets = np.concatenate(([0], np.cumsum(durations)))
    total_t = cum_offsets[-1]

    # Figure size: roughly proportional to total time
    width = max(14, total_t / 60)  # ~1 inch per minute
    height = 8.5
    fig, ax = plt.subplots(figsize=(width, height))

    # 3) Scatter spikes for each bank — depths already absolute (set by
    #    make_bank_probe in sort_survey.py: z = row * z_pitch with global row)
    for s_idx, sk in enumerate(used_surveys):
        t_off = cum_offsets[s_idx]
        for b in range(N_BANKS):
            if b not in survey_data[sk]:
                continue
            d = survey_data[sk][b]
            x = d['times_s'] + t_off
            y = d['depth_um']  # already absolute
            n = len(x)
            if n > 200000:
                idx = np.random.choice(n, 200000, replace=False)
                x, y, amp = x[idx], y[idx], d['amp'][idx]
            else:
                amp = d['amp']
            # Color by amplitude — red-ish hot colormap
            amp_norm = np.clip(amp / np.percentile(amp, 99), 0, 1)
            ax.scatter(x, y, s=0.3, c=amp_norm, cmap='Reds',
                        vmin=0, vmax=1, alpha=0.55,
                        marker='.', rasterized=True, linewidths=0)

        if s_idx > 0:
            ax.axvline(t_off, color='#888888', lw=0.4, alpha=0.6, zorder=2)
        ax.text(t_off + durations[s_idx] / 2, 25000 + 200,
                sk[4:], rotation=90, fontsize=7, ha='center', va='bottom',
                color='#555555')

    # 4) Bank boundary lines (only show those inside view)
    for b in range(1, N_BANKS):
        z = b * BANK_HEIGHT_UM
        if z >= 25000:
            break
        ax.axhline(z, color='#cccccc', lw=0.3, alpha=0.6, zorder=1)
        ax.text(total_t + 2, z - BANK_HEIGHT_UM / 2,
                 f'Bank {b-1}', ha='left', va='center', fontsize=9,
                 color='#666666')
    # Top bank label
    ax.text(total_t + 2, 25000 - 100, f'Bank {N_BANKS-1}+',
             ha='left', va='top', fontsize=9, color='#666666')

    ax.set_xlabel('Time within survey (s) — concatenated across surveys')
    ax.set_ylabel('Depth (µm)')
    ax.set_xlim(-2, total_t + 80)
    ax.set_ylim(0, 25000)
    title = (f'Spike depth vs time across {len(used_surveys)} surveys, '
             f'banks 0–{N_BANKS-1}')
    if args.neural_only:
        title += f'  (neural only, P ≥ {args.p_thresh:.2f})'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / f'drift_scatter_per_survey{suffix}.png'
    fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    fig.savefig(out.with_suffix('.svg'), bbox_inches='tight',
                 facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', out)


if __name__ == '__main__':
    main()
