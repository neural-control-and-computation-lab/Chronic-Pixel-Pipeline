"""
Minimalistic waveform plots for TRACKED PAIRS — the same neuron recorded in
two sessions far apart in time, matched with high confidence by UnitMatch.

Each plot overlays the two sessions' templates on a common channel geometry
(same bank → same electrode positions), two colours, no axes. The point is
the visual match between waveforms recorded weeks apart.

Selection: from tracking_summary.csv, keep pairs with probability >= --p-min,
dedupe to one pair per tracked neuron (UID), then rank by time gap (desc) then
probability (desc) and take the top N. Banks are spread where possible.

File name encodes: early date, late date, bank, both unit ids, gap, probability.

Usage:
  python make_minimal_pair_plots.py [N]      # default N=20
"""
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
P_MIN = 0.90
GAP_MIN = 45  # days
DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
TRACK_CSV = SORT_ROOT / 'tracking' / 'tracking_summary.csv'
OUT_DIR = DATA_ROOT / 'figures' / 'sorting' / 'minimal_pairs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EARLY_COLOR = '#1F4E9B'   # blue  — first session
LATE_COLOR = '#C03020'    # red   — later session


def pick_pairs(n):
    df = pd.read_csv(TRACK_CSV)
    df['s1'] = df['session_1'].astype(str)
    df['s2'] = df['session_2'].astype(str)

    def gap(r):
        d1 = datetime.strptime(r['s1'], '%Y%m%d')
        d2 = datetime.strptime(r['s2'], '%Y%m%d')
        return (d2 - d1).days

    df['gap_days'] = df.apply(gap, axis=1)
    # Orient so s1 is always the earlier session
    flip = df['gap_days'] < 0
    if flip.any():
        df.loc[flip, ['s1', 's2', 'unit_1', 'unit_2', 'depth_1', 'depth_2']] = \
            df.loc[flip, ['s2', 's1', 'unit_2', 'unit_1',
                          'depth_2', 'depth_1']].values
    df['gap_days'] = df['gap_days'].abs()

    cand = df[(df['probability'] >= P_MIN) & (df['gap_days'] >= GAP_MIN)].copy()
    # One representative pair per tracked neuron (UID): the largest-gap,
    # highest-probability instance.
    cand = cand.sort_values(['gap_days', 'probability'],
                             ascending=[False, False])
    cand = cand.drop_duplicates(subset=['bank', 'UID'], keep='first')

    # Spread across banks: round-robin over banks in gap-desc order so we
    # don't end up with 20 bank-3 pairs.
    cand = cand.sort_values(['gap_days', 'probability'],
                             ascending=[False, False])
    by_bank = {b: list(g.index) for b, g in cand.groupby('bank')}
    order = []
    while len(order) < n and any(by_bank.values()):
        for b in sorted(by_bank):
            if by_bank[b]:
                order.append(by_bank[b].pop(0))
                if len(order) >= n:
                    break
    return cand.loc[order].reset_index(drop=True)


def load_dense_template(survey, bank, unit_id):
    """Return (template (T, n_ch), ch_locs (n_ch, 2)) — dense over the bank."""
    import spikeinterface.full as si
    cache = SORT_ROOT / f'survey_{survey}' / f'bank{bank:02d}_ks4' / 'analyzer_cache'
    analyzer = si.load_sorting_analyzer(cache)
    try:
        analyzer.load_extension('templates')
    except Exception:
        pass
    templates = analyzer.get_extension('templates').get_data()
    unit_ids = list(analyzer.sorting.unit_ids)
    uid_idx = unit_ids.index(unit_id)
    return templates[uid_idx], analyzer.get_channel_locations()


def select_channels(template, ch_locs, n_rows=7):
    """Channel indices for top n_rows rows centered on the peak channel,
    both columns per row. Returns index array into the dense layout."""
    ptp = np.ptp(template, axis=0)
    peak_y = float(ch_locs[int(np.argmax(ptp)), 1])
    unique_y = np.sort(np.unique(ch_locs[:, 1]))
    keep_y = set(unique_y[np.argsort(np.abs(unique_y - peak_y))[:n_rows]])
    keep_idx = []
    for y in sorted(keep_y):
        ids = np.where(ch_locs[:, 1] == y)[0]
        if len(ids) == 0:
            continue
        xs = ch_locs[ids, 0]
        left = ids[int(np.argmin(xs))]
        right = ids[int(np.argmax(xs))]
        keep_idx.append(left)
        if right != left:
            keep_idx.append(right)
    return np.array(keep_idx)


def plot_pair(t_early, t_late, ch_locs, out_path):
    """Overlay two templates (already restricted to the same channel set, on
    identical positions ch_locs) with no axes. Narrow aspect via a virtual
    column projection."""
    if len(ch_locs) == 0:
        return
    n_samples = t_early.shape[0]
    amp_max = float(max(np.abs(t_early).max(), np.abs(t_late).max()))
    if amp_max == 0:
        amp_max = 1.0

    unique_y = np.sort(np.unique(ch_locs[:, 1]))
    y_pitch = float(np.min(np.diff(unique_y))) if len(unique_y) > 1 else 20.0
    n_rows = len(unique_y)
    target_x_extent = (n_rows * y_pitch) * 0.5
    x_pitch_virtual = target_x_extent / 1.6

    unique_x = np.sort(np.unique(ch_locs[:, 0]))
    if len(unique_x) > 1:
        x_min = float(unique_x[0])
        x_virtual = np.where(np.isclose(ch_locs[:, 0], x_min),
                             0.0, x_pitch_virtual)
    else:
        x_virtual = np.zeros(len(ch_locs))

    amp_scale = (y_pitch * 0.85) / amp_max
    t_extent = x_pitch_virtual * 0.8
    t_scale = t_extent / (n_samples - 1)
    t = np.arange(n_samples) * t_scale - (n_samples - 1) * t_scale / 2

    fig, ax = plt.subplots(figsize=(3, 6))
    for tmpl, color in ((t_early, EARLY_COLOR), (t_late, LATE_COLOR)):
        for i in range(tmpl.shape[1]):
            x0 = float(x_virtual[i])
            y0 = float(ch_locs[i, 1])
            ax.plot(x0 + t, y0 + tmpl[:, i] * amp_scale,
                    color=color, lw=1.6, alpha=0.85,
                    solid_capstyle='round', solid_joinstyle='round')
    ax.set_aspect('equal')
    ax.set_axis_off()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=220, bbox_inches='tight',
                facecolor='white', pad_inches=0.05)
    fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight',
                facecolor='white', pad_inches=0.05)
    plt.close(fig)


def main():
    pairs = pick_pairs(N)
    print(f'Selected {len(pairs)} tracked pairs (p>={P_MIN}, gap>={GAP_MIN}d)')
    saved = 0
    for _, r in pairs.iterrows():
        bank = int(r['bank'])
        s1, s2 = r['s1'], r['s2']
        u1, u2 = int(r['unit_1']), int(r['unit_2'])
        prob, gap = float(r['probability']), int(r['gap_days'])
        try:
            t1, locs1 = load_dense_template(s1, bank, u1)
            t2, locs2 = load_dense_template(s2, bank, u2)
            keep = select_channels(t1, locs1)
            # Map session-2 channels to the same coordinates (positions are
            # identical for the same bank, so indices match).
            t1k = t1[:, keep]
            t2k = t2[:, keep]
            locs = locs1[keep]
            # Time-align: KS4 templates share nt, but trim to the shorter if
            # they differ.
            m = min(t1k.shape[0], t2k.shape[0])
            t1k, t2k = t1k[:m], t2k[:m]
        except Exception as e:
            print(f'  skip {s1}->{s2} bank{bank:02d} u{u1}/u{u2}: {e}')
            continue
        fname = (f'{s1}_to_{s2}_bank{bank:02d}_u{u1:03d}_u{u2:03d}'
                 f'_gap{gap}d_p{prob:.2f}.png')
        plot_pair(t1k, t2k, locs, OUT_DIR / fname)
        print(f'  Saved {fname}')
        saved += 1
    print(f'\nDone. {saved} pair figures in {OUT_DIR}')
    print(f'  {EARLY_COLOR} = earlier session, {LATE_COLOR} = later session')


if __name__ == '__main__':
    main()
