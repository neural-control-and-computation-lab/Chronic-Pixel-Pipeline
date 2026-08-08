"""
Minimalistic waveform plots: top-N largest-amplitude classifier-strict units.

For each unit:
  - Load template + sparse channel mask from its analyzer_cache
  - Plot multi-channel waveform overlay (one trace per sparse channel)
  - No axes, no labels, no title
  - File name encodes: survey + bank + unit id

Usage:
  python make_minimal_unit_plots.py [N]   # default N=20
"""
import os
import sys
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
OUT_DIR = DATA_ROOT / 'figures' / 'sorting' / 'minimal_examples'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_top_units(n):
    rows = []
    for sd in sorted(SORT_ROOT.glob('survey_*')):
        sk = sd.name.replace('survey_', '')
        mp = sd / 'merged' / 'merged_units.csv'
        lp = sd / 'auto_curation_labels_custom.csv'
        if not mp.exists() or not lp.exists():
            continue
        m = pd.read_csv(mp).rename(columns={'original_id': 'unit_id'})
        l = pd.read_csv(lp)[['unit_id', 'bank', 'custom_p_neural']]
        j = m.merge(l, on=['unit_id', 'bank'], how='inner')
        j['survey'] = sk
        rows.append(j[(j['custom_p_neural'] >= 0.8) & (j['bank'] <= 6)])
    all_df = pd.concat(rows, ignore_index=True)

    # Stratified sampling: ensure good spread across (survey, bank) cells.
    # For each cell, rank units by amplitude descending. Round-robin pick
    # rank-1 from every cell, then rank-2, ..., until we have N.
    all_df = all_df.sort_values(['survey', 'bank', 'amplitude'],
                                  ascending=[True, True, False])
    all_df['rank_in_cell'] = all_df.groupby(['survey', 'bank']).cumcount()

    picked = []
    for rank in sorted(all_df['rank_in_cell'].unique()):
        layer = all_df[all_df['rank_in_cell'] == rank]
        # Within a rank layer, order by amplitude desc so the absolute
        # biggest neurons are still preferred.
        layer = layer.sort_values('amplitude', ascending=False)
        for _, row in layer.iterrows():
            picked.append(row)
            if len(picked) >= n:
                break
        if len(picked) >= n:
            break
    return pd.DataFrame(picked).reset_index(drop=True)


def load_unit_waveform(survey, bank, unit_id, n_rows=7):
    """Return (template, ch_positions) for one unit. Template restricted to
    the top `n_rows` rows centered on the peak channel — i.e. n_rows × 2
    channels (one per column), symmetric layout."""
    import spikeinterface.full as si
    cache = SORT_ROOT / f'survey_{survey}' / f'bank{bank:02d}_ks4' / 'analyzer_cache'
    analyzer = si.load_sorting_analyzer(cache)
    for ext in ('templates',):
        try:
            analyzer.load_extension(ext)
        except Exception:
            pass

    templates = analyzer.get_extension('templates').get_data()
    unit_ids = list(analyzer.sorting.unit_ids)
    uid_idx = unit_ids.index(unit_id)
    tmpl_full = templates[uid_idx]  # (T, n_ch_dense)
    all_locs = analyzer.get_channel_locations()

    # Identify peak channel + the row it's in
    ptp = np.ptp(tmpl_full, axis=0)
    peak_ch = int(np.argmax(ptp))
    peak_y = float(all_locs[peak_ch, 1])

    # Unique row y-positions, sort by distance from peak row, take nearest n_rows
    unique_y = np.sort(np.unique(all_locs[:, 1]))
    row_dists = np.abs(unique_y - peak_y)
    keep_y = set(unique_y[np.argsort(row_dists)[:n_rows]])

    # Among kept rows, pick BOTH columns (smallest+largest x) for each row.
    # If a row only has one channel, that one is taken.
    keep_idx = []
    for y in sorted(keep_y):
        mask = all_locs[:, 1] == y
        ids = np.where(mask)[0]
        if len(ids) == 0:
            continue
        xs = all_locs[ids, 0]
        # left + right channel in this row
        left = ids[int(np.argmin(xs))]
        right = ids[int(np.argmax(xs))]
        keep_idx.append(left)
        if right != left:
            keep_idx.append(right)

    keep_idx = np.array(keep_idx)
    return tmpl_full[:, keep_idx], all_locs[keep_idx]


def plot_unit(template, ch_locs, out_path):
    """Minimal multi-channel waveform plot — no axes, just traces. Spacing
    is tuned so adjacent traces never overlap horizontally or vertically.

    To control aspect ratio independently of NP1032's wide column spacing
    (103 µm), we re-project the two columns onto a virtual narrower layout
    so the final image ends up about half as wide as it is tall.
    """
    if len(ch_locs) == 0:
        return

    n_samples = template.shape[0]
    amp_max = float(np.abs(template).max())
    if amp_max == 0:
        amp_max = 1.0

    # Row pitch (vertical spacing between adjacent rows).
    unique_y = np.sort(np.unique(ch_locs[:, 1]))
    if len(unique_y) > 1:
        y_pitch = float(np.min(np.diff(unique_y)))
    else:
        y_pitch = 20.0

    # VIRTUAL column pitch — drives the image aspect ratio. Choose so that
    # x_extent (~1.6 × x_pitch) is roughly half of y_extent (n_rows × y_pitch).
    n_rows = len(unique_y)
    target_x_extent = (n_rows * y_pitch) * 0.5
    x_pitch_virtual = target_x_extent / 1.6

    # Project actual x positions onto the virtual layout: column min → 0,
    # column max → x_pitch_virtual.
    unique_x = np.sort(np.unique(ch_locs[:, 0]))
    if len(unique_x) > 1:
        x_min, x_max = float(unique_x[0]), float(unique_x[-1])
        # Map each channel's x to {0, x_pitch_virtual}
        x_virtual = np.where(np.isclose(ch_locs[:, 0], x_min),
                              0.0, x_pitch_virtual)
    else:
        x_virtual = np.zeros(len(ch_locs))

    # Amplitude scale so peak waveform fits within ~85% of a row pitch.
    amp_scale = (y_pitch * 0.85) / amp_max

    # Time scale so each waveform spans ~80% of the virtual column pitch.
    t_extent = x_pitch_virtual * 0.8
    t_scale = t_extent / (n_samples - 1)
    t = np.arange(n_samples) * t_scale - (n_samples - 1) * t_scale / 2

    fig, ax = plt.subplots(figsize=(3, 6))
    for i in range(template.shape[1]):
        x0 = float(x_virtual[i])
        y0 = float(ch_locs[i, 1])
        ax.plot(x0 + t, y0 + template[:, i] * amp_scale,
                color='black', lw=1.8, solid_capstyle='round',
                solid_joinstyle='round')

    ax.set_aspect('equal')
    ax.set_axis_off()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=220, bbox_inches='tight',
                facecolor='white', pad_inches=0.05)
    fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight',
                facecolor='white', pad_inches=0.05)
    plt.close(fig)


def main():
    top = pick_top_units(N)
    print(f'Plotting top {len(top)} units...')
    saved = 0
    for _, row in top.iterrows():
        sk = row['survey']
        bank = int(row['bank'])
        uid = int(row['unit_id'])
        amp = row['amplitude']
        try:
            tmpl, locs = load_unit_waveform(sk, bank, uid)
        except Exception as e:
            print(f'  skip {sk} bank{bank:02d} u{uid}: {e}')
            continue
        fname = f'{sk}_bank{bank:02d}_unit{uid:04d}.png'
        out_path = OUT_DIR / fname
        plot_unit(tmpl, locs, out_path)
        print(f'  Saved {fname} ({amp:.0f} µV)')
        saved += 1
    print(f'\nDone. {saved} figures in {OUT_DIR}')


if __name__ == '__main__':
    main()
