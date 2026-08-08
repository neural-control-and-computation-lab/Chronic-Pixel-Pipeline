"""
Render the tracked-pair composite figure (mirrored-symmetry layout, the
design judged best). For each selected pair:

  - centre: multichannel template overlay (same neuron, two sessions) with
    µV / ms scale bars and a colour key directly above it
  - far left  (Day d1, blue):  ACG (shaded refractory window + ISI%),
                               amplitude-vs-time, metrics block
  - far right (Day d2, ochre): mirrored ACG, amplitude, metrics block
  - top: title + colour-coded days-post-implant legend

Modes:
  python make_pair_composite.py mock                 # render just the example
  python make_pair_composite.py all [N]              # render all selected pairs

'all' reads E:/sorting_output/tracking/selected_pairs.csv (from
select_tracked_pairs.py), extracts each pair's data via pair_supp_data, and
renders to E:/figures/sorting/minimal_pairs_v2/.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import pair_panels as pp

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
OUT_DIR = DATA_ROOT / 'figures' / 'sorting' / 'minimal_pairs_v2'
MOCK_STEM = OUT_DIR / '_mock' / 'mock_data'
SEL_CSV = DATA_ROOT / 'sorting_output' / 'tracking' / 'selected_pairs.csv'


def render_pair(stem, out_png, title=None):
    arrs, meta = pp.load(stem)
    m1, m2 = meta['metrics1'], meta['metrics2']

    fig = plt.figure(figsize=(13, 7.4))
    # 3 columns: left supp | centre overlay | right supp.
    outer = gridspec.GridSpec(2, 3, figure=fig,
                              width_ratios=[1.0, 1.15, 1.0],
                              height_ratios=[0.001, 1.0],
                              hspace=0.05, wspace=0.28,
                              left=0.05, right=0.97, top=0.84, bottom=0.10)

    # Title + legend band (manual, above everything)
    if title is None:
        title = (f"Single neuron tracked across {abs(meta['day2'] - meta['day1'])} "
                 f"days  (bank {meta['bank']})")
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.96)
    fig.text(0.5, 0.90, pp.day_label(meta, 1), color=pp.COL_EARLY,
             fontsize=11, fontweight='bold', ha='right')
    fig.text(0.505, 0.90, '   ·   ', color='#888888', fontsize=11, ha='center')
    fig.text(0.51, 0.90, pp.day_label(meta, 2), color=pp.COL_LATE,
             fontsize=11, fontweight='bold', ha='left')

    # Left rail (Day 1): ACG, amplitude, metrics
    left = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[1, 0], height_ratios=[1.0, 1.0, 0.7],
        hspace=0.55)
    axL_acg = fig.add_subplot(left[0])
    axL_amp = fig.add_subplot(left[1])
    axL_met = fig.add_subplot(left[2])
    pp.draw_acg(axL_acg, arrs['acg_centers'], arrs['acg1'], pp.COL_EARLY,
                label=f"ACG · Day {meta['day1']}",
                isi_pct=m1['isi_violations_ratio'] * 100)
    pp.draw_amp(axL_amp, arrs['amp_t1'], arrs['amp_v1'], pp.COL_EARLY,
                label=f"Amplitude · Day {meta['day1']}")
    pp.draw_metrics(axL_met, m1, pp.COL_EARLY, header=f"Day {meta['day1']}")

    # Centre overlay with a colour key just above it
    axC = fig.add_subplot(outer[1, 1])
    pp.draw_overlay(axC, arrs)
    axC.set_title('same neuron — template overlay', fontsize=9,
                  color='#444444', pad=14)
    axC.text(0.30, 1.015, '■', color=pp.COL_EARLY, fontsize=10,
             ha='right', va='bottom', transform=axC.transAxes)
    axC.text(0.31, 1.015, f"Day {meta['day1']}", color=pp.COL_EARLY,
             fontsize=8, ha='left', va='bottom', transform=axC.transAxes)
    axC.text(0.66, 1.015, '■', color=pp.COL_LATE, fontsize=10,
             ha='right', va='bottom', transform=axC.transAxes)
    axC.text(0.67, 1.015, f"Day {meta['day2']}", color=pp.COL_LATE,
             fontsize=8, ha='left', va='bottom', transform=axC.transAxes)

    # Right rail (Day 2): ACG, amplitude, metrics (mirrored)
    right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[1, 2], height_ratios=[1.0, 1.0, 0.7],
        hspace=0.55)
    axR_acg = fig.add_subplot(right[0])
    axR_amp = fig.add_subplot(right[1])
    axR_met = fig.add_subplot(right[2])
    pp.draw_acg(axR_acg, arrs['acg_centers'], arrs['acg2'], pp.COL_LATE,
                label=f"ACG · Day {meta['day2']}",
                isi_pct=m2['isi_violations_ratio'] * 100)
    pp.draw_amp(axR_amp, arrs['amp_t2'], arrs['amp_v2'], pp.COL_LATE,
                label=f"Amplitude · Day {meta['day2']}")
    pp.draw_metrics(axR_met, m2, pp.COL_LATE, header=f"Day {meta['day2']}")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor='white', bbox_inches='tight')
    fig.savefig(out_png.with_suffix('.svg'), facecolor='white',
                bbox_inches='tight')
    plt.close(fig)


def run_all(n=None):
    from pair_supp_data import extract, save
    sel = pd.read_csv(SEL_CSV)
    # Only the identity fields are required to render; amplitude/p_neural may
    # be NaN for a unit missing from the merged table and that's harmless.
    sel = sel.dropna(subset=['bank', 's1', 's2', 'unit_1', 'unit_2'])
    if n:
        sel = sel.head(n)
    tmp = OUT_DIR / '_data'
    saved = 0
    for _, r in sel.iterrows():
        bank = int(r['bank'])
        # CSV round-trip can read 8-digit survey keys as floats (20260311.0);
        # coerce back to clean integer strings.
        s1, s2 = str(int(float(r['s1']))), str(int(float(r['s2'])))
        u1, u2 = int(r['unit_1']), int(r['unit_2'])
        d1, d2 = int(r['d1']), int(r['d2'])
        stem = tmp / f'{s1}_b{bank:02d}_u{u1}_to_{s2}_u{u2}'
        try:
            blob = extract(s1, bank, u1, s2, u2)
            save(blob, stem)
            fname = (f'bank{bank:02d}_day{d1:03d}_to_day{d2:03d}'
                     f'_{s1}u{u1:03d}_{s2}u{u2:03d}_gap{int(r["gap_days"])}d'
                     f'_p{r["probability"]:.2f}.png')
            render_pair(stem, OUT_DIR / fname)
            print(f'  {fname}')
            saved += 1
        except Exception as e:
            print(f'  skip bank{bank:02d} {s1}u{u1}->{s2}u{u2}: {e}')
    print(f'\nDone. {saved} composite figures in {OUT_DIR}')


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'mock'
    if mode == 'mock':
        render_pair(MOCK_STEM, OUT_DIR / '_mock' / 'composite_mockup.png')
        print('Saved', OUT_DIR / '_mock' / 'composite_mockup.png')
    elif mode == 'all':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_all(n)


if __name__ == '__main__':
    main()
