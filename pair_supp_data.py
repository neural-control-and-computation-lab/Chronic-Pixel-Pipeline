"""
Extract everything needed to plot a tracked-pair composite figure, and dump it
to a self-contained .npz + .json so figure code (or workflow agents) can render
without touching SpikeInterface.

For each unit of the pair we collect:
  - the dense template restricted to a shared 7-row x 2-col channel block
    (centered on the early unit's peak), on real electrode coordinates
  - ACG (autocorrelogram) counts + bin centers (ms)
  - per-spike amplitude vs time (s) for the stability scatter
  - scalar quality metrics: snr, firing_rate, num_spikes,
    isi_violations_ratio, rp_contamination, amplitude_median

Usage:
  python pair_supp_data.py SURVEY1 BANK UNIT1 SURVEY2 UNIT2 OUTSTEM
  # e.g. python pair_supp_data.py 20260323 4 70 20260604 72 E:/tmp/mock
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
SURGERY = datetime(2026, 2, 19)


def _analyzer(survey, bank):
    import spikeinterface.full as si
    cache = SORT_ROOT / f'survey_{survey}' / f'bank{bank:02d}_ks4' / 'analyzer_cache'
    a = si.load_sorting_analyzer(cache)
    for e in ('templates', 'correlograms', 'spike_amplitudes', 'quality_metrics'):
        try:
            a.load_extension(e)
        except Exception:
            pass
    return a


def select_channels(template, ch_locs, n_rows=7):
    ptp = np.ptp(template, axis=0)
    peak_y = float(ch_locs[int(np.argmax(ptp)), 1])
    uy = np.sort(np.unique(ch_locs[:, 1]))
    keep_y = set(uy[np.argsort(np.abs(uy - peak_y))[:n_rows]])
    idx = []
    for y in sorted(keep_y):
        ids = np.where(ch_locs[:, 1] == y)[0]
        if len(ids) == 0:
            continue
        xs = ch_locs[ids, 0]
        left = ids[int(np.argmin(xs))]
        right = ids[int(np.argmax(xs))]
        idx.append(left)
        if right != left:
            idx.append(right)
    return np.array(idx)


def unit_blob(a, uid, keep_idx=None):
    ui = list(a.sorting.unit_ids).index(uid)
    templates = a.get_extension('templates').get_data()
    tmpl = templates[ui]                       # (T, n_ch)
    locs = a.get_channel_locations()
    ccgs, bins = a.get_extension('correlograms').get_data()
    acg = ccgs[ui, ui].astype(float)
    acg_centers = (bins[:-1] + bins[1:]) / 2.0
    sv = a.sorting.to_spike_vector()
    sa = np.asarray(a.get_extension('spike_amplitudes').get_data())
    mask = sv['unit_index'] == ui
    fs = float(a.sampling_frequency)
    amp_t = sv['sample_index'][mask] / fs
    amp_v = sa[mask]
    qm = a.get_extension('quality_metrics').get_data()
    row = qm.loc[uid].to_dict() if uid in qm.index else {}

    def g(k):
        v = row.get(k, np.nan)
        try:
            return float(v)
        except Exception:
            return float('nan')

    metrics = {
        'snr': g('snr'),
        'firing_rate': g('firing_rate'),
        'num_spikes': g('num_spikes'),
        'isi_violations_ratio': g('isi_violations_ratio'),
        'rp_contamination': g('rp_contamination'),
        'amplitude_median': g('amplitude_median'),
    }
    return tmpl, locs, acg, acg_centers, amp_t, amp_v, metrics


def days_post(sk):
    return (datetime.strptime(sk, '%Y%m%d') - SURGERY).days


def extract(survey1, bank, unit1, survey2, unit2):
    a1 = _analyzer(survey1, bank)
    a2 = _analyzer(survey2, bank)
    t1, locs1, acg1, acgc1, at1, av1, m1 = unit_blob(a1, unit1)
    keep = select_channels(t1, locs1)
    t2, locs2, acg2, acgc2, at2, av2, m2 = unit_blob(a2, unit2)
    nT = min(t1.shape[0], t2.shape[0])
    blob = {
        'bank': int(bank),
        'survey1': survey1, 'survey2': survey2,
        'unit1': int(unit1), 'unit2': int(unit2),
        'day1': days_post(survey1), 'day2': days_post(survey2),
        'keep_locs': locs1[keep],
        'tmpl1': t1[:nT][:, keep], 'tmpl2': t2[:nT][:, keep],
        'acg1': acg1, 'acg2': acg2, 'acg_centers': acgc1,
        'amp_t1': at1, 'amp_v1': av1, 'amp_t2': at2, 'amp_v2': av2,
        'metrics1': m1, 'metrics2': m2,
    }
    return blob


def save(blob, outstem):
    outstem = Path(outstem)
    outstem.parent.mkdir(parents=True, exist_ok=True)
    arr = {k: v for k, v in blob.items()
           if isinstance(v, np.ndarray)}
    np.savez(outstem.with_suffix('.npz'), **arr)
    meta = {k: v for k, v in blob.items()
            if not isinstance(v, np.ndarray)}
    outstem.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print('Saved', outstem.with_suffix('.npz'), 'and', outstem.with_suffix('.json'))


if __name__ == '__main__':
    s1, bank, u1, s2, u2, stem = sys.argv[1:7]
    blob = extract(s1, int(bank), int(u1), s2, int(u2))
    save(blob, stem)
    # quick sanity print
    print('day1', blob['day1'], 'day2', blob['day2'],
          'chans', blob['keep_locs'].shape,
          't1', blob['tmpl1'].shape, 'acg1 sum', blob['acg1'].sum(),
          'spikes1', len(blob['amp_v1']), 'spikes2', len(blob['amp_v2']))
    print('m1', blob['metrics1'])
    print('m2', blob['metrics2'])
