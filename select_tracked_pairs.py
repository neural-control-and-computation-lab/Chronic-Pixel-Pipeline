"""
Select tracked-neuron pairs for example figures, balancing time gap against
match + unit quality. Writes a ranked CSV.

Score per pair:
  track_conf  = UnitMatch probability
  neural_conf = sqrt(p_neural_1 * p_neural_2)          (classifier confidence)
  gap_norm    = gap_days / max_gap
  amp_norm    = clip(min(amp1, amp2) / 200 uV, 0, 1)
  score = track_conf * neural_conf
          * (0.55 + 0.45*gap_norm)        # gap modulates, doesn't dominate
          * (0.60 + 0.40*amp_norm)        # bigger units preferred

This lets an excellent short-gap pair outrank a mediocre long-gap one, while
still rewarding large separations. Dedupe to one pair per tracked neuron
(bank, UID); spread across banks.

Usage:
  python select_tracked_pairs.py [N]    # default 50
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
P_MIN = 0.85
GAP_MIN = 21
# Isolation gate — both units of a pair must look like clean single units so
# the supplementary panels actually convince. (mockup ref: SNR 13, ISI 0%.)
SNR_MIN = 3.0
ISI_MAX = 0.02      # isi_violations_ratio (2%)
RP_MAX = 0.20       # rp_contamination (20%)
SURGERY = datetime(2026, 2, 19)
DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
OUT_CSV = SORT_ROOT / 'tracking' / 'selected_pairs.csv'

_QM_CACHE = {}


def qm_lookup(survey, bank, unit):
    """(snr, isi_violations_ratio, rp_contamination) for a unit, or NaNs."""
    key = (survey, bank)
    if key not in _QM_CACHE:
        p = (SORT_ROOT / f'survey_{survey}' / f'bank{bank:02d}_ks4'
             / 'analyzer_cache' / 'extensions' / 'quality_metrics' / 'metrics.csv')
        _QM_CACHE[key] = pd.read_csv(p, index_col=0) if p.exists() else None
    d = _QM_CACHE[key]
    if d is None or unit not in d.index:
        return np.nan, np.nan, np.nan
    r = d.loc[unit]
    return (float(r.get('snr', np.nan)),
            float(r.get('isi_violations_ratio', np.nan)),
            float(r.get('rp_contamination', np.nan)))


def unit_quality_lookup():
    """(survey, bank, unit_id) -> (amplitude_uV, p_neural)."""
    amp = {}
    pn = {}
    for sd in sorted(SORT_ROOT.glob('survey_*')):
        sk = sd.name.replace('survey_', '')
        mp = sd / 'merged' / 'merged_units.csv'
        lp = sd / 'auto_curation_labels_custom.csv'
        if mp.exists():
            m = pd.read_csv(mp)
            id_col = 'original_id' if 'original_id' in m.columns else 'unit_id'
            for _, r in m.iterrows():
                amp[(sk, int(r['bank']), int(r[id_col]))] = float(r['amplitude'])
        if lp.exists():
            l = pd.read_csv(lp)
            for _, r in l.iterrows():
                pn[(sk, int(r['bank']), int(r['unit_id']))] = float(r['custom_p_neural'])
    return amp, pn


def days_post(sk):
    return (datetime.strptime(sk, '%Y%m%d') - SURGERY).days


def main():
    df = pd.read_csv(SORT_ROOT / 'tracking' / 'tracking_summary.csv')
    df['s1'] = df['session_1'].astype(str)
    df['s2'] = df['session_2'].astype(str)
    df['d1'] = df['s1'].map(days_post)
    df['d2'] = df['s2'].map(days_post)
    # Orient earlier -> later
    swap = df['d1'] > df['d2']
    for a, b in [('s1', 's2'), ('unit_1', 'unit_2'), ('depth_1', 'depth_2'),
                 ('d1', 'd2')]:
        df.loc[swap, [a, b]] = df.loc[swap, [b, a]].values
    df['gap_days'] = (df['d2'] - df['d1']).abs()

    cand = df[(df['probability'] >= P_MIN) & (df['gap_days'] >= GAP_MIN)].copy()

    amp, pn = unit_quality_lookup()

    def lookup(row, which):
        sk = row['s1'] if which == 1 else row['s2']
        u = int(row['unit_1'] if which == 1 else row['unit_2'])
        b = int(row['bank'])
        return amp.get((sk, b, u), np.nan), pn.get((sk, b, u), np.nan)

    a1 = []; a2 = []; p1 = []; p2 = []
    snr1 = []; snr2 = []; isi1 = []; isi2 = []; rp1 = []; rp2 = []
    for _, r in cand.iterrows():
        x = lookup(r, 1); y = lookup(r, 2)
        a1.append(x[0]); p1.append(x[1]); a2.append(y[0]); p2.append(y[1])
        q1 = qm_lookup(r['s1'], int(r['bank']), int(r['unit_1']))
        q2 = qm_lookup(r['s2'], int(r['bank']), int(r['unit_2']))
        snr1.append(q1[0]); isi1.append(q1[1]); rp1.append(q1[2])
        snr2.append(q2[0]); isi2.append(q2[1]); rp2.append(q2[2])
    cand['amp1'] = a1; cand['amp2'] = a2
    cand['pn1'] = p1; cand['pn2'] = p2
    cand['snr1'] = snr1; cand['snr2'] = snr2
    cand['isi1'] = isi1; cand['isi2'] = isi2
    cand['rp1'] = rp1; cand['rp2'] = rp2

    # Both units classified neural-ish AND well-isolated single units, so the
    # supplementary panels (ACG dip, low ISI viol, high SNR) are convincing.
    cand = cand[(cand['pn1'] >= 0.5) & (cand['pn2'] >= 0.5)
                & (cand['snr1'] >= SNR_MIN) & (cand['snr2'] >= SNR_MIN)
                & (cand['isi1'] <= ISI_MAX) & (cand['isi2'] <= ISI_MAX)
                & (cand['rp1'] <= RP_MAX) & (cand['rp2'] <= RP_MAX)].copy()
    print(f'After isolation gate (SNR>={SNR_MIN}, ISI<={ISI_MAX}, '
          f'RP<={RP_MAX}): {len(cand)} pairs')

    gap_max = cand['gap_days'].max()
    cand['neural_conf'] = np.sqrt(cand['pn1'] * cand['pn2'])
    cand['gap_norm'] = cand['gap_days'] / gap_max
    cand['snr_norm'] = np.clip(np.minimum(cand['snr1'], cand['snr2']) / 8.0,
                                0, 1)
    cand['score'] = (cand['probability'] * cand['neural_conf']
                     * (0.55 + 0.45 * cand['gap_norm'])
                     * (0.50 + 0.50 * cand['snr_norm']))

    cand = cand.sort_values('score', ascending=False)
    cand = cand.drop_duplicates(subset=['bank', 'UID'], keep='first')

    # Spread across banks: round-robin by score order.
    by_bank = {b: list(g.index) for b, g in cand.groupby('bank')}
    for b in by_bank:
        by_bank[b].sort(key=lambda i: -cand.loc[i, 'score'])
    order = []
    while len(order) < N and any(by_bank.values()):
        for b in sorted(by_bank):
            if by_bank[b]:
                order.append(by_bank[b].pop(0))
                if len(order) >= N:
                    break
    sel = cand.loc[order].copy()
    sel = sel.sort_values('score', ascending=False).reset_index(drop=True)

    cols = ['bank', 's1', 'd1', 'unit_1', 'amp1', 'pn1', 'snr1', 'isi1',
            's2', 'd2', 'unit_2', 'amp2', 'pn2', 'snr2', 'isi2',
            'probability', 'gap_days', 'score']
    sel[cols].to_csv(OUT_CSV, index=False)
    pd.set_option('display.width', 240)
    pd.set_option('display.max_columns', 40)
    print(f'Wrote {OUT_CSV} ({len(sel)} pairs)')
    print()
    show = sel[['bank', 'd1', 'unit_1', 'snr1', 'isi1', 'd2', 'unit_2',
                'snr2', 'isi2', 'probability', 'gap_days', 'score']].copy()
    for c in ['snr1', 'snr2', 'probability', 'score']:
        show[c] = show[c].round(2)
    for c in ['isi1', 'isi2']:
        show[c] = (show[c] * 100).round(1)
    print(show.head(50).to_string(index=False))


if __name__ == '__main__':
    main()
