"""
Pre-compute waveform quality metrics for boundary noise detection.

For each bank's analyzer cache, computes per unit:
  1. peak_amplitude     — max peak-to-peak voltage across channels (uV)
  2. n_spatial_channels — channels with PTP > 25% of peak channel PTP
  3. max_pointwise_snr  — peak(|mean_wf| / std_wf) at the peak channel
  4. amp_cv             — std(spike amplitudes) / |mean(spike amplitudes)|
  5. acg_refr_ratio     — sum(ACG counts within +/-2 ms) / sum(counts 5-20 ms)

Firing rate is read from the existing quality_metrics CSV (not recomputed).

A composite flag (boundary_noise_flag) is set using six criteria with tiered
thresholds based on spatial footprint size (n_spatial_channels >= 14 = large
footprint, likely a real neuron with relaxed thresholds):

  c1: n_spat <= 2 AND amp < 50  (unless snr_pw >= 4.0 AND amp_cv <= 0.25)
  c2: n_spat <= 3 AND amp < 50 AND (snr_pw < 3.5 OR amp_cv > 0.30)
  c3: amp < 45 AND fr < 1.0 (standard) | amp < 30 AND fr < 0.5 (large fp)
  c4: amp_cv > 0.80 (standard) | amp_cv > 1.00 (large fp)
  c5: snr_pw < 2.7 AND amp_cv > 0.40 AND NOT large footprint
  c6: n_spat <= 4 AND amp_cv > 0.65

Saves results as waveform_quality/metrics.csv inside each analyzer_cache/extensions/.
Reads raw NPY/CSV cache files directly — no SortingAnalyzer object needed.

Usage:
  python precompute_waveform_quality.py          # all surveys
  python precompute_waveform_quality.py 20260309  # single survey
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'

SURVEYS = ['20260226', '20260302', '20260304', '20260306', '20260309', '20260311', '20260313', '20260316', '20260319', '20260323', '20260326', '20260331', '20260402', '20260407', '20260409', '20260414', '20260417', '20260421', '20260424', '20260428', '20260501', '20260507', '20260512', '20260522', '20260526', '20260604', '20260611', '20260618', '20260625', '20260723', '20260730', '20260813', '20260820', '20260826', '20260914', '20260924']
N_BANKS = 7


def compute_flag(df):
    """Compute boundary_noise_flag from metrics columns.

    Criteria (tiered by spatial footprint):
      large_fp: n_spatial_channels >= 14 (large footprint = likely real neuron)

      c1: n_spat <= 2 AND amp < 50, unless rescued by snr_pw >= 4 AND amp_cv <= 0.25
      c2: n_spat <= 3 AND amp < 50 AND (snr_pw < 3.5 OR amp_cv > 0.30)
      c3: amp < 45 AND fr < 1.0  (standard)
          amp < 30 AND fr < 0.5  (large footprint)
      c4: amp_cv > 0.80  (standard)
          amp_cv > 1.00  (large footprint)
      c5: snr_pw < 2.7 AND amp_cv > 0.40 AND NOT large_fp
      c6: n_spat <= 4 AND amp_cv > 0.65
    """
    n_spat = df['n_spatial_channels']
    amp = df['peak_amplitude']
    snr_pw = df['max_pointwise_snr'].fillna(0)
    amp_cv_filled = df['amp_cv'].fillna(999)
    fr = df['firing_rate'].fillna(999)
    amp_cv_raw = df['amp_cv']  # keep NaN as NaN

    large_fp = n_spat >= 14

    rescue = (snr_pw >= 4.0) & (amp_cv_filled <= 0.25)
    c1 = (n_spat <= 2) & (amp < 50) & ~rescue
    c2 = (n_spat <= 3) & (amp < 50) & ((snr_pw < 3.5) | (amp_cv_filled > 0.30))

    # C3: tiered by footprint size
    c3_standard = (amp < 45) & (fr < 1.0) & ~large_fp
    c3_large = (amp < 30) & (fr < 0.5) & large_fp
    c3 = c3_standard | c3_large

    # C4: tiered by footprint size
    c4_standard = (amp_cv_raw > 0.80) & ~large_fp
    c4_large = (amp_cv_raw > 1.00) & large_fp
    c4 = c4_standard | c4_large

    # C5: low SNR + moderate instability (small units only)
    c5 = (snr_pw < 2.7) & (amp_cv_filled > 0.40) & ~large_fp

    # C6: very small footprint + unstable amplitude
    c6 = (n_spat <= 4) & (amp_cv_raw > 0.65)

    return c1 | c2 | c3 | c4 | c5 | c6


def compute_bank_metrics(cache_dir):
    """Compute waveform quality metrics for one bank from cached files.

    Returns a DataFrame with columns:
        peak_amplitude, n_spatial_channels, max_pointwise_snr, amp_cv,
        acg_refr_ratio, firing_rate, boundary_noise_flag
    indexed by unit_id.
    """
    # Load arrays
    templates = np.load(cache_dir / 'extensions/templates/average.npy')
    sparsity = np.load(cache_dir / 'sparsity_mask.npy')
    wf = np.load(cache_dir / 'extensions/waveforms/waveforms.npy',
                 mmap_mode='r')
    rs_indices = np.load(
        cache_dir / 'extensions/random_spikes/random_spikes_indices.npy')
    spikes = np.load(cache_dir / 'sorting/spikes.npy')
    amplitudes = np.load(
        cache_dir / 'extensions/spike_amplitudes/amplitudes.npy')
    ccgs = np.load(cache_dir / 'extensions/correlograms/ccgs.npy')
    bins = np.load(cache_dir / 'extensions/correlograms/bins.npy')

    info = json.loads(
        (cache_dir / 'sorting/numpysorting_info.json').read_text())
    unit_ids = info['unit_ids']
    n_units = len(unit_ids)

    # Load firing rate from quality_metrics
    qm_path = cache_dir / 'extensions/quality_metrics/metrics.csv'
    qm = pd.read_csv(qm_path, index_col=0) if qm_path.exists() else None

    # Pre-compute mapping: random spike → unit index
    rs_unit_idx = spikes['unit_index'][rs_indices]

    # ACG bin masks (same for all units)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    refr_mask = np.abs(bin_centers) <= 2      # ±2 ms refractory zone
    shoulder_mask = ((np.abs(bin_centers) >= 5) &
                     (np.abs(bin_centers) <= 20))  # 5–20 ms baseline

    rows = []
    for uid_idx in range(n_units):
        uid = unit_ids[uid_idx]

        # ── peak_amplitude & n_spatial_channels ───────────────────
        ptp_per_ch = np.ptp(templates[uid_idx], axis=0)
        peak_ch = int(np.argmax(ptp_per_ch))
        peak_amp = float(ptp_per_ch[peak_ch])
        n_spat = int(np.sum(ptp_per_ch > 0.25 * peak_amp))

        # ── max_pointwise_snr ──────────────────────────────────────
        sparse_ch = np.where(sparsity[uid_idx])[0]
        peak_in_sparse = np.where(sparse_ch == peak_ch)[0]

        if len(peak_in_sparse) > 0:
            wf_mask = rs_unit_idx == uid_idx
            n_wf = int(np.sum(wf_mask))
            if n_wf >= 2:
                peak_wfs = wf[wf_mask][:, :, peak_in_sparse[0]]
                mean_wf = np.mean(peak_wfs, axis=0)
                std_wf = np.std(peak_wfs, axis=0)
                snr_pw = np.abs(mean_wf) / np.clip(std_wf, 1e-10, None)
                max_snr = float(np.max(snr_pw))
            else:
                max_snr = np.nan
        else:
            max_snr = np.nan

        # ── amp_cv ─────────────────────────────────────────────────
        spike_mask = spikes['unit_index'] == uid_idx
        unit_amps = amplitudes[spike_mask]
        if len(unit_amps) >= 5:
            amp_mean = np.abs(np.mean(unit_amps))
            amp_cv = float(np.std(unit_amps) / amp_mean) if amp_mean > 0 else np.nan
        else:
            amp_cv = np.nan

        # ── acg_refr_ratio ─────────────────────────────────────────
        auto_ccg = ccgs[uid_idx, uid_idx, :]
        refr_count = float(np.sum(auto_ccg[refr_mask]))
        shoulder_count = float(np.sum(auto_ccg[shoulder_mask]))
        acg_ratio = (refr_count / shoulder_count
                     if shoulder_count > 0 else np.nan)

        # ── firing_rate (from quality_metrics) ─────────────────────
        fr = float(qm.loc[uid, 'firing_rate']) if (qm is not None and uid in qm.index) else np.nan

        rows.append({
            'unit_id': uid,
            'peak_amplitude': round(peak_amp, 2),
            'n_spatial_channels': n_spat,
            'max_pointwise_snr': round(max_snr, 4) if not np.isnan(max_snr) else np.nan,
            'amp_cv': round(amp_cv, 4) if not np.isnan(amp_cv) else np.nan,
            'acg_refr_ratio': round(acg_ratio, 4) if not np.isnan(acg_ratio) else np.nan,
            'firing_rate': round(fr, 4) if not np.isnan(fr) else np.nan,
        })

    df = pd.DataFrame(rows).set_index('unit_id')
    df['boundary_noise_flag'] = compute_flag(df)
    return df


def process_survey(survey_key):
    """Process all banks for one survey."""
    sort_dir = SORT_ROOT / f'survey_{survey_key}'
    n_saved = 0
    n_flagged = 0
    n_units = 0

    for bank_id in range(N_BANKS):
        cache_dir = sort_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'

        # Check all required files exist
        required = [
            'extensions/templates/average.npy',
            'sparsity_mask.npy',
            'extensions/waveforms/waveforms.npy',
            'extensions/random_spikes/random_spikes_indices.npy',
            'sorting/spikes.npy',
            'extensions/spike_amplitudes/amplitudes.npy',
            'extensions/correlograms/ccgs.npy',
            'extensions/correlograms/bins.npy',
            'sorting/numpysorting_info.json',
        ]
        missing = [f for f in required if not (cache_dir / f).exists()]
        if missing:
            if cache_dir.exists():
                logger.warning("  Bank %d: missing %s, skipping", bank_id, missing)
            continue

        try:
            out_dir = cache_dir / 'extensions' / 'waveform_quality'
            if (out_dir / 'metrics.csv').exists():
                df = pd.read_csv(out_dir / 'metrics.csv', index_col=0)
                bank_flagged = int(df['boundary_noise_flag'].sum())
                n_saved += 1
                n_flagged += bank_flagged
                n_units += len(df)
                logger.info("  Bank %d: %d units (cached)", bank_id, len(df))
                continue

            df = compute_bank_metrics(cache_dir)

            # Save
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / 'metrics.csv')

            bank_flagged = int(df['boundary_noise_flag'].sum())
            n_saved += 1
            n_flagged += bank_flagged
            n_units += len(df)
            logger.info("  Bank %d: %d units, %d flagged as boundary noise",
                        bank_id, len(df), bank_flagged)
        except Exception as e:
            logger.warning("  Bank %d: failed — %s", bank_id, e)

    logger.info("Survey %s: %d banks processed, %d/%d units flagged",
                survey_key, n_saved, n_flagged, n_units)
    return n_flagged, n_units


def main():
    surveys = sys.argv[1:] if len(sys.argv) > 1 else SURVEYS
    total_flagged = 0
    total_units = 0

    for key in surveys:
        logger.info("=" * 60)
        logger.info("SURVEY %s", key)
        logger.info("=" * 60)
        flagged, units = process_survey(key)
        total_flagged += flagged
        total_units += units

    logger.info("\nDone! %d/%d units flagged across %d surveys.",
                total_flagged, total_units, len(surveys))


if __name__ == '__main__':
    main()
