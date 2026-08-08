"""
Correlate neural activity with IMU signals and decode IMU from neural populations.

For each bank of the wireless recording, simultaneously recorded neurons and
IMU data (accel/gyro/mag) are analyzed:
  1. Per-neuron correlation with each IMU channel (Pearson, Spearman, lag-optimized)
  2. Population decoding of IMU signals via Ridge regression (TimeSeriesSplit CV)

Each bank was recorded as a separate ~4 min segment, so only neurons from the
same bank are correlated with that segment's IMU data.

IMU conversion uses assumed ICM-20948 default sensitivity:
  accel: raw / 16384 -> g    (±2g range)
  gyro:  raw / 131   -> °/s  (±250 dps range)
  mag:   raw * 0.15  -> µT   (±4900 µT range)

Usage:
  python analyze_neural_imu.py E:/March23_wireless
  python analyze_neural_imu.py E:/March23_wireless --bank 0 2
  python analyze_neural_imu.py E:/March23_wireless --bin-ms 50 --smooth-ms 25
  python analyze_neural_imu.py E:/March23_wireless --raw-imu
"""
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy import stats
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import sys
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


# ── IMU Loading & Processing ─────────────────────────────────────────

def load_per_segment_imu(data_dir, output_dir):
    """Re-extract IMU per .dat segment, with caching.

    Returns {bank_id: {'accel': (N,3), 'gyro': (N,3), 'mag': (N,3),
                        'sampling_rate': 2500.0}}
    """
    import chronic_pixel_pipeline as cpp
    from sort_wireless import discover_segments

    segments = discover_segments(data_dir)
    result = {}

    for seg in segments:
        bank_id = seg['bank']
        cache_dir = output_dir / f'bank{bank_id:02d}_ks4' / 'imu'

        # Check cache
        if (cache_dir / 'accel.npy').exists():
            logger.info("  Bank %d: loading cached IMU", bank_id)
            result[bank_id] = {
                'accel': np.load(cache_dir / 'accel.npy'),
                'gyro': np.load(cache_dir / 'gyro.npy'),
                'mag': np.load(cache_dir / 'mag.npy'),
                'sampling_rate': 2500.0,
            }
            continue

        logger.info("  Bank %d: extracting IMU from %s...",
                     bank_id, seg['path'].name)
        try:
            imu = cpp.load_imu(str(seg['path']))
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / 'accel.npy', imu['accel'])
            np.save(cache_dir / 'gyro.npy', imu['gyro'])
            np.save(cache_dir / 'mag.npy', imu['mag'])
            result[bank_id] = imu
            logger.info("    %d samples (%.1fs)",
                         len(imu['accel']),
                         len(imu['accel']) / imu['sampling_rate'])
        except Exception as e:
            logger.warning("  Bank %d: IMU extraction failed: %s", bank_id, e)

    return result


def convert_imu_to_physical(imu_raw):
    """Convert raw int16 IMU to physical units (assumed ICM-20948 defaults).

    Returns dict with same keys but float64 arrays in physical units.
    """
    return {
        'accel': imu_raw['accel'].astype(np.float64) / 16384.0,   # g
        'gyro': imu_raw['gyro'].astype(np.float64) / 131.0,       # deg/s
        'mag': imu_raw['mag'].astype(np.float64) * 0.15,          # µT
        'sampling_rate': imu_raw['sampling_rate'],
    }


def compute_derived_imu(imu_phys):
    """Compute derived IMU channels: magnitudes, bandpass-filtered accel.

    Returns dict with 15 channels total:
      accel_{x,y,z}, gyro_{x,y,z}, mag_{x,y,z}  (9 raw)
      accel_mag, gyro_mag, jerk_mag               (3 magnitude)
      accel_filt_{x,y,z}                          (3 bandpass 0.5-50 Hz)
    """
    fs = imu_phys['sampling_rate']
    # Trim all to same length (sensors can differ by 1-2 samples)
    n = min(len(imu_phys['accel']), len(imu_phys['gyro']),
            len(imu_phys['mag']))
    accel = imu_phys['accel'][:n]
    gyro = imu_phys['gyro'][:n]
    mag = imu_phys['mag'][:n]

    # Magnitudes
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    jerk = np.diff(accel, axis=0) * fs  # (n-1, 3)
    jerk = np.vstack([jerk, jerk[-1:]])  # pad to (n, 3)
    jerk_mag = np.linalg.norm(jerk, axis=1)

    # Lowpass all signals at 10 Hz to remove high-freq sensor noise
    sos_lp = sig.butter(4, 10, btype='low', fs=fs, output='sos')
    accel_lp = sig.sosfiltfilt(sos_lp, accel, axis=0)
    gyro_lp = sig.sosfiltfilt(sos_lp, gyro, axis=0)
    mag_lp = sig.sosfiltfilt(sos_lp, mag, axis=0)

    # Recompute magnitudes and jerk from filtered signals
    accel_mag_lp = np.linalg.norm(accel_lp, axis=1)
    gyro_mag_lp = np.linalg.norm(gyro_lp, axis=1)
    jerk_lp = np.diff(accel_lp, axis=0) * fs
    jerk_lp = np.vstack([jerk_lp, jerk_lp[-1:]])
    jerk_mag_lp = np.linalg.norm(jerk_lp, axis=1)

    # Bandpass accel (remove gravity DC + high-freq noise)
    sos_bp = sig.butter(4, [0.5, 10], btype='bandpass', fs=fs, output='sos')
    accel_filt = sig.sosfiltfilt(sos_bp, accel, axis=0)

    return {
        'signals': np.column_stack([
            accel_lp, gyro_lp, mag_lp,
            accel_mag_lp, gyro_mag_lp, jerk_mag_lp,
            accel_filt,
        ]),
        'names': [
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z',
            'accel_mag', 'gyro_mag', 'jerk_mag',
            'accel_filt_x', 'accel_filt_y', 'accel_filt_z',
        ],
        'sampling_rate': fs,
    }


def bin_imu_signals(imu_derived, bin_ms):
    """Gaussian-smooth IMU signals, then downsample to bin_ms resolution.

    Applies very aggressive Gaussian smoothing (sigma = 20 * bin_ms) at the
    native IMU rate, then decimates. The heavy smoothing overcomes the coarse
    quantization (values are multiples of 64 due to 10-bit packing).

    Returns (imu_binned, t_bins, channel_names).
      imu_binned: (n_bins, n_channels) float64
      t_bins: (n_bins,) bin center times in seconds
    """
    fs = imu_derived['sampling_rate']
    signals = imu_derived['signals']
    n_samples = signals.shape[0]

    # Gaussian smooth at native rate (sigma = 10x bin size = 1s)
    sigma_samples = 10.0 * (bin_ms / 1000.0) * fs
    smoothed = gaussian_filter1d(signals, sigma=sigma_samples, axis=0)

    # Downsample to bin_ms resolution
    samples_per_bin = int(fs * bin_ms / 1000)
    n_bins = n_samples // samples_per_bin
    # Pick center of each bin from smoothed signal
    indices = (np.arange(n_bins) * samples_per_bin +
               samples_per_bin // 2).astype(int)
    imu_binned = smoothed[indices]

    t_bins = indices / fs

    return imu_binned, t_bins, imu_derived['names']


# ── Spike Train Binning ──────────────────────────────────────────────

def bin_spike_trains(cache_dir, good_unit_ids, bin_ms, smooth_ms, duration_s):
    """Gaussian-smooth spike trains and downsample to firing rate matrix.

    First bins spikes at 1ms resolution, applies aggressive Gaussian smoothing
    (sigma = 5 * bin_ms), then decimates to bin_ms resolution. This matches
    the IMU smoothing approach.

    Returns:
      fr_binned: (n_bins, n_units) smoothed firing rates in Hz
      unit_ids_used: list of unit IDs actually used
    """
    spikes = np.load(cache_dir / 'sorting' / 'spikes.npy')
    info = json.loads(
        (cache_dir / 'sorting' / 'numpysorting_info.json').read_text())
    all_unit_ids = info['unit_ids']
    fs = info['sampling_frequency']

    # Fine-resolution binning at 1ms
    fine_bin_s = 0.001
    n_fine = int(duration_s / fine_bin_s)
    fine_edges = np.arange(n_fine + 1) * fine_bin_s

    # Gaussian smooth: sigma = 10 * bin_ms in fine-bin units (1s)
    sigma_fine = 10.0 * bin_ms  # bin_ms is in ms, fine bins are 1ms each

    # Downsample indices
    bin_s = bin_ms / 1000.0
    n_bins = int(duration_s / bin_s)
    decimate_indices = (np.arange(n_bins) * (bin_ms) +
                        bin_ms // 2).astype(int)
    decimate_indices = decimate_indices[decimate_indices < n_fine]
    n_bins = len(decimate_indices)

    unit_ids_used = []
    fr_list = []

    for uid in good_unit_ids:
        if uid not in all_unit_ids:
            continue
        uid_idx = all_unit_ids.index(uid)
        spike_samples = spikes['sample_index'][spikes['unit_index'] == uid_idx]
        spike_times = spike_samples / fs

        counts, _ = np.histogram(spike_times, bins=fine_edges)
        # Convert to instantaneous rate (Hz) at 1ms resolution
        fr_fine = counts / fine_bin_s

        # Aggressive Gaussian smooth
        fr_smooth = gaussian_filter1d(fr_fine, sigma=sigma_fine)

        # Decimate
        fr_list.append(fr_smooth[decimate_indices])
        unit_ids_used.append(uid)

    if not fr_list:
        return np.empty((n_bins, 0)), []

    return np.column_stack(fr_list), unit_ids_used


# ── Correlation Analysis ─────────────────────────────────────────────

def compute_correlations(fr_binned, imu_binned, unit_ids, imu_names,
                         bank_id, bin_ms, max_lag_ms=500):
    """Compute per-neuron correlations with each IMU channel.

    Searches lags up to ±max_lag_ms for peak correlation.
    Returns DataFrame with columns: unit_id, bank, imu_channel,
      pearson_r, pearson_p, spearman_r, spearman_p, best_lag_ms,
      r_at_best_lag.
    """
    n_bins, n_units = fr_binned.shape
    n_imu = imu_binned.shape[1]
    max_lag_bins = int(max_lag_ms / bin_ms)

    rows = []
    for u_idx in range(n_units):
        fr = fr_binned[:, u_idx]
        uid = unit_ids[u_idx]

        for ch_idx in range(n_imu):
            imu_ch = imu_binned[:, ch_idx]

            # Zero-lag correlations
            r_p, p_p = stats.pearsonr(fr, imu_ch)
            r_s, p_s = stats.spearmanr(fr, imu_ch)

            # Lag search
            best_r = r_p
            best_lag = 0
            for lag in range(-max_lag_bins, max_lag_bins + 1):
                if lag == 0:
                    continue
                if lag > 0:
                    r_lag, _ = stats.pearsonr(fr[lag:], imu_ch[:-lag])
                else:
                    r_lag, _ = stats.pearsonr(fr[:lag], imu_ch[-lag:])
                if abs(r_lag) > abs(best_r):
                    best_r = r_lag
                    best_lag = lag

            rows.append({
                'unit_id': uid,
                'bank': bank_id,
                'imu_channel': imu_names[ch_idx],
                'pearson_r': round(r_p, 6),
                'pearson_p': p_p,
                'spearman_r': round(r_s, 6),
                'spearman_p': p_s,
                'best_lag_ms': best_lag * bin_ms,
                'r_at_best_lag': round(best_r, 6),
            })

    return pd.DataFrame(rows)


# ── Spike-Triggered Average ──────────────────────────────────────────

def compute_sta_significance(spike_samples, imu_signal, imu_fs=2500,
                             neural_fs=30000, window_ms=500, n_shuffles=100):
    """Compute STA with circular-shift shuffle significance test.

    Optimized: precomputes IMU index offsets once, then uses vectorized
    circular shifts for all shuffles in batch.

    Returns
    -------
    sta : (n_window, n_channels) mean STA
    shuffle_mean : (n_window, n_channels) mean of shuffle STAs
    shuffle_std : (n_window, n_channels) std of shuffle STAs
    p_values : (n_channels,) fraction of shuffles with ptp >= real ptp
    n_spikes : int
    """
    downsample = int(neural_fs / imu_fs)  # 12
    half_win = int(window_ms / 1000 * imu_fs)
    n_window = 2 * half_win + 1
    n_imu = len(imu_signal)
    n_ch = imu_signal.shape[1]

    # Convert spike times to IMU indices
    spike_imu = spike_samples // downsample
    valid = (spike_imu >= half_win) & (spike_imu < n_imu - half_win)
    spike_imu = spike_imu[valid]
    n_spikes = len(spike_imu)

    if n_spikes == 0:
        z = np.zeros((n_window, n_ch))
        return z, z.copy(), np.ones_like(z), np.ones(n_ch), 0

    # Precompute offset array
    offsets = np.arange(-half_win, half_win + 1)  # (n_window,)

    # Real STA — vectorized
    indices = spike_imu[:, None] + offsets[None, :]  # (n_spikes, n_window)
    sta = imu_signal[indices].mean(axis=0)  # (n_window, n_ch)
    real_ptp = np.ptp(sta, axis=0)

    # Shuffles — vectorized circular shifts on IMU indices
    rng = np.random.default_rng(42)
    n_total_imu = n_imu - 2 * half_win  # valid range
    shifts = rng.integers(n_total_imu // 4, 3 * n_total_imu // 4,
                          size=n_shuffles)

    # Compute all shuffle STAs efficiently
    shuffle_ptps = np.empty((n_shuffles, n_ch))
    shuffle_sum = np.zeros((n_window, n_ch))
    shuffle_sum_sq = np.zeros((n_window, n_ch))

    for i in range(n_shuffles):
        # Shift IMU indices (equivalent to circular shift of spikes)
        shifted_imu = (spike_imu + shifts[i]) % n_imu
        valid_sh = (shifted_imu >= half_win) & (shifted_imu < n_imu - half_win)
        sh_imu = shifted_imu[valid_sh]
        if len(sh_imu) == 0:
            shuffle_ptps[i] = 0
            continue
        sh_indices = sh_imu[:, None] + offsets[None, :]
        shuf_sta = imu_signal[sh_indices].mean(axis=0)
        shuffle_ptps[i] = np.ptp(shuf_sta, axis=0)
        shuffle_sum += shuf_sta
        shuffle_sum_sq += shuf_sta ** 2

    shuffle_mean = shuffle_sum / n_shuffles
    shuffle_std = np.sqrt(np.maximum(
        shuffle_sum_sq / n_shuffles - shuffle_mean ** 2, 0))

    p_values = (shuffle_ptps >= real_ptp[None, :]).mean(axis=0)

    return sta, shuffle_mean, shuffle_std, p_values, n_spikes


def run_sta_analysis(cache_dir, good_unit_ids, imu_derived, bank_id,
                     imu_names, window_ms=500, n_shuffles=500,
                     exclude_mag=True):
    """Run STA analysis for all good units on one bank.

    Returns list of dicts with STA results per unit.
    """
    import json

    spikes = np.load(cache_dir / 'sorting' / 'spikes.npy')
    info = json.loads(
        (cache_dir / 'sorting' / 'numpysorting_info.json').read_text())
    all_unit_ids = info['unit_ids']
    fs = info['sampling_frequency']

    imu_signal = imu_derived['signals']  # (n_samples, 15) at 2500 Hz
    imu_fs = imu_derived['sampling_rate']

    # Exclude magnetometer channels (noisy, slow, artifact-prone)
    if exclude_mag:
        keep = [i for i, name in enumerate(imu_names) if 'mag' not in name]
        imu_signal = imu_signal[:, keep]
        imu_names = [imu_names[i] for i in keep]

    # Trim first 5s of IMU (settling artifacts)
    trim_samples = int(5 * imu_fs)
    imu_signal = imu_signal[trim_samples:]
    trim_neural = int(5 * fs)

    results = []
    for uid in good_unit_ids:
        if uid not in all_unit_ids:
            continue
        uid_idx = all_unit_ids.index(uid)
        spike_samples = spikes['sample_index'][spikes['unit_index'] == uid_idx]

        # Trim first 5s of spikes to match IMU trimming
        spike_samples = spike_samples[spike_samples >= trim_neural] - trim_neural

        sta, shuf_mean, shuf_std, p_vals, n_spk = compute_sta_significance(
            spike_samples, imu_signal, imu_fs, fs, window_ms, n_shuffles)

        # Best channel by real STA peak-to-peak
        real_ptp = np.ptp(sta, axis=0)
        best_ch = int(np.argmax(real_ptp))
        best_p = float(p_vals[best_ch])

        results.append({
            'unit_id': uid,
            'bank': bank_id,
            'sta': sta,
            'shuffle_mean': shuf_mean,
            'shuffle_std': shuf_std,
            'p_values': p_vals,
            'n_spikes': n_spk,
            'best_channel': imu_names[best_ch],
            'best_ptp': float(real_ptp[best_ch]),
            'best_p': best_p,
            'sta_imu_names': imu_names,
        })

        if n_spk > 0:
            logger.info("    Unit %d: %d spikes, best=%s ptp=%.3f p=%.4f",
                         uid, n_spk, imu_names[best_ch],
                         real_ptp[best_ch], best_p)

    return results


def plot_sta_bank(sta_results, bank_id, merged_df, fig_dir,
                  window_ms=500, n_top=12):
    """Plot STA results for top units on one bank."""
    plt = _setup_matplotlib()

    BANK_OFFSET_UM = 192 * 20
    imu_fs = 2500
    half_win = int(window_ms / 1000 * imu_fs)
    t_ms = np.arange(-half_win, half_win + 1) / imu_fs * 1000

    # Sort by best_p (ascending = most significant first)
    ranked = sorted(sta_results, key=lambda r: r['best_p'])[:n_top]

    if not ranked:
        return

    n_plots = len(ranked)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots),
                             squeeze=False)

    for idx, r in enumerate(ranked):
        ax = axes[idx, 0]
        uid = r['unit_id']
        sta_names = r['sta_imu_names']
        best_ch_idx = sta_names.index(r['best_channel'])

        sta_trace = r['sta'][:, best_ch_idx]
        shuf_mean = r['shuffle_mean'][:, best_ch_idx]
        shuf_std = r['shuffle_std'][:, best_ch_idx]

        # Shuffle envelope: ±2 SD shaded, mean line
        lower = shuf_mean - 2 * shuf_std
        upper = shuf_mean + 2 * shuf_std
        ax.fill_between(t_ms, lower, upper,
                        color='#E0E0E0', alpha=0.8, label='Shuffle ±2SD',
                        zorder=1)
        ax.plot(t_ms, shuf_mean, color='#9E9E9E', linewidth=0.8, zorder=2)
        # Real STA on top
        ax.plot(t_ms, sta_trace, color='#1565C0', linewidth=2.0,
                label='STA', zorder=3)
        ax.axvline(0, color='red', linewidth=1.0, linestyle='--', alpha=0.7,
                   zorder=4)

        # Depth
        depth_row = merged_df[(merged_df['unit_id'] == uid) &
                              (merged_df['bank'] == bank_id)]
        if len(depth_row) > 0:
            depth_abs = depth_row['depth_um'].values[0] + bank_id * BANK_OFFSET_UM
            depth_str = f"{depth_abs / 1000:.1f}mm"
        else:
            depth_str = "?"

        sig_str = '*' if r['best_p'] < 0.01 else ''
        ax.set_title(f"Bk{bank_id} U{uid} ({depth_str})  {r['best_channel']}{sig_str}  "
                     f"ptp={r['best_ptp']:.3f}  p={r['best_p']:.4f}  "
                     f"n={r['n_spikes']}",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel(r['best_channel'], fontsize=8)
        ax.grid(True, alpha=0.15)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    axes[-1, 0].set_xlabel('Time from spike (ms)')
    fig.suptitle(f'Spike-Triggered Average — Bank {bank_id}', fontsize=13,
                 y=1.002)
    fig.tight_layout()
    fig.savefig(fig_dir / f'sta_bank{bank_id:02d}.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    logger.info("    Saved sta_bank%02d.png", bank_id)


def plot_sta_summary(all_sta_results, merged_df, fig_dir):
    """Summary plot: STA significance across all units and IMU channels."""
    plt = _setup_matplotlib()

    if not all_sta_results:
        return

    # Get channel names from first result
    sta_imu_names = all_sta_results[0]['sta_imu_names']

    # Build summary DataFrame
    rows = []
    for r in all_sta_results:
        for ch_idx, ch_name in enumerate(r['sta_imu_names']):
            ptp = float(np.ptp(r['sta'][:, ch_idx]))
            rows.append({
                'unit_id': r['unit_id'],
                'bank': r['bank'],
                'imu_channel': ch_name,
                'sta_ptp': ptp,
                'p_value': float(r['p_values'][ch_idx]),
                'n_spikes': r['n_spikes'],
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: p-value distribution
    ax = axes[0]
    ax.hist(df['p_value'], bins=50, color='#4472C4', edgecolor='k',
            linewidth=0.3, alpha=0.8)
    ax.axvline(0.01, color='red', linestyle='--', linewidth=1.5,
               label='p=0.01')
    n_sig = (df['p_value'] < 0.01).sum()
    ax.set_xlabel('p-value')
    ax.set_ylabel('Count')
    ax.set_title(f'STA Significance: {n_sig}/{len(df)} pairs (p<0.01)')
    ax.legend()

    # Panel 2: Fraction significant per IMU channel
    ax = axes[1]
    sig_frac = df.groupby('imu_channel').apply(
        lambda x: (x['p_value'] < 0.01).mean()).reindex(sta_imu_names)
    ax.barh(range(len(sta_imu_names)), sig_frac.values, color='#ED7D31')
    ax.set_yticks(range(len(sta_imu_names)))
    ax.set_yticklabels(sta_imu_names, fontsize=8)
    ax.set_xlabel('Fraction significant (p<0.01)')
    ax.set_title('Significant STAs per Channel')
    ax.invert_yaxis()

    # Panel 3: Best STA ptp per unit (across channels)
    ax = axes[2]
    best_per_unit = df.loc[df.groupby(['unit_id', 'bank'])['sta_ptp'].idxmax()]
    ax.scatter(best_per_unit['sta_ptp'], best_per_unit['p_value'],
               s=15, alpha=0.6, c='#4472C4')
    ax.axhline(0.01, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Best STA peak-to-peak')
    ax.set_ylabel('p-value')
    ax.set_title('STA Amplitude vs Significance')
    ax.set_yscale('log')

    fig.suptitle('Spike-Triggered Average Summary', fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / 'sta_summary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved sta_summary.png")

    return df


# ── Population Decoding ──────────────────────────────────────────────

def decode_imu(fr_binned, imu_binned, imu_names, n_folds=5, n_shuffles=10):
    """Decode IMU signals from neural population using Ridge regression.

    Uses TimeSeriesSplit for cross-validation (respects temporal order).
    Includes shuffle control for null distribution.

    Returns dict keyed by IMU channel name, each containing:
      r2_folds, r2_mean, corr_mean, y_pred, y_true, coefs, shuffle_r2_mean
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    n_bins, n_units = fr_binned.shape
    if n_units == 0:
        return {}

    tscv = TimeSeriesSplit(n_splits=n_folds)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    results = {}

    for ch_idx, ch_name in enumerate(imu_names):
        y = imu_binned[:, ch_idx]
        r2_folds = []
        corr_folds = []
        y_pred_all = np.full(n_bins, np.nan)
        coefs_sum = np.zeros(n_units)
        n_coef_folds = 0

        for train_idx, test_idx in tscv.split(fr_binned):
            X_train, X_test = fr_binned[train_idx], fr_binned[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = RidgeCV(alphas=alphas)
            model.fit(X_train_s, y_train)
            y_hat = model.predict(X_test_s)

            ss_res = np.sum((y_test - y_hat) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_folds.append(r2)

            if len(y_test) > 2:
                corr_folds.append(stats.pearsonr(y_test, y_hat)[0])

            y_pred_all[test_idx] = y_hat
            coefs_sum += model.coef_
            n_coef_folds += 1

        # Shuffle control
        shuffle_r2s = []
        rng = np.random.RandomState(42)
        for _ in range(n_shuffles):
            y_shuf = y.copy()
            # Circular shift by random amount (preserves autocorrelation structure)
            shift = rng.randint(n_bins // 4, 3 * n_bins // 4)
            y_shuf = np.roll(y_shuf, shift)

            shuf_r2_folds = []
            for train_idx, test_idx in tscv.split(fr_binned):
                X_train_s = StandardScaler().fit_transform(fr_binned[train_idx])
                X_test_s = StandardScaler().fit(fr_binned[train_idx]).transform(
                    fr_binned[test_idx])
                model = RidgeCV(alphas=alphas)
                model.fit(X_train_s, y_shuf[train_idx])
                y_hat = model.predict(X_test_s)
                ss_res = np.sum((y_shuf[test_idx] - y_hat) ** 2)
                ss_tot = np.sum((y_shuf[test_idx] - np.mean(y_shuf[test_idx])) ** 2)
                shuf_r2_folds.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
            shuffle_r2s.append(np.mean(shuf_r2_folds))

        results[ch_name] = {
            'r2_folds': r2_folds,
            'r2_mean': np.mean(r2_folds),
            'r2_std': np.std(r2_folds),
            'corr_mean': np.mean(corr_folds) if corr_folds else 0.0,
            'y_pred': y_pred_all,
            'y_true': y,
            'coefs': coefs_sum / max(n_coef_folds, 1),
            'shuffle_r2_mean': np.mean(shuffle_r2s),
            'shuffle_r2_std': np.std(shuffle_r2s),
        }

    return results


# ── Plotting ─────────────────────────────────────────────────────────

def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_correlation_heatmap(corr_df, bank_id, merged_df, fig_dir):
    """Heatmap of neurons (by depth) x IMU channels, color=Pearson r."""
    plt = _setup_matplotlib()

    bdf = corr_df[corr_df['bank'] == bank_id]
    if bdf.empty:
        return

    # Get depth ordering from merged_df
    bank_units = merged_df[(merged_df['bank'] == bank_id) &
                           (merged_df['is_good'])].sort_values('depth_um')
    unit_order = bank_units['unit_id'].tolist()

    imu_channels = [n for n in bdf['imu_channel'].unique()]
    n_units = len(unit_order)
    n_ch = len(imu_channels)

    matrix = np.zeros((n_units, n_ch))
    for i, uid in enumerate(unit_order):
        for j, ch in enumerate(imu_channels):
            row = bdf[(bdf['unit_id'] == uid) & (bdf['imu_channel'] == ch)]
            if not row.empty:
                matrix[i, j] = row.iloc[0]['r_at_best_lag']

    # Bonferroni threshold
    n_tests = n_units * n_ch
    p_thresh = 0.05 / max(n_tests, 1)

    fig, ax = plt.subplots(figsize=(14, max(6, n_units * 0.25)))
    vmax = max(0.3, np.percentile(np.abs(matrix), 95))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    # Mark significant correlations
    for i, uid in enumerate(unit_order):
        for j, ch in enumerate(imu_channels):
            row = bdf[(bdf['unit_id'] == uid) & (bdf['imu_channel'] == ch)]
            if not row.empty and row.iloc[0]['pearson_p'] < p_thresh:
                ax.text(j, i, '*', ha='center', va='center',
                        fontsize=8, fontweight='bold')

    depths = bank_units['depth_um'].values + bank_id * 192 * 20
    ax.set_yticks(range(n_units))
    ax.set_yticklabels([f"{d / 1000:.1f}" for d in depths], fontsize=7)
    ax.set_ylabel('Depth (mm)')
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(imu_channels, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'Bank {bank_id}: Neural-IMU Correlation (r at best lag)\n'
                 f'{n_units} good units, * = p < {p_thresh:.1e} (Bonferroni)')
    fig.colorbar(im, ax=ax, label='Pearson r', shrink=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / f'correlation_heatmap_bank{bank_id:02d}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("    Saved correlation_heatmap_bank%02d.png", bank_id)


def plot_top_unit_traces(corr_df, fr_binned_dict, imu_binned_dict,
                        t_bins_dict, unit_ids_dict, imu_names, merged_df,
                        fig_dir, n_top=16, output_dir=None):
    """Plot multi-channel waveform + dual-axis FR vs IMU for top units.

    Each row: [multi-channel template traces at spatial positions | FR vs IMU with dual y-axes].
    """
    plt = _setup_matplotlib()
    import matplotlib.gridspec as gridspec
    import json

    BANK_OFFSET_UM = 192 * 20  # µm per bank

    # Find top units by |r_at_best_lag|
    top = corr_df.loc[corr_df['r_at_best_lag'].abs().nlargest(n_top * 3).index]
    top = top.sort_values('r_at_best_lag', key=abs, ascending=False)
    top = top.drop_duplicates(subset=['unit_id', 'bank']).head(n_top)

    n_plots = len(top)
    fig = plt.figure(figsize=(22, 3.0 * n_plots))
    outer = gridspec.GridSpec(n_plots, 2, width_ratios=[1, 3.5], wspace=0.3,
                              hspace=0.5)

    color_imu = '#E53935'
    color_fr = '#1565C0'

    for idx, (_, row) in enumerate(top.iterrows()):
        bank_id = int(row['bank'])
        uid = int(row['unit_id'])
        ch_name = row['imu_channel']
        r_val = row['r_at_best_lag']
        lag_ms = int(row['best_lag_ms'])

        fr = fr_binned_dict.get(bank_id)
        imu = imu_binned_dict.get(bank_id)
        t = t_bins_dict.get(bank_id)
        uids = unit_ids_dict.get(bank_id)

        if fr is None or uid not in uids:
            continue
        u_idx = uids.index(uid)
        ch_idx = imu_names.index(ch_name)

        # Absolute depth
        depth_local = merged_df[(merged_df['unit_id'] == uid) &
                                (merged_df['bank'] == bank_id)]['depth_um'].values
        if len(depth_local) > 0:
            depth_abs = depth_local[0] + bank_id * BANK_OFFSET_UM
            depth_str = f"{depth_abs / 1000:.1f}mm"
        else:
            depth_str = "?"

        # ── Left panel: multi-channel template traces at spatial positions ──
        ax_wf = fig.add_subplot(outer[idx, 0])
        cache_dir = output_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
        tmpl_path = cache_dir / 'extensions' / 'templates' / 'average.npy'
        if tmpl_path.exists():
            templates = np.load(tmpl_path)

            # Load channel positions from probegroup
            pg_path = cache_dir / 'recording_info' / 'probegroup.json'
            ch_pos = None
            if pg_path.exists():
                pg = json.loads(pg_path.read_text())
                ch_pos = np.array(pg['probes'][0]['contact_positions'])

            info_path = cache_dir / 'sorting' / 'numpysorting_info.json'
            if info_path.exists():
                info = json.loads(info_path.read_text())
                all_uids = info.get('unit_ids', [])
            else:
                all_uids = list(range(templates.shape[0]))

            if uid in all_uids and ch_pos is not None:
                tmpl_idx = all_uids.index(uid)
                template = templates[tmpl_idx]  # (n_time, n_ch)
                ptp_per_ch = np.ptp(template, axis=0)
                peak_ch = np.argmax(ptp_per_ch)
                max_ptp = ptp_per_ch[peak_ch]
                peak_y = ch_pos[peak_ch, 1]

                # Sparse channels: those with >15% of peak amplitude
                sparse_ch = np.where(ptp_per_ch > 0.15 * max_ptp)[0]

                # Plot template traces at spatial positions (same style as single neuron plots)
                x_unique = np.sort(np.unique(ch_pos[sparse_ch, 0]))
                sorted_depths = np.sort(np.unique(ch_pos[sparse_ch, 1]))
                col_gap = (x_unique[-1] - x_unique[0]) if len(x_unique) > 1 else 80
                row_gap = (np.min(np.diff(sorted_depths))
                           if len(sorted_depths) > 1 else 20)

                n_t = template.shape[0]
                t_wf = np.linspace(-0.5, 0.5, n_t)
                x_scale = col_gap * 0.45
                y_scale = row_gap * 2.5 / max_ptp if max_ptp > 0 else 1

                for ch in sparse_ch:
                    cx, cy = ch_pos[ch]
                    trace = template[:, ch]
                    x_plot = cx + t_wf * x_scale
                    y_plot = cy + trace * y_scale
                    is_peak = (ch == peak_ch)
                    ax_wf.plot(x_plot, y_plot,
                               color='blue' if is_peak else 'grey',
                               lw=1.5 if is_peak else 0.5,
                               alpha=1.0 if is_peak else 0.5)

                ax_wf.set_xlim(x_unique[0] - col_gap * 0.55,
                               x_unique[-1] + col_gap * 0.55)
                ax_wf.set_ylim(peak_y - 400, peak_y + 400)
                ax_wf.set_ylabel('Depth (µm)', fontsize=7)
                ax_wf.set_title(f'Bk{bank_id} U{uid}  {max_ptp:.0f}µV',
                                fontsize=8, fontweight='bold')
            else:
                ax_wf.text(0.5, 0.5, 'N/A', transform=ax_wf.transAxes,
                           ha='center', va='center', fontsize=9, color='gray')
        else:
            ax_wf.text(0.5, 0.5, 'N/A', transform=ax_wf.transAxes,
                       ha='center', va='center', fontsize=9, color='gray')
        ax_wf.tick_params(labelsize=6)

        # ── Right panel: dual-axis FR vs IMU ──
        ax_fr = fig.add_subplot(outer[idx, 1])
        imu_sig = imu[:, ch_idx]
        fr_sig = fr[:, u_idx]

        # Invert IMU if negative correlation so traces visually match
        if r_val < 0:
            imu_plot = -imu_sig
            imu_label = f'-{ch_name}'
        else:
            imu_plot = imu_sig
            imu_label = ch_name

        # Plot FR on left axis
        ax_fr.plot(t, fr_sig, color=color_fr, linewidth=0.8, alpha=0.85)
        ax_fr.set_ylabel('FR (Hz)', color=color_fr, fontsize=8)
        ax_fr.tick_params(axis='y', labelcolor=color_fr, labelsize=7)
        ax_fr.tick_params(axis='x', labelsize=7)

        # Plot IMU on right axis (inverted if negative r)
        ax_imu = ax_fr.twinx()
        ax_imu.plot(t, imu_plot, color=color_imu, linewidth=0.8, alpha=0.75)
        ax_imu.set_ylabel(imu_label, color=color_imu, fontsize=8)
        ax_imu.tick_params(axis='y', labelcolor=color_imu, labelsize=7)

        sign = '+' if r_val > 0 else ''
        inv_str = ' (inv)' if r_val < 0 else ''
        lag_str = f'  lag={lag_ms}ms' if lag_ms != 0 else ''
        ax_fr.set_title(f'Bk{bank_id} U{uid} ({depth_str})  vs {ch_name}{inv_str}  '
                        f'r={sign}{r_val:.3f}{lag_str}',
                        fontsize=9, fontweight='bold', pad=3)
        if idx == n_plots - 1:
            ax_fr.set_xlabel('Time (s)', fontsize=8)
        ax_fr.grid(True, alpha=0.15)

    fig.suptitle('Top Correlated Neurons: Multi-Channel Waveform + FR (blue) vs IMU (red)',
                 fontsize=13, y=1.002)
    fig.savefig(fig_dir / 'top_unit_traces.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved top_unit_traces.png")


def plot_correlation_summary(corr_df, fig_dir):
    """Summary across all banks: max |r| distributions, depth dependence."""
    plt = _setup_matplotlib()

    # Per-neuron max |r|
    max_r = corr_df.groupby(['unit_id', 'bank']).agg(
        max_abs_r=('r_at_best_lag', lambda x: x.abs().max()),
        best_channel=('r_at_best_lag', lambda x: corr_df.loc[x.abs().idxmax(), 'imu_channel']),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histogram of max |r|
    ax = axes[0]
    ax.hist(max_r['max_abs_r'], bins=30, color='steelblue', edgecolor='white')
    ax.axvline(0.1, color='red', linestyle='--', linewidth=1, label='r=0.1')
    ax.set_xlabel('Max |r| across IMU channels')
    ax.set_ylabel('Number of neurons')
    ax.set_title('Distribution of Max Correlation')
    ax.legend()

    # Best channel distribution
    ax = axes[1]
    ch_counts = max_r['best_channel'].value_counts()
    ax.barh(range(len(ch_counts)), ch_counts.values, color='steelblue')
    ax.set_yticks(range(len(ch_counts)))
    ax.set_yticklabels(ch_counts.index, fontsize=8)
    ax.set_xlabel('Number of neurons')
    ax.set_title('Most-Correlated IMU Channel')

    # Per-bank fraction significant
    ax = axes[2]
    n_tests_per_unit = corr_df['imu_channel'].nunique()
    corr_df['significant'] = corr_df['pearson_p'] < 0.05 / n_tests_per_unit
    sig_per_bank = corr_df.groupby('bank')['significant'].mean()
    ax.bar(sig_per_bank.index, sig_per_bank.values * 100, color='steelblue')
    ax.set_xlabel('Bank')
    ax.set_ylabel('% significant correlations')
    ax.set_title('Significant Correlations per Bank')

    fig.suptitle(f'Neural-IMU Correlation Summary ({len(max_r)} neurons)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / 'correlation_summary.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved correlation_summary.png")


def plot_decoding_performance(all_decode_results, fig_dir):
    """Bar chart of R² per IMU channel per bank + shuffle baseline."""
    plt = _setup_matplotlib()

    # Collect R² values
    rows = []
    for bank_id, decode_results in all_decode_results.items():
        for ch_name, res in decode_results.items():
            rows.append({
                'bank': bank_id,
                'imu_channel': ch_name,
                'r2': res['r2_mean'],
                'r2_std': res['r2_std'],
                'shuffle_r2': res['shuffle_r2_mean'],
            })
    df = pd.DataFrame(rows)

    if df.empty:
        return

    # Top 6 IMU channels by mean R² across banks
    mean_r2 = df.groupby('imu_channel')['r2'].mean().sort_values(ascending=False)
    top_channels = mean_r2.head(6).index.tolist()
    df_top = df[df['imu_channel'].isin(top_channels)]

    banks = sorted(df_top['bank'].unique())
    n_banks = len(banks)
    n_ch = len(top_channels)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_ch)
    width = 0.8 / n_banks

    for i, bank_id in enumerate(banks):
        bdf = df_top[df_top['bank'] == bank_id]
        r2_vals = [bdf[bdf['imu_channel'] == ch]['r2'].values[0]
                   if ch in bdf['imu_channel'].values else 0
                   for ch in top_channels]
        ax.bar(x + i * width, r2_vals, width, label=f'Bank {bank_id}',
               alpha=0.8)

    # Shuffle baseline
    shuf_vals = [df_top[df_top['imu_channel'] == ch]['shuffle_r2'].mean()
                 for ch in top_channels]
    ax.plot(x + width * n_banks / 2, shuf_vals, 'k--', linewidth=2,
            label='Shuffle control', marker='_', markersize=15)

    ax.set_xticks(x + width * n_banks / 2)
    ax.set_xticklabels(top_channels, rotation=30, ha='right')
    ax.set_ylabel('R² (cross-validated)')
    ax.set_title('Population Decoding Performance (Top 6 IMU Channels)')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(fig_dir / 'decoding_r2_summary.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved decoding_r2_summary.png")


def plot_decoded_traces(decode_results, bank_id, t_bins, fig_dir):
    """Overlay predicted vs actual for top 3 decoded IMU channels."""
    plt = _setup_matplotlib()

    if not decode_results:
        return

    # Top 3 by R²
    sorted_ch = sorted(decode_results.items(), key=lambda x: x[1]['r2_mean'],
                       reverse=True)[:3]

    fig, axes = plt.subplots(len(sorted_ch), 1, figsize=(16, 3 * len(sorted_ch)),
                             sharex=True)
    if len(sorted_ch) == 1:
        axes = [axes]

    for ax, (ch_name, res) in zip(axes, sorted_ch):
        valid = ~np.isnan(res['y_pred'])
        ax.plot(t_bins[valid], res['y_true'][valid], 'b-', linewidth=0.8,
                alpha=0.7, label='Actual')
        ax.plot(t_bins[valid], res['y_pred'][valid], 'r-', linewidth=0.8,
                alpha=0.7, label='Predicted')
        ax.set_ylabel(ch_name)
        ax.set_title(f'{ch_name}: R²={res["r2_mean"]:.3f} '
                     f'(shuffle={res["shuffle_r2_mean"]:.3f})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Bank {bank_id}: Decoded vs Actual IMU', fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / f'decoded_traces_bank{bank_id:02d}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("    Saved decoded_traces_bank%02d.png", bank_id)


# ── Save Results ─────────────────────────────────────────────────────

def save_results(corr_df, all_decode_results, output_dir, subdir='imu_analysis'):
    """Save correlation and decoding results to CSV/NPY."""
    out_dir = output_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_df.to_csv(out_dir / 'correlations.csv', index=False)
    logger.info("  Saved correlations.csv (%d rows)", len(corr_df))

    # Decoding summary
    rows = []
    for bank_id, decode_results in all_decode_results.items():
        for ch_name, res in decode_results.items():
            rows.append({
                'bank': bank_id,
                'imu_channel': ch_name,
                'r2_mean': round(res['r2_mean'], 6),
                'r2_std': round(res['r2_std'], 6),
                'corr_mean': round(res['corr_mean'], 6),
                'shuffle_r2_mean': round(res['shuffle_r2_mean'], 6),
                'shuffle_r2_std': round(res['shuffle_r2_std'], 6),
            })
        # Save coefficients
        first_ch = list(decode_results.values())[0]
        coef_matrix = np.column_stack(
            [res['coefs'] for res in decode_results.values()])
        np.save(out_dir / f'decoding_coefs_bank{bank_id:02d}.npy', coef_matrix)

    pd.DataFrame(rows).to_csv(out_dir / 'decoding_summary.csv', index=False)
    logger.info("  Saved decoding_summary.csv")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Correlate neural activity with IMU and decode')
    parser.add_argument('data_dir', help='Directory containing .dat segments')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--fig-dir', type=str, default=None)
    parser.add_argument('--bank', type=int, nargs='+', default=None)
    parser.add_argument('--bin-ms', type=int, default=100)
    parser.add_argument('--smooth-ms', type=int, default=50)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--raw-imu', action='store_true',
                        help='Skip IMU unit conversion (use raw int16)')
    parser.add_argument('--no-sta', action='store_true',
                        help='Skip spike-triggered average analysis')
    parser.add_argument('--sta-shuffles', type=int, default=500,
                        help='Number of shuffles for STA significance')
    parser.add_argument('--neuron-set', choices=['legacy', 'classifier',
                        'classifier-strict'], default='legacy',
                        help="Which units to analyze: 'legacy' (is_good "
                              "tier), 'classifier' (custom_p_neural>=0.5), "
                              "'classifier-strict' (>=0.8)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_root = Path(os.environ.get('DATA_ROOT', 'E:/'))
    output_dir = Path(args.output) if args.output else (
        data_root / 'sorting_output' / data_dir.name)
    suffix = '' if args.neuron_set == 'legacy' else (
        '_classifier' if args.neuron_set == 'classifier' else '_classifier_strict')
    fig_dir = Path(args.fig_dir) if args.fig_dir else (
        data_root / 'figures' / 'sorting' / data_dir.name
        / f'imu_analysis{suffix}')
    fig_dir.mkdir(parents=True, exist_ok=True)
    bank_filter = set(args.bank) if args.bank else None

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("NEURAL-IMU ANALYSIS: %s", data_dir.name)
    logger.info("=" * 70)
    logger.info("Bin size: %d ms, Smooth: %d ms, CV folds: %d",
                 args.bin_ms, args.smooth_ms, args.n_folds)
    if not args.raw_imu:
        logger.warning("Using assumed ICM-20948 IMU calibration — "
                       "physical units may be approximate")

    # Load merged units
    merged_df = pd.read_csv(output_dir / 'merged' / 'merged_units.csv')
    if args.neuron_set != 'legacy':
        col = ('custom_neural' if args.neuron_set == 'classifier'
               else 'custom_neural_strict')
        if col not in merged_df.columns:
            logger.warning("merged_units.csv has no %s column; falling back "
                           "to legacy is_good", col)
        else:
            merged_df['is_good'] = merged_df[col].astype(bool)
            logger.info("Neuron set: %s (%s) — %d units",
                         args.neuron_set, col, int(merged_df['is_good'].sum()))
    good_df = merged_df[merged_df['is_good']]
    logger.info("Good units: %d across %d banks",
                 len(good_df), good_df['bank'].nunique())

    # Extract IMU per segment
    logger.info("\n[1/4] EXTRACTING IMU per segment")
    logger.info("=" * 70)
    imu_segments = load_per_segment_imu(data_dir, output_dir)

    all_corr_dfs = []
    all_decode_results = {}
    all_sta_results = []
    fr_binned_dict = {}
    imu_binned_dict = {}
    t_bins_dict = {}
    unit_ids_dict = {}
    imu_names_out = None

    banks = sorted(good_df['bank'].unique())
    if bank_filter:
        banks = [b for b in banks if b in bank_filter]

    for bank_id in banks:
        logger.info("\n[2/4] BANK %d — correlation & decoding", bank_id)
        logger.info("=" * 70)

        if bank_id not in imu_segments:
            logger.warning("  No IMU data for bank %d, skipping", bank_id)
            continue

        # Process IMU
        imu_raw = imu_segments[bank_id]

        # Check movement
        gyro_var = np.var(imu_raw['gyro'].astype(float))
        accel_var = np.var(imu_raw['accel'].astype(float))
        logger.info("  IMU variance — accel: %.1f, gyro: %.1f",
                     accel_var, gyro_var)
        if gyro_var < 10 and accel_var < 10:
            logger.warning("  Very low IMU variance — animal may not be moving")

        if args.raw_imu:
            imu_phys = {
                'accel': imu_raw['accel'].astype(np.float64),
                'gyro': imu_raw['gyro'].astype(np.float64),
                'mag': imu_raw['mag'].astype(np.float64),
                'sampling_rate': imu_raw['sampling_rate'],
            }
        else:
            imu_phys = convert_imu_to_physical(imu_raw)

        imu_derived = compute_derived_imu(imu_phys)
        imu_binned, t_bins, imu_names = bin_imu_signals(imu_derived,
                                                         args.bin_ms)

        # Bin spike trains
        cache_dir = output_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
        bank_good_ids = good_df[good_df['bank'] == bank_id]['unit_id'].tolist()
        duration_s = len(imu_raw['accel']) / imu_raw['sampling_rate']

        fr_binned, unit_ids_used = bin_spike_trains(
            cache_dir, bank_good_ids, args.bin_ms, args.smooth_ms, duration_s)

        if fr_binned.shape[1] == 0:
            logger.warning("  No good units for bank %d", bank_id)
            continue

        # Align lengths and trim first 5s (IMU settling artifacts)
        n_bins = min(fr_binned.shape[0], imu_binned.shape[0])
        trim_bins = int(5000 / args.bin_ms)  # 5 seconds
        fr_binned = fr_binned[trim_bins:n_bins]
        imu_binned_bank = imu_binned[trim_bins:n_bins]
        t_bins_bank = t_bins[trim_bins:n_bins]

        logger.info("  %d good units, %d time bins (%.1fs)",
                     fr_binned.shape[1], n_bins, n_bins * args.bin_ms / 1000)

        # Store for top-unit trace plots
        fr_binned_dict[bank_id] = fr_binned
        imu_binned_dict[bank_id] = imu_binned_bank
        t_bins_dict[bank_id] = t_bins_bank
        unit_ids_dict[bank_id] = unit_ids_used
        imu_names_out = imu_names

        # Correlations
        logger.info("  Computing correlations...")
        corr_df = compute_correlations(
            fr_binned, imu_binned_bank, unit_ids_used, imu_names,
            bank_id, args.bin_ms)
        all_corr_dfs.append(corr_df)

        # Report top correlations
        top = corr_df.loc[corr_df['r_at_best_lag'].abs().nlargest(3).index]
        for _, row in top.iterrows():
            logger.info("    Unit %d × %s: r=%.3f (lag=%dms)",
                         row['unit_id'], row['imu_channel'],
                         row['r_at_best_lag'], row['best_lag_ms'])

        # Per-bank figures
        plot_correlation_heatmap(corr_df, bank_id, merged_df, fig_dir)

        # STA analysis
        if not args.no_sta:
            logger.info("  Computing spike-triggered averages (%d shuffles)...",
                         args.sta_shuffles)
            cache_dir = output_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
            bank_good_ids = good_df[good_df['bank'] == bank_id]['unit_id'].tolist()
            sta_results = run_sta_analysis(
                cache_dir, bank_good_ids, imu_derived,
                bank_id, imu_names, window_ms=500,
                n_shuffles=args.sta_shuffles)
            all_sta_results.extend(sta_results)
            plot_sta_bank(sta_results, bank_id, merged_df, fig_dir)

    # Merge and save
    logger.info("\n[3/4] SUMMARY & FIGURES")
    logger.info("=" * 70)

    if all_corr_dfs:
        all_corr = pd.concat(all_corr_dfs, ignore_index=True)
        plot_correlation_summary(all_corr, fig_dir)
        plot_top_unit_traces(all_corr, fr_binned_dict, imu_binned_dict,
                             t_bins_dict, unit_ids_dict, imu_names_out,
                             merged_df, fig_dir, output_dir=output_dir)

        # STA summary
        if all_sta_results and not args.no_sta:
            sta_df = plot_sta_summary(all_sta_results, merged_df,
                                      fig_dir)
            if sta_df is not None:
                sta_out = output_dir / f'imu_analysis{suffix}'
                sta_out.mkdir(parents=True, exist_ok=True)
                sta_df.to_csv(sta_out / 'sta_results.csv', index=False)
                logger.info("  Saved sta_results.csv")

        logger.info("\n[4/4] SAVING RESULTS")
        logger.info("=" * 70)
        save_results(all_corr, {}, output_dir, subdir=f'imu_analysis{suffix}')
    else:
        all_corr = pd.DataFrame()
        logger.warning("  No results to save")

    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE in %.1f min", elapsed / 60)
    logger.info("Banks analyzed: %d", len(fr_binned_dict))
    if not all_corr.empty:
        n_sig = (all_corr['pearson_p'] < 0.05 / all_corr['imu_channel'].nunique()).sum()
        logger.info("Significant correlations: %d/%d (%.1f%%)",
                     n_sig, len(all_corr), 100 * n_sig / len(all_corr))
    logger.info("Output: %s", output_dir / 'imu_analysis')
    logger.info("Figures: %s", fig_dir)


if __name__ == '__main__':
    main()
