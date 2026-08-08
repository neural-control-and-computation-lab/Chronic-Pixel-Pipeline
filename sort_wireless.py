"""
Sort a multi-segment SpikeGadgets wireless .dat recording.

Each .dat segment corresponds to a different bank of the NP1032 probe.
Segments are sorted independently per bank (like sort_survey.py), then
merged into a full-probe summary.

Pipeline per bank:
  1. Load .dat segment, detect bank from header, apply probe geometry
  2. Preprocess (bandpass 300-6000 Hz, phase shift)
  3. Spike sort with KS4 (nblocks=1 for ~4 min segments)
  4. Build SortingAnalyzer, compute quality metrics

Global:
  5. Extract IMU data across all segments
  6. Merge all bank results into full-probe summary
  7. Precompute waveform quality flags
  8. Generate per-unit figures (good vs suspect)

Usage:
  python sort_wireless.py E:/March23_wireless
  python sort_wireless.py E:/March23_wireless --skip-imu
  python sort_wireless.py E:/March23_wireless --bank 0 2 5  # specific banks only
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import sys
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _detect_torch_device():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA device detected: %s", name)
            return 'cuda'
    except ImportError:
        pass
    logger.info("No CUDA — using CPU")
    return 'cpu'


# ── Phase 1: Discover segments ───────────────────────────────────────

def discover_segments(data_dir):
    """Find all .dat segments and detect their bank numbers.

    Returns list of dicts: {path, bank, n_channels, has_imu}
    sorted by bank number.
    """
    from chronic_pixel_pipeline.packet import detect_format

    data_dir = Path(data_dir)
    dat_files = sorted(data_dir.glob('*.dat'), key=lambda p: p.name.lower())

    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {data_dir}")

    segments = []
    for f in dat_files:
        bank, n_channels, packet_size, has_imu = detect_format(str(f))
        segments.append({
            'path': f,
            'bank': bank,
            'n_channels': n_channels,
            'has_imu': has_imu,
        })
        logger.info("  %s: bank %d, %d ch, IMU=%s",
                     f.name, bank, n_channels, has_imu)

    segments.sort(key=lambda s: s['bank'])
    return segments


# ── Phase 2: IMU extraction ──────────────────────────────────────────

def extract_imu(segments, output_dir):
    """Extract IMU data from all segments and concatenate."""
    import chronic_pixel_pipeline as cpp

    output_dir = Path(output_dir)
    all_accel, all_gyro, all_mag = [], [], []

    for seg in segments:
        logger.info("  Bank %d: extracting IMU from %s...",
                     seg['bank'], seg['path'].name)
        try:
            imu = cpp.load_imu(str(seg['path']))
            all_accel.append(imu['accel'])
            all_gyro.append(imu['gyro'])
            all_mag.append(imu['mag'])
            logger.info("    %d samples (%.1fs @ %.0f Hz)",
                         len(imu['accel']),
                         len(imu['accel']) / imu['sampling_rate'],
                         imu['sampling_rate'])
        except Exception as e:
            logger.warning("    IMU extraction failed: %s", e)

    if not all_accel:
        logger.warning("  No IMU data extracted!")
        return None

    imu_data = {
        'accel': np.concatenate(all_accel, axis=0),
        'gyro': np.concatenate(all_gyro, axis=0),
        'mag': np.concatenate(all_mag, axis=0),
        'sampling_rate': 2500.0,
    }

    imu_dir = output_dir / 'imu'
    imu_dir.mkdir(parents=True, exist_ok=True)
    np.save(imu_dir / 'accel.npy', imu_data['accel'])
    np.save(imu_dir / 'gyro.npy', imu_data['gyro'])
    np.save(imu_dir / 'mag.npy', imu_data['mag'])

    total_dur = len(imu_data['accel']) / imu_data['sampling_rate']
    logger.info("  IMU total: %.1fs, saved to %s", total_dur, imu_dir)

    return imu_data


def plot_imu(imu_data, fig_dir):
    """Generate IMU diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fs = imu_data['sampling_rate']
    n_samples = min(len(imu_data['accel']), len(imu_data['gyro']),
                    len(imu_data['mag']))
    t = np.arange(n_samples) / fs

    # Timeseries
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for ax, key, title in zip(axes, ['accel', 'gyro', 'mag'],
                               ['Accelerometer', 'Gyroscope', 'Magnetometer']):
        for i, label in enumerate(['X', 'Y', 'Z']):
            ax.plot(t, imu_data[key][:n_samples, i], linewidth=0.5, label=label)
        ax.set_ylabel(f'{title} (raw)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'IMU Data — {t[-1]:.0f}s @ {fs:.0f} Hz', fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / 'imu_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved imu_timeseries.png")

    # Gyro spectrogram
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = axes[i]
        f, t_spec, Sxx = spectrogram(
            imu_data['gyro'][:n_samples, i].astype(float),
            fs=fs, nperseg=2048, noverlap=1024)
        ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                       shading='gouraud', cmap='viridis', vmin=-20, vmax=40)
        ax.set_ylabel(f'Gyro {label}\nFreq (Hz)')
        ax.set_ylim(0, 200)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Gyroscope Spectrogram', fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / 'gyro_spectrogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved gyro_spectrogram.png")


# ── Phase 3: Per-bank sorting ────────────────────────────────────────

def sort_one_bank(seg, trodesconf_path, output_dir, torch_device, force=False,
                   notch=True):
    """Load, preprocess, and sort one bank segment.

    Returns (sorting, rec_pp, bank_id, n_units) or (None, None, bank_id, 0)
    on failure.
    """
    import spikeinterface.full as si
    import chronic_pixel_pipeline as cpp
    from chronic_pixel_pipeline.preprocessing import preprocess
    from chronic_pixel_pipeline.probe_geometry import (
        make_probe_for_bank, make_probe_from_trodesconf,
    )

    bank_id = seg['bank']
    dat_path = seg['path']
    sort_dir = output_dir / f'bank{bank_id:02d}_ks4'

    logger.info("\n" + "=" * 70)
    logger.info("BANK %d — %s", bank_id, dat_path.name)
    logger.info("=" * 70)

    # Load
    rec = cpp.load_segment(str(dat_path), attach_probe=False)
    canonical_ids = [str(c) for c in range(rec.get_num_channels())]
    rec = rec.rename_channels(new_channel_ids=canonical_ids)

    # Attach probe geometry
    if trodesconf_path:
        probe = make_probe_from_trodesconf(trodesconf_path, None,
                                            rec.get_num_channels())
    else:
        probe = make_probe_for_bank(bank_id, rec.get_num_channels(), 'NP1032')
    rec.set_probe(probe, in_place=True)

    dur = rec.get_total_duration()
    logger.info("  %d channels, %.1fs (%.1f min)", rec.get_num_channels(),
                 dur, dur / 60)

    # Preprocess
    # Wireless-specific: notch out SpikeGadgets transmitter clock noise at
    # 1700 Hz (and its 2nd harmonic at 3400 Hz). The noise has a wide hump
    # (~150 Hz FWHM in raw, broader at the shoulder) so use Q=10 to cover
    # 1500-1900 Hz fully. CMR helps further. Pass --no-notch to skip.
    pp_kwargs = dict(freq_min=300, freq_max=6000, destripe=False,
                      cmr=True, phase_shift=True)
    if notch:
        pp_kwargs['notch_freqs'] = [1700, 3400]
        pp_kwargs['notch_q'] = 10
        logger.info("  Preprocessing with Q=10 notch at 1700+3400 Hz")
    else:
        logger.info("  Preprocessing WITHOUT notch (raw bandpass + CMR only)")
    rec_pp = preprocess(rec, **pp_kwargs)

    # KS4 params
    ks_kwargs = dict(
        verbose=False,
        torch_device=torch_device,
        nblocks=1,
        templates_from_data=False,
        batch_size=30000,
        dminx=103,
        max_channel_distance=150,
        nearest_chans=20,
    )

    # Resume check
    existing = sort_dir / 'sorter_output'
    if not force and existing.exists() and (existing / 'spike_clusters.npy').exists():
        logger.info("  Resuming — loading existing KS4 output")
        try:
            sorting = si.read_sorter_folder(sort_dir)
            n_units = len(sorting.unit_ids)
            total_spikes = sum(sorting.count_num_spikes_per_unit().values())
            logger.info("  Loaded: %d units, %d spikes", n_units, total_spikes)
            return sorting, rec_pp, bank_id, n_units
        except Exception as e:
            logger.warning("  Failed to load: %s. Re-sorting...", e)

    # Sort
    logger.info("  Running KS4 (nblocks=1, %d ch, %.1fs)...",
                 rec_pp.get_num_channels(), dur)
    t0 = time.time()
    try:
        sorting = si.run_sorter(
            'kilosort4', rec_pp, folder=sort_dir,
            remove_existing_folder=True, **ks_kwargs)
        elapsed = time.time() - t0
        n_units = len(sorting.unit_ids)
        total_spikes = sum(sorting.count_num_spikes_per_unit().values())
        logger.info("  KS4 complete in %.0fs: %d units, %d spikes",
                     elapsed, n_units, total_spikes)
    except Exception as e:
        err_str = str(e)
        is_template_error = (
            'n_samples=' in err_str and 'n_clusters=' in err_str
        ) or (
            'Found array with 0 sample' in err_str
        ) or (
            'extract_wPCA_wTEMP' in err_str
        )
        if is_template_error:
            logger.warning("  Template extraction failed, retrying with "
                          "templates_from_data=False...")
            ks_fallback = {**ks_kwargs, 'templates_from_data': False}
            try:
                sorting = si.run_sorter(
                    'kilosort4', rec_pp, folder=sort_dir,
                    remove_existing_folder=True, **ks_fallback)
                n_units = len(sorting.unit_ids)
                logger.info("  Retry succeeded: %d units", n_units)
            except Exception as e2:
                logger.error("  KS4 retry also failed: %s", e2)
                return None, rec_pp, bank_id, 0
        else:
            logger.error("  KS4 failed: %s", e)
            return None, rec_pp, bank_id, 0

    return sorting, rec_pp, bank_id, n_units


# ── Phase 4: Per-bank analysis ───────────────────────────────────────

def analyze_one_bank(sorting, recording, bank_id, output_dir):
    """Build SortingAnalyzer and quality metrics for one bank.

    Returns analyzer or None on failure.
    """
    from chronic_pixel_pipeline.analysis import build_sorting_analyzer

    cache_dir = output_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'

    logger.info("  Bank %d: building analyzer (%d units)...",
                 bank_id, len(sorting.unit_ids))
    t0 = time.time()
    try:
        analyzer = build_sorting_analyzer(sorting, recording,
                                          cache_dir=cache_dir)
        logger.info("  Bank %d: analyzer built in %.0fs",
                     bank_id, time.time() - t0)
        return analyzer
    except Exception as e:
        logger.error("  Bank %d: analyzer failed: %s", bank_id, e)
        return None


# ── Phase 5: Merge + quality + figures ────────────────────────────────

def merge_and_classify(bank_results, output_dir, fig_dir, skip_figures=False):
    """Merge all bank results, compute waveform quality, generate figures."""
    from chronic_pixel_pipeline.analysis import classify_units
    from precompute_waveform_quality import compute_bank_metrics

    output_dir = Path(output_dir)
    fig_dir = Path(fig_dir)

    # Precompute waveform quality per bank
    logger.info("\n  Computing waveform quality metrics...")
    for bank_id, sorting, rec_pp, analyzer in bank_results:
        if analyzer is None:
            continue
        cache_dir = output_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
        try:
            wq_df = compute_bank_metrics(cache_dir)
            wq_dir = cache_dir / 'extensions' / 'waveform_quality'
            wq_dir.mkdir(parents=True, exist_ok=True)
            wq_df.to_csv(wq_dir / 'metrics.csv')
            n_flagged = int(wq_df['boundary_noise_flag'].sum())
            logger.info("    Bank %d: %d units, %d flagged",
                         bank_id, len(wq_df), n_flagged)
        except Exception as e:
            logger.warning("    Bank %d: waveform quality failed: %s",
                           bank_id, e)

    # Build merged unit table
    logger.info("\n  Building merged unit table...")
    all_rows = []
    global_unit_idx = 0
    for bank_id, sorting, rec_pp, analyzer in bank_results:
        if analyzer is None:
            continue

        good_ids, snr_thresh = classify_units(analyzer)
        templates = analyzer.get_extension('templates').get_data()
        ch_locs = analyzer.get_channel_locations()

        # Load waveform quality
        wq_path = (output_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
                    / 'extensions' / 'waveform_quality' / 'metrics.csv')
        wq_df = pd.read_csv(wq_path, index_col=0) if wq_path.exists() else None

        for uid_idx, uid in enumerate(sorting.unit_ids):
            template = templates[uid_idx]
            ptp_per_ch = np.ptp(template, axis=0)
            peak_ch = int(np.argmax(ptp_per_ch))
            depth = float(ch_locs[peak_ch, 1])
            amplitude = float(ptp_per_ch[peak_ch])
            spike_train = sorting.get_unit_spike_train(uid, segment_index=0)
            fr = len(spike_train) / rec_pp.get_total_duration()
            is_baseline = uid in good_ids
            is_noise = (bool(wq_df.loc[uid, 'boundary_noise_flag'])
                        if wq_df is not None and uid in wq_df.index else False)

            all_rows.append({
                'global_unit_idx': global_unit_idx,
                'unit_id': uid,
                'bank': bank_id,
                'depth_um': round(depth, 1),
                'amplitude_uv': round(amplitude, 2),
                'firing_rate': round(fr, 4),
                'n_spikes': len(spike_train),
                'is_baseline_good': is_baseline,
                'boundary_noise_flag': is_noise,
                'is_good': is_baseline and not is_noise,
            })
            global_unit_idx += 1

    merged_df = pd.DataFrame(all_rows)
    merged_dir = output_dir / 'merged'
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(merged_dir / 'merged_units.csv', index=False)

    n_total = len(merged_df)
    n_good = merged_df['is_good'].sum()
    n_baseline = merged_df['is_baseline_good'].sum()
    logger.info("  Merged: %d total, %d baseline, %d strict good",
                 n_total, n_baseline, n_good)
    for bank_id in sorted(merged_df['bank'].unique()):
        bdf = merged_df[merged_df['bank'] == bank_id]
        logger.info("    Bank %d: %d units (%d good)",
                     bank_id, len(bdf), bdf['is_good'].sum())

    # Generate figures
    if not skip_figures:
        logger.info("\n  Generating per-unit figures...")
        _generate_unit_figures(bank_results, merged_df, fig_dir)

    return merged_df


def _generate_unit_figures(bank_results, merged_df, fig_dir):
    """Generate per-unit summary figures split into good/suspect."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Import the plotting function from run_single_neuron_plots
    from run_single_neuron_plots import plot_custom_unit_summary

    fig_dir = Path(fig_dir)
    good_dir = fig_dir / 'units'
    suspect_dir = fig_dir / 'units_suspect'
    good_dir.mkdir(parents=True, exist_ok=True)
    suspect_dir.mkdir(parents=True, exist_ok=True)

    baseline_set = set(merged_df[merged_df['is_baseline_good']]['global_unit_idx'])
    strict_set = set(merged_df[merged_df['is_good']]['global_unit_idx'])
    suspect_set = baseline_set - strict_set

    n_good = len(strict_set)
    n_suspect = len(suspect_set)
    logger.info("  Figures: %d good + %d suspect = %d total",
                 n_good, n_suspect, n_good + n_suspect)

    count = 0
    for bank_id, sorting, rec_pp, analyzer in bank_results:
        if analyzer is None:
            continue

        bank_rows = merged_df[merged_df['bank'] == bank_id]

        for _, row in bank_rows.iterrows():
            gidx = row['global_unit_idx']
            uid = row['unit_id']

            if gidx not in baseline_set:
                continue

            out_dir = good_dir if gidx in strict_set else suspect_dir

            fig = plt.figure(figsize=(18, 6), dpi=100)
            try:
                plot_custom_unit_summary(analyzer, uid, fig)
                fig.suptitle(
                    f"Unit {gidx:04d} | Bank {bank_id} | "
                    f"Depth {row['depth_um']:.0f}µm | "
                    f"FR {row['firing_rate']:.2f}Hz | "
                    f"Amp {row['amplitude_uv']:.0f}µV",
                    fontsize=11, y=0.98)
                fig.savefig(out_dir / f"unit_{gidx:04d}_bank{bank_id:02d}.png",
                            dpi=100, bbox_inches='tight')
            except Exception as e:
                logger.warning("  Unit %d bank %d: figure failed: %s",
                               gidx, bank_id, e)
            finally:
                plt.close(fig)

            count += 1
            if count % 50 == 0:
                logger.info("    %d/%d figures done", count,
                             n_good + n_suspect)

    logger.info("  Generated %d figures", count)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sort multi-bank wireless recording (per-bank)')
    parser.add_argument('data_dir', help='Directory containing .dat segments')
    parser.add_argument('--bank', type=int, nargs='+', default=None,
                        help='Only sort specific bank(s)')
    parser.add_argument('--trodesconf', type=str,
                        default=os.environ.get('TRODESCONF'),
                        help='Path to .trodesconf for channel mapping '
                              '(defaults to the TRODESCONF env var)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--skip-figures', action='store_true')
    parser.add_argument('--skip-imu', action='store_true')
    parser.add_argument('--force', action='store_true',
                        help='Force re-sort even if output exists')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no-notch', action='store_true',
                        help='Skip wireless-specific notch filtering (use '
                              'plain bandpass + CMR + phase_shift)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_root = Path(os.environ.get('DATA_ROOT', 'E:/'))
    output_dir = Path(args.output) if args.output else (
        data_root / 'sorting_output' / data_dir.name)
    fig_dir = data_root / 'figures' / 'sorting' / data_dir.name
    torch_device = args.device or _detect_torch_device()
    bank_filter = set(args.bank) if args.bank else None

    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("WIRELESS SORT (per-bank): %s", data_dir.name)
    logger.info("=" * 70)
    logger.info("Data: %s", data_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Trodesconf: %s", args.trodesconf)
    if bank_filter:
        logger.info("Bank filter: %s", bank_filter)

    # Phase 1: Discover
    logger.info("\n[1/5] DISCOVERING segments")
    logger.info("=" * 70)
    segments = discover_segments(data_dir)
    logger.info("  Found %d banks: %s", len(segments),
                 [s['bank'] for s in segments])

    if bank_filter:
        segments = [s for s in segments if s['bank'] in bank_filter]
        logger.info("  After filter: %d banks", len(segments))

    # Phase 2: IMU
    if not args.skip_imu and any(s['has_imu'] for s in segments):
        logger.info("\n[2/5] EXTRACTING IMU data")
        logger.info("=" * 70)
        imu_data = extract_imu(segments, output_dir)
        if imu_data is not None:
            plot_imu(imu_data, fig_dir)
    else:
        logger.info("\n[2/5] IMU — skipped")

    # Phase 3: Sort each bank
    logger.info("\n[3/5] SORTING each bank with KS4")
    logger.info("=" * 70)
    sort_results = []  # (bank_id, sorting, rec_pp)
    for seg in segments:
        sorting, rec_pp, bank_id, n_units = sort_one_bank(
            seg, args.trodesconf, output_dir, torch_device, force=args.force,
            notch=not args.no_notch)
        sort_results.append((bank_id, sorting, rec_pp))

    # Phase 4: Analyze each bank
    logger.info("\n[4/5] BUILDING analyzers per bank")
    logger.info("=" * 70)
    bank_results = []  # (bank_id, sorting, rec_pp, analyzer)
    for bank_id, sorting, rec_pp in sort_results:
        if sorting is None:
            bank_results.append((bank_id, None, rec_pp, None))
            continue
        analyzer = analyze_one_bank(sorting, rec_pp, bank_id, output_dir)
        bank_results.append((bank_id, sorting, rec_pp, analyzer))

    # Phase 5: Merge + quality + figures
    logger.info("\n[5/5] MERGE + QUALITY + FIGURES")
    logger.info("=" * 70)
    merged_df = merge_and_classify(bank_results, output_dir, fig_dir,
                                    skip_figures=args.skip_figures)

    # Summary
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE in %.0f min", elapsed / 60)
    logger.info("Banks sorted: %d", sum(1 for _, s, _, _ in bank_results
                                         if s is not None))
    logger.info("Units: %d total, %d good",
                 len(merged_df), merged_df['is_good'].sum())
    logger.info("Output: %s", output_dir)
    logger.info("Figures: %s", fig_dir)


if __name__ == '__main__':
    main()
