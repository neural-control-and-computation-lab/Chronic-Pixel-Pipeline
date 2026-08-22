"""
Sort SpikeGLX survey recording across ALL probe banks.

Survey mode records from each of the 12 banks of the NP1032 probe sequentially
(~40s per bank, 2s settle). This script:
  1. Parses the svySBTT (Survey Shank-Bank Time Table) from the .meta file
  2. Extracts each bank's stable data window
  3. Assigns correct probe geometry per bank (depth offset)
  4. Preprocesses and spike-sorts each bank independently (Kilosort4)
  5. Builds SortingAnalyzer per bank with quality metrics
  6. Merges all bank sortings into one full-probe sorting object
  7. Generates full-probe visualization

Usage:
  python sort_survey.py                            # sort latest survey
  python sort_survey.py 20260226                   # sort by date key
  python sort_survey.py all                        # sort ALL surveys
  python sort_survey.py <survey_dir> <output_dir>  # sort specific path

Environment:
  Set DATA_ROOT env var to override the default data directory.
  CUDA is auto-detected; falls back to CPU if unavailable.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import logging
import platform
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _default_data_root():
    """Return platform-appropriate default data root."""
    env = os.environ.get('DATA_ROOT')
    if env:
        return Path(env)
    system = platform.system()
    if system == 'Darwin':
        return Path.home() / 'Library/CloudStorage/Dropbox/Lab Files/TRODES test'
    elif system == 'Windows':
        return Path.home() / 'data' / 'TRODES_test'
    else:
        return Path.home() / 'data' / 'TRODES_test'


def _detect_torch_device():
    """Auto-detect best available torch device."""
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


DATA_ROOT = _default_data_root()

# All known survey directories (paths are relative to DATA_ROOT)
SURVEYS = {
    '20260226': 'SvyPrb_2026-02-26T11.58.07_g0',
    '20260302': 'SvyPrb_2026-03-02T14.10.26_g0',
    '20260304': 'SvyPrb_2026-03-04T10.17.50_g0',
    '20260306': 'SvyPrb_2026-03-06T10.01.15_g0',
    '20260309': 'SvyPrb_2026-03-09T13.27.11_g0',
    '20260311': 'SvyPrb_2026-03-11T09.52.30_g0',
    '20260313': 'SvyPrb_2026-03-13T13.14.59_g0',
    '20260316': 'SvyPrb_2026-03-16T10.56.17_g0',
    '20260319': 'SvyPrb_2026-03-19T10.11.39_g0',
    '20260323': 'SvyPrb_2026-03-23T11.18.04_g0',
    '20260326': 'SvyPrb_2026-03-26T11.33.18_g0',
    '20260331': 'SvyPrb_2026-03-31T14.35.53_g0',
    '20260402': 'SvyPrb_2026-04-02T14.08.31_g0',
    '20260407': 'SvyPrb_2026-04-07T13.09.12_g0',
    '20260409': 'SvyPrb_2026-04-09T14.36.02_g0',
    '20260414': 'SvyPrb_2026-04-14T13.19.39_g0',
    '20260417': 'SvyPrb_2026-04-17T10.24.24_g0',
    '20260421': 'SvyPrb_2026-04-21T14.44.42_g0',
    '20260424': 'SvyPrb_2026-04-24T12.31.37_g0',
    '20260428': 'SvyPrb_2026-04-28T12.56.18_g0',
    '20260501': 'SvyPrb_2026-05-01T13.06.12_g0',
    '20260507': 'SvyPrb_2026-05-07T14.16.53_g0',
    '20260512': 'SvyPrb_2026-05-12T13.38.51_g0',
    '20260522': 'SvyPrb_2026-05-22T12.22.23_g0',
    '20260526': 'SvyPrb_2026-05-26T13.16.22_g0',
    '20260604': 'SvyPrb_2026-06-04T13.34.16_g0',
    '20260611': 'SvyPrb_2026-06-11T14.09.05_g0',
    '20260618': 'SvyPrb_2026-06-18T15.12.37_g0',
    '20260625': 'SvyPrb_2026-06-25T10.11.07_g0',
    '20260723': 'SvyPrb_2026-07-23T14.18.28_g0',
    '20260730': 'SvyPrb_2026-07-30T14.27.42_g0',
    '20260813': 'SvyPrb_2026-08-13T14.08.28_g0',
    '20260820': 'SvyPrb_2026-08-20T15.03.05_g0',
}


def _survey_dir(key):
    """Return the raw SpikeGLX data directory for a survey."""
    return DATA_ROOT / SURVEYS[key]


def _output_dir(key):
    """Return the sorting output directory for a survey (inside DATA_ROOT)."""
    return DATA_ROOT / 'sorting_output' / f'survey_{key}'


def _fig_dir(key):
    """Return the figure directory for a survey (inside DATA_ROOT)."""
    return DATA_ROOT / 'figures' / 'sorting' / f'survey_{key}'


def parse_meta(meta_path):
    """Parse SpikeGLX .meta file into a dict."""
    meta = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                meta[k] = v
    return meta


def parse_svySBTT(meta):
    """Parse the Survey Shank-Bank Time Table.

    Format: (shank bank settle_start settle_end) for each bank.
    The stable data for bank N runs from settle_end to the next bank's
    settle_start (or end of file for the last bank).

    Returns list of dicts with keys: shank, bank, settle_start, settle_end
    """
    sbtt_str = meta.get('~svySBTT', '')
    entries = re.findall(r'\((\d+)\s+(\d+)\s+(\d+)\s+(\d+)\)', sbtt_str)
    banks = []
    for shank, bank, start, end in entries:
        banks.append({
            'shank': int(shank),
            'bank': int(bank),
            'settle_start': int(start),
            'settle_end': int(end),
        })
    return banks


def make_bank_probe(bank_id, n_channels=384, z_pitch=20, n_rows_per_bank=192):
    """Create a SpikeInterface Probe object for a given bank.

    For NP1032: 2 columns (X=11, X=114), rows spaced by z_pitch.
    Bank N covers rows N*n_rows_per_bank to (N+1)*n_rows_per_bank - 1.
    Bank 11 (last) may have fewer rows (only 96 for NP1032 with 2208 total rows).
    """
    from probeinterface import Probe

    total_rows = 2208  # NP1032 total rows
    row_start = bank_id * n_rows_per_bank
    row_end = min((bank_id + 1) * n_rows_per_bank, total_rows)
    actual_rows = row_end - row_start
    actual_channels = actual_rows * 2  # 2 columns

    n_contacts = min(actual_channels, n_channels)

    positions = np.zeros((n_contacts, 2))
    for i in range(actual_rows):
        row = row_start + i
        z = row * z_pitch
        positions[2 * i, :] = [11, z]
        positions[2 * i + 1, :] = [114, z]

    probe = Probe(ndim=2)
    probe.set_contacts(positions=positions, shapes='square', shape_params={'width': 12})
    probe.set_device_channel_indices(np.arange(n_contacts))
    probe.create_auto_shape(probe_type='rect')

    return probe, actual_channels


def find_meta_file(survey_dir):
    """Find the .ap.meta file in a SpikeGLX survey directory."""
    metas = list(Path(survey_dir).glob('*.ap.meta'))
    if not metas:
        raise FileNotFoundError(f"No .ap.meta file found in {survey_dir}")
    return metas[0]


def sort_and_analyze_banks(survey_dir, output_dir, fig_dir, torch_device=None,
                           bank_filter=None, force=False):
    """Sort all banks and return results list."""
    import spikeinterface.full as si
    from chronic_pixel_pipeline.preprocessing import preprocess
    from chronic_pixel_pipeline.analysis import build_sorting_analyzer, classify_units

    if torch_device is None:
        torch_device = _detect_torch_device()

    survey_dir = Path(survey_dir)
    output_dir = Path(output_dir)
    fig_dir = Path(fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Parse metadata ----
    meta_file = find_meta_file(survey_dir)
    logger.info("Parsing metadata from %s ...", meta_file.name)
    meta = parse_meta(meta_file)
    fs = float(meta['imSampRate'])
    max_bank = int(meta['imSvyMaxBnk'])
    file_time = float(meta['fileTimeSecs'])

    logger.info("  %.0f Hz, banks 0-%d, %.1fs total", fs, max_bank, file_time)

    # ---- 2. Parse svySBTT ----
    banks = parse_svySBTT(meta)
    total_samples = int(file_time * fs)
    bank_windows = []
    for i, b in enumerate(banks):
        data_start = b['settle_end']
        data_end = banks[i + 1]['settle_start'] if i + 1 < len(banks) else total_samples
        duration = (data_end - data_start) / fs
        bank_windows.append({
            'bank': b['bank'],
            'data_start': data_start,
            'data_end': data_end,
            'duration': duration,
        })
        logger.info("  Bank %2d: %.1fs stable data", b['bank'], duration)

    # ---- 3. Load full recording ----
    logger.info("Loading full survey recording...")
    rec_full = si.read_spikeglx(str(survey_dir), stream_id='imec0.ap')

    # ---- 4. Sort each bank ----
    all_bank_results = []

    for bw in bank_windows:
        bank_id = bw['bank']
        if bank_filter is not None and bank_id not in bank_filter:
            logger.info("  Skipping bank %d (not in filter)", bank_id)
            continue
        logger.info("\n" + "=" * 70)
        logger.info("BANK %d — %.1fs of data", bank_id, bw['duration'])
        logger.info("=" * 70)

        # Extract time window
        rec_bank = rec_full.frame_slice(
            start_frame=bw['data_start'], end_frame=bw['data_end'])

        # Set correct probe geometry
        probe, actual_ch = make_bank_probe(bank_id)
        rec_bank = rec_bank.set_probe(probe, group_mode='by_probe')

        if actual_ch < 384:
            logger.info("  Bank %d: %d active channels (partial)", bank_id, actual_ch)
            good_ids = rec_bank.get_channel_ids()[:actual_ch]
            rec_bank = rec_bank.select_channels(channel_ids=good_ids)

        # Quick signal check
        snippet_dur = min(2.0, bw['duration'] - 1.0)
        snip = rec_bank.get_traces(
            start_frame=int(1.0 * fs), end_frame=int((1.0 + snippet_dur) * fs),
            return_in_uV=True)
        raw_rms = np.sqrt(np.mean(snip ** 2, axis=0))
        logger.info("  Raw RMS: median=%.1f µV", np.median(raw_rms))

        # Preprocess
        rec_pp = preprocess(rec_bank, freq_min=300, freq_max=6000,
                            destripe=False, cmr=False, phase_shift=True)

        snip_pp = rec_pp.get_traces(
            start_frame=int(1.0 * fs), end_frame=int((1.0 + snippet_dur) * fs),
            return_in_uV=True)
        pp_rms = np.sqrt(np.mean(snip_pp ** 2, axis=0))
        mad_noise = np.median(np.abs(snip_pp), axis=0) / 0.6745
        logger.info("  Preprocessed RMS: median=%.1f µV, MAD noise: %.1f µV",
                     np.median(pp_rms), np.median(mad_noise))

        n_active = 0
        for ch in range(snip_pp.shape[1]):
            thresh = -4 * mad_noise[ch]
            below = snip_pp[:, ch] < thresh
            xings = np.sum(np.diff(below.astype(int)) == 1) / snippet_dur
            if xings > 1:
                n_active += 1
        logger.info("  Active channels: %d/%d", n_active, rec_pp.get_num_channels())

        # Spike sort with KS4
        # templates_from_data=False: use pre-built universal templates.
        # Data-driven templates (True) fail on banks with few spikes because
        # extract_wPCA_wTEMP needs >=6 clips for KMeans but sparse banks
        # don't produce enough above the Th_single_ch threshold.
        # nblocks=1: drift correction enabled (single block sufficient for
        # ~58s survey banks).
        sort_dir = output_dir / f'bank{bank_id:02d}_ks4'
        ks_kwargs = dict(
            verbose=False, torch_device=torch_device, nblocks=1,
            templates_from_data=False, batch_size=30000,
            dminx=103,                 # NP1032 column spacing (103µm, not 32µm)
            max_channel_distance=150,  # Include both columns (103µm apart)
            nearest_chans=20,          # More neighbors for multi-channel spread
        )

        # Resume support: skip KS4 if sorter_output already exists
        existing_sorter = sort_dir / 'sorter_output'
        if not force and existing_sorter.exists() and (existing_sorter / 'spike_clusters.npy').exists():
            logger.info("  Resuming — loading existing KS4 output from %s", sort_dir)
            try:
                sorting = si.read_sorter_folder(sort_dir)
                n_units = len(sorting.unit_ids)
                total_spikes = sum(sorting.count_num_spikes_per_unit().values())
                logger.info("  KS4 (cached): %d units, %d spikes", n_units, total_spikes)
            except Exception as e:
                logger.warning("  Failed to load cached sorting: %s — re-running", e)
                sorting = None
        else:
            sorting = None

        if sorting is None:
            logger.info("  Running kilosort4...")
            try:
                sorting = si.run_sorter(
                    'kilosort4', rec_pp, folder=sort_dir,
                    remove_existing_folder=True, **ks_kwargs)
                n_units = len(sorting.unit_ids)
                total_spikes = sum(sorting.count_num_spikes_per_unit().values())
                logger.info("  KS4: %d units, %d spikes", n_units, total_spikes)
            except Exception as e:
                err_str = str(e)
                # Detect KS4 template extraction failures (ValueError wrapped
                # by SpikeInterface's SpikeSortingError) — extract_wPCA_wTEMP
                # fails when too few spike clips exceed Th_single_ch threshold
                # for KMeans (needs n_clusters=6 samples). Retry with pre-built
                # universal templates instead of data-driven ones.
                is_template_error = (
                    'n_samples=' in err_str and 'n_clusters=' in err_str
                ) or (
                    'Found array with 0 sample' in err_str
                ) or (
                    'extract_wPCA_wTEMP' in err_str
                )
                if not is_template_error:
                    logger.warning("  KS4 failed: %s", e)
                    sorting = None
                    n_units = 0
                    total_spikes = 0
                else:
                    logger.warning("  KS4 template extraction failed: %s", e)
                    logger.info("  Retrying with templates_from_data=False "
                                "(pre-built universal templates)...")
                    try:
                        ks_fallback = {**ks_kwargs, 'templates_from_data': False}
                        sorting = si.run_sorter(
                            'kilosort4', rec_pp, folder=sort_dir,
                            remove_existing_folder=True, **ks_fallback)
                        n_units = len(sorting.unit_ids)
                        total_spikes = sum(sorting.count_num_spikes_per_unit().values())
                        logger.info("  KS4 (universal templates): %d units, %d spikes",
                                     n_units, total_spikes)
                    except Exception as e2:
                        logger.warning("  KS4 fallback also failed: %s", e2)
                        sorting = None
                        n_units = 0
                        total_spikes = 0

        result = {
            'bank': bank_id,
            'sorting': sorting,
            'recording': rec_pp,
            'rec_bank_raw': rec_bank,
            'n_units': n_units,
            'total_spikes': total_spikes,
            'n_active_channels': n_active,
            'median_rms': float(np.median(pp_rms)),
            'median_noise': float(np.median(mad_noise)),
            'duration': bw['duration'],
            'probe': probe,
            'actual_channels': actual_ch,
            'bank_window': bw,
            '_sort_dir': str(sort_dir),
            '_fs': float(fs),
        }

        # Build analyzer
        if sorting is not None and n_units > 0:
            logger.info("  Building SortingAnalyzer...")
            try:
                cache = sort_dir / 'analyzer_cache'
                analyzer = build_sorting_analyzer(sorting, rec_pp, cache_dir=cache)
                qm = analyzer.get_extension("quality_metrics").get_data()
                good_ids, snr_thresh = classify_units(analyzer)
                logger.info("  %d/%d good units (SNR>%.2f)",
                             len(good_ids), n_units, snr_thresh)
                result['analyzer'] = analyzer
                result['good_ids'] = good_ids
                result['snr_thresh'] = snr_thresh
                result['qm'] = qm
            except Exception as e:
                logger.warning("  Analyzer failed: %s", e)

        all_bank_results.append(result)

    return all_bank_results


def merge_bank_sortings(all_bank_results, output_dir):
    """Merge per-bank sortings into one full-probe sorting object.

    Creates a merged sorting where each unit ID is prefixed with its bank
    number (e.g., 'b00_u005') and positioned at the correct absolute depth
    on the full NP1032 probe.

    Saves merged data to output_dir/merged/.
    """
    import spikeinterface.core as sc

    merged_dir = Path(output_dir) / 'merged'
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Collect all good units across banks with their metadata
    all_units = []
    for r in all_bank_results:
        if r['sorting'] is None or r['n_units'] == 0:
            continue
        bank_id = r['bank']
        good_ids = r.get('good_ids', [])
        sorting = r['sorting']
        probe = r['probe']
        positions = probe.contact_positions

        for uid in sorting.unit_ids:
            is_good = uid in good_ids if good_ids is not None else False
            # Get unit depth from template peak channel
            depth = np.nan
            amplitude = 0.0
            if 'analyzer' in r:
                try:
                    templates = r['analyzer'].get_extension("templates").get_data()
                    uid_idx = list(sorting.unit_ids).index(uid)
                    template = templates[uid_idx]
                    peak_ch = np.argmax(np.ptp(template, axis=0))
                    if peak_ch < len(positions):
                        depth = positions[peak_ch, 1]
                    amplitude = float(np.ptp(template[:, peak_ch]))
                except Exception:
                    pass

            spike_train = sorting.get_unit_spike_train(uid)
            fr = len(spike_train) / r['duration']

            all_units.append({
                'bank': bank_id,
                'original_id': uid,
                'merged_id': f"b{bank_id:02d}_u{uid}",
                'is_good': is_good,
                'depth': depth,
                'amplitude': amplitude,
                'firing_rate': fr,
                'n_spikes': len(spike_train),
            })

    # Save merged unit table
    import pandas as pd
    df = pd.DataFrame(all_units)
    df.to_csv(merged_dir / 'merged_units.csv', index=False)

    if len(df) == 0:
        logger.info("\nMerged sorting: 0 units (no banks produced output)")
        return df

    n_good = df['is_good'].sum()
    n_total = len(df)
    logger.info("\nMerged sorting: %d total units, %d good units across %d banks",
                n_total, n_good, len(set(df['bank'])))

    # Save per-bank summary
    for bank_id in sorted(df['bank'].unique()):
        bdf = df[df['bank'] == bank_id]
        n_good_bank = bdf['is_good'].sum()
        logger.info("  Bank %02d: %3d units, %3d good", bank_id, len(bdf), n_good_bank)

    return df


def plot_drift_diagnostics(all_bank_results, fig_dir):
    """Plot drift correction diagnostics for all banks.

    For each bank, loads the KS4 ops.npy to get:
      - dshift: drift in microns over time per block
      - spike_positions: spike depth scatter over time
    Generates a combined figure showing drift across the full probe.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect drift data across banks
    banks_with_drift = []
    for r in all_bank_results:
        sort_dir = r.get('_sort_dir')
        if sort_dir is None:
            continue
        ops_path = Path(sort_dir) / 'sorter_output' / 'ops.npy'
        if not ops_path.exists():
            continue
        ops = np.load(ops_path, allow_pickle=True).item()
        if ops.get('dshift') is None:
            continue
        banks_with_drift.append({
            'bank': r['bank'],
            'dshift': np.array(ops['dshift']),
            'yblk': np.array(ops.get('yblk', [0])),
            'batch_size': ops.get('batch_size', 30000),
            'fs': ops.get('fs', 30000),
            'mean_drift': ops.get('mean_drift', np.nan),
            'duration': r['duration'],
        })

    if not banks_with_drift:
        logger.info("  No drift data available (nblocks=0?)")
        return

    # ---- Figure: Drift amount per bank ----
    n_banks = len(banks_with_drift)
    fig, axes = plt.subplots(n_banks, 1, figsize=(14, 2.5 * n_banks),
                             squeeze=False, sharex=True)
    fig.suptitle('Drift Correction — All Banks', fontsize=14, fontweight='bold')

    for idx, bd in enumerate(banks_with_drift):
        ax = axes[idx, 0]
        dshift = bd['dshift']
        n_batches = dshift.shape[0]
        t = np.arange(n_batches) * bd['batch_size'] / bd['fs']

        if dshift.ndim == 1:
            ax.plot(t, dshift, color='#4472C4', linewidth=0.8)
        else:
            for col in range(dshift.shape[1]):
                ax.plot(t, dshift[:, col], linewidth=0.8, alpha=0.8)

        ax.set_ylabel('Drift (um)')
        ax.set_title(f'Bank {bd["bank"]}  (mean |drift| = {bd["mean_drift"]:.1f} um)',
                     fontsize=10, loc='left')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    axes[-1, 0].set_xlabel('Time (s)')
    fig.tight_layout()
    out_path = fig_dir / 'drift_correction.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    # ---- Figure: Spike depth scatter per bank ----
    banks_with_spikes = []
    for r in all_bank_results:
        if r['sorting'] is None or r['n_units'] == 0:
            continue
        sort_dir = r.get('_sort_dir')
        if sort_dir is None:
            continue
        pos_path = Path(sort_dir) / 'sorter_output' / 'spike_positions.npy'
        st_path = Path(sort_dir) / 'sorter_output' / 'spike_templates.npy'
        amp_path = Path(sort_dir) / 'sorter_output' / 'amplitudes.npy'
        if not pos_path.exists():
            continue
        try:
            spike_pos = np.load(pos_path)
            spike_amps = np.load(amp_path) if amp_path.exists() else None
            banks_with_spikes.append({
                'bank': r['bank'],
                'spike_pos': spike_pos,
                'spike_amps': spike_amps,
                'fs': r.get('_fs', 30000),
                'duration': r['duration'],
            })
        except Exception:
            continue

    if banks_with_spikes:
        n_sp = len(banks_with_spikes)
        fig, axes = plt.subplots(n_sp, 1, figsize=(14, 3 * n_sp),
                                 squeeze=False, sharex=True)
        fig.suptitle('Spike Depth Over Time — All Banks',
                     fontsize=14, fontweight='bold')

        for idx, bs in enumerate(banks_with_spikes):
            ax = axes[idx, 0]
            pos = bs['spike_pos']
            # pos columns: x, y (depth)
            # Subsample for speed
            n_total = len(pos)
            max_plot = 50000
            if n_total > max_plot:
                sel = np.random.default_rng(42).choice(n_total, max_plot, replace=False)
                sel = np.sort(sel)
            else:
                sel = np.arange(n_total)

            y_depth = pos[sel, 1] if pos.ndim > 1 else pos[sel]
            # Use spike index as proxy for time
            t = sel / n_total * bs['duration']

            c = '#4472C4'
            if bs['spike_amps'] is not None:
                amps = np.abs(bs['spike_amps'][sel].ravel())
                c = np.clip(np.log10(amps + 1), 0, None)

            ax.scatter(t, y_depth, s=0.3, c=c, alpha=0.3,
                       cmap='hot', rasterized=True)
            ax.set_ylabel('Depth (um)')
            ax.set_title(f'Bank {bs["bank"]}  ({n_total} spikes)',
                         fontsize=10, loc='left')

        axes[-1, 0].set_xlabel('Time (s)')
        fig.tight_layout()
        out_path = fig_dir / 'drift_scatter.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info("  Saved: %s", out_path)


def plot_full_probe(all_bank_results, fig_dir):
    """Generate combined full-probe visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 20))

    all_depths = []
    all_frs = []
    all_amps = []
    all_bank_labels = []
    all_good = []

    for r in all_bank_results:
        if r['n_units'] == 0 or 'qm' not in r:
            continue
        bank_id = r['bank']
        sorting = r['sorting']
        probe = r['probe']
        positions = probe.contact_positions

        for uid in sorting.unit_ids:
            st = sorting.get_unit_spike_train(uid)
            fr = len(st) / r['duration']
            depth = np.nan
            amp = 0
            if 'analyzer' in r:
                try:
                    templates = r['analyzer'].get_extension("templates").get_data()
                    uid_idx = list(sorting.unit_ids).index(uid)
                    template = templates[uid_idx]
                    peak_ch = np.argmax(np.ptp(template, axis=0))
                    if peak_ch < len(positions):
                        depth = positions[peak_ch, 1]
                    amp = np.ptp(template[:, peak_ch])
                except Exception:
                    pass
            all_depths.append(depth)
            all_frs.append(fr)
            all_amps.append(amp)
            all_bank_labels.append(bank_id)
            all_good.append('good_ids' in r and uid in r['good_ids'])

    all_depths = np.array(all_depths)
    all_frs = np.array(all_frs)
    all_amps = np.array(all_amps)
    all_good = np.array(all_good)
    valid = ~np.isnan(all_depths)

    total_units = sum(r['n_units'] for r in all_bank_results)
    total_good = int(np.sum(all_good))
    fig.suptitle(f'NP1032 Survey — Full Probe Overview\n'
                 f'{total_units} units ({total_good} good) across '
                 f'{len(all_bank_results)} banks',
                 fontsize=16, fontweight='bold')

    # Panel A: Unit locations colored by firing rate
    ax = axes[0]
    if np.any(valid):
        sc = ax.scatter(np.random.normal(62.5, 15, np.sum(valid)),
                        all_depths[valid], c=all_frs[valid], s=20,
                        cmap='hot', alpha=0.7,
                        norm=Normalize(vmin=0, vmax=np.percentile(all_frs[valid], 95)),
                        edgecolors='k', linewidth=0.2)
        plt.colorbar(sc, ax=ax, label='Firing Rate (Hz)', shrink=0.3)
    for bk in range(13):
        y = bk * 192 * 20
        ax.axhline(y, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
        if bk < 12:
            ax.text(125, y + 1900, f'Bank {bk}', fontsize=7, color='gray',
                    ha='center', va='center')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title('Unit Locations (color=FR)')
    ax.set_xlim(-20, 150)
    ax.invert_yaxis()

    # Panel B: Firing rate by depth
    ax = axes[1]
    if np.any(valid):
        colors = ['#70AD47' if g else '#4472C4' for g in all_good[valid]]
        ax.scatter(all_frs[valid], all_depths[valid], s=15, c=colors,
                   alpha=0.6, edgecolors='k', linewidth=0.2)
    for bk in range(13):
        ax.axhline(bk * 192 * 20, color='gray', linewidth=0.5,
                    linestyle='--', alpha=0.3)
    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title('Firing Rate by Depth\n(green=good, blue=all)')
    ax.invert_yaxis()

    # Panel C: Amplitude by depth
    ax = axes[2]
    if np.any(valid):
        ax.scatter(all_amps[valid], all_depths[valid], s=15,
                   c='#ED7D31', alpha=0.6, edgecolors='k', linewidth=0.2)
    for bk in range(13):
        ax.axhline(bk * 192 * 20, color='gray', linewidth=0.5,
                    linestyle='--', alpha=0.3)
    ax.set_xlabel('Template PTP (µV)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title('Amplitude by Depth')
    ax.invert_yaxis()

    # Panel D: Units per bank
    ax = axes[3]
    bank_ids_plot = [r['bank'] for r in all_bank_results]
    units_per_bank = [r['n_units'] for r in all_bank_results]
    good_per_bank = [len(r.get('good_ids', [])) for r in all_bank_results]
    x_pos = np.arange(len(bank_ids_plot))
    ax.barh(x_pos, units_per_bank, color='#4472C4', alpha=0.7, label='All units')
    ax.barh(x_pos, good_per_bank, color='#70AD47', alpha=0.8, label='Good units')
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f'Bank {b}' for b in bank_ids_plot])
    ax.set_xlabel('Number of Units')
    ax.set_title('Units per Bank')
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    out_path = fig_dir / 'survey_full_probe.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: %s", out_path)


def sort_one_survey(key, survey_dir, output_dir, fig_dir=None, torch_device=None,
                    bank_filter=None, force=False):
    """Sort a single survey: all banks, merge, visualize."""
    if fig_dir is None:
        fig_dir = DATA_ROOT / 'figures' / 'sorting' / output_dir.name
    logger.info("Survey: %s", survey_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Figures: %s", fig_dir)
    if bank_filter is not None:
        logger.info("Bank filter: %s", bank_filter)

    all_bank_results = sort_and_analyze_banks(
        survey_dir, output_dir, fig_dir, torch_device=torch_device,
        bank_filter=bank_filter, force=force)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SURVEY SUMMARY — %s — ALL BANKS", key)
    logger.info("=" * 70)
    total_units = sum(r['n_units'] for r in all_bank_results)
    logger.info("  Total units: %d", total_units)
    logger.info("  %-6s %6s %8s %6s %8s %7s", "Bank", "Units", "Spikes",
                "ActCh", "RMS(uV)", "Dur(s)")
    logger.info("  %s", "-" * 50)
    for r in all_bank_results:
        logger.info("  Bank%02d %6d %8d %6d %8.1f %7.1f",
                     r['bank'], r['n_units'], r['total_spikes'],
                     r['n_active_channels'], r['median_rms'], r['duration'])

    # Merge bank sortings
    merged_df = merge_bank_sortings(all_bank_results, output_dir)

    # Drift diagnostics
    logger.info("\nGenerating drift diagnostics...")
    try:
        plot_drift_diagnostics(all_bank_results, fig_dir)
    except Exception as e:
        logger.warning("Drift diagnostics failed: %s", e)

    # Full-probe visualization
    logger.info("\nGenerating full-probe visualization...")
    try:
        plot_full_probe(all_bank_results, fig_dir)
    except Exception as e:
        logger.warning("Visualization failed: %s", e)

    logger.info("\nSurvey %s complete.", key)
    return all_bank_results, merged_df


def main():
    import argparse as _ap

    parser = _ap.ArgumentParser(description='Sort SpikeGLX survey recording')
    parser.add_argument('survey', nargs='?', default=None,
                        help='Survey date key (e.g. 20260304), "all", or path')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output directory (only with explicit path)')
    parser.add_argument('--bank', type=int, nargs='+', default=None,
                        help='Only sort specific bank(s), e.g. --bank 0 or --bank 0 1 2')
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device (default: auto-detect)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-sort even if output already exists')
    args = parser.parse_args()

    torch_device = args.device or _detect_torch_device()
    bank_filter = set(args.bank) if args.bank is not None else None
    force = args.force
    logger.info("DATA_ROOT: %s", DATA_ROOT)

    if args.survey and args.output:
        # Explicit paths: sort_survey.py <survey_dir> <output_dir>
        survey_dir = Path(args.survey)
        output_dir = Path(args.output)
        sort_one_survey('custom', survey_dir, output_dir,
                        torch_device=torch_device, bank_filter=bank_filter,
                        force=force)

    elif args.survey:
        arg = args.survey

        if arg == 'all':
            logger.info("Sorting ALL %d surveys...", len(SURVEYS))
            for key in sorted(SURVEYS.keys()):
                survey_dir = _survey_dir(key)
                output_dir = _output_dir(key)
                fig_dir = _fig_dir(key)
                if not survey_dir.exists():
                    logger.warning("Skipping %s — data not found: %s",
                                   key, survey_dir)
                    continue
                logger.info("\n" + "#" * 70)
                logger.info("# SURVEY %s", key)
                logger.info("#" * 70)
                sort_one_survey(key, survey_dir, output_dir,
                                fig_dir=fig_dir, torch_device=torch_device,
                                bank_filter=bank_filter, force=force)
        elif arg in SURVEYS:
            sort_one_survey(arg, _survey_dir(arg), _output_dir(arg),
                            fig_dir=_fig_dir(arg), torch_device=torch_device,
                            bank_filter=bank_filter, force=force)
        else:
            survey_dir = Path(arg)
            date_str = survey_dir.name.split('_')[1].split('T')[0].replace('-', '')
            output_dir = DATA_ROOT / 'sorting_output' / f'survey_{date_str}'
            sort_one_survey(date_str, survey_dir, output_dir,
                            torch_device=torch_device, bank_filter=bank_filter,
                            force=force)
    else:
        # Default: latest survey
        key = '20260820'
        sort_one_survey(key, _survey_dir(key), _output_dir(key),
                        fig_dir=_fig_dir(key), torch_device=torch_device,
                        bank_filter=bank_filter, force=force)


if __name__ == '__main__':
    main()
