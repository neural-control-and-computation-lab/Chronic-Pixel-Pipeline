"""
Mega-sort: unified spike sorting across all banks and sessions.

Stacks banks 0-5 by depth into a single 2,304-channel recording per survey,
then concatenates all surveys in time order. Runs KS4 (nblocks=1) and KS2
(via Docker) on the same preprocessed binary for comparison.

Pipeline:
  1. Load each survey's SpikeGLX recording
  2. Extract banks 0-5, preprocess, stack by depth (2,304 channels)
  3. Concatenate surveys in time order
  4. Write preprocessed data to binary file on disk
  5. Run KS4 with nblocks=1
  6. Run KS2 via Docker
  7. Build SortingAnalyzer + quality metrics for each sorter
  8. Generate per-unit summary figures for each sorter

Usage:
  python mega_sort.py                   # run full pipeline (KS4 + KS2)
  python mega_sort.py --ks4-only        # only run KS4
  python mega_sort.py --ks2-only        # only run KS2 (assumes binary exists)
  python mega_sort.py --skip-figures    # sort + analyze, skip figure generation
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
import sys
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

# ── Imports from existing pipeline ────────────────────────────────────
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from sort_survey import (
    _default_data_root, _detect_torch_device,
    parse_meta, parse_svySBTT, make_bank_probe, find_meta_file,
    SURVEYS,
)

DATA_ROOT = _default_data_root()
OUTPUT_DIR = DATA_ROOT / 'sorting_output' / 'mega_sort'
FIG_DIR = DATA_ROOT / 'figures' / 'sorting' / 'mega_sort'
BANKS = list(range(7))  # banks 0-6
N_CHANNELS_PER_BANK = 384


# ── Phase 1: Load and assemble recordings ─────────────────────────────

def load_survey_stacked(survey_key, survey_dirname):
    """Load one survey, extract banks 0-5, preprocess, and stack by depth.

    Returns a single recording with 2,304 channels (6 banks x 384 channels),
    trimmed to the shortest bank duration, with correct probe geometry.
    """
    import spikeinterface.full as si
    from chronic_pixel_pipeline.preprocessing import preprocess

    survey_dir = DATA_ROOT / survey_dirname
    meta_file = find_meta_file(survey_dir)
    meta = parse_meta(meta_file)
    fs = float(meta['imSampRate'])
    file_time = float(meta['fileTimeSecs'])
    total_samples = int(file_time * fs)

    # Parse bank windows
    banks = parse_svySBTT(meta)
    bank_windows = []
    for i, b in enumerate(banks):
        data_start = b['settle_end']
        data_end = banks[i + 1]['settle_start'] if i + 1 < len(banks) else total_samples
        bank_windows.append({
            'bank': b['bank'],
            'data_start': data_start,
            'data_end': data_end,
            'duration': (data_end - data_start) / fs,
        })

    # Load full recording
    rec_full = si.read_spikeglx(str(survey_dir), stream_id='imec0.ap')

    # Extract and preprocess each bank
    bank_recs = []
    durations = []
    for bw in bank_windows:
        bank_id = bw['bank']
        if bank_id not in BANKS:
            continue

        rec_bank = rec_full.frame_slice(
            start_frame=bw['data_start'], end_frame=bw['data_end'])

        # Set probe geometry with absolute depth
        probe, actual_ch = make_bank_probe(bank_id)
        rec_bank = rec_bank.set_probe(probe, group_mode='by_probe')

        # Preprocess
        rec_pp = preprocess(rec_bank, freq_min=300, freq_max=6000,
                            destripe=False, cmr=False, phase_shift=True)

        # Rename channels to be unique across banks
        new_ids = [f"b{bank_id:02d}_ch{c:03d}" for c in range(rec_pp.get_num_channels())]
        rec_pp = rec_pp.rename_channels(new_channel_ids=new_ids)

        bank_recs.append(rec_pp)
        durations.append(rec_pp.get_num_samples())
        logger.info("    Bank %d: %d samples (%.1fs), %d channels",
                     bank_id, rec_pp.get_num_samples(),
                     rec_pp.get_num_samples() / fs, rec_pp.get_num_channels())

    # Trim all banks to shortest duration
    min_samples = min(durations)
    trimmed = [rec.frame_slice(start_frame=0, end_frame=min_samples)
               for rec in bank_recs]

    logger.info("    Trimmed to %d samples (%.1fs)", min_samples, min_samples / fs)

    # Stack by depth using aggregate_channels
    stacked = si.aggregate_channels(trimmed)
    logger.info("    Stacked: %d channels x %d samples",
                 stacked.get_num_channels(), stacked.get_num_samples())

    return stacked


def load_survey_single_bank(survey_key, survey_dirname, bank_id):
    """Load one bank from one survey, preprocess, and set probe geometry.

    Returns a single 384-channel recording for the given bank.
    """
    import spikeinterface.full as si
    from chronic_pixel_pipeline.preprocessing import preprocess

    survey_dir = DATA_ROOT / survey_dirname
    meta_file = find_meta_file(survey_dir)
    meta = parse_meta(meta_file)
    fs = float(meta['imSampRate'])
    file_time = float(meta['fileTimeSecs'])
    total_samples = int(file_time * fs)

    banks = parse_svySBTT(meta)
    bank_windows = []
    for i, b in enumerate(banks):
        data_start = b['settle_end']
        data_end = banks[i + 1]['settle_start'] if i + 1 < len(banks) else total_samples
        bank_windows.append({
            'bank': b['bank'],
            'data_start': data_start,
            'data_end': data_end,
        })

    # Find the window for this bank
    bw = next((w for w in bank_windows if w['bank'] == bank_id), None)
    if bw is None:
        return None

    rec_full = si.read_spikeglx(str(survey_dir), stream_id='imec0.ap')
    rec_bank = rec_full.frame_slice(
        start_frame=bw['data_start'], end_frame=bw['data_end'])

    probe, actual_ch = make_bank_probe(bank_id)
    rec_bank = rec_bank.set_probe(probe, group_mode='by_probe')

    rec_pp = preprocess(rec_bank, freq_min=300, freq_max=6000,
                        destripe=False, cmr=False, phase_shift=True)
    return rec_pp


def run_ks2_per_bank(output_dir):
    """Run KS2 via Docker per-bank (384 ch each), then merge results.

    For each bank, concatenates all surveys in time, writes a per-bank
    binary, and runs KS2. Returns a merged sorting with adjusted unit IDs.
    """
    import spikeinterface.full as si

    output_dir = Path(output_dir)
    survey_keys = sorted(SURVEYS.keys())
    all_sortings = []
    all_recordings = []
    bank_unit_offset = 0

    for bank_id in BANKS:
        logger.info("\n" + "=" * 70)
        logger.info("KS2 PER-BANK: Bank %d", bank_id)
        logger.info("=" * 70)

        bank_dir = output_dir / f'ks2_bank{bank_id:02d}'
        sort_dir = bank_dir / 'ks2_sort' / 'sorter_output'

        # Check for existing sort
        if sort_dir.exists() and (sort_dir / 'spike_clusters.npy').exists():
            logger.info("  Loading existing KS2 output for bank %d", bank_id)
            try:
                sorting = si.read_sorter_folder(bank_dir / 'ks2_sort')
                n_units = len(sorting.unit_ids)
                logger.info("  Loaded: %d units", n_units)
                all_sortings.append((bank_id, sorting))
                continue
            except Exception as e:
                logger.warning("  Failed to load: %s. Re-sorting...", e)

        # Load and concatenate this bank across all surveys
        bank_recs = []
        for key in survey_keys:
            rec = load_survey_single_bank(key, SURVEYS[key], bank_id)
            if rec is not None:
                bank_recs.append(rec)
                logger.info("  %s bank %d: %d samples (%.1fs)",
                            key, bank_id, rec.get_num_samples(),
                            rec.get_total_duration())

        if not bank_recs:
            logger.warning("  No recordings for bank %d, skipping", bank_id)
            continue

        # Concatenate in time
        concatenated = si.concatenate_recordings(bank_recs)
        logger.info("  Concatenated: %d ch x %.1fs",
                     concatenated.get_num_channels(),
                     concatenated.get_total_duration())

        # Write per-bank binary
        binary_dir = bank_dir / 'preprocessed'
        if (binary_dir / 'si_folder.json').exists():
            recording_saved = si.load_extractor(binary_dir)
            logger.info("  Reusing existing binary")
        else:
            logger.info("  Writing per-bank binary...")
            recording_saved = concatenated.save(
                folder=binary_dir, n_jobs=4, chunk_duration='5s',
                progress_bar=True)

        # Run KS2 via Docker (output dir separate from binary dir)
        ks2_sort_dir = bank_dir / 'ks2_sort'
        logger.info("  Running KS2 (%d ch, %.1fs)...",
                     recording_saved.get_num_channels(),
                     recording_saved.get_total_duration())
        t0 = time.time()
        try:
            sorting = si.run_sorter(
                'kilosort2', recording_saved,
                folder=ks2_sort_dir,
                remove_existing_folder=True,
                docker_image='spikeinterface/kilosort2-compiled-base:latest',
                installation_mode='pypi',
                spikeinterface_version='0.103.2',
                verbose=True,
                detect_threshold=5,
                nfilt_factor=8,
                minFR=0.002,
                minfr_goodchannels=0.01,
                sigmaMask=60,
                car=False,
            )
            elapsed = time.time() - t0
            n_units = len(sorting.unit_ids)
            logger.info("  Bank %d KS2 complete in %.0fs: %d units",
                         bank_id, elapsed, n_units)
            all_sortings.append((bank_id, sorting))
        except Exception as e:
            logger.error("  Bank %d KS2 FAILED: %s", bank_id, e)

    if not all_sortings:
        raise RuntimeError("No banks sorted successfully")

    # Merge sortings: remap unit IDs to be unique across banks
    logger.info("\nMerging %d bank sortings...", len(all_sortings))
    merged_units = {}
    merged_trains = {}
    fs = all_sortings[0][1].get_sampling_frequency()

    for bank_id, sorting in all_sortings:
        for uid in sorting.unit_ids:
            new_id = bank_unit_offset + uid
            train = sorting.get_unit_spike_train(uid, segment_index=0)
            merged_units[new_id] = bank_id
            merged_trains[new_id] = train
        bank_unit_offset += max(sorting.unit_ids) + 1 if len(sorting.unit_ids) > 0 else 0

    # Build merged NumpySorting
    from spikeinterface.core import NumpySorting
    total_samples = max(t.max() for t in merged_trains.values()) + 1 if merged_trains else 0
    merged_sorting = NumpySorting.from_unit_dict(
        merged_trains, sampling_frequency=fs)

    logger.info("  Merged: %d units across %d banks",
                 len(merged_units), len(all_sortings))

    # Save bank mapping for post-processing
    import pandas as pd
    pd.DataFrame([
        {'unit_id': uid, 'bank': bank_id}
        for uid, bank_id in merged_units.items()
    ]).to_csv(output_dir / 'ks2_bank_mapping.csv', index=False)

    return merged_sorting


def assemble_all_surveys():
    """Load all surveys, stack banks, concatenate in time order.

    Returns the concatenated recording (2,304 channels) and metadata.
    """
    import spikeinterface.full as si

    survey_keys = sorted(SURVEYS.keys())
    survey_recs = []
    survey_meta = []

    for idx, key in enumerate(survey_keys):
        logger.info("  [%d/%d] Loading survey %s...", idx + 1, len(survey_keys), key)
        t0 = time.time()
        rec = load_survey_stacked(key, SURVEYS[key])
        elapsed = time.time() - t0
        survey_recs.append(rec)
        survey_meta.append({
            'key': key,
            'dirname': SURVEYS[key],
            'n_samples': rec.get_num_samples(),
            'n_channels': rec.get_num_channels(),
            'duration_s': rec.get_total_duration(),
        })
        logger.info("    Done in %.0fs", elapsed)

    # Concatenate in time
    logger.info("  Concatenating %d surveys in time...", len(survey_recs))
    concatenated = si.concatenate_recordings(survey_recs)

    total_dur = sum(m['duration_s'] for m in survey_meta)
    logger.info("  Final recording: %d channels x %.1fs (%.1f min)",
                 concatenated.get_num_channels(), total_dur, total_dur / 60)

    return concatenated, survey_meta


# ── Phase 2: Sort ──────────────────────────────────────────────────────

def run_ks4_sort(recording, output_dir, torch_device):
    """Write preprocessed recording to disk and run KS4."""
    import spikeinterface.full as si

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sort_dir = output_dir / 'ks4_output'
    existing_sorter = sort_dir / 'sorter_output'
    if existing_sorter.exists() and (existing_sorter / 'spike_clusters.npy').exists():
        logger.info("  Loading existing KS4 output from %s", sort_dir)
        try:
            sorting = si.read_sorter_folder(sort_dir)
            n_units = len(sorting.unit_ids)
            total_spikes = sum(sorting.count_num_spikes_per_unit().values())
            logger.info("  Loaded: %d units, %d spikes", n_units, total_spikes)
            return sorting
        except Exception as e:
            logger.warning("  Failed to load existing sort: %s. Re-sorting...", e)

    # Write preprocessed data to binary
    binary_dir = output_dir / 'preprocessed_binary'
    if not (binary_dir / 'si_folder.json').exists():
        logger.info("  Writing preprocessed data to binary (%d channels)...",
                     recording.get_num_channels())
        t0 = time.time()
        recording_saved = recording.save(folder=binary_dir, n_jobs=4, chunk_duration='5s',
                                          progress_bar=True)
        elapsed = time.time() - t0
        logger.info("  Binary write complete in %.0fs", elapsed)
    else:
        logger.info("  Loading preprocessed binary from %s", binary_dir)
        recording_saved = si.load_extractor(binary_dir)

    # KS4 parameters
    ks_kwargs = dict(
        verbose=True,
        torch_device=torch_device,
        nblocks=1,
        templates_from_data=False,
        batch_size=30000,
        dminx=103,
        max_channel_distance=150,
        nearest_chans=20,
    )

    logger.info("  Running KS4 (nblocks=1, %d channels, %.1fs)...",
                 recording_saved.get_num_channels(),
                 recording_saved.get_total_duration())
    logger.info("  KS4 params: %s", ks_kwargs)
    t0 = time.time()

    try:
        sorting = si.run_sorter(
            'kilosort4', recording_saved,
            folder=sort_dir,
            remove_existing_folder=True,
            **ks_kwargs,
        )
        elapsed = time.time() - t0
        n_units = len(sorting.unit_ids)
        total_spikes = sum(sorting.count_num_spikes_per_unit().values())
        logger.info("  KS4 complete in %.0fs: %d units, %d spikes",
                     elapsed, n_units, total_spikes)
        return sorting
    except Exception as e:
        logger.error("  KS4 FAILED: %s", e)
        raise


def run_ks2_sort(recording, output_dir):
    """Run Kilosort 2 via Docker on the same preprocessed binary."""
    import spikeinterface.full as si

    output_dir = Path(output_dir)
    sort_dir = output_dir / 'ks2_output'
    existing_sorter = sort_dir / 'sorter_output'
    if existing_sorter.exists() and (existing_sorter / 'spike_clusters.npy').exists():
        logger.info("  Loading existing KS2 output from %s", sort_dir)
        try:
            sorting = si.read_sorter_folder(sort_dir)
            n_units = len(sorting.unit_ids)
            total_spikes = sum(sorting.count_num_spikes_per_unit().values())
            logger.info("  Loaded: %d units, %d spikes", n_units, total_spikes)
            return sorting
        except Exception as e:
            logger.warning("  Failed to load existing KS2 sort: %s. Re-sorting...", e)

    # Load preprocessed binary
    binary_dir = output_dir / 'preprocessed_binary'
    if (binary_dir / 'si_folder.json').exists():
        recording_saved = si.load_extractor(binary_dir)
    else:
        logger.info("  Writing preprocessed data to binary (%d channels)...",
                     recording.get_num_channels())
        t0 = time.time()
        recording_saved = recording.save(folder=binary_dir, n_jobs=4, chunk_duration='5s',
                                          progress_bar=True)
        logger.info("  Binary write complete in %.0fs", time.time() - t0)

    logger.info("  Running KS2 via Docker (%d channels, %.1fs)...",
                 recording_saved.get_num_channels(),
                 recording_saved.get_total_duration())
    t0 = time.time()

    try:
        sorting = si.run_sorter(
            'kilosort2', recording_saved,
            folder=sort_dir,
            remove_existing_folder=True,
            docker_image='spikeinterface/kilosort2-compiled-base:latest',
            installation_mode='pypi',
            spikeinterface_version='0.103.2',
            verbose=True,
            # Tuned for 2688-channel mega recording (7 banks × 384 ch)
            detect_threshold=5,           # more sensitive (default 6)
            nfilt_factor=8,               # more templates for 7 banks (default 4)
            minFR=0.002,                  # keep low-FR units in short surveys (default 0.02)
            minfr_goodchannels=0.01,      # don't discard sparse banks (default 0.1)
            sigmaMask=60,                 # wider spatial mask for NP1032 2-col (default 30)
            car=False,                    # already CAR'd in preprocessing
        )
        elapsed = time.time() - t0
        n_units = len(sorting.unit_ids)
        total_spikes = sum(sorting.count_num_spikes_per_unit().values())
        logger.info("  KS2 complete in %.0fs: %d units, %d spikes",
                     elapsed, n_units, total_spikes)
        return sorting
    except Exception as e:
        logger.error("  KS2 FAILED: %s", e)
        raise


# ── Phase 3: Analysis ──────────────────────────────────────────────────

def build_mega_analyzer(sorting, recording, sort_output_dir):
    """Build SortingAnalyzer and compute quality + waveform metrics."""
    from chronic_pixel_pipeline.analysis import build_sorting_analyzer, classify_units

    cache_dir = Path(sort_output_dir) / 'analyzer_cache'

    logger.info("  Building SortingAnalyzer (%d units)...", len(sorting.unit_ids))
    t0 = time.time()
    analyzer = build_sorting_analyzer(sorting, recording, cache_dir=cache_dir)
    elapsed = time.time() - t0
    logger.info("  Analyzer built in %.0fs", elapsed)

    # Classify units (baseline quality gate)
    good_ids, snr_thresh = classify_units(analyzer)
    logger.info("  Baseline gate: %d/%d good units (SNR>%.2f)",
                 len(good_ids), len(sorting.unit_ids), snr_thresh)

    return analyzer, good_ids, snr_thresh


def compute_waveform_quality(analyzer, sort_output_dir):
    """Compute waveform quality metrics and boundary noise flags.

    Adapted from precompute_waveform_quality.py for a single mega analyzer.
    """
    from precompute_waveform_quality import compute_flag

    cache_dir = Path(sort_output_dir) / 'analyzer_cache'

    # Load cached arrays
    templates = np.load(cache_dir / 'extensions/templates/average.npy')
    sparsity = np.load(cache_dir / 'sparsity_mask.npy')
    wf = np.load(cache_dir / 'extensions/waveforms/waveforms.npy', mmap_mode='r')
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

    # Firing rate from quality_metrics
    qm_path = cache_dir / 'extensions/quality_metrics/metrics.csv'
    qm = pd.read_csv(qm_path, index_col=0) if qm_path.exists() else None

    rs_unit_idx = spikes['unit_index'][rs_indices]

    # ACG bin masks
    bin_centers = (bins[:-1] + bins[1:]) / 2
    refr_mask = np.abs(bin_centers) <= 2
    shoulder_mask = (np.abs(bin_centers) >= 5) & (np.abs(bin_centers) <= 20)

    logger.info("  Computing waveform quality for %d units...", n_units)
    rows = []
    for uid_idx in range(n_units):
        uid = unit_ids[uid_idx]

        # peak_amplitude & n_spatial_channels
        ptp_per_ch = np.ptp(templates[uid_idx], axis=0)
        peak_ch = int(np.argmax(ptp_per_ch))
        peak_amp = float(ptp_per_ch[peak_ch])
        n_spat = int(np.sum(ptp_per_ch > 0.25 * peak_amp))

        # max_pointwise_snr
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

        # amp_cv
        spike_mask = spikes['unit_index'] == uid_idx
        unit_amps = amplitudes[spike_mask]
        if len(unit_amps) >= 5:
            amp_mean = np.abs(np.mean(unit_amps))
            amp_cv = float(np.std(unit_amps) / amp_mean) if amp_mean > 0 else np.nan
        else:
            amp_cv = np.nan

        # acg_refr_ratio
        auto_ccg = ccgs[uid_idx, uid_idx, :]
        refr_count = float(np.sum(auto_ccg[refr_mask]))
        shoulder_count = float(np.sum(auto_ccg[shoulder_mask]))
        acg_ratio = refr_count / shoulder_count if shoulder_count > 0 else np.nan

        # firing_rate
        fr = float(qm.loc[uid, 'firing_rate']) if (
            qm is not None and uid in qm.index) else np.nan

        rows.append({
            'unit_id': uid,
            'peak_amplitude': round(peak_amp, 2),
            'n_spatial_channels': n_spat,
            'max_pointwise_snr': round(max_snr, 4) if not np.isnan(max_snr) else np.nan,
            'amp_cv': round(amp_cv, 4) if not np.isnan(amp_cv) else np.nan,
            'acg_refr_ratio': round(acg_ratio, 4) if not np.isnan(acg_ratio) else np.nan,
            'firing_rate': round(fr, 4) if not np.isnan(fr) else np.nan,
        })

        if (uid_idx + 1) % 200 == 0:
            logger.info("    %d/%d units processed", uid_idx + 1, n_units)

    df = pd.DataFrame(rows).set_index('unit_id')
    df['boundary_noise_flag'] = compute_flag(df)

    # Save
    out_dir = cache_dir / 'extensions' / 'waveform_quality'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'metrics.csv')

    n_flagged = int(df['boundary_noise_flag'].sum())
    logger.info("  Waveform quality: %d/%d flagged as boundary noise",
                 n_flagged, len(df))

    return df


def build_unit_table(analyzer, wq_df, good_ids, output_dir, suffix=''):
    """Build a comprehensive unit table with bank assignment, depth, and quality."""
    sorting = analyzer.sorting
    templates = analyzer.get_extension('templates').get_data()
    ch_locs = analyzer.get_channel_locations()

    fs = analyzer.sampling_frequency
    total_duration = analyzer.get_num_samples() / fs

    rows = []
    for uid_idx, uid in enumerate(sorting.unit_ids):
        template = templates[uid_idx]
        ptp_per_ch = np.ptp(template, axis=0)
        peak_ch = int(np.argmax(ptp_per_ch))
        depth = float(ch_locs[peak_ch, 1])
        amplitude = float(ptp_per_ch[peak_ch])

        # Determine which bank this unit belongs to based on depth
        bank_id = int(depth // (192 * 20))  # 192 rows * 20µm pitch per bank
        bank_id = min(bank_id, max(BANKS))

        spike_train = sorting.get_unit_spike_train(uid, segment_index=0)
        fr = len(spike_train) / total_duration

        is_baseline_good = uid in good_ids
        is_noise = bool(wq_df.loc[uid, 'boundary_noise_flag']) if uid in wq_df.index else False
        is_good = is_baseline_good and not is_noise

        rows.append({
            'unit_id': uid,
            'bank': bank_id,
            'depth_um': round(depth, 1),
            'amplitude_uv': round(amplitude, 2),
            'firing_rate': round(fr, 4),
            'n_spikes': len(spike_train),
            'is_baseline_good': is_baseline_good,
            'boundary_noise_flag': is_noise,
            'is_good': is_good,
            'peak_amplitude': wq_df.loc[uid, 'peak_amplitude'] if uid in wq_df.index else np.nan,
            'n_spatial_channels': int(wq_df.loc[uid, 'n_spatial_channels']) if uid in wq_df.index else 0,
            'max_pointwise_snr': wq_df.loc[uid, 'max_pointwise_snr'] if uid in wq_df.index else np.nan,
            'amp_cv': wq_df.loc[uid, 'amp_cv'] if uid in wq_df.index else np.nan,
        })

    df = pd.DataFrame(rows)
    fname = f'mega_units{suffix}.csv' if suffix else 'mega_units.csv'
    out_path = Path(output_dir) / fname
    df.to_csv(out_path, index=False)

    n_good = df['is_good'].sum()
    n_baseline = df['is_baseline_good'].sum()
    logger.info("  Unit table: %d total, %d baseline good, %d strict good",
                 len(df), n_baseline, n_good)
    logger.info("  Depth range: %.0f - %.0f µm", df['depth_um'].min(), df['depth_um'].max())
    for bank_id in sorted(df['bank'].unique()):
        bdf = df[df['bank'] == bank_id]
        logger.info("    Bank %d: %d units (%d good)",
                     bank_id, len(bdf), bdf['is_good'].sum())

    return df


# ── Phase 4: Figures ───────────────────────────────────────────────────

def generate_figures(analyzer, unit_df, fig_dir):
    """Generate per-unit summary figures, split into good vs suspect."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_single_neuron_plots import plot_custom_unit_summary

    fig_dir = Path(fig_dir)
    good_dir = fig_dir / 'units'
    suspect_dir = fig_dir / 'units_suspect'
    good_dir.mkdir(parents=True, exist_ok=True)
    suspect_dir.mkdir(parents=True, exist_ok=True)

    baseline_good = set(unit_df[unit_df['is_baseline_good']]['unit_id'])
    strict_good = set(unit_df[unit_df['is_good']]['unit_id'])
    suspect = baseline_good - strict_good

    all_plot_units = sorted(baseline_good)
    n_total = len(all_plot_units)
    n_good = len(strict_good)
    n_suspect = len(suspect)

    logger.info("  Generating figures: %d good + %d suspect = %d total",
                 n_good, n_suspect, n_total)

    for count, uid in enumerate(all_plot_units):
        row = unit_df[unit_df['unit_id'] == uid].iloc[0]
        bank_id = int(row['bank'])

        if uid in strict_good:
            out_dir = good_dir
        else:
            out_dir = suspect_dir

        fig = plt.figure(figsize=(18, 6), dpi=100)
        try:
            plot_custom_unit_summary(analyzer, uid, fig)
            fig.suptitle(f"Unit {uid} | Bank {bank_id} | "
                         f"Depth {row['depth_um']:.0f}µm | "
                         f"FR {row['firing_rate']:.2f}Hz | "
                         f"Amp {row['amplitude_uv']:.0f}µV",
                         fontsize=11, y=0.98)
            out_path = out_dir / f"unit_{uid:04d}_bank{bank_id:02d}.png"
            fig.savefig(out_path, bbox_inches='tight', dpi=100)
        except Exception as e:
            logger.warning("  Unit %d figure failed: %s", uid, e)
        finally:
            plt.close(fig)

        if (count + 1) % 50 == 0 or count + 1 == n_total:
            logger.info("    Figures: %d/%d done", count + 1, n_total)


def generate_overview_figure(analyzer, unit_df, fig_dir, suffix=''):
    """Generate full-probe overview figure showing all units by depth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    good = unit_df[unit_df['is_good']]
    suspect = unit_df[unit_df['is_baseline_good'] & ~unit_df['is_good']]

    fig, axes = plt.subplots(1, 3, figsize=(14, 20))

    # Panel 1: Unit locations by depth
    ax = axes[0]
    if len(suspect) > 0:
        ax.scatter(np.zeros(len(suspect)), suspect['depth_um'],
                   c='orange', s=10, alpha=0.5, label='Suspect')
    if len(good) > 0:
        ax.scatter(np.zeros(len(good)), good['depth_um'],
                   c='blue', s=10, alpha=0.7, label='Good')
    ax.set_ylabel('Depth (µm)')
    ax.set_title(f'Unit Locations\n{len(good)} good, {len(suspect)} suspect')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # Add bank boundaries
    for bank_id in BANKS:
        y = bank_id * 192 * 20
        ax.axhline(y, color='grey', linewidth=0.5, linestyle='--')
        ax.text(0.5, y + 100, f'Bank {bank_id}', fontsize=7,
                ha='center', color='grey')

    # Panel 2: Firing rate by depth
    ax = axes[1]
    if len(good) > 0:
        ax.scatter(good['firing_rate'], good['depth_um'],
                   c='blue', s=10, alpha=0.7)
    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title('Firing Rate by Depth')
    ax.invert_yaxis()

    # Panel 3: Amplitude by depth
    ax = axes[2]
    if len(good) > 0:
        ax.scatter(good['amplitude_uv'], good['depth_um'],
                   c='blue', s=10, alpha=0.7)
    ax.set_xlabel('Amplitude (µV)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title('Amplitude by Depth')
    ax.invert_yaxis()

    label = f' ({suffix.upper()})' if suffix else ''
    fig.suptitle(f'Mega-Sort: Full Probe Overview (Banks 0-{max(BANKS)}){label}',
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f'mega_full_probe{("_" + suffix) if suffix else ""}.png'
    fig.savefig(fig_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved %s", fname)


# ── Main ───────────────────────────────────────────────────────────────

def _load_ks2_per_bank_labels(output_dir):
    """Load KS2 'good' labels from per-bank sort outputs and build a set
    of merged unit IDs that KS2 labelled as good."""
    good_uids = set()
    bank_unit_offset = 0
    for bank_id in BANKS:
        label_path = (Path(output_dir) / f'ks2_bank{bank_id:02d}' /
                      'ks2_sort' / 'sorter_output' / 'cluster_KSLabel.tsv')
        if not label_path.exists():
            continue
        df = pd.read_csv(label_path, sep='\t')
        for _, row in df.iterrows():
            if row['KSLabel'] == 'good':
                good_uids.add(bank_unit_offset + row['cluster_id'])
        max_id = int(df['cluster_id'].max()) + 1 if len(df) > 0 else 0
        bank_unit_offset += max_id
    return good_uids


def _run_postprocessing(sorting, recording, output_dir, fig_dir, suffix,
                        skip_figures=False):
    """Run analysis + figures for one sorter's output."""
    sort_output = Path(output_dir) / f'{suffix}_output'
    cache_dir = sort_output / 'analyzer_cache' if sort_output.exists() else Path(output_dir) / f'{suffix}_output'

    logger.info("  --- Post-processing %s ---", suffix.upper())
    analyzer, good_ids, snr_thresh = build_mega_analyzer(
        sorting, recording, sort_output)

    wq_df = compute_waveform_quality(analyzer, sort_output)
    unit_df = build_unit_table(analyzer, wq_df, good_ids, output_dir,
                                suffix=f'_{suffix}')

    # For KS2 per-bank: override is_good with KS2's own labels
    if suffix == 'ks2':
        ks2_good = _load_ks2_per_bank_labels(output_dir)
        if ks2_good:
            unit_df['is_good'] = unit_df['unit_id'].isin(ks2_good)
            unit_df['is_baseline_good'] = unit_df['is_good']
            fname = f'mega_units_{suffix}.csv'
            unit_df.to_csv(Path(output_dir) / fname, index=False)
            n_good = unit_df['is_good'].sum()
            logger.info("  Overrode with KS2 labels: %d/%d good", n_good, len(unit_df))
            for bank_id in sorted(unit_df['bank'].unique()):
                bdf = unit_df[unit_df['bank'] == bank_id]
                logger.info("    Bank %d: %d units (%d good)",
                             bank_id, len(bdf), bdf['is_good'].sum())

    if not skip_figures:
        sorter_fig_dir = Path(fig_dir) if suffix == 'ks4' else Path(str(fig_dir) + f'_{suffix}')
        generate_overview_figure(analyzer, unit_df, sorter_fig_dir, suffix=suffix)
        generate_figures(analyzer, unit_df, sorter_fig_dir)

    return unit_df


def main():
    parser = argparse.ArgumentParser(
        description='Mega-sort: unified spike sorting across all banks and sessions')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--ks4-only', action='store_true',
                        help='Only run KS4 (skip KS2)')
    parser.add_argument('--ks2-only', action='store_true',
                        help='Only run KS2 (assumes preprocessed binary exists)')
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device (default: auto-detect)')
    args = parser.parse_args()

    run_ks4 = not args.ks2_only
    run_ks2 = not args.ks4_only
    torch_device = args.device or _detect_torch_device()
    t_start = time.time()

    sorters = []
    if run_ks4:
        sorters.append('KS4')
    if run_ks2:
        sorters.append('KS2')

    logger.info("=" * 70)
    logger.info("MEGA-SORT: %s across all banks and sessions", ' + '.join(sorters))
    logger.info("=" * 70)
    logger.info("DATA_ROOT: %s", DATA_ROOT)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Banks: 0-%d (%d channels)", max(BANKS), len(BANKS) * N_CHANNELS_PER_BANK)
    logger.info("Surveys: %s", ', '.join(sorted(SURVEYS.keys())))
    logger.info("")

    import spikeinterface.full as si
    binary_dir = OUTPUT_DIR / 'preprocessed_binary'

    # ── Phase 1: Load and assemble ──
    if not (binary_dir / 'si_folder.json').exists():
        logger.info("[1] LOADING — assembling multi-session recording")
        logger.info("=" * 70)
        concatenated, survey_meta = assemble_all_surveys()
        pd.DataFrame(survey_meta).to_csv(OUTPUT_DIR / 'survey_metadata.csv', index=False)
        logger.info("")
    else:
        logger.info("[1] LOADING — reusing existing preprocessed binary")
        concatenated = None
        logger.info("")

    # ── Phase 2a: KS4 Sort ──
    ks4_sorting = None
    if run_ks4:
        logger.info("[2a] SORTING — running KS4 with nblocks=1")
        logger.info("=" * 70)
        rec_for_sort = concatenated if concatenated is not None else si.load_extractor(binary_dir)
        ks4_sorting = run_ks4_sort(rec_for_sort, OUTPUT_DIR, torch_device)
        logger.info("")

    # ── Phase 2b: KS2 Sort (per-bank) ──
    ks2_sorting = None
    if run_ks2:
        logger.info("[2b] SORTING — running KS2 per-bank via Docker")
        logger.info("=" * 70)
        ks2_sorting = run_ks2_per_bank(OUTPUT_DIR)
        logger.info("")

    # ── Phase 3: Analysis ──
    logger.info("[3] ANALYSIS — building SortingAnalyzer and quality metrics")
    logger.info("=" * 70)

    recording_for_analyzer = si.load_extractor(binary_dir) if (binary_dir / 'si_folder.json').exists() else concatenated

    if ks4_sorting is not None:
        ks4_df = _run_postprocessing(
            ks4_sorting, recording_for_analyzer, OUTPUT_DIR, FIG_DIR, 'ks4',
            skip_figures=args.skip_figures)

    if ks2_sorting is not None:
        ks2_df = _run_postprocessing(
            ks2_sorting, recording_for_analyzer, OUTPUT_DIR, FIG_DIR, 'ks2',
            skip_figures=args.skip_figures)

    # ── Summary ──
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("MEGA-SORT COMPLETE")
    logger.info("=" * 70)
    logger.info("Total time: %.0f min", elapsed / 60)
    if ks4_sorting is not None:
        logger.info("KS4: %d total, %d good", len(ks4_df), ks4_df['is_good'].sum())
    if ks2_sorting is not None:
        logger.info("KS2: %d total, %d good", len(ks2_df), ks2_df['is_good'].sum())
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Figures: %s", FIG_DIR)


if __name__ == '__main__':
    main()
