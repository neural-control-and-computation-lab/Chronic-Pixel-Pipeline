"""
Estimate chronic probe drift across sessions with a single KS4 drift-only run.

Strategy (depth-range mode, the recommended workflow):
  1. For each session, take 5 × 2 s slices evenly distributed across the
     stable window of every bank that contributes channels to the requested
     depth band. Concatenate the 5 slices in time and aggregate the
     contributing banks as CHANNELS, so each session becomes a 10 s
     recording covering the depth band.
  2. Concatenate sessions in time. KS4 sees one long 'super-recording'
     where each 2 s batch belongs to exactly one session (5 batches/session).
  3. Call kilosort directly (initialize_ops → compute_preprocessing →
     compute_drift_correction). Stop before spike extraction / clustering.
     `ops['dshift']` ends up shape (n_batches,) → 5 estimates per session.
  4. Plot session-median drift across days since surgery, raster of detected
     spikes, and a side-by-side raw-vs-corrected raster (each session's
     depths shifted by +session_median).

Canonical command (the 'final' parameter set):
  python drift_concat.py --preset chronic

Equivalent to:
  python drift_concat.py --depth-range 12000 16000 --segment-s 10 \\
      --th-universal 14 --binning-depth 30 --distribute-batches \\
      --plot-median --exclude 20260226

Outputs (per run, with tag suffix encoding the parameters):
  E:/sorting_output/drift_concat/{tag}_ks4/sorter_output/{ops.npy,st0.npy}
  E:/figures/sorting/drift_concat/{tag}_drift_median.{png,svg}
  E:/figures/sorting/drift_concat/{tag}_raster.{png,svg}
  E:/figures/sorting/drift_concat/{tag}_raster_corrected.{png,svg}
  E:/sorting_output/drift_concat/segment_boundaries.csv

Other modes:
  --single-bank N           one bank only, no depth restriction
  (no flag)                 stack ALL 7 banks as channels (super-probe)

Usage:
  python drift_concat.py --preset chronic
  python drift_concat.py --preset chronic --exclude 20260226 20260512
  python drift_concat.py --depth-range 14000 18000 --preset chronic
  python drift_concat.py --single-bank 3 --th-universal 14 --binning-depth 30
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sort_survey import (SURVEYS as SURVEY_DIRS,
                          make_bank_probe, find_meta_file,
                          parse_meta, parse_svySBTT,
                          _detect_torch_device)

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
FIG_ROOT = DATA_ROOT / 'figures' / 'sorting'
OUT_DIR = SORT_ROOT / 'drift_concat'
FIG_DIR = FIG_ROOT / 'drift_concat'

N_BANKS = 7
CH_PER_BANK = 384
TOTAL_CH = N_BANKS * CH_PER_BANK

# Channel pitch in µm (NP1032 column spacing)
SURGERY_DATE = datetime(2026, 2, 19)

# Default excluded surveys (first post-surgery session is too unstable to
# participate in cross-session drift estimation).
DEFAULT_EXCLUDE = {'20260226'}

# Recommended chronic-drift parameters (the 'final' validated config).
CHRONIC_PRESET = {
    'depth_range': (12000.0, 16000.0),  # ventral cortex / hippocampus band
    'segment_s': 10.0,                    # 5 × 2 s batches/session
    'th_universal': 14.0,                 # large-amp spikes only
    'binning_depth': 30.0,                # coarse spatial bins
    'distribute_batches': True,           # spread batches across stable window
    'plot_median': True,                  # one median point per session
}


def _patch_zero_channel_pad():
    """Workaround for two SpikeInterface bugs in zero_channel_pad. Same as
    earlier — needed only if a bank has < 384 active channels.
    """
    from spikeinterface.preprocessing.zero_channel_pad import (
        ZeroChannelPaddedRecording, ZeroChannelPaddedRecordingSegment)
    _orig_init = ZeroChannelPaddedRecording.__init__

    def _patched_init(self, recording=None, num_channels=None,
                       channel_mapping=None, parent_recording=None):
        rec = recording if recording is not None else parent_recording
        if channel_mapping is not None:
            self.channel_mapping = np.asarray(channel_mapping)
        _orig_init(self, rec, num_channels, channel_mapping=channel_mapping)

    def _patched_get_traces(self, start_frame, end_frame, channel_indices):
        parent_n_ch = len(self.channel_mapping)
        parent_traces = self.parent_recording_segment.get_traces(
            start_frame=start_frame, end_frame=end_frame,
            channel_indices=np.arange(parent_n_ch))
        n_t = parent_traces.shape[0]
        traces = np.zeros((n_t, self.num_channels), dtype=parent_traces.dtype)
        traces[:, self.channel_mapping] = parent_traces
        if channel_indices is None:
            return traces
        return traces[:, channel_indices]

    ZeroChannelPaddedRecording.__init__ = _patched_init
    ZeroChannelPaddedRecordingSegment.get_traces = _patched_get_traces


_patch_zero_channel_pad()


# ────────────────────────────────────────────────────────────────────────
# Probe construction
# ────────────────────────────────────────────────────────────────────────
def build_super_probe_positions():
    """Return (xc, yc) numpy arrays for a 2688-channel super-probe spanning
    banks 0..N_BANKS-1. Channel order: bank 0 chan 0..383, then bank 1, ...
    """
    xc = np.zeros(TOTAL_CH, dtype=np.float32)
    yc = np.zeros(TOTAL_CH, dtype=np.float32)
    for b in range(N_BANKS):
        probe, n_ch = make_bank_probe(b)
        pos = probe.contact_positions  # (n_ch, 2)
        sl = slice(b * CH_PER_BANK, b * CH_PER_BANK + n_ch)
        xc[sl] = pos[:n_ch, 0]
        yc[sl] = pos[:n_ch, 1]
        # If bank is partial, fill remaining slots with placeholder positions
        # well above the highest real depth so they don't influence drift.
        if n_ch < CH_PER_BANK:
            top_z = yc[:b * CH_PER_BANK + n_ch].max() + 20
            for i in range(n_ch, CH_PER_BANK):
                xc[b * CH_PER_BANK + i] = 11 if (i % 2 == 0) else 114
                yc[b * CH_PER_BANK + i] = top_z + (i - n_ch) * 20
    return xc, yc


def build_super_probe_si():
    """SI Probe object for the same 2688-channel super-probe (used to set
    on the aggregated recording before saving to binary)."""
    from probeinterface import Probe
    xc, yc = build_super_probe_positions()
    positions = np.column_stack([xc, yc])
    probe = Probe(ndim=2)
    probe.set_contacts(positions=positions, shapes='square',
                       shape_params={'width': 12})
    probe.set_device_channel_indices(np.arange(TOTAL_CH))
    probe.create_auto_shape(probe_type='rect')
    return probe


# ────────────────────────────────────────────────────────────────────────
# Build session recordings (banks-as-channels)
# ────────────────────────────────────────────────────────────────────────
def _bank_window(meta, bank_id):
    """Return (start_sample, end_sample) for bank's stable window."""
    fs = float(meta['imSampRate'])
    file_time = float(meta['fileTimeSecs'])
    total_samples = int(file_time * fs)
    banks_meta = parse_svySBTT(meta)
    for i, bk in enumerate(banks_meta):
        if bk['bank'] != bank_id:
            continue
        ds = bk['settle_end']
        de = (banks_meta[i + 1]['settle_start']
               if i + 1 < len(banks_meta) else total_samples)
        return ds, de
    return None


def build_session_recording(survey_key, segment_s, super_probe):
    """Build a single 2 s × 2688-channel recording for one session by
    stacking banks 0..N_BANKS-1 as channels. Missing banks get zero-padded.
    """
    import spikeinterface.full as si
    from chronic_pixel_pipeline.preprocessing import preprocess

    # sort_survey.SURVEYS is a flat {key: 'SvyPrb_xxx_g0'} mapping of the
    # raw data directory name relative to DATA_ROOT.
    survey_dir = DATA_ROOT / SURVEY_DIRS[survey_key]

    if not survey_dir.exists():
        logger.warning('  %s: missing %s', survey_key, survey_dir)
        return None, None

    meta = parse_meta(find_meta_file(survey_dir))
    fs = float(meta['imSampRate'])
    max_bank = int(meta['imSvyMaxBnk'])
    seg_samples = int(round(segment_s * fs))

    rec_full = si.read_spikeglx(str(survey_dir), stream_id='imec0.ap')

    bank_recs = []
    banks_present = []
    for b in range(N_BANKS):
        if b > max_bank:
            bank_recs.append(None)
            continue
        bw = _bank_window(meta, b)
        if bw is None:
            bank_recs.append(None)
            continue
        win_dur_samples = bw[1] - bw[0]
        if win_dur_samples < seg_samples:
            start = bw[0]
            end = bw[1]
        else:
            mid = (bw[0] + bw[1]) // 2
            start = mid - seg_samples // 2
            end = start + seg_samples

        rec_bank = rec_full.frame_slice(start_frame=start, end_frame=end)
        probe, actual_ch = make_bank_probe(b)
        rec_bank = rec_bank.set_probe(probe, group_mode='by_probe')
        if actual_ch < CH_PER_BANK:
            good = rec_bank.get_channel_ids()[:actual_ch]
            rec_bank = rec_bank.select_channels(channel_ids=good)
        rec_pp = preprocess(rec_bank, freq_min=300, freq_max=6000,
                             destripe=False, cmr=False, phase_shift=True)
        # Pad up to CH_PER_BANK
        if rec_pp.get_num_channels() < CH_PER_BANK:
            n = rec_pp.get_num_channels()
            rec_pp = si.zero_channel_pad(rec_pp, num_channels=CH_PER_BANK,
                                          channel_mapping=list(range(n)))
        # Trim to exactly seg_samples
        if rec_pp.get_num_samples() != seg_samples:
            rec_pp = rec_pp.frame_slice(start_frame=0, end_frame=seg_samples)
        bank_recs.append(rec_pp)
        banks_present.append(b)

    if not banks_present:
        return None, None

    # For missing banks, make a zero recording of the right shape using
    # zero_channel_pad on a present bank's recording.
    template = bank_recs[banks_present[0]]
    for b in range(N_BANKS):
        if bank_recs[b] is None:
            bank_recs[b] = si.zero_channel_pad(template,
                                                num_channels=CH_PER_BANK,
                                                channel_mapping=[])
            # That empty-mapping form is unusual; instead, pad an empty rec
            # with zeros via channel_mapping=[]. If SI doesn't accept that,
            # fall back to padding template (still all zeros are OK because
            # the actual values get overwritten by traces from template; we
            # really want zero traces here). Use a wrapper that zeroes out.
            bank_recs[b] = _zero_clone(template)

    # Aggregate banks as channels — strip per-bank probes first
    stripped = []
    for rec in bank_recs:
        rec_clean = _strip_probe(rec)
        stripped.append(rec_clean)
    rec_session = si.aggregate_channels(stripped)
    # Set the super-probe geometry
    rec_session = rec_session.set_probe(super_probe, group_mode='by_probe')
    return rec_session, banks_present


def _strip_probe(rec):
    """Return a copy of rec without probe metadata so aggregate_channels
    treats it as plain channels."""
    rec_clone = rec.clone()
    for k in ['contact_vector', 'location', 'group']:
        try:
            rec_clone.delete_property(k)
        except Exception:
            pass
    # Also clear the probe annotations
    try:
        rec_clone.clear_channel_locations()
    except Exception:
        pass
    try:
        rec_clone.clear_channel_groups()
    except Exception:
        pass
    return rec_clone


def _zero_clone(rec):
    """Return a recording with the same shape & probe as rec but all zeros."""
    from spikeinterface.core import NumpyRecording
    import numpy as np
    n_samp = rec.get_num_samples()
    n_ch = rec.get_num_channels()
    data = np.zeros((n_samp, n_ch), dtype='float32')
    out = NumpyRecording(traces_list=[data], sampling_frequency=rec.sampling_frequency)
    return out


# ────────────────────────────────────────────────────────────────────────
# Drift-only KS4
# ────────────────────────────────────────────────────────────────────────
def run_drift_only(rec_concat, super_probe_dict, out_dir, batch_size,
                    force=False, th_universal=None, binning_depth=None,
                    drift_smoothing=None, sig_interp=None, nblocks=None):
    """Write rec_concat to a binary file then run kilosort's drift-only
    sequence (initialize_ops, compute_preprocessing, compute_drift_correction).
    Saves ops.npy with dshift populated."""
    import torch
    from kilosort.run_kilosort import (initialize_ops, compute_preprocessing,
                                        compute_drift_correction)
    from kilosort.parameters import DEFAULT_SETTINGS

    out_dir = Path(out_dir)
    sorter_out = out_dir / 'sorter_output'
    sorter_out.mkdir(parents=True, exist_ok=True)
    ops_path = sorter_out / 'ops.npy'
    if not force and ops_path.exists():
        logger.info('  Cached ops.npy exists, skipping KS4')
        return

    # 1) Write the recording to a binary file
    bin_path = out_dir / 'recording.bin'
    logger.info('  Writing %d ch × %.1f s to %s ...',
                 rec_concat.get_num_channels(),
                 rec_concat.get_num_samples() / rec_concat.sampling_frequency,
                 bin_path)
    t0 = time.time()
    rec_concat.save(format='binary', folder=out_dir / 'binary_cache',
                     n_jobs=1, overwrite=True, dtype='int16')
    logger.info('  Wrote binary in %.1fs', time.time() - t0)
    bin_files = list((out_dir / 'binary_cache').rglob('traces_cached_seg0.raw'))
    if not bin_files:
        raise RuntimeError('binary write did not produce traces_cached_seg0.raw')
    bin_path = bin_files[0]

    fs = float(rec_concat.sampling_frequency)
    n_chan = rec_concat.get_num_channels()

    # 2) Build settings
    settings = DEFAULT_SETTINGS.copy()
    update = {
        'filename': str(bin_path),
        'n_chan_bin': n_chan,
        'fs': fs,
        'nblocks': nblocks if nblocks is not None else 1,
        'dminx': 103,
        'max_channel_distance': 150,
        'nearest_chans': 20,
        'templates_from_data': False,
    }
    if batch_size is not None:
        update['batch_size'] = batch_size
    if th_universal is not None:
        update['Th_universal'] = th_universal
    if binning_depth is not None:
        update['binning_depth'] = binning_depth
    if drift_smoothing is not None:
        update['drift_smoothing'] = drift_smoothing
    if sig_interp is not None:
        update['sig_interp'] = sig_interp
    settings.update(update)
    # Some settings must align with the probe channel count
    if settings['nearest_chans'] > n_chan:
        settings['nearest_chans'] = n_chan

    # 3) Probe + ops + drift
    device = _detect_torch_device()
    if isinstance(device, str):
        device = torch.device(device)
    logger.info('  Running KS4 drift-only on device=%s', device)
    ops, _settings = initialize_ops(
        settings, super_probe_dict, data_dtype='int16',
        do_CAR=True, invert_sign=False, device=device,
        save_preprocessed_copy=False)
    t1 = time.time()
    ops = compute_preprocessing(ops, device)
    logger.info('  Preprocessing: %.1fs', time.time() - t1)
    t1 = time.time()
    ops, bfile, st0 = compute_drift_correction(ops, device)
    logger.info('  Drift correction: %.1fs (dshift shape %s)',
                 time.time() - t1, np.asarray(ops['dshift']).shape)

    # 4) Save ops + detected spikes (st0) for downstream raster plotting
    np.save(ops_path, ops, allow_pickle=True)
    st0_path = sorter_out / 'st0.npy'
    np.save(st0_path, np.asarray(st0))
    logger.info('  Saved %s (%d detected spikes)', st0_path, len(st0))


def load_drift(out_dir):
    ops_path = Path(out_dir) / 'sorter_output' / 'ops.npy'
    if not ops_path.exists():
        return None
    ops = np.load(ops_path, allow_pickle=True).item()
    d = ops.get('dshift')
    if d is None:
        return None
    return {'dshift': np.asarray(d),
             'batch_size': int(ops.get('batch_size', 60000)),
             'fs': float(ops.get('fs', 30000.0))}


def load_spike_array(out_dir):
    p = Path(out_dir) / 'sorter_output' / 'st0.npy'
    if not p.exists():
        return None
    return np.load(p)


def plot_drift_corrected_raster(st0, dshift, survey_keys, segment_s,
                                  batch_size_s, out_png, title_suffix=''):
    """Stacked raw vs drift-corrected raster.

    For each spike we look up the session it falls in (based on its concat
    time), compute the MEDIAN dshift across that session's batches, and add
    it to the spike depth (KS4 sign convention: aligned_depth = raw + dshift).
    """
    if st0 is None or len(st0) == 0:
        logger.warning('No spikes for corrected raster.')
        return None

    n_batches = len(dshift)
    n_sessions = len(survey_keys)
    n_batch_per_session = max(1, int(round(segment_s / batch_size_s)))
    # Trust n_batch_per_session; otherwise assume evenly distributed
    if n_batches < n_sessions:
        logger.warning('Not enough batches (%d) for %d sessions',
                       n_batches, n_sessions)
        return None
    # If batches/session evenly divides
    if n_batches == n_sessions * n_batch_per_session:
        session_med = np.array([
            float(np.median(dshift[i * n_batch_per_session:
                                    (i + 1) * n_batch_per_session]))
            for i in range(n_sessions)])
    else:
        # Proportional split
        edges = np.linspace(0, n_batches, n_sessions + 1).astype(int)
        session_med = np.array([
            float(np.median(dshift[edges[i]:edges[i + 1]]))
            for i in range(n_sessions)])

    t_s = st0[:, 0]
    depth_raw = st0[:, 1]
    amp = st0[:, 2]

    session_idx = np.clip((t_s // segment_s).astype(int), 0, n_sessions - 1)
    depth_corr = depth_raw + session_med[session_idx]

    # Subsample
    n = len(t_s)
    if n > 300000:
        idx = np.random.default_rng(0).choice(n, 300000, replace=False)
        t_s = t_s[idx]; depth_raw = depth_raw[idx]
        depth_corr = depth_corr[idx]; amp = amp[idx]
    amp_norm = np.clip(np.abs(amp) / np.percentile(np.abs(amp), 99), 0, 1)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    titles = [
        'RAW depths (uncorrected)',
        f'DRIFT-CORRECTED depths (each session shifted by its '
        f'median dshift, range {session_med.min():+.1f} to '
        f'{session_med.max():+.1f} µm)']
    for ax, y, ttitle in zip(axes, [depth_raw, depth_corr], titles):
        ax.scatter(t_s, y, s=2.0, c=amp_norm, cmap='inferno_r',
                    vmin=0, vmax=1, alpha=0.5, marker='.',
                    rasterized=True, linewidths=0)
        for i in range(n_sessions + 1):
            x = i * segment_s
            if i > 0:
                ax.axvline(x, color='#888888', lw=0.4, alpha=0.7, zorder=2)
            if i < n_sessions:
                mid = x + segment_s / 2
                ax.text(mid, y.max() + (y.max() - y.min()) * 0.02,
                         survey_keys[i][4:], rotation=90, fontsize=6,
                         ha='center', va='bottom', color='#555555')
        ax.set_ylabel('Spike depth (µm)')
        ax.set_title(ttitle, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[-1].set_xlabel('Concatenated time (s) — each segment = one session')
    axes[-1].set_xlim(0, n_sessions * segment_s)
    y_min = min(depth_raw.min(), depth_corr.min())
    y_max = max(depth_raw.max(), depth_corr.max())
    pad = (y_max - y_min) * 0.03
    for ax in axes:
        ax.set_ylim(y_min - pad, y_max + pad)

    fig.suptitle('Drift-correction test — raw vs corrected spike depths' +
                  (f'\n{title_suffix}' if title_suffix else ''),
                  fontsize=12, fontweight='bold', y=1.0)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=140, bbox_inches='tight', facecolor='white')
    fig.savefig(out_png.with_suffix('.svg'), bbox_inches='tight',
                 facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', out_png)
    return session_med


def plot_spike_raster(st0, fs, survey_keys, segment_s, out_png,
                       title='Spike raster (KS4 universal-template detection)'):
    """Scatter plot: x=concatenated time (seconds), y=spike depth (µm).
    st0 columns from KS4: [time_sample, depth, amplitude, template_id, batch, ...]
    """
    if st0 is None or len(st0) == 0:
        logger.warning('No spikes to plot.')
        return

    # KS4's st0 columns: [time_seconds, depth_µm, amplitude, template_id, batch, channel]
    t_s = st0[:, 0]
    depth = st0[:, 1]
    amp = st0[:, 2]

    fig, ax = plt.subplots(figsize=(15, 6))
    # Subsample if too many
    n = len(t_s)
    if n > 300000:
        idx = np.random.default_rng(0).choice(n, 300000, replace=False)
        t_s = t_s[idx]; depth = depth[idx]; amp = amp[idx]
    amp_norm = np.clip(np.abs(amp) / np.percentile(np.abs(amp), 99), 0, 1)
    # Larger markers + constant alpha so sparse later sessions stay visible
    ax.scatter(t_s, depth, s=2.0, c=amp_norm, cmap='inferno_r',
                vmin=0, vmax=1, alpha=0.5, marker='.',
                rasterized=True, linewidths=0)

    # Session boundary lines + date labels
    n_sessions = len(survey_keys)
    cum = 0.0
    for i in range(n_sessions + 1):
        if i > 0:
            ax.axvline(cum, color='#888888', lw=0.4, alpha=0.7, zorder=2)
        if i < n_sessions:
            mid = cum + segment_s / 2
            ax.text(mid, depth.max() + (depth.max() - depth.min()) * 0.02,
                     survey_keys[i][4:], rotation=90, fontsize=6,
                     ha='center', va='bottom', color='#555555')
        cum += segment_s

    ax.set_xlabel('Concatenated time (s) — each segment is one session')
    ax.set_ylabel('Spike depth on probe (µm)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, cum)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    fig.savefig(out_png.with_suffix('.svg'), bbox_inches='tight',
                 facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', out_png)


# ────────────────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────────────────
def plot_drift(drift, survey_order, out_png, segment_s=None, mode='per_batch'):
    """mode: 'per_batch' (default, every batch as its own marker) or
    'median' (one median per session, with a thin vertical bar showing min/max).
    """
    dshift = drift['dshift']
    if dshift.ndim == 2 and dshift.shape[1] == 1:
        dshift = dshift[:, 0]

    n_batches = dshift.shape[0]
    n_sessions = len(survey_order)
    dates = [datetime.strptime(k, '%Y%m%d') for k in survey_order]
    days = np.array([(d - SURGERY_DATE).days for d in dates])

    # If batches don't match sessions, aggregate batches per session via the
    # known seconds-per-session (segment_s). batches_per_session = batch
    # count divided across sessions in proportion to segment lengths.
    per_session = None  # list of per-session arrays of batch values
    if dshift.ndim == 1:
        if n_batches == n_sessions:
            per_session = [np.atleast_1d(dshift[i]) for i in range(n_sessions)]
        else:
            # Map each batch to a session by proportional position
            edges = np.linspace(0, n_batches, n_sessions + 1)
            per_session = []
            for i in range(n_sessions):
                lo, hi = int(np.floor(edges[i])), int(np.ceil(edges[i + 1]))
                lo = max(0, lo); hi = min(n_batches, max(lo + 1, hi))
                per_session.append(dshift[lo:hi])
            logger.info('Aggregated %d batches into %d sessions '
                         '(~%.1f batches/session)',
                         n_batches, n_sessions, n_batches / n_sessions)

    fig, ax = plt.subplots(figsize=(12, 5))
    if dshift.ndim == 1 and per_session is not None and mode == 'median':
        # One marker per session at the median of its batches.
        medians = np.array([float(np.median(p)) for p in per_session])
        ax.plot(days, medians, color='#1F3864', lw=1.0, alpha=0.6, zorder=3)
        ax.scatter(days, medians, color='#1F3864', s=34, zorder=4,
                    label=f'session median (of {len(per_session[0])} batches)')
        ax.legend(fontsize=9, loc='upper right')
    elif dshift.ndim == 1 and per_session is not None:
        # Plot every batch as its own point. Spread the n batches of each
        # session across a small day-width so multiple batches at the same
        # session are visible. Connect with a thin line to show drift trace.
        xs = []
        ys = []
        for i, p in enumerate(per_session):
            nb = len(p)
            if nb == 1:
                xs.append(float(days[i]))
                ys.append(float(p[0]))
            else:
                # Spread within ±0.18 days around the session day position
                offsets = np.linspace(-0.18, 0.18, nb)
                for off, val in zip(offsets, p):
                    xs.append(float(days[i]) + float(off))
                    ys.append(float(val))
        xs = np.array(xs)
        ys = np.array(ys)
        ax.plot(xs, ys, color='#1F3864', lw=0.8, alpha=0.55, zorder=2)
        ax.scatter(xs, ys, color='#1F3864', s=22, zorder=3,
                    label=f'per-batch drift ({len(xs)} batches)')
        if any(len(p) > 1 for p in per_session):
            ax.legend(fontsize=9, loc='upper right')
    elif dshift.ndim == 1:
        ax.plot(days[:n_batches], dshift, color='#1F3864', lw=1.6,
                marker='o', markersize=5)
    else:
        for c in range(dshift.shape[1]):
            y = dshift[:, c]
            ax.plot(days[:len(y)], y, lw=1.2, alpha=0.85, marker='o',
                    markersize=4, label=f'block {c}')
        ax.legend(fontsize=8, loc='upper right')

    # Surgery line
    ymax = float(np.max(dshift)) if dshift.size else 1
    ymin = float(np.min(dshift)) if dshift.size else -1
    pad = max(20, (ymax - ymin) * 0.12)
    y_top = ymax + pad
    y_bot = ymin - pad
    ax.axvline(0, color='#C0504D', linestyle='--', lw=1.4, alpha=0.9, zorder=2)
    ax.annotate('Surgery', xy=(0, y_top), xytext=(4, -4),
                 textcoords='offset points', fontsize=10,
                 color='#C0504D', fontweight='bold', va='top', ha='left')

    ax.set_xlim(-3, max(days) + 3)
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel('Days since surgery')
    ax.set_ylabel('Drift (µm)')
    title = drift.get('title') or (
        f'Chronic probe drift — banks 0–{N_BANKS-1} stacked as channels')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(out_png.with_suffix('.svg'), bbox_inches='tight',
                 facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', out_png)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def build_single_bank_session(survey_key, bank_id, segment_s):
    """Build a middle-5s slice of a single bank from one survey. Returns
    a 384-channel recording with that bank's absolute-depth probe set, and
    the actual number of channels (may be < CH_PER_BANK for partial banks).
    """
    import spikeinterface.full as si
    from chronic_pixel_pipeline.preprocessing import preprocess

    survey_dir = DATA_ROOT / SURVEY_DIRS[survey_key]
    if not survey_dir.exists():
        logger.warning('  %s: missing %s', survey_key, survey_dir)
        return None

    meta = parse_meta(find_meta_file(survey_dir))
    fs = float(meta['imSampRate'])
    max_bank = int(meta['imSvyMaxBnk'])
    if bank_id > max_bank:
        logger.info('  %s: bank %d not present (max %d), skipping',
                     survey_key, bank_id, max_bank)
        return None
    bw = _bank_window(meta, bank_id)
    if bw is None:
        return None
    seg_samples = int(round(segment_s * fs))
    win_dur_samples = bw[1] - bw[0]
    if win_dur_samples < seg_samples:
        start, end = bw
    else:
        mid = (bw[0] + bw[1]) // 2
        start = mid - seg_samples // 2
        end = start + seg_samples

    rec_full = si.read_spikeglx(str(survey_dir), stream_id='imec0.ap')
    rec_bank = rec_full.frame_slice(start_frame=start, end_frame=end)
    probe, actual_ch = make_bank_probe(bank_id)
    rec_bank = rec_bank.set_probe(probe, group_mode='by_probe')
    if actual_ch < CH_PER_BANK:
        good = rec_bank.get_channel_ids()[:actual_ch]
        rec_bank = rec_bank.select_channels(channel_ids=good)
    rec_pp = preprocess(rec_bank, freq_min=300, freq_max=6000,
                         destripe=False, cmr=False, phase_shift=True)
    # Trim to exactly seg_samples for clean concatenation
    if rec_pp.get_num_samples() != seg_samples:
        rec_pp = rec_pp.frame_slice(start_frame=0,
                                     end_frame=min(seg_samples,
                                                    rec_pp.get_num_samples()))
    return rec_pp


def main_single_bank(args):
    """Single-bank drift run: concat middle slices of one bank across
    sessions, run KS4 drift correction. Snaps segment_s to an integer
    multiple of batch_size so KS4 batches never straddle session boundaries.
    """
    import spikeinterface.full as si
    from kilosort.parameters import DEFAULT_SETTINGS

    bank_id = args.single_bank
    survey_keys = sorted(SURVEY_DIRS.keys())
    excluded = set(args.exclude or [])
    survey_keys = [k for k in survey_keys if k not in excluded]
    logger.info('Single-bank mode: bank %d, %d sessions (excluded: %s)',
                 bank_id, len(survey_keys), sorted(excluded))

    # Determine the KS4 batch_size we'll use; snap segment_s to a multiple
    # of batch_size_s so every batch lies inside one session.
    fs_first = 30000.0  # placeholder; refined after first read
    batch_size = args.batch_size or DEFAULT_SETTINGS.get('batch_size', 60000)
    batch_size_s = batch_size / fs_first
    n_batches_per_session = max(1, int(round(args.segment_s / batch_size_s)))
    snapped_segment_s = n_batches_per_session * batch_size_s
    if abs(snapped_segment_s - args.segment_s) > 1e-6:
        logger.info('Snapping segment_s %.3fs -> %.3fs to align with '
                     'batch_size %.3fs (%d batches/session)',
                     args.segment_s, snapped_segment_s, batch_size_s,
                     n_batches_per_session)
    seg_s = snapped_segment_s

    session_recs = []
    used_keys = []
    for sk in survey_keys:
        rec = build_single_bank_session(sk, bank_id, seg_s)
        if rec is None:
            continue
        fs_first = rec.sampling_frequency  # confirmed
        session_recs.append(rec)
        used_keys.append(sk)
        logger.info('  %s bank%d: %d ch × %.1fs', sk, bank_id,
                     rec.get_num_channels(),
                     rec.get_num_samples() / rec.sampling_frequency)

    if not session_recs:
        logger.error('No sessions built.')
        return

    rec_concat = si.concatenate_recordings(session_recs)
    total_dur = rec_concat.get_num_samples() / rec_concat.sampling_frequency
    logger.info('Concat: %d ch × %.1fs (%d sessions)',
                 rec_concat.get_num_channels(), total_dur, len(used_keys))

    pd.DataFrame([{'survey': k, 'duration_s': args.segment_s}
                   for k in used_keys]).to_csv(
        OUT_DIR / 'segment_boundaries.csv', index=False)

    # Probe dict for KS4 (single-bank probe, absolute depths)
    probe_si, actual_ch = make_bank_probe(bank_id)
    pos = probe_si.contact_positions[:rec_concat.get_num_channels()]
    n_chan = rec_concat.get_num_channels()
    probe_dict = {
        'chanMap': np.arange(n_chan).astype(np.int32),
        'xc': pos[:, 0].astype(np.float32),
        'yc': pos[:, 1].astype(np.float32),
        'kcoords': np.zeros(n_chan, dtype=np.float32),
        'n_chan': n_chan,
    }

    batch_size = args.batch_size  # may be None → KS4 default (60000)
    suffix_parts = []
    if args.th_universal is not None:
        suffix_parts.append(f'th{args.th_universal:g}')
    if args.binning_depth is not None:
        suffix_parts.append(f'bin{args.binning_depth:g}')
    base_tag = f'bank{bank_id:02d}_only'
    tag = base_tag + (('_' + '_'.join(suffix_parts)) if suffix_parts else '')
    out_dir = OUT_DIR / f'{tag}_ks4'
    run_drift_only(rec_concat, probe_dict, out_dir,
                    batch_size=batch_size, force=args.force,
                    th_universal=args.th_universal,
                    binning_depth=args.binning_depth)

    drift = load_drift(out_dir)
    if drift is None:
        logger.error('No dshift recovered.')
        return
    extra = []
    if args.th_universal is not None:
        extra.append(f'Th={args.th_universal:g}')
    if args.binning_depth is not None:
        extra.append(f'bin={args.binning_depth:g}µm')
    extra_str = (' • ' + ', '.join(extra)) if extra else ''
    drift['title'] = (f'Chronic probe drift — bank {bank_id} only, '
                       f'{len(used_keys)} sessions × {seg_s:g}s '
                       f'(batch {batch_size_s:g}s, {n_batches_per_session}/session)'
                       f'{extra_str}')
    out_png = FIG_DIR / f'{tag}_drift.png'
    plot_drift(drift, used_keys, out_png)

    # Spike raster from KS4-detected spikes (st0)
    st0 = load_spike_array(out_dir)
    if st0 is not None:
        plot_spike_raster(
            st0, fs=drift['fs'], survey_keys=used_keys,
            segment_s=seg_s,
            out_png=FIG_DIR / f'{tag}_raster.png',
            title=f'Spike raster — bank {bank_id} concat across '
                   f'{len(used_keys)} sessions × {seg_s:g}s '
                   f'(KS4 universal-template detection)')


def build_depth_range_session(survey_key, depth_min, depth_max, segment_s,
                                n_distributed=1, batch_size_s=2.0):
    """Per session: for each bank that has channels in [depth_min, depth_max],
    sample `n_distributed` chunks of `batch_size_s` seconds each, evenly
    distributed across the bank's stable window. Select channels in the
    depth band, then aggregate per-bank stacks as channels.

    n_distributed=1 → original behaviour: one contiguous middle-segment_s
    slice. Otherwise: n_distributed * batch_size_s == segment_s, and the
    chunks are concatenated in time so KS4 still sees a `segment_s`-long
    recording per session but each batch comes from a different point in
    the session.
    """
    import spikeinterface.full as si
    from chronic_pixel_pipeline.preprocessing import preprocess

    survey_dir = DATA_ROOT / SURVEY_DIRS[survey_key]
    if not survey_dir.exists():
        return None, None

    meta = parse_meta(find_meta_file(survey_dir))
    fs = float(meta['imSampRate'])
    max_bank = int(meta['imSvyMaxBnk'])
    seg_samples = int(round(segment_s * fs))
    batch_samples = int(round(batch_size_s * fs))

    rec_full = si.read_spikeglx(str(survey_dir), stream_id='imec0.ap')

    bank_recs = []
    depths_all = []
    for b in range(N_BANKS):
        if b > max_bank:
            continue
        probe, actual_ch = make_bank_probe(b)
        bank_yc = probe.contact_positions[:actual_ch, 1]
        bank_xc = probe.contact_positions[:actual_ch, 0]
        sel = np.where((bank_yc >= depth_min) & (bank_yc < depth_max))[0]
        if len(sel) == 0:
            continue
        bw = _bank_window(meta, b)
        if bw is None:
            continue
        win_dur_samples = bw[1] - bw[0]

        # Decide on the per-chunk start/end positions
        if n_distributed <= 1:
            # Original: one contiguous middle slice
            if win_dur_samples < seg_samples:
                slices = [bw]
            else:
                mid = (bw[0] + bw[1]) // 2
                start = mid - seg_samples // 2
                end = start + seg_samples
                slices = [(start, end)]
        else:
            # Distributed: split stable window into n_distributed equal
            # chunks; take batch_samples from the middle of each chunk.
            if win_dur_samples < seg_samples:
                # Window too short — fall back to contiguous
                slices = [bw]
            else:
                chunk_size = win_dur_samples // n_distributed
                slices = []
                for i in range(n_distributed):
                    chunk_start = bw[0] + i * chunk_size
                    chunk_end = chunk_start + chunk_size
                    s = chunk_start + max(0, (chunk_size - batch_samples) // 2)
                    e = s + batch_samples
                    e = min(e, chunk_end)
                    s = max(bw[0], min(s, bw[1] - batch_samples))
                    slices.append((s, s + batch_samples))

        # Build per-slice frame_slice recordings and concatenate them in time
        per_slice = []
        for s, e in slices:
            r = rec_full.frame_slice(start_frame=s, end_frame=e)
            r = r.set_probe(probe, group_mode='by_probe')
            ch_ids = r.get_channel_ids()[:actual_ch]
            keep_ids = [ch_ids[i] for i in sel]
            r = r.select_channels(channel_ids=keep_ids)
            per_slice.append(r)

        if len(per_slice) == 1:
            rec_bank = per_slice[0]
        else:
            # Concatenate the n_distributed chunks in time
            rec_bank = si.concatenate_recordings(per_slice)

        rec_pp = preprocess(rec_bank, freq_min=300, freq_max=6000,
                             destripe=False, cmr=False, phase_shift=True)
        if rec_pp.get_num_samples() != seg_samples:
            rec_pp = rec_pp.frame_slice(start_frame=0,
                                         end_frame=min(seg_samples,
                                                        rec_pp.get_num_samples()))
        bank_recs.append(rec_pp)
        depths_all.append(np.column_stack([bank_xc[sel], bank_yc[sel]]))

    if not bank_recs:
        return None, None

    # Strip probes from children before aggregating
    stripped = [_strip_probe(r) for r in bank_recs]
    rec_session = si.aggregate_channels(stripped)
    coords = np.vstack(depths_all)
    return rec_session, coords


def main_depth_range(args):
    """Drift run restricted to a depth band [depth_min, depth_max] across
    whichever banks contribute channels there. Aggregates banks as channels.
    """
    import spikeinterface.full as si
    from kilosort.parameters import DEFAULT_SETTINGS

    depth_min, depth_max = args.depth_range
    survey_keys = sorted(SURVEY_DIRS.keys())
    excluded = set(args.exclude or [])
    survey_keys = [k for k in survey_keys if k not in excluded]
    logger.info('Depth-range mode: %.0f-%.0f µm, %d sessions (excluded: %s)',
                 depth_min, depth_max, len(survey_keys), sorted(excluded))

    fs_first = 30000.0
    batch_size = args.batch_size or DEFAULT_SETTINGS.get('batch_size', 60000)
    batch_size_s = batch_size / fs_first
    n_batches_per_session = max(1, int(round(args.segment_s / batch_size_s)))
    snapped_segment_s = n_batches_per_session * batch_size_s
    if abs(snapped_segment_s - args.segment_s) > 1e-6:
        logger.info('Snapping segment_s %.3fs -> %.3fs to align with '
                     'batch_size %.3fs (%d batches/session)',
                     args.segment_s, snapped_segment_s, batch_size_s,
                     n_batches_per_session)
    seg_s = snapped_segment_s

    n_dist = n_batches_per_session if args.distribute_batches else 1
    if args.distribute_batches:
        logger.info('Distributing %d × %.2fs batches evenly across each '
                     "session's stable window", n_dist, batch_size_s)
    session_recs = []
    used_keys = []
    coords = None
    for sk in survey_keys:
        rec, c = build_depth_range_session(
            sk, depth_min, depth_max, seg_s,
            n_distributed=n_dist, batch_size_s=batch_size_s)
        if rec is None:
            continue
        fs_first = rec.sampling_frequency
        session_recs.append(rec)
        used_keys.append(sk)
        if coords is None:
            coords = c
        logger.info('  %s: %d ch in [%.0f, %.0f) µm, %.1fs', sk,
                     rec.get_num_channels(), depth_min, depth_max,
                     rec.get_num_samples() / rec.sampling_frequency)

    if not session_recs:
        logger.error('No sessions built.')
        return

    n_chan = session_recs[0].get_num_channels()
    rec_concat = si.concatenate_recordings(session_recs)
    total_dur = rec_concat.get_num_samples() / rec_concat.sampling_frequency
    logger.info('Concat: %d ch × %.1fs (%d sessions)',
                 rec_concat.get_num_channels(), total_dur, len(used_keys))

    pd.DataFrame([{'survey': k, 'duration_s': seg_s,
                    'depth_min': depth_min, 'depth_max': depth_max,
                    'n_chan': n_chan}
                   for k in used_keys]).to_csv(
        OUT_DIR / 'segment_boundaries.csv', index=False)

    probe_dict = {
        'chanMap': np.arange(n_chan).astype(np.int32),
        'xc': coords[:, 0].astype(np.float32),
        'yc': coords[:, 1].astype(np.float32),
        'kcoords': np.zeros(n_chan, dtype=np.float32),
        'n_chan': n_chan,
    }

    tag = f'depth_{int(depth_min)}_{int(depth_max)}'
    suffix_parts = []
    if args.th_universal is not None:
        suffix_parts.append(f'th{args.th_universal:g}')
    if args.binning_depth is not None:
        suffix_parts.append(f'bin{args.binning_depth:g}')
    if args.drift_smoothing is not None:
        suffix_parts.append(
            'sm' + '-'.join(f'{x:g}' for x in args.drift_smoothing))
    if args.sig_interp is not None:
        suffix_parts.append(f'sig{args.sig_interp:g}')
    if args.nblocks is not None and args.nblocks != 1:
        suffix_parts.append(f'nb{args.nblocks:d}')
    if abs(seg_s - 4.0) > 1e-6:
        suffix_parts.append(f'seg{seg_s:g}s')
    if args.distribute_batches:
        suffix_parts.append('dist')
    if suffix_parts:
        tag = tag + '_' + '_'.join(suffix_parts)
    out_dir = OUT_DIR / f'{tag}_ks4'
    run_drift_only(rec_concat, probe_dict, out_dir,
                    batch_size=batch_size, force=args.force,
                    th_universal=args.th_universal,
                    binning_depth=args.binning_depth,
                    drift_smoothing=args.drift_smoothing,
                    sig_interp=args.sig_interp,
                    nblocks=args.nblocks)

    drift = load_drift(out_dir)
    if drift is None:
        return
    extra = []
    if args.th_universal is not None:
        extra.append(f'Th={args.th_universal:g}')
    if args.binning_depth is not None:
        extra.append(f'bin={args.binning_depth:g}µm')
    if args.drift_smoothing is not None:
        extra.append('sm=[' + ','.join(f'{x:g}' for x in args.drift_smoothing) + ']')
    if args.sig_interp is not None:
        extra.append(f'sig={args.sig_interp:g}')
    if args.nblocks is not None and args.nblocks != 1:
        extra.append(f'nblocks={args.nblocks}')
    extra_str = (' • ' + ', '.join(extra)) if extra else ''
    drift['title'] = (f'Chronic probe drift — depth {int(depth_min)}-'
                       f'{int(depth_max)} µm, {len(used_keys)} sessions × '
                       f'{seg_s:g}s '
                       f'(batch {batch_size_s:g}s, {n_batches_per_session}/session)'
                       f'{extra_str}')
    mode = 'median' if args.plot_median else 'per_batch'
    out_drift = FIG_DIR / f'{tag}_drift{"_median" if args.plot_median else ""}.png'
    plot_drift(drift, used_keys, out_drift, mode=mode)

    # tag also used for raster file


    st0 = load_spike_array(out_dir)
    if st0 is not None:
        plot_spike_raster(
            st0, fs=drift['fs'], survey_keys=used_keys,
            segment_s=seg_s,
            out_png=FIG_DIR / f'{tag}_raster.png',
            title=f'Spike raster — depth {int(depth_min)}-{int(depth_max)} µm '
                   f'across {len(used_keys)} sessions × {seg_s:g}s '
                   f'(KS4 universal-template detection)')
        # Drift-corrected raster (raw vs corrected) — uses session medians
        session_med = plot_drift_corrected_raster(
            st0, drift['dshift'].flatten()
            if drift['dshift'].ndim > 1 else drift['dshift'],
            survey_keys=used_keys, segment_s=seg_s,
            batch_size_s=batch_size_s,
            out_png=FIG_DIR / f'{tag}_raster_corrected.png',
            title_suffix=(f'depth {int(depth_min)}-{int(depth_max)} µm  •  '
                          f'{len(used_keys)} sessions × {seg_s:g}s  •  '
                          f'{n_batches_per_session} batches/session'
                          + extra_str))
        if session_med is not None:
            pd.DataFrame({
                'survey': used_keys,
                'session_median_drift_um': session_med,
            }).to_csv(out_dir / 'session_median_drift.csv', index=False)
            logger.info('  Saved session_median_drift.csv')

    # Sync drift_concat figures to Dropbox so the lab-shared tree stays fresh
    try:
        from figure_sync import sync_to_dropbox
        sync_to_dropbox('drift_concat')
    except Exception as e:
        logger.warning('Dropbox sync failed: %s', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment-s', type=float, default=2.0,
                        help='Seconds of each session to use (default 2 for '
                              'all-banks; pass 5 for single-bank)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='KS4 batch_size in samples '
                              '(default: KS4 internal default 60000 = 2s)')
    parser.add_argument('--single-bank', type=int, default=None,
                        help='Run on a single bank only (do not stack banks '
                              'as channels)')
    parser.add_argument('--depth-range', type=float, nargs=2, default=None,
                        metavar=('MIN_UM', 'MAX_UM'),
                        help='Restrict to channels with z in [MIN, MAX) µm '
                              'across bank boundaries')
    parser.add_argument('--exclude', type=str, nargs='*', default=None,
                        help='Survey keys to exclude (e.g. 20260226)')
    parser.add_argument('--th-universal', type=float, default=None,
                        help='KS4 Th_universal (default 9). Higher = only '
                              'large-amplitude spikes used for drift '
                              '(try 12–14 to focus on big units)')
    parser.add_argument('--binning-depth', type=float, default=None,
                        help='KS4 binning_depth in µm (default 5). Coarser '
                              '(10–15) stabilises sparse bands')
    parser.add_argument('--drift-smoothing', type=float, nargs=3,
                        default=None, metavar=('S0', 'S1', 'S2'),
                        help='KS4 drift_smoothing (default 0.5 0.5 0.5). '
                              'Lower → less temporal smoothing across batches')
    parser.add_argument('--sig-interp', type=float, default=None,
                        help='KS4 sig_interp (default 20). Lower → sharper '
                              'drift correlation peaks')
    parser.add_argument('--nblocks', type=int, default=None,
                        help='KS4 nblocks (default 1, rigid). >1 = non-rigid '
                              '(per-depth drift)')
    parser.add_argument('--plot-median', action='store_true',
                        help='Plot one median per session (vs. per-batch '
                              'scatter)')
    parser.add_argument('--distribute-batches', action='store_true',
                        help='Spread each session\'s batches evenly across '
                              'its stable window (instead of one contiguous '
                              'middle slice)')
    parser.add_argument('--preset', choices=['chronic'], default=None,
                        help="Apply the validated 'chronic' parameter set: "
                              "depth-range 12000-16000, segment-s 10, "
                              "th-universal 14, binning-depth 30, "
                              "distribute-batches, plot-median, "
                              "exclude 20260226. Individual --flag overrides "
                              "still take effect.")
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    # Apply preset (fills any unset values; explicit --flag still wins)
    if args.preset == 'chronic':
        p = CHRONIC_PRESET
        if args.depth_range is None:
            args.depth_range = list(p['depth_range'])
        if args.segment_s == 2.0:
            args.segment_s = p['segment_s']
        if args.th_universal is None:
            args.th_universal = p['th_universal']
        if args.binning_depth is None:
            args.binning_depth = p['binning_depth']
        if not args.distribute_batches:
            args.distribute_batches = p['distribute_batches']
        if not args.plot_median:
            args.plot_median = p['plot_median']
        if args.exclude is None:
            args.exclude = sorted(DEFAULT_EXCLUDE)
        logger.info('Preset chronic applied: depth %s, seg %ss, Th=%g, bin=%g, '
                     'distribute=%s, median=%s, exclude=%s',
                     args.depth_range, args.segment_s, args.th_universal,
                     args.binning_depth, args.distribute_batches,
                     args.plot_median, args.exclude)
    # Even without preset, apply DEFAULT_EXCLUDE if user did not specify
    if args.exclude is None:
        args.exclude = sorted(DEFAULT_EXCLUDE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.depth_range is not None:
        return main_depth_range(args)
    if args.single_bank is not None:
        return main_single_bank(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Build the super-probe (SI and KS dict forms)
    super_probe_si = build_super_probe_si()
    xc, yc = build_super_probe_positions()
    super_probe_dict = {
        'chanMap': np.arange(TOTAL_CH).astype(np.int32),
        'xc': xc.astype(np.float32),
        'yc': yc.astype(np.float32),
        'kcoords': np.zeros(TOTAL_CH, dtype=np.float32),
        'n_chan': TOTAL_CH,
    }

    survey_keys = sorted(SURVEY_DIRS.keys())
    logger.info('Building session recordings (%d surveys × banks-as-channels)...',
                 len(survey_keys))

    import spikeinterface.full as si
    session_recs = []
    used_keys = []
    boundaries = []
    fs_first = None
    for sk in survey_keys:
        rec_sess, banks_present = build_session_recording(
            sk, args.segment_s, super_probe_si)
        if rec_sess is None:
            continue
        fs_first = fs_first or rec_sess.sampling_frequency
        n_samp = rec_sess.get_num_samples()
        session_recs.append(rec_sess)
        used_keys.append(sk)
        boundaries.append({'survey': sk, 'banks_present': banks_present,
                            'n_samples': n_samp,
                            'duration_s': n_samp / rec_sess.sampling_frequency})
        logger.info('  %s: %d ch × %.2fs (banks %s)',
                     sk, rec_sess.get_num_channels(),
                     n_samp / rec_sess.sampling_frequency, banks_present)

    if not session_recs:
        logger.error('No sessions built; aborting.')
        return

    pd.DataFrame([{'survey': b['survey'],
                    'banks_present': ','.join(map(str, b['banks_present'])),
                    'n_samples': b['n_samples'],
                    'duration_s': b['duration_s']} for b in boundaries]).to_csv(
        OUT_DIR / 'segment_boundaries.csv', index=False)

    rec_concat = si.concatenate_recordings(session_recs)
    total_dur = rec_concat.get_num_samples() / rec_concat.sampling_frequency
    logger.info('Concat: %d ch × %.1fs (%d sessions)',
                 rec_concat.get_num_channels(), total_dur, len(used_keys))

    batch_size = args.batch_size or int(round(args.segment_s * fs_first))
    logger.info('Batch size: %d samples (%.2fs)', batch_size, batch_size / fs_first)

    out_dir = OUT_DIR / 'all_banks_ks4'
    run_drift_only(rec_concat, super_probe_dict, out_dir,
                    batch_size=batch_size, force=args.force)

    drift = load_drift(out_dir)
    if drift is None:
        logger.error('No dshift recovered.')
        return
    plot_drift(drift, used_keys, FIG_DIR / 'all_banks_drift.png')


if __name__ == '__main__':
    main()
