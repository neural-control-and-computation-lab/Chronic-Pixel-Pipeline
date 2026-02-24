"""Signal preprocessing for Neuropixels recordings.

Provides standard preprocessing steps: bad channel removal, bandpass filtering,
and IBL destriping (highpass spatial filter). These functions wrap SpikeInterface
operations into a simple API for the Chronic Pixel Pipeline.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def remove_bad_channels(recording, dead_threshold=10, noisy_factor=10,
                        snippet_duration=2.0):
    """Remove dead and noisy channels based on raw RMS.

    Uses a snippet from the middle of the recording to compute per-channel RMS
    in raw int16 units. Channels below ``dead_threshold`` RMS are considered
    dead (no signal); channels above ``noisy_factor`` × median RMS are
    considered noisy (artifact-dominated).

    Parameters
    ----------
    recording : BaseRecording
        Input recording (should have gains set for µV scaling).
    dead_threshold : float
        RMS below this value (in raw int16 units) marks a channel as dead.
        Default 10.
    noisy_factor : float
        Channels with RMS > this factor × median RMS are marked noisy.
        Default 10.
    snippet_duration : float
        Duration (seconds) of the analysis snippet taken from the recording
        midpoint. Default 2.0.

    Returns
    -------
    recording_clean : BaseRecording
        Recording with bad channels removed.
    good_channel_ids : list
        Channel IDs that were kept.
    info : dict
        Diagnostic information with keys: ``n_dead``, ``n_noisy``,
        ``n_removed``, ``n_kept``, ``median_rms_uv``.
    """
    fs = recording.get_sampling_frequency()
    n_samp = recording.get_num_samples()
    ch_ids = np.array(recording.get_channel_ids())

    # Use a snippet from the middle of the recording
    mid = n_samp // 2
    half_snippet = int(snippet_duration / 2 * fs)
    sf = max(0, mid - half_snippet)
    ef = min(n_samp, mid + half_snippet)

    # Use RAW data for bad channel detection (int16 units)
    snippet = recording.get_traces(
        start_frame=sf, end_frame=ef, return_in_uV=False
    ).astype(np.float64)
    rms_raw = np.sqrt(np.mean(snippet ** 2, axis=0))

    median_rms = np.median(rms_raw)
    bad_low = rms_raw < dead_threshold
    bad_high = rms_raw > median_rms * noisy_factor
    bad_mask = bad_low | bad_high

    good_ids = ch_ids[~bad_mask].tolist()
    n_dead = int(bad_low.sum())
    n_noisy = int(bad_high.sum())
    n_removed = int(bad_mask.sum())

    # Convert median RMS to µV for reporting
    gains = recording.get_channel_gains()
    gain_val = gains[0] if gains is not None else 1.0
    median_rms_uv = float(median_rms * gain_val)

    info = {
        "n_dead": n_dead,
        "n_noisy": n_noisy,
        "n_removed": n_removed,
        "n_kept": len(good_ids),
        "median_rms_uv": median_rms_uv,
    }

    logger.info(
        "Bad channels: %d (dead=%d, noisy=%d), keeping %d/%d, "
        "median RMS=%.2f µV",
        n_removed, n_dead, n_noisy, len(good_ids), len(ch_ids), median_rms_uv,
    )

    return recording.select_channels(channel_ids=good_ids), good_ids, info


def preprocess(recording, freq_min=300, freq_max=6000, destripe=True):
    """Apply standard Neuropixels preprocessing.

    Steps:
      1. Bandpass filter (default 300–6000 Hz for AP band)
      2. IBL destriping via ``highpass_spatial_filter`` (optional)

    Parameters
    ----------
    recording : BaseRecording
        Input recording (should already have bad channels removed).
    freq_min : float
        High-pass cutoff in Hz. Default 300.
    freq_max : float
        Low-pass cutoff in Hz. Default 6000.
    destripe : bool
        If True, apply IBL's highpass spatial filter after bandpass.
        Default True.

    Returns
    -------
    recording_preprocessed : BaseRecording
        Preprocessed recording.
    """
    import spikeinterface.full as si

    rec = si.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)

    if destripe:
        rec = si.highpass_spatial_filter(rec)

    logger.info(
        "Preprocessed: %d ch, %.1fs, bandpass=%d–%d Hz%s",
        rec.get_num_channels(),
        rec.get_total_duration(),
        freq_min, freq_max,
        " + destripe" if destripe else "",
    )

    return rec
