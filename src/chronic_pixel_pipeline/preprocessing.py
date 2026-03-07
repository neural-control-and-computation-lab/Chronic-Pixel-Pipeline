"""Signal preprocessing for Neuropixels recordings.

Provides standard preprocessing steps: bad channel removal, phase-shift
correction (for ADC multiplexing), bandpass filtering, notch filtering
(for narrowband artifact removal), and spatial referencing (IBL destriping
or common median reference). These functions wrap SpikeInterface operations
into a simple API for the Chronic Pixel Pipeline.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def remove_bad_channels(recording, dead_threshold=10, noisy_factor=10,
                        snippet_duration=2.0):
    """Remove dead and noisy channels based on raw RMS.

    Uses a snippet from the middle of the recording to compute per-channel RMS
    in raw int16 units. Channels below ``dead_threshold`` RMS are considered
    dead (no signal); channels above ``noisy_factor`` x median RMS are
    considered noisy (artifact-dominated).

    Parameters
    ----------
    recording : BaseRecording
        Input recording (should have gains set for uV scaling).
    dead_threshold : float
        RMS below this value (in raw int16 units) marks a channel as dead.
        Default 10.
    noisy_factor : float
        Channels with RMS > this factor x median RMS are marked noisy.
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

    # Convert median RMS to uV for reporting
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
        "median RMS=%.2f uV",
        n_removed, n_dead, n_noisy, len(good_ids), len(ch_ids), median_rms_uv,
    )

    return recording.select_channels(channel_ids=good_ids), good_ids, info


def preprocess(recording, freq_min=300, freq_max=6000, destripe=True,
               notch_freqs=None, notch_q=30, cmr=False, phase_shift=True):
    """Apply standard Neuropixels preprocessing.

    Steps:
      1. Phase-shift correction for ADC multiplexing (if ``inter_sample_shift``
         property is present on the recording)
      2. Notch filter at specified frequencies (optional, for artifact removal)
      3. Bandpass filter (default 300-6000 Hz for AP band)
      4. IBL destriping via ``highpass_spatial_filter`` (optional)
         OR common median reference (if ``cmr=True``)

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
        Default True. Ignored if ``cmr=True``.
    notch_freqs : list[float] or None
        Frequencies (Hz) at which to apply notch filters before bandpass.
        Useful for removing narrowband artifacts (e.g., 1602 Hz wireless
        artifact + harmonics). Default None (no notch filtering).
    notch_q : float
        Quality factor for notch filters. Higher = narrower notch.
        Default 30.
    cmr : bool
        If True, apply global common median reference instead of destriping.
        Default False.
    phase_shift : bool
        If True and the recording has an ``inter_sample_shift`` property,
        apply phase-shift correction to compensate for the Neuropixels
        multiplexed ADC sampling offsets. Default True.

    Returns
    -------
    recording_preprocessed : BaseRecording
        Preprocessed recording.
    """
    import spikeinterface.full as si

    rec = recording

    # Step 0: Phase-shift correction for ADC multiplexing
    phase_label = ""
    if phase_shift and 'inter_sample_shift' in rec.get_property_keys():
        rec = si.phase_shift(rec)
        phase_label = "phase_shift + "
        logger.info("  Phase-shift correction applied (ADC demux)")
    elif phase_shift:
        logger.info("  Phase-shift skipped (no inter_sample_shift property)")

    # Step 1: Notch filtering (before bandpass to remove narrowband artifacts)
    if notch_freqs is not None:
        for freq in notch_freqs:
            if freq_min < freq < freq_max:
                rec = si.notch_filter(rec, freq=freq, q=notch_q)
                logger.info("  Notch filter: %.0f Hz (Q=%d)", freq, notch_q)
            else:
                logger.info("  Skipping notch at %.0f Hz (outside bandpass)", freq)

    # Step 2: Bandpass filter
    rec = si.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max)

    # Step 3: Spatial filtering
    if cmr:
        rec = si.common_reference(rec, reference="global", operator="median")
        spatial_label = " + CMR"
    elif destripe:
        rec = si.highpass_spatial_filter(rec)
        spatial_label = " + destripe"
    else:
        spatial_label = ""

    notch_label = f" + notch({len(notch_freqs)})" if notch_freqs else ""
    logger.info(
        "Preprocessed: %d ch, %.1fs, %sbandpass=%d-%d Hz%s%s",
        rec.get_num_channels(),
        rec.get_total_duration(),
        phase_label,
        freq_min, freq_max,
        notch_label, spatial_label,
    )

    return rec
