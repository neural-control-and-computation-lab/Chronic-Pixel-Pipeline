"""Post-spike-sorting analysis and visualization.

Provides functions to build a SortingAnalyzer with standard extensions
and generate a set of summary figures from spike sorting results.
"""

import numpy as np


# ── Default quality metric names ─────────────────────────────────────────────
DEFAULT_METRIC_NAMES = [
    "num_spikes", "firing_rate", "presence_ratio", "snr",
    "isi_violation", "rp_violation", "amplitude_cutoff",
    "amplitude_median", "drift", "firing_range",
]


def build_sorting_analyzer(sorting, recording, sparse=True,
                           radius_um=150.0,
                           max_spikes_per_unit=500,
                           ms_before=1.5, ms_after=2.5,
                           correlogram_window_ms=50.0,
                           correlogram_bin_ms=1.0,
                           metric_names=None,
                           cache_dir=None):
    """Create a SortingAnalyzer and compute standard extensions.

    Computes: random_spikes, waveforms, templates, noise_levels,
    spike_amplitudes, correlograms, template_similarity, unit_locations,
    spike_locations, and quality_metrics.

    Parameters
    ----------
    sorting : BaseSorting
        Spike sorting result.
    recording : BaseRecording
        Recording the sorting was run on.
    sparse : bool
        Use sparse waveform extraction (recommended).
    radius_um : float
        Radius in micrometers for sparse channel estimation. Default 150
        ensures both columns of two-column probes (e.g., NP1032 with
        103 um column spacing) are included in each unit's channel set.
    max_spikes_per_unit : int
        Max random spikes to extract per unit.
    ms_before, ms_after : float
        Waveform window (ms).
    correlogram_window_ms, correlogram_bin_ms : float
        Correlogram parameters.
    metric_names : list[str] or None
        Quality metrics to compute. Uses DEFAULT_METRIC_NAMES if None.
    cache_dir : str, Path, or None
        Directory for caching the analyzer to disk. If provided and a
        cached analyzer already exists there, it is loaded directly
        (skipping all computation). If None, the analyzer is created
        in memory with no caching.

    Returns
    -------
    SortingAnalyzer
        Fully populated analyzer.
    """
    import spikeinterface.full as si
    from pathlib import Path

    if metric_names is None:
        metric_names = list(DEFAULT_METRIC_NAMES)

    # Try loading from cache
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if (cache_dir / "spikeinterface_info.json").exists():
            try:
                analyzer = si.load_sorting_analyzer(cache_dir)
                return analyzer
            except Exception:
                pass  # cache invalid, recompute

    fmt = "binary_folder" if cache_dir is not None else "memory"
    kwargs = {"sorting": sorting, "recording": recording,
              "format": fmt, "sparse": sparse,
              "radius_um": radius_um}
    if cache_dir is not None:
        kwargs["folder"] = cache_dir
        kwargs["overwrite"] = True
    analyzer = si.create_sorting_analyzer(**kwargs)

    analyzer.compute("random_spikes", max_spikes_per_unit=max_spikes_per_unit)
    analyzer.compute("waveforms", ms_before=ms_before, ms_after=ms_after)
    analyzer.compute("templates")
    analyzer.compute("noise_levels")
    analyzer.compute("spike_amplitudes")
    analyzer.compute("correlograms",
                     window_ms=correlogram_window_ms,
                     bin_ms=correlogram_bin_ms)
    analyzer.compute("template_similarity")
    analyzer.compute("unit_locations", method="center_of_mass")
    analyzer.compute("spike_locations", method="center_of_mass")
    analyzer.compute("quality_metrics", metric_names=metric_names)
    analyzer.compute("template_metrics", include_multi_channel_metrics=True)

    return analyzer


def classify_units(analyzer, snr_thresh=None, fr_thresh=0.1,
                   isi_thresh=0.5, presence_thresh=0.5):
    """Classify units as 'good' based on quality metrics.

    Parameters
    ----------
    analyzer : SortingAnalyzer
        Must have quality_metrics extension computed.
    snr_thresh : float or None
        SNR threshold. If None, auto-selects: 3.0 for scaled data,
        75th percentile for unscaled data (SNR median < 5).
    fr_thresh : float
        Minimum firing rate (Hz).
    isi_thresh : float
        Maximum ISI violations ratio.
    presence_thresh : float
        Minimum presence ratio.

    Returns
    -------
    good_ids : list
        Unit IDs passing all criteria.
    snr_thresh : float
        The SNR threshold used (useful when auto-selected).
    """
    qm = analyzer.get_extension("quality_metrics").get_data()

    if snr_thresh is None:
        snr_median = qm['snr'].median()
        if snr_median < 5:
            snr_thresh = qm['snr'].quantile(0.75)
        else:
            snr_thresh = 3.0

    mask = (
        (qm['firing_rate'] > fr_thresh) &
        (qm['snr'] > snr_thresh) &
        (qm['isi_violations_ratio'] < isi_thresh)
    )
    if qm['presence_ratio'].notna().any():
        mask = mask & (qm['presence_ratio'].fillna(1.0) > presence_thresh)

    return qm.index[mask].tolist(), snr_thresh


def classify_units_advanced(analyzer,
                            # Noise thresholds (Stage A)
                            max_peaks=2,
                            max_spread_um=800,
                            max_isi_violations=0.5,
                            min_amplitude_uv=25.0,
                            max_n_spatial_channels=20,
                            min_n_spatial_channels=4,
                            max_half_width_ms=0.8,
                            min_amp_spatial_product=80.0,
                            # Axonal thresholds (Stage B)
                            axonal_pt_ratio_thresh=1.0,
                            axonal_max_spread_um=80.0,
                            # Cell type threshold (Stage C)
                            ns_peak_to_valley_s=0.00042,
                            # Artifact voting thresholds (Stage D)
                            artifact_voting=True,
                            artifact_min_violations=2,
                            artifact_max_spatial_decay_um=525.9,
                            artifact_min_isi_cv=0.90,
                            artifact_max_duration_10pct_ms=1.57):
    """Multi-stage unit classification using template and quality metrics.

    Three-stage pipeline inspired by Bombcell (Fabre et al., 2023) and
    CellExplorer (Petersen et al., 2021):
      A) Noise detection — multi-peaked, ISI violations, low amplitude,
         insufficient spatial footprint, excessive spatial spread,
         too-few spatial channels, overly broad waveform
      B) Somatic vs axonal — positive-dominant waveform = axonal
      C) Cell type — narrow-spiking (NS) vs broad-spiking (BS) for somatic

    Note on peak_trough_ratio: SpikeInterface computes peak_val / trough_val
    where trough is the most negative point and peak is the subsequent max.
    For standard neurons this is always negative (e.g., -0.2 means the
    positive recovery is 20% of the negative deflection). More negative
    values (e.g., -3.0) mean the positive peak dominates — typical of
    axonal recordings.

    Parameters
    ----------
    analyzer : SortingAnalyzer
        Must have template_metrics and quality_metrics extensions computed.
    max_peaks : int
        Max positive or negative peaks before flagging as noise.
    max_spread_um : float
        Spatial spread above which a unit is flagged as noise (um).
    max_isi_violations : float
        ISI violations ratio above which a unit is noise.
    min_amplitude_uv : float
        Minimum peak-to-peak template amplitude (uV) on peak channel.
        Units below this are flagged as noise regardless of spatial extent.
    max_n_spatial_channels : int
        Maximum number of channels with PTP > 25% of peak amplitude.
        Real neurons produce a localized dipole that decays rapidly with
        distance; units detectable on too many channels are typically
        electromagnetic interference or motion artifacts. Default 20.
    min_n_spatial_channels : int
        Minimum number of channels with PTP > 25% of peak amplitude.
        Real neurons generate dipole fields detectable on multiple
        nearby channels; units visible on too few channels are likely
        spike-sorter artifacts or localized electrical noise. Default 4.
    max_half_width_ms : float
        Maximum half-width of the template waveform (ms) on peak channel.
        Broad, symmetric waveforms exceeding this are low-frequency
        artifacts, not real action potentials. Real broad-spiking neurons
        have half-widths < 0.45 ms. Default 0.8.
    min_amp_spatial_product : float
        Minimum product of amplitude (uV) x n_spatial_channels.
        Real neurons generate dipole fields that spread to neighboring
        channels via volume conduction; units with low amplitude AND
        narrow spatial footprint are likely spike-sorter artifacts
        (noise fluctuations picked up on a single channel). Default 80
        requires e.g. 80 uV on 1 channel, 40 uV on 2, or 27 uV on 3.
    axonal_pt_ratio_thresh : float
        Absolute peak-trough ratio above which a non-noise unit is axonal
        (positive peak dominates trough).
    axonal_max_spread_um : float
        Max spatial spread for axonal classification (um).
    ns_peak_to_valley_s : float
        Peak-to-valley duration threshold: <= this = narrow-spiking (s).
    artifact_voting : bool
        Enable Stage D artifact voting. When True, non-noise units that
        violate >= ``artifact_min_violations`` of 3 novel metric criteria
        are reclassified as noise.
    artifact_min_violations : int
        Minimum number of criteria violations to flag as artifact (default 2).
    artifact_max_spatial_decay_um : float
        Spatial decay constant above which a unit is flagged (diffuse signal).
    artifact_min_isi_cv : float
        ISI coefficient of variation below which a unit is flagged
        (too-regular firing, not Poisson-like).
    artifact_max_duration_10pct_ms : float
        Waveform duration at 10% amplitude above which a unit is flagged
        (overly wide waveform).

    Returns
    -------
    pandas.DataFrame
        Indexed by unit_id with columns: unit_class, recording_type,
        cell_type, amplitude_uv, n_spatial_channels, template_rms,
        spatial_decay_um, isi_cv, duration_10pct_ms,
        plus all template metric columns.
    """
    import pandas as pd

    tm = analyzer.get_extension("template_metrics").get_data()
    qm = analyzer.get_extension("quality_metrics").get_data()
    templates = analyzer.get_extension("templates").get_data()

    # Start with template metrics as the base DataFrame
    df = tm.copy()

    # Compute per-unit template amplitude, spatial extent, and RMS
    unit_ids = list(df.index)
    all_unit_ids = list(analyzer.sorting.unit_ids)
    ptp_amps = np.full(len(df), np.nan)
    n_spatial_ch = np.full(len(df), 0, dtype=int)
    t_rms = np.full(len(df), np.nan)
    for i, uid in enumerate(unit_ids):
        uid_idx = all_unit_ids.index(uid)
        template = templates[uid_idx]          # (n_samples, n_channels)
        ptp_per_ch = np.ptp(template, axis=0)
        peak_ch = np.argmax(ptp_per_ch)
        peak_ptp = ptp_per_ch[peak_ch]
        ptp_amps[i] = peak_ptp
        n_spatial_ch[i] = int(np.sum(ptp_per_ch > 0.25 * peak_ptp))
        t_rms[i] = np.sqrt(np.mean(template[:, peak_ch] ** 2))
    df['amplitude_uv'] = ptp_amps
    df['n_spatial_channels'] = n_spatial_ch
    df['template_rms'] = t_rms

    # Initialize classification columns
    df['unit_class'] = 'good'
    df['recording_type'] = pd.NA
    df['cell_type'] = pd.NA

    # ── Stage A: Noise detection ─────────────────────────────────────────
    noise_mask = np.zeros(len(df), dtype=bool)

    # Multi-peaked waveforms (>2 peaks = likely artifact or multi-unit)
    if 'num_negative_peaks' in df.columns:
        noise_mask |= (df['num_negative_peaks'] > max_peaks)
    if 'num_positive_peaks' in df.columns:
        noise_mask |= (df['num_positive_peaks'] > max_peaks)

    # Spatially too wide (volume-conducted noise)
    if 'spread' in df.columns:
        noise_mask |= (df['spread'].notna() & (df['spread'] > max_spread_um))

    # No refractory period
    if 'isi_violations_ratio' in qm.columns:
        isi_vals = qm['isi_violations_ratio'].reindex(df.index)
        noise_mask |= (isi_vals > max_isi_violations)

    # Too-low amplitude — below this threshold, the signal is not
    # distinguishable from noise/crosstalk regardless of other metrics
    noise_mask |= (df['amplitude_uv'] < min_amplitude_uv)

    # Insufficient amplitude-spatial product — real neurons produce dipole
    # fields that must be detectable on multiple nearby channels.  Units
    # with both low amplitude and narrow spatial footprint are likely
    # spike-sorter artifacts (noise fluctuations on a single channel).
    amp_spatial = df['amplitude_uv'] * df['n_spatial_channels']
    noise_mask |= (amp_spatial < min_amp_spatial_product)

    # Too many channels with significant signal — real neurons produce a
    # localized dipole field that decays with distance.  Units detectable
    # at >25% of peak amplitude on too many channels are electromagnetic
    # interference, motion artifacts, or volume-conducted noise.
    noise_mask |= (df['n_spatial_channels'] > max_n_spatial_channels)

    # Too few channels with significant signal — real neurons generate
    # dipole fields detectable on multiple nearby channels via volume
    # conduction.  Units visible on fewer than min_n_spatial_channels
    # are likely spike-sorter artifacts or localized electrical noise.
    noise_mask |= (df['n_spatial_channels'] < min_n_spatial_channels)

    # Overly broad waveform — real action potentials have half-widths
    # well below 0.8 ms (even broad-spiking neurons peak around 0.45 ms).
    # Wider waveforms are low-frequency artifacts or motion noise.
    if 'half_width' in df.columns:
        hw_ms = df['half_width'] * 1000  # convert s → ms
        noise_mask |= (hw_ms.notna() & (hw_ms > max_half_width_ms))

    df.loc[noise_mask, 'unit_class'] = 'noise'

    # ── Stage B: Somatic vs axonal (non-noise only) ──────────────────────
    # Axonal recordings have positive-dominant waveforms: the peak (positive)
    # is larger in magnitude than the trough (negative). In SI's convention,
    # peak_trough_ratio = peak/trough, so |ratio| > 1 means peak dominates.
    good_mask = df['unit_class'] == 'good'

    axonal_mask = good_mask.copy()
    if 'peak_trough_ratio' in df.columns:
        # |ratio| > threshold means the positive peak dominates the trough
        axonal_mask &= (df['peak_trough_ratio'].abs() > axonal_pt_ratio_thresh)
    else:
        axonal_mask[:] = False
    # Narrow spatial footprint required for axonal
    if 'spread' in df.columns:
        has_spread = df['spread'].notna()
        axonal_mask &= (~has_spread | (df['spread'] < axonal_max_spread_um))

    df.loc[good_mask & ~axonal_mask, 'recording_type'] = 'somatic'
    df.loc[axonal_mask, 'recording_type'] = 'axonal'

    # ── Stage C: Cell type for somatic units ──────────────────────────────
    somatic_mask = df['recording_type'] == 'somatic'

    if 'peak_to_valley' in df.columns:
        ns_mask = somatic_mask & (df['peak_to_valley'] <= ns_peak_to_valley_s)
        bs_mask = somatic_mask & (df['peak_to_valley'] > ns_peak_to_valley_s)
        df.loc[ns_mask, 'cell_type'] = 'NS'
        df.loc[bs_mask, 'cell_type'] = 'BS'

    # ── Stage D: Artifact voting (2-of-3 novel metric rule) ───────────
    if artifact_voting:
        fs = analyzer.recording.get_sampling_frequency()
        positions = analyzer.get_channel_locations()

        # Compute novel metrics for all units
        spatial_decay_vals = np.full(len(df), np.nan)
        isi_cv_vals = np.full(len(df), np.nan)
        dur_10pct_vals = np.full(len(df), np.nan)

        for i, uid in enumerate(unit_ids):
            uid_idx = all_unit_ids.index(uid)
            template = templates[uid_idx]
            ptp_per_ch = np.ptp(template, axis=0)
            peak_ch = np.argmax(ptp_per_ch)
            peak_ptp = ptp_per_ch[peak_ch]

            # Spatial decay rate
            if peak_ptp > 1e-12:
                peak_pos = positions[peak_ch]
                distances = np.sqrt(np.sum(
                    (positions - peak_pos) ** 2, axis=1))
                valid = ((ptp_per_ch > 0.01 * peak_ptp) &
                         (distances > 0))
                if np.sum(valid) >= 3:
                    d = distances[valid]
                    log_a = np.log(np.clip(
                        ptp_per_ch[valid] / peak_ptp, 1e-10, None))
                    try:
                        coeffs = np.polyfit(d, log_a, 1)
                        if coeffs[0] < 0:
                            spatial_decay_vals[i] = -1.0 / coeffs[0]
                    except Exception:
                        pass

            # ISI coefficient of variation
            spike_train = analyzer.sorting.get_unit_spike_train(uid)
            if len(spike_train) >= 10:
                isi = np.diff(spike_train) / fs
                mean_isi = np.mean(isi)
                if mean_isi > 1e-12:
                    isi_cv_vals[i] = np.std(isi) / mean_isi

            # Duration at 10% amplitude
            wf = template[:, peak_ch]
            baseline = np.mean(wf[:max(3, int(0.1 * len(wf)))])
            ptp_wf = np.ptp(wf)
            if ptp_wf > 1e-12:
                thresh_10 = 0.10 * ptp_wf
                above = np.where(
                    np.abs(wf - baseline) > thresh_10)[0]
                if len(above) >= 2:
                    dur_10pct_vals[i] = (
                        (above[-1] - above[0]) / fs * 1000)

        df['spatial_decay_um'] = spatial_decay_vals
        df['isi_cv'] = isi_cv_vals
        df['duration_10pct_ms'] = dur_10pct_vals

        # Apply voting: flag units that violate >= N of 3 criteria
        still_good = df['unit_class'] == 'good'
        v1 = (df['spatial_decay_um'] > artifact_max_spatial_decay_um) | (
            df['spatial_decay_um'].isna())
        v2 = (df['isi_cv'] < artifact_min_isi_cv) | df['isi_cv'].isna()
        v3 = (df['duration_10pct_ms'] > artifact_max_duration_10pct_ms) | (
            df['duration_10pct_ms'].isna())

        n_violations = v1.astype(int) + v2.astype(int) + v3.astype(int)
        artifact_mask = still_good & (n_violations >= artifact_min_violations)

        df.loc[artifact_mask, 'unit_class'] = 'noise'
        df.loc[artifact_mask, 'recording_type'] = pd.NA
        df.loc[artifact_mask, 'cell_type'] = pd.NA

    return df


def plot_summary(analyzer, sorting, recording, output_dir,
                 good_ids=None, snr_thresh=3.0, dpi=150):
    """Generate a standard set of summary figures.

    Produces 10 figure types (plus per-unit summaries) and saves them
    as PNG files to *output_dir*.

    Parameters
    ----------
    analyzer : SortingAnalyzer
        Must have all standard extensions computed.
    sorting : BaseSorting
        Spike sorting result.
    recording : BaseRecording
        Recording the sorting was run on.
    output_dir : str or Path
        Directory for saving figures (created if needed).
    good_ids : list or None
        Unit IDs classified as 'good'. If None, all units are colored the same.
    snr_thresh : float
        SNR threshold (for display on plots).
    dpi : int
        Figure resolution.

    Returns
    -------
    list[Path]
        Paths to all saved figures.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import spikeinterface.full as si
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    qm = analyzer.get_extension("quality_metrics").get_data()
    locs = analyzer.get_extension("unit_locations").get_data()
    unit_ids = sorting.unit_ids
    if good_ids is None:
        good_ids = []
    good_set = set(good_ids)

    def _save(fig, name):
        p = output_dir / name
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved.append(p)

    colors = ['tab:blue' if uid in good_set else 'tab:gray'
              for uid in qm.index]

    # ── 1. Unit locations overview ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    si.plot_unit_locations(analyzer, ax=axes[0], with_channel_ids=False)
    axes[0].set_title("Unit locations")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")

    depths = locs[:, 1]
    axes[1].scatter(qm['firing_rate'].values, depths, c=colors, s=15,
                    alpha=0.7, edgecolors='none')
    axes[1].set_xlabel("Firing rate (Hz)")
    axes[1].set_ylabel("Depth (um)")
    axes[1].set_title("Depth vs Firing Rate")
    axes[1].set_xscale('log')

    axes[2].scatter(qm['snr'].values, depths, c=colors, s=15,
                    alpha=0.7, edgecolors='none')
    axes[2].set_xlabel("SNR")
    axes[2].set_ylabel("Depth (um)")
    axes[2].set_title("Depth vs SNR")
    axes[2].axvline(snr_thresh, color='red', ls='--', alpha=0.5,
                    label=f'SNR={snr_thresh:.2f}')
    axes[2].legend()
    fig.suptitle(f"Sorting Overview — {len(unit_ids)} units "
                 f"({len(good_ids)} good)", fontsize=14)
    fig.tight_layout()
    _save(fig, "01_unit_locations.png")

    # ── 2. Quality metric histograms ─────────────────────────────────────
    metrics_to_plot = ['firing_rate', 'snr', 'isi_violations_ratio',
                       'presence_ratio', 'amplitude_cutoff',
                       'amplitude_median']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    thresholds = {
        'firing_rate': 0.1, 'snr': snr_thresh,
        'isi_violations_ratio': 0.5, 'presence_ratio': 0.5,
    }
    for ax, metric in zip(axes.flat, metrics_to_plot):
        vals = qm[metric].dropna().values
        ax.hist(vals, bins=40, color='steelblue', edgecolor='white',
                alpha=0.8)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        if metric == 'firing_rate':
            ax.set_xscale('log')
        if metric in thresholds:
            ax.axvline(thresholds[metric], color='red', ls='--', alpha=0.7,
                       label=f'threshold={thresholds[metric]}')
            ax.legend(fontsize=8)
    fig.suptitle("Quality Metrics Distributions", fontsize=14)
    fig.tight_layout()
    _save(fig, "02_quality_metrics.png")

    # ── 3. Quality metric scatter matrix ─────────────────────────────────
    scatter_metrics = ['firing_rate', 'snr', 'isi_violations_ratio',
                       'presence_ratio']
    n = len(scatter_metrics)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    for i, mi in enumerate(scatter_metrics):
        for j, mj in enumerate(scatter_metrics):
            ax = axes[i, j]
            if i == j:
                ax.hist(qm[mi].dropna().values, bins=30,
                        color='steelblue', edgecolor='white', alpha=0.8)
            else:
                xi, yi = qm[mj].values, qm[mi].values
                mask = np.isfinite(xi) & np.isfinite(yi)
                c = ['tab:blue' if uid in good_set else 'tab:gray'
                     for uid, m in zip(qm.index, mask) if m]
                ax.scatter(xi[mask], yi[mask], c=c, s=8, alpha=0.5,
                           edgecolors='none')
            if i == n - 1:
                ax.set_xlabel(mj, fontsize=8)
            if j == 0:
                ax.set_ylabel(mi, fontsize=8)
    fig.suptitle("Quality Metrics Scatter Matrix (blue=good, gray=other)",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, "03_qm_scatter.png")

    # ── 4. Template similarity ───────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    si.plot_template_similarity(analyzer, ax=ax)
    ax.set_title("Template Similarity Matrix")
    fig.tight_layout()
    _save(fig, "04_template_similarity.png")

    # ── 5. Autocorrelograms (top 20 by SNR) ──────────────────────────────
    top_snr = qm.nlargest(min(20, len(qm)), 'snr').index.tolist()
    fig = plt.figure(figsize=(18, 12))
    si.plot_autocorrelograms(analyzer, unit_ids=top_snr, figure=fig)
    fig.suptitle("Autocorrelograms (top 20 units by SNR)", fontsize=14,
                 y=1.01)
    fig.tight_layout()
    _save(fig, "05_autocorrelograms.png")

    # ── 6. Waveforms (top 12 by SNR) ─────────────────────────────────────
    top12 = qm.nlargest(min(12, len(qm)), 'snr').index.tolist()
    fig = plt.figure(figsize=(18, 14))
    si.plot_unit_waveforms(analyzer, unit_ids=top12, figure=fig)
    fig.suptitle("Unit Waveforms (top 12 by SNR)", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "06_waveforms.png")

    # ── 7. Spike amplitudes over time ────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    si.plot_amplitudes(analyzer, figure=fig)
    fig.suptitle("Spike Amplitudes Over Time", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "07_amplitudes.png")

    # ── 8. Depth raster ──────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fs = recording.get_sampling_frequency()
    for i, uid in enumerate(unit_ids):
        st = sorting.get_unit_spike_train(uid) / fs
        ax.plot(st, np.full_like(st, locs[i, 1], dtype=float), '|',
                markersize=1, alpha=0.3, color='k')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (um)")
    ax.set_title("Spike Raster by Depth")
    fig.tight_layout()
    _save(fig, "08_depth_raster.png")

    # ── 9. Individual unit summaries (top 6 good by SNR) ─────────────────
    if good_ids:
        good_sorted = qm.loc[good_ids].nlargest(min(6, len(good_ids)),
                                                 'snr')
        for uid in good_sorted.index:
            try:
                fig = plt.figure(figsize=(14, 8))
                si.plot_unit_summary(analyzer, unit_id=uid, figure=fig)
                fig.suptitle(
                    f"Unit {uid} — SNR={qm.loc[uid,'snr']:.1f}, "
                    f"FR={qm.loc[uid,'firing_rate']:.1f} Hz, "
                    f"ISI viol={qm.loc[uid,'isi_violations_ratio']:.3f}",
                    fontsize=12, y=1.01)
                fig.tight_layout()
                _save(fig, f"09_unit_summary_{uid}.png")
            except Exception:
                plt.close('all')

    # ── 10. Firing rate heatmap ──────────────────────────────────────────
    duration = recording.get_total_duration()
    bin_s = max(1.0, duration / 100)  # at least 1s bins
    n_bins = int(np.ceil(duration / bin_s))
    fr_matrix = np.zeros((len(unit_ids), n_bins))
    for i, uid in enumerate(unit_ids):
        st = sorting.get_unit_spike_train(uid) / fs
        hist, _ = np.histogram(st, bins=n_bins, range=(0, duration))
        fr_matrix[i] = hist / bin_s
    depth_order = np.argsort(locs[:, 1])
    fr_sorted = fr_matrix[depth_order]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    im = ax.imshow(fr_sorted, aspect='auto', origin='lower',
                   extent=[0, duration, 0, len(unit_ids)],
                   vmin=0, vmax=np.percentile(fr_sorted, 95),
                   cmap='inferno')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit (sorted by depth)")
    ax.set_title(f"Firing Rate Heatmap ({bin_s:.0f}s bins, sorted by depth)")
    plt.colorbar(im, ax=ax, label="Firing rate (Hz)")
    fig.tight_layout()
    _save(fig, "10_firing_rate_heatmap.png")

    return saved
