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
                           max_spikes_per_unit=500,
                           ms_before=1.5, ms_after=2.5,
                           correlogram_window_ms=50.0,
                           correlogram_bin_ms=1.0,
                           metric_names=None):
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
    max_spikes_per_unit : int
        Max random spikes to extract per unit.
    ms_before, ms_after : float
        Waveform window (ms).
    correlogram_window_ms, correlogram_bin_ms : float
        Correlogram parameters.
    metric_names : list[str] or None
        Quality metrics to compute. Uses DEFAULT_METRIC_NAMES if None.

    Returns
    -------
    SortingAnalyzer
        Fully populated analyzer.
    """
    import spikeinterface.full as si

    if metric_names is None:
        metric_names = list(DEFAULT_METRIC_NAMES)

    analyzer = si.create_sorting_analyzer(
        sorting, recording, format="memory", sparse=sparse,
    )

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
