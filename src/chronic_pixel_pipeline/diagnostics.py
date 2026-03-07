"""Diagnostic visualization for neural recordings and spike sorting results.

Provides two main capabilities:

1. **Sorting diagnostics** — Generate 12 diagnostic figure types from spike
   sorting results (``plot_sorting_diagnostics``).

2. **Timecourse bio-signal scan** — Slide a window across a recording and
   compute per-window biological signal quality metrics without spike sorting
   (``compute_window_metrics``, ``run_timescan``, ``plot_timecourse_dashboard``,
   ``plot_best_windows``).
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Timescan constants ────────────────────────────────────────────────────────
_N_BEFORE = 45   # 1.5 ms at 30 kHz
_N_AFTER = 75    # 2.5 ms at 30 kHz
_THRESH_SIGMA = -4.0  # threshold in MAD units


# ══════════════════════════════════════════════════════════════════════════════
# SORTING DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def plot_sorting_diagnostics(analyzer, sorting, recording, output_dir,
                             recording_name="", dpi=200):
    """Generate 12 diagnostic figures from spike sorting results.

    All SortingAnalyzer data (templates, waveforms, noise_levels,
    spike_amplitudes) are already in µV because the analyzer uses
    ``return_in_uV=True`` by default. No manual gain multiplication needed.

    Figures generated:
      01. Unit quality summary table
      02. Mean waveforms (top 20 by SNR)
      03. Individual waveform overlays (top 6)
      04. Autocorrelograms (top 12)
      05. Spike amplitudes over time (top 8)
      06. Raw traces with spike markers (top 6)
      07. Spatial footprint / multi-channel templates (top 6)
      08. ISI histograms (top 12)
      09. Quality metrics scatter plots
      10. Spike raster by depth
      11. Noise floor analysis
      12. PTP amplitude histogram

    Parameters
    ----------
    analyzer : SortingAnalyzer
        Must have standard extensions computed (see
        :func:`~chronic_pixel_pipeline.analysis.build_sorting_analyzer`).
    sorting : BaseSorting
        Spike sorting result.
    recording : BaseRecording
        Recording the sorting was run on.
    output_dir : str or Path
        Directory for saving figures (created if needed).
    recording_name : str
        Label for figure titles.
    dpi : int
        Figure resolution. Default 200.

    Returns
    -------
    list[Path]
        Paths to all saved figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    qm = analyzer.get_extension("quality_metrics").get_data()
    templates = analyzer.get_extension("templates").get_data()  # already µV
    locs = analyzer.get_extension("unit_locations").get_data()
    unit_ids = sorting.unit_ids
    fs = recording.get_sampling_frequency()
    n_units = len(unit_ids)

    def _save(fig, name):
        p = output_dir / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        logger.info("  saved: %s", name)

    # ── Template time axis ────────────────────────────────────────────────
    n_template_samples = templates.shape[1]
    ms_before = 1.5
    t_ms = np.arange(n_template_samples) / fs * 1000 - ms_before

    # ── SNR-ordered unit list ─────────────────────────────────────────────
    snr_order = qm.nlargest(min(20, n_units), "snr").index.tolist()
    n_show = len(snr_order)

    # ==================================================================
    # 1. OVERVIEW: Unit summary table
    # ==================================================================
    fig, ax = plt.subplots(figsize=(14, max(4, 0.3 * n_units + 2)))
    ax.axis("off")
    cols = ["unit_id", "firing_rate", "snr", "isi_violations_ratio",
            "presence_ratio", "amplitude_median", "num_spikes"]
    table_data = []
    for uid in unit_ids:
        row = [str(uid)]
        for col in cols[1:]:
            val = qm.loc[uid, col] if col in qm.columns else "N/A"
            row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        table_data.append(row)
    if table_data:
        tbl = ax.table(cellText=table_data, colLabels=cols,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.auto_set_column_width(list(range(len(cols))))
    ax.set_title(f"Unit Quality Summary — {recording_name} ({n_units} units)",
                 fontsize=13, fontweight="bold")
    _save(fig, "01_unit_summary_table.png")

    # ==================================================================
    # 2. MEAN WAVEFORMS (µV)
    # ==================================================================
    ncols = min(5, n_show)
    nrows = max(1, int(np.ceil(n_show / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and nrows > 1:
        axes = axes.T

    for idx, uid in enumerate(snr_order):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        uid_idx = list(unit_ids).index(uid)
        template = templates[uid_idx]
        peak_chan = np.argmin(np.min(template, axis=0))
        waveform = template[:, peak_chan]
        ax.plot(t_ms, waveform, "k-", lw=1.5)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axvline(0, color="red", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("µV", fontsize=8)
        peak_amp = waveform.min() if abs(waveform.min()) > abs(waveform.max()) else waveform.max()
        ax.set_title(f"Unit {uid}\nSNR={qm.loc[uid, 'snr']:.1f}, "
                     f"FR={qm.loc[uid, 'firing_rate']:.1f}Hz\n"
                     f"peak={peak_amp:.1f}µV", fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(n_show, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    fig.suptitle(f"Mean Waveforms (top {n_show} by SNR) — {recording_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_mean_waveforms_uV.png")

    # ==================================================================
    # 3. INDIVIDUAL WAVEFORM OVERLAYS
    # ==================================================================
    wf_ext = analyzer.get_extension("waveforms")
    sparsity = analyzer.sparsity
    top6 = snr_order[:min(6, n_show)]
    fig, axes_arr = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes_arr.flatten()
    for idx, uid in enumerate(top6):
        ax = axes_flat[idx]
        try:
            wfs = wf_ext.get_waveforms_one_unit(uid)
            uid_idx = list(unit_ids).index(uid)
            template = templates[uid_idx]
            peak_chan = np.argmin(np.min(template, axis=0))
            if sparsity is not None:
                sparse_mask = sparsity.mask[uid_idx]
                sparse_indices = np.flatnonzero(sparse_mask)
                if peak_chan in sparse_indices:
                    local_peak = np.searchsorted(sparse_indices, peak_chan)
                else:
                    local_peak = np.argmax(np.ptp(wfs.mean(axis=0), axis=0))
                    peak_chan = sparse_indices[local_peak] if local_peak < len(sparse_indices) else 0
            else:
                local_peak = peak_chan
            wfs_uv = wfs[:, :, local_peak]
            mean_uv = template[:, peak_chan]
            n_plot = min(100, wfs_uv.shape[0])
            for i in range(n_plot):
                ax.plot(t_ms, wfs_uv[i], color="steelblue", alpha=0.1, lw=0.5)
            ax.plot(t_ms, mean_uv, "k-", lw=2, label="Mean")
            ax.axhline(0, color="gray", ls=":", lw=0.5)
            ax.axvline(0, color="red", ls=":", lw=0.5, alpha=0.5)
            ax.set_xlabel("Time (ms)", fontsize=9)
            ax.set_ylabel("µV", fontsize=9)
            ax.set_title(f"Unit {uid} — {wfs_uv.shape[0]} spikes on ch {peak_chan}\n"
                         f"SNR={qm.loc[uid, 'snr']:.1f}, "
                         f"FR={qm.loc[uid, 'firing_rate']:.1f}Hz", fontsize=9)
            ax.legend(fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
    for idx in range(len(top6), 6):
        axes_flat[idx].set_visible(False)
    fig.suptitle(f"Waveform Overlays — {recording_name}\n"
                 f"Blue: individual spikes, Black: mean (µV)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_waveform_overlays_uV.png")

    # ==================================================================
    # 4. AUTOCORRELOGRAMS
    # ==================================================================
    ccg_ext = analyzer.get_extension("correlograms")
    ccgs, bins = ccg_ext.get_data()
    n_acg = min(12, n_units)
    acg_ids = snr_order[:n_acg]
    ncols_acg = min(4, n_acg)
    nrows_acg = max(1, int(np.ceil(n_acg / ncols_acg)))
    fig, axes_arr = plt.subplots(nrows_acg, ncols_acg,
                                 figsize=(4 * ncols_acg, 3 * nrows_acg))
    axes_arr = np.atleast_2d(axes_arr)
    if axes_arr.shape[0] == 1 and nrows_acg > 1:
        axes_arr = axes_arr.T
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for idx, uid in enumerate(acg_ids):
        r, c = divmod(idx, ncols_acg)
        ax = axes_arr[r, c]
        uid_idx = list(unit_ids).index(uid)
        acg = ccgs[uid_idx, uid_idx, :]
        ax.bar(bin_centers, acg, width=bins[1] - bins[0], color="steelblue",
               edgecolor="white", linewidth=0.3)
        ax.axvspan(-2, 2, alpha=0.15, color="red", label="Refractory (±2ms)")
        ax.set_xlabel("Lag (ms)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(f"Unit {uid} — ISI viol={qm.loc[uid, 'isi_violations_ratio']:.3f}\n"
                     f"SNR={qm.loc[uid, 'snr']:.1f}, "
                     f"{int(qm.loc[uid, 'num_spikes'])} spikes", fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(n_acg, nrows_acg * ncols_acg):
        axes_arr[idx // ncols_acg, idx % ncols_acg].set_visible(False)
    fig.suptitle(f"Autocorrelograms — {recording_name}\n"
                 f"Refractory period dip near 0 ms indicates real neurons",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_autocorrelograms.png")

    # ==================================================================
    # 5. SPIKE AMPLITUDES OVER TIME
    # ==================================================================
    amp_ext = analyzer.get_extension("spike_amplitudes")
    amps = amp_ext.get_data()
    n_amp = min(8, n_units)
    amp_ids = snr_order[:n_amp]
    ncols_amp = min(4, n_amp)
    nrows_amp = max(1, int(np.ceil(n_amp / ncols_amp)))
    fig, axes_arr = plt.subplots(nrows_amp, ncols_amp,
                                 figsize=(4 * ncols_amp, 3 * nrows_amp))
    axes_arr = np.atleast_2d(axes_arr)
    if axes_arr.shape[0] == 1 and nrows_amp > 1:
        axes_arr = axes_arr.T
    # Use the analyzer's spike vector (which matches amps after trimming)
    spikes = analyzer.sorting.to_spike_vector()
    # If analyzer trimmed spikes, amps length may differ; use min length
    n_spikes_amps = len(amps)
    spikes_for_amps = spikes[:n_spikes_amps]
    for idx, uid in enumerate(amp_ids):
        r, c = divmod(idx, ncols_amp)
        ax = axes_arr[r, c]
        uid_idx = list(unit_ids).index(uid)
        spike_mask = spikes_for_amps["unit_index"] == uid_idx
        spike_times = spikes_for_amps["sample_index"][spike_mask] / fs
        spike_amps = amps[spike_mask]
        ax.scatter(spike_times, spike_amps, s=3, alpha=0.5,
                   c="steelblue", edgecolors="none")
        med = np.median(spike_amps)
        ax.axhline(med, color="red", ls="--", lw=1, alpha=0.5,
                   label=f"median={med:.1f}µV")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("µV", fontsize=8)
        ax.set_title(f"Unit {uid} — {len(spike_amps)} spikes", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    for idx in range(n_amp, nrows_amp * ncols_amp):
        axes_arr[idx // ncols_amp, idx % ncols_amp].set_visible(False)
    fig.suptitle(f"Spike Amplitudes Over Time — {recording_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_spike_amplitudes_time.png")

    # ==================================================================
    # 6. RAW TRACES WITH SPIKE MARKERS
    # ==================================================================
    n_trace = min(6, n_units)
    trace_ids = snr_order[:n_trace]
    fig, axes_list = plt.subplots(n_trace, 1, figsize=(18, 3.5 * n_trace))
    if n_trace == 1:
        axes_list = [axes_list]
    n_samp = recording.get_num_samples()
    mid = n_samp // 2
    snippet_sf = max(0, int(mid - 0.5 * fs))
    snippet_ef = min(n_samp, int(mid + 0.5 * fs))
    t_snippet = np.arange(snippet_sf, snippet_ef) / fs
    for idx, uid in enumerate(trace_ids):
        ax = axes_list[idx]
        uid_idx = list(unit_ids).index(uid)
        uid_template = templates[uid_idx]
        peak_chan = np.argmin(np.min(uid_template, axis=0))
        ch_ids_all = recording.channel_ids
        peak_ch_id = ch_ids_all[peak_chan]
        trace = recording.get_traces(
            start_frame=snippet_sf, end_frame=snippet_ef,
            channel_ids=[peak_ch_id], return_in_uV=True
        ).flatten()
        ax.plot(t_snippet, trace, "k-", lw=0.3, alpha=0.7)
        spike_train = sorting.get_unit_spike_train(uid)
        spikes_in_win = spike_train[(spike_train >= snippet_sf) &
                                    (spike_train < snippet_ef)]
        for sp in spikes_in_win:
            ax.axvline(sp / fs, color="red", alpha=0.4, lw=0.8)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("µV", fontsize=9)
        ax.set_title(f"Unit {uid} — Ch {peak_chan} — "
                     f"SNR={qm.loc[uid, 'snr']:.1f}, "
                     f"{len(spikes_in_win)} spikes in window", fontsize=10)
    fig.suptitle(f"Raw Traces with Spike Markers — {recording_name}\n"
                 f"Black: filtered trace (µV), Red: detected spikes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_raw_traces_spikes.png")

    # ==================================================================
    # 7. SPATIAL FOOTPRINT
    # ==================================================================
    n_foot = min(6, n_units)
    foot_ids = snr_order[:n_foot]
    fig, axes_arr = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes_arr.flatten()
    for idx, uid in enumerate(foot_ids):
        ax = axes_flat[idx]
        uid_idx = list(unit_ids).index(uid)
        template_uv = templates[uid_idx]
        ptp = np.ptp(template_uv, axis=0)
        peak_chan = np.argmax(ptp)
        ch_start = max(0, peak_chan - 15)
        ch_end = min(template_uv.shape[1], peak_chan + 16)
        max_ptp = ptp[peak_chan]
        spacing = max(max_ptp * 2, 5)
        for ci, ch in enumerate(range(ch_start, ch_end)):
            wf = template_uv[:, ch]
            offset = ci * spacing
            alpha = 1.0 if ch == peak_chan else 0.3
            color = "red" if ch == peak_chan else "steelblue"
            ax.plot(t_ms, wf + offset, color=color, alpha=alpha, lw=0.8)
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Channel (offset, µV)", fontsize=9)
        ax.set_title(f"Unit {uid} — peak ch {peak_chan}\n"
                     f"PTP={ptp[peak_chan]:.1f}µV, "
                     f"SNR={qm.loc[uid, 'snr']:.1f}", fontsize=9)
    for idx in range(n_foot, 6):
        axes_flat[idx].set_visible(False)
    fig.suptitle(f"Spatial Footprint — {recording_name}\n"
                 f"Red = peak channel, Blue = neighbors",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_spatial_footprint.png")

    # ==================================================================
    # 8. ISI HISTOGRAMS
    # ==================================================================
    n_isi = min(12, n_units)
    isi_ids = snr_order[:n_isi]
    ncols_isi = min(4, n_isi)
    nrows_isi = max(1, int(np.ceil(n_isi / ncols_isi)))
    fig, axes_arr = plt.subplots(nrows_isi, ncols_isi,
                                 figsize=(4 * ncols_isi, 3 * nrows_isi))
    axes_arr = np.atleast_2d(axes_arr)
    if axes_arr.shape[0] == 1 and nrows_isi > 1:
        axes_arr = axes_arr.T
    for idx, uid in enumerate(isi_ids):
        r, c = divmod(idx, ncols_isi)
        ax = axes_arr[r, c]
        spike_train = sorting.get_unit_spike_train(uid)
        if len(spike_train) > 1:
            isis = np.diff(spike_train) / fs * 1000
            ax.hist(isis[isis < 100], bins=100, color="steelblue",
                    edgecolor="white", linewidth=0.3)
            ax.axvspan(0, 2, alpha=0.2, color="red")
            n_violations = np.sum(isis < 2.0)
            pct = 100.0 * n_violations / len(isis)
            ax.set_xlabel("ISI (ms)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title(f"Unit {uid} — {n_violations} viol ({pct:.1f}%)\n"
                         f"{len(spike_train)} spikes, "
                         f"FR={qm.loc[uid, 'firing_rate']:.1f}Hz", fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(n_isi, nrows_isi * ncols_isi):
        axes_arr[idx // ncols_isi, idx % ncols_isi].set_visible(False)
    fig.suptitle(f"ISI Distributions — {recording_name}\n"
                 f"Red zone: refractory period (<2ms)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "08_isi_histograms.png")

    # ==================================================================
    # 9. QUALITY METRICS SCATTER
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    sc = ax.scatter(qm["firing_rate"], qm["snr"],
                    c=qm["isi_violations_ratio"], cmap="RdYlGn_r",
                    vmin=0, vmax=0.5, s=30, edgecolors="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="ISI violations ratio")
    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("SNR")
    ax.set_xscale("log")
    ax.set_title("SNR vs Firing Rate\n(color = ISI violations)")
    ax = axes[1]
    if "amplitude_median" in qm.columns:
        ax.scatter(qm["snr"], qm["amplitude_median"],
                   c="steelblue", s=20, edgecolors="k", linewidth=0.3)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Amplitude median (µV)")
        ax.set_title("Amplitude vs SNR")
    ax = axes[2]
    ax.scatter(qm["firing_rate"], qm["presence_ratio"],
               c=qm["snr"], cmap="viridis", s=30, edgecolors="k", linewidth=0.3)
    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Presence Ratio")
    ax.set_xscale("log")
    ax.set_title("Presence Ratio vs Firing Rate\n(color = SNR)")
    plt.colorbar(ax.collections[0], ax=ax, label="SNR")
    fig.suptitle(f"Quality Metrics — {recording_name} ({n_units} units)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "09_quality_scatter.png")

    # ==================================================================
    # 10. DEPTH RASTER
    # ==================================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, uid in enumerate(unit_ids):
        st = sorting.get_unit_spike_train(uid) / fs
        depth = locs[i, 1]
        color = "tab:blue" if qm.loc[uid, "snr"] > qm["snr"].quantile(0.75) else "tab:gray"
        ax.plot(st, np.full_like(st, depth, dtype=float), "|",
                markersize=2, alpha=0.5, color=color)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Depth (µm)", fontsize=11)
    ax.set_title(f"Spike Raster by Depth — {recording_name}\n"
                 f"Blue = top 25% SNR, Gray = rest",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "10_depth_raster.png")

    # ==================================================================
    # 11. NOISE FLOOR
    # ==================================================================
    noise_uv = analyzer.get_extension("noise_levels").get_data()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(noise_uv, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Noise level (µV RMS)", fontsize=11)
    ax.set_ylabel("Channel count", fontsize=11)
    ax.set_title(f"Channel Noise Distribution\nMedian: {np.median(noise_uv):.2f} µV")
    ax.axvline(np.median(noise_uv), color="red", ls="--",
               label=f"median={np.median(noise_uv):.2f}µV")
    ax.legend()
    ax = axes[1]
    ax.plot(noise_uv, "o-", ms=2, lw=0.5, color="steelblue")
    ax.set_xlabel("Channel index", fontsize=11)
    ax.set_ylabel("Noise level (µV RMS)", fontsize=11)
    ax.set_title("Noise Level per Channel")
    ax.axhline(np.median(noise_uv), color="red", ls="--", alpha=0.5)
    fig.suptitle(f"Noise Floor — {recording_name}\n"
                 f"Expected NP1.0 thermal noise: ~5–10 µV RMS",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "11_noise_floor.png")

    # ==================================================================
    # 12. PTP AMPLITUDE HISTOGRAM
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ptps = [np.max(np.ptp(templates[i], axis=0)) for i in range(len(unit_ids))]
    ax.hist(ptps, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Peak-to-Peak Amplitude (µV)", fontsize=11)
    ax.set_ylabel("Unit count", fontsize=11)
    ax.axvline(np.median(ptps), color="red", ls="--",
               label=f"median={np.median(ptps):.1f}µV")
    ax.legend()
    ax.set_title(f"Template PTP Distribution — {recording_name}\n"
                 f"Expected for real neurons: >20 µV PTP",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "12_ptp_histogram.png")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# TIMECOURSE BIO-SIGNAL SCAN
# ══════════════════════════════════════════════════════════════════════════════

def compute_window_metrics(raw_int16, uv_per_bit, n_channels, fs=30_000.0):
    """Compute bio-quality metrics for a single time window.

    Applies bandpass filtering (300–6000 Hz) and common-median-referencing,
    then computes signal quality metrics without spike sorting.

    Parameters
    ----------
    raw_int16 : ndarray, shape (n_samples, n_channels)
        Raw unscaled trace data (int16 values).
    uv_per_bit : float
        Gain to convert raw → µV.
    n_channels : int
        Number of channels.
    fs : float
        Sampling rate in Hz. Default 30000.

    Returns
    -------
    dict
        Metrics including: ``rms_median``, ``rms_p95``, ``kurt_median``,
        ``kurt_p95``, ``xrate_median``, ``xrate_p95``,
        ``refrac_compliance``, ``mean_asymmetry``, ``active_frac``,
        ``bio_score``, ``best_waveforms``, ``cmr``.
    """
    from scipy import signal as scipy_signal
    from scipy.stats import kurtosis

    raw_uv = raw_int16.astype(np.float32) * uv_per_bit

    # Bandpass filter + CMR
    sos = scipy_signal.butter(3, [300, 6000], btype="bandpass", fs=fs, output="sos")
    bp = scipy_signal.sosfiltfilt(sos, raw_uv, axis=0).astype(np.float32)
    cmr = bp - np.median(bp, axis=1, keepdims=True)

    n_samples = cmr.shape[0]
    window_dur = n_samples / fs

    # 1. AP-band RMS
    cmr_rms = np.sqrt(np.mean(cmr ** 2, axis=0))
    rms_median = float(np.median(cmr_rms))
    rms_p95 = float(np.percentile(cmr_rms, 95))

    # 2. Kurtosis
    kurt_vals = np.array([kurtosis(cmr[:, ci]) for ci in range(n_channels)])
    kurt_vals = np.where(np.isfinite(kurt_vals), kurt_vals, 0.0)
    kurt_median = float(np.median(kurt_vals))
    kurt_p95 = float(np.percentile(kurt_vals, 95))

    # 3. Threshold crossings per channel
    crossing_rates = np.zeros(n_channels)
    mad_per_ch = np.zeros(n_channels)
    for ci in range(n_channels):
        trace = cmr[:, ci]
        med = np.median(trace)
        mad = np.median(np.abs(trace - med)) * 1.4826
        mad_per_ch[ci] = mad
        if mad > 1e-10:
            thresh = _THRESH_SIGMA * mad
            crossings = np.sum(np.diff((trace < thresh).astype(np.int8)) == 1)
            crossing_rates[ci] = crossings / window_dur

    xrate_median = float(np.median(crossing_rates))
    xrate_p95 = float(np.percentile(crossing_rates, 95))

    # 4–6. Waveform analysis on top-20 kurtosis channels
    top_k_chs = np.argsort(kurt_vals)[-20:]
    refrac_compliances = []
    asymmetries = []
    best_waveforms = []

    for ci in top_k_chs:
        trace = cmr[:, ci]
        mad = mad_per_ch[ci]
        if mad < 1e-10:
            continue
        thresh = _THRESH_SIGMA * mad
        cross_idx = np.where(np.diff((trace < thresh).astype(np.int8)) == 1)[0]
        if len(cross_idx) < 3:
            continue

        # Refine to trough
        troughs = []
        for cx in cross_idx:
            s = max(0, cx - 15)
            e = min(len(trace), cx + 15)
            troughs.append(s + np.argmin(trace[s:e]))
        troughs = np.unique(np.array(troughs))

        if len(troughs) > 1:
            isi_ms = np.diff(troughs) / fs * 1000
            refrac = float(np.sum(isi_ms > 1.5) / len(isi_ms))
            refrac_compliances.append(refrac)
        else:
            continue

        # Extract waveform snippets
        snips = []
        for tr in troughs:
            if _N_BEFORE <= tr < len(trace) - _N_AFTER:
                snips.append(trace[tr - _N_BEFORE:tr + _N_AFTER])
        if len(snips) < 3:
            continue

        snips = np.array(snips)
        mean_wf = np.mean(snips, axis=0)
        neg = float(np.abs(np.min(mean_wf)))
        pos = float(np.abs(np.max(mean_wf)))
        asym = neg / (pos + 1e-10)
        asymmetries.append(asym)

        trough_i = np.argmin(mean_wf)
        half_val = mean_wf[trough_i] / 2
        hw_samp = np.sum(mean_wf < half_val)
        hw_ms = hw_samp / fs * 1000

        post_trough = mean_wf[trough_i:]
        p2t_ms = float(np.argmax(post_trough) / fs * 1000) if len(post_trough) > 3 else 0

        best_waveforms.append({
            "ci": int(ci), "asym": asym, "hw_ms": hw_ms, "p2t_ms": p2t_ms,
            "rate": float(len(troughs) / window_dur),
            "amplitude": neg, "kurt": float(kurt_vals[ci]),
            "refrac": refrac, "snips": snips, "mean_wf": mean_wf,
            "troughs": troughs, "isi_ms": isi_ms,
        })

    refrac_compliance = float(np.mean(refrac_compliances)) if refrac_compliances else 0.0
    mean_asymmetry = float(np.mean(asymmetries)) if asymmetries else 1.0

    # 6. Spatial localization
    active_frac = float(np.sum(crossing_rates > 10) / n_channels)

    # 7. Composite bio-score
    kurt_score = min(kurt_p95 / 10.0, 1.0)
    refrac_score = refrac_compliance
    asym_score = min(mean_asymmetry / 5.0, 1.0)
    local_score = 1.0 - min(active_frac / 0.3, 1.0)
    bio_score = (kurt_score * 0.3 + refrac_score * 0.3
                 + asym_score * 0.2 + local_score * 0.2)

    best_waveforms.sort(key=lambda x: x["asym"], reverse=True)

    return {
        "rms_median": rms_median, "rms_p95": rms_p95,
        "kurt_median": kurt_median, "kurt_p95": kurt_p95,
        "xrate_median": xrate_median, "xrate_p95": xrate_p95,
        "refrac_compliance": refrac_compliance,
        "mean_asymmetry": mean_asymmetry,
        "active_frac": active_frac, "bio_score": bio_score,
        "best_waveforms": best_waveforms, "cmr": cmr,
    }


def run_timescan(recording, window_s=2.0, stride_s=1.0):
    """Scan a recording's full timecourse for biological signal quality.

    Slides a window across the recording and computes per-window bio-quality
    metrics. No spike sorting — just fast signal-level analysis.

    Parameters
    ----------
    recording : BaseRecording
        Input recording (with gains set for µV scaling).
    window_s : float
        Analysis window duration in seconds. Default 2.0.
    stride_s : float
        Stride between windows in seconds. Default 1.0.

    Returns
    -------
    dict
        Results with keys:
        - ``times``: list of window start times (seconds)
        - ``metrics``: list of metric dicts from ``compute_window_metrics``
        - ``best_indices``: array of window indices sorted by bio-score (best first)
        - ``scores``: array of bio-scores
    """
    fs = recording.get_sampling_frequency()
    n_ch = recording.get_num_channels()
    n_samp = recording.get_num_samples()
    uv_per_bit = recording.get_channel_gains()[0]

    window_samples = int(window_s * fs)
    stride_samples = int(stride_s * fs)

    window_starts = []
    sf = 0
    while sf + window_samples <= n_samp:
        window_starts.append(sf)
        sf += stride_samples

    n_windows = len(window_starts)
    window_times = [s / fs for s in window_starts]

    logger.info("Scanning %d windows (%.1fs window, %.1fs stride)",
                n_windows, window_s, stride_s)

    metrics_list = []
    for wi, sf in enumerate(window_starts):
        ef = sf + window_samples
        raw = recording.get_traces(start_frame=sf, end_frame=ef,
                                   return_in_uV=False)
        m = compute_window_metrics(raw, uv_per_bit, n_ch, fs)
        metrics_list.append(m)

        if (wi + 1) % 10 == 0 or wi == n_windows - 1:
            logger.info(
                "  [%d/%d] t=%.0fs  bio=%.3f  kurt_p95=%.1f  "
                "refrac=%.2f  asym=%.1f",
                wi + 1, n_windows, sf / fs,
                m["bio_score"], m["kurt_p95"],
                m["refrac_compliance"], m["mean_asymmetry"],
            )

    scores = np.array([m["bio_score"] for m in metrics_list])
    best_indices = np.argsort(scores)[::-1]

    return {
        "times": window_times,
        "metrics": metrics_list,
        "best_indices": best_indices,
        "scores": scores,
    }


def plot_timecourse_dashboard(scan_result, recording_name, output_path,
                              window_s=2.0, stride_s=1.0):
    """Plot the 6-panel timecourse dashboard.

    Parameters
    ----------
    scan_result : dict
        Output from :func:`run_timescan`.
    recording_name : str
        Label for figure titles.
    output_path : str or Path
        Where to save the figure.
    window_s : float
        Window duration (for display). Default 2.0.
    stride_s : float
        Stride (for display). Default 1.0.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = scan_result["times"]
    metrics_list = scan_result["metrics"]
    best_indices = scan_result["best_indices"]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    t = np.array(times)

    def _highlight_best(ax):
        for bi in best_indices[:3]:
            ax.axvspan(t[bi] - stride_s / 2, t[bi] + window_s - stride_s / 2,
                       alpha=0.15, color="gold", zorder=0)
        ax.axvline(t[best_indices[0]], color="green", ls="--", lw=1, alpha=0.7,
                   label=f"Best @ {t[best_indices[0]]:.0f}s")

    # Row 1: RMS + crossing rate
    ax = axes[0, 0]
    ax.plot(t, [m["rms_median"] for m in metrics_list], "b-", lw=1.5, label="Median")
    ax.plot(t, [m["rms_p95"] for m in metrics_list], "b--", lw=0.8, alpha=0.5, label="P95")
    _highlight_best(ax)
    ax.set_ylabel("CMR RMS (µV)")
    ax.set_title("AP-Band RMS Over Time", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, [m["xrate_median"] for m in metrics_list], "g-", lw=1.5, label="Median")
    ax.plot(t, [m["xrate_p95"] for m in metrics_list], "g--", lw=0.8, alpha=0.5, label="P95")
    _highlight_best(ax)
    ax.set_ylabel("Crossings/s")
    ax.set_title("Threshold Crossing Rate (-4×MAD)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: Kurtosis + refractory compliance
    ax = axes[1, 0]
    ax.plot(t, [m["kurt_median"] for m in metrics_list], "orange", lw=1.5, label="Median")
    ax.plot(t, [m["kurt_p95"] for m in metrics_list], "red", lw=1, alpha=0.7, label="P95")
    ax.axhline(3, color="gray", ls=":", lw=0.8, label="Gaussian baseline")
    _highlight_best(ax)
    ax.set_ylabel("Kurtosis")
    ax.set_title("Channel Kurtosis Over Time (>3 = spike activity)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, [m["refrac_compliance"] for m in metrics_list], "purple", lw=1.5)
    ax.axhline(0.9, color="green", ls=":", lw=0.8, label=">90% = biological")
    _highlight_best(ax)
    ax.set_ylabel("Refractory compliance")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Refractory Period Compliance (ISI > 1.5ms)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 3: Asymmetry + composite bio-score
    ax = axes[2, 0]
    ax.plot(t, [m["mean_asymmetry"] for m in metrics_list], "teal", lw=1.5)
    ax.axhline(2.0, color="green", ls=":", lw=0.8, label=">2.0 = spike-like")
    _highlight_best(ax)
    ax.set_ylabel("Neg/Pos asymmetry")
    ax.set_xlabel("Time (s)")
    ax.set_title("Waveform Asymmetry", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    bio_scores = [m["bio_score"] for m in metrics_list]
    ax.plot(t, bio_scores, "k-", lw=2)
    ax.fill_between(t, bio_scores, alpha=0.2, color="steelblue")
    _highlight_best(ax)
    ax.set_ylabel("Bio-score")
    ax.set_xlabel("Time (s)")
    ax.set_title("Composite Biological Score", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    best_t = t[best_indices[0]]
    best_s = metrics_list[best_indices[0]]["bio_score"]
    fig.suptitle(f"Timecourse Bio-Signal Scan — {recording_name}\n"
                 f"Window={window_s}s, Stride={stride_s}s | "
                 f"Best: {best_t:.0f}–{best_t + window_s:.0f}s "
                 f"(bio-score={best_s:.3f})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_best_windows(scan_result, ch_ids, recording_name, output_path,
                      fs=30_000.0):
    """Deep-dive visualization of the top-3 bio-scoring windows.

    Parameters
    ----------
    scan_result : dict
        Output from :func:`run_timescan`.
    ch_ids : array-like
        Channel IDs for labeling.
    recording_name : str
        Label for figure titles.
    output_path : str or Path
        Where to save the figure.
    fs : float
        Sampling rate. Default 30000.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = scan_result["times"]
    metrics_list = scan_result["metrics"]
    best_indices = scan_result["best_indices"]
    window_s = times[1] - times[0] if len(times) > 1 else 2.0

    n_best = min(3, len(best_indices))
    t_wf = np.arange(_N_BEFORE + _N_AFTER) / fs * 1000 - _N_BEFORE / fs * 1000

    fig, axes = plt.subplots(n_best, 3, figsize=(20, 5 * n_best))
    if n_best == 1:
        axes = axes[np.newaxis, :]

    for row, bi in enumerate(best_indices[:n_best]):
        m = metrics_list[bi]
        t_start = times[bi]
        cmr = m["cmr"]
        wfs = m["best_waveforms"]

        # Column 1: CMR traces (10ms)
        ax = axes[row, 0]
        n_show_samples = int(0.010 * fs)
        t_ms = np.arange(n_show_samples) / fs * 1000
        if wfs:
            top_cis = [w["ci"] for w in wfs[:5]]
        else:
            cmr_rms = np.sqrt(np.mean(cmr ** 2, axis=0))
            top_cis = list(np.argsort(cmr_rms)[-5:])
        offset = 0
        for ci in top_cis:
            trace_10ms = cmr[:n_show_samples, ci]
            ptp = np.ptp(trace_10ms)
            ax.plot(t_ms, trace_10ms + offset, lw=0.6, alpha=0.8,
                    label=f"ch{ch_ids[ci]}")
            offset += max(ptp * 1.5, 0.5)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV, offset)")
        ax.set_title(f"CMR traces (10ms)\nWindow {t_start:.0f}–"
                     f"{t_start + window_s:.0f}s, "
                     f"score={m['bio_score']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right")

        # Column 2: Best waveforms
        ax = axes[row, 1]
        n_wf_show = min(4, len(wfs))
        if n_wf_show > 0:
            colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
            offset = 0
            for wi in range(n_wf_show):
                w = wfs[wi]
                snips = w["snips"]
                n_overlay = min(30, snips.shape[0])
                for si_idx in range(n_overlay):
                    ax.plot(t_wf, snips[si_idx] + offset, color=colors[wi],
                            alpha=0.07, lw=0.3)
                ax.plot(t_wf, w["mean_wf"] + offset, color=colors[wi], lw=2,
                        label=f"ch{ch_ids[w['ci']]} asym={w['asym']:.1f} "
                              f"rate={w['rate']:.0f}/s")
                offset += np.ptp(w["mean_wf"]) * 2.5
            ax.legend(fontsize=6, loc="upper right")
        else:
            ax.text(0.5, 0.5, "No spike-like waveforms found",
                    transform=ax.transAxes, ha="center", fontsize=10)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axvline(0, color="red", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"Best waveforms (top {n_wf_show} by asymmetry)\n"
                     f"Refrac compliance: {m['refrac_compliance']:.2f}",
                     fontsize=10, fontweight="bold")

        # Column 3: ISI histogram
        ax = axes[row, 2]
        if wfs and "isi_ms" in wfs[0] and len(wfs[0]["isi_ms"]) > 2:
            isi = wfs[0]["isi_ms"]
            ax.hist(isi[isi < 50], bins=50, color="steelblue",
                    edgecolor="white", linewidth=0.3)
            ax.axvspan(0, 1.5, alpha=0.2, color="red",
                       label="Refractory (<1.5ms)")
            n_viol = np.sum(isi < 1.5)
            pct_viol = 100 * n_viol / len(isi)
            ax.set_xlabel("ISI (ms)")
            ax.set_ylabel("Count")
            ax.set_title(f"ISI: ch{ch_ids[wfs[0]['ci']]} — "
                         f"{n_viol}/{len(isi)} violations ({pct_viol:.1f}%)\n"
                         f"Rate={wfs[0]['rate']:.1f}/s, "
                         f"{len(wfs[0]['troughs'])} events",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Not enough events for ISI",
                    transform=ax.transAxes, ha="center", fontsize=10)

    fig.suptitle(f"Best Windows Deep Dive — {recording_name}\n"
                 f"Top {n_best} windows by composite bio-score",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)
