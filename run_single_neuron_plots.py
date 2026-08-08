"""
Generate individual unit summary figures for all surveys.

One figure per unit labelled 'neural' by the custom classifier
(labels_custom.csv, written by apply_custom_classifier.py), for banks 0-6.
All figures go to figures/sorting/survey_<key>/units/.

Each figure is a 5-panel layout:
  1. Probe location map (electrode positions, peak channel highlighted)
  2. Multi-channel template waveforms (sparse channels)
  3. Peak-channel mean waveform +/- SD
  4. Autocorrelogram (1 ms bins, +/-50 ms window)
  5. Spike amplitudes over time (stability check)

Uses cached SortingAnalyzer data — no raw recording needed.

Usage:
  python run_single_neuron_plots.py              # all surveys
  python run_single_neuron_plots.py 20260316     # single survey
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import spikeinterface.full as si

# ── Add repo root to path so we can import from track_neurons ──
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from track_neurons import (load_analyzer_data, get_good_unit_ids,
                            get_ks4_good_unit_ids, get_classifier_good_unit_ids)

# ── Config ────────────────────────────────────────────────────────────
DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
SORT_ROOT = DATA_ROOT / 'sorting_output'
FIG_ROOT = DATA_ROOT / 'figures' / 'sorting'

SURVEYS = ['20260226', '20260302', '20260304', '20260306', '20260309', '20260311', '20260313', '20260316', '20260319', '20260323', '20260326', '20260331', '20260402', '20260407', '20260409', '20260414', '20260417', '20260421', '20260424', '20260428', '20260501', '20260507', '20260512', '20260522', '20260526', '20260604', '20260611', '20260618', '20260625', '20260723', '20260730']
N_BANKS = 7


# ── Custom unit summary ────────────────────────────────────────────────
def plot_custom_unit_summary(analyzer, uid, fig):
    """Custom unit summary: probe layout, template traces, peak waveform, ACG, amplitudes."""
    sorting = analyzer.sorting
    fs = analyzer.sampling_frequency
    uid_idx = list(sorting.unit_ids).index(uid)

    # Data extraction
    templates = analyzer.get_extension('templates').get_data()
    template = templates[uid_idx]
    ch_locs = analyzer.get_channel_locations()
    sparse_ch = analyzer.sparsity.unit_id_to_channel_indices[uid]

    ptp_per_ch = np.ptp(template, axis=0)
    peak_ch = np.argmax(ptp_per_ch)
    max_ptp = ptp_per_ch[peak_ch]

    wf_ext = analyzer.get_extension('waveforms')
    nbefore = int(wf_ext.nbefore)
    n_samples = template.shape[0]
    t_ms = (np.arange(n_samples) - nbefore) / fs * 1000

    waveforms = wf_ext.get_waveforms_one_unit(uid)
    peak_in_sparse = np.where(sparse_ch == peak_ch)[0]

    ccgs, bins = analyzer.get_extension('correlograms').get_data()
    auto_ccg = ccgs[uid_idx, uid_idx, :]

    amp_data = analyzer.get_extension('spike_amplitudes').get_data(
        outputs='by_unit', concatenated=False)
    amps = amp_data[0][uid]
    spike_train = sorting.get_unit_spike_train(uid, segment_index=0)
    spike_times_s = spike_train / fs
    if len(spike_times_s) > 0:
        spike_times_s = spike_times_s - spike_times_s.min()

    # Layout
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[3, 1],
                           wspace=0.4, hspace=0.4)
    ax_probe = fig.add_subplot(gs[0, 0])
    ax_traces = fig.add_subplot(gs[0, 1])
    ax_wf = fig.add_subplot(gs[0, 2])
    ax_acg = fig.add_subplot(gs[0, 3])
    ax_amp = fig.add_subplot(gs[1, :])

    peak_y = ch_locs[peak_ch, 1]

    # Panel 1: Probe layout
    ax_probe.scatter(ch_locs[:, 0], ch_locs[:, 1], s=6, c='lightgrey',
                     marker='s', edgecolors='none', zorder=1)
    ax_probe.scatter(ch_locs[sparse_ch, 0], ch_locs[sparse_ch, 1],
                     s=18, c=ptp_per_ch[sparse_ch], cmap='hot',
                     marker='s', edgecolors='none', zorder=2)
    ax_probe.scatter([ch_locs[peak_ch, 0]], [ch_locs[peak_ch, 1]],
                     s=60, facecolors='none', edgecolors='blue',
                     marker='o', linewidths=2, zorder=3)
    ax_probe.set_ylim(peak_y - 400, peak_y + 400)
    ax_probe.set_xlim(ch_locs[:, 0].min() - 20, ch_locs[:, 0].max() + 20)
    ax_probe.set_ylabel('Depth (µm)')
    ax_probe.set_title('Probe', fontsize=10)

    # Panel 2: Template traces at spatial positions
    x_unique = np.sort(np.unique(ch_locs[sparse_ch, 0]))
    col_gap = (x_unique[-1] - x_unique[0]) if len(x_unique) > 1 else 80
    sorted_depths = np.sort(np.unique(ch_locs[sparse_ch, 1]))
    row_gap = (np.min(np.diff(sorted_depths))
               if len(sorted_depths) > 1 else 20)

    t_range = t_ms[-1] - t_ms[0]
    t_center = (t_ms[0] + t_ms[-1]) / 2
    x_scale = col_gap * 0.45 / t_range
    y_scale = row_gap * 2.5 / max_ptp if max_ptp > 0 else 1

    for ch_idx in sparse_ch:
        trace = template[:, ch_idx]
        cx, cy = ch_locs[ch_idx]
        x_plot = cx + (t_ms - t_center) * x_scale
        y_plot = cy + trace * y_scale
        is_peak = (ch_idx == peak_ch)
        ax_traces.plot(x_plot, y_plot,
                       color='blue' if is_peak else 'grey',
                       lw=1.5 if is_peak else 0.5,
                       alpha=1.0 if is_peak else 0.5)

    bar_y = sorted_depths[0] - row_gap * 2
    bar_x0 = x_unique[0] - 0.5 * x_scale
    bar_x1 = bar_x0 + 1.0 * x_scale
    ax_traces.plot([bar_x0, bar_x1], [bar_y, bar_y], 'k-', lw=2, clip_on=False)
    ax_traces.text((bar_x0 + bar_x1) / 2, bar_y - row_gap * 0.5,
                   '1 ms', ha='center', va='top', fontsize=7)

    ax_traces.set_xlim(x_unique[0] - col_gap * 0.55,
                       x_unique[-1] + col_gap * 0.55)
    ax_traces.set_ylim(peak_y - 400, peak_y + 400)
    ax_traces.set_xlabel('Position (µm)')
    ax_traces.set_ylabel('Depth (µm)')
    ax_traces.set_title('Templates', fontsize=10)

    # Panel 3: Peak waveform overlay
    if len(peak_in_sparse) > 0:
        pi = peak_in_sparse[0]
        n_show = min(50, waveforms.shape[0])
        show_idx = (np.random.choice(waveforms.shape[0], n_show, replace=False)
                    if waveforms.shape[0] > n_show
                    else np.arange(waveforms.shape[0]))
        for idx in show_idx:
            ax_wf.plot(t_ms, waveforms[idx, :, pi],
                       color='grey', alpha=0.15, lw=0.5)
        mean_wf = np.mean(waveforms[:, :, pi], axis=0)
        std_wf = np.std(waveforms[:, :, pi], axis=0)
        ax_wf.fill_between(t_ms, mean_wf - std_wf, mean_wf + std_wf,
                           alpha=0.25, color='blue')
        ax_wf.plot(t_ms, mean_wf, color='blue', lw=2)
    ax_wf.axvline(0, color='k', ls=':', alpha=0.3, lw=0.5)
    ax_wf.set_xlabel('Time (ms)')
    ax_wf.set_ylabel('µV')
    ax_wf.set_title('Peak waveform', fontsize=10)

    # Panel 4: Autocorrelogram
    bin_c = (bins[:-1] + bins[1:]) / 2
    bw = bins[1] - bins[0]
    ax_acg.bar(bin_c, auto_ccg, width=bw * 0.9,
               color='steelblue', edgecolor='none')
    ax_acg.set_xlabel('Lag (ms)')
    ax_acg.set_ylabel('Count')
    ax_acg.set_title('ACG', fontsize=10)

    # Panel 5: Amplitude over time
    n_pts = min(len(amps), len(spike_times_s))
    if n_pts > 0:
        ax_amp.scatter(spike_times_s[:n_pts], amps[:n_pts],
                       s=1, alpha=0.3, c='steelblue', rasterized=True)
        ax_amp.axhline(np.median(amps[:n_pts]), color='red',
                       lw=1, ls='--', alpha=0.7)
    ax_amp.set_xlabel('Time (s)')
    ax_amp.set_ylabel('Amplitude')
    ax_amp.set_title('Amplitude over time', fontsize=10)


# ── Main ──────────────────────────────────────────────────────────────
def run_survey(survey_key):
    """Generate single-neuron plots for one survey.

    Uses the custom classifier's 'neural' label (labels_custom.csv, written by
    apply_custom_classifier.py) — one figure per neural unit, all in units/.
    """
    sort_dir = SORT_ROOT / f'survey_{survey_key}'
    fig_dir_good = FIG_ROOT / f'survey_{survey_key}' / 'units'
    fig_dir_good.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SURVEY %s", survey_key)
    logger.info("  units -> %s", fig_dir_good)
    logger.info("=" * 60)

    total_good_saved = 0
    global_unit_idx = 0

    for bank_id in range(N_BANKS):
        cache_dir = sort_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
        if not cache_dir.exists():
            logger.warning("  Bank %d: no analyzer cache, skipping", bank_id)
            continue

        # Load analyzer from cache
        try:
            analyzer = si.load_sorting_analyzer(cache_dir)
            for ext_name in analyzer.get_saved_extension_names():
                try:
                    analyzer.load_extension(ext_name)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("  Bank %d: failed to load analyzer: %s", bank_id, e)
            continue

        sorting = analyzer.sorting
        n_units = len(sorting.unit_ids)
        if n_units == 0:
            logger.info("  Bank %d: 0 units, skipping", bank_id)
            continue

        # Use the custom classifier's neural/noise labels
        data = load_analyzer_data(cache_dir)
        good_ids = set(get_classifier_good_unit_ids(sort_dir, bank_id))

        qm = data['qm']
        templates_arr = data['templates']

        bank_good = 0
        for uid_idx, uid in enumerate(sorting.unit_ids):
            if uid not in good_ids:
                global_unit_idx += 1
                continue

            # Unit metadata for title
            template = templates_arr[uid_idx]
            ptp_per_ch = np.ptp(template, axis=0)
            peak_ptp = np.max(ptp_per_ch)
            peak_ch = np.argmax(ptp_per_ch)
            n_spat = int(np.sum(ptp_per_ch > 0.25 * peak_ptp))
            ch_locs = analyzer.get_channel_locations()
            depth = ch_locs[peak_ch, 1] if peak_ch < len(ch_locs) else np.nan

            fr = float(qm.loc[uid, 'firing_rate']) if uid in qm.index else 0
            snr_val = float(qm.loc[uid, 'snr']) if uid in qm.index else 0

            # Noise floor at peak channel
            noise_str = ''
            if data['noise_levels'] is not None:
                noise_str = f"  |  Noise={data['noise_levels'][peak_ch]:.1f}uV"

            try:
                fig = plt.figure(figsize=(16, 9))
                plot_custom_unit_summary(analyzer, uid, fig)

                fig.suptitle(
                    f"Unit {global_unit_idx}  |  Bank {bank_id}  |  "
                    f"{depth / 1000:.1f} mm  |  "
                    f"Amp={peak_ptp:.0f} uV  |  FR={fr:.1f} Hz  |  "
                    f"SNR={snr_val:.1f}{noise_str}",
                    fontsize=13, fontweight='bold', y=1.02)

                fname = f'unit_{global_unit_idx:04d}_bank{bank_id:02d}.png'
                fig.savefig(fig_dir_good / fname, dpi=150, bbox_inches='tight',
                            facecolor='white')
                plt.close(fig)
                bank_good += 1
            except Exception as e:
                logger.warning("  Failed to plot unit %d (uid %s): %s",
                               global_unit_idx, uid, e)
                plt.close('all')

            global_unit_idx += 1

        logger.info("  Bank %d: %d figures saved", bank_id, bank_good)
        total_good_saved += bank_good

    logger.info("Survey %s: %d figures", survey_key, total_good_saved)
    return total_good_saved, 0


def main():
    surveys = sys.argv[1:] if len(sys.argv) > 1 else SURVEYS
    grand_good = 0
    grand_suspect = 0
    for key in surveys:
        good, suspect = run_survey(key)
        grand_good += good
        grand_suspect += suspect
    logger.info("\nDone! %d good + %d suspect = %d total figures across %d surveys.",
                 grand_good, grand_suspect, grand_good + grand_suspect, len(surveys))


if __name__ == '__main__':
    main()
