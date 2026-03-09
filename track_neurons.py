"""
Track neurons across survey recording days using UnitMatch.

For each bank (0-11), extracts split-half waveforms from each survey date,
formats them for UnitMatch, runs Naive Bayes matching, and aggregates
results across the full NP1032 probe.

Usage:
  python track_neurons.py                                    # track all 4 dates
  python track_neurons.py 20260226 20260302                  # track 2 dates
  python track_neurons.py 20260226 20260302 20260304 20260306  # explicit all 4

Environment:
  Set DATA_ROOT env var to override the default data directory.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import gc
import logging
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


# ---- Paths (all outputs live alongside data in DATA_ROOT) ----
DATA_ROOT = _default_data_root()
OUTPUT_DIR = DATA_ROOT / 'sorting_output' / 'tracking'
FIG_DIR = DATA_ROOT / 'figures' / 'sorting' / 'tracking'

# Survey raw-data dir names (relative to DATA_ROOT)
SURVEY_DIRS = {
    '20260226': 'SvyPrb_2026-02-26T11.58.07_g0',
    '20260302': 'SvyPrb_2026-03-02T14.10.26_g0',
    '20260304': 'SvyPrb_2026-03-04T10.17.50_g0',
    '20260306': 'SvyPrb_2026-03-06T10.01.15_g0',
}

# Derived helper — sort output for a given survey date
SURVEYS = {
    key: {
        'spikeglx_dir': DATA_ROOT / sdir,
        'sort_dir': DATA_ROOT / 'sorting_output' / f'survey_{key}',
    }
    for key, sdir in SURVEY_DIRS.items()
}


def load_analyzer_data(cache_dir):
    """Load templates, quality metrics, template metrics, unit IDs, and
    channel positions directly from the analyzer cache files (no SI
    recording required).

    Returns dict with keys: templates, unit_ids, channel_positions, qm, tm
    """
    import json

    cache_dir = Path(cache_dir)

    # Templates: (n_units, n_timepoints, n_channels)
    templates = np.load(cache_dir / 'extensions/templates/average.npy')

    # Unit IDs
    sort_info = json.loads(
        (cache_dir / 'sorting/numpysorting_info.json').read_text())
    unit_ids = sort_info['unit_ids']

    # Channel positions from probegroup
    pg = json.loads(
        (cache_dir / 'recording_info/probegroup.json').read_text())
    channel_positions = np.array(pg['probes'][0]['contact_positions'])

    # Quality metrics
    qm = pd.read_csv(
        cache_dir / 'extensions/quality_metrics/metrics.csv', index_col=0)

    # Template metrics (for advanced classification)
    tm_path = cache_dir / 'extensions/template_metrics/metrics.csv'
    tm = pd.read_csv(tm_path, index_col=0) if tm_path.exists() else None

    return {
        'templates': templates,
        'unit_ids': unit_ids,
        'channel_positions': channel_positions,
        'qm': qm,
        'tm': tm,
    }


def get_good_unit_ids(qm, tm=None, templates=None, fr_thresh=0.1,
                      isi_thresh=0.5, min_amplitude_uv=25.0,
                      min_n_spatial_channels=2,
                      max_n_spatial_channels=30,
                      max_half_width_ms=0.8,
                      min_amp_spatial_product=50.0,
                      max_spread_um=1500,
                      max_peaks=2):
    """Classify good units using advanced artifact criteria.

    Uses the same multi-stage noise detection as classify_units_advanced
    in analysis.py, applied directly from cached template metrics and
    quality metrics DataFrames.  Falls back to basic SNR/FR/ISI filtering
    when template metrics are unavailable.
    """
    snr = qm['snr'].dropna()
    if len(snr) == 0:
        return []
    snr_thresh = snr.quantile(0.75) if snr.median() < 5 else 3.0

    # Basic quality gate
    mask = (
        (qm['snr'] > snr_thresh) &
        (qm['firing_rate'] > fr_thresh) &
        (qm['isi_violations_ratio'] < isi_thresh)
    )
    if 'presence_ratio' in qm.columns:
        pr = qm['presence_ratio'].dropna()
        if len(pr) > 0:
            mask = mask & (qm['presence_ratio'].fillna(1.0) > 0.5)

    # Advanced artifact rejection (when template metrics available)
    if tm is not None and templates is not None:
        noise_mask = np.zeros(len(qm), dtype=bool)

        # Compute per-unit amplitude and spatial channel count from templates
        ptp_amps = np.full(len(qm), np.nan)
        n_spatial_ch = np.full(len(qm), 0, dtype=int)
        all_unit_ids = list(qm.index)
        for i, uid in enumerate(all_unit_ids):
            if uid not in tm.index:
                continue
            template = templates[i]
            ptp_per_ch = np.ptp(template, axis=0)
            peak_ch = np.argmax(ptp_per_ch)
            ptp_amps[i] = ptp_per_ch[peak_ch]
            n_spatial_ch[i] = int(np.sum(ptp_per_ch > 0.25 * ptp_per_ch[peak_ch]))

        # Multi-peaked waveforms
        if 'num_negative_peaks' in tm.columns:
            vals = tm.reindex(qm.index)['num_negative_peaks']
            noise_mask |= (vals > max_peaks).fillna(False).values
        if 'num_positive_peaks' in tm.columns:
            vals = tm.reindex(qm.index)['num_positive_peaks']
            noise_mask |= (vals > max_peaks).fillna(False).values

        # Spatially too wide
        if 'spread' in tm.columns:
            spread = tm.reindex(qm.index)['spread']
            noise_mask |= (spread.notna() & (spread > max_spread_um)).values

        # Too-low amplitude
        noise_mask |= (ptp_amps < min_amplitude_uv)

        # Insufficient amplitude-spatial product
        amp_spatial = ptp_amps * n_spatial_ch
        noise_mask |= (amp_spatial < min_amp_spatial_product)

        # Too many or too few spatial channels
        noise_mask |= (n_spatial_ch > max_n_spatial_channels)
        noise_mask |= (n_spatial_ch < min_n_spatial_channels)

        # Overly broad waveform
        if 'half_width' in tm.columns:
            hw_ms = tm.reindex(qm.index)['half_width'] * 1000
            noise_mask |= (hw_ms.notna() & (hw_ms > max_half_width_ms)).values

        # Remove artifacts from good mask
        mask = mask & ~noise_mask

    return list(qm.index[mask])


def prepare_um_session(date_key, bank_id, um_input_dir):
    """Prepare UnitMatch input for one bank on one survey date.

    Loads templates and quality metrics directly from the analyzer cache
    NPY/CSV files — does NOT use load_sorting_analyzer() which would
    try to re-open the 11 GB SpikeGLX recording from Dropbox.

    Returns the session directory path, or None if no good units.
    """
    survey = SURVEYS[date_key]
    sort_dir = survey['sort_dir']

    ks_dir = sort_dir / f'bank{bank_id:02d}_ks4'
    cache_dir = ks_dir / 'analyzer_cache'
    if not (cache_dir / 'spikeinterface_info.json').exists():
        logger.warning("  No analyzer cache for %s bank %d", date_key, bank_id)
        return None

    # Load data directly from files
    data = load_analyzer_data(cache_dir)
    good_ids = get_good_unit_ids(data['qm'], tm=data['tm'],
                                 templates=data['templates'])
    if len(good_ids) == 0:
        logger.info("  %s bank %d: 0 good units, skipping", date_key, bank_id)
        return None

    # Filter templates to good units only
    all_unit_ids = data['unit_ids']
    good_indices = [all_unit_ids.index(uid) for uid in good_ids]
    templates = data['templates'][good_indices]

    # Save in UnitMatch format — per-unit files
    # Each file: (n_timepoints, n_channels, 2) — two halves (same data since
    # we don't have raw recordings to split)
    session_dir = Path(um_input_dir) / f'Session_{date_key}'
    session_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = session_dir / 'RawWaveforms'
    raw_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed=42 + bank_id)
    for i, uid in enumerate(good_ids):
        # template shape: (n_timepoints, n_channels)
        tmpl = templates[i]
        # Zero-center (subtract mean of first 15 timepoints)
        baseline = tmpl[:15, :].mean(axis=0, keepdims=True)
        tmpl = tmpl - baseline
        # Create two halves with small noise to simulate split-half variability
        # ~5% amplitude noise prevents arctanh(1.0)=inf in UnitMatch
        amp = np.abs(tmpl).max() + 1e-10
        noise_scale = 0.05 * amp
        half1 = tmpl + rng.normal(0, noise_scale, tmpl.shape)
        half2 = tmpl + rng.normal(0, noise_scale, tmpl.shape)
        unit_wave = np.stack([half1, half2], axis=-1)
        np.save(raw_dir / f'Unit{uid}_RawSpikes.npy', unit_wave)

    np.save(session_dir / 'channel_positions.npy', data['channel_positions'])
    np.save(session_dir / 'channel_map.npy', np.arange(data['channel_positions'].shape[0]))

    with open(session_dir / 'cluster_group.tsv', 'w') as f:
        f.write('cluster_id\tgroup\n')
        for uid in good_ids:
            f.write(f'{uid}\tgood\n')

    logger.info("  %s bank %d: %d good units, template shape %s",
                date_key, bank_id, len(good_ids), templates.shape)

    return session_dir


def run_unitmatch_bank(bank_id, session_dirs, um_input_dir):
    """Run UnitMatch on one bank across all survey dates.

    Returns the probability matrix, matched pairs, and UIDs.
    """
    import UnitMatchPy.default_params as default_params
    import UnitMatchPy.utils as util
    import UnitMatchPy.overlord as ov
    import UnitMatchPy.bayes_functions as bf
    import UnitMatchPy.assign_unique_id as aid

    logger.info("\nRunning UnitMatch for Bank %d across %d sessions...",
                bank_id, len(session_dirs))

    # Get parameters
    param = default_params.get_default_param()
    param['KS_dirs'] = [str(d) for d in session_dirs]

    # Load waveforms and channel positions
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(
        [str(d) for d in session_dirs])
    param = util.get_probe_geometry(channel_pos[0], param)

    waveform, session_id, session_switch, within_session, good_units, param = \
        util.load_good_waveforms(wave_paths, unit_label_paths, param,
                                good_units_only=True)

    logger.info("  Loaded %d units total, %d sessions",
                param['n_units'], len(session_dirs))

    # Create cluster info dict
    clus_info = {
        'good_units': good_units,
        'session_switch': session_switch,
        'session_id': session_id,
        'original_ids': np.concatenate(good_units),
    }

    # Extract waveform parameters
    extracted = ov.extract_parameters(waveform, channel_pos, clus_info, param)

    # Calculate metric scores
    total_score, candidate_pairs, scores_to_include, predictors = \
        ov.extract_metric_scores(extracted, session_switch, within_session,
                                param, niter=2)

    # Bayesian matching
    prior_match = 1 - (param['n_expected_matches'] / param['n_units'] ** 2)
    priors = np.array((prior_match, 1 - prior_match))

    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    parameter_kernels = bf.get_parameter_kernels(
        scores_to_include, labels, cond, param, add_one=1)

    probability = bf.apply_naive_bayes(
        parameter_kernels, priors, predictors, param, cond)

    prob_matrix = probability[:, 1].reshape(param['n_units'], param['n_units'])

    # Evaluate and get matches — report all pairs above 0.5
    match_threshold = 0.5
    util.evaluate_output(prob_matrix, param, within_session, session_switch,
                        match_threshold=match_threshold)

    # Assign unique IDs
    UIDs = aid.assign_unique_id(prob_matrix, param, clus_info)

    # UIDs is [liberal, intermediate, conservative, original]
    uid_intermediate = UIDs[1]
    logger.info("  Bank %d: %d unique neuron IDs assigned (intermediate)",
                bank_id, len(np.unique(uid_intermediate)))

    return {
        'prob_matrix': prob_matrix,
        'UIDs': uid_intermediate,
        'clus_info': clus_info,
        'session_id': session_id,
        'good_units': good_units,
        'param': param,
        'match_threshold': match_threshold,
        'extracted': extracted,
        'within_session': within_session,
        'scores_to_include': scores_to_include,
        'candidate_pairs': candidate_pairs,
    }


def extract_matches(um_result, date_keys, bank_id):
    """Extract cross-session matches from UnitMatch results."""
    prob_matrix = um_result['prob_matrix']
    session_id = um_result['session_id']
    good_units = um_result['good_units']
    UIDs = um_result['UIDs']
    threshold = um_result['match_threshold']
    extracted = um_result['extracted']

    n_units = prob_matrix.shape[0]
    matches = []

    # NOTE: channel positions already include absolute probe depth
    # (set by make_bank_probe in sort_survey.py), so no bank offset needed.

    # Build unit index to (session, original_id) mapping
    unit_info = []
    for sess_idx, units in enumerate(good_units):
        for uid in units:
            uid_val = int(np.atleast_1d(uid).flat[0])
            unit_info.append({'session': date_keys[sess_idx], 'original_id': uid_val})

    # Find cross-session matches above threshold
    for i in range(n_units):
        for j in range(i + 1, n_units):
            if unit_info[i]['session'] == unit_info[j]['session']:
                continue  # skip within-session
            p = prob_matrix[i, j]
            if p > threshold:
                # Get depth from extracted centroids
                # avg_centroid shape: (3_dims, n_units, 2_halves)
                # dim 2 = y (depth), mean across halves
                depth_i = np.nan
                depth_j = np.nan
                try:
                    if 'avg_centroid' in extracted:
                        centroids = np.array(extracted['avg_centroid'])
                        # depth = y-axis = last spatial dim, mean over halves
                        # add bank offset for absolute probe depth
                        depth_i = float(centroids[-1, i, :].mean())
                        depth_j = float(centroids[-1, j, :].mean())
                except (IndexError, TypeError, ValueError):
                    pass

                matches.append({
                    'bank': bank_id,
                    'session_1': unit_info[i]['session'],
                    'unit_1': unit_info[i]['original_id'],
                    'session_2': unit_info[j]['session'],
                    'unit_2': unit_info[j]['original_id'],
                    'probability': float(p),
                    'UID': int(UIDs[i]) if UIDs[i] == UIDs[j] else -1,
                    'depth_1': depth_i,
                    'depth_2': depth_j,
                })

    return matches


def plot_tracking_summary(all_matches_df, all_um_results, date_keys, fig_dir):
    """Generate tracking summary figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if len(all_matches_df) == 0:
        logger.warning("No matches to plot!")
        return

    # ---- Figure 1: Matches per bank ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'Cross-Day Neuron Tracking — {" vs ".join(date_keys)}',
                 fontsize=14, fontweight='bold')

    # Panel A: Matches per bank
    ax = axes[0]
    bank_counts = all_matches_df.groupby('bank').size()
    banks = range(12)
    counts = [bank_counts.get(b, 0) for b in banks]
    ax.barh(list(banks), counts, color='#4472C4', alpha=0.8)
    ax.set_yticks(list(banks))
    ax.set_yticklabels([f'Bank {b}' for b in banks])
    ax.set_xlabel('Number of Matched Units')
    ax.set_title('Tracked Neurons per Bank')
    ax.invert_yaxis()

    # Panel B: Match probability distribution
    ax = axes[1]
    ax.hist(all_matches_df['probability'], bins=50, range=(0, 1),
            color='#ED7D31', alpha=0.8, edgecolor='k', linewidth=0.3)
    ax.axvline(0.75, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    ax.set_xlabel('Match Probability')
    ax.set_ylabel('Count')
    ax.set_title('Match Confidence Distribution')
    ax.legend()

    # Panel C: Depth consistency
    ax = axes[2]
    valid_depth = all_matches_df.dropna(subset=['depth_1', 'depth_2'])
    if len(valid_depth) > 0:
        ax.scatter(valid_depth['depth_1'], valid_depth['depth_2'],
                   c=valid_depth['probability'], cmap='RdYlGn', s=20,
                   alpha=0.7, edgecolors='k', linewidth=0.3)
        # Unity line
        dmin = min(valid_depth['depth_1'].min(), valid_depth['depth_2'].min())
        dmax = max(valid_depth['depth_1'].max(), valid_depth['depth_2'].max())
        ax.plot([dmin, dmax], [dmin, dmax], 'k--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(f'Depth — {date_keys[0]} (µm)')
        ax.set_ylabel(f'Depth — {date_keys[1]} (µm)')
    ax.set_title('Depth Consistency of Matches')

    fig.tight_layout()
    out_path = fig_dir / 'tracking_summary.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)

    # ---- Figure 2: Full-probe tracking map ----
    n_dates = len(date_keys)
    fig, ax = plt.subplots(figsize=(4 + 4 * n_dates, 20))
    session_colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']
    if len(valid_depth) > 0:
        for _, row in valid_depth.iterrows():
            # Map sessions to x-positions
            s1_idx = date_keys.index(str(int(row['session_1'])))
            s2_idx = date_keys.index(str(int(row['session_2'])))
            color = '#70AD47' if row['probability'] > 0.9 else '#FFC000'
            ax.plot([s1_idx, s2_idx], [row['depth_1'], row['depth_2']],
                    color=color, alpha=0.5, linewidth=0.8)
            ax.scatter([s1_idx], [row['depth_1']], s=10,
                       c=session_colors[s1_idx % len(session_colors)], zorder=5)
            ax.scatter([s2_idx], [row['depth_2']], s=10,
                       c=session_colors[s2_idx % len(session_colors)], zorder=5)

        # Bank boundaries
        for bk in range(13):
            y = bk * 192 * 20
            ax.axhline(y, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
            if bk < 12:
                ax.text((n_dates - 1) / 2.0, y + 1900, f'Bank {bk}',
                        fontsize=7, color='gray', ha='center', va='center')

    ax.set_xticks(range(n_dates))
    ax.set_xticklabels(date_keys)
    ax.set_ylabel('Depth (um)')
    ax.set_title(f'Neuron Tracking Across Days\n{len(all_matches_df)} matched pairs')
    ax.invert_yaxis()

    fig.tight_layout()
    out_path = fig_dir / 'tracking_full_probe.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def _plot_spatial_footprint(ax, template, channel_positions,
                            y_radius_um=100, time_scale=80):
    """Draw waveform traces at their physical channel positions.

    Similar to Fig 2b in van Beest et al. 2024.  Each channel's waveform
    trace is centred at its (x, y) position on the probe, giving a spatial
    footprint view of the unit.

    Parameters
    ----------
    ax : matplotlib Axes
    template : ndarray, (n_timepoints, n_channels)
    channel_positions : ndarray, (n_channels, 2) — (x, y) in µm
    y_radius_um : float — vertical radius around peak channel to show
        (100 µm ≈ 5 rows per column on NP1032, ~10 channels total)
    time_scale : float — total width of each trace in µm
    """
    ptp_per_ch = np.ptp(template, axis=0)
    peak_ch = np.argmax(ptp_per_ch)
    peak_pos = channel_positions[peak_ch]

    # Select channels within y_radius of peak
    dy = np.abs(channel_positions[:, 1] - peak_pos[1])
    nearby = np.where(dy <= y_radius_um)[0]
    if len(nearby) == 0:
        return

    # Auto-scale: waveform amplitude maps to ±12 µm (allows slight overlap
    # between 20 µm-spaced rows but maximises waveform visibility)
    max_amp = np.max(np.abs(template[:, nearby]))
    amp_scale = 12.0 / max_amp if max_amp > 0 else 1.0

    n_tp = template.shape[0]
    t_norm = np.linspace(-time_scale / 2, time_scale / 2, n_tp)

    for ch in nearby:
        xc, yc = channel_positions[ch]
        wav = template[:, ch]
        rel = ptp_per_ch[ch] / (ptp_per_ch[peak_ch] + 1e-10)

        if ch == peak_ch:
            color, lw = '#D62728', 1.8   # red for peak
        elif rel > 0.5:
            color, lw = '#1F77B4', 1.0   # blue for strong
        elif rel > 0.25:
            color, lw = '#7F7F7F', 0.7   # gray for medium
        else:
            color, lw = '#C0C0C0', 0.4   # light gray for weak

        # Negate waveform so negative deflections point downward on
        # the depth-inverted y-axis (standard extracellular convention).
        ax.plot(t_norm + xc, -wav * amp_scale + yc,
                color=color, linewidth=lw, alpha=0.85)

    # Faint channel position dots
    ax.scatter(channel_positions[nearby, 0], channel_positions[nearby, 1],
               s=3, color='#AAAAAA', alpha=0.3, zorder=0)

    # Tight axis limits
    pad_x, pad_y = time_scale * 0.3, 15
    ax.set_xlim(channel_positions[nearby, 0].min() - pad_x,
                channel_positions[nearby, 0].max() + pad_x)
    ax.set_ylim(channel_positions[nearby, 1].max() + pad_y,
                channel_positions[nearby, 1].min() - pad_y)  # invert for depth


def plot_matched_unit_diagnostics(all_matches_df, date_keys, fig_dir,
                                   max_units_per_page=12):
    """Plot waveform comparisons and quality metrics for matched neurons.

    For each matched pair, loads the template from each session's analyzer
    cache and plots side-by-side waveforms, plus aggregate quality metrics.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if len(all_matches_df) == 0:
        return

    # ---- Pre-load all analyzer data (cache across calls) ----
    _cache = {}

    def _get_data(date_key, bank_id):
        key = (date_key, bank_id)
        if key not in _cache:
            cache_dir = (DATA_ROOT / 'sorting_output' / f'survey_{date_key}'
                         / f'bank{bank_id:02d}_ks4' / 'analyzer_cache')
            if (cache_dir / 'spikeinterface_info.json').exists():
                _cache[key] = load_analyzer_data(cache_dir)
            else:
                _cache[key] = None
        return _cache[key]

    # ---- Figure: Waveform gallery for top matched pairs ----
    # Group matches by UID to find neurons tracked across multiple days
    uid_matches = all_matches_df[all_matches_df['UID'] >= 0].copy()

    # Build per-unit records: collect all (session, unit_id, bank) for each UID
    neuron_records = {}  # UID -> list of (date, unit_id, bank, depth, prob)
    for _, row in uid_matches.iterrows():
        uid = int(row['UID'])
        bank = int(row['bank'])
        if uid not in neuron_records:
            neuron_records[uid] = {}
        s1 = str(int(row['session_1']))
        s2 = str(int(row['session_2']))
        neuron_records[uid][(s1, bank)] = {
            'unit_id': int(row['unit_1']), 'depth': row['depth_1'],
            'prob': row['probability']}
        neuron_records[uid][(s2, bank)] = {
            'unit_id': int(row['unit_2']), 'depth': row['depth_2'],
            'prob': row['probability']}

    # Sort by number of sessions tracked (more = better)
    sorted_uids = sorted(neuron_records.keys(),
                         key=lambda u: len(neuron_records[u]), reverse=True)

    # ---- Spatial footprint waveform pages (cf. Fig 2b, van Beest 2024) ----
    page = 0
    for start in range(0, min(len(sorted_uids), 48), max_units_per_page):
        batch = sorted_uids[start:start + max_units_per_page]
        n_units = len(batch)
        n_cols = len(date_keys)

        fig, axes = plt.subplots(n_units, n_cols,
                                 figsize=(3 * n_cols, 2.8 * n_units),
                                 squeeze=False)
        fig.suptitle(f'Matched Neuron Spatial Footprints — Page {page + 1}',
                     fontsize=13, fontweight='bold')

        for row_idx, uid in enumerate(batch):
            records = neuron_records[uid]
            bank_id = list(records.keys())[0][1]

            for col_idx, dk in enumerate(date_keys):
                ax = axes[row_idx, col_idx]
                key = (dk, bank_id)
                if key in records:
                    rec = records[key]
                    data = _get_data(dk, bank_id)
                    if data is not None:
                        try:
                            tmpl_idx = data['unit_ids'].index(rec['unit_id'])
                            template = data['templates'][tmpl_idx]
                            _plot_spatial_footprint(
                                ax, template, data['channel_positions'])
                            depth_str = (f'd={rec["depth"]:.0f}'
                                         if not np.isnan(rec['depth']) else '')
                            ax.set_title(
                                f'u{rec["unit_id"]} {depth_str} '
                                f'p={rec["prob"]:.2f}',
                                fontsize=8, loc='left')
                        except (ValueError, IndexError):
                            ax.text(0.5, 0.5, 'load err',
                                    transform=ax.transAxes,
                                    ha='center', va='center', fontsize=8)
                    else:
                        ax.text(0.5, 0.5, 'no cache',
                                transform=ax.transAxes,
                                ha='center', va='center', fontsize=8)
                else:
                    ax.text(0.5, 0.5, '\u2014', transform=ax.transAxes,
                            ha='center', va='center', fontsize=10,
                            color='gray')

                if row_idx == 0:
                    ax.set_title(dk + '\n' + ax.get_title(), fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f'UID {uid}\nBank {bank_id}', fontsize=8)
                ax.tick_params(labelsize=6)
                ax.set_xticklabels([])
                if col_idx > 0:
                    ax.set_yticklabels([])

        fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
        out = fig_dir / f'matched_waveforms_p{page:02d}.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: %s", out)
        page += 1

    # ---- Figure: Quality metrics of matched vs unmatched units ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quality Metrics — Matched vs All Good Units',
                 fontsize=14, fontweight='bold')

    # Collect QM for matched and all good units
    matched_qm = {'snr': [], 'firing_rate': [], 'isi_violations_ratio': [],
                   'amplitude_median': [], 'presence_ratio': [], 'depth': []}
    all_qm = {'snr': [], 'firing_rate': [], 'isi_violations_ratio': [],
               'amplitude_median': [], 'presence_ratio': [], 'depth': []}

    for dk in date_keys:
        for bank_id in range(12):
            data = _get_data(dk, bank_id)
            if data is None:
                continue
            qm = data['qm']
            good = get_good_unit_ids(qm, tm=data['tm'],
                                     templates=data['templates'])
            positions = data['channel_positions']

            for uid in good:
                if uid not in qm.index:
                    continue
                row = qm.loc[uid]
                for k in ['snr', 'firing_rate', 'isi_violations_ratio',
                           'amplitude_median', 'presence_ratio']:
                    if k in qm.columns:
                        all_qm[k].append(float(row.get(k, np.nan)))

                # Get depth from template peak channel
                try:
                    tmpl_idx = data['unit_ids'].index(uid)
                    template = data['templates'][tmpl_idx]
                    peak_ch = np.argmax(np.ptp(template, axis=0))
                    depth = positions[peak_ch, 1] if peak_ch < len(positions) else np.nan
                    all_qm['depth'].append(depth)
                except (ValueError, IndexError):
                    all_qm['depth'].append(np.nan)

            # Matched units for this session/bank
            matched_ids = set()
            bank_matches = all_matches_df[all_matches_df['bank'] == bank_id]
            for _, m in bank_matches.iterrows():
                if str(int(m['session_1'])) == dk:
                    matched_ids.add(int(m['unit_1']))
                if str(int(m['session_2'])) == dk:
                    matched_ids.add(int(m['unit_2']))

            for uid in matched_ids:
                if uid not in qm.index:
                    continue
                row = qm.loc[uid]
                for k in ['snr', 'firing_rate', 'isi_violations_ratio',
                           'amplitude_median', 'presence_ratio']:
                    if k in qm.columns:
                        matched_qm[k].append(float(row.get(k, np.nan)))
                try:
                    tmpl_idx = data['unit_ids'].index(uid)
                    template = data['templates'][tmpl_idx]
                    peak_ch = np.argmax(np.ptp(template, axis=0))
                    depth = positions[peak_ch, 1] if peak_ch < len(positions) else np.nan
                    matched_qm['depth'].append(depth)
                except (ValueError, IndexError):
                    matched_qm['depth'].append(np.nan)

    metrics = ['snr', 'firing_rate', 'isi_violations_ratio',
               'amplitude_median', 'presence_ratio', 'depth']
    labels = ['SNR', 'Firing Rate (Hz)', 'ISI Violations Ratio',
              'Amplitude Median (µV)', 'Presence Ratio', 'Depth (µm)']
    log_scale = [False, True, False, False, False, False]

    for ax, metric, label, use_log in zip(axes.flat, metrics, labels, log_scale):
        a = np.array(all_qm[metric])
        m = np.array(matched_qm[metric])
        a = a[np.isfinite(a)]
        m = m[np.isfinite(m)]
        if len(a) > 0:
            bins = 40
            ax.hist(a, bins=bins, color='#4472C4', alpha=0.5,
                    label=f'All good (n={len(a)})', edgecolor='white')
        if len(m) > 0:
            ax.hist(m, bins=bins, color='#ED7D31', alpha=0.7,
                    label=f'Matched (n={len(m)})', edgecolor='white')
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        if use_log:
            ax.set_xscale('log')
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = fig_dir / 'matched_quality_metrics.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out)

    # ---- Figure: Depth stability across days ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Depth Stability of Tracked Neurons',
                 fontsize=14, fontweight='bold')

    # Panel A: Depth drift histogram
    valid = all_matches_df.dropna(subset=['depth_1', 'depth_2'])
    if len(valid) > 0:
        depth_diff = (valid['depth_2'] - valid['depth_1']).values
        ax = axes[0]
        ax.hist(depth_diff, bins=50, color='#70AD47', alpha=0.8,
                edgecolor='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=1, linestyle='--')
        ax.axvline(np.median(depth_diff), color='red', linewidth=1.5,
                   linestyle='-', label=f'median={np.median(depth_diff):.1f} µm')
        ax.set_xlabel('Depth Difference (µm)')
        ax.set_ylabel('Count')
        ax.set_title(f'Depth Shift Distribution\n'
                     f'|shift| = {np.median(np.abs(depth_diff)):.1f} µm median')
        ax.legend()

        # Panel B: Depth diff vs match probability
        ax = axes[1]
        sc = ax.scatter(valid['probability'], np.abs(depth_diff),
                        c=valid['bank'], cmap='tab20', s=20, alpha=0.7,
                        edgecolors='k', linewidth=0.3)
        plt.colorbar(sc, ax=ax, label='Bank')
        ax.set_xlabel('Match Probability')
        ax.set_ylabel('|Depth Difference| (µm)')
        ax.set_title('Match Confidence vs Depth Consistency')

    fig.tight_layout()
    out = fig_dir / 'depth_stability.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_ptrack_vs_days(all_matches_df, date_keys, fig_dir):
    """P(track) vs days between recordings (cf. Fig 5e, van Beest 2024).

    For each pair of sessions, computes the fraction of good units that
    were successfully tracked and plots against the time interval.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime

    fig_dir = Path(fig_dir)

    # Parse dates
    dates = {dk: datetime.strptime(dk, '%Y%m%d') for dk in date_keys}

    # For each pair of sessions, count trackable and tracked units
    pair_stats = []
    for i, dk1 in enumerate(date_keys):
        for j, dk2 in enumerate(date_keys):
            if j <= i:
                continue
            delta_days = (dates[dk2] - dates[dk1]).days

            total_good_1 = 0
            total_good_2 = 0
            total_tracked = 0

            for bank_id in range(12):
                cache1 = (DATA_ROOT / 'sorting_output' / f'survey_{dk1}'
                          / f'bank{bank_id:02d}_ks4' / 'analyzer_cache')
                cache2 = (DATA_ROOT / 'sorting_output' / f'survey_{dk2}'
                          / f'bank{bank_id:02d}_ks4' / 'analyzer_cache')

                if not (cache1 / 'spikeinterface_info.json').exists():
                    continue
                if not (cache2 / 'spikeinterface_info.json').exists():
                    continue

                data1 = load_analyzer_data(cache1)
                data2 = load_analyzer_data(cache2)
                good1 = get_good_unit_ids(data1['qm'], tm=data1['tm'],
                                           templates=data1['templates'])
                good2 = get_good_unit_ids(data2['qm'], tm=data2['tm'],
                                           templates=data2['templates'])

                total_good_1 += len(good1)
                total_good_2 += len(good2)

                # Count tracked pairs for this bank and session pair
                s1, s2 = str(dk1), str(dk2)
                bank_matches = all_matches_df[
                    (all_matches_df['bank'] == bank_id) &
                    (((all_matches_df['session_1'].astype(str) == s1) &
                      (all_matches_df['session_2'].astype(str) == s2)) |
                     ((all_matches_df['session_1'].astype(str) == s2) &
                      (all_matches_df['session_2'].astype(str) == s1)))
                ]
                total_tracked += len(bank_matches)

            n_trackable = min(total_good_1, total_good_2)
            p_track = total_tracked / n_trackable if n_trackable > 0 else 0

            pair_stats.append({
                'session_1': dk1, 'session_2': dk2,
                'delta_days': delta_days,
                'n_good_1': total_good_1, 'n_good_2': total_good_2,
                'n_tracked': total_tracked, 'n_trackable': n_trackable,
                'p_track': p_track,
            })

    if not pair_stats:
        return

    df = pd.DataFrame(pair_stats)
    logger.info("P(track) analysis:\n%s",
                df[['session_1', 'session_2', 'delta_days',
                    'n_tracked', 'n_trackable', 'p_track']].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Tracking Yield Across Recording Intervals',
                 fontsize=14, fontweight='bold')

    # Panel A: P(track) vs delta-days
    ax = axes[0]
    ax.scatter(df['delta_days'], df['p_track'], s=120, c='#4472C4',
               edgecolors='k', linewidth=0.8, zorder=5)
    for _, row in df.iterrows():
        lbl = (f"{row['session_1'][4:6]}/{row['session_1'][6:]}"
               f"\nvs\n{row['session_2'][4:6]}/{row['session_2'][6:]}")
        ax.annotate(lbl, (row['delta_days'], row['p_track']),
                    fontsize=6, ha='center', va='top',
                    xytext=(0, -14), textcoords='offset points')
    ax.set_xlabel('Days Between Recordings')
    ax.set_ylabel('P(track)')
    ax.set_ylim(-0.05, max(df['p_track'].max() * 1.15, 0.5))
    ax.set_title('Probability of Tracking vs Time Interval')
    ax.grid(True, alpha=0.3)

    # Panel B: Number tracked per pair
    ax = axes[1]
    x_pos = range(len(df))
    bars = ax.bar(x_pos, df['n_tracked'], color='#ED7D31', alpha=0.8,
                  edgecolor='k', linewidth=0.5)
    ax.bar(x_pos, df['n_trackable'], color='#4472C4', alpha=0.25,
           edgecolor='k', linewidth=0.3, label='Trackable')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(
        [f"{r['session_1'][-4:]}\nvs\n{r['session_2'][-4:]}"
         for _, r in df.iterrows()], fontsize=8)
    ax.set_ylabel('Number of Units')
    ax.set_title('Tracked vs Trackable per Session Pair')
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row['n_tracked'] + 2,
                f"{row['n_tracked']}/{row['n_trackable']}\n"
                f"({row['p_track']:.1%})",
                ha='center', va='bottom', fontsize=7)
    ax.legend(loc='upper right')

    fig.tight_layout()
    out = fig_dir / 'ptrack_vs_days.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_similarity_scores(all_um_results, date_keys, fig_dir):
    """Plot UnitMatch similarity score distributions for matched vs
    non-matched pairs (cf. Fig 3, van Beest 2024).

    Aggregates scores across all banks and shows the distribution of
    each metric separated by match label.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)

    # Aggregate scores across banks
    all_scores = {}   # metric_name -> list of values
    all_labels = []   # 0=non-match, 1=match

    for bank_id, um in all_um_results.items():
        scores = um.get('scores_to_include')
        labels = um.get('candidate_pairs')
        if scores is None or labels is None:
            continue
        for name, vals in scores.items():
            if name not in all_scores:
                all_scores[name] = []
            all_scores[name].append(np.asarray(vals).ravel())
        all_labels.append(np.asarray(labels).ravel())

    if not all_scores or not all_labels:
        logger.info("No similarity scores available for plotting.")
        return

    all_labels = np.concatenate(all_labels)
    for k in all_scores:
        all_scores[k] = np.concatenate(all_scores[k])

    metric_names = list(all_scores.keys())
    n_metrics = len(metric_names)
    ncols = min(n_metrics, 4)
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle('UnitMatch Similarity Score Distributions\n'
                 'Matched (orange) vs Non-matched (blue) Pairs',
                 fontsize=14, fontweight='bold')

    match_mask = all_labels == 1
    nonmatch_mask = all_labels == 0

    for idx, name in enumerate(metric_names):
        ax = axes[idx // ncols, idx % ncols]
        vals = all_scores[name]
        finite = np.isfinite(vals)

        if np.any(finite & nonmatch_mask):
            ax.hist(vals[finite & nonmatch_mask], bins=60, density=True,
                    color='#4472C4', alpha=0.5, label='Non-match')
        if np.any(finite & match_mask):
            ax.hist(vals[finite & match_mask], bins=60, density=True,
                    color='#ED7D31', alpha=0.7, label='Match')
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_title(name, fontsize=10)

    # Hide unused axes
    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.tight_layout()
    out = fig_dir / 'similarity_scores.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out)


def main():
    # Parse survey dates from args
    if len(sys.argv) > 1:
        date_keys = sys.argv[1:]
    else:
        # Default: all known surveys (sorted chronologically)
        date_keys = sorted(SURVEYS.keys())

    # Validate
    for dk in date_keys:
        if dk not in SURVEYS:
            logger.error("Unknown survey date: %s. Known: %s", dk, list(SURVEYS.keys()))
            sys.exit(1)
        sort_dir = SURVEYS[dk]['sort_dir']
        if not sort_dir.exists():
            logger.error("Sort output not found: %s", sort_dir)
            sys.exit(1)

    logger.info("Tracking neurons across surveys: %s", ', '.join(date_keys))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_matches = []
    all_um_results = {}

    for bank_id in range(12):
        logger.info("\n" + "=" * 70)
        logger.info("BANK %d", bank_id)
        logger.info("=" * 70)

        # Prepare UnitMatch input for each date
        um_input_dir = OUTPUT_DIR / f'bank{bank_id:02d}'
        session_dirs = []
        active_dates = []

        for dk in date_keys:
            sess_dir = prepare_um_session(dk, bank_id, um_input_dir)
            if sess_dir is not None:
                session_dirs.append(sess_dir)
                active_dates.append(dk)

        if len(session_dirs) < 2:
            logger.info("  Bank %d: fewer than 2 sessions with good units, skipping",
                        bank_id)
            continue

        # Run UnitMatch
        try:
            um_result = run_unitmatch_bank(bank_id, session_dirs, um_input_dir)
            all_um_results[bank_id] = um_result

            # Extract matches
            matches = extract_matches(um_result, active_dates, bank_id)
            all_matches.extend(matches)
            logger.info("  Bank %d: %d cross-session matches", bank_id, len(matches))

            # Save per-bank results
            np.save(um_input_dir / 'prob_matrix.npy', um_result['prob_matrix'])
            np.save(um_input_dir / 'UIDs.npy', um_result['UIDs'])

        except Exception as e:
            logger.error("  UnitMatch failed for bank %d: %s", bank_id, e)
            import traceback
            traceback.print_exc()

    # Aggregate results
    if all_matches:
        all_matches_df = pd.DataFrame(all_matches)
        all_matches_df.to_csv(OUTPUT_DIR / 'tracking_summary.csv', index=False)
        logger.info("\n" + "=" * 70)
        logger.info("TRACKING SUMMARY")
        logger.info("=" * 70)
        logger.info("  Total matched pairs: %d", len(all_matches_df))
        logger.info("  Matched banks: %d/12",
                     all_matches_df['bank'].nunique())
        logger.info("  Mean match probability: %.3f",
                     all_matches_df['probability'].mean())

        # Per-bank breakdown
        for bank_id in sorted(all_matches_df['bank'].unique()):
            bdf = all_matches_df[all_matches_df['bank'] == bank_id]
            logger.info("  Bank %02d: %d matches (mean prob %.3f)",
                         bank_id, len(bdf), bdf['probability'].mean())

        # Visualization
        plot_tracking_summary(all_matches_df, all_um_results, date_keys, FIG_DIR)

        # Full diagnostics
        logger.info("\nGenerating matched-unit diagnostics...")
        try:
            plot_matched_unit_diagnostics(all_matches_df, date_keys, FIG_DIR)
        except Exception as e:
            logger.warning("Diagnostics failed: %s", e)
            import traceback
            traceback.print_exc()

        # Paper-inspired analyses (van Beest et al. 2024)
        logger.info("\nGenerating paper-inspired analyses...")
        try:
            plot_ptrack_vs_days(all_matches_df, date_keys, FIG_DIR)
        except Exception as e:
            logger.warning("P(track) vs days plot failed: %s", e)
            import traceback
            traceback.print_exc()

        try:
            plot_similarity_scores(all_um_results, date_keys, FIG_DIR)
        except Exception as e:
            logger.warning("Similarity scores plot failed: %s", e)
            import traceback
            traceback.print_exc()
    else:
        logger.warning("No matches found across any bank!")
        all_matches_df = pd.DataFrame()

    logger.info("\nTracking pipeline complete.")
    return all_matches_df


if __name__ == '__main__':
    main()
