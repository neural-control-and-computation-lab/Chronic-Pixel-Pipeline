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
    """Load templates, quality metrics, unit IDs, and channel positions
    directly from the analyzer cache files (no SI recording required).

    Returns dict with keys: templates, unit_ids, channel_positions, qm
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

    return {
        'templates': templates,
        'unit_ids': unit_ids,
        'channel_positions': channel_positions,
        'qm': qm,
    }


def get_good_unit_ids(qm, fr_thresh=0.1, isi_thresh=0.5):
    """Classify good units from quality metrics DataFrame."""
    snr = qm['snr'].dropna()
    if len(snr) == 0:
        return []
    snr_thresh = snr.quantile(0.75) if snr.median() < 5 else 3.0
    mask = (
        (qm['snr'] > snr_thresh) &
        (qm['firing_rate'] > fr_thresh) &
        (qm['isi_violations_ratio'] < isi_thresh)
    )
    if 'presence_ratio' in qm.columns:
        pr = qm['presence_ratio'].dropna()
        if len(pr) > 0:
            mask = mask & (qm['presence_ratio'] > 0.5)
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
    good_ids = get_good_unit_ids(data['qm'])
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

    # Bank depth offset: each bank covers 192 electrodes at 20 µm pitch
    bank_depth_offset = bank_id * 192 * 20  # µm

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
                        depth_i = float(centroids[-1, i, :].mean()) + bank_depth_offset
                        depth_j = float(centroids[-1, j, :].mean()) + bank_depth_offset
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
    else:
        logger.warning("No matches found across any bank!")
        all_matches_df = pd.DataFrame()

    logger.info("\nTracking pipeline complete.")
    return all_matches_df


if __name__ == '__main__':
    main()
