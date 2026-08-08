"""
Robust neural-IMU decoding analysis for wireless recordings.

Three complementary analyses per bank:
  1. Movement vs Rest — detect movement bouts, compare firing rates
  2. Temporal Prediction — train Ridge on 80%, predict held-out 20%
  3. Single-neuron scatters — FR vs best IMU channel with regression line

Usage:
  python decode_neural_imu.py E:/March31_wireless
  python decode_neural_imu.py E:/March23_wireless --bank 0
"""
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import sys
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from analyze_neural_imu import (
    load_per_segment_imu, convert_imu_to_physical, compute_derived_imu,
    bin_imu_signals, bin_spike_trains,
)

BANK_OFFSET_UM = 192 * 20


# ── Analysis 1: Movement vs Rest ────────────────────────────────────

def detect_movement_bouts(imu_derived, threshold_percentile=75, min_bout_s=0.5):
    """Detect movement bouts from gyro magnitude.

    Returns binary movement mask at IMU rate (2500 Hz) and bout boundaries.
    """
    fs = imu_derived['sampling_rate']
    names = imu_derived['names']
    signals = imu_derived['signals']

    # Use gyro magnitude
    gyro_mag_idx = names.index('gyro_mag')
    gyro_mag = signals[:, gyro_mag_idx]

    # Smooth with 200ms Gaussian
    sigma = int(0.2 * fs)
    gyro_smooth = gaussian_filter1d(gyro_mag, sigma=sigma)

    # Threshold
    thresh = np.percentile(gyro_smooth, threshold_percentile)
    is_moving = gyro_smooth > thresh

    # Enforce minimum bout duration
    min_samples = int(min_bout_s * fs)
    # Remove short bouts
    changes = np.diff(is_moving.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if is_moving[0]:
        starts = np.concatenate([[0], starts])
    if is_moving[-1]:
        ends = np.concatenate([ends, [len(is_moving)]])

    for s, e in zip(starts, ends):
        if e - s < min_samples:
            is_moving[s:e] = False

    return is_moving, gyro_smooth, thresh


def movement_vs_rest_analysis(cache_dir, good_unit_ids, imu_derived,
                               bank_id, bin_ms=500):
    """Compare firing rates during movement vs rest for each unit.

    Uses 500ms bins for robust rate estimates.
    """
    fs_imu = imu_derived['sampling_rate']
    duration_s = len(imu_derived['signals']) / fs_imu

    # Detect movement
    is_moving, gyro_smooth, thresh = detect_movement_bouts(imu_derived)

    # Trim first 5s
    trim = int(5 * fs_imu)
    is_moving = is_moving[trim:]

    # Bin movement into same bins as firing rates
    bin_samples = int(bin_ms / 1000 * fs_imu)
    n_bins = (len(is_moving)) // bin_samples
    is_moving_trimmed = is_moving[:n_bins * bin_samples]
    move_binned = is_moving_trimmed.reshape(n_bins, bin_samples).mean(axis=1)
    move_mask = move_binned > 0.5  # bin is "movement" if >50% of samples are moving

    # Get firing rates
    fr_binned, unit_ids_used = bin_spike_trains(
        cache_dir, good_unit_ids, bin_ms, bin_ms * 2, duration_s - 5)
    # Trim to match
    n = min(len(move_mask), fr_binned.shape[0])
    move_mask = move_mask[:n]
    fr_binned = fr_binned[:n]

    n_move = move_mask.sum()
    n_rest = (~move_mask).sum()

    results = []
    for i, uid in enumerate(unit_ids_used):
        fr_move = fr_binned[move_mask, i]
        fr_rest = fr_binned[~move_mask, i]

        if len(fr_move) < 3 or len(fr_rest) < 3:
            continue

        # Wilcoxon rank-sum test
        stat, p_val = stats.mannwhitneyu(fr_move, fr_rest, alternative='two-sided')

        results.append({
            'unit_id': uid,
            'bank': bank_id,
            'fr_move': float(np.mean(fr_move)),
            'fr_rest': float(np.mean(fr_rest)),
            'fr_ratio': float(np.mean(fr_move) / max(np.mean(fr_rest), 0.01)),
            'p_value': float(p_val),
            'n_move_bins': int(n_move),
            'n_rest_bins': int(n_rest),
        })

    return pd.DataFrame(results)


# ── Analysis 1b: Binary movement classification ─────────────────────

def binary_movement_classification(cache_dir, good_unit_ids, imu_derived,
                                    bank_id, bin_ms=500, n_folds=5,
                                    n_shuffles=200):
    """Classify moving vs not-moving from neural population.

    Uses logistic regression with 5-fold stratified CV. Includes
    permutation test: shuffle movement labels, re-run CV, repeat
    n_shuffles times to build null distribution of accuracy.

    Returns dict with accuracy, shuffle distribution, fold predictions.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    fs_imu = imu_derived['sampling_rate']
    duration_s = len(imu_derived['signals']) / fs_imu

    # Detect movement
    is_moving, gyro_smooth, thresh = detect_movement_bouts(imu_derived)

    # Trim first 5s
    trim = int(5 * fs_imu)
    is_moving = is_moving[trim:]

    # Bin movement labels
    bin_samples = int(bin_ms / 1000 * fs_imu)
    n_bins = len(is_moving) // bin_samples
    move_binned = is_moving[:n_bins * bin_samples].reshape(n_bins, bin_samples).mean(axis=1)
    y = (move_binned > 0.5).astype(int)

    # Get firing rates
    fr_binned, unit_ids_used = bin_spike_trains(
        cache_dir, good_unit_ids, bin_ms, bin_ms * 2, duration_s - 5)
    n = min(len(y), fr_binned.shape[0])
    y = y[:n]
    X = fr_binned[:n]

    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return None

    # Standardize
    scaler = StandardScaler()

    # Cross-validated classification
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    y_pred_all = np.zeros_like(y)
    y_prob_all = np.zeros(len(y))
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegressionCV(Cs=10, cv=3, max_iter=1000,
                                    solver='lbfgs', random_state=42,
                                    class_weight='balanced')
        clf.fit(X_train, y[train_idx])
        y_pred_all[test_idx] = clf.predict(X_test)
        y_prob_all[test_idx] = clf.predict_proba(X_test)[:, 1]
        fold_accs.append(clf.score(X_test, y[test_idx]))

    # Balanced accuracy: mean of recall per class
    from sklearn.metrics import balanced_accuracy_score
    real_acc = balanced_accuracy_score(y, y_pred_all)

    # Permutation test
    rng = np.random.default_rng(42)
    shuffle_accs = []
    for _ in range(n_shuffles):
        y_shuf = y.copy()
        # Circular shift (preserve autocorrelation)
        shift = rng.integers(len(y) // 4, 3 * len(y) // 4)
        y_shuf = np.roll(y_shuf, shift)

        shuf_preds = np.zeros_like(y_shuf)
        for train_idx, test_idx in skf.split(X, y_shuf):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegressionCV(Cs=5, cv=3, max_iter=500,
                                        solver='lbfgs', random_state=42,
                                        class_weight='balanced')
            clf.fit(X_train, y_shuf[train_idx])
            shuf_preds[test_idx] = clf.predict(X_test)
        shuffle_accs.append(balanced_accuracy_score(y_shuf, shuf_preds))

    shuffle_accs = np.array(shuffle_accs)
    p_value = (shuffle_accs >= real_acc).mean()

    return {
        'accuracy': float(real_acc),
        'fold_accs': fold_accs,
        'shuffle_accs': shuffle_accs,
        'p_value': float(p_value),
        'y_true': y,
        'y_pred': y_pred_all,
        'y_prob': y_prob_all,
        'n_move': int(y.sum()),
        'n_rest': int((1 - y).sum()),
        'n_units': X.shape[1],
    }


# ── Analysis 2: Movement envelope prediction ────────────────────────

def movement_envelope_prediction(cache_dir, good_unit_ids, imu_derived,
                                  bank_id, bin_ms=500, n_folds=5,
                                  n_shuffles=200):
    """Predict smoothed gyro magnitude (movement intensity) from neural pop.

    Uses Ridge with 5-fold CV (every bin gets predicted). Includes
    permutation test with circular shifts.

    Returns dict with predicted trace, actual trace, R², shuffle distribution.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    fs_imu = imu_derived['sampling_rate']
    duration_s = len(imu_derived['signals']) / fs_imu
    names = imu_derived['names']
    signals = imu_derived['signals']

    # Get gyro magnitude — already smoothed in compute_derived_imu
    gyro_mag_idx = names.index('gyro_mag')
    gyro_mag = signals[:, gyro_mag_idx]

    # Bin gyro magnitude
    trim = int(5 * fs_imu)
    gyro_mag = gyro_mag[trim:]
    bin_samples = int(bin_ms / 1000 * fs_imu)
    n_bins = len(gyro_mag) // bin_samples
    y = gyro_mag[:n_bins * bin_samples].reshape(n_bins, bin_samples).mean(axis=1)

    # Get firing rates
    fr_binned, unit_ids_used = bin_spike_trains(
        cache_dir, good_unit_ids, bin_ms, bin_ms * 2, duration_s - 5)
    n = min(len(y), fr_binned.shape[0])
    y = y[:n]
    X = fr_binned[:n]

    if X.shape[1] == 0:
        return None

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 5-fold CV — every bin gets a prediction
    kf = KFold(n_splits=n_folds, shuffle=False)
    y_pred_all = np.full(len(y), np.nan)

    for train_idx, test_idx in kf.split(X):
        X_train = scaler_X.fit_transform(X[train_idx])
        X_test = scaler_X.transform(X[test_idx])
        y_train = scaler_y.fit_transform(y[train_idx].reshape(-1, 1)).ravel()

        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge.fit(X_train, y_train)

        y_pred_z = ridge.predict(X_test)
        y_pred_all[test_idx] = scaler_y.inverse_transform(
            y_pred_z.reshape(-1, 1)).ravel()

    # R² on full cross-validated predictions
    valid = ~np.isnan(y_pred_all)
    ss_res = np.sum((y[valid] - y_pred_all[valid]) ** 2)
    ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
    real_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    real_r, _ = stats.pearsonr(y[valid], y_pred_all[valid])

    # Permutation test
    rng = np.random.default_rng(42)
    shuffle_r2s = []
    for _ in range(n_shuffles):
        shift = rng.integers(len(y) // 4, 3 * len(y) // 4)
        y_shuf = np.roll(y, shift)

        y_pred_shuf = np.full(len(y), np.nan)
        for train_idx, test_idx in kf.split(X):
            X_train = scaler_X.fit_transform(X[train_idx])
            X_test = scaler_X.transform(X[test_idx])
            y_train = scaler_y.fit_transform(
                y_shuf[train_idx].reshape(-1, 1)).ravel()
            ridge = RidgeCV(alphas=[1, 10, 100])
            ridge.fit(X_train, y_train)
            y_pred_z = ridge.predict(X_test)
            y_pred_shuf[test_idx] = scaler_y.inverse_transform(
                y_pred_z.reshape(-1, 1)).ravel()

        v = ~np.isnan(y_pred_shuf)
        ss_r = np.sum((y_shuf[v] - y_pred_shuf[v]) ** 2)
        ss_t = np.sum((y_shuf[v] - y_shuf[v].mean()) ** 2)
        shuffle_r2s.append(1 - ss_r / ss_t if ss_t > 0 else 0)

    shuffle_r2s = np.array(shuffle_r2s)
    p_value = (shuffle_r2s >= real_r2).mean()

    # Time vector
    t = (np.arange(n) * bin_ms / 1000) + 5  # offset for trimmed 5s

    return {
        'r2': float(real_r2),
        'r': float(real_r),
        'p_value': float(p_value),
        'y_true': y,
        'y_pred': y_pred_all,
        't': t,
        'shuffle_r2s': shuffle_r2s,
        'n_units': X.shape[1],
    }


# ── Analysis 2 (old): Temporal Prediction ───────────────────────────

def temporal_prediction(cache_dir, good_unit_ids, imu_derived, bank_id,
                        bin_ms=200, train_frac=0.8):
    """Train Ridge on first 80% of data, predict IMU on held-out 20%.

    Returns predicted and actual IMU traces for the test segment.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    fs_imu = imu_derived['sampling_rate']
    duration_s = len(imu_derived['signals']) / fs_imu - 5  # trim 5s

    # Bin both signals
    imu_binned, t_bins, imu_names = bin_imu_signals(imu_derived, bin_ms)
    fr_binned, unit_ids_used = bin_spike_trains(
        cache_dir, good_unit_ids, bin_ms, bin_ms * 5, duration_s)

    # Trim first 5s
    trim_bins = int(5000 / bin_ms)
    n = min(fr_binned.shape[0], imu_binned.shape[0])
    fr = fr_binned[trim_bins:n]
    imu = imu_binned[trim_bins:n]
    t = t_bins[trim_bins:n]

    if fr.shape[1] == 0:
        return None

    # Exclude magnetometer
    keep = [i for i, name in enumerate(imu_names) if 'mag' not in name]
    imu = imu[:, keep]
    imu_names_filt = [imu_names[i] for i in keep]

    # Train/test split (temporal — no shuffling)
    n_train = int(len(fr) * train_frac)
    X_train, X_test = fr[:n_train], fr[n_train:]
    t_test = t[n_train:]

    # Z-score both
    scaler_X = StandardScaler().fit(X_train)
    X_train_z = scaler_X.transform(X_train)
    X_test_z = scaler_X.transform(X_test)

    results = {}
    for ch_idx, ch_name in enumerate(imu_names_filt):
        y_train = imu[:n_train, ch_idx]
        y_test = imu[n_train:, ch_idx]

        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_z = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        y_test_z = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # Ridge with cross-validation on training set
        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge.fit(X_train_z, y_train_z)

        y_pred_z = ridge.predict(X_test_z)

        # Back to original scale for plotting
        y_pred = scaler_y.inverse_transform(y_pred_z.reshape(-1, 1)).ravel()

        # R² on test set
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Correlation
        r_val, _ = stats.pearsonr(y_test, y_pred)

        results[ch_name] = {
            'r2': float(r2),
            'r': float(r_val),
            'y_test': y_test,
            'y_pred': y_pred,
            'alpha': float(ridge.alpha_),
        }

    return {
        't_test': t_test,
        'channels': results,
        'imu_names': imu_names_filt,
        'n_train': n_train,
        'n_test': len(t_test),
        'n_units': fr.shape[1],
        'unit_ids': unit_ids_used,
    }


# ── Analysis 3: Single-neuron scatters ──────────────────────────────

def single_neuron_scatters(cache_dir, good_unit_ids, imu_derived, bank_id,
                           bin_ms=500):
    """Compute per-neuron correlation with each IMU channel.

    Returns top neurons with their best IMU channel scatter data.
    """
    fs_imu = imu_derived['sampling_rate']
    duration_s = len(imu_derived['signals']) / fs_imu - 5

    imu_binned, t_bins, imu_names = bin_imu_signals(imu_derived, bin_ms)
    fr_binned, unit_ids_used = bin_spike_trains(
        cache_dir, good_unit_ids, bin_ms, bin_ms * 5, duration_s)

    trim_bins = int(5000 / bin_ms)
    n = min(fr_binned.shape[0], imu_binned.shape[0])
    fr = fr_binned[trim_bins:n]
    imu = imu_binned[trim_bins:n]

    # Exclude magnetometer
    keep = [i for i, name in enumerate(imu_names) if 'mag' not in name]
    imu = imu[:, keep]
    imu_names_filt = [imu_names[i] for i in keep]

    results = []
    for u_idx, uid in enumerate(unit_ids_used):
        best_r = 0
        best_ch = 0
        for ch_idx in range(imu.shape[1]):
            r, p = stats.pearsonr(fr[:, u_idx], imu[:, ch_idx])
            if abs(r) > abs(best_r):
                best_r = r
                best_ch = ch_idx
                best_p = p

        results.append({
            'unit_id': uid,
            'bank': bank_id,
            'best_channel': imu_names_filt[best_ch],
            'r': float(best_r),
            'p': float(best_p),
            'fr_data': fr[:, u_idx],
            'imu_data': imu[:, best_ch],
        })

    results.sort(key=lambda x: abs(x['r']), reverse=True)
    return results


# ── Plotting ────────────────────────────────────────────────────────

def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_movement_vs_rest(mvr_df, bank_id, merged_df, fig_dir):
    """Bar chart: firing rate during movement vs rest, sorted by effect size."""
    plt = _setup_matplotlib()

    sig = mvr_df[mvr_df['p_value'] < 0.01].sort_values('fr_ratio', ascending=False)
    nonsig = mvr_df[mvr_df['p_value'] >= 0.01]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: All units — movement modulation index
    ax = axes[0]
    sorted_df = mvr_df.sort_values('fr_ratio', ascending=False)
    colors = ['#E53935' if p < 0.01 else '#BDBDBD'
              for p in sorted_df['p_value']]
    ax.bar(range(len(sorted_df)), sorted_df['fr_ratio'].values - 1,
           color=colors, edgecolor='none')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Unit (sorted by modulation)')
    ax.set_ylabel('Movement Modulation Index\n(FR_move/FR_rest - 1)')
    ax.set_title(f'Bank {bank_id}: {len(sig)}/{len(mvr_df)} units significant (p<0.01)')

    # Panel 2: Top significant units — paired bars
    ax = axes[1]
    top = sig.head(15)
    if len(top) > 0:
        x = np.arange(len(top))
        w = 0.35
        ax.bar(x - w/2, top['fr_rest'], w, color='#90CAF9', label='Rest')
        ax.bar(x + w/2, top['fr_move'], w, color='#E53935', label='Movement')
        ax.set_xticks(x)

        # Get depths
        labels = []
        for _, row in top.iterrows():
            d = merged_df[(merged_df['unit_id'] == row['unit_id']) &
                          (merged_df['bank'] == bank_id)]['depth_um']
            depth = (d.values[0] + bank_id * BANK_OFFSET_UM) / 1000 if len(d) > 0 else 0
            labels.append(f"U{row['unit_id']}\n{depth:.1f}mm")
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title('Top Movement-Modulated Neurons')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No significant units', transform=ax.transAxes,
                ha='center', fontsize=12, color='gray')

    fig.suptitle(f'Movement vs Rest — Bank {bank_id}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / f'movement_vs_rest_bank{bank_id:02d}.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("    Saved movement_vs_rest_bank%02d.png", bank_id)


def plot_binary_classification(clf_result, bank_id, fig_dir):
    """Plot classification accuracy + permutation test + predicted time series."""
    plt = _setup_matplotlib()

    if clf_result is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Permutation test histogram
    ax = axes[0]
    ax.hist(clf_result['shuffle_accs'], bins=30, color='#BDBDBD',
            edgecolor='k', linewidth=0.3, alpha=0.8, label='Shuffled')
    ax.axvline(clf_result['accuracy'], color='#E53935', linewidth=2.5,
               label=f"Real: {clf_result['accuracy']:.1%}")
    ax.axvline(0.5, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Count')
    ax.set_title(f"p = {clf_result['p_value']:.4f}")
    ax.legend(fontsize=9)

    # Panel 2: Predicted probability over time
    ax = axes[1]
    t = np.arange(len(clf_result['y_true'])) * 0.5  # 500ms bins
    ax.fill_between(t, 0, 1, where=clf_result['y_true'] == 1,
                    color='#FFCDD2', alpha=0.5, label='Actual movement')
    ax.plot(t, clf_result['y_prob'], color='#1565C0', linewidth=0.8,
            label='Predicted P(moving)')
    ax.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('P(moving)')
    ax.set_title('Predicted Movement Probability')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: Confusion summary
    ax = axes[2]
    y_t = clf_result['y_true']
    y_p = clf_result['y_pred']
    tp = ((y_t == 1) & (y_p == 1)).sum()
    tn = ((y_t == 0) & (y_p == 0)).sum()
    fp = ((y_t == 0) & (y_p == 1)).sum()
    fn = ((y_t == 1) & (y_p == 0)).sum()
    labels = ['True Pos', 'True Neg', 'False Pos', 'False Neg']
    vals = [tp, tn, fp, fn]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title(f'Balanced Acc={clf_result["accuracy"]:.1%}  '
                 f'({clf_result["n_move"]} move, {clf_result["n_rest"]} rest bins)')

    fig.suptitle(f'Bank {bank_id}: Binary Movement Classification '
                 f'({clf_result["n_units"]} neurons, 5-fold CV)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / f'classification_bank{bank_id:02d}.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("    Saved classification_bank%02d.png", bank_id)


def plot_envelope_prediction(env_result, bank_id, fig_dir):
    """Plot predicted vs actual movement envelope + permutation test."""
    plt = _setup_matplotlib()

    if env_result is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                              gridspec_kw={'height_ratios': [3, 1]})

    # Panel 1: Actual vs predicted traces
    ax = axes[0]
    t = env_result['t']
    valid = ~np.isnan(env_result['y_pred'])
    ax.plot(t[valid], env_result['y_true'][valid], color='#BDBDBD',
            linewidth=1.0, alpha=0.9, label='Actual gyro magnitude')
    ax.plot(t[valid], env_result['y_pred'][valid], color='#1565C0',
            linewidth=1.2, alpha=0.85, label='Predicted from neurons')
    ax.set_ylabel('Gyro Magnitude (°/s)')
    ax.set_title(f'Bank {bank_id}: Movement Envelope Prediction  '
                 f'R²={env_result["r2"]:.3f}  r={env_result["r"]:.3f}  '
                 f'p={env_result["p_value"]:.4f}  '
                 f'({env_result["n_units"]} neurons, 5-fold CV)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # Panel 2: Permutation histogram
    ax = axes[1]
    ax.hist(env_result['shuffle_r2s'], bins=30, color='#BDBDBD',
            edgecolor='k', linewidth=0.3, alpha=0.8, label='Shuffled R²')
    ax.axvline(env_result['r2'], color='#E53935', linewidth=2.5,
               label=f"Real R²={env_result['r2']:.3f}")
    ax.set_xlabel('R²')
    ax.set_ylabel('Count')
    ax.set_title(f"Permutation test: p={env_result['p_value']:.4f}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / f'envelope_prediction_bank{bank_id:02d}.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("    Saved envelope_prediction_bank%02d.png", bank_id)


def plot_temporal_prediction(pred_result, bank_id, fig_dir):
    """Overlay predicted vs actual IMU for the best channels."""
    plt = _setup_matplotlib()

    if pred_result is None:
        return

    # Rank channels by R²
    ranked = sorted(pred_result['channels'].items(),
                    key=lambda x: x[1]['r2'], reverse=True)

    # Plot top 4 channels
    n_plot = min(4, len(ranked))
    fig, axes = plt.subplots(n_plot, 1, figsize=(16, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    t = pred_result['t_test']

    for idx, (ch_name, res) in enumerate(ranked[:n_plot]):
        ax = axes[idx]
        ax.plot(t, res['y_test'], color='#BDBDBD', linewidth=1.0, alpha=0.9,
                label='Actual')
        ax.plot(t, res['y_pred'], color='#1565C0', linewidth=1.2, alpha=0.85,
                label='Predicted')
        ax.set_ylabel(ch_name, fontsize=9)
        r2_str = f"R²={res['r2']:.3f}" if res['r2'] > -1 else f"R²={res['r2']:.1f}"
        ax.set_title(f"{ch_name}  {r2_str}  r={res['r']:.3f}  "
                     f"(alpha={res['alpha']:.0f})",
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Bank {bank_id}: Temporal Prediction (train 80% → predict 20%)\n'
                 f'{pred_result["n_units"]} neurons, {pred_result["n_test"]} test bins',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / f'temporal_prediction_bank{bank_id:02d}.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("    Saved temporal_prediction_bank%02d.png", bank_id)


def plot_neuron_scatters(scatter_results, bank_id, merged_df, fig_dir, n_top=12):
    """Scatter plots: FR vs best IMU channel for top correlated neurons."""
    plt = _setup_matplotlib()
    from sklearn.linear_model import LinearRegression

    top = scatter_results[:n_top]
    n_cols = 4
    n_rows = (len(top) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, r in enumerate(top):
        ax = axes[idx]
        fr = r['fr_data']
        imu = r['imu_data']

        ax.scatter(fr, imu, s=8, alpha=0.4, color='#1565C0', edgecolors='none')

        # Regression line
        reg = LinearRegression().fit(fr.reshape(-1, 1), imu)
        x_line = np.linspace(fr.min(), fr.max(), 100)
        ax.plot(x_line, reg.predict(x_line.reshape(-1, 1)),
                color='#E53935', linewidth=2)

        # Depth
        d = merged_df[(merged_df['unit_id'] == r['unit_id']) &
                      (merged_df['bank'] == bank_id)]['depth_um']
        depth = (d.values[0] + bank_id * BANK_OFFSET_UM) / 1000 if len(d) > 0 else 0

        sig = '*' if r['p'] < 0.01 else ''
        ax.set_title(f"U{r['unit_id']} ({depth:.1f}mm){sig}\n"
                     f"r={r['r']:.3f}  {r['best_channel']}",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('FR (Hz)', fontsize=7)
        ax.set_ylabel(r['best_channel'], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.15)

    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Bank {bank_id}: Single-Neuron Correlations (500ms bins)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / f'neuron_scatters_bank{bank_id:02d}.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("    Saved neuron_scatters_bank%02d.png", bank_id)


def plot_summary(all_mvr, all_pred, fig_dir):
    """Cross-bank summary figure."""
    plt = _setup_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Fraction movement-modulated per bank
    ax = axes[0]
    if len(all_mvr) > 0:
        bank_sig = all_mvr.groupby('bank').apply(
            lambda x: (x['p_value'] < 0.01).mean())
        ax.bar(bank_sig.index, bank_sig.values, color='#E53935', alpha=0.8)
        ax.set_xlabel('Bank')
        ax.set_ylabel('Fraction Significant (p<0.01)')
        ax.set_title('Movement-Modulated Neurons')
        ax.set_ylim(0, 1)

    # Panel 2: Movement envelope R² per bank
    ax = axes[1]
    if all_pred:
        bank_ids = sorted(all_pred.keys())
        for bk in bank_ids:
            res = all_pred[bk]
            if res is None:
                continue
            r2 = max(res.get('r2', 0), 0)
            ax.bar(bk, r2, color='#1565C0', alpha=0.8)
        ax.set_xlabel('Bank')
        ax.set_ylabel('R² (5-fold CV)')
        ax.set_title('Movement Envelope Prediction')

    # Panel 3: Distribution of single-neuron |r| across all banks
    ax = axes[2]
    if len(all_mvr) > 0:
        ax.hist(all_mvr['fr_ratio'] - 1, bins=40, color='#4472C4',
                edgecolor='k', linewidth=0.3, alpha=0.8)
        ax.axvline(0, color='red', linewidth=1, linestyle='--')
        ax.set_xlabel('Movement Modulation Index')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Movement Effects')

    fig.suptitle('Neural-IMU Decoding Summary', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / 'decoding_summary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved decoding_summary.png")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Robust neural-IMU decoding analysis')
    parser.add_argument('data_dir', help='Directory containing .dat segments')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--fig-dir', type=str, default=None)
    parser.add_argument('--bank', type=int, nargs='+', default=None)
    parser.add_argument('--neuron-set', choices=['legacy', 'classifier',
                        'classifier-strict'], default='legacy',
                        help="Which units to analyze: 'legacy' (is_good "
                              "tier), 'classifier' (custom_p_neural>=0.5), "
                              "'classifier-strict' (>=0.8)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_root = Path(os.environ.get('DATA_ROOT', 'E:/'))
    output_dir = Path(args.output) if args.output else (
        data_root / 'sorting_output' / data_dir.name)
    suffix = '' if args.neuron_set == 'legacy' else (
        '_classifier' if args.neuron_set == 'classifier' else '_classifier_strict')
    fig_dir = Path(args.fig_dir) if args.fig_dir else (
        data_root / 'figures' / 'sorting' / data_dir.name
        / f'decoding{suffix}')
    fig_dir.mkdir(parents=True, exist_ok=True)
    bank_filter = set(args.bank) if args.bank else None

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("NEURAL-IMU DECODING: %s", data_dir.name)
    logger.info("=" * 70)

    # Load merged units
    merged_df = pd.read_csv(output_dir / 'merged' / 'merged_units.csv')
    if args.neuron_set != 'legacy':
        col = ('custom_neural' if args.neuron_set == 'classifier'
               else 'custom_neural_strict')
        if col not in merged_df.columns:
            logger.warning("merged_units.csv has no %s column; falling back "
                           "to legacy is_good", col)
        else:
            merged_df['is_good'] = merged_df[col].astype(bool)
            logger.info("Neuron set: %s (%s) — %d units",
                         args.neuron_set, col, int(merged_df['is_good'].sum()))
    good_df = merged_df[merged_df['is_good']]
    logger.info("Good units: %d across %d banks",
                 len(good_df), good_df['bank'].nunique())

    # Load IMU
    logger.info("\n[1/4] LOADING IMU")
    logger.info("=" * 70)
    imu_segments = load_per_segment_imu(data_dir, output_dir)

    banks = sorted(good_df['bank'].unique())
    if bank_filter:
        banks = [b for b in banks if b in bank_filter]

    all_mvr_dfs = []
    all_pred = {}
    bank_summary_rows = []

    for bank_id in banks:
        logger.info("\n[2/4] BANK %d — analysis", bank_id)
        logger.info("=" * 70)

        if bank_id not in imu_segments:
            logger.warning("  No IMU for bank %d", bank_id)
            continue

        imu_raw = imu_segments[bank_id]
        imu_phys = convert_imu_to_physical(imu_raw)
        imu_derived = compute_derived_imu(imu_phys)

        cache_dir = output_dir / f'bank{bank_id:02d}_ks4' / 'analyzer_cache'
        bank_good_ids = good_df[good_df['bank'] == bank_id]['unit_id'].tolist()

        if not bank_good_ids:
            logger.warning("  No good units for bank %d", bank_id)
            continue

        # Analysis 1: Movement vs Rest
        logger.info("  [A] Movement vs Rest...")
        mvr_df = movement_vs_rest_analysis(
            cache_dir, bank_good_ids, imu_derived, bank_id)
        n_sig = (mvr_df['p_value'] < 0.01).sum() if len(mvr_df) > 0 else 0
        logger.info("    %d/%d significant (p<0.01)", n_sig, len(mvr_df))
        all_mvr_dfs.append(mvr_df)
        plot_movement_vs_rest(mvr_df, bank_id, merged_df, fig_dir)

        # Analysis 1b: Binary classification (moving vs not)
        logger.info("  [B] Binary movement classification (200 shuffles)...")
        clf_result = binary_movement_classification(
            cache_dir, bank_good_ids, imu_derived, bank_id)
        if clf_result is not None:
            logger.info("    Accuracy: %.1f%% (p=%.4f, chance=%.1f%%)",
                         clf_result['accuracy'] * 100,
                         clf_result['p_value'],
                         clf_result['n_move'] / (clf_result['n_move'] + clf_result['n_rest']) * 100)
        plot_binary_classification(clf_result, bank_id, fig_dir)

        # Analysis 2: Movement envelope prediction
        logger.info("  [C] Movement envelope prediction (200 shuffles)...")
        env_result = movement_envelope_prediction(
            cache_dir, bank_good_ids, imu_derived, bank_id)
        all_pred[bank_id] = env_result
        if env_result is not None:
            logger.info("    R²=%.3f  r=%.3f  p=%.4f",
                         env_result['r2'], env_result['r'], env_result['p_value'])
        plot_envelope_prediction(env_result, bank_id, fig_dir)

        # Analysis 3: Single-neuron scatters
        logger.info("  [D] Single-neuron scatters...")
        scatters = single_neuron_scatters(
            cache_dir, bank_good_ids, imu_derived, bank_id)
        if scatters:
            logger.info("    Top: U%d %s r=%.3f",
                         scatters[0]['unit_id'],
                         scatters[0]['best_channel'],
                         scatters[0]['r'])
        plot_neuron_scatters(scatters, bank_id, merged_df, fig_dir)

        # Collect per-bank summary row
        row = {
            'bank': bank_id,
            'n_units': len(bank_good_ids),
            'mvr_n_significant': int((mvr_df['p_value'] < 0.01).sum())
                if len(mvr_df) > 0 else 0,
            'clf_accuracy': clf_result['accuracy'] if clf_result else np.nan,
            'clf_p_value': clf_result['p_value'] if clf_result else np.nan,
            'env_r2': env_result['r2'] if env_result else np.nan,
            'env_r': env_result['r'] if env_result else np.nan,
            'env_p_value': env_result['p_value'] if env_result else np.nan,
            'top_scatter_unit': scatters[0]['unit_id'] if scatters else np.nan,
            'top_scatter_channel': scatters[0]['best_channel'] if scatters else '',
            'top_scatter_r': scatters[0]['r'] if scatters else np.nan,
        }
        bank_summary_rows.append(row)

    # Summary
    logger.info("\n[3/4] SUMMARY")
    logger.info("=" * 70)
    all_mvr = pd.concat(all_mvr_dfs, ignore_index=True) if all_mvr_dfs else pd.DataFrame()
    plot_summary(all_mvr, all_pred, fig_dir)

    # Save
    logger.info("\n[4/4] SAVING")
    logger.info("=" * 70)
    out_dir = output_dir / f'decoding{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(all_mvr) > 0:
        all_mvr.to_csv(out_dir / 'movement_vs_rest.csv', index=False)
        logger.info("  Saved movement_vs_rest.csv")
    if bank_summary_rows:
        bank_summary_df = pd.DataFrame(bank_summary_rows)
        bank_summary_df.to_csv(out_dir / 'bank_decoding_summary.csv',
                                index=False)
        logger.info("  Saved bank_decoding_summary.csv (%d banks)",
                     len(bank_summary_df))

    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE in %.1f min", elapsed / 60)
    logger.info("Output: %s", fig_dir)


if __name__ == '__main__':
    main()
