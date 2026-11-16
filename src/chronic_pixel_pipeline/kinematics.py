"""Continuity-block-aware movement bout segmentation, PETHs, lag tuning, and neuron-dropping curves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def _blocked_ridge_cv_r2(
    x: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
) -> float:
    """Evaluate contiguous blocked k-fold Ridge regression R^2."""
    n = y.size
    folds = np.array_split(np.arange(n), max(2, int(n_folds)))
    preds = np.empty(n, dtype=np.float64)
    for te in folds:
        if te.size == 0:
            continue
        tr = np.setdiff1d(np.arange(n), te, assume_unique=True)
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr])
        x_te = scaler.transform(x[te])
        model = RidgeCV(alphas=alphas)
        model.fit(x_tr, y[tr])
        preds[te] = model.predict(x_te)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def detect_spontaneous_movement_bouts(
    t_s: np.ndarray,
    acc_dyn_g: np.ndarray,
    block_ids: np.ndarray,
    rest_thresh_g: float = 0.035,
    active_thresh_g: float = 0.08,
    min_quiet_pre_s: float = 1.5,
    min_bout_s: float = 0.5,
    min_gap_between_bouts_s: float = 2.0,
    edge_guard_s: float = 2.0,
) -> dict[str, Any]:
    """Identify spontaneous rest epochs and clean movement-onset bouts within continuity blocks.

    All boundary checks operate strictly inside each packet-continuity block so gap jumps
    never trigger false movement onsets or contaminate quiet-rest baselines.
    """
    t = np.asarray(t_s, dtype=np.float64).ravel()
    acc = np.asarray(acc_dyn_g, dtype=np.float64).ravel()
    bids = np.asarray(block_ids, dtype=np.int64).ravel()
    if t.size != acc.size or acc.size != bids.size:
        raise ValueError("t_s, acc_dyn_g, and block_ids must have identical length")
    if t.size < 4:
        return {
            "onset_times_s": np.empty(0, dtype=np.float64),
            "bout_durations_s": np.empty(0, dtype=np.float64),
            "rest_mask": np.zeros(t.size, dtype=bool),
            "active_mask": np.zeros(t.size, dtype=bool),
            "rest_intervals_s": [],
            "active_intervals_s": [],
        }

    dt_s = float(np.median(np.diff(t)))
    pre_bins = max(1, int(round(min_quiet_pre_s / max(dt_s, 1e-4))))
    bout_bins = max(1, int(round(min_bout_s / max(dt_s, 1e-4))))
    guard_bins = max(1, int(round(edge_guard_s / max(dt_s, 1e-4))))
    refrac_bins = max(1, int(round(min_gap_between_bouts_s / max(dt_s, 1e-4))))

    rest_mask = np.zeros(t.size, dtype=bool)
    active_mask = np.zeros(t.size, dtype=bool)
    onset_times: list[float] = []
    bout_durs: list[float] = []
    rest_intervals: list[tuple[float, float]] = []
    active_intervals: list[tuple[float, float]] = []

    for bid in np.unique(bids):
        idx = np.flatnonzero(bids == bid)
        if idx.size <= 2 * guard_bins + pre_bins + bout_bins:
            continue
        inner = idx[guard_bins:-guard_bins]
        sub_acc = acc[inner]
        sub_t = t[inner]

        r_local = sub_acc <= float(rest_thresh_g)
        a_local = sub_acc >= float(active_thresh_g)
        rest_mask[inner] = r_local
        active_mask[inner] = a_local

        # Collect contiguous rest & active intervals inside block
        for flag_arr, target_list in ((r_local, rest_intervals), (a_local, active_intervals)):
            pad = np.concatenate(([False], flag_arr, [False]))
            starts = np.flatnonzero(~pad[:-1] & pad[1:])
            ends = np.flatnonzero(pad[:-1] & ~pad[1:])
            for s, e in zip(starts, ends):
                if e - s >= bout_bins:
                    target_list.append((float(sub_t[s]), float(sub_t[e - 1] + dt_s)))

        last_onset = -refrac_bins - 1
        k = pre_bins
        limit = sub_acc.size - bout_bins
        while k < limit:
            if k - last_onset >= refrac_bins:
                pre_win = sub_acc[k - pre_bins : k]
                post_win = sub_acc[k : k + bout_bins]
                if (
                    float(np.median(pre_win)) <= float(rest_thresh_g)
                    and float(np.max(pre_win)) < float(active_thresh_g)
                    and float(np.min(post_win)) >= float(active_thresh_g)
                ):
                    onset_times.append(float(sub_t[k]))
                    end_k = k + bout_bins
                    while end_k < sub_acc.size and sub_acc[end_k] >= float(rest_thresh_g):
                        end_k += 1
                    bout_durs.append(float((end_k - k) * dt_s))
                    last_onset = k
                    k = end_k + 1
                    continue
            k += 1

    return {
        "onset_times_s": np.asarray(onset_times, dtype=np.float64),
        "bout_durations_s": np.asarray(bout_durs, dtype=np.float64),
        "rest_mask": rest_mask,
        "active_mask": active_mask,
        "rest_intervals_s": rest_intervals,
        "active_intervals_s": active_intervals,
    }


def compute_movement_onset_peths(
    unit_spikes_s: list[np.ndarray],
    onset_times_s: np.ndarray,
    window_s: tuple[float, float] = (-2.0, 3.0),
    bin_s: float = 0.05,
    baseline_s: tuple[float, float] = (-1.5, -0.3),
    response_s: tuple[float, float] = (-0.1, 1.0),
) -> dict[str, Any]:
    """Compute peri-movement-onset firing rate PETHs (Hz and baseline-z-scored) per neuron."""
    onsets = np.asarray(onset_times_s, dtype=np.float64).ravel()
    w0, w1 = float(window_s[0]), float(window_s[1])
    edges_rel = np.arange(w0, w1 + 0.5 * float(bin_s), float(bin_s), dtype=np.float64)
    centers_rel = 0.5 * (edges_rel[:-1] + edges_rel[1:])
    n_units = len(unit_spikes_s)
    n_bins = centers_rel.size

    if n_units == 0 or onsets.size == 0:
        return {
            "t_rel_s": centers_rel,
            "peth_hz": np.empty((n_units, n_bins), dtype=np.float64),
            "peth_z": np.empty((n_units, n_bins), dtype=np.float64),
            "trial_rates_hz": np.empty((n_units, onsets.size, n_bins), dtype=np.float64),
            "sort_order": np.arange(n_units, dtype=np.int64),
            "mod_score": np.zeros(n_units, dtype=np.float64),
        }

    trials = np.zeros((n_units, onsets.size, n_bins), dtype=np.float64)
    for ui, sp in enumerate(unit_spikes_s):
        st = np.sort(np.asarray(sp, dtype=np.float64).ravel())
        for ti, ton in enumerate(onsets):
            edges = ton + edges_rel
            i0 = int(np.searchsorted(st, edges[0], side="left"))
            i1 = int(np.searchsorted(st, edges[-1], side="right"))
            cnt, _ = np.histogram(st[i0:i1], bins=edges)
            trials[ui, ti] = cnt.astype(np.float64) / float(bin_s)

    peth_hz = np.mean(trials, axis=1)
    base_mask = (centers_rel >= baseline_s[0]) & (centers_rel <= baseline_s[1])
    resp_mask = (centers_rel >= response_s[0]) & (centers_rel <= response_s[1])
    mu = np.mean(peth_hz[:, base_mask], axis=1, keepdims=True)
    sd = np.std(peth_hz[:, base_mask], axis=1, keepdims=True)
    sd = np.maximum(sd, np.maximum(0.5, 0.1 * np.abs(mu)))
    peth_z = (peth_hz - mu) / sd

    resp_vals = np.mean(peth_z[:, resp_mask], axis=1)
    sort_order = np.argsort(resp_vals)[::-1].astype(np.int64)

    return {
        "t_rel_s": centers_rel,
        "peth_hz": peth_hz,
        "peth_z": peth_z,
        "trial_rates_hz": trials,
        "sort_order": sort_order,
        "mod_score": resp_vals,
    }


def compute_neuron_kinematic_lag_tuning(
    rate_mat: np.ndarray,
    kin_vec: np.ndarray,
    block_ids: np.ndarray,
    bin_dt_s: float = 0.05,
    max_lag_s: float = 0.5,
) -> pd.DataFrame:
    """Compute block-safe single-neuron Pearson cross-correlation across kinematic lags.

    Negative ``best_lag_ms`` indicates neural modulation preceding head kinematics
    (motor-leading), whereas positive lags indicate reafferent/post-movement modulation.
    """
    x = np.asarray(rate_mat, dtype=np.float64)
    y = np.asarray(kin_vec, dtype=np.float64).ravel()
    bids = np.asarray(block_ids, dtype=np.int64).ravel()
    max_k = max(1, int(round(float(max_lag_s) / float(bin_dt_s))))
    lags_k = np.arange(-max_k, max_k + 1, dtype=int)
    lags_ms = lags_k.astype(np.float64) * float(bin_dt_s) * 1000.0

    block_slices = [np.flatnonzero(bids == bid) for bid in np.unique(bids)]
    n_units = x.shape[1]
    r_mat = np.zeros((n_units, lags_k.size), dtype=np.float64)

    for li, k in enumerate(lags_k):
        xs_l: list[np.ndarray] = []
        ys_l: list[np.ndarray] = []
        for sl in block_slices:
            m = sl.size
            if m <= abs(k) + 4:
                continue
            # Positive k: neural(t) correlated with kin(t + k) -> kin lags neural if k > 0?
            # Standard convention: lag = t_neural - t_kin; negative lag => neural precedes kin.
            if k < 0:
                xs_l.append(x[sl[: m + k]])
                ys_l.append(y[sl[-k:]])
            elif k > 0:
                xs_l.append(x[sl[k:]])
                ys_l.append(y[sl[: m - k]])
            else:
                xs_l.append(x[sl])
                ys_l.append(y[sl])
        if not xs_l:
            continue
        x_cat = np.vstack(xs_l)
        y_cat = np.concatenate(ys_l)
        yc = y_cat - np.mean(y_cat)
        y_norm = float(np.sqrt(np.sum(yc * yc)))
        if y_norm <= 1e-12:
            continue
        xc = x_cat - np.mean(x_cat, axis=0, keepdims=True)
        x_norm = np.sqrt(np.sum(xc * xc, axis=0))
        corr = (xc.T @ yc) / np.maximum(x_norm * y_norm, 1e-12)
        corr[x_norm <= 1e-12] = 0.0
        r_mat[:, li] = corr

    rows: list[dict[str, Any]] = []
    for u in range(n_units):
        curve = r_mat[u]
        best_i = int(np.argmax(np.abs(curve)))
        rows.append(
            {
                "unit_idx": int(u),
                "peak_r": float(curve[best_i]),
                "abs_peak_r": float(abs(curve[best_i])),
                "best_lag_ms": float(lags_ms[best_i]),
                "r_zero_lag": float(curve[max_k]),
            }
        )
    return pd.DataFrame(rows)


def compute_neuron_dropping_curve(
    rate_mat: np.ndarray,
    target_vec: np.ndarray,
    block_ids: np.ndarray,
    counts: list[int] | None = None,
    n_draws: int = 20,
    n_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute Trautmann Fig. 3g neuron-dropping Ridge R^2 scaling curves across ensemble sizes."""
    x = np.asarray(rate_mat, dtype=np.float64)
    y = np.asarray(target_vec, dtype=np.float64).ravel()
    _ = np.asarray(block_ids, dtype=np.int64).ravel()
    n_units = x.shape[1]

    if counts is None:
        base_grid = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
        valid_counts = sorted({k for k in base_grid if k < n_units} | {n_units})
    else:
        valid_counts = sorted({min(int(k), n_units) for k in counts if int(k) >= 1})

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for k in valid_counts:
        draws = 1 if k == n_units else int(n_draws)
        vals = np.empty(draws, dtype=np.float64)
        for d in range(draws):
            cols = np.arange(n_units) if k == n_units else rng.choice(n_units, size=k, replace=False)
            vals[d] = _blocked_ridge_cv_r2(x[:, cols], y, n_folds=n_folds)
        rows.append(
            {
                "n_neurons": int(k),
                "n_draws": int(draws),
                "mean_r2": float(np.mean(vals)),
                "sd_r2": float(np.std(vals)),
                "median_r2": float(np.median(vals)),
                "iqr_low": float(np.percentile(vals, 25)),
                "iqr_high": float(np.percentile(vals, 75)),
            }
        )
    return pd.DataFrame(rows)
