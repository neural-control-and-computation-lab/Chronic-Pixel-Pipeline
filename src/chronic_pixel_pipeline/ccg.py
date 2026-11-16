"""Exact jitter-corrected spike-timing cross-correlograms (Trautmann et al. 2025 method)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _interval_jitter_expected_counts(
    spike_trains_s: list[np.ndarray],
    t_start_s: float,
    t_end_s: float,
    lags_ms: np.ndarray,
    bin_ms: float = 1.0,
    jitter_win_ms: float = 25.0,
) -> np.ndarray:
    """Exact expected coincidences under independent uniform spike jitter within disjoint windows.

    Because neurons ``i != j`` are jittered independently across disjoint intervals of width
    ``jitter_win_ms``, multiplying the uniform per-ms density of each neuron inside every window
    and shifting by integer lags ``tau`` matches Monte Carlo interval jitter in the limit.
    """
    n_units = len(spike_trains_s)
    n_lags = len(lags_ms)
    if n_units < 2 or t_end_s <= t_start_s:
        return np.zeros((n_units, n_units, n_lags), dtype=np.float64)

    win_s = float(jitter_win_ms) / 1000.0
    bin_s = float(bin_ms) / 1000.0
    w_bins = max(1, int(round(float(jitter_win_ms) / float(bin_ms))))
    n_win = max(1, int(np.ceil((t_end_s - t_start_s) / win_s)))

    # Density per bin (spikes per 1-ms bin) inside each disjoint jitter window
    dens = np.zeros((n_units, n_win), dtype=np.float64)
    for i, sp in enumerate(spike_trains_s):
        w_idx = np.floor((sp - t_start_s) / win_s).astype(np.int64)
        valid = (w_idx >= 0) & (w_idx < n_win)
        if valid.any():
            np.add.at(dens[i], w_idx[valid], 1.0 / float(w_bins))

    # For lag tau (in 1-ms bins), two windows u and v overlap by ov_bins(d = v - u).
    # Only small window offsets |v - u| <= ceil(max_lag / w_bins) have nonzero overlap.
    expected = np.zeros((n_units, n_units, n_lags), dtype=np.float64)
    max_w_shift = int(np.ceil(np.max(np.abs(lags_ms)) / float(w_bins))) + 1

    for l_idx, tau_bins in enumerate(lags_ms.astype(int)):
        acc = np.zeros((n_units, n_units), dtype=np.float64)
        for dw in range(-max_w_shift, max_w_shift + 1):
            # Overlap count between shifted [u*w + tau, (u+1)*w + tau) and [(u+dw)*w, (u+dw+1)*w)
            s_a = tau_bins
            e_a = s_a + w_bins
            s_b = dw * w_bins
            e_b = s_b + w_bins
            ov = min(e_a, e_b) - max(s_a, s_b)
            if ov <= 0:
                continue
            u0 = max(0, -dw)
            u1 = min(n_win, n_win - dw)
            if u1 <= u0:
                continue
            acc += ov * (dens[:, u0:u1] @ dens[:, (u0 + dw) : (u1 + dw)].T)
        expected[:, :, l_idx] = acc

    return expected


def compute_jitter_corrected_ccgs(
    spike_trains_s: list[np.ndarray],
    max_lag_ms: int = 50,
    bin_ms: float = 1.0,
    jitter_win_ms: float = 25.0,
    sig_lag_ms: float = 10.0,
    flank_min_ms: float = 25.0,
    z_thresh: float = 7.0,
    min_peak_count: int = 10,
    min_rate_hz: float = 1.0,
    valid_duration_s: float | None = None,
) -> dict[str, Any]:
    """Compute raw and 25-ms jitter-corrected CCGs and identify putative connected pairs.

    Normalization follows Trautmann et al. (2025) Methods:
    ``CCG(tau) = C(tau) / (theta(tau) * sqrt(lambda_1 * lambda_2))`` expressed in
    coincidences per geometric-mean spike, with significance defined by peak Z-score
    relative to the flanking noise distribution (``25 <= |tau| <= 50 ms``).
    """
    raw_trains = [np.sort(np.asarray(s, dtype=np.float64).reshape(-1)) for s in spike_trains_s]
    non_empty = [s for s in raw_trains if s.size > 0]
    n_lags = 2 * int(max_lag_ms / bin_ms) + 1
    lags_ms = np.linspace(-float(max_lag_ms), float(max_lag_ms), n_lags)

    if not non_empty:
        return {
            "eligible_indices": np.empty(0, dtype=np.int64),
            "lags_ms": lags_ms,
            "ccg_jc": np.empty((0, 0, n_lags), dtype=np.float64),
            "ccg_raw": np.empty((0, 0, n_lags), dtype=np.float64),
            "pairs_df": pd.DataFrame(),
            "duration_s": 0.0,
        }

    t_min = min(float(s[0]) for s in non_empty)
    t_max = max(float(s[-1]) for s in non_empty)
    if valid_duration_s is not None and valid_duration_s > 0:
        duration_s = float(valid_duration_s)
        t_end = t_min + duration_s
    else:
        duration_s = max(t_max - t_min, 1e-3)
        t_end = t_max

    rates = np.array([float(s.size) / max(duration_s, 1e-6) for s in raw_trains], dtype=np.float64)
    elig_idx = np.flatnonzero(rates >= float(min_rate_hz)).astype(np.int64)
    elig_trains = [raw_trains[i] for i in elig_idx]
    n_e = len(elig_trains)

    if n_e < 2:
        return {
            "eligible_indices": elig_idx,
            "lags_ms": lags_ms,
            "ccg_jc": np.empty((n_e, n_e, n_lags), dtype=np.float64),
            "ccg_raw": np.empty((n_e, n_e, n_lags), dtype=np.float64),
            "pairs_df": pd.DataFrame(),
            "duration_s": duration_s,
        }

    # Integer bin edges shifted by -0.5 * bin_ms so bin center at 0 covers [-0.5, +0.5) ms
    bin_s = float(bin_ms) / 1000.0
    max_s = (float(max_lag_ms) + 0.5 * float(bin_ms)) / 1000.0
    counts_raw = np.zeros((n_e, n_e, n_lags), dtype=np.float64)

    for i in range(n_e):
        ti = elig_trains[i]
        for j in range(i + 1, n_e):
            tj = elig_trains[j]
            # Vectorized searchsorted window scan over smaller/reference train
            lo = np.searchsorted(tj, ti - max_s, side="left")
            hi = np.searchsorted(tj, ti + max_s, side="right")
            span = hi - lo
            nz = np.flatnonzero(span > 0)
            if not nz.size:
                continue
            dt_chunks = [tj[lo[k] : hi[k]] - ti[k] for k in nz]
            dts_ms = np.concatenate(dt_chunks) * 1000.0
            bin_pos = np.floor((dts_ms + float(max_lag_ms) + 0.5 * float(bin_ms)) / float(bin_ms)).astype(np.int64)
            valid_b = (bin_pos >= 0) & (bin_pos < n_lags)
            if valid_b.any():
                c_ij = np.bincount(bin_pos[valid_b], minlength=n_lags).astype(np.float64)
                counts_raw[i, j, :] = c_ij
                counts_raw[j, i, :] = c_ij[::-1]

    expected_counts = _interval_jitter_expected_counts(
        elig_trains,
        t_start_s=t_min,
        t_end_s=t_end,
        lags_ms=lags_ms,
        bin_ms=bin_ms,
        jitter_win_ms=jitter_win_ms,
    )

    theta_frac = np.maximum(1.0 - np.abs(lags_ms) * 0.001 / max(duration_s, 1.0), 1e-3)
    n_sp = np.array([float(s.size) for s in elig_trains], dtype=np.float64)
    geo_norm = np.sqrt(np.outer(n_sp, n_sp))[:, :, None] * theta_frac[None, None, :]

    ccg_raw_norm = counts_raw / np.maximum(geo_norm, 1e-12)
    ccg_exp_norm = expected_counts / np.maximum(geo_norm, 1e-12)
    ccg_jc = ccg_raw_norm - ccg_exp_norm

    sig_mask = np.abs(lags_ms) <= float(sig_lag_ms)
    flank_mask = np.abs(lags_ms) >= float(flank_min_ms)
    sig_lags = lags_ms[sig_mask]

    pair_rows: list[dict[str, Any]] = []
    for i in range(n_e):
        for j in range(i + 1, n_e):
            jc = ccg_jc[i, j]
            flank = jc[flank_mask]
            flank_mean = float(np.mean(flank))
            flank_sd = float(np.std(flank))
            if flank_sd <= 1e-12:
                continue

            z_curve = (jc - flank_mean) / flank_sd
            z_sig = z_curve[sig_mask]
            max_pos_idx = int(np.argmax(z_sig))
            min_neg_idx = int(np.argmin(z_sig))
            max_z = float(z_sig[max_pos_idx])
            min_z = float(z_sig[min_neg_idx])
            raw_peak = float(np.max(counts_raw[i, j, sig_mask]))

            if max_z > float(z_thresh) and raw_peak >= float(min_peak_count):
                peak_lag = float(sig_lags[max_pos_idx])
                z_val = max_z
                peak_jc = float(jc[sig_mask][max_pos_idx])
                if abs(peak_lag) <= 0.5:
                    conn_type = "synchronous_0ms"
                elif 1.0 <= abs(peak_lag) <= 5.0:
                    conn_type = "monosynaptic_exc"
                else:
                    conn_type = "broad_lag"
            elif abs(min_z) > float(z_thresh) and raw_peak >= float(min_peak_count):
                peak_lag = float(sig_lags[min_neg_idx])
                z_val = min_z
                peak_jc = float(jc[sig_mask][min_neg_idx])
                conn_type = "inhibitory_trough"
            else:
                continue

            pair_rows.append(
                {
                    "elig_i": int(i),
                    "elig_j": int(j),
                    "unit_idx_i": int(elig_idx[i]),
                    "unit_idx_j": int(elig_idx[j]),
                    "peak_lag_ms": peak_lag,
                    "z_score": z_val,
                    "ccg_jc_peak": peak_jc,
                    "raw_peak_coincidences": int(raw_peak),
                    "connection_type": conn_type,
                }
            )

    return {
        "eligible_indices": elig_idx,
        "lags_ms": lags_ms,
        "ccg_jc": ccg_jc,
        "ccg_raw": ccg_raw_norm,
        "ccg_expected": ccg_exp_norm,
        "pairs_df": pd.DataFrame(pair_rows),
        "duration_s": duration_s,
    }
