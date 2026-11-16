"""Single-spike drift clouds and Kilosort4 within-session drift trajectory utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats


def load_ks4_drift_trace(bank_dir: Path, fs_hz: float = 30000.0) -> dict[str, Any]:
    """Load Kilosort4 `ops.npy` per-batch drift (`dshift`) trace for one sorted bank.

    Parameters
    ----------
    bank_dir : Path
        Path to `bankXX_ks4` containing `sorter_output/ops.npy`.
    fs_hz : float
        AP sampling rate in Hz (default 30,000 Hz).

    Returns
    -------
    dict[str, Any]
        Contains `t_s`, `dshift_um` (per-batch median across depth blocks),
        `dshift_centered_um`, `dshift_blocks_um`, and summary spread metrics
        (`drift_ptp_um`, `drift_sd_um`, `drift_mad_um`).
    """
    ops_path = Path(bank_dir) / "sorter_output" / "ops.npy"
    if not ops_path.exists():
        raise FileNotFoundError(f"Kilosort4 ops.npy not found: {ops_path}")

    ops = np.load(ops_path, allow_pickle=True).item()
    raw_dshift = ops.get("dshift")
    if raw_dshift is None:
        empty = np.empty(0, dtype=np.float64)
        return {
            "t_s": empty,
            "dshift_um": empty,
            "dshift_centered_um": empty,
            "dshift_blocks_um": np.empty((0, 1), dtype=np.float64),
            "drift_ptp_um": 0.0,
            "drift_sd_um": 0.0,
            "drift_mad_um": 0.0,
            "n_batches": 0,
        }

    arr = np.asarray(raw_dshift, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_batches = arr.shape[0]
    batch_samp = float(ops.get("batch_size", ops.get("nt", fs_hz)))
    dt_s = max(batch_samp / float(fs_hz), 1e-3)
    t_s = (np.arange(n_batches, dtype=np.float64) + 0.5) * dt_s

    med_trace = np.nanmedian(arr, axis=1) if n_batches else np.empty(0, dtype=np.float64)
    finite = med_trace[np.isfinite(med_trace)]
    if finite.size:
        med0 = float(np.median(finite))
        centered = med_trace - med0
        ptp = float(np.ptp(finite))
        sd = float(np.std(finite))
        mad = float(np.median(np.abs(finite - med0)))
    else:
        centered = med_trace.copy()
        ptp = sd = mad = 0.0

    return {
        "t_s": t_s,
        "dshift_um": med_trace,
        "dshift_centered_um": centered,
        "dshift_blocks_um": arr,
        "drift_ptp_um": ptp,
        "drift_sd_um": sd,
        "drift_mad_um": mad,
        "n_batches": int(n_batches),
    }


def _load_neural_cluster_ids(bank_dir: Path, p_neural_min: float) -> set[int] | None:
    """Resolve curated neural cluster IDs from `labels_custom.csv` if available."""
    cand_paths = [
        bank_dir / "analyzer_cache" / "extensions" / "auto_curation" / "labels_custom.csv",
        bank_dir / "sorter_output" / "labels_custom.csv",
    ]
    for lp in cand_paths:
        if lp.exists():
            df = pd.read_csv(lp)
            if "custom_p_neural" in df.columns and "unit_id" in df.columns:
                mask = df["custom_p_neural"].astype(float) >= float(p_neural_min)
                return set(df.loc[mask, "unit_id"].astype(int).tolist())
    return None


def extract_bank_spike_cloud(
    bank_dir: Path,
    bank_index: int,
    bank_step_um: float = 3840.0,
    p_neural_min: float | None = 0.5,
    fs_hz: float = 30000.0,
    max_spikes: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Extract global-depth spike times, depths (µm), and normalized amplitudes for one bank.

    Local coordinates are normalized into ``[0, bank_step_um)`` before adding
    ``bank_index * bank_step_um`` so outputs remain aligned across surveys whose
    sorters stored either local bank coordinates or pre-offset global coordinates.
    """
    bdir = Path(bank_dir)
    sdir = bdir / "sorter_output"
    st_path = sdir / "spike_times.npy"
    if not st_path.exists():
        raise FileNotFoundError(f"Missing spike_times.npy in {sdir}")

    spike_times = np.load(st_path).reshape(-1).astype(np.int64)
    if spike_times.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "times_s": empty,
            "depth_um": empty,
            "amp_norm": empty,
            "duration_s": 0.0,
            "n_total_spikes": 0,
        }

    spike_clusters = np.load(sdir / "spike_clusters.npy").reshape(-1).astype(np.int64)
    amps = np.abs(np.load(sdir / "amplitudes.npy").reshape(-1).astype(np.float64))

    sp_pos_path = sdir / "spike_positions.npy"
    if sp_pos_path.exists():
        sp_pos = np.load(sp_pos_path)
        y_raw = np.asarray(sp_pos[:, 1], dtype=np.float64)
    else:
        spike_templates = np.load(sdir / "spike_templates.npy").reshape(-1).astype(np.int64)
        chan_pos = np.load(sdir / "channel_positions.npy")
        templates = np.load(sdir / "templates.npy")
        peak_ch = np.argmax(np.ptp(templates, axis=1), axis=1)
        y_raw = np.asarray(chan_pos[peak_ch[spike_templates], 1], dtype=np.float64)

    n = min(spike_times.size, spike_clusters.size, amps.size, y_raw.size)
    spike_times = spike_times[:n]
    spike_clusters = spike_clusters[:n]
    amps = amps[:n]
    y_raw = y_raw[:n]

    if p_neural_min is not None:
        neural_ids = _load_neural_cluster_ids(bdir, p_neural_min)
        if neural_ids is not None:
            if not neural_ids:
                empty = np.empty(0, dtype=np.float64)
                return {
                    "times_s": empty,
                    "depth_um": empty,
                    "amp_norm": empty,
                    "duration_s": float(spike_times.max() / fs_hz),
                    "n_total_spikes": 0,
                }
            keep = np.isin(spike_clusters, np.fromiter(neural_ids, dtype=np.int64))
            spike_times = spike_times[keep]
            amps = amps[keep]
            y_raw = y_raw[keep]

    if spike_times.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "times_s": empty,
            "depth_um": empty,
            "amp_norm": empty,
            "duration_s": 0.0,
            "n_total_spikes": 0,
        }

    # Normalize coordinates once into global bank depth to avoid double-offsetting.
    offset_um = float(bank_index) * float(bank_step_um)
    med_y = float(np.nanmedian(y_raw))
    if bank_index > 0 and med_y >= 0.75 * offset_um:
        y_local = y_raw - offset_um
    elif bank_index == 0 and med_y >= 1.5 * bank_step_um:
        y_local = np.mod(y_raw, bank_step_um)
    else:
        y_local = y_raw
    depth_um = y_local + offset_um

    times_s = spike_times.astype(np.float64) / float(fs_hz)
    duration_s = float(np.max(times_s))
    n_total = int(times_s.size)

    p99 = float(np.percentile(amps, 99)) if amps.size else 1.0
    amp_norm = np.clip(amps / max(p99, 1e-6), 0.0, 1.0)

    if max_spikes is not None and n_total > int(max_spikes):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=int(max_spikes), replace=False))
        times_s = times_s[idx]
        depth_um = depth_um[idx]
        amp_norm = amp_norm[idx]

    return {
        "times_s": times_s,
        "depth_um": depth_um,
        "amp_norm": amp_norm,
        "duration_s": duration_s,
        "n_total_spikes": n_total,
    }


def _safe_pearson_overlap(a: np.ndarray, b: np.ndarray, min_overlap: int) -> float:
    """Normalized correlation over jointly finite overlapping elements."""
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < min_overlap:
        return np.nan
    va = a[mask] - float(np.mean(a[mask]))
    vb = b[mask] - float(np.mean(b[mask]))
    denom = float(np.sqrt(np.dot(va, va) * np.dot(vb, vb)))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def register_1d_profiles_across_sessions(
    profiles_2d: np.ndarray,
    pitch_um: float = 20.0,
    max_disp_um: float = 600.0,
    min_corr: float = 0.15,
    ref_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Globally align 1D depth profiles across longitudinal sessions via weighted pairwise xcorr.

    Positive displacement ``displacement_um[j] > displacement_um[i]`` indicates that
    session ``j``'s spatial landmarks appear shifted toward larger coordinate values
    relative to session ``i`` (i.e. ``profile[j, y + disp_j] ≈ profile[i, y + disp_i]``).
    """
    prof = np.asarray(profiles_2d, dtype=np.float64)
    if prof.ndim != 2:
        raise ValueError(f"Expected 2D array (n_sessions, n_depths), got shape {prof.shape}")
    n_sess, n_depths = prof.shape
    max_lag_bins = max(1, int(round(float(max_disp_um) / float(pitch_um))))
    min_overlap = max(20, n_depths // 10)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1, dtype=np.int64)

    valid_sess = np.array([int(np.isfinite(prof[s]).sum()) >= min_overlap for s in range(n_sess)], dtype=bool)
    pair_lags = np.full((n_sess, n_sess), np.nan, dtype=np.float64)
    pair_corrs = np.full((n_sess, n_sess), np.nan, dtype=np.float64)
    for s in range(n_sess):
        if valid_sess[s]:
            pair_lags[s, s] = 0.0
            pair_corrs[s, s] = 1.0

    for i in range(n_sess):
        if not valid_sess[i]:
            continue
        pi = prof[i]
        for j in range(i + 1, n_sess):
            if not valid_sess[j]:
                continue
            pj = prof[j]
            c_arr = np.full(lags.size, np.nan, dtype=np.float64)
            for li, k in enumerate(lags):
                if k >= 0:
                    a = pi[: n_depths - k]
                    b = pj[k:]
                else:
                    a = pi[-k:]
                    b = pj[: n_depths + k]
                c_arr[li] = _safe_pearson_overlap(a, b, min_overlap)
            if not np.any(np.isfinite(c_arr)):
                continue
            best_idx = int(np.nanargmax(c_arr))
            r_peak = float(c_arr[best_idx])
            sub_bin = 0.0
            if 0 < best_idx < lags.size - 1:
                c0, c1, c2 = c_arr[best_idx - 1], c_arr[best_idx], c_arr[best_idx + 1]
                if np.isfinite(c0) and np.isfinite(c1) and np.isfinite(c2):
                    denom = c0 - 2.0 * c1 + c2
                    if denom < -1e-9:
                        sub_bin = float(np.clip(0.5 * (c0 - c2) / denom, -0.5, 0.5))
            lag_um = float(lags[best_idx] + sub_bin) * float(pitch_um)
            pair_lags[i, j] = lag_um
            pair_lags[j, i] = -lag_um
            pair_corrs[i, j] = pair_corrs[j, i] = r_peak

    disp = np.full(n_sess, np.nan, dtype=np.float64)
    active_idx = np.flatnonzero(valid_sess)
    n_act = active_idx.size
    if n_act > 0:
        lap = np.zeros((n_act, n_act), dtype=np.float64)
        rhs = np.zeros(n_act, dtype=np.float64)
        for a_i, si in enumerate(active_idx):
            for a_j in range(a_i + 1, n_act):
                sj = active_idx[a_j]
                r_val = pair_corrs[si, sj]
                l_val = pair_lags[si, sj]
                if not (np.isfinite(r_val) and np.isfinite(l_val)) or r_val <= min_corr:
                    continue
                w = float((r_val - min_corr) ** 2)
                lap[a_i, a_i] += w
                lap[a_j, a_j] += w
                lap[a_i, a_j] -= w
                lap[a_j, a_i] -= w
                rhs[a_i] -= w * l_val
                rhs[a_j] += w * l_val
        # Anchor gauge constraint so sparse disconnected rows cannot diverge.
        lap += 1e-6 * np.eye(n_act) + 1e-4 * np.ones((n_act, n_act))
        z_act = np.linalg.solve(lap, rhs)
        disp[active_idx] = z_act

        if ref_mask is not None:
            rm = np.asarray(ref_mask, dtype=bool) & valid_sess & np.isfinite(disp)
            offset = float(np.nanmedian(disp[rm])) if np.any(rm) else float(np.nanmedian(disp[active_idx]))
        else:
            offset = float(np.nanmedian(disp[active_idx]))
        disp[active_idx] -= offset

    return {
        "displacement_um": disp,
        "pairwise_lags_um": pair_lags,
        "pairwise_corrs": pair_corrs,
    }


def fit_chronic_settling_trajectory(
    days: np.ndarray,
    pos_um: np.ndarray,
    plateau_min_day: float = 35.0,
) -> dict[str, Any]:
    """Fit exponential relaxation + chronic plateau stability metrics on longitudinal drift.

    Centers ``pos_um`` to ``0.0 um`` at the ``days >= plateau_min_day`` median so early
    post-surgical settling and late-chronic plateau scatter share a common reference frame.
    """
    d_arr = np.asarray(days, dtype=np.float64)
    p_arr = np.asarray(pos_um, dtype=np.float64)
    if d_arr.shape != p_arr.shape:
        raise ValueError(f"Shape mismatch: days {d_arr.shape} vs pos_um {p_arr.shape}")

    fin = np.isfinite(d_arr) & np.isfinite(p_arr)
    centered = np.full_like(p_arr, np.nan, dtype=np.float64)
    fit_curve = np.full_like(p_arr, np.nan, dtype=np.float64)
    if int(fin.sum()) < 4:
        return {
            "n_sessions": int(fin.sum()),
            "centered_um": centered,
            "fit_um": fit_curve,
        }

    order = np.argsort(d_arr[fin])
    d_fin = d_arr[fin][order]
    p_fin = p_arr[fin][order]
    plat_mask = d_fin >= float(plateau_min_day)
    ref_med = float(np.median(p_fin[plat_mask])) if np.any(plat_mask) else float(np.median(p_fin))
    c_fin = p_fin - ref_med
    centered[fin] = p_arr[fin] - ref_med

    d0 = float(d_fin[0])

    def _exp_model(x: np.ndarray, y_inf: float, amp: float, tau: float) -> np.ndarray:
        return y_inf - amp * np.exp(-(x - d0) / tau)

    p0 = (0.0, float(-c_fin[0]), 12.0)
    try:
        popt, _ = optimize.curve_fit(
            _exp_model,
            d_fin,
            c_fin,
            p0=p0,
            bounds=([-250.0, -2500.0, 1.0], [250.0, 2500.0, 250.0]),
            maxfev=5000,
        )
        y_inf, amp, tau = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        y_inf, amp, tau = 0.0, float(-c_fin[0]), 15.0

    y_hat_fin = _exp_model(d_fin, y_inf, amp, tau)
    fit_curve[fin] = _exp_model(d_arr[fin], y_inf, amp, tau)
    ss_res = float(np.sum((c_fin - y_hat_fin) ** 2))
    ss_tot = float(np.sum((c_fin - np.mean(c_fin)) ** 2))
    fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    fit_rmse = float(np.sqrt(ss_res / max(d_fin.size, 1)))

    plat_vals = c_fin[plat_mask] if np.any(plat_mask) else c_fin
    plat_days = d_fin[plat_mask] if np.any(plat_mask) else d_fin
    plat_med = float(np.median(plat_vals))
    q25, q75 = np.percentile(plat_vals, [25.0, 75.0])
    step_diffs = np.abs(np.diff(plat_vals))
    if plat_days.size >= 3 and np.ptp(plat_days) > 0:
        lr = stats.linregress(plat_days, plat_vals)
        slope_30d = float(lr.slope * 30.0)
        slope_p = float(lr.pvalue)
    else:
        slope_30d, slope_p = 0.0, 1.0

    return {
        "n_sessions": int(d_fin.size),
        "n_plateau_sessions": int(plat_vals.size),
        "day_start": d0,
        "day_end": float(d_fin[-1]),
        "plateau_ref_um": ref_med,
        "initial_observed_um": float(c_fin[0]),
        "total_excursion_um": float(np.ptp(c_fin)),
        "exp_y_inf_um": y_inf,
        "exp_amp_um": amp,
        "exp_start_to_asymp_um": float(y_hat_fin[0] - y_inf),
        "tau_days": tau,
        "t95_days": float(d0 + tau * np.log(20.0)),
        "fit_r2": float(fit_r2),
        "fit_rmse_um": fit_rmse,
        "plateau_iqr_um": float(q75 - q25),
        "plateau_mad_um": float(np.median(np.abs(plat_vals - plat_med))),
        "plateau_rms_um": float(np.sqrt(np.mean(plat_vals ** 2))),
        "plateau_range_um": float(np.ptp(plat_vals)),
        "step_abs_median_um": float(np.median(step_diffs)) if step_diffs.size else 0.0,
        "plateau_slope_um_per_30d": slope_30d,
        "plateau_slope_p": slope_p,
        "centered_um": centered,
        "fit_um": fit_curve,
    }
