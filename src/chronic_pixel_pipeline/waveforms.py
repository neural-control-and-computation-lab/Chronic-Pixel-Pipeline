"""Spatial 2-column waveform footprints, biophysical width classification, and spike train metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def extract_2col_spatial_footprint(
    templates_3d: np.ndarray,
    channel_positions_um: np.ndarray,
    unit_index: int,
    n_rows: int = 7,
) -> dict[str, Any]:
    """Extract dense 2-column multichannel waveform footprint centered on a unit's peak channel.

    Parameters
    ----------
    templates_3d : np.ndarray
        Unit template array of shape ``(n_units, n_samples, n_channels)`` in µV.
    channel_positions_um : np.ndarray
        Electrode coordinates ``(n_channels, 2)`` in µm.
    unit_index : int
        Row index into ``templates_3d``.
    n_rows : int
        Number of vertical pitch rows (up to ``2 * n_rows`` channels across both columns).

    Returns
    -------
    dict[str, Any]
        Contains ``waveforms_uv`` ``(n_samples, n_sel_ch)``, ``positions_um`` ``(n_sel_ch, 2)``,
        ``channel_indices``, ``peak_channel``, ``ptp_uv``, and ``centroid_xy_um``.
    """
    tmpl = np.asarray(templates_3d[unit_index], dtype=np.float64)
    pos = np.asarray(channel_positions_um, dtype=np.float64)
    if tmpl.shape[1] != pos.shape[0]:
        raise ValueError(f"Template channels ({tmpl.shape[1]}) != position rows ({pos.shape[0]})")

    ptp_all = np.ptp(tmpl, axis=0)
    peak_ch = int(np.argmax(ptp_all))
    peak_ptp = float(ptp_all[peak_ch])
    peak_y = float(pos[peak_ch, 1])

    unique_y = np.sort(np.unique(np.round(pos[:, 1], 2)))
    chosen_y = np.sort(unique_y[np.argsort(np.abs(unique_y - peak_y))[: max(1, int(n_rows))]])

    sel_idx: list[int] = []
    for yv in chosen_y:
        matches = np.flatnonzero(np.abs(pos[:, 1] - yv) <= 1.5)
        if matches.size:
            order = matches[np.argsort(pos[matches, 0])]
            sel_idx.extend(int(c) for c in order)

    sel_arr = np.asarray(sel_idx, dtype=np.int64)
    active_mask = ptp_all >= max(0.25 * peak_ptp, 1e-6)
    weights = np.square(ptp_all[active_mask])
    if np.sum(weights) > 0:
        centroid = np.sum(pos[active_mask] * weights[:, None], axis=0) / np.sum(weights)
    else:
        centroid = pos[peak_ch].copy()

    return {
        "waveforms_uv": tmpl[:, sel_arr],
        "positions_um": pos[sel_arr],
        "channel_indices": sel_arr,
        "peak_channel": peak_ch,
        "ptp_uv": peak_ptp,
        "centroid_xy_um": centroid,
    }


def compute_waveform_celltype_metrics(
    templates_3d: np.ndarray,
    channel_positions_um: np.ndarray | None = None,
    template_metrics_df: pd.DataFrame | None = None,
    fs_hz: float = 30000.0,
    narrow_cutoff_ms: float = 0.42,
    upsample: int = 10,
) -> pd.DataFrame:
    """Compute upsampled waveform width, spread, and putative narrow/broad cell class.

    Uses 10x cubic-spline upsampling on the peak-amplitude channel to extract a uniform
    trough-to-peak duration (`trough_to_peak_ms`) independent of sorter metadata scaling.
    """
    arr = np.asarray(templates_3d, dtype=np.float64)
    n_units, n_samp, n_ch = arr.shape
    t_orig = np.arange(n_samp, dtype=np.float64) * (1000.0 / float(fs_hz))
    t_fine = np.linspace(t_orig[0], t_orig[-1], (n_samp - 1) * int(upsample) + 1)
    dt_fine_ms = float(t_fine[1] - t_fine[0]) if t_fine.size > 1 else (1000.0 / float(fs_hz))

    y_pos = (
        np.asarray(channel_positions_um[:, 1], dtype=np.float64)
        if channel_positions_um is not None and len(channel_positions_um) == n_ch
        else np.arange(n_ch, dtype=np.float64) * 20.0
    )

    rows: list[dict[str, Any]] = []
    n_pre = max(3, min(15, n_samp // 6))
    for u in range(n_units):
        tmpl = arr[u]
        ptp_ch = np.ptp(tmpl, axis=0)
        peak_ch = int(np.argmax(ptp_ch))
        ptp_uv = float(ptp_ch[peak_ch])

        wf = tmpl[:, peak_ch]
        wf = wf - float(np.median(wf[:n_pre]))
        cs = CubicSpline(t_orig, wf)
        wf_fine = cs(t_fine)

        min_idx = int(np.argmin(wf_fine))
        max_idx = int(np.argmax(wf_fine))
        min_v = float(wf_fine[min_idx])
        max_v = float(wf_fine[max_idx])
        is_positive = bool(max_v > abs(min_v) * 1.15)

        if is_positive:
            dom_idx = max_idx
            post_slice = wf_fine[dom_idx:]
            rep_rel = int(np.argmin(post_slice)) if post_slice.size else 0
            ttp_ms = float(rep_rel * dt_fine_ms)
            half_lvl = 0.5 * wf_fine[dom_idx]
            above = np.flatnonzero(wf_fine >= half_lvl)
        else:
            dom_idx = min_idx
            post_slice = wf_fine[dom_idx:]
            rep_rel = int(np.argmax(post_slice)) if post_slice.size else 0
            ttp_ms = float(rep_rel * dt_fine_ms)
            half_lvl = 0.5 * wf_fine[dom_idx]
            above = np.flatnonzero(wf_fine <= half_lvl)

        # Contiguous half-width around the dominant extremum
        if above.size >= 2:
            half_w_ms = float((above[-1] - above[0]) * dt_fine_ms)
        else:
            half_w_ms = dt_fine_ms

        ratio = float(abs(max_v) / max(abs(min_v), 1e-6)) if not is_positive else float(abs(min_v) / max(abs(max_v), 1e-6))

        if ptp_uv > 0:
            active_y = y_pos[ptp_ch >= 0.25 * ptp_uv]
            spread_um = float(np.max(active_y) - np.min(active_y)) if active_y.size >= 2 else 0.0
        else:
            spread_um = 0.0

        cell_class = "narrow" if ttp_ms < float(narrow_cutoff_ms) else "broad"
        rows.append(
            {
                "unit_idx": int(u),
                "peak_channel": peak_ch,
                "ptp_uv": ptp_uv,
                "trough_to_peak_ms": ttp_ms,
                "half_width_ms": half_w_ms,
                "peak_trough_ratio": ratio,
                "spread_um": spread_um,
                "is_positive_spike": is_positive,
                "cell_class": cell_class,
            }
        )

    out = pd.DataFrame(rows)
    if template_metrics_df is not None and len(template_metrics_df) == n_units:
        for extra_col in ("repolarization_slope", "recovery_slope", "velocity_above", "velocity_below"):
            if extra_col in template_metrics_df.columns:
                out[f"si_{extra_col}"] = template_metrics_df[extra_col].to_numpy(dtype=np.float64)
    return out


def compute_spike_train_biophysics(
    spike_samples: np.ndarray,
    fs_hz: float = 30000.0,
    valid_duration_s: float | None = None,
    acg_half_ms: int = 25,
) -> dict[str, Any]:
    """Compute firing rate, short-ISI burst index, local CV2, and ±25 ms autocorrelogram."""
    sp = np.sort(np.asarray(spike_samples, dtype=np.int64).reshape(-1))
    times_s = sp.astype(np.float64) / float(fs_hz)
    n_sp = int(times_s.size)

    if valid_duration_s is not None and valid_duration_s > 0:
        dur_s = float(valid_duration_s)
    elif n_sp >= 2:
        dur_s = max(float(times_s[-1] - times_s[0]), 1e-3)
    else:
        dur_s = 1.0

    fr_hz = float(n_sp / max(dur_s, 1e-6))
    half_s = float(acg_half_ms) / 1000.0
    n_bins = 2 * int(acg_half_ms)
    bin_edges = np.linspace(-float(acg_half_ms), float(acg_half_ms), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if n_sp < 2:
        return {
            "n_spikes": n_sp,
            "firing_rate_hz": fr_hz,
            "burst_index_10ms": 0.0,
            "cv2": float("nan"),
            "acg_bins_ms": bin_centers,
            "acg_hz": np.zeros(n_bins, dtype=np.float64),
        }

    isis_ms = np.diff(times_s) * 1000.0
    sub500 = isis_ms[(isis_ms > 0.0) & (isis_ms < 500.0)]
    burst_idx = float(np.mean(sub500 < 10.0)) if sub500.size else 0.0

    pos_isis = isis_ms[isis_ms > 0.0]
    if pos_isis.size >= 2:
        a = pos_isis[:-1]
        b = pos_isis[1:]
        cv2 = float(np.mean(2.0 * np.abs(b - a) / np.maximum(a + b, 1e-9)))
    else:
        cv2 = float("nan")

    # Fast exact positive-lag pair scan, mirrored for symmetric ±25 ms ACG
    counts_pos = np.zeros(int(acg_half_ms), dtype=np.int64)
    right = 0
    for i in range(n_sp):
        ti = times_s[i]
        limit = ti + half_s
        while right < n_sp and times_s[right] < limit:
            right += 1
        j = i + 1
        while j < right:
            dt_ms = (times_s[j] - ti) * 1000.0
            b_idx = int(dt_ms)
            if 0 <= b_idx < int(acg_half_ms):
                counts_pos[b_idx] += 1
            j += 1

    sym_counts = np.concatenate([counts_pos[::-1], counts_pos]).astype(np.float64)
    acg_hz = sym_counts / (float(n_sp) * 0.001)

    return {
        "n_spikes": n_sp,
        "firing_rate_hz": fr_hz,
        "burst_index_10ms": burst_idx,
        "cv2": cv2,
        "acg_bins_ms": bin_centers,
        "acg_hz": acg_hz,
    }
