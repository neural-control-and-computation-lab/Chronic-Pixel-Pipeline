"""Reusable Matplotlib plotting primitives for 2-column footprints, CCGs, and scaling curves."""

from __future__ import annotations

import matplotlib.axes as maxes
import numpy as np


def add_scalebar(
    ax: maxes.Axes,
    x0: float,
    y0: float,
    dx: float = 1.0,
    dy: float = 100.0,
    x_label: str = "1 ms",
    y_label: str = "100 uV",
    color: str = "#111111",
    fontsize: float = 6.5,
) -> None:
    """Draw clean L-shaped time/voltage scalebar on a trace or footprint axis."""
    ax.plot([x0, x0 + dx], [y0, y0], color=color, linewidth=1.2)
    ax.plot([x0, x0], [y0, y0 + dy], color=color, linewidth=1.2)
    if x_label:
        ax.text(x0 + 0.5 * dx, y0 - 0.15 * abs(dy), x_label, ha="center", va="top", fontsize=fontsize, color=color)
    if y_label:
        ax.text(x0 - 0.20 * abs(dx), y0 + 0.5 * dy, y_label, ha="right", va="center", fontsize=fontsize, color=color, rotation=90)


def plot_2col_footprint(
    ax: maxes.Axes,
    waveforms_uv: np.ndarray,
    positions_um: np.ndarray,
    color: str = "#1F4E9B",
    linestyle: str = "-",
    linewidth: float = 1.1,
    alpha: float = 0.92,
    amp_scale: float | None = None,
    x_width_um: float = 65.0,
    x_col_sep_um: float = 85.0,
    label: str | None = None,
    fs_hz: float = 30000.0,
) -> float:
    """Overlay a unit's 2-column spatial waveform footprint on an axis and return scale used."""
    wf = np.asarray(waveforms_uv, dtype=np.float64)
    pos = np.asarray(positions_um, dtype=np.float64)
    n_samp, n_ch = wf.shape
    if n_ch == 0:
        return 1.0

    if amp_scale is None:
        max_abs = float(np.max(np.abs(wf)))
        unique_y = np.sort(np.unique(np.round(pos[:, 1], 1)))
        dy = float(np.min(np.diff(unique_y))) if unique_y.size > 1 else 20.0
        amp_scale = (0.85 * dy) / max(max_abs, 1e-3)

    t = np.linspace(-0.5 * x_width_um, 0.5 * x_width_um, n_samp)
    ux = np.sort(np.unique(np.round(pos[:, 0], 1)))
    if ux.size > 1:
        x_mid = float(0.5 * (ux[0] + ux[-1]))
        x_disp = np.where(pos[:, 0] < x_mid, -0.5 * x_col_sep_um, 0.5 * x_col_sep_um)
    else:
        x_disp = np.zeros(n_ch, dtype=np.float64)

    for ci in range(n_ch):
        ax.plot(
            x_disp[ci] + t,
            pos[ci, 1] + wf[:, ci] * amp_scale,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label if ci == 0 else None,
        )
    _ = fs_hz
    return float(amp_scale)


def plot_shank_waveforms(
    ax: maxes.Axes,
    footprints: list[dict[str, np.ndarray]],
    colors: list[str],
    x_jitter_um: list[float] | None = None,
    linewidth: float = 0.7,
    alpha: float = 0.85,
) -> None:
    """Plot multiple 2-column or peak-channel waveforms at physical depth along a shank axis."""
    for idx, fp in enumerate(footprints):
        wf = np.asarray(fp["waveforms_uv"], dtype=np.float64)
        pos = np.asarray(fp["positions_um"], dtype=np.float64)
        dx = float(x_jitter_um[idx]) if x_jitter_um is not None else 0.0
        max_abs = max(float(np.max(np.abs(wf))), 1e-3)
        scale_y = 140.0 / max_abs
        t = np.linspace(-0.4, 0.4, wf.shape[0])
        for c in range(wf.shape[1]):
            ax.plot(dx + t, pos[c, 1] + wf[:, c] * scale_y, color=colors[idx], linewidth=linewidth, alpha=alpha)


def plot_ccg_with_flanks(
    ax: maxes.Axes,
    lags_ms: np.ndarray,
    ccg_jc: np.ndarray,
    z_thresh: float = 7.0,
    flank_min_ms: float = 25.0,
    bar_color: str = "#1F4E9B",
    title: str = "",
) -> None:
    """Plot a jitter-corrected cross-correlogram with ±z_thresh flanking noise band."""
    lags = np.asarray(lags_ms, dtype=np.float64)
    vals = np.asarray(ccg_jc, dtype=np.float64)
    flank = vals[np.abs(lags) >= float(flank_min_ms)]
    sd = float(np.std(flank)) if flank.size else 0.0
    mu = float(np.mean(flank)) if flank.size else 0.0

    ax.bar(lags, vals, width=0.85, color=bar_color, edgecolor="none", alpha=0.9)
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.7)
    ax.axhline(mu + z_thresh * sd, color="#C44E52", linestyle=":", linewidth=0.9, label=f"+{z_thresh:g} SD")
    if np.min(vals) < mu - 3.0 * sd:
        ax.axhline(mu - z_thresh * sd, color="#C44E52", linestyle=":", linewidth=0.9)
    if title:
        ax.set_title(title, fontsize=7.5)


def plot_neuron_dropping(
    ax: maxes.Axes,
    df_curve,
    color: str = "#1F4E9B",
    label: str | None = None,
) -> None:
    """Plot Trautmann-style neuron-dropping curve (median ± IQR shading)."""
    x = df_curve["n_neurons"].to_numpy(dtype=float)
    y = df_curve["mean_r2"].to_numpy(dtype=float)
    lo = df_curve["iqr_low"].to_numpy(dtype=float)
    hi = df_curve["iqr_high"].to_numpy(dtype=float)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.3, color=color, label=label)
