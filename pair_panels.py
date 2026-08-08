"""
Shared panel-drawing helpers for tracked-pair composite figures. Layout code
(and workflow design agents) compose these onto a gridspec; the low-level
rendering stays consistent across variants.

Data is the .npz + .json produced by pair_supp_data.py.
"""
import json
from pathlib import Path

import numpy as np

# Neutral, colourblind-distinguishable pair (steel blue vs ochre).
COL_EARLY = '#436A8C'
COL_LATE = '#B5823C'
FS = 30000.0  # sampling rate (Hz), constant across NP1032 surveys


def load(stem):
    stem = Path(stem)
    arrs = dict(np.load(stem.with_suffix('.npz'), allow_pickle=True))
    meta = json.loads(stem.with_suffix('.json').read_text())
    return arrs, meta


def draw_overlay(ax, arrs, lw=1.7, n_rows_hint=7, scalebars=True,
                 uv_bar=100.0, ms_bar=1.0):
    """Multichannel template overlay, two sessions, no axes. Uses a virtual
    narrow column projection so the block is ~2x taller than wide and traces
    never overlap. Optional µV / ms scale bars at lower-left."""
    locs = arrs['keep_locs']
    t1 = arrs['tmpl1']
    t2 = arrs['tmpl2']
    n = t1.shape[0]
    amp_max = float(max(np.abs(t1).max(), np.abs(t2).max())) or 1.0
    uy = np.sort(np.unique(locs[:, 1]))
    y_pitch = float(np.min(np.diff(uy))) if len(uy) > 1 else 20.0
    n_rows = len(uy)
    x_pitch_v = ((n_rows * y_pitch) * 0.5) / 1.6
    ux = np.sort(np.unique(locs[:, 0]))
    if len(ux) > 1:
        x_virtual = np.where(np.isclose(locs[:, 0], ux[0]), 0.0, x_pitch_v)
    else:
        x_virtual = np.zeros(len(locs))
    amp_scale = (y_pitch * 0.85) / amp_max
    t_extent = x_pitch_v * 0.8
    t_scale = t_extent / (n - 1)
    t = np.arange(n) * t_scale - (n - 1) * t_scale / 2
    for tmpl, color in ((t1, COL_EARLY), (t2, COL_LATE)):
        for i in range(tmpl.shape[1]):
            ax.plot(float(x_virtual[i]) + t,
                    float(locs[i, 1]) + tmpl[:, i] * amp_scale,
                    color=color, lw=lw, alpha=0.85,
                    solid_capstyle='round', solid_joinstyle='round')

    if scalebars:
        # Anchor at lower-left of the block.
        x0 = float(x_virtual.min()) + t[0] - 0.18 * x_pitch_v
        y0 = float(locs[:, 1].min()) - 0.45 * y_pitch
        # Vertical µV bar
        ax.plot([x0, x0], [y0, y0 + uv_bar * amp_scale],
                color='#444444', lw=1.4, solid_capstyle='butt')
        ax.text(x0 - 0.04 * x_pitch_v, y0 + uv_bar * amp_scale / 2,
                f'{uv_bar:.0f} µV', rotation=90, fontsize=6.5,
                color='#444444', ha='right', va='center')
        # Horizontal ms bar
        ms_data = (ms_bar / 1000.0) * FS * t_scale
        ax.plot([x0, x0 + ms_data], [y0, y0],
                color='#444444', lw=1.4, solid_capstyle='butt')
        ax.text(x0 + ms_data / 2, y0 - 0.10 * y_pitch,
                f'{ms_bar:.0f} ms', fontsize=6.5, color='#444444',
                ha='center', va='top')

    ax.set_aspect('equal')
    ax.set_axis_off()


def draw_acg(ax, centers, acg, color, label=None, isi_pct=None, refr_ms=2.0):
    """Autocorrelogram bar plot. A clean refractory dip at 0 = real unit.
    Shades the ±refr_ms refractory window and annotates ISI-violation %."""
    width = (centers[1] - centers[0]) if len(centers) > 1 else 1.0
    ymax = max(1.0, acg.max() * 1.15)
    ax.axvspan(-refr_ms, refr_ms, color='#C0504D', alpha=0.10, linewidth=0)
    ax.bar(centers, acg, width=width, color=color, alpha=0.85, linewidth=0)
    ax.axvline(0, color='#999999', lw=0.5, alpha=0.6)
    ax.set_xlim(centers[0], centers[-1])
    ax.set_ylim(0, ymax)
    ax.set_xlabel('lag (ms)', fontsize=7)
    ax.tick_params(labelsize=6, length=2)
    ax.set_yticks([])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    if isi_pct is not None:
        ax.text(0.97, 0.93, f'ISI viol {isi_pct:.1f}%', fontsize=6.5,
                color='#C0504D', ha='right', va='top', transform=ax.transAxes)
    if label:
        ax.set_title(label, fontsize=7, color=color, pad=2)


def draw_amp(ax, t, v, color, label=None):
    """Per-spike amplitude vs time — a flat band = stable, non-noise unit."""
    ax.scatter(t, v, s=2.0, c=color, alpha=0.4, linewidths=0, rasterized=True)
    if len(v) > 5:
        lo, hi = np.percentile(v, [1, 99])
        pad = (hi - lo) * 0.3 + 1
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, t.max() if len(t) else 1)
    ax.set_xlabel('time (s)', fontsize=7)
    ax.set_ylabel('amp', fontsize=7)
    ax.tick_params(labelsize=6, length=2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if label:
        ax.set_title(label, fontsize=7, color=color, pad=2)


def metrics_lines(m):
    """Human-readable quality summary for a unit."""
    return [
        f"SNR {m['snr']:.1f}",
        f"{m['firing_rate']:.1f} Hz · {int(m['num_spikes'])} spk",
        f"ISI viol {m['isi_violations_ratio']*100:.1f}%",
        f"refrac {m['rp_contamination']*100:.1f}%",
        f"amp {abs(m['amplitude_median']):.0f} uV",
    ]


def draw_metrics(ax, m, color, header=None):
    ax.set_axis_off()
    y = 1.0
    if header:
        ax.text(0.0, y, header, fontsize=8.5, color=color,
                fontweight='bold', va='top', transform=ax.transAxes)
        y -= 0.18
    for line in metrics_lines(m):
        ax.text(0.0, y, line, fontsize=7.5, color='#333333', va='top',
                transform=ax.transAxes)
        y -= 0.15


def day_label(meta, which):
    d = meta['day1'] if which == 1 else meta['day2']
    sk = meta['survey1'] if which == 1 else meta['survey2']
    pretty = f"{sk[:4]}-{sk[4:6]}-{sk[6:]}"
    return f"Day {d}  ({pretty})"


def draw_legend(ax, meta):
    ax.set_axis_off()
    ax.text(0.0, 0.7, day_label(meta, 1), fontsize=10, color=COL_EARLY,
            fontweight='bold', va='center', transform=ax.transAxes)
    ax.text(0.0, 0.3, day_label(meta, 2), fontsize=10, color=COL_LATE,
            fontweight='bold', va='center', transform=ax.transAxes)
