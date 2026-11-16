"""Depth-resolved bipolar LFP spectrograms and 1/f aperiodic parameterization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch

BAND_DEFINITIONS_HZ: dict[str, tuple[float, float]] = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "low_gamma_30_55": (30.0, 55.0),
    "high_gamma_65_100": (65.0, 100.0),
}


def compute_bipolar_depth_psd_map(
    lf_bin_path: Path,
    sample_start: int,
    sample_end: int,
    n_saved_ch: int = 385,
    n_active_ch: int = 384,
    fs_hz: float = 2500.0,
    uv_per_bit: float = 2.34375,
    win_s: float = 2.0,
    min_valid_wins: int = 4,
    abs_max_uv: float = 1100.0,
    ptp_min_uv: float = 5.0,
    freq_max_hz: float = 100.0,
) -> dict[str, Any]:
    """Compute range-4 artifact-screened within-column bipolar LFP PSD across 191 bank rows.

    Re-references adjacent vertical electrodes in each column (even ``0..380`` vs ``2..382``;
    odd ``1..381`` vs ``3..383``) to eliminate distant reference contamination before running
    Welch spectral estimation (1 Hz resolution) with 57-63 Hz mains notch interpolation.
    """
    src = Path(lf_bin_path)
    if not src.exists():
        raise FileNotFoundError(f"LFP binary file not found: {src}")

    total_samples = src.stat().st_size // (int(n_saved_ch) * 2)
    s0 = max(0, int(sample_start))
    s1 = min(total_samples, int(sample_end))
    mm = np.memmap(src, dtype="<i2", mode="r", shape=(total_samples, int(n_saved_ch)))
    seg = np.asarray(mm[s0:s1, :n_active_ch], dtype=np.float32) * float(uv_per_bit)

    even_bip = seg[:, 2:n_active_ch:2] - seg[:, 0 : n_active_ch - 2 : 2]
    odd_bip = seg[:, 3:n_active_ch:2] - seg[:, 1 : n_active_ch - 2 : 2]
    n_rows = even_bip.shape[1]  # 191 rows for 384 channels
    bipolar = np.concatenate([even_bip, odd_bip], axis=1)  # 382 bipolar pairs

    win_len = max(int(round(win_s * fs_hz)), 256)
    n_wins = bipolar.shape[0] // win_len
    freqs_full = np.fft.rfftfreq(min(int(round(fs_hz)), win_len), d=1.0 / fs_hz)
    f_sel = (freqs_full >= 1.0) & (freqs_full <= float(freq_max_hz))
    freqs_hz = freqs_full[f_sel]

    if n_wins < int(min_valid_wins):
        return {
            "z_local_um": np.arange(n_rows, dtype=np.float64) * 20.0 + 10.0,
            "freqs_hz": freqs_hz,
            "psd_depth_db": np.full((n_rows, freqs_hz.size), np.nan, dtype=np.float64),
            "band_powers_db": {k: np.full(n_rows, np.nan, dtype=np.float64) for k in BAND_DEFINITIONS_HZ},
            "aperiodic_exp": np.full(n_rows, np.nan, dtype=np.float64),
            "eligible_pairs_count": 0,
            "qc_pass": False,
        }

    trimmed = bipolar[: n_wins * win_len].reshape(n_wins, win_len, bipolar.shape[1])
    ptp = np.ptp(trimmed, axis=1)
    abs_max = np.max(np.abs(trimmed), axis=1)
    valid_mask = (abs_max < float(abs_max_uv)) & (ptp > float(ptp_min_uv))

    n_pairs = bipolar.shape[1]
    pair_psd = np.full((n_pairs, freqs_hz.size), np.nan, dtype=np.float64)
    eligible_pairs = 0

    nperseg = min(int(round(fs_hz)), win_len)
    for j in range(n_pairs):
        w_idx = np.flatnonzero(valid_mask[:, j])
        if w_idx.size < int(min_valid_wins):
            continue
        eligible_pairs += 1
        _, pxx = welch(trimmed[w_idx, :, j], fs=fs_hz, nperseg=nperseg, axis=1)
        mean_p = np.mean(pxx, axis=0)[f_sel]
        pair_psd[j] = mean_p

    # Average passing left + right columns at each physical vertical depth row
    psd_rows_lin = np.full((n_rows, freqs_hz.size), np.nan, dtype=np.float64)
    for r in range(n_rows):
        cands = [pair_psd[r], pair_psd[n_rows + r]]
        valid_cands = [c for c in cands if np.all(np.isfinite(c))]
        if valid_cands:
            psd_rows_lin[r] = np.mean(valid_cands, axis=0)

    # Interpolate across 57-63 Hz mains band in linear PSD before dB & 1/f fits
    notch_mask = (freqs_hz >= 57.0) & (freqs_hz <= 63.0)
    non_notch = ~notch_mask
    psd_rows_clean = psd_rows_lin.copy()
    for r in range(n_rows):
        if np.all(np.isfinite(psd_rows_clean[r])):
            psd_rows_clean[r, notch_mask] = np.interp(
                freqs_hz[notch_mask], freqs_hz[non_notch], psd_rows_clean[r, non_notch]
            )

    psd_depth_db = 10.0 * np.log10(np.maximum(psd_rows_clean, 1e-12))
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    assert trapz_fn is not None

    band_powers_db: dict[str, np.ndarray] = {}
    for bname, (f_lo, f_hi) in BAND_DEFINITIONS_HZ.items():
        bmask = (freqs_hz >= f_lo) & (freqs_hz <= f_hi)
        vals = np.full(n_rows, np.nan, dtype=np.float64)
        for r in range(n_rows):
            if np.all(np.isfinite(psd_rows_clean[r, bmask])):
                vals[r] = 10.0 * np.log10(max(float(trapz_fn(psd_rows_clean[r, bmask], freqs_hz[bmask])), 1e-12))
        band_powers_db[bname] = vals

    fit_mask = (freqs_hz >= 15.0) & (freqs_hz <= 95.0) & ~((freqs_hz >= 55.0) & (freqs_hz <= 65.0))
    log_f = np.log10(freqs_hz[fit_mask])
    aperiodic_exp = np.full(n_rows, np.nan, dtype=np.float64)
    for r in range(n_rows):
        if np.all(np.isfinite(psd_rows_clean[r, fit_mask])):
            slope, _ = np.polyfit(log_f, np.log10(np.maximum(psd_rows_clean[r, fit_mask], 1e-12)), 1)
            aperiodic_exp[r] = float(-slope)

    return {
        "z_local_um": np.arange(n_rows, dtype=np.float64) * 20.0 + 10.0,
        "freqs_hz": freqs_hz,
        "psd_depth_db": psd_depth_db,
        "band_powers_db": band_powers_db,
        "aperiodic_exp": aperiodic_exp,
        "eligible_pairs_count": int(eligible_pairs),
        "qc_pass": bool(eligible_pairs >= int(0.80 * n_pairs)),
    }
