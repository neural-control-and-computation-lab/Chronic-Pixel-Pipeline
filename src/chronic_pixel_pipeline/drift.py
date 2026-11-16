"""Single-spike drift clouds and Kilosort4 within-session drift trajectory utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
