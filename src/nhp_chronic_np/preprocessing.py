"""Signal preprocessing, matched-window slicing, and curated QC interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

from nhp_chronic_np.config import GOVERNANCE_CONTROLS, SAMPLE_RATE_HZ


def matched_window_sample_count(sec_per_bank: float = 40.0, fs_hz: float = SAMPLE_RATE_HZ) -> int:
    """Return exact sample count for early-vs-late duration-sensitivity checks (`40 s`)."""
    return int(round(float(sec_per_bank) * float(fs_hz)))


def build_discontinuity_exclusion_mask(
    n_packets: int,
    gap_packet_indices: np.ndarray | list[int],
    margin_s: float = 0.5,
    fs_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Return boolean keep-mask excluding `±margin_s` around wireless packet drops."""
    keep = np.ones(int(n_packets), dtype=bool)
    half_win = int(round(float(margin_s) * float(fs_hz)))
    for idx in gap_packet_indices:
        lo = max(0, int(idx) - half_win)
        hi = min(int(n_packets), int(idx) + half_win + 1)
        keep[lo:hi] = False
    return keep


def bandpass_and_cmr(
    traces_uv: np.ndarray,
    fs_hz: float = SAMPLE_RATE_HZ,
    low_hz: float = 300.0,
    high_hz: float = 6000.0,
    order: int = 3,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass followed by median common-average reference."""
    arr = np.asarray(traces_uv, dtype=np.float32)
    nyq = 0.5 * fs_hz
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    filt = filtfilt(b, a, arr, axis=0)
    median_ref = np.median(filt, axis=1, keepdims=True)
    return (filt - median_ref).astype(np.float32, copy=False)


def estimate_channel_mad_uv(traces_uv: np.ndarray) -> np.ndarray:
    """Robust per-channel RMS noise proxy (`MAD / 0.67449`) in microvolts."""
    arr = np.asarray(traces_uv, dtype=np.float64)
    med = np.median(arr, axis=0, keepdims=True)
    mad = np.median(np.abs(arr - med), axis=0)
    return mad / 0.67448975


@dataclass(frozen=True)
class HarmonizedClassifierPolicy:
    """Enforces frozen governance rules for the 48-site harmonized unit QC model."""

    n_spatial_features: int = 48
    adopted_model: bool = bool(GOVERNANCE_CONTROLS["adopted_model"])
    accepted_merges: int = int(GOVERNANCE_CONTROLS["accepted_merges"])
    masks_changed: bool = bool(GOVERNANCE_CONTROLS["masks_changed"])

    def validate_metrics(self, bal_acc: float, roc_auc: float) -> dict[str, Any]:
        """Package classifier benchmark metrics with mandatory non-adoption controls."""
        return {
            "n_spatial_features": self.n_spatial_features,
            "balanced_accuracy": float(bal_acc),
            "roc_auc": float(roc_auc),
            "adopted_model": self.adopted_model,
            "accepted_merges": self.accepted_merges,
            "masks_changed": self.masks_changed,
        }
