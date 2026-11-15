"""Continuity-block-aware neural-to-IMU kinematics decoding with circular-shift nulls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ImuDecodingResult:
    """Cross-validated R^2 and block-wise circular time-shift permutation null summary."""

    cv_r2: float
    null_r2_mean: float
    null_r2_p95: float
    p_value: float
    n_bins: int
    n_blocks: int
    n_permutations: int


def blockwise_circular_shift(
    matrix: np.ndarray,
    block_ids: np.ndarray,
    min_shift_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circularly shift rows strictly within each contiguous packet-continuity block.

    Preserves within-block temporal autocorrelation of both IMU kinematics and
    neural firing rates while breaking true sensor-neural coupling without
    leaking across telemetry packet drops.
    """
    shifted = np.empty_like(matrix)
    for bid in np.unique(block_ids):
        idx = np.flatnonzero(block_ids == bid)
        n = idx.size
        if n <= 2:
            shifted[idx] = matrix[idx]
            continue
        lo = min(int(min_shift_bins), max(1, n // 4))
        hi = max(lo + 1, n - lo)
        k = int(rng.integers(lo, hi))
        shifted[idx] = np.roll(matrix[idx], shift=k, axis=0)
    return shifted


def _blocked_cv_r2(
    x: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
) -> float:
    """Blocked contiguous k-fold ridge CV R^2 (no random sample shuffling)."""
    n = len(y)
    folds = np.array_split(np.arange(n), n_folds)
    preds = np.empty_like(y, dtype=np.float64)

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
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def decode_imu_with_circular_null(
    neural_rates: np.ndarray,
    imu_target: np.ndarray,
    block_ids: np.ndarray,
    bin_dt_s: float = 0.05,
    min_shift_s: float = 5.0,
    n_folds: int = 5,
    n_permutations: int = 200,
    seed: int = 42,
) -> ImuDecodingResult:
    """Evaluate neural-to-IMU decoding R^2 against block-wise circular-shift null."""
    x = np.asarray(neural_rates, dtype=np.float64)
    y = np.asarray(imu_target, dtype=np.float64).ravel()
    bids = np.asarray(block_ids, dtype=np.int64).ravel()
    if x.shape[0] != y.size or y.size != bids.size:
        raise ValueError("Mismatched sample lengths between neural_rates, imu_target, and block_ids")

    obs_r2 = _blocked_cv_r2(x, y, n_folds=n_folds)
    min_shift_bins = max(1, int(round(float(min_shift_s) / float(bin_dt_s))))
    rng = np.random.default_rng(seed)

    null_r2s = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        y_shuf = blockwise_circular_shift(y[:, None], bids, min_shift_bins=min_shift_bins, rng=rng).ravel()
        null_r2s[i] = _blocked_cv_r2(x, y_shuf, n_folds=n_folds)

    p_val = float((1 + np.sum(null_r2s >= obs_r2)) / (1 + n_permutations))
    return ImuDecodingResult(
        cv_r2=float(obs_r2),
        null_r2_mean=float(np.mean(null_r2s)),
        null_r2_p95=float(np.percentile(null_r2s, 95)),
        p_value=p_val,
        n_bins=int(y.size),
        n_blocks=int(np.unique(bids).size),
        n_permutations=int(n_permutations),
    )
