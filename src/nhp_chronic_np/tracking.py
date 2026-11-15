"""Within-bank Hungarian longitudinal unit matching and split-half calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchedUnitPair:
    """One-to-one matched unit identity between two recordings of the same bank."""

    source_idx: int
    target_idx: int
    similarity_score: float
    waveform_cosine: float
    spatial_distance_um: float
    rate_log_ratio: float


def _rowwise_l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def compute_similarity_matrix(
    waveforms_a: np.ndarray,
    positions_a_um: np.ndarray,
    rates_a_hz: np.ndarray,
    waveforms_b: np.ndarray,
    positions_b_um: np.ndarray,
    rates_b_hz: np.ndarray,
    max_spatial_dist_um: float = 40.0,
    rate_sigma_log: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise combined similarity, cosine, spatial distance, and log rate diff."""
    wa = _rowwise_l2_normalize(waveforms_a)
    wb = _rowwise_l2_normalize(waveforms_b)
    cos_sim = np.clip(wa @ wb.T, -1.0, 1.0)

    pa = np.asarray(positions_a_um, dtype=np.float64)[:, None, :]
    pb = np.asarray(positions_b_um, dtype=np.float64)[None, :, :]
    dist_um = np.linalg.norm(pa - pb, axis=2)

    ra = np.log10(np.maximum(np.asarray(rates_a_hz, dtype=np.float64), 1e-3))[:, None]
    rb = np.log10(np.maximum(np.asarray(rates_b_hz, dtype=np.float64), 1e-3))[None, :]
    rate_diff = np.abs(ra - rb)

    spatial_term = np.exp(-0.5 * (dist_um / (0.5 * max_spatial_dist_um)) ** 2)
    rate_term = np.exp(-0.5 * (rate_diff / rate_sigma_log) ** 2)
    combined = np.maximum(cos_sim, 0.0) * spatial_term * rate_term
    combined = np.where(dist_um <= max_spatial_dist_um, combined, 0.0)
    return combined, cos_sim, dist_um, rate_diff


def match_units_hungarian(
    waveforms_a: np.ndarray,
    positions_a_um: np.ndarray,
    rates_a_hz: np.ndarray,
    waveforms_b: np.ndarray,
    positions_b_um: np.ndarray,
    rates_b_hz: np.ndarray,
    min_score: float = 0.60,
    min_cosine: float = 0.85,
    max_spatial_dist_um: float = 40.0,
    bank_a: int | None = None,
    bank_b: int | None = None,
) -> list[MatchedUnitPair]:
    """Perform strict one-to-one Hungarian matching within the same probe bank.

    Enforces two critical safeguards:
    1. Cross-bank matches are forbidden unless evaluating an explicit negative null.
    2. Output pairs have zero many-to-one target or source collisions.
    """
    if bank_a is not None and bank_b is not None and bank_a != bank_b:
        raise ValueError(
            f"Refusing cross-bank matching (bank {bank_a} vs {bank_b}); "
            "sequential banks are not simultaneously recorded."
        )
    if len(waveforms_a) == 0 or len(waveforms_b) == 0:
        return []

    combined, cos_sim, dist_um, rate_diff = compute_similarity_matrix(
        waveforms_a,
        positions_a_um,
        rates_a_hz,
        waveforms_b,
        positions_b_um,
        rates_b_hz,
        max_spatial_dist_um=max_spatial_dist_um,
    )
    row_ind, col_ind = linear_sum_assignment(-combined)

    matches: list[MatchedUnitPair] = []
    for r, c in zip(row_ind, col_ind):
        sc = float(combined[r, c])
        cs = float(cos_sim[r, c])
        du = float(dist_um[r, c])
        if sc >= min_score and cs >= min_cosine and du <= max_spatial_dist_um:
            matches.append(
                MatchedUnitPair(
                    source_idx=int(r),
                    target_idx=int(c),
                    similarity_score=sc,
                    waveform_cosine=cs,
                    spatial_distance_um=du,
                    rate_log_ratio=float(rate_diff[r, c]),
                )
            )
    return matches


def evaluate_null_false_positive_rate(
    waveforms_a: np.ndarray,
    positions_a_um: np.ndarray,
    rates_a_hz: np.ndarray,
    waveforms_b: np.ndarray,
    positions_b_um: np.ndarray,
    rates_b_hz: np.ndarray,
    n_permutations: int = 200,
    seed: int = 0,
    **match_kwargs: float,
) -> dict[str, float]:
    """Estimate false-match rate under channel-position + template permutation null."""
    rng = np.random.default_rng(seed)
    n_a = len(waveforms_a)
    n_b = len(waveforms_b)
    if n_a == 0 or n_b == 0:
        return {"mean_null_matches": 0.0, "null_fpr": 0.0}

    counts = []
    denom = float(min(n_a, n_b))
    for _ in range(n_permutations):
        perm = rng.permutation(n_b)
        wb_rot = np.roll(waveforms_b, shift=int(rng.integers(1, waveforms_b.shape[1])), axis=1)
        m = match_units_hungarian(
            waveforms_a,
            positions_a_um,
            rates_a_hz,
            wb_rot[perm],
            positions_b_um,
            rates_b_hz[perm],
            **match_kwargs,
        )
        counts.append(len(m) / denom)
    return {
        "mean_null_matches": float(np.mean(counts) * denom),
        "null_fpr": float(np.mean(counts)),
        "p95_null_fpr": float(np.percentile(counts, 95)),
    }
