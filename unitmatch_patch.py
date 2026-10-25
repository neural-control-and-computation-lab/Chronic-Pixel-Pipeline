"""
Memory-safe replacement for UnitMatchPy's recentered-euclidean-distance metric.

Why
---
``UnitMatchPy.metric_functions.get_recentered_euclidean_dist`` (3.3.0) builds
the full pairwise comparison by tiling:

    x1 = tile(recentered[:, :, waveidx, 0, :], n_units)   # (3, n, 23, 2, n)
    x2 = swapaxes(tile(recentered[:, :, waveidx, 1, :], n_units), 1, 4)
    tmp_euclid = norm(x1 - x2, axis=0)                     # (n, 23, 2, n)

At n = 2,714 units (bank 3 of the chronic NP1032 series as of 37 sessions)
``x1``, ``x2`` and ``x1 - x2`` are ~7.5 GiB each in float64, so the call needs
25-30 GiB of transients and dies with

    numpy.core._exceptions._ArrayMemoryError: Unable to allocate 2.52 GiB for
    an array with shape (2714, 23, 2, 2714) and data type float64

even on a machine with 25 GiB free. The cost grows as n^2, so it worsens with
every survey added. track_neurons.py catches the error per bank and continues,
which means the run exits 0 having silently dropped its largest bank.

What this does
--------------
The quantity being computed is just a pairwise distance:

    out[i, t, s, j] = || R0[:, i, t, s] - R1[:, j, t, s] ||_2

where R0/R1 are the recentered waveforms at flip index 0/1. We evaluate it in
row blocks, so peak extra memory is O(block * n) instead of O(n^2). The output
array itself is unchanged, and results are numerically identical to the
original (same float64 dtype, same NaN handling).

Usage
-----
    import unitmatch_patch
    unitmatch_patch.apply()      # idempotent; call before running UnitMatch
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

_APPLIED = False

# Rows of the (n, 23, 2, n) output computed per block. 64 keeps the working
# set near 200 MB at n ~ 2,700; lower it if memory is tighter.
BLOCK = 64


def get_recentered_euclidean_dist_chunked(avg_waveform_per_tp_flip,
                                           avg_centroid, param):
    """Drop-in replacement computing the same array without O(n^2) temporaries.

    Parameters match the upstream function:
      avg_waveform_per_tp_flip : (3, n_units, spike_width, 2, 2)
      avg_centroid             : (3, n_units, 2)
      param                    : dict with 'waveidx', 'n_units', 'spike_width'

    Returns
    -------
    ndarray, shape (n_units, len(waveidx), 2, n_units), float64
    """
    waveidx = param['waveidx']
    n_units = param['n_units']
    spike_width = param['spike_width']

    # Recenter exactly as upstream does.
    avg_centroid_broadcast = np.tile(
        np.expand_dims(avg_centroid, axis=(3, 4)), (1, 1, 1, spike_width, 2))
    recentered = np.swapaxes(
        np.swapaxes(avg_waveform_per_tp_flip, 2, 3) - avg_centroid_broadcast,
        2, 3)

    # (3, n_units, n_wave, 2) for each flip index.
    r0 = recentered[:, :, waveidx, 0, :]
    r1 = recentered[:, :, waveidx, 1, :]
    n_wave = r0.shape[2]

    # r1 laid out so the *column* unit is the trailing axis: (3, 1, n_wave, 2, n)
    r1_cols = np.moveaxis(r1, 1, -1)[:, None, ...]

    out = np.empty((n_units, n_wave, 2, n_units), dtype=np.float64)

    for start in range(0, n_units, BLOCK):
        stop = min(start + BLOCK, n_units)
        # (3, block, n_wave, 2, 1) - (3, 1, n_wave, 2, n) -> (3, block, n_wave, 2, n)
        diff = r0[:, start:stop, ..., None] - r1_cols
        block_out = np.linalg.norm(diff, axis=0)
        # Upstream masks on the first coordinate only; mirror that exactly.
        block_out[np.isnan(diff[0])] = np.nan
        out[start:stop] = block_out
        del diff, block_out

    return out


# UnitMatchPy versions this patch has been checked against. A different
# version is not fatal — the self-check below is the real gate — but it is
# worth a warning so an upgrade gets a second look.
TESTED_VERSIONS = {'3.3.0'}

_TARGET = 'get_recentered_euclidean_dist'


class PatchError(RuntimeError):
    """Raised when the patch cannot be applied safely.

    Always fail loudly rather than silently leaving upstream in place: the
    unpatched code dies on bank 3, and a swallowed failure there is what let
    incomplete tracking results reach Dropbox in the first place.
    """


def _self_check(original):
    """Confirm our implementation still matches the installed upstream.

    Runs both on a small synthetic input (milliseconds) where upstream's
    O(n^2) tiling is harmless. Catches the case where a future UnitMatchPy
    keeps the function name but changes shape conventions or semantics, which
    would otherwise make us silently return wrong distances.
    """
    n, spike_width, n_wave = 6, 12, 4
    rng = np.random.default_rng(20261025)
    waveidx = np.arange(n_wave)
    param = {'waveidx': waveidx, 'n_units': n, 'spike_width': spike_width}
    awf = rng.normal(size=(3, n, spike_width, 2, 2))
    cen = rng.normal(size=(3, n, 2))
    # Exercise the NaN-mask path too.
    awf[0, 1, waveidx[0], 0, 0] = np.nan

    try:
        ref = original(awf.copy(), cen.copy(), param)
    except Exception as exc:
        raise PatchError(
            f"upstream {_TARGET} could not be evaluated on the reference "
            f"input, so its calling convention has probably changed: {exc!r}. "
            "Review unitmatch_patch.py against the installed UnitMatchPy "
            "before running tracking."
        ) from exc

    new = get_recentered_euclidean_dist_chunked(awf.copy(), cen.copy(), param)

    if ref.shape != new.shape:
        raise PatchError(
            f"shape mismatch vs upstream: {new.shape} != {ref.shape}")
    if not np.array_equal(np.isnan(ref), np.isnan(new)):
        raise PatchError("NaN placement differs from upstream")
    finite = ~np.isnan(ref)
    if finite.any() and not np.allclose(ref[finite], new[finite],
                                        rtol=0, atol=1e-12):
        worst = np.abs(ref[finite] - new[finite]).max()
        raise PatchError(
            f"values differ from upstream (max abs diff {worst:.3e})")


def apply(verify=True):
    """Monkey-patch UnitMatchPy in place. Idempotent and safe to call twice.

    Raises PatchError if the patch cannot be applied safely — if the upstream
    function has gone, or if our replacement no longer reproduces it.
    """
    global _APPLIED
    if _APPLIED:
        return False

    import UnitMatchPy.metric_functions as mf

    try:
        import importlib.metadata as _md
        version = _md.version('UnitMatchPy')
    except Exception:
        version = 'unknown'
    if version not in TESTED_VERSIONS:
        logger.warning(
            "UnitMatchPy %s has not been checked against this patch "
            "(tested: %s); relying on the self-check.",
            version, ', '.join(sorted(TESTED_VERSIONS)))

    # A bare setattr would happily create a NEW attribute if upstream renamed
    # or removed the function, leaving the real call path unpatched and the
    # O(n^2) blowup live. Require the target to exist.
    original = getattr(mf, _TARGET, None)
    if original is None:
        raise PatchError(
            f"UnitMatchPy.metric_functions.{_TARGET} does not exist in "
            f"version {version}. The memory fix would silently not apply; "
            "update unitmatch_patch.py to target the new API.")
    if getattr(original, '__module__', '').startswith('unitmatch_patch'):
        _APPLIED = True
        return False

    if verify:
        _self_check(original)

    mf.get_recentered_euclidean_dist = get_recentered_euclidean_dist_chunked

    # overlord.py calls it via the module alias `mf`, so patching the module
    # attribute is what matters; rebind a direct import too, if one appears.
    import UnitMatchPy.overlord as ov
    if hasattr(ov, _TARGET):
        ov.get_recentered_euclidean_dist = get_recentered_euclidean_dist_chunked

    if mf.get_recentered_euclidean_dist is not \
            get_recentered_euclidean_dist_chunked:
        raise PatchError("patch did not take effect on "
                         "UnitMatchPy.metric_functions")

    _APPLIED = True
    logger.info("Applied chunked %s patch (UnitMatchPy %s, block=%d, "
                "verified=%s)", _TARGET, version, BLOCK, verify)
    return True
