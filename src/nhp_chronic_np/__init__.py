"""Unified NHP chronic Neuropixels 1.0 + wireless SpikeGadgets analysis package."""

from nhp_chronic_np.config import (
    AP_PRIMARY_BANKS,
    CLEAN_SGLX_REF_DATE,
    EXCLUDED_BANKS,
    IMPLANT_DATE,
    LFP_EXPLORATORY_BANKS,
    LFP_PRIMARY_BANKS,
    LOCKED_EVAL_SESSIONS,
    PROBE_PART,
    PROBE_SERIAL,
    PROJECT_PATHS,
    WIRELESS_SESSIONS,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "AP_PRIMARY_BANKS",
    "CLEAN_SGLX_REF_DATE",
    "EXCLUDED_BANKS",
    "IMPLANT_DATE",
    "LFP_EXPLORATORY_BANKS",
    "LFP_PRIMARY_BANKS",
    "LOCKED_EVAL_SESSIONS",
    "PROBE_PART",
    "PROBE_SERIAL",
    "PROJECT_PATHS",
    "WIRELESS_SESSIONS",
]
