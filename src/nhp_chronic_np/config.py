"""Core probe geometry constants, bank boundaries, and env-configurable workspace paths."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

IMPLANT_DATE = dt.date(2026, 2, 19)
CLEAN_SGLX_REF_DATE = dt.date(2026, 2, 25)

ANIMAL_ID = "Olivia"
ANIMAL_SPECIES = "Macaca mulatta"
PROBE_PART = "NP1032"
PROBE_SERIAL = "22176114101"
PROBE_LENGTH_MM = 45.0
SAMPLE_RATE_HZ = 30000.0
PACKET_BYTES = 848

# Strictly enforced multi-bank policy (sequential banks are non-simultaneous)
AP_PRIMARY_BANKS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
LFP_PRIMARY_BANKS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
LFP_EXPLORATORY_BANKS: tuple[int, ...] = (7, 8)
EXCLUDED_BANKS: tuple[int, ...] = (9, 10, 11)

LOCKED_EVAL_SESSIONS: tuple[str, ...] = (
    "20260507",
    "20260604",
    "20260625",
    "20260730",
    "20260914",
    "20261013",
)

GOVERNANCE_CONTROLS: dict[str, object] = {
    "adopted_model": False,
    "accepted_merges": 0,
    "masks_changed": False,
    "single_animal_n": 1,
    "allow_flip_vflip_laminar_claims": False,
}

WIRELESS_SESSIONS: dict[str, str] = {
    "March23_wireless": "2026-03-23",
    "March31_wireless": "2026-03-31",
    "April7_wireless": "2026-04-07",
    "April21_wireless": "2026-04-21",
    "Nov6_wireless": "2026-11-06",
}


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved runtime paths configurable via DATA_ROOT / TRODES_ROOT."""

    e_raw_root: Path = Path(os.environ.get("DATA_ROOT", r"E:"))
    trodes_root: Path = Path(os.environ.get("TRODES_ROOT", r"E:\Dropbox\Lab Files\TRODES test"))
    sorting_output: Path = Path(os.environ.get("DATA_ROOT", r"E:")) / "sorting_output"
    reference_trodesconf: Path = (
        Path(os.environ.get("TRODES_ROOT", r"E:\Dropbox\Lab Files\TRODES test")) / "incagenew" / "config.trodesconf"
    )


PROJECT_PATHS = ProjectPaths()
