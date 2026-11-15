"""NP1032 (45 mm NHP long) probe geometry, SpikeGLX meta, and Trodes XML parsers."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProbeGeometry:
    """2D channel coordinates in micrometers relative to bank origin or tip."""

    probe_part: str
    shank_width_um: float
    tip_length_um: float
    channel_ids: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    connected: np.ndarray

    @property
    def positions_um(self) -> np.ndarray:
        """Return `(n_channels, 2)` float64 XY coordinates in micrometers."""
        return np.column_stack([self.x_um, self.y_um])


def parse_spikeglx_meta(meta_path: Path | str) -> dict[str, str]:
    """Parse key=value pairs from a SpikeGLX `.ap.meta` or `.lf.meta` file."""
    out: dict[str, str] = {}
    with Path(meta_path).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def geometry_from_sns_geom_map(meta: dict[str, str]) -> ProbeGeometry:
    """Build ProbeGeometry directly from SpikeGLX `~snsGeomMap` metadata."""
    geom_raw = meta.get("~snsGeomMap", "")
    hdr_m = re.match(r"^\(([^,]+),(\d+),[^,]+,([0-9.]+)\)(.*)$", geom_raw)
    if not hdr_m:
        raise ValueError(f"Invalid or missing ~snsGeomMap: {geom_raw[:80]}")
    part = hdr_m.group(1)
    shank_w = float(hdr_m.group(3))
    body = hdr_m.group(4)

    tuples = re.findall(r"\(\d+:([-+0-9.]+):([-+0-9.]+):([01])\)", body)
    if not tuples:
        raise ValueError("No site tuples parsed from ~snsGeomMap")

    n = len(tuples)
    xs = np.array([float(t[0]) for t in tuples], dtype=np.float64)
    ys = np.array([float(t[1]) for t in tuples], dtype=np.float64)
    conn = np.array([int(t[2]) == 1 for t in tuples], dtype=bool)
    tip_um = float(meta.get("imTipLength", "373"))
    return ProbeGeometry(
        probe_part=part,
        shank_width_um=shank_w,
        tip_length_um=tip_um,
        channel_ids=np.arange(n, dtype=np.int64),
        x_um=xs,
        y_um=ys,
        connected=conn,
    )


def np1032_bank_geometry(bank: int = 0, n_channels: int = 384, tip_length_um: float = 373.0) -> ProbeGeometry:
    """Analytic NP1032 NHP-long bank geometry (`x in {11, 114} um`, `20 um` row pitch).

    Matches verified `~snsGeomMap=(NP1032,1,0,125)(0:11:0:1)(0:114:0:1)...` shifted
    globally by `bank * (n_channels // 2) * 20 um` along the shank.
    """
    if not (0 <= bank <= 11):
        raise ValueError(f"NP1032 bank must be in 0..11, got {bank}")
    idx = np.arange(n_channels, dtype=np.int64)
    xs = np.where((idx % 2) == 0, 11.0, 114.0)
    rows_per_bank = n_channels // 2
    global_row = bank * rows_per_bank + (idx // 2)
    ys = global_row.astype(np.float64) * 20.0
    return ProbeGeometry(
        probe_part="NP1032",
        shank_width_um=125.0,
        tip_length_um=float(tip_length_um),
        channel_ids=idx,
        x_um=xs,
        y_um=ys,
        connected=np.ones(n_channels, dtype=bool),
    )


def spikegadgets_wireless_bank_geometry(
    bank: int = 0,
    n_channels: int = 384,
    global_z: bool = False,
    tip_length_um: float = 373.0,
) -> ProbeGeometry:
    """Verified SpikeGadgets wireless `.dat` hardware-channel-to-site mapping.

    Confirmed 100% identical to `20260311_100744.trodesconf` and all existing
    `April21_wireless` / `March31_wireless` / `March23_wireless` Kilosort4
    locations: consecutive `.dat` channels are ADC-interleaved rather than
    sequentially ordered along depth (`even ch -> x=114 um`, `odd ch -> x=11 um`,
    `row = (pair % 16) * 12 + (pair // 16)`).
    """
    if not (0 <= bank <= 11):
        raise ValueError(f"NP1032 bank must be in 0..11, got {bank}")
    hw = np.arange(n_channels, dtype=np.int64)
    pair = hw // 2
    col = 1 - (hw % 2)  # even hw -> right column (x=114), odd -> left column (x=11)
    row = (pair % 16) * 12 + (pair // 16)
    xs = np.where(col == 0, 11.0, 114.0).astype(np.float64)
    ys = row.astype(np.float64) * 20.0 + (float(bank) * 3840.0 if global_z else 0.0)
    return ProbeGeometry(
        probe_part="NP1032",
        shank_width_um=125.0,
        tip_length_um=float(tip_length_um),
        channel_ids=hw,
        x_um=xs,
        y_um=ys,
        connected=np.ones(n_channels, dtype=bool),
    )


def parse_trodesconf_channels(trodesconf_path: Path | str) -> list[dict[str, Any]]:
    """Parse `SpikeNTrode` -> `SpikeChannel hwChan` mapping from a `.trodesconf` XML."""
    root = ET.parse(Path(trodesconf_path)).getroot()
    spike_cfg = root.find("SpikeConfiguration")
    if spike_cfg is None:
        raise ValueError(f"Missing <SpikeConfiguration> in {trodesconf_path}")

    entries: list[dict[str, Any]] = []
    for ntrode in spike_cfg.findall("SpikeNTrode"):
        ntrode_id = int(ntrode.attrib["id"])
        raw_scale = float(ntrode.attrib.get("rawScalingToUv", "0.018311105685598315"))
        for ch in ntrode.findall("SpikeChannel"):
            entries.append(
                {
                    "ntrode_id": ntrode_id,
                    "hw_chan": int(ch.attrib["hwChan"]),
                    "raw_scaling_to_uv": raw_scale,
                    "trigger_on": int(ch.attrib.get("triggerOn", "1")),
                }
            )
    return entries
