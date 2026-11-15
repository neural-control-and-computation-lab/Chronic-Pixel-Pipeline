"""Gap-safe 848-byte SpikeGadgets wireless NP1.0 packet parser and header verifier.

Verified against `incagenew/config.trodesconf` and binary headers across all five
wireless sessions (`March23`, `March31`, `April7`, `April21`, `Nov6`):
- Byte 0: Sync byte (`0x55`)
- Byte 1: Controller_DIO / session state byte (LE uint16 magic `0x0d55`..`0x2455`)
- Byte 2: Multiplexed flags (`0x00`=status/idle, `0x01`=RFsync, `0x02`=Mag,
          `0x04`=Gyro, `0x08`=Accel, `0x10`=Ack1)
- Byte 3: Status bank ID when `byte 2 == 0x00` (`0..11`, or `15` on empty stub)
- Bytes 4..9: Signed `<3h` sensor XYZ values when `byte 2` has Gyro/Accel/Mag bit
- Bytes 10..75: `neuropixels_LFP` (2-byte 12-step cycle mask + 32 x `<h` LFP)
- Bytes 76..79: `<I` 30 kHz hardware packet counter
- Bytes 80..847: 384 x `<h` AP samples (`1.2 V / (65536 * ap_gain)`)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nhp_chronic_np.config import PACKET_BYTES, SAMPLE_RATE_HZ

SYNC_BYTE = 0x55
DIO_OFFSET = 1
MUX_FLAG_OFFSET = 2
STATUS_BANK_OFFSET = 3
MUX_XYZ_OFFSET = 4
LFP_MUX_OFFSET = 10
LFP_DATA_OFFSET = 12
TIMESTAMP_OFFSET = 76
AP_DATA_OFFSET = 80
N_AP_CHANNELS = 384

FLAG_RFSYNC = 0x01
FLAG_MAG = 0x02
FLAG_GYRO = 0x04
FLAG_ACCEL = 0x08
FLAG_ACK1 = 0x10

KNOWN_SESSION_MAGIC_U16: tuple[int, ...] = (
    0x0A55,  # incagenew
    0x0D55,  # March23_wireless
    0x0E55,  # March31_wireless
    0x1D55,  # April7_wireless
    0x1F55,  # April21_wireless
    0x2455,  # Nov6_wireless
)

DEFAULT_UV_PER_BIT_GAIN1000 = 1_200_000.0 / (65536.0 * 1000.0)


@dataclass(frozen=True)
class DatHeaderSummary:
    """Bounded header/tail integrity summary for a wireless `.dat` file."""

    file_path: str
    file_bytes: int
    n_packets: int
    rem_bytes: int
    sync_0x55_ok: bool
    magic_u16_hex: tuple[str, ...]
    status_banks: tuple[int, ...]
    first_counter: int
    last_counter: int
    duration_nominal_s: float
    is_stub: bool


def inspect_dat_headers_bounded(
    dat_path: Path | str,
    head_packets: int = 4096,
    tail_packets: int = 1024,
    stub_packet_threshold: int = 5000,
) -> DatHeaderSummary:
    """Verify packet alignment, 0x55 sync byte, magics, and status banks in O(1) I/O."""
    path = Path(dat_path)
    fsize = path.stat().st_size
    n_packets = fsize // PACKET_BYTES
    rem_bytes = fsize % PACKET_BYTES
    if n_packets == 0:
        raise ValueError(f"File contains zero complete {PACKET_BYTES}-byte packets: {path}")

    with path.open("rb") as fh:
        h_n = min(n_packets, head_packets)
        head = np.frombuffer(fh.read(h_n * PACKET_BYTES), dtype=np.uint8).reshape(h_n, PACKET_BYTES)
        t_n = min(n_packets, tail_packets)
        fh.seek((n_packets - t_n) * PACKET_BYTES)
        tail = np.frombuffer(fh.read(t_n * PACKET_BYTES), dtype=np.uint8).reshape(t_n, PACKET_BYTES)

    both = np.concatenate([head, tail], axis=0)
    sync_ok = bool(np.all(both[:, 0] == SYNC_BYTE))
    u16 = both[:, 0].astype(np.uint16) | (both[:, 1].astype(np.uint16) << 8)
    magic_hex = tuple(f"0x{int(x):04x}" for x in np.unique(u16))

    idle = both[:, MUX_FLAG_OFFSET] == 0x00
    banks = tuple(int(x) for x in np.unique(both[idle, STATUS_BANK_OFFSET])) if np.any(idle) else ()
    first_ts = int(np.frombuffer(head[0, TIMESTAMP_OFFSET : TIMESTAMP_OFFSET + 4].tobytes(), dtype="<u4")[0])
    last_ts = int(np.frombuffer(tail[-1, TIMESTAMP_OFFSET : TIMESTAMP_OFFSET + 4].tobytes(), dtype="<u4")[0])

    return DatHeaderSummary(
        file_path=str(path),
        file_bytes=fsize,
        n_packets=n_packets,
        rem_bytes=rem_bytes,
        sync_0x55_ok=sync_ok,
        magic_u16_hex=magic_hex,
        status_banks=banks,
        first_counter=first_ts,
        last_counter=last_ts,
        duration_nominal_s=n_packets / SAMPLE_RATE_HZ,
        is_stub=(n_packets < stub_packet_threshold) or (15 in banks),
    )


def find_continuity_blocks(counters: np.ndarray) -> dict[str, Any]:
    """Identify contiguous `dt == 1` intervals and forward/backward counter jumps.

    Never smooths or fills across discontinuities; assigns each packet a `block_id`.
    """
    ts = np.asarray(counters, dtype=np.uint64).ravel()
    n = ts.size
    if n == 0:
        return {"block_ids": np.empty(0, dtype=np.int64), "blocks": [], "gaps": []}

    dt = np.diff(ts.astype(np.int64))
    break_idx = np.flatnonzero(dt != 1) + 1
    starts = np.concatenate(([0], break_idx))
    ends = np.concatenate((break_idx, [n]))

    block_ids = np.empty(n, dtype=np.int64)
    blocks: list[dict[str, int]] = []
    for bid, (s, e) in enumerate(zip(starts, ends)):
        block_ids[s:e] = bid
        blocks.append(
            {
                "block_id": int(bid),
                "start_pkt": int(s),
                "end_pkt": int(e),
                "n_packets": int(e - s),
                "start_counter": int(ts[s]),
                "end_counter": int(ts[e - 1]),
            }
        )

    gaps = [
        {
            "packet_index": int(bi),
            "prev_counter": int(ts[bi - 1]),
            "curr_counter": int(ts[bi]),
            "delta_ticks": int(ts[bi]) - int(ts[bi - 1]),
        }
        for bi in break_idx
    ]
    return {"block_ids": block_ids, "blocks": blocks, "gaps": gaps}


def extract_imu_from_packets(packets: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Extract accel, gyro, mag, and RF-sync events using verified byte offsets 2..9 & 76..79."""
    arr = np.asarray(packets, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != PACKET_BYTES:
        raise ValueError(f"Expected (N, {PACKET_BYTES}) uint8 packet array, got {arr.shape}")

    if not np.all(arr[:, 0] == SYNC_BYTE):
        bad = int(np.flatnonzero(arr[:, 0] != SYNC_BYTE)[0])
        raise ValueError(f"Sync byte mismatch at packet {bad}: expected 0x55, got 0x{arr[bad, 0]:02x}")

    flags = arr[:, MUX_FLAG_OFFSET]
    counters = arr[:, TIMESTAMP_OFFSET : TIMESTAMP_OFFSET + 4].copy().view("<u4").ravel()
    cont = find_continuity_blocks(counters)
    block_ids = cont["block_ids"]
    xyz_all = arr[:, MUX_XYZ_OFFSET : MUX_XYZ_OFFSET + 6].copy().view("<i2").reshape(-1, 3)

    streams: dict[str, dict[str, np.ndarray]] = {}
    for name, mask_bit in (
        ("accel", FLAG_ACCEL),
        ("gyro", FLAG_GYRO),
        ("mag", FLAG_MAG),
        ("rf_sync", FLAG_RFSYNC),
    ):
        idx = np.flatnonzero((flags & mask_bit) != 0)
        xyz = xyz_all[idx].copy()
        rail_mask = np.any((xyz == -32768) | (xyz == 32767), axis=1)
        streams[name] = {
            "packet_index": idx,
            "time_s": idx.astype(np.float64) / SAMPLE_RATE_HZ,
            "counter_u32": counters[idx],
            "block_id": block_ids[idx],
            "xyz_raw": xyz,
            "rail_saturated": rail_mask,
        }

    streams["_continuity"] = {
        "n_blocks": np.array([len(cont["blocks"])], dtype=np.int64),
        "n_gaps": np.array([len(cont["gaps"])], dtype=np.int64),
    }
    return streams


def decode_ap_int16_uv(
    packets: np.ndarray,
    ap_gain: float = 1000.0,
    scaling_uv_per_bit: float | None = None,
) -> np.ndarray:
    """Decode `(N, 848)` packets into `(N, 384)` float32 AP microvolt traces."""
    arr = np.asarray(packets, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != PACKET_BYTES:
        raise ValueError(f"Expected (N, {PACKET_BYTES}) uint8 packet array, got {arr.shape}")
    scale = (
        float(scaling_uv_per_bit)
        if scaling_uv_per_bit is not None
        else 1_200_000.0 / (65536.0 * float(ap_gain))
    )
    raw_i16 = arr[:, AP_DATA_OFFSET : AP_DATA_OFFSET + 2 * N_AP_CHANNELS].copy().view("<i2")
    return raw_i16.astype(np.float32) * np.float32(scale)
