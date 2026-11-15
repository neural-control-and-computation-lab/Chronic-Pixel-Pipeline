"""Packet format constants and streaming parsing for SpikeGadgets Neuropixels wireless data logger."""

import os
import struct
from pathlib import Path

import numpy as np

# --- Format constants ---
SYNC_BYTE = 0x55
MAGIC_STANDARD = 0x1155  # Standard firmware (no IMU)
MAGIC_IMU = 0x1655       # IMU-enabled firmware v1
MAGIC_IMU_V2 = 0x1755
MAGIC_IMU_V3 = 0x1855
MAGIC_IMU_V4 = 0x1955
MAGIC_IMU_V5 = 0x0855
MAGIC_IMU_V6 = 0x0955
MAGIC_IMU_V7 = 0x0A55
MAGIC_IMU_V8 = 0x0B55
MAGIC_IMU_V9 = 0x1A55
MAGIC_IMU_V10 = 0x0C55
MAGIC_IMU_V11 = 0x1B55
MAGIC_IMU_V12 = 0x1C55
MAGIC_IMU_V13 = 0x0D55
MAGIC_IMU_V14 = 0x0E55
MAGIC_IMU_V15 = 0x1F55
MAGIC_IMU_V16 = 0x1D55
MAGIC_IMU_V17 = 0x2455
VALID_MAGICS = {
    MAGIC_STANDARD, MAGIC_IMU, MAGIC_IMU_V2, MAGIC_IMU_V3, MAGIC_IMU_V4,
    MAGIC_IMU_V5, MAGIC_IMU_V6, MAGIC_IMU_V7, MAGIC_IMU_V8, MAGIC_IMU_V9,
    MAGIC_IMU_V10, MAGIC_IMU_V11, MAGIC_IMU_V12, MAGIC_IMU_V13, MAGIC_IMU_V14,
    MAGIC_IMU_V15, MAGIC_IMU_V16, MAGIC_IMU_V17,
}
IMU_MAGICS = VALID_MAGICS - {MAGIC_STANDARD}

HEADER_SIZE = 80  # bytes per packet header
SAMPLING_RATE = 30_000.0  # Hz (Neuropixels AP band)
DTYPE = np.dtype("<i2")  # int16, little-endian
BYTES_PER_SAMPLE = 2
CHANNELS_PER_BANK = 48  # each bit in the idle bank bitmask = 48 channels

# Empirically verified header byte offsets (848-byte Trodes packet)
MAGIC_OFFSET = 0             # 2 bytes: [sync=0x55, Controller_DIO]
DIO_OFFSET = 1               # 1 byte: Controller_DIO
MUX_FLAG_OFFSET = 2          # 1 byte: multiplexed sensor/status flag
BANK_OFFSET = 3              # 1 byte: active probe bank ID (when mux_flag == 0x00)
MUX_XYZ_OFFSET = 4           # 6 bytes: 3 x <i2 sensor XYZ payload (when mux_flag != 0)
CHANNEL_BITMASK_OFFSET = 5   # 1 byte: active channel mask on idle packets
LFP_CYCLE_OFFSET = 10        # 2 bytes: 12-step cycle one-hot LFP mask
LFP_DATA_OFFSET = 12         # 64 bytes: 32 x <i2 LFP samples
COUNTER_OFFSET = 76          # 4 bytes: <u4 30 kHz hardware packet counter
DATA_OFFSET = 80             # neural AP data starts here (384 x <i2)

# Backwards-compatible aliases
CYCLE_BANK_OFFSET = LFP_CYCLE_OFFSET
CYCLE_MUX_OFFSET = MUX_FLAG_OFFSET
METADATA_OFFSET = LFP_DATA_OFFSET

# Multiplexed device flags (byte 2)
MUX_NONE = 0x00
MUX_RFSYNC = 0x01
MUX_MAG = 0x02
MUX_GYRO = 0x04
MUX_ACCEL = 0x08
MUX_ACK = 0x10
MUX_AIN = 0x20

# Empirically calibrated IMU conversion constants
ACCEL_LSB_PER_G = 16384.0    # +-2 g 16-bit full scale (resting norm ~ 0.92-0.99 g)
GYRO_LSB_PER_DPS = 131.072   # +-250 deg/s nominal 16-bit scale

IMU_CHANNELS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
]

IMU_MUX_MAP = {
    MUX_ACCEL: ("accel", 0, 1, 2),
    MUX_GYRO: ("gyro", 0, 1, 2),
    MUX_MAG: ("mag", 0, 1, 2),
}


def compute_n_channels(bitmask: int) -> int:
    """Compute number of active channels from the channel bitmask (48 ch/bit)."""
    n_ch = bin(int(bitmask)).count("1") * CHANNELS_PER_BANK
    return n_ch if n_ch > 0 else 384


def compute_packet_size(n_channels: int) -> int:
    """Compute total packet size in bytes given the number of channels."""
    return HEADER_SIZE + int(n_channels) * BYTES_PER_SAMPLE


def parse_header(header_bytes: bytes) -> dict:
    """Parse an 80-byte packet header."""
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError(
            f"Header must be at least {HEADER_SIZE} bytes, got {len(header_bytes)}"
        )

    sync = header_bytes[0]
    if sync != SYNC_BYTE:
        raise ValueError(
            f"Invalid sync byte: 0x{sync:02x} (expected 0x{SYNC_BYTE:02x})"
        )

    magic = struct.unpack_from("<H", header_bytes, MAGIC_OFFSET)[0]
    mux_id = header_bytes[MUX_FLAG_OFFSET]
    bank = header_bytes[BANK_OFFSET]
    bitmask = header_bytes[CHANNEL_BITMASK_OFFSET] if mux_id == MUX_NONE else 0xFF
    n_channels = compute_n_channels(bitmask)
    counter = struct.unpack_from("<I", header_bytes, COUNTER_OFFSET)[0]
    xyz = np.frombuffer(header_bytes, dtype=DTYPE, count=3, offset=MUX_XYZ_OFFSET).copy()
    metadata = np.frombuffer(header_bytes, dtype=DTYPE, count=32, offset=METADATA_OFFSET).copy()

    return {
        "magic": magic,
        "has_imu": (magic != MAGIC_STANDARD) and ((magic & 0xFF) == SYNC_BYTE),
        "bank": bank,
        "channel_bitmask": bitmask,
        "n_channels": n_channels,
        "packet_size": compute_packet_size(n_channels),
        "cycle_bank": struct.unpack_from("<H", header_bytes, LFP_CYCLE_OFFSET)[0],
        "mux_id": mux_id,
        "xyz": xyz,
        "counter": counter,
        "metadata": metadata,
    }


def detect_format(filepath: Path, sample_packets: int = 2000) -> tuple:
    """Auto-detect bank, n_channels, packet_size, and IMU availability via bounded read."""
    filepath = Path(filepath)
    fsize = filepath.stat().st_size
    if fsize < HEADER_SIZE:
        raise ValueError(f"File too small to contain a valid header: {filepath}")

    with open(filepath, "rb") as f:
        first_hdr = f.read(HEADER_SIZE)
        info0 = parse_header(first_hdr)
        packet_size = info0["packet_size"]
        n_packets = fsize // packet_size
        read_n = min(n_packets, max(1, int(sample_packets)))
        f.seek(0)
        buf = np.frombuffer(f.read(read_n * packet_size), dtype=np.uint8)
        pkts = buf.reshape(read_n, packet_size)

    if not np.all(pkts[:, 0] == SYNC_BYTE):
        raise ValueError(f"Sync byte 0x{SYNC_BYTE:02x} violation in {filepath}")

    idle_mask = pkts[:, MUX_FLAG_OFFSET] == MUX_NONE
    if np.any(idle_mask):
        bank = int(np.bincount(pkts[idle_mask, BANK_OFFSET]).argmax())
    else:
        bank = int(info0["bank"])

    return bank, int(info0["n_channels"]), int(packet_size), bool(info0["has_imu"])


def count_packets(filepath: Path, packet_size: int) -> int:
    """Count the number of complete packets in a file."""
    filepath = Path(filepath)
    return filepath.stat().st_size // int(packet_size)


def is_usable_segment(filepath: Path, min_packets: int = 5000) -> bool:
    """Return True if .dat file has enough packets and a valid neural bank (not bank 15 stub)."""
    bank, _n_ch, psize, _imu = detect_format(filepath)
    n_pkts = count_packets(filepath, psize)
    return (n_pkts >= int(min_packets)) and (0 <= int(bank) <= 11)


def find_continuity_blocks(counters: np.ndarray) -> dict:
    """Identify contiguous dt == 1 blocks and counter jumps (packet drops)."""
    ts = np.asarray(counters, dtype=np.uint64).ravel()
    n = ts.size
    if n == 0:
        return {"block_ids": np.empty(0, dtype=np.int64), "blocks": [], "gaps": []}
    dt = np.diff(ts.astype(np.int64))
    break_idx = np.flatnonzero(dt != 1) + 1
    starts = np.concatenate(([0], break_idx))
    ends = np.concatenate((break_idx, [n]))
    block_ids = np.empty(n, dtype=np.int64)
    blocks = []
    for bid, (s, e) in enumerate(zip(starts, ends)):
        block_ids[s:e] = bid
        blocks.append({
            "block_id": int(bid),
            "start_pkt": int(s),
            "end_pkt": int(e),
            "n_packets": int(e - s),
            "start_counter": int(ts[s]),
            "end_counter": int(ts[e - 1]),
        })
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


def extract_imu(filepath: Path, chunk_packets: int = 250_000) -> dict:
    """Stream-extract multiplexed IMU streams using verified offsets (byte 2, bytes 4..9, bytes 76..79).

    Reads large .dat files in chunks so multi-GB recordings require <250 MB RAM.
    Computes empirical IMU sampling rate from packet spans (~98.0 Hz) and scales
    accelerometer counts to gravitational units via ACCEL_LSB_PER_G (16,384 LSB/g).
    """
    filepath = Path(filepath)
    bank, n_channels, packet_size, has_imu = detect_format(filepath)
    if not has_imu:
        raise ValueError(f"File does not contain IMU firmware magic: {filepath}")

    n_packets = count_packets(filepath, packet_size)
    accel_chunks, accel_idx = [], []
    gyro_chunks, gyro_idx = [], []
    mag_chunks, mag_idx = [], []
    rf_idx = []
    counters = np.empty(n_packets, dtype=np.uint32)

    pkt_offset = 0
    with open(filepath, "rb") as f:
        while pkt_offset < n_packets:
            take = min(chunk_packets, n_packets - pkt_offset)
            buf = np.frombuffer(f.read(take * packet_size), dtype=np.uint8).reshape(take, packet_size)
            if not np.all(buf[:, 0] == SYNC_BYTE):
                bad = int(np.flatnonzero(buf[:, 0] != SYNC_BYTE)[0]) + pkt_offset
                raise ValueError(f"Sync byte error at packet {bad} in {filepath.name}")

            flags = buf[:, MUX_FLAG_OFFSET]
            xyz = np.ascontiguousarray(buf[:, MUX_XYZ_OFFSET : MUX_XYZ_OFFSET + 6]).view(DTYPE).reshape(take, 3)
            counters[pkt_offset : pkt_offset + take] = (
                np.ascontiguousarray(buf[:, COUNTER_OFFSET : COUNTER_OFFSET + 4]).view("<u4").ravel()
            )

            m_acc = (flags & MUX_ACCEL) != 0
            if np.any(m_acc):
                accel_chunks.append(xyz[m_acc].copy())
                accel_idx.append(np.flatnonzero(m_acc) + pkt_offset)

            m_gyr = (flags & MUX_GYRO) != 0
            if np.any(m_gyr):
                gyro_chunks.append(xyz[m_gyr].copy())
                gyro_idx.append(np.flatnonzero(m_gyr) + pkt_offset)

            m_mag = (flags & MUX_MAG) != 0
            if np.any(m_mag):
                mag_chunks.append(xyz[m_mag].copy())
                mag_idx.append(np.flatnonzero(m_mag) + pkt_offset)

            m_rf = (flags & MUX_RFSYNC) != 0
            if np.any(m_rf):
                rf_idx.append(np.flatnonzero(m_rf) + pkt_offset)

            pkt_offset += take

    accel = np.vstack(accel_chunks) if accel_chunks else np.empty((0, 3), dtype=DTYPE)
    gyro = np.vstack(gyro_chunks) if gyro_chunks else np.empty((0, 3), dtype=DTYPE)
    mag = np.vstack(mag_chunks) if mag_chunks else np.empty((0, 3), dtype=DTYPE)
    acc_pkt = np.concatenate(accel_idx) if accel_idx else np.empty(0, dtype=np.int64)
    gyr_pkt = np.concatenate(gyro_idx) if gyro_idx else np.empty(0, dtype=np.int64)
    mag_pkt = np.concatenate(mag_idx) if mag_idx else np.empty(0, dtype=np.int64)
    rf_pkt = np.concatenate(rf_idx) if rf_idx else np.empty(0, dtype=np.int64)

    cont = find_continuity_blocks(counters)
    dur_s = n_packets / SAMPLING_RATE
    emp_fs = (len(accel) / dur_s) if (dur_s > 0 and len(accel) > 0) else 98.0

    return {
        "bank": bank,
        "n_packets": int(n_packets),
        "duration_s": float(dur_s),
        "sampling_rate": float(emp_fs),
        "channels": IMU_CHANNELS,
        "accel": accel,
        "accel_g": accel.astype(np.float64) / ACCEL_LSB_PER_G,
        "gyro": gyro,
        "mag": mag,
        "accel_pkt_idx": acc_pkt,
        "gyro_pkt_idx": gyr_pkt,
        "mag_pkt_idx": mag_pkt,
        "rfsync_pkt_idx": rf_pkt,
        "packet_counters": counters,
        "continuity_blocks": cont["blocks"],
        "gaps": cont["gaps"],
        "n_gap_events": len(cont["gaps"]),
    }
