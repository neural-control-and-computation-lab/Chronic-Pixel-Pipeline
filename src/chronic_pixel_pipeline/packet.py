"""Packet format constants and parsing for SpikeGadgets Neuropixels wireless data logger."""

import struct
from pathlib import Path

import numpy as np

# --- Format constants ---
MAGIC_STANDARD = 0x1155  # Standard firmware (no IMU)
MAGIC_IMU = 0x1655       # IMU-enabled firmware (accel/gyro/mag)
VALID_MAGICS = {MAGIC_STANDARD, MAGIC_IMU}
HEADER_SIZE = 80  # bytes per packet header
SAMPLING_RATE = 30_000.0  # Hz (Neuropixels AP band)
DTYPE = np.dtype("<i2")  # int16, little-endian
BYTES_PER_SAMPLE = 2
CHANNELS_PER_BANK = 48  # each bit in the bank bitmask = 48 channels

# Header field byte offsets
MAGIC_OFFSET = 0  # 2 bytes, uint16 LE
BANK_OFFSET = 3  # 1 byte, probe group/bank number (0-11)
CHANNEL_BITMASK_OFFSET = 5  # 1 byte, popcount * 48 = n_channels
CYCLE_BANK_OFFSET = 10   # 1 byte, bank rotation bitmask (bits 0-7)
CYCLE_MUX_OFFSET = 11    # 1 byte, multiplexed device ID
METADATA_OFFSET = 12     # 64 bytes (32 x int16 status/metadata)
COUNTER_OFFSET = 76      # 4 bytes, packet counter
DATA_OFFSET = 80         # neural data starts here

# Multiplexed device IDs (byte 11)
MUX_NONE = 0x00
MUX_RFSYNC = 0x01
MUX_MAG = 0x02
MUX_GYRO = 0x04
MUX_ACCEL = 0x08
MUX_ACK = 0x10
MUX_AIN = 0x20

# IMU cycle: 12 packets total, 8 bank rotation + 4 sensor
# Effective IMU sample rate = SAMPLING_RATE / 12
IMU_CYCLE_LENGTH = 12
IMU_SAMPLING_RATE = SAMPLING_RATE / IMU_CYCLE_LENGTH  # 2500 Hz

# IMU channel names and mux IDs in extraction order
IMU_CHANNELS = ["accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z",
                "mag_x", "mag_y", "mag_z"]

# Mapping from mux ID to (channel_prefix, metadata int16 indices for X, Y, Z)
IMU_MUX_MAP = {
    MUX_ACCEL: ("accel", 0, 1, 2),
    MUX_GYRO:  ("gyro",  0, 1, 2),
    MUX_MAG:   ("mag",   0, 1, 2),
}


def compute_n_channels(bitmask: int) -> int:
    """Compute number of active channels from the channel bank bitmask (byte 5).

    Each set bit represents a bank of 48 channels.
    """
    return bin(bitmask).count("1") * CHANNELS_PER_BANK


def compute_packet_size(n_channels: int) -> int:
    """Compute total packet size in bytes given the number of channels."""
    return HEADER_SIZE + n_channels * BYTES_PER_SAMPLE


def parse_header(header_bytes: bytes) -> dict:
    """Parse an 80-byte packet header.

    Parameters
    ----------
    header_bytes : bytes
        Exactly 80 bytes from the start of a packet.

    Returns
    -------
    dict with keys: magic, has_imu, bank, n_channels, packet_size,
                    cycle_bank, mux_id, metadata
    """
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError(
            f"Header must be at least {HEADER_SIZE} bytes, got {len(header_bytes)}"
        )

    magic = struct.unpack_from("<H", header_bytes, MAGIC_OFFSET)[0]
    if magic not in VALID_MAGICS:
        raise ValueError(
            f"Invalid magic number: 0x{magic:04x} "
            f"(expected one of {', '.join(f'0x{m:04x}' for m in VALID_MAGICS)})"
        )

    bank = header_bytes[BANK_OFFSET]
    bitmask = header_bytes[CHANNEL_BITMASK_OFFSET]
    n_channels = compute_n_channels(bitmask)
    cycle_bank = header_bytes[CYCLE_BANK_OFFSET]
    mux_id = header_bytes[CYCLE_MUX_OFFSET]
    metadata = np.frombuffer(header_bytes, dtype=DTYPE, count=32, offset=METADATA_OFFSET)

    return {
        "magic": magic,
        "has_imu": magic == MAGIC_IMU,
        "bank": bank,
        "channel_bitmask": bitmask,
        "n_channels": n_channels,
        "packet_size": compute_packet_size(n_channels),
        "cycle_bank": cycle_bank,
        "mux_id": mux_id,
        "metadata": metadata.copy(),
    }


def detect_format(filepath: Path) -> tuple:
    """Auto-detect the format of a SpikeGadgets dat file from its first header.

    Parameters
    ----------
    filepath : Path or str
        Path to a .dat file.

    Returns
    -------
    (bank_number, n_channels, packet_size, has_imu) : tuple[int, int, int, bool]
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)

    if len(header_bytes) < HEADER_SIZE:
        raise ValueError(f"File too small to contain a valid header: {filepath}")

    info = parse_header(header_bytes)
    return info["bank"], info["n_channels"], info["packet_size"], info["has_imu"]


def count_packets(filepath: Path, packet_size: int) -> int:
    """Count the number of complete packets in a file.

    Parameters
    ----------
    filepath : Path or str
        Path to a .dat file.
    packet_size : int
        Size of each packet in bytes.

    Returns
    -------
    int
        Number of complete packets (partial trailing packet is ignored).
    """
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    return file_size // packet_size


def extract_imu(filepath: Path) -> dict:
    """Extract IMU (accelerometer, gyroscope, magnetometer) data from a .dat file.

    IMU data is embedded in the packet headers of IMU-enabled recordings
    (magic 0x1655). The data is multiplexed across a 12-packet cycle:
    8 bank-rotation packets + 1 RFsync + 1 magnetometer + 1 gyroscope + 1 accelerometer.

    Each sensor packet carries X, Y, Z values in metadata int16 positions 0, 1, 2.

    Parameters
    ----------
    filepath : Path or str
        Path to a .dat file with IMU-enabled firmware.

    Returns
    -------
    dict with keys:
        - "accel": np.ndarray, shape (N, 3), int16 — accelerometer X, Y, Z
        - "gyro":  np.ndarray, shape (N, 3), int16 — gyroscope X, Y, Z
        - "mag":   np.ndarray, shape (N, 3), int16 — magnetometer X, Y, Z
        - "sampling_rate": float — effective IMU sampling rate (2500 Hz)
        - "channels": list[str] — channel names in order

    Raises
    ------
    ValueError
        If the file does not use IMU-enabled firmware.
    """
    filepath = Path(filepath)
    bank, n_channels, packet_size, has_imu = detect_format(filepath)

    if not has_imu:
        raise ValueError(
            f"File does not contain IMU data (magic 0x{MAGIC_STANDARD:04x}, "
            f"expected 0x{MAGIC_IMU:04x}): {filepath}"
        )

    file_size = filepath.stat().st_size
    n_packets = file_size // packet_size

    # Read all packet headers at once using strided access
    raw = np.fromfile(filepath, dtype=np.uint8, count=n_packets * packet_size)
    raw = raw.reshape(n_packets, packet_size)

    # Extract mux_id (byte 11) for every packet
    mux_ids = raw[:, CYCLE_MUX_OFFSET]

    # Extract metadata region (bytes 12-75) as int16 for every packet
    # metadata is 64 bytes = 32 int16 values
    meta_bytes = raw[:, METADATA_OFFSET:METADATA_OFFSET + 64]
    meta_int16 = np.ascontiguousarray(meta_bytes).view(DTYPE).reshape(n_packets, 32)

    result = {
        "sampling_rate": IMU_SAMPLING_RATE,
        "channels": IMU_CHANNELS,
    }

    for mux_id, (prefix, ix, iy, iz) in IMU_MUX_MAP.items():
        mask = mux_ids == mux_id
        sensor_meta = meta_int16[mask]
        result[prefix] = sensor_meta[:, [ix, iy, iz]].copy()

    return result
