"""Packet format constants and parsing for SpikeGadgets Neuropixels wireless data logger."""

import struct
from pathlib import Path

import numpy as np

# --- Format constants ---
MAGIC = 0x1155
HEADER_SIZE = 80  # bytes per packet header
SAMPLING_RATE = 30_000.0  # Hz (Neuropixels AP band)
DTYPE = np.dtype("<i2")  # int16, little-endian
BYTES_PER_SAMPLE = 2
CHANNELS_PER_BANK = 48  # each bit in the bank bitmask = 48 channels

# Header field byte offsets
MAGIC_OFFSET = 0  # 2 bytes, uint16 LE
BANK_OFFSET = 3  # 1 byte, probe group/bank number (0-11)
CHANNEL_BITMASK_OFFSET = 5  # 1 byte, popcount * 48 = n_channels
CYCLE_OFFSET = 10  # 2 bytes, uint16 LE, cycling group bitmask
METADATA_OFFSET = 12  # 64 bytes (32 x int16 status/metadata)
SEPARATOR_OFFSET = 76  # 4 bytes, zeros
DATA_OFFSET = 80  # neural data starts here


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
    dict with keys: magic, bank, n_channels, packet_size, cycle_mask, metadata
    """
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError(
            f"Header must be at least {HEADER_SIZE} bytes, got {len(header_bytes)}"
        )

    magic = struct.unpack_from("<H", header_bytes, MAGIC_OFFSET)[0]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic number: 0x{magic:04x} (expected 0x{MAGIC:04x})")

    bank = header_bytes[BANK_OFFSET]
    bitmask = header_bytes[CHANNEL_BITMASK_OFFSET]
    n_channels = compute_n_channels(bitmask)
    cycle_mask = struct.unpack_from("<H", header_bytes, CYCLE_OFFSET)[0]
    metadata = np.frombuffer(header_bytes, dtype=DTYPE, count=32, offset=METADATA_OFFSET)

    return {
        "magic": magic,
        "bank": bank,
        "channel_bitmask": bitmask,
        "n_channels": n_channels,
        "packet_size": compute_packet_size(n_channels),
        "cycle_mask": cycle_mask,
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
    (bank_number, n_channels, packet_size) : tuple[int, int, int]
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)

    if len(header_bytes) < HEADER_SIZE:
        raise ValueError(f"File too small to contain a valid header: {filepath}")

    info = parse_header(header_bytes)
    return info["bank"], info["n_channels"], info["packet_size"]


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
