"""High-level API for loading SpikeGadgets Neuropixels recordings."""

from pathlib import Path

from .packet import detect_format, extract_imu
from .recording import SpikeGadgetsNPXRecording


def load_segment(file_path, probe_part_number="NP1032", attach_probe=True,
                  ap_gain=1000):
    """Load a single .dat segment file as a SpikeInterface recording.

    Parameters
    ----------
    file_path : str or Path
        Path to a single .dat file.
    probe_part_number : str
        Neuropixels probe part number for geometry. Default "NP1032" (NHP long linear).
        Use "NP1030" for the staggered variant.
    attach_probe : bool
        If True and probeinterface is installed, attach probe geometry.
    ap_gain : int
        Neuropixels AP band gain setting. Default 1000.
        Common values: 50, 125, 250, 500, 1000, 1500, 2000, 3000.

    Returns
    -------
    SpikeGadgetsNPXRecording
        Recording with channel IDs reflecting electrode positions on the probe.
    """
    return SpikeGadgetsNPXRecording(
        file_path, probe_part_number=probe_part_number, attach_probe=attach_probe,
        ap_gain=ap_gain,
    )


def load_all_segments(data_dir, file_pattern="*.dat", probe_part_number="NP1032",
                      attach_probe=True):
    """Load all .dat files in a directory, returning a dict keyed by bank number.

    Files are auto-sorted by bank number extracted from their headers.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .dat files.
    file_pattern : str
        Glob pattern for finding data files. Default "*.dat".
    probe_part_number : str
        Neuropixels probe part number for geometry.
    attach_probe : bool
        If True and probeinterface is installed, attach probe geometry.

    Returns
    -------
    dict[int, SpikeGadgetsNPXRecording]
        Mapping from bank number (0-11) to recording objects.
    """
    data_dir = Path(data_dir)
    dat_files = sorted(data_dir.glob(file_pattern))

    if not dat_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {data_dir}")

    recordings = {}
    for fpath in dat_files:
        bank, _, _, _ = detect_format(fpath)
        rec = SpikeGadgetsNPXRecording(
            fpath, probe_part_number=probe_part_number, attach_probe=attach_probe
        )
        recordings[bank] = rec

    return recordings


def load_concatenated(data_dir, file_pattern="*.dat", probe_part_number="NP1032",
                      attach_probe=True):
    """Load all .dat files and combine by channel into one recording.

    All segments are trimmed to the length of the shortest segment, then
    combined using spikeinterface.core.aggregate_channels() to produce
    a single recording with all electrode channels (e.g., 4,416 for the
    full NHP long probe).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .dat files.
    file_pattern : str
        Glob pattern for finding data files. Default "*.dat".
    probe_part_number : str
        Neuropixels probe part number for geometry.
    attach_probe : bool
        If True and probeinterface is installed, attach probe geometry.

    Returns
    -------
    BaseRecording
        Aggregated recording with all electrode channels.
    """
    from spikeinterface.core import aggregate_channels

    segments = load_all_segments(
        data_dir, file_pattern, probe_part_number, attach_probe=attach_probe
    )

    # Sort by bank number for consistent channel ordering
    sorted_banks = sorted(segments.keys())
    recording_list = [segments[bank] for bank in sorted_banks]

    # Trim all recordings to the shortest length
    min_samples = min(rec.get_num_samples(0) for rec in recording_list)
    trimmed = []
    for rec in recording_list:
        if rec.get_num_samples(0) > min_samples:
            rec = rec.frame_slice(start_frame=0, end_frame=min_samples)
        trimmed.append(rec)

    return aggregate_channels(trimmed)


def load_imu(file_path):
    """Extract IMU data (accelerometer, gyroscope, magnetometer) from a .dat file.

    Only works with IMU-enabled firmware recordings (magic 0x1655). The IMU
    data is multiplexed in the packet headers at 2500 Hz (30 kHz / 12-packet cycle).

    Parameters
    ----------
    file_path : str or Path
        Path to a .dat file recorded with IMU-enabled firmware.

    Returns
    -------
    dict with keys:
        - "accel": np.ndarray, shape (N, 3), int16 — accelerometer X, Y, Z
        - "gyro":  np.ndarray, shape (N, 3), int16 — gyroscope X, Y, Z
        - "mag":   np.ndarray, shape (N, 3), int16 — magnetometer X, Y, Z
        - "sampling_rate": float — 2500.0 Hz
        - "channels": list[str] — channel names

    Raises
    ------
    ValueError
        If the file does not use IMU-enabled firmware.
    """
    return extract_imu(file_path)
