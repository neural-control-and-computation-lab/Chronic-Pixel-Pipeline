"""Chronic Pixel Pipeline — SpikeGadgets Neuropixels wireless data reader."""

__version__ = "0.1.0"

from .io import load_all_segments, load_concatenated, load_imu, load_segment
from .packet import count_packets, detect_format, extract_imu, parse_header
from .probe_geometry import electrode_positions, make_full_probe, make_probe_for_bank
from .recording import SpikeGadgetsNPXRecording, SpikeGadgetsNPXRecordingSegment

__all__ = [
    "load_segment",
    "load_all_segments",
    "load_concatenated",
    "load_imu",
    "SpikeGadgetsNPXRecording",
    "SpikeGadgetsNPXRecordingSegment",
    "detect_format",
    "parse_header",
    "count_packets",
    "extract_imu",
    "electrode_positions",
    "make_probe_for_bank",
    "make_full_probe",
]
