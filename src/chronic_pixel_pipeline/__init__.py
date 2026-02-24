"""Chronic Pixel Pipeline — SpikeGadgets Neuropixels wireless data toolbox.

A Python library for reading, preprocessing, spike sorting, and visualizing
chronic Neuropixels long probe recordings from SpikeGadgets wireless data
loggers. Built on SpikeInterface.

Quick start::

    import chronic_pixel_pipeline as cpp

    # Load a recording
    rec = cpp.load_segment("recording.dat")

    # Remove bad channels and preprocess
    rec_clean, good_ids, info = cpp.remove_bad_channels(rec)
    rec_pp = cpp.preprocess(rec_clean)

See :mod:`chronic_pixel_pipeline.diagnostics` for visualization tools and
:mod:`chronic_pixel_pipeline.analysis` for post-sorting analysis.
"""

__version__ = "0.1.0"

# ── Data loading ──────────────────────────────────────────────────────────────
from .io import load_all_segments, load_concatenated, load_imu, load_segment

# ── Low-level packet access ───────────────────────────────────────────────────
from .packet import count_packets, detect_format, extract_imu, parse_header

# ── Probe geometry ────────────────────────────────────────────────────────────
from .probe_geometry import electrode_positions, make_full_probe, make_probe_for_bank

# ── Recording extractor ──────────────────────────────────────────────────────
from .recording import SpikeGadgetsNPXRecording, SpikeGadgetsNPXRecordingSegment

# ── Preprocessing ─────────────────────────────────────────────────────────────
from .preprocessing import preprocess, remove_bad_channels

__all__ = [
    # Loading
    "load_segment",
    "load_all_segments",
    "load_concatenated",
    "load_imu",
    # Recording
    "SpikeGadgetsNPXRecording",
    "SpikeGadgetsNPXRecordingSegment",
    # Packet
    "detect_format",
    "parse_header",
    "count_packets",
    "extract_imu",
    # Probe geometry
    "electrode_positions",
    "make_probe_for_bank",
    "make_full_probe",
    # Preprocessing
    "remove_bad_channels",
    "preprocess",
]
