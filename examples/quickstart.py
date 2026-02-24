"""Quickstart — load, inspect, and preprocess a SpikeGadgets recording.

Demonstrates the core API: loading a .dat file, inspecting its properties,
removing bad channels, and applying standard preprocessing.

Usage::

    python examples/quickstart.py <dat_file>

Example::

    python examples/quickstart.py recording.dat
"""

import sys
from pathlib import Path

import numpy as np

import chronic_pixel_pipeline as cpp

# ── Parse args ────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python quickstart.py <dat_file>")
    sys.exit(1)

dat_file = Path(sys.argv[1])

# ── 1. Load a recording ──────────────────────────────────────────────────────
print(f"Loading {dat_file.name}...")
rec = cpp.load_segment(dat_file)

print(f"  Channels:       {rec.get_num_channels()}")
print(f"  Samples:        {rec.get_num_samples()}")
print(f"  Duration:       {rec.get_total_duration():.1f} s")
print(f"  Sampling rate:  {rec.get_sampling_frequency()} Hz")
print(f"  µV/bit (gain):  {rec.get_channel_gains()[0]:.6f}")
print(f"  Bank number:    {rec.bank_number}")
print(f"  Channel IDs:    {rec.channel_ids[0]}..{rec.channel_ids[-1]}")

# ── 2. Inspect the file format ───────────────────────────────────────────────
bank, n_ch, pkt_size, has_imu = cpp.detect_format(dat_file)
print(f"\n  File format:")
print(f"    Bank:         {bank}")
print(f"    Channels:     {n_ch}")
print(f"    Packet size:  {pkt_size} bytes")
print(f"    Has IMU:      {has_imu}")

# ── 3. Read some traces ──────────────────────────────────────────────────────
# Raw int16 values
raw = rec.get_traces(start_frame=0, end_frame=300, return_in_uV=False)
print(f"\n  Raw traces shape: {raw.shape}, dtype: {raw.dtype}")
print(f"  Raw range: [{raw.min()}, {raw.max()}]")

# Scaled to µV
uv = rec.get_traces(start_frame=0, end_frame=300, return_in_uV=True)
print(f"  µV traces shape: {uv.shape}")
print(f"  µV range: [{uv.min():.2f}, {uv.max():.2f}]")

# ── 4. Remove bad channels ───────────────────────────────────────────────────
print("\nRemoving bad channels...")
rec_clean, good_ids, info = cpp.remove_bad_channels(rec)
print(f"  Dead:    {info['n_dead']}")
print(f"  Noisy:   {info['n_noisy']}")
print(f"  Kept:    {info['n_kept']}/{rec.get_num_channels()}")
print(f"  Median RMS: {info['median_rms_uv']:.2f} µV")

# ── 5. Preprocess ────────────────────────────────────────────────────────────
print("\nPreprocessing (bandpass 300-6000 Hz + IBL destripe)...")
rec_pp = cpp.preprocess(rec_clean)
print(f"  Output: {rec_pp.get_num_channels()} channels, "
      f"{rec_pp.get_total_duration():.1f}s")

# ── 6. Extract a time window ─────────────────────────────────────────────────
fs = rec.get_sampling_frequency()
total_dur = rec.get_total_duration()
if total_dur > 5:
    sf = int(2.0 * fs)
    ef = int(min(5.0, total_dur) * fs)
    rec_window = rec.frame_slice(start_frame=sf, end_frame=ef)
    print(f"\nExtracted window 2-{min(5.0, total_dur):.0f}s: "
          f"{rec_window.get_num_samples()} samples")

print("\nDone!")
