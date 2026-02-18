# Chronic Pixel Pipeline

Python toolbox for processing chronic Neuropixels recordings from the SpikeGadgets wireless data logger. Built on [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

## Features

- **SpikeGadgets .dat reader** — auto-detects packet format (80-byte header + N channels x int16), bank number, and channel count
- **IMU extraction** — accelerometer, gyroscope, and magnetometer at 2500 Hz from IMU-enabled firmware recordings
- **Probe geometry** — NP1032 (linear) and NP1030 (staggered) NHP long probe layouts with 4,416 electrodes across 12 banks
- **Two analysis modes**:
  - **Per-segment**: each .dat file as a separate SpikeInterface recording
  - **Concatenated**: all banks combined channel-wise into a single recording (trimmed to shortest)
- **Post-sorting analysis** — SortingAnalyzer with quality metrics, waveforms, correlograms, template similarity, and 10+ summary visualizations

## Installation

```bash
pip install -e .                  # core (reader + probe geometry)
pip install -e ".[analysis]"      # + matplotlib for plotting
pip install -e ".[sorting]"       # + kilosort4 + pytorch
pip install -e ".[all]"           # everything
```

## Quick Start

### Load a recording

```python
import chronic_pixel_pipeline as cpp

# Single segment (one .dat file = one probe bank)
rec = cpp.load_segment("Data/test_segment1.dat")

# All segments as a dict keyed by bank number
segments = cpp.load_all_segments("Data/")

# All banks concatenated channel-wise (4,416 channels)
full_rec = cpp.load_concatenated("Data/")
```

### Extract IMU data

```python
import chronic_pixel_pipeline as cpp

# Load accelerometer, gyroscope, and magnetometer from IMU-enabled firmware
imu = cpp.load_imu("Data/testgyro_segment1.dat")

imu["accel"]          # (N, 3) int16 — accelerometer X, Y, Z
imu["gyro"]           # (N, 3) int16 — gyroscope X, Y, Z
imu["mag"]            # (N, 3) int16 — magnetometer X, Y, Z
imu["sampling_rate"]  # 2500.0 Hz
```

### Run spike sorting

```python
import spikeinterface.full as si

rec = cpp.load_segment("Data/test_segment1.dat")

sorting = si.run_sorter(
    'kilosort4', rec,
    folder='output',
    torch_device='mps',   # Apple Silicon GPU ('cuda' for NVIDIA)
    batch_size=30000,
    nblocks=0,
)
```

### Post-sorting analysis

```python
from chronic_pixel_pipeline.analysis import (
    build_sorting_analyzer,
    classify_units,
    plot_summary,
)

# Build analyzer with all extensions (waveforms, quality metrics, etc.)
analyzer = build_sorting_analyzer(sorting, rec)

# Classify units by quality
good_ids, snr_thresh = classify_units(analyzer)

# Generate 10+ summary figures
plot_summary(analyzer, sorting, rec, "figures/",
             good_ids=good_ids, snr_thresh=snr_thresh)
```

## Data Format

SpikeGadgets wireless data logger files (`.dat`) use a packet-based binary format:

| Field | Offset | Size | Description |
|-------|--------|------|-------------|
| Magic | 0 | 2 bytes | `0x1155` (standard) or `0x1655` (IMU-enabled) |
| Bank | 3 | 1 byte | Probe bank number (0-11) |
| Channel bitmask | 5 | 1 byte | popcount x 48 = n_channels |
| Cycle bank | 10 | 1 byte | Bank rotation bitmask |
| Cycle mux ID | 11 | 1 byte | Multiplexed device ID (0=neural, 1=RFsync, 2=mag, 4=gyro, 8=accel) |
| Metadata | 12 | 64 bytes | 32 x int16 (IMU X/Y/Z in positions 0-2 on sensor packets) |
| Counter | 76 | 4 bytes | Packet counter |
| Neural data | 80 | N x 2 bytes | int16 LE, one sample per channel |

- Banks 0-10: 384 channels (848 bytes/packet)
- Bank 11: 192 channels (464 bytes/packet)
- 30 kHz sampling rate

### IMU Multiplexing (firmware 0x1655)

IMU-enabled recordings interleave sensor data in a 12-packet cycle:

| Packet | Byte 11 | Content |
|--------|---------|---------|
| 0-7 | 0x00 | Bank rotation (neural data in all packets) |
| 8 | 0x01 | RFsync |
| 9 | 0x02 | Magnetometer X/Y/Z in metadata[0:3] |
| 10 | 0x04 | Gyroscope X/Y/Z in metadata[0:3] |
| 11 | 0x08 | Accelerometer X/Y/Z in metadata[0:3] |

Effective IMU sampling rate: 30,000 / 12 = **2,500 Hz** per sensor.

## Probe Geometry

Neuropixels NHP long probe (NP1032/NP1030):
- 4,416 electrodes total
- 12 banks (11 x 384 + 1 x 192 channels)
- 2 columns, 20 um vertical pitch
- NP1032: 103 um horizontal pitch (linear)
- NP1030: 87 um horizontal pitch (staggered)

## Package Structure

```
src/chronic_pixel_pipeline/
    __init__.py          # Public API
    packet.py            # Packet format constants, parsing, and IMU extraction
    recording.py         # SpikeInterface recording extractor
    probe_geometry.py    # NP1032/NP1030 probe layouts
    io.py                # High-level load functions (load_segment, load_imu, etc.)
    analysis.py          # Post-sorting analysis and visualization

examples/
    analyze_sorting.py   # Post-sorting analysis example
    test_ks4_mps.py      # Kilosort 4 on Apple MPS GPU
    plot_imu.py          # IMU extraction and plotting
```

## License

MIT
