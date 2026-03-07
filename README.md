# Chronic Pixel Pipeline

A Python toolbox for processing chronic Neuropixels long probe recordings from SpikeGadgets wireless data loggers. Built on [SpikeInterface](https://spikeinterface.readthedocs.io/).

## Features

- **Read SpikeGadgets `.dat` files** — custom packetized binary format with 80-byte headers
- **Automatic format detection** — bank number, channel count, firmware version, IMU presence
- **Neuropixels NHP probe geometry** — NP1032 (linear) and NP1030 (staggered), 4,416 electrodes across 12 banks
- **Correct µV scaling** — 10-bit ADC conversion with configurable AP gain
- **IMU extraction** — accelerometer, gyroscope, magnetometer at 2500 Hz
- **Preprocessing** — bad channel removal, bandpass filtering, IBL destriping
- **Bio-signal timescan** — scan full recordings for biological signal quality without spike sorting
- **Spike sorting pipeline** — Kilosort 4 integration with full diagnostic figures
- **12 diagnostic figure types** — waveforms, autocorrelograms, ISI histograms, spatial footprints, and more
- **CLI tools** — `cpp-scan` and `cpp-sort` for command-line operation

## Installation

```bash
pip install -e .                  # core (reader + preprocessing)
pip install -e ".[analysis]"      # + matplotlib for plotting
pip install -e ".[sorting]"       # + kilosort4 + pytorch
pip install -e ".[all]"           # everything
```

## Quick Start

### Python API

```python
import chronic_pixel_pipeline as cpp

# Load a recording
rec = cpp.load_segment("recording.dat")
print(f"{rec.get_num_channels()} channels, {rec.get_total_duration():.1f}s")

# Remove bad channels
rec_clean, good_ids, info = cpp.remove_bad_channels(rec)

# Preprocess (bandpass 300-6000 Hz + IBL destripe)
rec_pp = cpp.preprocess(rec_clean)

# Extract a time window
fs = rec.get_sampling_frequency()
rec_window = rec.frame_slice(start_frame=int(5*fs), end_frame=int(19*fs))
```

### CLI Tools

**Timecourse bio-signal scan** — find the best segments in a recording:
```bash
cpp-scan recording.dat
cpp-scan recording.dat --output figures/my_scan --window 3 --stride 1.5
```

**Full spike sorting pipeline** — preprocess, sort with KS4, generate diagnostics:
```bash
cpp-sort recording.dat
cpp-sort recording.dat --t-start 5 --t-end 19
cpp-sort recording.dat --device cuda --output my_results
```

### Load All Banks

```python
# All segments as a dict keyed by bank number
segments = cpp.load_all_segments("Data/")

# All banks concatenated channel-wise (4,416 channels)
full_rec = cpp.load_concatenated("Data/")
```

### Extract IMU Data

```python
imu = cpp.load_imu("recording.dat")  # IMU-enabled firmware only

imu["accel"]          # (N, 3) int16 — accelerometer X, Y, Z
imu["gyro"]           # (N, 3) int16 — gyroscope X, Y, Z
imu["mag"]            # (N, 3) int16 — magnetometer X, Y, Z
imu["sampling_rate"]  # 2500.0 Hz
```

### Post-Sorting Analysis

```python
from chronic_pixel_pipeline.analysis import build_sorting_analyzer, classify_units
from chronic_pixel_pipeline.diagnostics import plot_sorting_diagnostics

# Build analyzer with all extensions
analyzer = build_sorting_analyzer(sorting, rec)

# Classify units by quality
good_ids, snr_thresh = classify_units(analyzer)

# Generate 12 diagnostic figures
plot_sorting_diagnostics(analyzer, sorting, rec, "figures/diagnostics/")
```

### Bio-Signal Timescan

```python
from chronic_pixel_pipeline.diagnostics import (
    run_timescan, plot_timecourse_dashboard, plot_best_windows
)

# Scan for biological signal quality (no spike sorting needed)
result = run_timescan(rec, window_s=2.0, stride_s=1.0)

# Visualize
plot_timecourse_dashboard(result, "my recording", "figures/timescan.png")
plot_best_windows(result, rec.channel_ids, "my recording", "figures/best.png")
```

## Data Format

SpikeGadgets wireless data logger files (`.dat`) use a packetized binary format:

| Field | Offset | Size | Description |
|-------|--------|------|-------------|
| Magic | 0 | 2 bytes | Format identifier (0x1155 standard, 0x1655+ IMU) |
| Bank | 3 | 1 byte | Probe bank number (0-11) |
| Channel bitmask | 5 | 1 byte | Active channel banks (popcount x 48 = n_channels) |
| Cycle bank | 10 | 1 byte | Bank rotation bitmask |
| Cycle mux ID | 11 | 1 byte | Multiplexed device ID |
| Metadata | 12 | 64 bytes | 32 x int16 (IMU X/Y/Z on sensor packets) |
| Counter | 76 | 4 bytes | Packet counter |
| Neural data | 80 | N x 2 bytes | int16 LE, one sample per channel |

- Sampling rate: 30 kHz
- Banks 0-10: 384 channels (848-byte packets)
- Bank 11: 192 channels (464-byte packets)

### µV Scaling

The Neuropixels 1.0 ASIC has a 10-bit SAR ADC with 1.2V full-scale range.
SpikeGadgets stores the 10-bit codes left-shifted by 6 in int16:

```
µV = raw_int16 x 1,200,000 / (65,536 x AP_gain)
```

At the default gain of 1000: **0.018311 µV/bit**.

### IMU Multiplexing

IMU-enabled recordings (magic 0x1655+) interleave sensor data in a 12-packet cycle:

| Packet | Mux ID | Content |
|--------|--------|---------|
| 0-7 | 0x00 | Bank rotation (neural data in all) |
| 8 | 0x01 | RFsync |
| 9 | 0x02 | Magnetometer X/Y/Z |
| 10 | 0x04 | Gyroscope X/Y/Z |
| 11 | 0x08 | Accelerometer X/Y/Z |

Effective IMU sampling rate: 30,000 / 12 = **2,500 Hz**.

## Survey Sorting Pipeline

Sort SpikeGLX survey-mode recordings across all 12 banks of the NP1032 probe, with Kilosort 4 drift correction.

### Setup (Windows with CUDA)

```bat
setup_env.bat
conda activate cpp
```

### Setup (Linux / macOS)

```bash
bash setup_env.sh
conda activate cpp
```

### Running

```bash
# Sort a single survey by date key
python sort_survey.py 20260226

# Sort ALL known surveys
python sort_survey.py all

# Sort + track neurons across days (full pipeline)
run_all.bat          # Windows
bash run_all.sh      # Linux / macOS
```

Set the `DATA_ROOT` environment variable to point to the directory containing the SpikeGLX survey folders (e.g., `SvyPrb_2026-02-26T11.58.07_g0/`). On Windows this defaults to `%USERPROFILE%\data\TRODES_test`.

### What it does

1. **Parses** the SpikeGLX survey timetable (`svySBTT`) from the `.meta` file
2. **Extracts** each bank's stable data window (skipping settle periods)
3. **Assigns** correct NP1032 probe geometry per bank
4. **Preprocesses** with bandpass + IBL destriping
5. **Spike-sorts** each bank independently with Kilosort 4 (CUDA auto-detected, drift correction enabled)
6. **Builds** SortingAnalyzer with quality metrics per bank
7. **Merges** all bank sortings into one full-probe summary
8. **Generates** diagnostic figures: full-probe overview, drift correction, spike depth scatter

### Output

```
sorting_output/survey_YYYYMMDD/
    bank00_ks4/          # Per-bank KS4 output
        sorter_output/   # Raw KS4 files (ops.npy with drift data)
        analyzer_cache/  # SortingAnalyzer cache (templates, quality metrics)
    bank01_ks4/
    ...
    merged/
        merged_units.csv # All units across banks with depth, amplitude, quality

figures/sorting/survey_YYYYMMDD/
    survey_full_probe.png  # Unit locations, firing rates, amplitudes by depth
    drift_correction.png   # Drift amount over time per bank
    drift_scatter.png      # Spike depth scatter over time per bank
```

## Cross-Day Neuron Tracking

Track neurons across multiple survey days using [UnitMatch](https://github.com/EnnyvanBeest/UnitMatch) (Naive Bayes waveform matching).

```bash
# Track across all 4 survey dates
python track_neurons.py

# Track specific dates
python track_neurons.py 20260226 20260302
```

### Output

```
sorting_output/tracking/
    tracking_summary.csv   # All matched pairs with probabilities and depths
    bank00/                # Per-bank UnitMatch results
        prob_matrix.npy    # Match probability matrix
        UIDs.npy           # Unique neuron IDs

figures/sorting/tracking/
    tracking_summary.png   # Matches per bank, confidence distribution, depth consistency
    tracking_full_probe.png # Full-probe tracking map across days
```

## Package Structure

```
src/chronic_pixel_pipeline/
    __init__.py          # Public API exports
    packet.py            # Packet format constants and parsing
    recording.py         # SpikeInterface recording extractor
    io.py                # High-level loading API
    probe_geometry.py    # NP1032/NP1030 probe geometry
    preprocessing.py     # Bad channel removal, bandpass, destripe
    analysis.py          # Post-sorting analysis (SortingAnalyzer, quality metrics)
    diagnostics.py       # Diagnostic visualization (sorting + timescan)
    scripts/
        timescan.py      # CLI: cpp-scan
        sort.py          # CLI: cpp-sort

sort_survey.py           # Survey-mode sorting pipeline (all banks + drift correction)
track_neurons.py         # Cross-day neuron tracking with UnitMatch
setup_env.sh / .bat      # Environment setup (conda + CUDA)
run_all.sh / .bat        # Full pipeline orchestration
```

## API Reference

### Loading

| Function | Description |
|----------|-------------|
| `cpp.load_segment(path)` | Load a single `.dat` file as a SpikeInterface recording |
| `cpp.load_all_segments(dir)` | Load all `.dat` files, keyed by bank number |
| `cpp.load_concatenated(dir)` | Load and aggregate all banks into one recording |
| `cpp.load_imu(path)` | Extract IMU data (accel/gyro/mag) from IMU-enabled files |

### Preprocessing

| Function | Description |
|----------|-------------|
| `cpp.remove_bad_channels(rec)` | Remove dead/noisy channels by raw RMS |
| `cpp.preprocess(rec)` | Bandpass filter + IBL destripe |

### Analysis (`chronic_pixel_pipeline.analysis`)

| Function | Description |
|----------|-------------|
| `build_sorting_analyzer(sorting, rec)` | Create SortingAnalyzer with all standard extensions |
| `classify_units(analyzer)` | Classify units as good/bad by quality metrics |
| `plot_summary(analyzer, sorting, rec, dir)` | Generate summary figures |

### Diagnostics (`chronic_pixel_pipeline.diagnostics`)

| Function | Description |
|----------|-------------|
| `plot_sorting_diagnostics(analyzer, ...)` | 12 diagnostic figure types |
| `run_timescan(rec)` | Scan recording for bio-signal quality |
| `compute_window_metrics(raw, ...)` | Bio-quality metrics for a single window |
| `plot_timecourse_dashboard(result, ...)` | Timecourse visualization |
| `plot_best_windows(result, ...)` | Deep-dive on best time windows |

## Probe Geometry

Neuropixels NHP long probe (NP1032/NP1030):
- 4,416 electrodes total
- 12 banks (11 x 384 + 1 x 192 channels)
- 2 columns, 20 µm vertical pitch
- NP1032: 103 µm horizontal pitch (linear)
- NP1030: 87 µm horizontal pitch (staggered)

## License

MIT
