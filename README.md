# Chronic Pixel Pipeline

Python toolbox for processing chronic Neuropixels recordings from the SpikeGadgets wireless data logger. Built on [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

## Features

- **SpikeGadgets .dat reader** — auto-detects packet format (80-byte header + N channels x int16), bank number, and channel count
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

Or run the example script:

```bash
python analyze_sorting.py
```

## Data Format

SpikeGadgets wireless data logger files (`.dat`) use a packet-based binary format:

| Field | Offset | Size | Description |
|-------|--------|------|-------------|
| Magic | 0 | 2 bytes | `0x1155` (uint16 LE) |
| Bank | 3 | 1 byte | Probe bank number (0-11) |
| Channel bitmask | 5 | 1 byte | popcount x 48 = n_channels |
| Cycle mask | 10 | 2 bytes | Cycling group bitmask |
| Metadata | 12 | 64 bytes | 32 x int16 status values |
| Separator | 76 | 4 bytes | Zeros |
| Neural data | 80 | N x 2 bytes | int16 LE, one sample per channel |

- Banks 0-10: 384 channels (848 bytes/packet)
- Bank 11: 192 channels (464 bytes/packet)
- 30 kHz sampling rate

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
    packet.py            # Packet format constants and parsing
    recording.py         # SpikeInterface recording extractor
    probe_geometry.py    # NP1032/NP1030 probe layouts
    io.py                # High-level load functions
    analysis.py          # Post-sorting analysis and visualization
```

## License

MIT
