"""Post-spike-sorting analysis — example script.

Loads existing KS4 sorting results, builds a SortingAnalyzer with quality
metrics, classifies units, and saves summary figures. This example shows how
to work with sorting results after running the pipeline.

Usage::

    python examples/analyze_sorting.py <dat_file> <ks4_output_dir> [figure_dir]

Example::

    python examples/analyze_sorting.py recording.dat sorting_output/recording figures/
"""

import sys
from pathlib import Path

import numpy as np
import spikeinterface.full as si

import chronic_pixel_pipeline as cpp
from chronic_pixel_pipeline.analysis import (
    build_sorting_analyzer,
    classify_units,
    plot_summary,
)
from chronic_pixel_pipeline.diagnostics import plot_sorting_diagnostics

# ── Parse args ────────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: python analyze_sorting.py <dat_file> <ks4_output_dir> [figure_dir]")
    sys.exit(1)

dat_file = Path(sys.argv[1])
ks_output = Path(sys.argv[2])
fig_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("figures")

# ── 1. Load recording and sorting ────────────────────────────────────────────
print(f"Loading recording: {dat_file}")
rec = cpp.load_segment(dat_file)
sorting = si.read_kilosort(ks_output, remove_empty_units=True)
print(f"  Recording : {rec.get_num_channels()} ch, "
      f"{rec.get_num_samples()} samples, "
      f"{rec.get_total_duration():.1f} s")
print(f"  Sorting   : {len(sorting.unit_ids)} units, "
      f"{sum(sorting.count_num_spikes_per_unit().values())} total spikes")

# ── 2. Build SortingAnalyzer ─────────────────────────────────────────────────
print("\nBuilding SortingAnalyzer...")
analyzer = build_sorting_analyzer(sorting, rec)

# ── 3. Classify units ────────────────────────────────────────────────────────
good_ids, snr_thresh = classify_units(analyzer)
qm = analyzer.get_extension("quality_metrics").get_data()
print(f"  {len(good_ids)} / {len(qm)} units classified as 'good' "
      f"(SNR > {snr_thresh:.2f})")

# ── 4. Generate figures ──────────────────────────────────────────────────────
print(f"\nGenerating sorting diagnostics in {fig_dir}/...")
saved = plot_sorting_diagnostics(
    analyzer, sorting, rec, fig_dir / "diagnostics",
    recording_name=dat_file.stem,
)
print(f"  Saved {len(saved)} diagnostic figures")

# Also generate the standard summary plots
saved2 = plot_summary(
    analyzer, sorting, rec, fig_dir / "summary",
    good_ids=good_ids, snr_thresh=snr_thresh,
)
print(f"  Saved {len(saved2)} summary figures")

# ── 5. Print summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SORTING SUMMARY")
print("=" * 60)
print(f"Total units:         {len(sorting.unit_ids)}")
print(f"Good units:          {len(good_ids)}")
print(f"Recording duration:  {rec.get_total_duration():.1f} s")
print(f"Sampling rate:       {rec.get_sampling_frequency()} Hz")
print(f"Channels:            {rec.get_num_channels()}")
print(f"\nGood unit criteria: FR>0.1 Hz, SNR>{snr_thresh:.2f}, ISI viol<50%")

noise = analyzer.get_extension("noise_levels").get_data()
templates = analyzer.get_extension("templates").get_data()
ptps = np.max(np.ptp(templates, axis=1), axis=1)
print(f"\nMedian noise (µV):        {np.median(noise):.4f}")
print(f"Median template PTP (µV): {np.median(ptps):.2f}")

if good_ids:
    print(f"\nQuality metrics (good units):")
    print(qm.loc[good_ids, ["firing_rate", "snr", "isi_violations_ratio",
                             "presence_ratio"]].describe().round(2))
print("=" * 60)
