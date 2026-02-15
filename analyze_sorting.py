"""Post-spike-sorting analysis — example script.

Loads KS4 results, builds a SortingAnalyzer with quality metrics,
classifies units, and saves summary figures.

Usage:
    python analyze_sorting.py
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from pathlib import Path
import spikeinterface.full as si
import chronic_pixel_pipeline as cpp
from chronic_pixel_pipeline.analysis import (
    build_sorting_analyzer,
    classify_units,
    plot_summary,
)

# ── paths ────────────────────────────────────────────────────────────────────
data_dir   = Path("Data")
ks_output  = Path("test_output/kilosort4_output")
fig_dir    = Path("figures")

# ── 1. Load recording and sorting ───────────────────────────────────────────
print("Loading recording and sorting...")
rec = cpp.load_segment(sorted(data_dir.glob("*.dat"))[0])
sorting = si.read_kilosort(ks_output, remove_empty_units=True)
print(f"  Recording : {rec.get_num_channels()} ch, "
      f"{rec.get_num_samples()} samples, "
      f"{rec.get_total_duration():.1f} s")
print(f"  Sorting   : {len(sorting.unit_ids)} units, "
      f"{sum(sorting.count_num_spikes_per_unit().values())} total spikes")

# ── 2. Build SortingAnalyzer with all extensions ────────────────────────────
print("\nBuilding SortingAnalyzer...")
analyzer = build_sorting_analyzer(sorting, rec)
print("  Done.")

# ── 3. Classify units ───────────────────────────────────────────────────────
good_ids, snr_thresh = classify_units(analyzer)
qm = analyzer.get_extension("quality_metrics").get_data()
print(f"\n  {len(good_ids)} / {len(qm)} units classified as 'good' "
      f"(SNR > {snr_thresh:.2f})")

# ── 4. Generate summary figures ─────────────────────────────────────────────
print(f"\nGenerating figures in {fig_dir}/...")
saved = plot_summary(analyzer, sorting, rec, fig_dir,
                     good_ids=good_ids, snr_thresh=snr_thresh)
for p in saved:
    print(f"  {p}")

# ── 5. Print summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SORTING SUMMARY")
print("=" * 60)
print(f"Total units:         {len(sorting.unit_ids)}")
print(f"Good units:          {len(good_ids)}")
print(f"Recording duration:  {rec.get_total_duration():.1f} s")
print(f"Sampling rate:       {rec.get_sampling_frequency()} Hz")
print(f"Channels:            {rec.get_num_channels()}")
print(f"\nGood unit criteria: FR>0.1 Hz, SNR>{snr_thresh:.2f}, ISI viol<50%")
if good_ids:
    print(f"\nQuality metrics summary (good units only):")
    print(qm.loc[good_ids, ['firing_rate', 'snr', 'isi_violations_ratio',
                             'presence_ratio']].describe().round(2))
print(f"\nFigures saved to: {fig_dir.resolve()}")
print("=" * 60)
