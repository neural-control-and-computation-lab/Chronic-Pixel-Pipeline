"""Timecourse bio-signal scan for SpikeGadgets Neuropixels recordings.

Slides a window across the full recording and computes per-window biological
signal quality metrics — no spike sorting, just fast signal-level analysis.

Usage::

    cpp-scan recording.dat
    cpp-scan recording.dat --output figures/scan
    cpp-scan recording.dat --window 3 --stride 1.5
"""

import argparse
import logging
import sys

import numpy as np


def main(argv=None):
    """Entry point for ``cpp-scan``."""
    parser = argparse.ArgumentParser(
        description="Scan a SpikeGadgets .dat file for biological signal quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  cpp-scan recording.dat\n"
            "  cpp-scan recording.dat --output figures/my_scan\n"
            "  cpp-scan recording.dat --window 3 --stride 1.5\n"
        ),
    )
    parser.add_argument("dat_file", help="Path to a .dat file")
    parser.add_argument(
        "--output", "-o", default="figures/timescan",
        help="Output directory for figures (default: figures/timescan)",
    )
    parser.add_argument(
        "--window", type=float, default=2.0,
        help="Analysis window duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--stride", type=float, default=1.0,
        help="Stride between windows in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--probe", default="NP1032",
        help="Probe part number: NP1032 (linear) or NP1030 (staggered). "
             "Default: NP1032",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args(argv)

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from pathlib import Path
    import chronic_pixel_pipeline as cpp
    from chronic_pixel_pipeline.diagnostics import (
        run_timescan,
        plot_timecourse_dashboard,
        plot_best_windows,
    )

    dat_file = Path(args.dat_file)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dat_file.exists():
        logger.error("File not found: %s", dat_file)
        sys.exit(1)

    # ── Load recording ────────────────────────────────────────────────────
    logger.info("Loading %s ...", dat_file.name)
    rec = cpp.load_segment(dat_file, probe_part_number=args.probe)
    fs = rec.get_sampling_frequency()
    n_ch = rec.get_num_channels()
    n_samp = rec.get_num_samples()
    duration_s = n_samp / fs
    ch_ids = np.array(rec.get_channel_ids())
    uv_per_bit = rec.get_channel_gains()[0]

    logger.info("  %d channels, %.1fs (%d samples), µV/bit=%.6f",
                n_ch, duration_s, n_samp, uv_per_bit)

    # ── Run timescan ──────────────────────────────────────────────────────
    scan_result = run_timescan(rec, window_s=args.window, stride_s=args.stride)

    # ── Print results ─────────────────────────────────────────────────────
    times = scan_result["times"]
    metrics = scan_result["metrics"]
    best_indices = scan_result["best_indices"]

    print(f"\n  Top 5 windows by bio-score:")
    print(f"  {'Rank':>4} {'Time':>8} {'Score':>7} {'Kurt95':>7} {'Refrac':>7} "
          f"{'Asym':>6} {'Xrate95':>8} {'Active%':>8} {'RMS':>7}")
    for rank, bi in enumerate(best_indices[:5]):
        m = metrics[bi]
        t = times[bi]
        print(f"  {rank + 1:>4} {t:>5.0f}-{t + args.window:>3.0f}s "
              f"{m['bio_score']:>7.3f} "
              f"{m['kurt_p95']:>7.1f} "
              f"{m['refrac_compliance']:>7.2f} "
              f"{m['mean_asymmetry']:>6.1f} "
              f"{m['xrate_p95']:>8.0f}/s "
              f"{m['active_frac'] * 100:>7.1f}% "
              f"{m['rms_median']:>7.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    name = dat_file.stem
    plot_timecourse_dashboard(
        scan_result, recording_name=f"{name} ({duration_s:.0f}s)",
        output_path=output_dir / f"timescan_{name}.png",
        window_s=args.window, stride_s=args.stride,
    )

    # Free CMR data from non-best windows to save memory before deep dive
    keep_set = set(best_indices[:3].tolist())
    for wi, m in enumerate(metrics):
        if wi not in keep_set:
            m.pop("cmr", None)
            for w in m.get("best_waveforms", []):
                w.pop("snips", None)

    plot_best_windows(
        scan_result, ch_ids, recording_name=f"{name} ({duration_s:.0f}s)",
        output_path=output_dir / f"timescan_{name}_best.png", fs=fs,
    )

    logger.info("Done. Figures saved to %s/", output_dir)


if __name__ == "__main__":
    main()
