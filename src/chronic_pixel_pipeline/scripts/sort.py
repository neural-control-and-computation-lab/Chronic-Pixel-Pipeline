"""Full spike sorting pipeline for SpikeGadgets Neuropixels recordings.

Loads a .dat file, removes bad channels, applies preprocessing (notch +
bandpass + CMR/destripe), runs Kilosort 4, and generates diagnostic figures.

Usage::

    cpp-sort recording.dat
    cpp-sort recording.dat --t-start 5 --t-end 19
    cpp-sort recording.dat --device cuda --output my_results
    cpp-sort recording.dat --ap-gain 250 --notch-freqs 1602 3204 4806 --cmr
"""

import argparse
import logging
import os
import sys

# Prevent segfault from duplicate OpenMP runtimes (PyTorch + FAISS on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main(argv=None):
    """Entry point for ``cpp-sort``."""
    parser = argparse.ArgumentParser(
        description="Run the full spike sorting pipeline on a SpikeGadgets .dat file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  cpp-sort recording.dat\n"
            "  cpp-sort recording.dat --t-start 5 --t-end 19\n"
            "  cpp-sort recording.dat --device cuda\n"
            "  cpp-sort recording.dat --ap-gain 250 --notch-freqs 1602 3204 4806 --cmr\n"
            "  cpp-sort recording.dat --no-sort  # preprocessing + diagnostics only\n"
        ),
    )
    parser.add_argument("dat_file", help="Path to a .dat file")
    parser.add_argument(
        "--output", "-o", default="sorting_output",
        help="Output directory for sorting results (default: sorting_output)",
    )
    parser.add_argument(
        "--figures", default="figures/sorting",
        help="Output directory for diagnostic figures (default: figures/sorting)",
    )
    parser.add_argument(
        "--t-start", type=float, default=None,
        help="Start time in seconds (default: beginning of recording)",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="End time in seconds (default: end of recording)",
    )
    parser.add_argument(
        "--freq-min", type=float, default=300,
        help="Bandpass high-pass cutoff in Hz (default: 300)",
    )
    parser.add_argument(
        "--freq-max", type=float, default=6000,
        help="Bandpass low-pass cutoff in Hz (default: 6000)",
    )
    parser.add_argument(
        "--no-destripe", action="store_true",
        help="Skip IBL destriping (highpass spatial filter)",
    )
    parser.add_argument(
        "--cmr", action="store_true",
        help="Use common median reference instead of IBL destriping",
    )
    parser.add_argument(
        "--notch-freqs", type=float, nargs="+", default=None,
        help="Notch filter frequencies in Hz (e.g., 1602 3204 4806 for "
             "SpikeGadgets artifact removal)",
    )
    parser.add_argument(
        "--notch-q", type=float, default=30,
        help="Notch filter quality factor (default: 30)",
    )
    parser.add_argument(
        "--ap-gain", type=int, default=1000,
        help="AP band gain setting (default: 1000). Use 250 if gain "
             "mismatch is suspected.",
    )
    parser.add_argument(
        "--probe", default="NP1032",
        help="Probe part number: NP1032 or NP1030. Default: NP1032",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device for Kilosort 4: cpu, cuda, or mps. Default: cpu",
    )
    parser.add_argument(
        "--no-sort", action="store_true",
        help="Skip spike sorting (run preprocessing + bad channel removal only)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=30000,
        help="KS4 batch size (default: 30000)",
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
    import numpy as np
    import chronic_pixel_pipeline as cpp
    from chronic_pixel_pipeline.preprocessing import remove_bad_channels, preprocess
    from chronic_pixel_pipeline.analysis import build_sorting_analyzer
    from chronic_pixel_pipeline.diagnostics import plot_sorting_diagnostics

    dat_file = Path(args.dat_file)
    output_dir = Path(args.output)
    fig_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not dat_file.exists():
        logger.error("File not found: %s", dat_file)
        sys.exit(1)

    name = dat_file.stem
    t_label = ""

    # -- 1. Load ---------------------------------------------------------------
    logger.info("Loading %s (ap_gain=%d)...", dat_file.name, args.ap_gain)
    rec_full = cpp.load_segment(dat_file, probe_part_number=args.probe,
                                ap_gain=args.ap_gain)
    fs = rec_full.get_sampling_frequency()
    n_total = rec_full.get_num_samples()
    total_dur = n_total / fs

    logger.info("  %d channels, %.1fs (%d samples), uV/bit=%.6f",
                rec_full.get_num_channels(), total_dur, n_total,
                rec_full.get_channel_gains()[0])

    # -- 2. Extract time window ------------------------------------------------
    rec = rec_full
    if args.t_start is not None or args.t_end is not None:
        t_start = args.t_start if args.t_start is not None else 0.0
        t_end = args.t_end if args.t_end is not None else total_dur
        sf = int(t_start * fs)
        ef = min(int(t_end * fs), n_total)
        rec = rec_full.frame_slice(start_frame=sf, end_frame=ef)
        t_label = f" (s{t_start:.0f}-{t_end:.0f})"
        logger.info("  Extracted seconds %.1f-%.1f: %d samples, %.1fs",
                     t_start, t_end, rec.get_num_samples(),
                     rec.get_total_duration())

    # -- 3. Remove bad channels ------------------------------------------------
    logger.info("Removing bad channels...")
    rec_clean, good_ch_ids, bad_info = remove_bad_channels(rec)
    logger.info("  Removed %d (dead=%d, noisy=%d), keeping %d channels",
                bad_info["n_removed"], bad_info["n_dead"],
                bad_info["n_noisy"], bad_info["n_kept"])

    # -- 4. Preprocess ---------------------------------------------------------
    use_cmr = args.cmr
    use_destripe = not args.no_destripe and not use_cmr
    pp_parts = [f"bandpass {args.freq_min}-{args.freq_max} Hz"]
    if args.notch_freqs:
        pp_parts.insert(0, f"notch {args.notch_freqs}")
    if use_cmr:
        pp_parts.append("CMR")
    elif use_destripe:
        pp_parts.append("destripe")
    logger.info("Preprocessing: %s...", " + ".join(pp_parts))

    rec_pp = preprocess(rec_clean, freq_min=args.freq_min,
                        freq_max=args.freq_max,
                        destripe=use_destripe,
                        notch_freqs=args.notch_freqs,
                        notch_q=args.notch_q,
                        cmr=use_cmr)
    logger.info("  Done: %d ch, %.1fs",
                rec_pp.get_num_channels(), rec_pp.get_total_duration())

    if args.no_sort:
        logger.info("--no-sort specified, skipping spike sorting.")
        logger.info("Pipeline complete (preprocessing only).")
        return

    # -- 5. Run Kilosort 4 -----------------------------------------------------
    logger.info("Running Kilosort 4 (device=%s)...", args.device)
    import spikeinterface.full as si

    ks_out = output_dir / name
    sorting = si.run_sorter(
        "kilosort4", rec_pp,
        folder=ks_out,
        remove_existing_folder=True,
        verbose=args.verbose,
        torch_device=args.device,
        nblocks=0,
        templates_from_data=False,
        batch_size=args.batch_size,
    )

    n_units = len(sorting.unit_ids)
    total_spikes = sum(sorting.count_num_spikes_per_unit().values())
    logger.info("  Sorting complete: %d units, %d total spikes",
                n_units, total_spikes)

    # -- 6. Diagnostics --------------------------------------------------------
    logger.info("Building SortingAnalyzer...")
    analyzer = build_sorting_analyzer(sorting, rec_pp)

    qm = analyzer.get_extension("quality_metrics").get_data()
    noise = analyzer.get_extension("noise_levels").get_data()
    templates_data = analyzer.get_extension("templates").get_data()
    ptps = np.max(np.ptp(templates_data, axis=1), axis=1)

    logger.info("  Quality metrics:")
    logger.info("    Median SNR: %.2f", qm["snr"].median())
    logger.info("    Median FR:  %.2f Hz", qm["firing_rate"].median())
    logger.info("    Median ISI viol: %.3f", qm["isi_violations_ratio"].median())
    logger.info("    Median noise (uV): %.4f", np.median(noise))
    logger.info("    Median template PTP (uV): %.2f", np.median(ptps))
    logger.info("    Max template PTP (uV): %.2f", np.max(ptps))

    logger.info("Generating diagnostic figures...")
    recording_name = f"{name}{t_label}"
    saved = plot_sorting_diagnostics(
        analyzer, sorting, rec_pp, fig_dir / name,
        recording_name=recording_name,
    )
    logger.info("  Saved %d figures to %s/", len(saved), fig_dir / name)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
