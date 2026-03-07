#!/bin/bash
# Run the full pipeline: sort all 4 surveys, then track neurons across days.
#
# Usage:
#   bash run_all.sh                    # sort all + track
#   bash run_all.sh sort               # sort only
#   bash run_all.sh track              # track only (assumes sorting done)
#
# Prerequisites:
#   1. Run setup_env.sh first to create the conda environment
#   2. Set DATA_ROOT to your SpikeGLX data directory:
#        export DATA_ROOT=~/data/TRODES_test
#   3. Copy the 4 survey directories into DATA_ROOT:
#        SvyPrb_2026-02-26T11.58.07_g0/
#        SvyPrb_2026-03-02T14.10.26_g0/
#        SvyPrb_2026-03-04T10.17.50_g0/
#        SvyPrb_2026-03-06T10.01.15_g0/

set -e

# Default DATA_ROOT if not set
export DATA_ROOT="${DATA_ROOT:-$HOME/data/TRODES_test}"

echo "=================================================="
echo "Chronic Pixel Pipeline — Full Run"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "=================================================="

# Check data exists
for dir in \
    "SvyPrb_2026-02-26T11.58.07_g0" \
    "SvyPrb_2026-03-02T14.10.26_g0" \
    "SvyPrb_2026-03-04T10.17.50_g0" \
    "SvyPrb_2026-03-06T10.01.15_g0"; do
    if [ ! -d "${DATA_ROOT}/${dir}" ]; then
        echo "WARNING: Missing data directory: ${DATA_ROOT}/${dir}"
    fi
done

MODE="${1:-all}"

if [ "$MODE" = "sort" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 1: Spike sorting all 4 surveys with KS4"
    echo "=================================================="
    python sort_survey.py all 2>&1 | tee logs/sort_all.log
fi

if [ "$MODE" = "track" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 2: Cross-day neuron tracking with UnitMatch"
    echo "=================================================="
    python track_neurons.py 20260226 20260302 20260304 20260306 2>&1 | tee logs/track_neurons.log
fi

echo ""
echo "=================================================="
echo "Pipeline complete!"
echo "=================================================="
echo "Outputs:"
echo "  Sorting:  sorting_output/survey_*/  (per-bank KS4 + analyzer caches)"
echo "  Tracking: sorting_output/tracking/  (tracking_summary.csv)"
echo "  Figures:  figures/sorting/           (PNG visualizations)"
