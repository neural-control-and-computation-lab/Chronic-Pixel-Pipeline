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
#   3. Copy the 4 survey directories into DATA_ROOT
#
# All outputs (sorting_output, figures, logs) are written inside DATA_ROOT
# so nothing goes through Dropbox or other cloud sync.

set -e

# Default DATA_ROOT if not set
export DATA_ROOT="${DATA_ROOT:-$HOME/data/TRODES_test}"

LOG_DIR="${DATA_ROOT}/logs"

echo "=================================================="
echo "Chronic Pixel Pipeline — Full Run"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "Logs:      ${LOG_DIR}"
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

mkdir -p "${LOG_DIR}"

MODE="${1:-all}"

if [ "$MODE" = "sort" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 1: Spike sorting all 4 surveys with KS4"
    echo "=================================================="
    python sort_survey.py all 2>&1 | tee "${LOG_DIR}/sort_all.log"
fi

if [ "$MODE" = "track" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 2: Cross-day neuron tracking with UnitMatch"
    echo "=================================================="
    python track_neurons.py 20260226 20260302 20260304 20260306 2>&1 | tee "${LOG_DIR}/track_neurons.log"
fi

echo ""
echo "=================================================="
echo "Pipeline complete!"
echo "=================================================="
echo "Outputs (all inside DATA_ROOT):"
echo "  Sorting:  ${DATA_ROOT}/sorting_output/survey_*/"
echo "  Tracking: ${DATA_ROOT}/sorting_output/tracking/"
echo "  Figures:  ${DATA_ROOT}/figures/sorting/"
echo "  Logs:     ${LOG_DIR}/"
