#!/bin/bash
# Run the full pipeline: sort, compute quality metrics, generate figures, track neurons.
#
# Usage:
#   bash run_all.sh                    # full pipeline (sort + metrics + figures + track)
#   bash run_all.sh sort               # sort only
#   bash run_all.sh metrics            # precompute waveform quality only
#   bash run_all.sh figures            # generate unit figures only
#   bash run_all.sh track              # track only (assumes sorting done)
#
# Prerequisites:
#   1. Run setup_env.sh first to create the conda environment
#   2. Set DATA_ROOT to your SpikeGLX data directory:
#        export DATA_ROOT=~/data/TRODES_test
#   3. Survey directories should exist in DATA_ROOT
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

mkdir -p "${LOG_DIR}"

MODE="${1:-all}"

if [ "$MODE" = "sort" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 1: Spike sorting all surveys with KS4"
    echo "=================================================="
    python sort_survey.py all 2>&1 | tee "${LOG_DIR}/sort_all.log"
fi

if [ "$MODE" = "metrics" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 2: Precompute waveform quality metrics"
    echo "=================================================="
    python precompute_waveform_quality.py 2>&1 | tee "${LOG_DIR}/precompute_quality.log"
fi

if [ "$MODE" = "figures" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 3: Generate single-neuron figures"
    echo "=================================================="
    python run_single_neuron_plots.py 2>&1 | tee "${LOG_DIR}/single_neuron_plots.log"
fi

if [ "$MODE" = "track" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 4: Cross-day neuron tracking with UnitMatch"
    echo "=================================================="
    python track_neurons.py 2>&1 | tee "${LOG_DIR}/track_neurons.log"
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
