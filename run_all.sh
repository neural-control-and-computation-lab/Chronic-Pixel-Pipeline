#!/bin/bash
# Run the full survey pipeline, end to end.
#
# Usage:
#   bash run_all.sh                    # full pipeline (all six steps)
#   bash run_all.sh sort               # 1. sort only
#   bash run_all.sh metrics            # 2. precompute waveform quality only
#   bash run_all.sh classify           # 3. apply the neural/noise classifier only
#   bash run_all.sh figures            # 4. generate unit figures only
#   bash run_all.sh track              # 5. track only (assumes sorting done)
#   bash run_all.sh aggregate          # 6. aggregate figures + Dropbox sync only
#
# Prerequisites:
#   1. Run setup_env.sh first to create the conda environment
#   2. Set DATA_ROOT to your SpikeGLX data directory:
#        export DATA_ROOT=~/data/TRODES_test
#   3. Survey directories should exist in DATA_ROOT
#   4. Step 3 needs a trained model in DATA_ROOT/classifier_model/
#      (see train_classifier.py)
#
# All outputs (sorting_output, figures, logs) are written inside DATA_ROOT.
# The only thing that leaves DATA_ROOT is the explicit figure sync in step 6.

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

if [ "$MODE" = "classify" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 3: Apply the neural/noise classifier"
    echo "=================================================="
    python apply_custom_classifier.py all 2>&1 | tee "${LOG_DIR}/apply_classifier.log"
fi

if [ "$MODE" = "figures" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 4: Generate single-neuron figures"
    echo "=================================================="
    python run_single_neuron_plots.py 2>&1 | tee "${LOG_DIR}/single_neuron_plots.log"
fi

if [ "$MODE" = "track" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 5: Cross-day neuron tracking with UnitMatch"
    echo "=================================================="
    python track_neurons.py 2>&1 | tee "${LOG_DIR}/track_neurons.log"
fi

if [ "$MODE" = "aggregate" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "=================================================="
    echo "Step 6: Aggregate tracking figures + Dropbox sync"
    echo "=================================================="
    python regen_tracking_plots.py 2>&1 | tee "${LOG_DIR}/regen_tracking_plots.log"
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
