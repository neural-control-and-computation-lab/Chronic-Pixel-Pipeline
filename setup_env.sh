#!/bin/bash
# Setup environment for chronic pixel pipeline on Ubuntu + CUDA
# Run once: bash setup_env.sh
set -e

echo "=== Setting up Chronic Pixel Pipeline environment ==="

# Create conda env (or use existing)
ENV_NAME="cpp"
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.11 -y
else
    echo "Environment ${ENV_NAME} already exists"
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# PyTorch with CUDA (adjust cuda version as needed)
echo "=== Installing PyTorch with CUDA ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core pipeline
echo "=== Installing SpikeInterface + Kilosort4 ==="
pip install "spikeinterface[full]>=0.100"
pip install "kilosort>=4.0"
pip install probeinterface

# UnitMatch for neuron tracking
echo "=== Installing UnitMatch ==="
pip install UnitMatchPy mat73

# Visualization and utilities
pip install matplotlib pandas

# Install the chronic-pixel-pipeline package itself (editable)
echo "=== Installing chronic-pixel-pipeline ==="
pip install -e ".[all]"

echo ""
echo "=== Environment ready ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Verify CUDA:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
