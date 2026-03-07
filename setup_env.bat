@echo off
REM Setup environment for chronic pixel pipeline on Windows + CUDA
REM Run once from Anaconda Prompt: setup_env.bat
setlocal

echo =================================================
echo  Setting up Chronic Pixel Pipeline environment
echo =================================================

set ENV_NAME=cpp

REM Create conda env (or use existing)
conda info --envs | findstr /B "%ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo Creating conda environment: %ENV_NAME%
    conda create -n %ENV_NAME% python=3.11 -y
) else (
    echo Environment %ENV_NAME% already exists
)

call conda activate %ENV_NAME%

REM PyTorch with CUDA (adjust cuda version as needed)
echo === Installing PyTorch with CUDA ===
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Core pipeline
echo === Installing SpikeInterface + Kilosort4 ===
pip install "spikeinterface[full]>=0.100"
pip install "kilosort>=4.0"
pip install probeinterface

REM UnitMatch for neuron tracking
echo === Installing UnitMatch ===
pip install UnitMatchPy mat73

REM Visualization and utilities
pip install matplotlib pandas

REM Install the chronic-pixel-pipeline package itself (editable)
echo === Installing chronic-pixel-pipeline ===
pip install -e ".[all]"

echo.
echo =================================================
echo  Environment ready
echo =================================================
echo Activate with: conda activate %ENV_NAME%
echo.
echo Verify CUDA:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else chr(34)none chr(34)}')"

endlocal
