#!/usr/bin/env bash

set -euo pipefail

# Linux environment setup for EDU (Python 3.10) with selectable PyTorch build
# Usage:
#   bash setup_env_linux.sh [cpu|cuda12|cuda11]
# Defaults to CPU if no argument provided.

CHOICE="${1:-cpu}"

echo "=========================================="
echo "Linux setup for conda env 'EDU' (Python 3.10)"
echo "Requested PyTorch build: ${CHOICE}"
echo "=========================================="

# Ensure conda is available and initialized for non-interactive shells
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found in PATH. If Miniconda/Anaconda is installed, sourcing base..."
  # Try common installation paths
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Error: conda not found. Please install Miniconda/Anaconda and retry." >&2
    exit 1
  fi
fi

# Initialize conda shell hook for bash
eval "$(conda shell.bash hook)"

echo "Creating conda environment 'EDU' with Python 3.10..."
conda create -n EDU python=3.10 -y

echo "Activating 'EDU'..."
conda activate EDU

echo "Installing PyTorch (${CHOICE})..."
case "$CHOICE" in
  cpu|CPU)
    conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
    ;;
  cuda12|CUDA12|cuda12.1|CUDA12.1)
    conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
    ;;
  cuda11|CUDA11|cuda11.8|CUDA11.8)
    conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
    ;;
  *)
    echo "Unknown option '${CHOICE}'. Use one of: cpu | cuda12 | cuda11" >&2
    exit 1
    ;;
esac

echo "Installing Ultralytics YOLOv8 and dependencies..."
pip install --upgrade pip
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install pandas
pip install seaborn
pip install pillow
pip install pyyaml
pip install tqdm
pip install scipy

echo "Verifying installation..."
python - <<'PY'
import torch, ultralytics
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (if any): {torch.version.cuda}")
print(f"Ultralytics version: {ultralytics.__version__}")
PY

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "To activate next time: conda activate EDU"
echo "=========================================="


