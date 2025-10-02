#!/bin/bash

# Create conda environment named EDU with Python 3.10
echo "Creating conda environment 'EDU'..."
conda create -n EDU python=3.10 -y

# Activate the environment
echo "Activating EDU environment..."
source activate EDU

# Install PyTorch (Mac-compatible version)
echo "Installing PyTorch..."
# For Mac with Apple Silicon (M1/M2/M3)
conda install pytorch torchvision torchaudio -c pytorch -y

# For Intel Mac, use:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install Ultralytics YOLO
echo "Installing Ultralytics YOLOv8..."
pip install ultralytics

# Install additional dependencies
echo "Installing additional packages..."
pip install opencv-python
pip install matplotlib
pip install pandas
pip install seaborn
pip install pillow
pip install pyyaml
pip install tqdm
pip install scipy

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "To activate: conda activate EDU"
echo "=========================================="