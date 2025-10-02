# Hack_Aura - Safety Equipment Detection System

A YOLOv8-based computer vision system for detecting safety equipment in industrial and emergency settings. This project can identify 7 different types of safety equipment including oxygen tanks, fire extinguishers, first aid boxes, and more.

## 🎯 Overview

This project implements a custom YOLOv8 model trained to detect critical safety equipment in workplace environments. The system can help in automated safety compliance monitoring, emergency response planning, and facility management.

### Detected Equipment Classes
- **OxygenTank** - Medical/industrial oxygen tanks
- **NitrogenTank** - Industrial nitrogen tanks  
- **FirstAidBox** - Emergency first aid kits
- **FireAlarm** - Fire detection and alarm systems
- **SafetySwitchPanel** - Emergency safety switch panels
- **EmergencyPhone** - Emergency communication devices
- **FireExtinguisher** - Fire suppression equipment

## 🚀 Features

- **Custom YOLOv8 Training**: Optimized hyperparameters for safety equipment detection
- **Real-time Prediction**: Fast inference on images with bounding box visualization
- **Dataset Visualization**: Interactive tool to browse training/validation data
- **Flexible Configuration**: YAML-based configuration for easy parameter tuning
- **Cross-platform Support**: Works on CPU and GPU (CUDA)

## 📁 Project Structure

```
Hack_Aura/
├── Hackathon2_scripts/          # Main scripts and model files
│   ├── train.py                 # Model training script
│   ├── predict.py               # Inference and prediction script
│   ├── visualize.py             # Dataset visualization tool
│   ├── yolo_params.yaml         # Dataset configuration
│   ├── yolov8s.pt              # Pre-trained YOLOv8 weights
│   ├── classes.txt             # Class names definition
│   └── ENV_SETUP/              # Environment setup scripts
├── train_2/                    # Training dataset (gitignored)
│   ├── train2/                 # Training images and labels
│   └── val2/                   # Validation images and labels
├── test2/                      # Test dataset (gitignored)
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8

### Setup Environment

#### For Windows:
```bash
cd Hackathon2_scripts/ENV_SETUP
setup_env.bat
```

#### For macOS/Linux:
```bash
cd Hackathon2_scripts/ENV_SETUP
chmod +x setup_env.sh
./setup_env.sh
```

#### Manual Installation:
```bash
pip install ultralytics opencv-python PyYAML
```

## 🏃‍♂️ Usage

### Training the Model

Train a custom YOLOv8 model on your safety equipment dataset:

```bash
cd Hackathon2_scripts
python train.py --epochs 100 --lr0 0.001 --device cuda
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 10)
- `--mosaic`: Mosaic augmentation probability (default: 0.4)
- `--optimizer`: Optimizer type (default: 'AdamW')
- `--momentum`: SGD momentum (default: 0.9)
- `--lr0`: Initial learning rate (default: 0.0001)
- `--lrf`: Final learning rate (default: 0.0001)
- `--device`: Training device - 'cpu', 'cuda', or GPU ID (default: 'cpu')

### Running Predictions

Perform inference on test images:

```bash
cd Hackathon2_scripts
python predict.py
```

The script will:
1. Load the best trained model from `runs/detect/train*/weights/best.pt`
2. Process all images in the test directory
3. Save predicted images with bounding boxes to `predictions/images/`
4. Save label files in YOLO format to `predictions/labels/`
5. Display validation metrics

### Visualizing Dataset

Interactively browse your training and validation data:

```bash
cd Hackathon2_scripts
python visualize.py
```

**Controls:**
- `d` - Next image
- `a` - Previous image  
- `t` - Switch to training set
- `v` - Switch to validation set
- `q` or `ESC` - Quit

## ⚙️ Configuration

Edit `yolo_params.yaml` to customize dataset paths and parameters:

```yaml
train: ../train_2/train2      # Training data path
val: ../train_2/val2          # Validation data path  
test: ../test2                # Test data path
nc: 7                         # Number of classes
names: ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
```

## 📊 Model Performance

The model uses YOLOv8s architecture with the following specifications:
- **Parameters**: 3,011,043 trainable parameters
- **GFLOPs**: 8.2 computational complexity
- **Layers**: 225 total layers
- **Input Resolution**: Configurable (default: 640x640)

### Training Tips
- Mosaic augmentation improves validation but may reduce test performance
- AdamW optimizer with low learning rates works best for this dataset
- CPU training is supported but GPU training is recommended for speed

## 📝 License

This project is part of the Hack_Aura hackathon submission. Please refer to the competition guidelines for usage terms.

## 🔗 Dependencies

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📞 Support

For questions and support, please open an issue in the repository or contact the development team.

---

*Built with ❤️ for safety and automation*
