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
├── src/                        # Source code and scripts
│   ├── train_model.py          # Model training script
│   ├── run_inference.py        # Inference and prediction script
│   ├── visualize_dataset.py    # Dataset visualization tool
│   ├── models/                 # Model files
│   │   ├── safety_equipment_model.pt    # Trained model weights
│   │   └── pretrained_yolov8s.pt       # Pre-trained YOLOv8 weights
│   ├── configs/                # Configuration files
│   │   ├── dataset_config.yaml # Dataset configuration
│   │   └── class_names.txt     # Class names definition
│   └── ENV_SETUP/              # Environment setup scripts
├── apps/                       # Application interfaces
│   ├── streamlit/              # Web application
│   │   └── streamlit_app.py    # Streamlit web interface
│   └── desktop/                # Desktop application
│       └── tkinter_app.py      # Tkinter desktop interface
├── datasets/                   # Dataset directories
│   ├── training/               # Training images and labels
│   ├── validation/             # Validation images and labels
│   └── testing/                # Test images and labels
├── docs/                       # Documentation
│   ├── Application.md          # Application documentation
│   ├── falcon_integration_plan.md  # Falcon integration plan
│   └── logs.md                 # Log files
├── run_streamlit.py            # Launcher for web app
├── run_desktop.py              # Launcher for desktop app
└── README.md

- [Download training dataset](https://storage.googleapis.com/duality-public-share/Datasets/hackathon2_train_3.zip)
- [Download test dataset](https://storage.googleapis.com/duality-public-share/Datasets/hackathon2_test3.zip)
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
cd src/ENV_SETUP
setup_env.bat
```

#### For macOS:
```bash
cd src/ENV_SETUP
chmod +x setup_env.sh
./setup_env.sh
```

#### For Linux:
```bash
cd src/ENV_SETUP
bash setup_env_linux.sh            # CPU-only install (default)

# or specify CUDA build if you have an NVIDIA GPU and drivers installed
bash setup_env_linux.sh cuda12     # CUDA 12.1 build
bash setup_env_linux.sh cuda11     # CUDA 11.8 build
```

#### Manual Installation:
```bash
pip install ultralytics opencv-python PyYAML
```

## 🏃‍♂️ Usage

### Quick Start

#### Web Application (Streamlit)
```bash
# Easy launcher
python run_streamlit.py

# Or run directly
streamlit run apps/streamlit/streamlit_app.py
```

#### Desktop Application (Tkinter)
```bash
# Easy launcher
python run_desktop.py

# Or run directly
python apps/desktop/tkinter_app.py
```

### Training the Model

Train a custom YOLOv8 model on your safety equipment dataset:

```bash
cd src
python train_model.py --epochs 100 --lr0 0.001 --device cuda
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
cd src
python run_inference.py
```

The script will:
1. Load the best trained model from `models/safety_equipment_model.pt`
2. Process all images in the test directory
3. Save predicted images with bounding boxes to `predictions/images/`
4. Save label files in YOLO format to `predictions/labels/`
5. Display validation metrics

### Visualizing Dataset

Interactively browse your training and validation data:

```bash
cd src
python visualize_dataset.py
```

**Controls:**
- `d` - Next image
- `a` - Previous image  
- `t` - Switch to training set
- `v` - Switch to validation set
- `q` or `ESC` - Quit

## ⚙️ Configuration

Edit `configs/dataset_config.yaml` to customize dataset paths and parameters:

```yaml
train: ../datasets/training      # Training data path
val: ../datasets/validation      # Validation data path  
test: ../datasets/testing        # Test data path
nc: 7                            # Number of classes
names: ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
```

## 📊 Model Performance

The model uses YOLOv8s architecture with the following specifications:
- **Parameters**: 3,011,043 trainable parameters
- **GFLOPs**: 8.2 computational complexity
- **Layers**: 225 total layers
- **Input Resolution**: Configurable (default: 640x640)

### Validation Results

The model has been validated on a comprehensive test dataset with the following performance metrics:

#### Performance Curves

**F1-Confidence Curve**
![F1-Confidence Curve](src/runs/val_test/BoxF1_curve.png)

**Precision-Recall Curve**
![Precision-Recall Curve](src/runs/val_test/BoxPR_curve.png)

**Precision Curve**
![Precision Curve](src/runs/val_test/BoxP_curve.png)

**Recall Curve**
![Recall Curve](src/runs/val_test/BoxR_curve.png)

#### Class-wise Performance
The model shows varying performance across different safety equipment classes:

| Class | Peak F1 Score | Optimal Confidence |
|-------|---------------|-------------------|
| OxygenTank | ~0.75 | 0.2-0.3 |
| NitrogenTank | ~0.70 | 0.2-0.3 |
| FirstAidBox | ~0.70 | 0.2-0.3 |
| FireExtinguisher | ~0.60 | 0.2-0.3 |
| SafetySwitchPanel | ~0.58-0.60 | 0.2-0.3 |
| FireAlarm | ~0.48-0.50 | 0.2-0.3 |
| EmergencyPhone | ~0.45 | 0.2-0.3 |

#### Validation Visualizations

**Confusion Matrix**
![Confusion Matrix](src/runs/val_test/confusion_matrix.png)

**Normalized Confusion Matrix**
![Normalized Confusion Matrix](src/runs/val_test/confusion_matrix_normalized.png)

**Sample Validation Predictions**

*Ground Truth vs Predictions - Batch 0*
![Validation Batch 0 - Labels](src/runs/val_test/val_batch0_labels.jpg)
![Validation Batch 0 - Predictions](src/runs/val_test/val_batch0_pred.jpg)

*Ground Truth vs Predictions - Batch 1*
![Validation Batch 1 - Labels](src/runs/val_test/val_batch1_labels.jpg)
![Validation Batch 1 - Predictions](src/runs/val_test/val_batch1_pred.jpg)

*Ground Truth vs Predictions - Batch 2*
![Validation Batch 2 - Labels](src/runs/val_test/val_batch2_labels.jpg)
![Validation Batch 2 - Predictions](src/runs/val_test/val_batch2_pred.jpg)

### Training Tips
- Mosaic augmentation improves validation but may reduce test performance
- AdamW optimizer with low learning rates works best for this dataset
- CPU training is supported but GPU training is recommended for speed
- Optimal confidence threshold of 0.225 provides best overall F1 score

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
