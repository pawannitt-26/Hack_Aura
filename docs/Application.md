# 🛰️ Space Station Safety Equipment Detector

An AI-powered application for detecting critical safety equipment in space station environments using YOLOv8 object detection.

## 🎯 Features

- **Real-time Detection**: Identify 7 types of safety equipment
- **High Accuracy**: Trained on synthetic data from Falcon digital twin platform
- **User-Friendly Interface**: Web-based (Streamlit) or Desktop (Tkinter) options
- **Confidence Adjustments**: Configurable detection thresholds
- **Detailed Reports**: Per-object statistics and visualizations
- **Continuous Updates**: Integration with Falcon for ongoing model improvements

## 🔍 Detected Objects

1. 🔥 Fire Extinguisher
2. 🚨 Fire Alarm
3. 📦 First Aid Box
4. 💨 Oxygen Tank
5. 🌡️ Nitrogen Tank
6. ⚡ Safety Switch Panel
7. 📞 Emergency Phone

## 📋 Requirements

### System Requirements
- **OS**: macOS, Linux, or Windows
- **Python**: 3.8-3.11
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB free space

### For Apple Silicon Mac (M1/M2/M3)
- GPU acceleration supported via MPS

## 🎮 Usage

### Option 1: Streamlit Web App (Recommended)

```bash
# Start the application
streamlit run apps/streamlit/streamlit_app.py

# Open in browser (automatic)
# Default: http://localhost:8501
```

**Features:**
- Upload images or videos
- Adjust confidence and IoU thresholds in real-time
- View detection statistics
- Download annotated results

### Option 2: Desktop GUI App

```bash
# Run the desktop application
python apps/desktop/tkinter_app.py
```

**Features:**
- File browser for image selection
- Adjustable detection settings
- Real-time results panel
- Simple and fast

## 📊 Model Performance

- **mAP@0.5**: 76.6%
- **mAP@0.5-0.95**: 63.4%
- **Inference Speed**: ~238.7ms per image (CPU)
- **Training Dataset**: 2,103 images (1,767 train + 336 val)
- **Architecture**: YOLOv8s (Small)

### Per-Class Performance
| Object | Precision | Recall | mAP@0.5 |
|--------|-----------|--------|---------|
| Fire Extinguisher | 93.2% | 56.8% | 71.2% |
| Fire Alarm | 93.5% | 76.1% | 86.0% |
| First Aid Box | 87.9% | 68.9% | 82.6% |
| Oxygen Tank | 95.7% | 76.4% | 86.1% |
| Nitrogen Tank | 88.3% | 69.3% | 81.0% |
| Safety Switch Panel | 91.3% | 55.0% | 68.2% |
| Emergency Phone | 91.4% | 52.2% | 61.1% |

## 🔄 Keeping Model Updated with Falcon

### Monthly Update Pipeline

1. **Monitor Production Performance**
   ```bash
   python monitor_production.py --logs logs/month_1/
   ```

2. **Generate New Scenarios in Falcon**
   - Identify weak detection areas
   - Create targeted scenarios (lighting, angles, occlusions)
   - Generate 500-1000 new labeled images

3. **Merge Datasets**
   ```bash
   python merge_datasets.py \
       --original datasets/v1.0 \
       --new falcon_generated/month_2 \
       --output datasets/v1.1
   ```

4. **Retrain Model**
   ```bash
   python train.py \
       --data datasets/v1.1/config.yaml \
       --epochs 50 \
       --weights models/v1.0/best.pt \
       --name v1.1
   ```

5. **Validate & Deploy**
   ```bash
   # Compare performance
   python compare_models.py --model1 v1.0 --model2 v1.1
   
   # If improved, deploy
   cp runs/train/v1.1/weights/best.pt ./src/models/safety_equipment_model.pt
   ```

### Falcon Integration Benefits
- ✅ **Unlimited Data**: Generate synthetic images on-demand
- ✅ **Perfect Labels**: Automatic, error-free annotations
- ✅ **Scenario Control**: Precise control over lighting, angles, occlusions
- ✅ **Cost Effective**: No manual labeling required
- ✅ **Rapid Iteration**: Hours instead of weeks

See `falcon_integration_plan.md` for detailed implementation guide.

## 🎯 Tips for Best Results

### Detection Settings
- **High Precision Needed**: `conf=0.4, iou=0.5`
- **High Recall Needed**: `conf=0.2, iou=0.4`
- **Balanced**: `conf=0.25, iou=0.45` (default)

### Image Quality
- ✅ Good lighting (not too dark or bright)
- ✅ Clear, focused images
- ✅ Objects not extremely small (<20px)
- ✅ Minimal motion blur

### Common Issues
- **No detections**: Lower confidence threshold
- **Too many false positives**: Raise confidence threshold
- **Overlapping boxes**: Adjust IoU threshold
- **Slow inference**: Use smaller model (yolov8n) or reduce image size

## 🏆 Hackathon Submission Components

This application fulfills the **bonus challenge** requirements:

### ✅ Application Features
1. Working detection interface (web/desktop)
2. Model inference integration
3. User-friendly controls
4. Results visualization

### ✅ Falcon Integration Plan
1. Continuous learning pipeline
2. Monthly update schedule
3. Scenario generation strategy
4. Performance monitoring system

### ✅ Documentation
1. Complete usage instructions
2. Installation guide
3. Update procedures
4. Technical specifications