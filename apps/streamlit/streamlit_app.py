import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import tempfile

# Page configuration
st.set_page_config(
    page_title="Space Station Safety Detector",
    page_icon="🛰️",
    layout="wide"
)

# Title and description
st.title("🛰️ Space Station Safety Equipment Detector")
st.markdown("""
This application detects critical safety equipment in space station environments:
- 🔴 Fire Extinguisher
- 🚨 Fire Alarm
- 📦 First Aid Box
- 🧯 Oxygen Tank
- 💨 Nitrogen Tank
- ⚡ Safety Switch Panel
- 📞 Emergency Phone
""")

# Sidebar for settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    step=0.05
)

# Model loading with caching
@st.cache_resource
def load_model():
    try:
        # Try to load from models directory first, then fallback to absolute path
        model_path = '../src/models/safety_equipment_model.pt'
        if not Path(model_path).exists():
            model_path = '/Users/pawankumar/Hack_Aura/src/models/safety_equipment_model.pt'
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

if model is None:
    st.error("⚠️ Please ensure 'safety_equipment_model.pt' model file is in the models directory")
    st.stop()

# Main content area
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "ℹ️ About"])

# Tab 1: Image Detection
with tab1:
    st.header("Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image of space station safety equipment"
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Run detection
        with st.spinner("🔍 Detecting safety equipment..."):
            results = model.predict(
                source=image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Get annotated image
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_img, use_container_width=True)
        
        # Display detection statistics
        st.subheader("📊 Detection Summary")
        
        detections = results[0].boxes
        if len(detections) > 0:
            class_names = results[0].names
            detected_objects = {}
            
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls_id]
                
                if class_name not in detected_objects:
                    detected_objects[class_name] = []
                detected_objects[class_name].append(conf)
            
            # Display results in columns
            cols = st.columns(4)
            for idx, (obj_name, confidences) in enumerate(detected_objects.items()):
                with cols[idx % 4]:
                    st.metric(
                        label=obj_name,
                        value=len(confidences),
                        delta=f"Avg: {np.mean(confidences):.2f}"
                    )
            
            # Detailed table
            st.subheader("🔍 Detailed Detection List")
            detection_data = []
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detection_data.append({
                    "Object": class_names[cls_id],
                    "Confidence": f"{conf:.2%}",
                    "Location": f"({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})"
                })
            
            st.table(detection_data)
            
            # Safety check
            st.subheader("✅ Safety Equipment Status")
            required_equipment = [
                "FireExtinguisher", "FireAlarm", "FirstAidBox", 
                "EmergencyPhone", "OxygenTank"
            ]
            
            for equipment in required_equipment:
                if equipment in detected_objects:
                    st.success(f"✅ {equipment}: Present ({len(detected_objects[equipment])} detected)")
                else:
                    st.warning(f"⚠️ {equipment}: Not detected")
        else:
            st.warning("⚠️ No safety equipment detected in the image")

# Tab 2: Video Detection
with tab2:
    st.header("Upload Video for Detection")
    st.info("📹 Upload a video file to detect safety equipment across frames")
    
    video_file = st.file_uploader(
        "Choose a video...", 
        type=["mp4", "avi", "mov"],
        help="Upload a video of space station environment"
    )
    
    if video_file is not None:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        st.video(video_file)
        
        if st.button("🎬 Process Video"):
            with st.spinner("Processing video... This may take a while"):
                # Process video
                results = model.predict(
                    source=tfile.name,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    save=True,
                    project='runs/detect',
                    name='video_results'
                )
                
                st.success("✅ Video processing complete!")
                st.info("Processed video saved to: runs/detect/video_results/")

# Tab 3: About
with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application uses AI-powered object detection to identify critical safety equipment 
    in space station environments, ensuring operational safety and compliance.
    
    ### 🤖 Technology Stack
    - **Model**: YOLOv8 (Ultralytics)
    - **Framework**: PyTorch
    - **Interface**: Streamlit
    - **Training Data**: Synthetic data from Falcon Digital Twin Platform
    
    ### 🔄 Model Updates with Falcon
    
    #### Continuous Improvement Pipeline:
    1. **Generate New Scenarios**: Use Falcon to create new lighting conditions, 
       equipment positions, and edge cases
    2. **Synthetic Data Augmentation**: Add wear patterns, damage, occlusions
    3. **Retrain & Validate**: Combine new data with existing dataset
    4. **Deploy Updated Model**: Replace `best.pt` with improved version
    
    #### Falcon Integration Benefits:
    - 🌓 Simulate various lighting conditions (day/night cycles)
    - 📐 Generate new camera angles and distances
    - 🔧 Add equipment variations (different models, wear states)
    - 🚧 Create occlusion scenarios (blocked visibility)
    - ⚡ Test emergency situations
    
    #### Update Schedule:
    - **Monthly**: Generate 500+ new synthetic images
    - **Quarterly**: Major model retraining
    - **On-Demand**: When new equipment types are added
    
    ### 📊 Model Performance
    - **mAP@0.5**: 0.766 (76.6%)
    - **mAP@0.5-0.95**: 0.634 (63.4%)
    - **Inference Speed**: ~238.7ms per image (CPU)
    - **Classes Detected**: 7 safety equipment types
    - **Training Epochs**: 10 epochs completed
    - **Training Time**: 7.072 hours
    
    ### 👨‍💻 Developed for
    Duality AI Space Station Challenge: Safety Object Detection #2
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ for safer space operations")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Quick Tips:**
- Lower confidence = More detections
- Higher confidence = More accurate
- IoU affects overlapping boxes
""")