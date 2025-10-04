from http.server import BaseHTTPRequestHandler
import json
import base64
import io
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os
import requests

# --- Hugging Face Model Info ---
HF_REPO = "pawan-kumar/safety-equipment-model"  # 🔁 replace with your actual repo name
MODEL_FILENAME = "safety_equipment_model.pt"
MODEL_PATH = f"/tmp/{MODEL_FILENAME}"
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # add this in Vercel if model is private

# --- Model Caching ---
_model = None

def download_model():
    """Download model from Hugging Face if not cached."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    print("📥 Downloading model from Hugging Face...")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{MODEL_FILENAME}"

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("✅ Model downloaded successfully:", MODEL_PATH)
    return MODEL_PATH

def load_model():
    """Load YOLO model (cached globally)."""
    global _model
    if _model is None:
        try:
            model_path = download_model()
            _model = YOLO(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            _model = None
    return _model

# --- Request Handler ---
class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Parse incoming JSON
            length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(length)
            data = json.loads(post_data.decode('utf-8'))

            image_data = data.get('image')
            conf = data.get('confidence', 0.25)
            iou = data.get('iou', 0.45)

            if not image_data:
                self.wfile.write(json.dumps({'error': 'No image data provided'}).encode())
                return

            # Decode Base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Load YOLO model
            model = load_model()
            if model is None:
                self.wfile.write(json.dumps({'error': 'Model could not be loaded'}).encode())
                return

            # Run inference
            results = model.predict(image, conf=conf, iou=iou, verbose=False)
            detections = results[0].boxes
            class_names = results[0].names

            detection_data = []
            detected_objects = {}

            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = class_names[cls_id]
                detection_data.append({
                    'object': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                detected_objects.setdefault(class_name, []).append(conf)

            # Annotated image
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated_img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            annotated_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Response
            response = {
                'success': True,
                'detections': detection_data,
                'detected_objects': {k: len(v) for k, v in detected_objects.items()},
                'annotated_image': f'data:image/png;base64,{annotated_base64}',
                'total_detections': len(detection_data)
            }

            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
