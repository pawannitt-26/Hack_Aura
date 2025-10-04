from http.server import BaseHTTPRequestHandler
import json
import base64
import io
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os
from pathlib import Path

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            # Set CORS headers
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Get content length
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON data
            data = json.loads(post_data.decode('utf-8'))
            
            # Get image data and parameters
            image_data = data.get('image')
            confidence_threshold = data.get('confidence', 0.25)
            iou_threshold = data.get('iou', 0.45)
            
            if not image_data:
                self.wfile.write(json.dumps({
                    'error': 'No image data provided'
                }).encode())
                return

            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Load model (cached in global scope)
            model = load_model()
            if model is None:
                self.wfile.write(json.dumps({
                    'error': 'Model could not be loaded'
                }).encode())
                return

            # Run prediction
            results = model.predict(
                source=image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Process results
            detections = results[0].boxes
            class_names = results[0].names
            
            detection_data = []
            detected_objects = {}
            
            if len(detections) > 0:
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
                    
                    if class_name not in detected_objects:
                        detected_objects[class_name] = []
                    detected_objects[class_name].append(conf)
            
            # Get annotated image
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Convert annotated image to base64
            pil_annotated = Image.fromarray(annotated_img)
            buffer = io.BytesIO()
            pil_annotated.save(buffer, format='PNG')
            annotated_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare response
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
            self.wfile.write(json.dumps({
                'error': str(e)
            }).encode())

# Global model variable for caching
_model = None

def load_model():
    global _model
    if _model is None:
        try:
            # Try different model paths
            model_paths = [
                'src/models/safety_equipment_model.pt',
                '/var/task/src/models/safety_equipment_model.pt',
                'safety_equipment_model.pt'
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print("No model file found")
                return None
                
            _model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    return _model
