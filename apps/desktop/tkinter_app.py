#!/usr/bin/env python3
"""
Desktop GUI Application for Space Station Safety Equipment Detection
Uses Tkinter for a simple, cross-platform interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import threading
import os

class SafetyDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛰️ Space Station Safety Equipment Detector")
        self.root.geometry("1200x800")
        
        # Model variables
        self.model = None
        self.current_image = None
        self.current_image_path = None
        self.detection_results = None
        
        # UI variables
        self.confidence_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.45)
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Configure root window
        self.root.configure(bg='#f0f0f0')
        
        # Main frame with better styling
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header frame with title and description
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Title with better styling
        title_label = ttk.Label(header_frame, text="🛰️ Space Station Safety Equipment Detector", 
                               font=("Arial", 18, "bold"), foreground="#2c3e50")
        title_label.pack()
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, text="AI-Powered Detection of Critical Safety Equipment", 
                                  font=("Arial", 10), foreground="#7f8c8d")
        subtitle_label.pack(pady=(5, 0))
        
        # Control panel with better styling
        control_frame = ttk.LabelFrame(main_frame, text="🎛️ Controls & Settings", padding="15")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        # File selection section
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        file_frame.columnconfigure(0, weight=1)
        
        ttk.Label(file_frame, text="📁 Select Image", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.select_btn = ttk.Button(file_frame, text="Choose Image File", 
                                   command=self.select_image, style="Accent.TButton")
        self.select_btn.grid(row=1, column=0, sticky=tk.W+tk.E, pady=(0, 5))
        
        self.file_label = ttk.Label(file_frame, text="No image selected", 
                                  foreground="#7f8c8d", font=("Arial", 9))
        self.file_label.grid(row=2, column=0, sticky=tk.W)
        
        # Detection section
        detect_frame = ttk.Frame(control_frame)
        detect_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        detect_frame.columnconfigure(0, weight=1)
        
        ttk.Label(detect_frame, text="🔍 Detection", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.detect_btn = ttk.Button(detect_frame, text="Detect Safety Equipment", 
                                   command=self.detect_objects, state="disabled")
        self.detect_btn.grid(row=1, column=0, sticky=tk.W+tk.E, pady=(0, 5))
        
        # Settings section
        settings_frame = ttk.Frame(control_frame)
        settings_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        settings_frame.columnconfigure(0, weight=1)
        
        ttk.Label(settings_frame, text="⚙️ Detection Settings", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        conf_frame.columnconfigure(1, weight=1)
        
        ttk.Label(conf_frame, text="Confidence:", font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W)
        self.confidence_label = ttk.Label(conf_frame, text="0.25", font=("Arial", 9, "bold"), 
                                        foreground="#3498db")
        self.confidence_label.grid(row=0, column=2, sticky=tk.E)
        
        confidence_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, pady=(2, 0))
        confidence_scale.configure(command=self.update_confidence_label)
        
        # IoU threshold
        iou_frame = ttk.Frame(settings_frame)
        iou_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        iou_frame.columnconfigure(1, weight=1)
        
        ttk.Label(iou_frame, text="IoU Overlap:", font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W)
        self.iou_label = ttk.Label(iou_frame, text="0.45", font=("Arial", 9, "bold"), 
                                 foreground="#e74c3c")
        self.iou_label.grid(row=0, column=2, sticky=tk.E)
        
        iou_scale = ttk.Scale(iou_frame, from_=0.1, to=1.0, 
                            variable=self.iou_var, orient=tk.HORIZONTAL)
        iou_scale.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, pady=(2, 0))
        iou_scale.configure(command=self.update_iou_label)
        
        # Quick tips
        tips_frame = ttk.Frame(control_frame)
        tips_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        
        ttk.Label(tips_frame, text="💡 Quick Tips:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(tips_frame, text="• Lower confidence = More detections", 
                 font=("Arial", 8), foreground="#7f8c8d").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(tips_frame, text="• Higher confidence = More accurate", 
                 font=("Arial", 8), foreground="#7f8c8d").grid(row=2, column=0, sticky=tk.W)
        
        # Results panel with better layout
        results_frame = ttk.LabelFrame(main_frame, text="🖼️ Image & Detection Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Image display with better styling
        image_frame = ttk.Frame(results_frame)
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        self.image_label = ttk.Label(image_frame, text="📷 Select an image to begin detection", 
                                   anchor="center", font=("Arial", 12), foreground="#7f8c8d")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Detection info panel
        info_frame = ttk.LabelFrame(main_frame, text="📊 Detection Information", padding="10")
        info_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(info_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        
        self.summary_text = tk.Text(summary_frame, height=8, width=35, wrap=tk.WORD, 
                                   font=("Arial", 9), bg="#f8f9fa", fg="#2c3e50", 
                                   relief="flat", selectbackground="#3498db")
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Details tab
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="Details")
        
        self.details_text = tk.Text(details_frame, height=8, width=35, wrap=tk.WORD, 
                                   font=("Arial", 9), bg="#f8f9fa", fg="#2c3e50", 
                                   relief="flat", selectbackground="#3498db")
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Safety tab
        safety_frame = ttk.Frame(self.notebook)
        self.notebook.add(safety_frame, text="Safety Check")
        
        self.safety_text = tk.Text(safety_frame, height=8, width=35, wrap=tk.WORD, 
                                  font=("Arial", 9), bg="#f8f9fa", fg="#2c3e50", 
                                  relief="flat", selectbackground="#3498db")
        self.safety_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar with better styling
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(15, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="🟢 Ready - Select an image to begin")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, 
                              relief="sunken", padding="5", font=("Arial", 9))
        status_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Progress bar (initially hidden)
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.progress.grid_remove()  # Hide initially
    
    def update_confidence_label(self, value):
        """Update confidence label when scale changes"""
        self.confidence_label.config(text=f"{float(value):.2f}")
    
    def update_iou_label(self, value):
        """Update IoU label when scale changes"""
        self.iou_label.config(text=f"{float(value):.2f}")
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            # Try to find the best model
            model_paths = [
                '../src/models/safety_equipment_model.pt',
                '/Users/pawankumar/Hack_Aura/src/models/safety_equipment_model.pt',
                '/opt/homebrew/runs/detect/train2/weights/best.pt'
            ]
            
            model_path = None
            for path in model_paths:
                if Path(path).exists():
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("No trained model found")
            
            self.model = YOLO(model_path)
            self.status_var.set(f"Model loaded: {Path(model_path).name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Model loading failed")
    
    def select_image(self):
        """Select an image file"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select an image",
            filetypes=filetypes
        )
        
        if filename:
            self.current_image_path = filename
            self.load_image()
            self.detect_btn.config(state="normal")
    
    def load_image(self):
        """Load and display the selected image"""
        try:
            # Load image
            image = Image.open(self.current_image_path)
            
            # Resize for display while maintaining aspect ratio
            display_size = (600, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(image)
            
            # Update display
            self.image_label.config(image=self.current_image, text="")
            
            # Update file label
            self.file_label.config(text=f"📁 {Path(self.current_image_path).name}", 
                                 foreground="#27ae60")
            
            # Clear previous results and show image info
            self.clear_results()
            self.update_summary(f"📷 Image: {Path(self.current_image_path).name}\n"
                              f"📏 Size: {image.size[0]} × {image.size[1]} pixels\n"
                              f"📁 Path: {self.current_image_path}\n\n"
                              f"🔍 Ready for detection!\n"
                              f"Click 'Detect Safety Equipment' to analyze.")
            
            self.status_var.set(f"🟢 Image loaded: {Path(self.current_image_path).name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("❌ Image loading failed")
    
    def detect_objects(self):
        """Run object detection on the current image"""
        if not self.model or not self.current_image_path:
            return
        
        def detection_thread():
            try:
                # Update UI for detection start
                self.status_var.set("🔍 Detecting safety equipment...")
                self.detect_btn.config(state="disabled", text="Detecting...")
                self.progress.grid()  # Show progress bar
                self.progress.start()
                self.root.update()
                
                # Run detection
                results = self.model.predict(
                    source=self.current_image_path,
                    conf=self.confidence_var.get(),
                    iou=self.iou_var.get(),
                    verbose=False
                )
                
                # Process results
                self.detection_results = results[0]
                self.display_results()
                
                # Update UI for completion
                self.status_var.set("✅ Detection complete")
                self.detect_btn.config(state="normal", text="Detect Safety Equipment")
                self.progress.stop()
                self.progress.grid_remove()  # Hide progress bar
                
            except Exception as e:
                messagebox.showerror("Error", f"Detection failed: {str(e)}")
                self.status_var.set("❌ Detection failed")
                self.detect_btn.config(state="normal", text="Detect Safety Equipment")
                self.progress.stop()
                self.progress.grid_remove()
        
        # Run detection in separate thread to prevent UI freezing
        thread = threading.Thread(target=detection_thread)
        thread.daemon = True
        thread.start()
    
    def clear_results(self):
        """Clear all result displays"""
        self.summary_text.delete(1.0, tk.END)
        self.details_text.delete(1.0, tk.END)
        self.safety_text.delete(1.0, tk.END)
    
    def update_summary(self, text):
        """Update summary tab"""
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, text)
    
    def update_details(self, text):
        """Update details tab"""
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, text)
    
    def update_safety(self, text):
        """Update safety check tab"""
        self.safety_text.delete(1.0, tk.END)
        self.safety_text.insert(tk.END, text)
    
    def display_results(self):
        """Display detection results"""
        if not self.detection_results:
            return
        
        # Get annotated image
        annotated_img = self.detection_results.plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(annotated_img)
        pil_img.thumbnail((600, 400), Image.Resampling.LANCZOS)
        
        # Update display
        self.current_image = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.current_image)
        
        # Process detection results
        detections = self.detection_results.boxes
        if len(detections) > 0:
            class_names = self.detection_results.names
            detected_objects = {}
            
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls_id]
                
                if class_name not in detected_objects:
                    detected_objects[class_name] = []
                detected_objects[class_name].append(conf)
            
            # Update Summary tab
            summary_text = f"🎯 Detection Summary\n"
            summary_text += f"{'='*30}\n\n"
            summary_text += f"📊 Total Objects: {len(detections)}\n"
            summary_text += f"🏷️ Classes Found: {len(detected_objects)}\n\n"
            summary_text += f"🔍 Detected Equipment:\n"
            for obj_name, confidences in detected_objects.items():
                avg_conf = np.mean(confidences)
                summary_text += f"• {obj_name}: {len(confidences)} ({avg_conf:.1%})\n"
            
            self.update_summary(summary_text)
            
            # Update Details tab
            details_text = f"📋 Detailed Detection Results\n"
            details_text += f"{'='*40}\n\n"
            details_text += f"Image: {Path(self.current_image_path).name}\n"
            details_text += f"Confidence Threshold: {self.confidence_var.get():.2f}\n"
            details_text += f"IoU Threshold: {self.iou_var.get():.2f}\n\n"
            
            details_text += f"Per-Class Breakdown:\n"
            details_text += f"{'-'*25}\n"
            for obj_name, confidences in detected_objects.items():
                avg_conf = np.mean(confidences)
                max_conf = max(confidences)
                min_conf = min(confidences)
                details_text += f"\n{obj_name}:\n"
                details_text += f"  Count: {len(confidences)}\n"
                details_text += f"  Avg Confidence: {avg_conf:.2%}\n"
                details_text += f"  Range: {min_conf:.2%} - {max_conf:.2%}\n"
            
            self.update_details(details_text)
            
            # Update Safety Check tab
            required_equipment = [
                "FireExtinguisher", "FireAlarm", "FirstAidBox", 
                "EmergencyPhone", "OxygenTank"
            ]
            
            safety_text = f"🛡️ Safety Equipment Status\n"
            safety_text += f"{'='*35}\n\n"
            safety_text += f"Critical Equipment Check:\n"
            safety_text += f"{'-'*25}\n\n"
            
            present_count = 0
            for equipment in required_equipment:
                if equipment in detected_objects:
                    count = len(detected_objects[equipment])
                    avg_conf = np.mean(detected_objects[equipment])
                    safety_text += f"✅ {equipment}: {count} detected ({avg_conf:.1%})\n"
                    present_count += 1
                else:
                    safety_text += f"⚠️ {equipment}: Not detected\n"
            
            safety_text += f"\n📊 Safety Score: {present_count}/{len(required_equipment)} ({present_count/len(required_equipment)*100:.0f}%)\n\n"
            
            if present_count == len(required_equipment):
                safety_text += f"🎉 All critical safety equipment present!\n"
            elif present_count >= len(required_equipment) * 0.8:
                safety_text += f"⚠️ Most safety equipment present, but some missing.\n"
            else:
                safety_text += f"🚨 Multiple safety equipment items missing!\n"
            
            self.update_safety(safety_text)
            
        else:
            # No detections found
            no_detection_text = f"❌ No Safety Equipment Detected\n"
            no_detection_text += f"{'='*35}\n\n"
            no_detection_text += f"Image: {Path(self.current_image_path).name}\n"
            no_detection_text += f"Confidence: {self.confidence_var.get():.2f}\n\n"
            no_detection_text += f"💡 Suggestions:\n"
            no_detection_text += f"• Lower the confidence threshold\n"
            no_detection_text += f"• Check if image contains safety equipment\n"
            no_detection_text += f"• Try a different image\n"
            
            self.update_summary(no_detection_text)
            self.update_details(no_detection_text)
            self.update_safety(no_detection_text)

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = SafetyDetectorApp(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
