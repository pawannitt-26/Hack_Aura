import os
import cv2
import sys
from functools import lru_cache


class YoloVisualizer:
    MODE_TRAIN = 0
    MODE_VAL = 1
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        classes_file = os.path.join(dataset_folder, "configs", "class_names.txt")
        
        # Validate dataset folder exists
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")
        
        # Load classes with error handling
        try:
            with open(classes_file, "r") as f:
                self.classes = f.read().splitlines()
            self.classes = {i: c for i, c in enumerate(self.classes)}
        except FileNotFoundError:
            print(f"Error: class_names.txt not found in {dataset_folder}/configs")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading classes: {e}")
            sys.exit(1)
            
        self.set_mode(YoloVisualizer.MODE_TRAIN)
    
    def _find_valid_path(self, possible_paths, path_type):
        """Find the first valid path from a list of possible paths"""
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found {path_type} at: {path}")
                return path
        # If no valid path found, return the first one for error reporting
        return possible_paths[0]
    
    def set_mode(self, mode=MODE_TRAIN):
        if mode == self.MODE_TRAIN:
            # Try different possible paths for training data
            possible_train_paths = [
                os.path.join(self.dataset_folder, "..", "datasets", "training", "images"),
                os.path.join(self.dataset_folder, "train", "images"),
                os.path.join(self.dataset_folder, "train2", "images")
            ]
            self.images_folder = self._find_valid_path(possible_train_paths, "training images")
            self.labels_folder = self.images_folder.replace("images", "labels")
            mode_name = "training"
        else:
            # Try different possible paths for validation data
            possible_val_paths = [
                os.path.join(self.dataset_folder, "..", "datasets", "validation", "images"),
                os.path.join(self.dataset_folder, "val", "images"),
                os.path.join(self.dataset_folder, "val2", "images")
            ]
            self.images_folder = self._find_valid_path(possible_val_paths, "validation images")
            self.labels_folder = self.images_folder.replace("images", "labels")
            mode_name = "validation"
        
        # Validate folders exist
        if not os.path.exists(self.images_folder):
            print(f"Error: {mode_name} images folder not found: {self.images_folder}")
            sys.exit(1)
        if not os.path.exists(self.labels_folder):
            print(f"Error: {mode_name} labels folder not found: {self.labels_folder}")
            sys.exit(1)
        
        try:
            self.num_images = len(os.listdir(self.images_folder))
            num_labels = len(os.listdir(self.labels_folder))
            self.label_names = sorted(os.listdir(self.labels_folder))
            self.image_names = sorted(os.listdir(self.images_folder))
            
            if self.num_images != num_labels:
                print(f"Warning: Mismatch between images ({self.num_images}) and labels ({num_labels}) in {mode_name} set")
            if self.num_images == 0:
                print(f"Error: No images found in {mode_name} set")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error accessing {mode_name} data: {e}")
            sys.exit(1)
            
        self.frame_index = 0
        print(f"Loaded {mode_name} set with {self.num_images} images")


    def next_frame(self):
        self.frame_index += 1
        if self.frame_index >= self.num_images:
            self.frame_index = 0

    def previous_frame(self):
        self.frame_index -= 1
        if self.frame_index < 0:
            self.frame_index = self.num_images - 1
    
    @lru_cache(maxsize=10)  # Cache last 10 images for better performance
    def _load_image_cached(self, image_path):
        """Cached image loading for better performance"""
        return cv2.imread(image_path)
    
    def seek_frame(self, idx):
        if idx < 0 or idx >= self.num_images:
            print(f"Error: Invalid frame index {idx}. Must be between 0 and {self.num_images - 1}")
            return None
            
        image_file = os.path.join(self.images_folder, self.image_names[idx])
        label_file = os.path.join(self.labels_folder, self.label_names[idx])
        
        # Load image with caching and error handling
        image = self._load_image_cached(image_file)
        if image is None:
            print(f"Error: Could not load image {image_file}")
            return None
        
        # Load labels with error handling
        try:
            with open(label_file, "r") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            print(f"Warning: Label file not found: {label_file}")
            return image
        except Exception as e:
            print(f"Error loading labels: {e}")
            return image
        
        # Draw bounding boxes
        for line in lines:
            if not line.strip():  # Skip empty lines
                continue
            try:
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Invalid label format in {label_file}: {line}")
                    continue
                    
                class_index, x, y, w, h = map(float, parts)
                class_index = int(class_index)
                
                if class_index not in self.classes:
                    print(f"Warning: Unknown class index {class_index} in {label_file}")
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                cx = int(x * image.shape[1])
                cy = int(y * image.shape[0])
                w = int(w * image.shape[1])
                h = int(h * image.shape[0])
                x = cx - w // 2
                y = cy - h // 2
                
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, self.classes[class_index], (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except ValueError as e:
                print(f"Warning: Invalid label data in {label_file}: {line} - {e}")
                continue
        
        return image

    def run(self):
        print("\n=== YOLO Dataset Visualizer ===")
        print("Controls:")
        print("  d/SPACE - Next image")
        print("  a/BACKSPACE - Previous image")
        print("  t - Switch to training set")
        print("  v - Switch to validation set")
        print("  r - Reset to first frame")
        print("  e - Jump to last frame")
        print("  1-9 - Jump to specific frame")
        print("  h/? - Show help")
        print("  q/ESC - Quit")
        print("=" * 30)
        
        while True:
            frame = self.seek_frame(self.frame_index)
            if frame is None:
                print("Error: Could not load frame, skipping...")
                self.next_frame()
                continue
                
            # Resize frame for display
            frame = cv2.resize(frame, (640, 480))
            
            # Add frame information overlay
            mode_text = "TRAIN" if hasattr(self, 'images_folder') and 'train' in self.images_folder else "VAL"
            info_text = f"{mode_text} - Frame {self.frame_index + 1}/{self.num_images}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Add filename overlay
            filename = self.image_names[self.frame_index] if self.frame_index < len(self.image_names) else "Unknown"
            cv2.putText(frame, filename, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, filename, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cv2.imshow(f"Yolo Visualizer - {os.path.basename(self.dataset_folder)}", frame)
            key = cv2.waitKey(0)
            
            if key == ord('q') or key == 27 or key == -1:
                break
            elif key == ord('d') or key == ord(' '):  # Space bar also works for next
                self.next_frame()
            elif key == ord('a') or key == 8:  # Backspace also works for previous
                self.previous_frame()
            elif key == ord('t'):
                self.set_mode(YoloVisualizer.MODE_TRAIN)
            elif key == ord('v'):
                self.set_mode(YoloVisualizer.MODE_VAL)
            elif key == ord('h') or key == ord('?'):
                print("\n=== Help ===")
                print("d or SPACE - Next image")
                print("a or BACKSPACE - Previous image")
                print("t - Switch to training set")
                print("v - Switch to validation set")
                print("h or ? - Show this help")
                print("q or ESC - Quit")
                print("=" * 15)
            elif key == ord('r'):  # Reset to first frame
                self.frame_index = 0
                print("Reset to first frame")
            elif key == ord('e'):  # Go to end
                self.frame_index = self.num_images - 1
                print(f"Jumped to last frame ({self.num_images})")
            elif key >= ord('0') and key <= ord('9'):  # Jump to specific frame
                try:
                    target_frame = int(chr(key))
                    if target_frame == 0:
                        target_frame = 10  # 0 key goes to frame 10
                    if target_frame <= self.num_images:
                        self.frame_index = target_frame - 1
                        print(f"Jumped to frame {target_frame}")
                    else:
                        print(f"Frame {target_frame} does not exist (max: {self.num_images})")
                except ValueError:
                    pass
        
        cv2.destroyAllWindows()
        print("Visualizer closed.")


if __name__ == "__main__":
    try:
        dataset_path = os.path.dirname(__file__)
        print(f"Loading dataset from: {dataset_path}")
        vis = YoloVisualizer(dataset_path)
        vis.run()
    except KeyboardInterrupt:
        print("\nVisualizer interrupted by user.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
