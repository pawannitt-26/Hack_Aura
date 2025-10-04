from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import time
import sys
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional
import gc


def setup_logging(log_file: str = "prediction.log") -> logging.Logger:
    """Setup logging configuration for the prediction process."""
    logger = logging.getLogger("predict")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def predict_and_save(model, image_path: Path, output_path: Path, output_path_txt: Path, 
                    logger: logging.Logger, conf_threshold: float = 0.5) -> Tuple[bool, str]:
    """
    Perform prediction and save results for a single image.
    
    Args:
        model: YOLO model instance
        image_path: Path to input image
        output_path: Path to save predicted image
        output_path_txt: Path to save bounding box labels
        logger: Logger instance
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Perform prediction
        results = model.predict(
            str(image_path), 
            conf=conf_threshold, 
            save=False, 
            verbose=False,
            device='cpu'  # Force CPU to avoid GPU memory issues
        )

        result = results[0]
        
        # Draw boxes on the image
        img = result.plot()  # Plots the predictions directly on the image

        # Save the result
        cv2.imwrite(str(output_path), img)
        
        # Save the bounding box data
        with open(output_path_txt, 'w') as f:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract the class id and bounding box coordinates
                    cls_id = int(box.cls)
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    
                    # Write bbox information in the format [class_id, x_center, y_center, width, height]
                    f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
            else:
                # No detections found
                f.write("")
        
        return True, f"Successfully processed {image_path.name} - {len(result.boxes) if result.boxes else 0} detections"
        
    except Exception as e:
        error_msg = f"Error processing {image_path.name}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def process_images_batch(model, image_paths: List[Path], output_dir: Path, 
                        logger: logging.Logger, conf_threshold: float = 0.5,
                        batch_size: int = 10) -> Tuple[int, int, List[str]]:
    """
    Process images in batches with progress tracking and error handling.
    
    Args:
        model: YOLO model instance
        image_paths: List of image paths to process
        output_dir: Output directory for results
        logger: Logger instance
        conf_threshold: Confidence threshold for predictions
        batch_size: Number of images to process in each batch
        
    Returns:
        Tuple of (successful_count, failed_count, error_messages)
    """
    # Create output directories
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_count = 0
    failed_count = 0
    error_messages = []
    
    # Process images in batches
    total_images = len(image_paths)
    logger.info(f"Starting prediction for {total_images} images in batches of {batch_size}")
    
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        for i in range(0, total_images, batch_size):
            batch = image_paths[i:i + batch_size]
            batch_start_time = time.time()
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
            
            for img_path in batch:
                if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    pbar.update(1)
                    continue
                    
                output_path_img = images_output_dir / img_path.name
                output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name
                
                success, message = predict_and_save(
                    model, img_path, output_path_img, output_path_txt, logger, conf_threshold
                )
                
                if success:
                    successful_count += 1
                    logger.debug(message)
                else:
                    failed_count += 1
                    error_messages.append(message)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': successful_count,
                    'Failed': failed_count,
                    'Current': img_path.name[:20] + '...' if len(img_path.name) > 20 else img_path.name
                })
            
            # Memory cleanup after each batch
            gc.collect()
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch completed in {batch_time:.2f}s")
    
    return successful_count, failed_count, error_messages


def find_model_path(this_dir: Path, logger: logging.Logger) -> Path:
    """Find the best available model path with fallback options."""
    model_path = None
    
    # First, try to find the trained model from common locations
    possible_paths = [
        Path("models/safety_equipment_model.pt"),  # Local models directory
        Path("../models/safety_equipment_model.pt"),  # Parent models directory
        Path("/opt/homebrew/runs/detect/train2/weights/best.pt"),  # Homebrew on macOS
        Path("runs/detect/safety_equipment_training/weights/best.pt"),  # Local runs directory
        Path("runs/train/safety_equipment_training/weights/best.pt"),  # Alternative runs structure
    ]
    
    model_path = None
    for trained_model_path in possible_paths:
        if trained_model_path.exists():
            model_path = trained_model_path
            logger.info(f"Using trained model from: {model_path}")
            break
    
    if model_path is None:
        # Try local runs directory
        detect_path = this_dir / "runs" / "detect"
        if detect_path.exists():
            train_folders = [
                f for f in os.listdir(detect_path)
                if os.path.isdir(detect_path / f) and f.startswith("train")
            ]
            if len(train_folders) > 0:
                # Auto-select the most recent training run (non-interactive)
                most_recent = max(
                    train_folders,
                    key=lambda f: (detect_path / f).stat().st_mtime
                )
                candidate = detect_path / most_recent / "weights" / "best.pt"
                if candidate.exists():
                    model_path = candidate
                    logger.info(f"Using local trained model from: {model_path}")
    
    # Fallback to a local weights file if no training runs are found
    if model_path is None:
        fallback_weights = this_dir / "models" / "pretrained_yolov8s.pt"
        if fallback_weights.exists():
            model_path = fallback_weights
            logger.info(f"Using fallback model from: {model_path}")
        else:
            raise FileNotFoundError(
                f"No trained weights found. Please ensure the model is available at:\n"
                f"1. {trained_model_path}\n"
                f"2. {detect_path}/train*/weights/best.pt\n"
                f"3. {fallback_weights}"
            )
    
    return model_path


def main():
    """Main function to run the prediction process."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting prediction process")
    
    start_time = time.time()
    
    try:
        # Get script directory and change to it
        this_dir = Path(__file__).parent
        os.chdir(this_dir)
        
        # Load configuration
        config_file = this_dir / 'configs' / 'dataset_config.yaml'
        if not config_file.exists():
            logger.error(f"Configuration file {config_file} not found")
            sys.exit(1)
            
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
            if 'test' in data and data['test'] is not None:
                images_dir = Path(data['test']) / 'images'
            else:
                logger.error("No test field found in yolo_params.yaml, please add the test field with the path to the test images")
                sys.exit(1)
        
        # Validate images directory
        if not images_dir.exists():
            logger.error(f"Images directory {images_dir} does not exist")
            sys.exit(1)

        if not images_dir.is_dir():
            logger.error(f"Images directory {images_dir} is not a directory")
            sys.exit(1)
        
        if not any(images_dir.iterdir()):
            logger.error(f"Images directory {images_dir} is empty")
            sys.exit(1)
        
        # Find and load model
        logger.info("Loading YOLO model...")
        model_path = find_model_path(this_dir, logger)
        model = YOLO(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
        
        # Get list of images to process
        image_paths = [img_path for img_path in images_dir.glob('*') 
                      if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        if not image_paths:
            logger.error("No valid image files found in the test directory")
            sys.exit(1)
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process images
        output_dir = this_dir / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images in batches
        successful_count, failed_count, error_messages = process_images_batch(
            model, image_paths, output_dir, logger, conf_threshold=0.5, batch_size=10
        )
        
        # Print summary
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {len(image_paths)}")
        logger.info(f"Successful predictions: {successful_count}")
        logger.info(f"Failed predictions: {failed_count}")
        logger.info(f"Success rate: {(successful_count/len(image_paths)*100):.1f}%")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Average time per image: {total_time/len(image_paths):.2f} seconds")
        logger.info(f"Predicted images saved in: {output_dir / 'images'}")
        logger.info(f"Bounding box labels saved in: {output_dir / 'labels'}")
        
        if error_messages:
            logger.warning(f"Errors encountered: {len(error_messages)}")
            for error in error_messages[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
            if len(error_messages) > 5:
                logger.warning(f"  ... and {len(error_messages) - 5} more errors")
        
        # Run validation if requested
        logger.info("Running model validation...")
        try:
            metrics = model.val(
                data=str(config_file),
                split="test",
                project=str(this_dir / "runs"),
                name="val_test",
                save=True,
                verbose=False
            )
            logger.info("Validation completed successfully")
        except Exception as e:
            logger.warning(f"Validation failed: {str(e)}")
        
        logger.info("Prediction process completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in prediction process: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()