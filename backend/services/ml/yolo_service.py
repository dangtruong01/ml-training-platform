import subprocess
import os
import zipfile
import uuid
import cv2
import numpy as np
import json
import yaml
import shutil
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from fastapi import UploadFile
from ultralytics import YOLO

# Import new modular components
from .base.training_monitor import training_monitor
from .datasets.yolo_processor import YOLODatasetProcessor
from .detection.yolo_trainer import YOLODetectionTrainer
from .segmentation.yolo_trainer import YOLOSegmentationTrainer
from .anomaly.sklearn_trainer import SklearnAnomalyTrainer
from .anomaly.pytorch_trainer import PytorchAnomalyTrainer
from .prediction.yolo_predictor import YOLOPredictor
from .pre_annotation.opencv_annotator import OpenCVAnnotator
from .raw_annotation.raw_processor import RawAnnotationProcessor

# Import database service for cloud training methods
try:
    from backend.services.core.database_service import database_service
except ImportError:
    from services.core.database_service import database_service

class YoloService:
    def __init__(self, scripts_dir: str = "ml/scripts", datasets_dir: str = "ml/datasets"):
        self.scripts_dir = os.path.abspath(scripts_dir)
        self.datasets_dir = os.path.abspath(datasets_dir)
        self.results_dir = os.path.abspath("ml/results")
        
        # Initialize modular components
        self.dataset_processor = YOLODatasetProcessor(datasets_dir)
        self.detection_trainer = YOLODetectionTrainer(scripts_dir, self.results_dir)
        self.segmentation_trainer = YOLOSegmentationTrainer(scripts_dir, self.results_dir)
        self.sklearn_anomaly_trainer = SklearnAnomalyTrainer(self.results_dir)
        self.pytorch_anomaly_trainer = PytorchAnomalyTrainer(self.results_dir)
        self.yolo_predictor = YOLOPredictor("ml/models")
        self.opencv_annotator = OpenCVAnnotator(self.results_dir)
        self.raw_processor = RawAnnotationProcessor(datasets_dir)
        # Use global training_monitor instance (imported from base module)
        
        # Legacy training progress tracking (will be phased out)
        self.training_processes: Dict[str, dict] = {}
        self.training_logs: Dict[str, list] = {}
        
        # Ensure directories exist
        os.makedirs(self.scripts_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Try to use the best available model
        model_priority = ["yolov8l.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]
        
        for model_name in model_priority:
            model_path = os.path.join("ml", "models", model_name)
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Using model: {model_name}")
                break
        else:
            # Fallback to downloading nano model
            self.model = YOLO("yolov8n.pt")
            print("Using fallback nano model")

    async def handle_dataset_upload(self, file: UploadFile, task_type: str) -> str | None:
        """Saves and unzips an uploaded dataset, returning the path to data.yaml."""
        # Use new modular dataset processor
        return await self.dataset_processor.process_uploaded_dataset(file, task_type)

    def _find_data_yaml(self, directory: str) -> Optional[str]:
        """Recursively find data.yaml file in directory"""
        # Delegate to dataset processor
        return self.dataset_processor.find_data_yaml(directory)

    def _validate_dataset_structure(self, data_yaml_path: str) -> bool:
        """Validate that the dataset has proper YOLO structure"""
        # Delegate to dataset processor
        return self.dataset_processor.validate_dataset_structure(data_yaml_path)

    def train_detection(self, dataset_path: str, device: str = "cpu") -> str:
        """Train a YOLO detection model using the provided dataset"""
        # Delegate to modular detection trainer
        config = {
            'epochs': 10,
            'batch_size': 16,
            'device': device,
            'model_size': 'n'
        }
        return self.detection_trainer.train(dataset_path, config)

    def train_segmentation(self, dataset_path: str, device: str = "cpu") -> str:
        """Train a YOLO segmentation model using the provided dataset"""
        # Delegate to modular segmentation trainer
        config = {
            'epochs': 10,
            'batch_size': 16,
            'device': device,
            'model_size': 'n'
        }
        return self.segmentation_trainer.train(dataset_path, config)

    # Legacy training monitoring methods removed - now handled by TrainingMonitor

    def get_training_status(self, task_id: str) -> dict:
        """Get the current status of a training task"""
        # Use new training monitor, fall back to legacy if not found
        status = training_monitor.get_task_status(task_id)
        if status.get('status') != 'not_found':
            return status
        
        # Legacy fallback
        if task_id not in self.training_processes:
            return {'error': 'Task not found'}
        
        status_info = self.training_processes[task_id].copy()
        
        # Add recent logs
        if task_id in self.training_logs:
            status_info['recent_logs'] = self.training_logs[task_id][-10:]  # Last 10 entries
        
        return status_info

    def get_training_logs(self, task_id: str, lines: int = 50) -> dict:
        """Get recent training logs"""
        # Use new training monitor, fall back to legacy if not found
        logs = training_monitor.get_task_logs(task_id, lines)
        if logs.get('status') != 'not_found':
            return logs
        
        # Legacy fallback
        if task_id not in self.training_processes:
            return {'error': 'Task not found'}
        
        # Try to read from log file if it exists
        log_file = self.training_processes[task_id].get('log_file')
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()
                    recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    return {
                        'task_id': task_id,
                        'logs': [line.strip() for line in recent_lines],
                        'total_lines': len(all_lines)
                    }
            except Exception as e:
                pass
        
        # Fallback to in-memory logs
        if task_id in self.training_logs:
            recent_logs = self.training_logs[task_id][-lines:]
            return {
                'task_id': task_id,
                'logs': [f"[{log['timestamp']}] {log['message']}" for log in recent_logs],
                'total_lines': len(self.training_logs[task_id])
            }
        
        return {'task_id': task_id, 'logs': [], 'total_lines': 0}

    def list_training_tasks(self) -> dict:
        """List all training tasks and their status"""
        # Use new training monitor first
        return training_monitor.list_tasks()

    # Removed _store_training_process - handled by TrainingMonitor

    # Prediction methods - delegated to modular predictor
    def predict_detection(self, image_path: str, model_path: str = None) -> dict:
        """Run detection prediction with quality assessment"""
        return self.yolo_predictor.predict_detection(image_path, model_path)

    def predict_segmentation(self, image_path: str, model_path: str = None) -> dict:
        """Run segmentation prediction with quality assessment"""
        return self.yolo_predictor.predict_segmentation(image_path, model_path)

    def pre_annotate_detection(self, image_path: str) -> str:
        """Pre-annotate image for detection using OpenCV"""
        return self.opencv_annotator.annotate_detection(image_path)
    
    def pre_annotate_segmentation(self, image_path: str) -> str:
        """Pre-annotate image for segmentation using OpenCV"""
        return self.opencv_annotator.annotate_segmentation(image_path)
    
    def pre_annotate_sam2_detection(self, image_path: str) -> str:
        try:
            try:
                from backend.services.annotation.sam2_service import sam2_service
            except ImportError:
                from services.annotation.sam2_service import sam2_service
            return sam2_service.annotate_detection(image_path)
        except ImportError:
            raise RuntimeError("SAM2 service not available. Please install SAM2 dependencies.")
        except Exception as e:
            print(f"SAM2 detection failed: {e}. Falling back to OpenCV.")
            return self._opencv_detection_annotation(image_path)
    
    def pre_annotate_sam2_segmentation(self, image_path: str) -> str:
        try:
            try:
                from backend.services.annotation.sam2_service import sam2_service
            except ImportError:
                from services.annotation.sam2_service import sam2_service
            return sam2_service.annotate_segmentation(image_path)
        except ImportError:
            raise RuntimeError("SAM2 service not available. Please install SAM2 dependencies.")
        except Exception as e:
            print(f"SAM2 segmentation failed: {e}. Falling back to OpenCV.")
            return self._opencv_segmentation_annotation(image_path)

    # Old OpenCV annotation and prediction helper methods removed - now handled by modular components
    
    # Keep remaining utility methods...
    async def upload_model(self, file, model_type: str) -> dict:
        """Upload and save a trained model"""
        models_dir = os.path.join("ml", "models", "uploaded")
        os.makedirs(models_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        annotated_img = img.copy()
        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width
        
        # Save original for debugging
        original_debug = os.path.join(results_dir, f"debug_original_{os.path.basename(image_path)}")
        cv2.imwrite(original_debug, img)
        
        # METHOD 1: Enhanced edge-based detection
        valid_contours = self._edge_based_detection(img, results_dir, image_path)
        
        # METHOD 2: If insufficient contours, try adaptive thresholding
        if len(valid_contours) < 1:
            valid_contours.extend(self._adaptive_threshold_detection(img, img_area, results_dir, image_path))
        
        # METHOD 3: If still insufficient, try color-based detection
        if len(valid_contours) < 1:
            valid_contours.extend(self._color_based_detection(img, img_area))
        
        # Remove duplicates and filter by area
        valid_contours = self._filter_and_merge_contours(valid_contours, img_area)
        
        # Draw bounding boxes (detection style)
        for i, c in enumerate(valid_contours[:5]):
            x, y, w, h = cv2.boundingRect(c)
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)][i % 5]
            
            # Draw bounding rectangle
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
            
            # Add area and confidence info
            area = cv2.contourArea(c)
            label = f"Object {i+1} ({int(area)}px)"
            cv2.putText(annotated_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add detection summary
        summary = f"Detection: {len(valid_contours)} objects found"
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        output_path = os.path.join(results_dir, f"detection_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        return output_path
    
    def _opencv_segmentation_annotation(self, image_path: str) -> str:
        """Enhanced OpenCV-based segmentation with edge detection for metallic objects"""
        results_dir = os.path.abspath(os.path.join("ml", "results", "pre_annotation"))
        os.makedirs(results_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        annotated_img = img.copy()
        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width
        
        # Use enhanced edge detection for segmentation too
        valid_contours = self._edge_based_detection(img, results_dir, image_path)
        
        # If insufficient contours, try adaptive thresholding
        if len(valid_contours) < 1:
            valid_contours.extend(self._adaptive_threshold_detection(img, img_area, results_dir, image_path))
        
        # If still insufficient, try color-based segmentation
        if len(valid_contours) < 1:
            valid_contours.extend(self._color_based_segmentation(img, img_area))
        
        # Filter and merge for segmentation
        valid_contours = self._filter_and_merge_contours(valid_contours, img_area)
        
        # Draw detailed contours (segmentation style) with more precise contours
        for i, c in enumerate(valid_contours[:3]):
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i % 3]
            
            # For segmentation, we want more precise contours
            # Approximate contour to reduce noise while preserving shape
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx_contour = cv2.approxPolyDP(c, epsilon, True)
            
            # Draw filled contour with transparency
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [approx_contour], 255)
            
            # Create colored overlay
            colored_mask = np.zeros_like(img)
            colored_mask[mask == 255] = color
            annotated_img = cv2.addWeighted(annotated_img, 0.7, colored_mask, 0.3, 0)
            
            # Draw precise contour outline
            cv2.drawContours(annotated_img, [approx_contour], -1, color, 2)
            
            # Add label with area information
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            label = f"Segment {i+1} ({int(area)}px)"
            cv2.putText(annotated_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add segmentation summary
        summary = f"Segmentation: {len(valid_contours)} segments found"
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        output_path = os.path.join(results_dir, f"segmentation_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        return output_path
    
    def _color_based_detection(self, img, img_area):
        """Helper method for color-based object detection"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-blue areas (metallic objects typically not blue)
        lower_blue = np.array([90, 70, 70])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        non_blue_mask = cv2.bitwise_not(blue_mask)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(non_blue_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if 1000 < cv2.contourArea(c) < 0.8 * img_area]
    
    def _color_based_segmentation(self, img, img_area):
        """Helper method for color-based segmentation"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # More refined color segmentation
        lower_blue = np.array([90, 70, 70])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        non_blue_mask = cv2.bitwise_not(blue_mask)
        
        # Finer morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(non_blue_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours with better approximation
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return [c for c in contours if 1000 < cv2.contourArea(c) < 0.8 * img_area]
    
    def _edge_based_detection(self, img, results_dir, image_path):
        """Enhanced edge-based detection for metallic objects with varying backgrounds"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle different lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Save enhanced image for debugging
        enhanced_debug = os.path.join(results_dir, f"debug_enhanced_{os.path.basename(image_path)}")
        cv2.imwrite(enhanced_debug, enhanced)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Use Canny edge detection with adaptive thresholds
        # Calculate dynamic thresholds based on image statistics
        v = np.median(bilateral)
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        
        edges = cv2.Canny(bilateral, lower_thresh, upper_thresh)
        
        # Save edges for debugging
        edges_debug = os.path.join(results_dir, f"debug_edges_{os.path.basename(image_path)}")
        cv2.imwrite(edges_debug, edges)
        
        # Dilate edges to connect nearby edge pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Close gaps in edges
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close)
        
        # Save processed edges for debugging
        edges_processed_debug = os.path.join(results_dir, f"debug_edges_processed_{os.path.basename(image_path)}")
        cv2.imwrite(edges_processed_debug, edges_closed)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = img.shape[0] * img.shape[1]
        valid_contours = []
        
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            # Filter by area (not too small, not too large)
            if area < 1500 or area > 0.7 * img_area:
                continue
                
            # Filter by aspect ratio and solidity for metallic objects
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Metallic objects usually have reasonable aspect ratios and decent solidity
            if 0.1 < aspect_ratio < 10 and solidity > 0.3:
                valid_contours.append(c)
        
        return valid_contours
    
    def _adaptive_threshold_detection(self, img, img_area, results_dir, image_path):
        """Adaptive thresholding for different lighting conditions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Use adaptive threshold to handle varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Save adaptive threshold for debugging
        adaptive_debug = os.path.join(results_dir, f"debug_adaptive_{os.path.basename(image_path)}")
        cv2.imwrite(adaptive_debug, adaptive_thresh)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if 1200 < area < 0.6 * img_area:
                # Additional filtering for shape characteristics
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                if 0.1 < aspect_ratio < 8:  # Reasonable aspect ratio
                    valid_contours.append(c)
        
        return valid_contours
    
    def _filter_and_merge_contours(self, contours, img_area):
        """Filter and merge overlapping contours"""
        if not contours:
            return []
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        merged_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip if too small or too large
            if area < 800 or area > 0.8 * img_area:
                continue
            
            # Check if this contour overlaps significantly with existing ones
            is_duplicate = False
            current_rect = cv2.boundingRect(contour)
            
            for existing in merged_contours:
                existing_rect = cv2.boundingRect(existing)
                
                # Calculate intersection over union (IoU)
                iou = self._calculate_bbox_iou(current_rect, existing_rect)
                if iou > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_contours.append(contour)
                
                # Limit to avoid too many detections
                if len(merged_contours) >= 5:
                    break
        
        return merged_contours
    
    def _calculate_bbox_iou(self, box1, box2):
        """Calculate Intersection over Union for bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _process_detection_results(self, result, image_path: str) -> dict:
        """Process YOLO detection results and assess quality"""
        import cv2
        import base64
        from pathlib import Path
        
        # Load original image
        img = cv2.imread(image_path)
        annotated_img = img.copy()
        
        detections = []
        total_defects = 0
        confidence_scores = []
        
        if result.boxes is not None:
            for box in result.boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(annotated_img, f"{class_name}: {confidence:.2f}", 
                           (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                detections.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
                
                # Count defects (assuming scratch/dent classes indicate defects)
                if class_name.lower() in ['scratch', 'dent', 'defect', 'damage']:
                    total_defects += 1
                    confidence_scores.append(confidence)
        
        # Quality assessment logic
        quality_status = self._assess_quality_from_detections(total_defects, confidence_scores)
        
        # Save annotated image
        results_dir = os.path.join("ml", "results", "predictions")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"prediction_{Path(image_path).stem}.jpg")
        cv2.imwrite(output_path, annotated_img)
        
        # Convert image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "quality": quality_status,
            "detections": detections,
            "total_defects": total_defects,
            "image_base64": img_base64,
            "output_path": output_path
        }

    def _process_segmentation_results(self, result, image_path: str) -> dict:
        """Process YOLO segmentation results and assess quality"""
        import cv2
        import base64
        import numpy as np
        from pathlib import Path
        
        # Load original image
        img = cv2.imread(image_path)
        annotated_img = img.copy()
        
        detections = []
        total_defects = 0
        confidence_scores = []
        total_defect_area = 0
        
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            
            for i, mask in enumerate(masks):
                if boxes is not None:
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    
                    # Resize mask to image size
                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Calculate defect area
                    defect_area = np.sum(mask_binary)
                    
                    # Create colored overlay
                    color = (0, 0, 255) if class_name.lower() in ['scratch', 'dent', 'defect'] else (0, 255, 0)
                    annotated_img[mask_binary == 1] = annotated_img[mask_binary == 1] * 0.6 + np.array(color) * 0.4
                    
                    detections.append({
                        "class": class_name,
                        "confidence": float(confidence),
                        "area": int(defect_area)
                    })
                    
                    # Count defects
                    if class_name.lower() in ['scratch', 'dent', 'defect', 'damage']:
                        total_defects += 1
                        confidence_scores.append(confidence)
                        total_defect_area += defect_area
        
        # Quality assessment for segmentation
        quality_status = self._assess_quality_from_segmentation(total_defects, confidence_scores, total_defect_area, img.shape[0] * img.shape[1])
        
        # Save result
        results_dir = os.path.join("ml", "results", "predictions")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"segmentation_{Path(image_path).stem}.jpg")
        cv2.imwrite(output_path, annotated_img)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "quality": quality_status,
            "detections": detections,
            "total_defects": total_defects,
            "total_defect_area": total_defect_area,
            "image_base64": img_base64,
            "output_path": output_path
        }

    def _assess_quality_from_detections(self, total_defects: int, confidence_scores: list) -> dict:
        """Assess quality based on detection results"""
        # Quality rules for detection
        if total_defects == 0:
            return {
                "status": "OK",
                "confidence": 0.95,
                "reason": "No defects detected"
            }
        elif total_defects <= 2 and all(conf < 0.6 for conf in confidence_scores):
            return {
                "status": "OK",
                "confidence": 0.75,
                "reason": f"Minor defects detected ({total_defects}) with low confidence"
            }
        else:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            return {
                "status": "NG",
                "confidence": float(avg_confidence),
                "reason": f"Significant defects detected ({total_defects})"
            }

    def _assess_quality_from_segmentation(self, total_defects: int, confidence_scores: list, total_area: int, image_area: int) -> dict:
        """Assess quality based on segmentation results"""
        area_percentage = (total_area / image_area) * 100 if image_area > 0 else 0
        
        if total_defects == 0:
            return {
                "status": "OK",
                "confidence": 0.95,
                "reason": "No defects detected"
            }
        elif area_percentage < 1.0 and all(conf < 0.7 for conf in confidence_scores):
            return {
                "status": "OK",
                "confidence": 0.70,
                "reason": f"Minor defects (<1% area)"
            }
        else:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            return {
                "status": "NG",
                "confidence": float(avg_confidence),
                "reason": f"Significant defects ({area_percentage:.1f}% area)"
            }

    async def upload_model(self, file, model_type: str) -> dict:
        """Upload and save a trained model"""
        models_dir = os.path.join("ml", "models", "uploaded")
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate unique filename
        import uuid
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{model_type}_{uuid.uuid4().hex[:8]}.{file_extension}"
        model_path = os.path.join(models_dir, unique_filename)
        
        # Save file
        with open(model_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Test load the model to validate
        try:
            test_model = YOLO(model_path)
            model_info = {
                "status": "success",
                "model_path": model_path,
                "model_type": model_type,
                "filename": unique_filename,
                "original_name": file.filename,
                "classes": test_model.names if hasattr(test_model, 'names') else None
            }
            print(f"✅ Model uploaded successfully: {model_path}")
            return model_info
        except Exception as e:
            # Remove invalid model file
            if os.path.exists(model_path):
                os.remove(model_path)
            raise RuntimeError(f"Invalid model file: {e}")

    def list_available_models(self) -> dict:
        """List all available models"""
        models = {
            "pretrained": [],
            "trained": [],
            "uploaded": []
        }
        
        # Pretrained models
        pretrained_dir = os.path.join("ml", "models")
        if os.path.exists(pretrained_dir):
            for file in os.listdir(pretrained_dir):
                if file.endswith('.pt'):
                    models["pretrained"].append({
                        "name": file,
                        "path": os.path.join(pretrained_dir, file),
                        "type": "segmentation" if "seg" in file else "detection"
                    })
        
        # Trained models from training results
        results_dir = os.path.join("ml", "results", "detection")
        if os.path.exists(results_dir):
            for training_folder in os.listdir(results_dir):
                weights_dir = os.path.join(results_dir, training_folder, "yolo_detection_model", "weights")
                if os.path.exists(weights_dir):
                    best_path = os.path.join(weights_dir, "best.pt")
                    if os.path.exists(best_path):
                        models["trained"].append({
                            "name": f"{training_folder}_best.pt",
                            "path": best_path,
                            "type": "detection",
                            "training_id": training_folder
                        })
        
        # Uploaded models
        uploaded_dir = os.path.join("ml", "models", "uploaded")
        if os.path.exists(uploaded_dir):
            for file in os.listdir(uploaded_dir):
                if file.endswith('.pt'):
                    models["uploaded"].append({
                        "name": file,
                        "path": os.path.join(uploaded_dir, file),
                        "type": "detection" if "detection" in file else "segmentation"
                    })
        
        return models

    async def predict_batch(self, files, model_path: str = None, task_type: str = "detection"):
        """Predict on multiple images"""
        temp_files = []
        image_paths = []
        
        try:
            for file in files:
                # Save temp file
                temp_path = f"temp_batch_{file.filename}"
                temp_files.append(temp_path)
                image_paths.append(temp_path)
                
                with open(temp_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
            
            # Use modular predictor's batch prediction
            result = self.yolo_predictor.predict_batch(image_paths, model_path, task_type)
            return result
        
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def assess_quality(self, image_path: str, model_path: str = None) -> dict:
        """Standalone quality assessment"""
        return self.yolo_predictor.assess_quality_standalone(image_path, model_path)
    
    def process_raw_annotations(self, 
                               raw_folder_path: str, 
                               output_name: str,
                               train_split: float = 0.8,
                               val_split: float = 0.2) -> dict:
        """Process raw annotation folder into YOLO format
        
        Args:
            raw_folder_path: Path to folder containing images/, labels/, classes.txt
            output_name: Name for the processed dataset
            train_split: Fraction of data for training (default: 0.8)
            val_split: Fraction of data for validation (default: 0.2)
            
        Returns:
            dict: Processing results including paths and statistics
        """
        return self.raw_processor.process_raw_folder(raw_folder_path, output_name, train_split, val_split)
    
    def get_dataset_statistics(self, dataset_path: str) -> dict:
        """Get statistics about a YOLO dataset
        
        Args:
            dataset_path: Path to YOLO dataset directory
            
        Returns:
            dict: Dataset statistics including file counts and validation results
        """
        return self.raw_processor.get_dataset_statistics(dataset_path)

    def _generate_batch_summary(self, results: list) -> dict:
        """Generate summary statistics for batch prediction"""
        total = len(results)
        ok_count = sum(1 for r in results if r["quality"]["status"] == "OK")
        ng_count = total - ok_count
        
        return {
            "total_images": total,
            "ok_count": ok_count,
            "ng_count": ng_count,
            "ok_percentage": (ok_count / total * 100) if total > 0 else 0,
            "ng_percentage": (ng_count / total * 100) if total > 0 else 0
        }

    # =============================================================================
    # PROJECT-BASED TRAINING METHODS
    # =============================================================================

    def train_detection_from_project(self, project_id: str, training_config: dict, algorithm: str = "yolo_v8") -> str:
        """Train detection model from project dataset with algorithm-specific routing"""
        # Add algorithm to config and delegate to detection trainer
        training_config['algorithm'] = algorithm
        return self.detection_trainer.train_from_project(project_id, training_config)

    # Old YOLO training method removed - now handled by modular YOLODetectionTrainer

    def train_segmentation_from_project(self, project_id: str, training_config: dict) -> str:
        """Train segmentation model from project dataset"""
        # Delegate to segmentation trainer
        return self.segmentation_trainer.train_from_project(project_id, training_config)

    def train_anomaly_from_project_cloud(self, project_id: str, training_config: dict, algorithm: str = "isolation_forest") -> str:
        """Train anomaly detection model using Vertex AI cloud training"""
        try:
            from services.cloud.vertex_ai_service import vertex_ai_service
            
            task_id = f"anomaly_training_{uuid.uuid4().hex[:8]}"
            print(f"🚀 Starting Vertex AI training for project {project_id} with task ID: {task_id}")
            
            # Submit job to Vertex AI
            result = vertex_ai_service.submit_training_job(
                task_id=task_id,
                project_id=project_id,
                model_type='anomaly',
                algorithm=algorithm,
                training_config=training_config
            )
            
            if result['status'] == 'submitted':
                # Add to local tracking for monitoring
                self.training_processes[task_id] = {
                    'status': 'submitted',
                    'progress': 0,
                    'current_epoch': 0,
                    'total_epochs': training_config.get('epochs', 100),
                    'project_id': project_id,
                    'started_at': datetime.now().isoformat(),
                    'training_config': training_config,
                    'model_type': 'anomaly_detection',
                    'vertex_ai_job_id': result['vertex_ai_job_id'],
                    'cloud_training': True
                }
                self.training_logs[task_id] = [
                    f"🚀 Submitted training job to Vertex AI: {result['vertex_ai_job_id']}",
                    f"🎯 Machine type: {result.get('machine_type', 'CPU')}",
                    f"🔧 Accelerator: {result.get('accelerator_type', 'None')}"
                ]
                
                return task_id
            else:
                raise RuntimeError(f"Failed to submit to Vertex AI: {result['message']}")
                
        except Exception as e:
            print(f"❌ Cloud training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_detection_from_project_cloud(self, project_id: str, training_config: dict, algorithm: str = "yolo_v8") -> str:
        """Train object detection model using Vertex AI cloud training"""
        try:
            from services.cloud.vertex_ai_service import vertex_ai_service

            task_id = f"detection_training_{uuid.uuid4().hex[:8]}"
            print(f"🚀 Starting Vertex AI detection training for project {project_id} with task ID: {task_id}")

            # Submit job to Vertex AI
            result = vertex_ai_service.submit_training_job(
                task_id=task_id,
                project_id=project_id,
                model_type='object_detection',
                algorithm=algorithm,
                training_config=training_config
            )

            if result['status'] == 'submitted':
                # Add to local tracking for monitoring
                self.training_processes[task_id] = {
                    'status': 'submitted',
                    'progress': 0,
                    'current_epoch': 0,
                    'total_epochs': training_config.get('epochs', 100),
                    'project_id': project_id,
                    'started_at': datetime.now().isoformat(),
                    'training_config': training_config,
                    'model_type': 'object_detection',
                    'vertex_ai_job_id': result['vertex_ai_job_id'],
                    'cloud_training': True
                }

                # Save to database
                db_result = database_service.create_training_job(
                    task_id=task_id,
                    project_id=project_id,
                    model_type='object_detection',
                    algorithm=algorithm,
                    training_config={
                        **training_config,
                        'vertex_ai_job_id': result['vertex_ai_job_id']
                    },
                    total_epochs=training_config.get('epochs', 100)
                )
                print(f"✅ Detection training job saved to database: {db_result}")

                self.training_logs[task_id] = [
                    f"🚀 Submitted detection training job to Vertex AI: {result['vertex_ai_job_id']}",
                    f"🎯 Machine type: {result.get('machine_type', 'CPU')}",
                    f"🔧 Accelerator: {result.get('accelerator_type', 'None')}"
                ]

                return task_id
            else:
                raise Exception(f"Failed to submit Vertex AI job: {result}")

        except Exception as e:
            print(f"❌ Cloud detection training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_segmentation_from_project_cloud(self, project_id: str, training_config: dict, algorithm: str = "yolo_v8") -> str:
        """Train segmentation model using Vertex AI cloud training"""
        try:
            from services.cloud.vertex_ai_service import vertex_ai_service

            task_id = f"segmentation_training_{uuid.uuid4().hex[:8]}"
            print(f"🚀 Starting Vertex AI segmentation training for project {project_id} with task ID: {task_id}")

            # Submit job to Vertex AI
            result = vertex_ai_service.submit_training_job(
                task_id=task_id,
                project_id=project_id,
                model_type='segmentation',
                algorithm=algorithm,
                training_config=training_config
            )

            if result['status'] == 'submitted':
                # Add to local tracking for monitoring
                self.training_processes[task_id] = {
                    'status': 'submitted',
                    'progress': 0,
                    'current_epoch': 0,
                    'total_epochs': training_config.get('epochs', 100),
                    'project_id': project_id,
                    'started_at': datetime.now().isoformat(),
                    'training_config': training_config,
                    'model_type': 'segmentation',
                    'vertex_ai_job_id': result['vertex_ai_job_id'],
                    'cloud_training': True
                }

                # Save to database
                db_result = database_service.create_training_job(
                    task_id=task_id,
                    project_id=project_id,
                    model_type='segmentation',
                    algorithm=algorithm,
                    training_config={
                        **training_config,
                        'vertex_ai_job_id': result['vertex_ai_job_id']
                    },
                    total_epochs=training_config.get('epochs', 100)
                )
                print(f"✅ Segmentation training job saved to database: {db_result}")

                self.training_logs[task_id] = [
                    f"🚀 Submitted segmentation training job to Vertex AI: {result['vertex_ai_job_id']}",
                    f"🎯 Machine type: {result.get('machine_type', 'CPU')}",
                    f"🔧 Accelerator: {result.get('accelerator_type', 'None')}"
                ]

                return task_id
            else:
                raise Exception(f"Failed to submit Vertex AI job: {result}")

        except Exception as e:
            print(f"❌ Cloud segmentation training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_anomaly_from_project(self, project_id: str, training_config: dict, algorithm: str = "isolation_forest") -> str:
        """Train anomaly detection model from project dataset"""
        # Add algorithm to config and delegate to appropriate trainer
        training_config['algorithm'] = algorithm
        
        if algorithm in ['isolation_forest', 'one_class_svm', 'local_outlier_factor']:
            return self.sklearn_anomaly_trainer.train_from_project(project_id, training_config)
        elif algorithm == 'autoencoder':
            return self.pytorch_anomaly_trainer.train_from_project(project_id, training_config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
    def train_anomaly(self, dataset_path: str, algorithm: str = "isolation_forest", **kwargs) -> str:
        """Train anomaly detection model using the provided dataset"""
        # Prepare config from kwargs
        config = {
            'algorithm': algorithm,
            'epochs': kwargs.get('epochs', 100),
            'batch_size': kwargs.get('batch_size', 16),
            'learning_rate': kwargs.get('learning_rate', 0.001)
        }
        
        if algorithm in ['isolation_forest', 'one_class_svm', 'local_outlier_factor']:
            return self.sklearn_anomaly_trainer.train(dataset_path, config)
        elif algorithm == 'autoencoder':
            return self.pytorch_anomaly_trainer.train(dataset_path, config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def get_model_files(self, task_id: str) -> list:
        """Get list of model files for a completed training task"""
        try:
            results_dir = None
            
            # First try to get results_dir from memory (for active sessions)
            if task_id in self.training_processes:
                task_info = self.training_processes[task_id]
                results_dir = task_info.get('results_dir')
            
            # If not found in memory, try to find it in filesystem (for server restarts)
            if not results_dir:
                # Determine model type from task_id
                if task_id.startswith('anomaly_training_'):
                    results_dir = os.path.join(self.results_dir, 'anomaly', task_id)
                elif task_id.startswith('detection_training_'):
                    results_dir = os.path.join(self.results_dir, 'detection', task_id)  
                elif task_id.startswith('segmentation_training_'):
                    results_dir = os.path.join(self.results_dir, 'segmentation', task_id)
            
            if not results_dir or not os.path.exists(results_dir):
                return []
            
            model_files = []
            
            # Look for common model file patterns
            model_patterns = ['*.pt', '*.pth', '*.pkl', '*.h5', '*.onnx']
            
            for pattern in model_patterns:
                import glob
                pattern_path = os.path.join(results_dir, '**', pattern)
                files = glob.glob(pattern_path, recursive=True)
                
                for file_path in files:
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        model_files.append({
                            'filename': os.path.basename(file_path),
                            'filepath': file_path,
                            'file_size': file_size,
                            'created_at': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                        })
            
            return model_files
            
        except Exception as e:
            print(f"❌ Error getting model files for {task_id}: {e}")
            return []

    def delete_training_task(self, task_id: str) -> dict:
        """Delete a training task and its files"""
        try:
            if task_id not in self.training_processes:
                return {'status': 'error', 'message': 'Task not found'}
            
            task_info = self.training_processes[task_id]
            results_dir = task_info.get('results_dir')
            
            # Remove task from tracking
            del self.training_processes[task_id]
            if task_id in self.training_logs:
                del self.training_logs[task_id]
            
            # Remove files
            deleted_files = 0
            if results_dir and os.path.exists(results_dir):
                import shutil
                shutil.rmtree(results_dir)
                deleted_files = 1
            
            return {
                'status': 'success',
                'message': f'Training task {task_id} deleted successfully',
                'deleted_files': deleted_files
            }
            
        except Exception as e:
            print(f"❌ Error deleting training task {task_id}: {e}")
            return {'status': 'error', 'message': str(e)}

    # Old anomaly training helper methods removed - now handled by modular anomaly trainers

# Create global instance
yolo_service = YoloService()