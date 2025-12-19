import os
import cv2
import base64
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from ultralytics import YOLO

from ..base.base_predictor import BasePredictor


class YOLOPredictor(BasePredictor):
    """YOLO-based predictor for detection and segmentation"""
    
    def __init__(self, models_dir: str = "ml/models"):
        super().__init__(models_dir)
        self.detection_model = None
        self.segmentation_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLO models with best available weights"""
        # Detection model priority
        detection_priority = ["yolov8l.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]
        
        for model_name in detection_priority:
            model_path = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_path):
                self.detection_model = YOLO(model_path)
                print(f"Using detection model: {model_name}")
                break
        else:
            # Fallback to downloading nano model
            self.detection_model = YOLO("yolov8n.pt")
            print("Using fallback nano detection model")
        
        # Segmentation model priority
        seg_priority = ["yolov8l-seg.pt", "yolov8m-seg.pt", "yolov8s-seg.pt", "yolov8n-seg.pt"]
        
        for model_name in seg_priority:
            model_path = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_path):
                self.segmentation_model = YOLO(model_path)
                print(f"Using segmentation model: {model_name}")
                break
        else:
            # Fallback to downloading nano segmentation model
            self.segmentation_model = YOLO("yolov8n-seg.pt")
            print("Using fallback nano segmentation model")
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm this predictor handles"""
        return "YOLO"
    
    def predict(self, image_path: str, model_path: str = None, task_type: str = "detection", **kwargs) -> Dict[str, Any]:
        """Run YOLO prediction on an image
        
        Args:
            image_path: Path to input image
            model_path: Optional path to specific model
            task_type: Type of task ('detection' or 'segmentation')
            **kwargs: Additional parameters
            
        Returns:
            dict: Prediction results with quality assessment
        """
        if not self.validate_image(image_path):
            raise ValueError(f"Invalid image: {image_path}")
        
        if task_type == "detection":
            return self.predict_detection(image_path, model_path)
        elif task_type == "segmentation":
            return self.predict_segmentation(image_path, model_path)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def predict_detection(self, image_path: str, model_path: str = None) -> Dict[str, Any]:
        """Run detection prediction with quality assessment"""
        try:
            # Load model
            if model_path and os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"Using custom model: {model_path}")
            else:
                model = self.detection_model
                print(f"Using default detection model")
            
            # Run prediction
            results = model(image_path)
            
            # Process results for quality assessment
            return self._process_detection_results(results[0], image_path)
            
        except Exception as e:
            print(f"Detection prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_segmentation(self, image_path: str, model_path: str = None) -> Dict[str, Any]:
        """Run segmentation prediction with quality assessment"""
        try:
            # Load segmentation model
            if model_path and os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"Using custom segmentation model: {model_path}")
            else:
                model = self.segmentation_model
                print(f"Using default segmentation model")
            
            # Run prediction
            results = model(image_path)
            
            # Process results for quality assessment
            return self._process_segmentation_results(results[0], image_path)
            
        except Exception as e:
            print(f"Segmentation prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(self, image_paths: List[str], model_path: str = None, task_type: str = "detection", **kwargs) -> Dict[str, Any]:
        """Predict on multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                if task_type == "detection":
                    result = self.predict_detection(image_path, model_path)
                else:
                    result = self.predict_segmentation(image_path, model_path)
                
                result["filename"] = os.path.basename(image_path)
                results.append(result)
                
            except Exception as e:
                results.append({
                    "filename": os.path.basename(image_path),
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "success",
            "total_images": len(image_paths),
            "results": results,
            "summary": self._generate_batch_summary(results)
        }
    
    def _process_detection_results(self, result, image_path: str) -> Dict[str, Any]:
        """Process YOLO detection results and assess quality"""
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
    
    def _process_segmentation_results(self, result, image_path: str) -> Dict[str, Any]:
        """Process YOLO segmentation results and assess quality"""
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
    
    def _generate_batch_summary(self, results: list) -> dict:
        """Generate summary statistics for batch prediction"""
        total = len([r for r in results if r.get('status') != 'error'])
        ok_count = sum(1 for r in results if r.get("quality", {}).get("status") == "OK")
        ng_count = total - ok_count
        
        return {
            "total_images": total,
            "ok_count": ok_count,
            "ng_count": ng_count,
            "ok_percentage": (ok_count / total * 100) if total > 0 else 0,
            "ng_percentage": (ng_count / total * 100) if total > 0 else 0
        }
    
    def assess_quality_standalone(self, image_path: str, model_path: str = None) -> dict:
        """Standalone quality assessment"""
        result = self.predict_detection(image_path, model_path)
        return {
            "status": "success",
            "filename": os.path.basename(image_path),
            "quality": result["quality"],
            "defects_found": result["total_defects"],
            "detections": result["detections"]
        }