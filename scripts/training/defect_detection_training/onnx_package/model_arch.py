"""
YOLOv8 Defect Detection Model Architecture and Inference
"""
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

class YOLOv8DefectDetector:
    """
    YOLOv8 Defect Detection Model Wrapper

    Provides easy-to-use interface for defect detection inference
    with post-processing and visualization capabilities.
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize the defect detector

        Args:
            model_path: Path to the .pt model file
            config_path: Optional path to config.json file
        """
        self.model = YOLO(model_path)
        self.classes = ["dent", "dirt", "scratch"]

        # Load configuration if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.classes = self.config['model_info']['classes']
                self.conf_threshold = self.config['inference']['confidence_threshold']
                self.iou_threshold = self.config['inference']['iou_threshold']
        else:
            self.config = None
            self.conf_threshold = 0.25
            self.iou_threshold = 0.45

    def predict(self, image_path: str, save_results: bool = False,
                output_path: Optional[str] = None) -> Dict:
        """
        Run defect detection on an image

        Args:
            image_path: Path to input image
            save_results: Whether to save annotated image
            output_path: Output path for annotated image

        Returns:
            Dictionary containing detection results
        """
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save_results,
            project=output_path if output_path else None
        )

        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class_id': int(box.cls[0]),
                        'class_name': self.classes[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        'bbox_normalized': box.xywhn[0].tolist()  # [x_center, y_center, width, height] normalized
                    }
                    detections.append(detection)

        return {
            'image_path': image_path,
            'detections': detections,
            'num_detections': len(detections),
            'model_info': {
                'classes': self.classes,
                'confidence_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold
            }
        }

    def predict_batch(self, image_paths: List[str], save_results: bool = False) -> List[Dict]:
        """
        Run batch prediction on multiple images

        Args:
            image_paths: List of image paths
            save_results: Whether to save annotated images

        Returns:
            List of detection results for each image
        """
        all_results = []
        for image_path in image_paths:
            result = self.predict(image_path, save_results=save_results)
            all_results.append(result)
        return all_results

    def visualize_detections(self, image_path: str, output_path: str,
                           show_confidence: bool = True) -> str:
        """
        Create visualization of detections with custom styling

        Args:
            image_path: Input image path
            output_path: Output image path
            show_confidence: Whether to show confidence scores

        Returns:
            Path to saved visualization
        """
        image = cv2.imread(image_path)
        results = self.predict(image_path)

        # Color mapping for classes
        colors = {
            'dent': (0, 0, 255),      # Red
            'dirt': (0, 165, 255),    # Orange
            'scratch': (0, 255, 0)    # Green
        }

        for detection in results['detections']:
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            color = colors.get(class_name, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}"
            if show_confidence:
                label += f" {confidence:.2f}"

            # Calculate label size and position
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Save visualization
        cv2.imwrite(output_path, image)
        return output_path

    def get_model_info(self) -> Dict:
        """Get model information and statistics"""
        return {
            'model_type': 'YOLOv8 Object Detection',
            'classes': self.classes,
            'num_classes': len(self.classes),
            'input_size': '640x640',
            'confidence_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'auto'
        }

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLOv8DefectDetector("model.pt", "config.json")

    # Single image prediction
    result = detector.predict("test_image.jpg", save_results=True)
    print(f"Found {result['num_detections']} defects")

    # Print detections
    for detection in result['detections']:
        print(f"- {detection['class_name']}: {detection['confidence']:.2f}")

    # Custom visualization
    detector.visualize_detections("test_image.jpg", "output_with_detections.jpg")

    # Model info
    info = detector.get_model_info()
    print(f"Model info: {info}")
