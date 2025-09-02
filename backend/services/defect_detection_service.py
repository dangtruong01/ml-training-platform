import cv2
import numpy as np
import os
import uuid
import base64
from typing import List, Dict
from pathlib import Path

from .grounding_dino_service import grounding_dino_service
from .sam2_service import sam2_service

class DefectDetectionService:
    def __init__(self):
        self.grounding_dino = grounding_dino_service
        self.sam2 = sam2_service
        self.results_dir = os.path.abspath(os.path.join("ml", "results", "hybrid_annotation"))
        os.makedirs(self.results_dir, exist_ok=True)

    def predict_with_hybrid_model(self, image_path: str, prompts: List[str], confidence_threshold: float = 0.3) -> Dict:
        """
        Orchestrates GroundingDINO and SAM2 for high-precision defect detection.
        """
        print("🚀 Starting Hybrid Defect Detection...")

        # Step 1: Get coarse bounding boxes from GroundingDINO
        print("1. Running GroundingDINO for initial detection...")
        dino_results = self.grounding_dino.annotate_with_prompts(image_path, prompts, confidence_threshold)

        if dino_results["status"] != "success" or not dino_results["detections"]:
            print("⚠️ GroundingDINO found no initial detections. Skipping SAM.")
            return {
                "status": "success",
                "message": "No objects detected by GroundingDINO.",
                "detections": [],
                "annotated_image_path": None,
                "image_base64": None
            }

        # Step 2: Use SAM2 to get precise masks for each bounding box
        print(f"2. Running SAM2 for precise segmentation on {len(dino_results['detections'])} boxes...")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        all_masks = []
        for detection in dino_results["detections"]:
            bbox = detection["bbox"]  # [x1, y1, x2, y2]
            print(f"   - Segmenting box: {bbox} for class '{detection['class']}'")
            
            # Get precise mask from SAM2 using the bounding box
            mask, score, _ = self.sam2.segment_with_box(image, bbox)
            
            if mask is not None:
                # Convert mask to contours for JSON serialization
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_points = []
                for contour in contours:
                    if len(contour) > 5:  # Filter small contours
                        points = contour.reshape(-1, 2).tolist()
                        contour_points.append(points)
                
                all_masks.append({
                    "mask": mask,  # Keep for internal processing
                    "contours": contour_points,  # Serializable version
                    "mask_area": int(np.sum(mask)),
                    "class": detection["class"],
                    "confidence": float(score),  # Ensure it's a Python float
                    "dino_confidence": float(detection["confidence"]),
                    "bbox": detection["bbox"]
                })

        if not all_masks:
            print("⚠️ SAM2 did not produce any valid masks from the given boxes.")
            return {
                "status": "success",
                "message": "SAM2 could not segment the detected objects.",
                "detections": [],
                "annotated_image_path": None,
                "image_base64": None
            }

        # Step 3: Create a final annotated image with precise masks
        print("3. Creating final annotated image...")
        annotated_image = self._create_annotated_image_with_masks(image, all_masks)
        
        # Save and encode the final image
        output_filename = f"hybrid_{Path(image_path).stem}_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(self.results_dir, output_filename)
        cv2.imwrite(output_path, annotated_image)

        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Create serializable detections (without numpy arrays)
        serializable_detections = []
        for mask_data in all_masks:
            serializable_detections.append({
                "class": mask_data["class"],
                "confidence": mask_data["confidence"],
                "dino_confidence": mask_data["dino_confidence"],
                "bbox": mask_data["bbox"],
                "contours": mask_data["contours"],
                "mask_area": mask_data["mask_area"]
            })

        print("✅ Hybrid detection completed successfully.")
        return {
            "status": "success",
            "detections": serializable_detections,
            "total_detections": len(serializable_detections),
            "annotated_image_path": output_path,
            "image_base64": image_base64
        }

    def _create_annotated_image_with_masks(self, image: np.ndarray, masks_data: List[Dict]) -> np.ndarray:
        """Draws precise segmentation masks and labels on the image."""
        annotated_img = image.copy()
        overlay = image.copy()

        colors = {
            'scratch': (0, 0, 255),    # Red
            'dent': (255, 0, 0),       # Blue
            'dirt': (0, 255, 0),       # Green
            'corrosion': (0, 165, 255), # Orange
            'crack': (255, 0, 255),    # Magenta
            'stain': (255, 255, 0),    # Cyan
            'defect': (128, 128, 128)  # Gray (default)
        }

        for data in masks_data:
            mask = data["mask"]
            class_name = data["class"]
            color = colors.get(class_name.lower(), colors['defect'])

            # Draw filled mask on the overlay
            overlay[mask > 0] = color

            # Find contours to draw the outline
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_img, contours, -1, color, 2)

            # Add label
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{class_name} ({data['confidence']:.2f})"
                cv2.putText(annotated_img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Blend the overlay with the original image
        final_image = cv2.addWeighted(overlay, 0.4, annotated_img, 0.6, 0)
        return final_image

# Create a global instance
defect_detection_service = DefectDetectionService()