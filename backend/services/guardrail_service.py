import os
import cv2
import numpy as np
from .anomaly_service import anomaly_service
from .grounding_dino_service import grounding_dino_service

class GuardrailService:
    def __init__(self):
        self.anomaly_service = anomaly_service
        self.grounding_dino = grounding_dino_service
        self.results_dir = os.path.abspath(os.path.join("ml", "results", "guardrail_annotation"))
        os.makedirs(self.results_dir, exist_ok=True)

    def train_anomaly_model(self, good_images: list[str], project_name: str) -> str:
        """
        Train an anomaly detection model.
        """
        return self.anomaly_service.train_patchcore(good_images, project_name)

    def annotate_with_guardrail(self, image_path: str, model_path: str, prompts: list[str], confidence_threshold: float = 0.3) -> dict:
        """
        Annotate an image using the Guardrail pipeline.
        """
        # Step 1: Detect anomalies
        anomaly_result = self.anomaly_service.predict_anomaly(image_path, model_path)

        if anomaly_result["pred_score"] < 0.5: # No anomaly detected
            return {
                "status": "success",
                "message": "No anomalies detected.",
                "detections": [],
                "annotated_image": anomaly_result["annotated_image"]
            }

        # Step 2: Get bounding boxes from the anomaly map
        anomaly_map = (anomaly_result["anomaly_map"] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(anomaly_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "status": "success",
                "message": "Anomaly score was high, but no distinct anomaly regions were found.",
                "detections": [],
                "annotated_image": anomaly_result["annotated_image"]
            }

        # Step 3: Classify anomalies with GroundingDINO
        image = cv2.imread(image_path)
        all_detections = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Crop the anomalous region
            cropped_image = image[y:y+h, x:x+w]
            
            # Save the cropped image temporarily to pass to GroundingDINO
            temp_crop_path = os.path.join(self.results_dir, "temp_crop.jpg")
            cv2.imwrite(temp_crop_path, cropped_image)

            # Use GroundingDINO to classify the cropped region
            dino_results = self.grounding_dino.annotate_with_prompts(
                temp_crop_path, prompts, confidence_threshold
            )

            if dino_results["status"] == "success" and dino_results["detections"]:
                for detection in dino_results["detections"]:
                    # Adjust bbox to original image coordinates
                    detection["bbox"][0] += x
                    detection["bbox"][1] += y
                    detection["bbox"][2] += x
                    detection["bbox"][3] += y
                    all_detections.append(detection)

        # Step 4: Create final annotated image
        annotated_image = self._create_final_annotated_image(image, all_detections)

        return {
            "status": "success",
            "detections": all_detections,
            "annotated_image": annotated_image
        }

    def _create_final_annotated_image(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Create the final annotated image with bounding boxes and labels.
        """
        annotated_img = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return annotated_img

# Create a global instance
guardrail_service = GuardrailService()