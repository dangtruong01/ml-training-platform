import argparse
from ultralytics import YOLO
import cv2
import os

def pre_annotate_detection(source_path: str):
    """
    Performs object detection on an image and saves the annotated image.
    """
    # Build absolute path to the model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    model_path = os.path.join(project_root, "yolov8n.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = YOLO(model_path)
    results = model(source_path)

    # Load the image with OpenCV
    img = cv2.imread(source_path)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            conf = float(box.conf)
            cls = int(box.cls)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image to an absolute path
    output_dir = os.path.join(project_root, "runs/detect/pre_annotate")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(source_path)
    output_path = os.path.join(output_dir, base_name)
    cv2.imwrite(output_path, img)
    
    # Print the absolute path to stdout
    print(os.path.abspath(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-annotate an image for object detection.")
    parser.add_argument("--source", required=True, help="Path to the source image.")
    args = parser.parse_args()
    pre_annotate_detection(args.source)