import argparse
from ultralytics import YOLO
import os

def main(source_image):
    # Assuming the model is saved in the 'models' directory
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n-seg.pt')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'segmentation')
    os.makedirs(results_dir, exist_ok=True)

    model = YOLO(model_path)
    results = model(source_image)
    
    # Save the results
    saved_path = results[0].save(filename=os.path.join(results_dir, os.path.basename(source_image)))
    
    # Print the path for the backend to capture
    print(saved_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the source image")
    args = parser.parse_args()
    
    main(args.source)