#!/usr/bin/env python3
"""
Test script to verify YOLOv8 model output format
"""
import torch
from ultralytics import YOLO
import numpy as np

def test_model_output():
    """Test our trained model to see exact output format"""

    # Load our trained model
    model_path = "defect_detection_model/weights/best.pt"
    model = YOLO(model_path)

    print("🔍 Testing YOLO Model Output Format")
    print("="*50)

    # Create a dummy 640x640 input
    dummy_input = torch.randn(1, 3, 640, 640)

    # Get raw model output (not post-processed)
    model.model.eval()
    with torch.no_grad():
        raw_output = model.model(dummy_input)

    # Print raw output information
    if isinstance(raw_output, (tuple, list)):
        print(f"📊 Output is {type(raw_output).__name__} with {len(raw_output)} elements")
        for i, out in enumerate(raw_output):
            if hasattr(out, 'shape'):
                print(f"   Output {i}: shape {out.shape}, dtype {out.dtype}")
                if hasattr(out, 'numel') and out.numel() < 50:  # Small tensors
                    print(f"   Sample values: {out.flatten()[:10]}")
            else:
                print(f"   Output {i}: type {type(out)}, value: {out}")
    else:
        print(f"📊 Output shape: {raw_output.shape}")
        print(f"📊 Output dtype: {raw_output.dtype}")

    print()

    # Test with actual inference
    print("🧪 Testing with actual inference...")
    results = model.predict(dummy_input, verbose=False)

    if results and len(results) > 0:
        result = results[0]

        # Check if we have detections
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"✅ Found {len(result.boxes)} detections")

            # Show first detection details
            box = result.boxes[0]
            print(f"📦 First detection:")
            print(f"   Coordinates (xyxy): {box.xyxy[0]}")
            print(f"   Coordinates (xywh): {box.xywh[0]}")
            print(f"   Confidence: {box.conf[0]}")
            print(f"   Class: {box.cls[0]}")

        else:
            print("⚠️ No detections found")

        # Print model output format info
        print(f"\n📋 Model Info:")
        print(f"   Classes: {model.names}")
        print(f"   Task: {model.task}")

        # Check raw output format by looking at names
        print(f"\n🏗️ Model Architecture:")
        model_yaml = model.model.yaml
        if 'nc' in model_yaml:
            print(f"   Number of classes: {model_yaml['nc']}")

    print("\n" + "="*50)
    print("🎯 Expected Output Format for YOLOv8:")
    print("   [batch, 8400, 8] where 8 = [cx, cy, w, h, class0, class1, class2, objectness]")
    print("   OR")
    print("   [batch, 8400, 7] where 7 = [cx, cy, w, h, class0, class1, class2]")
    print("   Confidence = max(class probabilities)")

if __name__ == "__main__":
    test_model_output()