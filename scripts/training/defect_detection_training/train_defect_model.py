#!/usr/bin/env python3
"""
Standalone YOLOv8 Training Script for Defect Detection
Created for training a defect detection model using the yolo_defect_detection dataset.
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml

def check_system():
    """Check system capabilities"""
    print("="*60)
    print("🔧 SYSTEM CHECK")
    print("="*60)

    # Check PyTorch and CUDA
    print(f"🐍 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"⚡ CUDA version: {torch.version.cuda}")
        print(f"🎯 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"🚀 GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️ No GPU available - training will use CPU (slower)")

    print("="*60)

def setup_data_yaml():
    """Create a local data.yaml file with an absolute path"""
    script_dir = Path(__file__).parent.absolute()
    project_dir = script_dir.parent
    dataset_dir = project_dir / "yolo_defect_detection"

    # Use an absolute path to ensure the dataset is always found
    data_config = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': ['dent', 'dirt', 'scratch']
    }

    yaml_path = script_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"📋 Created data.yaml at: {yaml_path}")
    print(f"📂 Absolute dataset path: {data_config['path']}")

    # Verify dataset exists
    train_images = dataset_dir / "images" / "train"
    val_images = dataset_dir / "images" / "val"
    train_labels = dataset_dir / "labels" / "train"
    val_labels = dataset_dir / "labels" / "val"

    print(f"✅ Train images: {len(list(train_images.glob('*.jpg')))} files")
    print(f"✅ Val images: {len(list(val_images.glob('*.jpg')))} files")
    print(f"✅ Train labels: {len(list(train_labels.glob('*.txt')))} files")
    print(f"✅ Val labels: {len(list(val_labels.glob('*.txt')))} files")

    return str(yaml_path)

def train_model():
    """Train YOLOv8 model for defect detection"""
    print("="*60)
    print("🚀 STARTING YOLO TRAINING")
    print("="*60)

    # Setup paths
    script_dir = Path(__file__).parent.absolute()
    data_yaml = setup_data_yaml()

    # Training configuration
    model_size = 's'  # yolov8s.pt - good balance of speed and accuracy
    epochs = 20
    batch_size = 8   # Small batch size due to limited dataset
    image_size = 640
    patience = 20    # Early stopping patience

    print(f"📦 Model: YOLOv8{model_size}")
    print(f"🎯 Epochs: {epochs}")
    print(f"📊 Batch size: {batch_size}")
    print(f"🖼️ Image size: {image_size}")
    print(f"⏰ Patience: {patience}")
    print()

    try:
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')  # Download pre-trained model

        # Start training
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            patience=patience,
            save=True,
            plots=True,
            verbose=True,
            project=str(script_dir),
            name='defect_detection_model',
            exist_ok=True,
            # Data augmentation settings
            hsv_h=0.015,      # Image HSV-Hue augmentation
            hsv_s=0.7,        # Image HSV-Saturation augmentation
            hsv_v=0.4,        # Image HSV-Value augmentation
            degrees=10.0,     # Image rotation (+/- deg)
            translate=0.1,    # Image translation (+/- fraction)
            scale=0.5,        # Image scale (+/- gain)
            shear=2.0,        # Image shear (+/- deg)
            perspective=0.0,  # Image perspective (+/- fraction)
            flipud=0.0,       # Image flip up-down (probability)
            fliplr=0.5,       # Image flip left-right (probability)
            mosaic=1.0,       # Image mosaic (probability)
            mixup=0.0,        # Image mixup (probability)
            copy_paste=0.0,   # Segment copy-paste (probability)
        )

        print("="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)

        # Model paths
        best_model = script_dir / "defect_detection_model" / "weights" / "best.pt"
        last_model = script_dir / "defect_detection_model" / "weights" / "last.pt"

        print(f"🏆 Best model: {best_model}")
        print(f"📄 Last model: {last_model}")
        print(f"📊 Results: {script_dir / 'defect_detection_model'}")

        # Validate the model
        if best_model.exists():
            print("\n🧪 VALIDATING TRAINED MODEL...")
            model = YOLO(str(best_model))
            val_results = model.val()

            print(f"📈 mAP50: {val_results.box.map50:.3f}")
            print(f"📈 mAP50-95: {val_results.box.map:.3f}")

            # Test inference on a sample image
            print("\n🔍 TESTING INFERENCE...")
            dataset_dir = script_dir.parent / "yolo_defect_detection"
            sample_images = list((dataset_dir / "images" / "val").glob("*.jpg"))

            if sample_images:
                sample_image = sample_images[0]
                results = model(str(sample_image))

                print(f"✅ Successfully ran inference on: {sample_image.name}")
                print(f"🎯 Detected {len(results[0].boxes)} objects")

                # Save prediction result
                pred_path = script_dir / "sample_prediction.jpg"
                results[0].save(str(pred_path))
                print(f"💾 Saved prediction to: {pred_path}")

            # Create both ONNX and PyTorch packages for production deployment
            print("\n📦 CREATING DEPLOYMENT PACKAGES...")

            # Create ONNX package
            print("\n🔧 Creating ONNX package...")
            onnx_package_path = create_onnx_package(model, script_dir, val_results)

            # Create PyTorch package
            print("\n🔧 Creating PyTorch package...")
            pytorch_package_path = create_pytorch_package(model, best_model, script_dir, val_results)

            print("\n" + "="*60)
            print("🎉 DEFECT DETECTION MODEL READY FOR USE!")
            print("="*60)
            print(f"📁 PyTorch Model: {best_model}")
            print(f"📦 ONNX Package: {onnx_package_path}")
            print(f"� PyTorch Package: {pytorch_package_path}")
            print("�🔧 Usage options:")
            print(f"   - Direct PyTorch: YOLO('{best_model}')")
            print(f"   - ONNX Production: Upload {onnx_package_path} to vision systems")
            print(f"   - PyTorch Production: Upload {pytorch_package_path} to Python systems")
            print("   - Classes: dent, dirt, scratch")

            return str(best_model)

        return None

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_onnx_package(model, script_dir, val_results):
    """Create a complete ONNX package for production deployment"""
    import json
    import zipfile
    from pathlib import Path

    try:
        # Create package directory
        package_dir = script_dir / "onnx_package"
        package_dir.mkdir(exist_ok=True)

        # 1. Export ONNX model
        print("🔄 Exporting ONNX model...")
        onnx_path = package_dir / "model.onnx"
        # Find the exported ONNX file and move it
        exported_onnx_source = Path(model.export(format='onnx', imgsz=640, opset=11))
        exported_onnx_source.rename(onnx_path)
        print(f"✅ ONNX model: {onnx_path}")

        # 2. Create classes.txt
        print("📝 Creating classes.txt...")
        classes_path = package_dir / "classes.txt"
        classes = ["dent", "dirt", "scratch"]
        with open(classes_path, 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        print(f"✅ Classes file: {classes_path}")

        # 3. Create config.json
        print("⚙️ Creating config.json...")
        config_path = package_dir / "config.json"

        config = {
            "model_info": {
                "name": "Defect Detection Model",
                "version": "1.0",
                "type": "object_detection",
                "framework": "YOLOv8",
                "classes": classes,
                "num_classes": len(classes)
            },
            "input": {
                "width": 640,
                "height": 640,
                "channels": 3,
                "format": "RGB",
                "normalize": True,
                "mean": [0.0, 0.0, 0.0],
                "std": [255.0, 255.0, 255.0]
            },
            "inference": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 100
            },
            "performance": {
                "map50": float(val_results.box.map50) if val_results else 0.0,
                "map50_95": float(val_results.box.map) if val_results else 0.0
            },
            "preprocessing": {
                "resize_mode": "letterbox",
                "pad_color": [114, 114, 114]
            },
            "postprocessing": {
                "nms_enabled": True,
                "class_agnostic_nms": False
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config file: {config_path}")

        # 4. Create class_colors.json
        print("🎨 Creating class_colors.json...")
        colors_path = package_dir / "class_colors.json"

        class_colors = {
            "dent": [255, 0, 0],      # Red
            "dirt": [255, 165, 0],    # Orange
            "scratch": [0, 255, 0]    # Green
        }

        with open(colors_path, 'w') as f:
            json.dump(class_colors, f, indent=2)
        print(f"✅ Colors file: {colors_path}")

        # 5. Create README.md
        print("📖 Creating README.md...")
        readme_path = package_dir / "README.md"

        readme_content = f"""# Defect Detection Model Package

## Model Information
- **Model Type**: YOLOv8 Object Detection
- **Classes**: {', '.join(classes)}
- **Input Size**: 640x640
- **Performance**: mAP50: {val_results.box.map50:.3f}, mAP50-95: {val_results.box.map:.3f}

## Files Included
- `model.onnx`: ONNX model file for inference
- `classes.txt`: Class names in training order
- `config.json`: Model configuration and thresholds
- `class_colors.json`: Visualization colors for each class
- `README.md`: This documentation

## Usage
1. Load the ONNX model using your inference engine
2. Preprocess images to 640x640 RGB format
3. Run inference with confidence threshold >= 0.25
4. Apply NMS with IoU threshold >= 0.45
5. Map class IDs to names using classes.txt

## Model Details
- **Training Images**: 12
- **Validation Images**: 3
- **Epochs**: 20
- **Data Augmentation**: Enabled
- **Optimizer**: AdamW

## Integration
This package is designed for production vision systems that support ONNX models.
Simply upload this package to your system for automatic model deployment.
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✅ README file: {readme_path}")

        # 6. Create ZIP package
        print("📦 Creating ZIP package...")
        zip_path = script_dir / "object_detection_onnx.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)

        print(f"✅ Package created: {zip_path}")
        print(f"📦 Package size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Verify package contents
        print("\n📋 Package contents:")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for filename in zipf.namelist():
                file_info = zipf.getinfo(filename)
                size_kb = file_info.file_size / 1024
                print(f"   📄 {filename} ({size_kb:.1f} KB)")

        return str(zip_path)

    except Exception as e:
        print(f"❌ ONNX package creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main training function"""
    print("🎯 YOLOv8 Defect Detection Training")
    print("🏭 Training custom model for dent, dirt, and scratch detection")
    print()

    # Check system
    check_system()

    # Train model
    model_path = train_model()

    if model_path:
        print(f"\n✅ Training completed! Model saved to: {model_path}")
        print("🚀 You can now use this model in any vision software that supports YOLOv8!")
    else:
        print("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()