# Defect Detection Model Training

This folder contains a standalone YOLOv8 training script to create a custom defect detection model using your `yolo_defect_detection` dataset.

## Dataset Overview
- **Classes**: dent, dirt, scratch (3 classes)
- **Training Images**: 14 images with labels
- **Validation Images**: 5 images with labels
- **Format**: YOLO format (normalized bounding boxes)

## Quick Start

### 1. Install Dependencies
```bash
cd defect_detection_training
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train_defect_model.py
```

## What the Script Does

1. **System Check**: Verifies PyTorch, CUDA, and GPU availability
2. **Dataset Setup**: Creates proper data.yaml configuration
3. **Model Training**: Trains YOLOv8s model for 100 epochs with data augmentation
4. **Validation**: Tests the trained model and saves results
5. **Sample Inference**: Runs prediction on a validation image

## Output

After training completes, you'll find:

- **`defect_detection_model/weights/best.pt`** - Best performing model (use this!)
- **`defect_detection_model/weights/last.pt`** - Final epoch model
- **`defect_detection_model/`** - Training logs, plots, and metrics
- **`sample_prediction.jpg`** - Example prediction result

## Using the Trained Model

### In Python:
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('defect_detection_model/weights/best.pt')

# Run inference
results = model('your_image.jpg')

# Get detections
for r in results:
    boxes = r.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = box.conf[0]
        coordinates = box.xyxy[0].tolist()

        # Classes: 0=dent, 1=dirt, 2=scratch
        class_names = ['dent', 'dirt', 'scratch']
        print(f"Detected {class_names[class_id]} with confidence {confidence:.2f}")
```

### In Vision Software:
Most vision software platforms support YOLO models:
- Load the `.pt` file directly
- Set input size to 640x640
- Classes will be mapped as: 0=dent, 1=dirt, 2=scratch

## Training Configuration

- **Model**: YOLOv8s (small) - Good balance of speed and accuracy
- **Epochs**: 100 with early stopping (patience=20)
- **Batch Size**: 8 (suitable for small dataset)
- **Image Size**: 640x640
- **Data Augmentation**: Enabled (rotation, scaling, flipping, etc.)

## Performance Tips

1. **More Data**: Consider adding more training images for better accuracy
2. **Model Size**: Try YOLOv8m or YOLOv8l for higher accuracy (slower inference)
3. **Hyperparameters**: Adjust learning rate, batch size based on your hardware
4. **Transfer Learning**: The script uses pre-trained weights for faster convergence

## Troubleshooting

- **CUDA Issues**: Install PyTorch with CUDA support for GPU training
- **Memory Errors**: Reduce batch size or use CPU training
- **Dataset Errors**: Ensure all images have corresponding label files