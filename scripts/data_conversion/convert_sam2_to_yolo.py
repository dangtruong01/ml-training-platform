import os
import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
import argparse

def extract_bboxes_from_sam2_image(image_path):
    """
    Extract bounding boxes from SAM2 annotated images by detecting green rectangles
    Returns list of normalized bounding boxes
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return []
    
    h, w = image.shape[:2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color (SAM2 uses bright green for main objects)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    # Create mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours of green rectangles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        # Get bounding rectangle
        x, y, box_w, box_h = cv2.boundingRect(contour)
        
        # Filter out very small boxes (likely noise)
        if box_w < 20 or box_h < 20:
            continue
            
        # Convert to YOLO format (normalized center coordinates)
        center_x = (x + box_w / 2) / w
        center_y = (y + box_h / 2) / h
        norm_width = box_w / w
        norm_height = box_h / h
        
        # Ensure values are within [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        norm_width = max(0, min(1, norm_width))
        norm_height = max(0, min(1, norm_height))
        
        bboxes.append((center_x, center_y, norm_width, norm_height))
    
    return bboxes

def convert_sam2_to_yolo(source_dir, output_dir, class_names=None, train_split=0.8):
    """
    Convert SAM2 annotated detection batch to YOLO training format
    
    Args:
        source_dir: Path to annotated_sam2-detection_batch folder
        output_dir: Path to output YOLO dataset
        class_names: List of class names (default: ['object'])
        train_split: Ratio of images for training (rest for validation)
    """
    
    if class_names is None:
        class_names = ['object']  # Default single class
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create YOLO dataset structure
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Find all image files in source directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(source_path.glob(f'*{ext}'))
        image_files.extend(source_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    if len(image_files) == 0:
        print("No images found! Check the source directory.")
        return False
    
    # Split into train/val
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)} images, Val: {len(val_files)} images")
    
    # Process each split
    for split_name, files in [('train', train_files), ('val', val_files)]:
        print(f"\nProcessing {split_name} split...")
        
        for i, image_file in enumerate(files):
            print(f"Processing {i+1}/{len(files)}: {image_file.name}")
            
            # Copy image to appropriate directory
            dest_image = output_path / 'images' / split_name / image_file.name
            shutil.copy2(image_file, dest_image)
            
            # Extract bounding boxes from annotated image
            bboxes = extract_bboxes_from_sam2_image(str(image_file))
            
            # Create label file
            label_file = output_path / 'labels' / split_name / f"{image_file.stem}.txt"
            
            with open(label_file, 'w') as f:
                if bboxes:
                    for bbox in bboxes:
                        # Use class_id 0 for single class, or modify as needed
                        class_id = 0
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                else:
                    # Create empty label file if no objects detected
                    pass
            
            print(f"  Found {len(bboxes)} bounding boxes")
    
    # Create data.yaml file
    data_yaml = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✅ Dataset conversion complete!")
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Classes: {class_names}")
    print(f"🏷️  Total images: {len(image_files)}")
    print(f"🚂 Training images: {len(train_files)}")
    print(f"✅ Validation images: {len(val_files)}")
    print(f"\n📋 Next steps:")
    print(f"1. Review the generated labels in {output_path}/labels/")
    print(f"2. Zip the entire {output_path} folder")
    print(f"3. Upload through your training interface")
    
    return True

def verify_yolo_dataset(dataset_dir):
    """
    Verify the YOLO dataset structure and content
    """
    dataset_path = Path(dataset_dir)
    
    # Check required files and directories
    required_paths = [
        dataset_path / 'data.yaml',
        dataset_path / 'images' / 'train',
        dataset_path / 'images' / 'val',
        dataset_path / 'labels' / 'train',
        dataset_path / 'labels' / 'val'
    ]
    
    print("🔍 Verifying dataset structure...")
    for path in required_paths:
        if path.exists():
            print(f"✅ {path.relative_to(dataset_path)}")
        else:
            print(f"❌ {path.relative_to(dataset_path)} - MISSING!")
            return False
    
    # Check data.yaml content
    try:
        with open(dataset_path / 'data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        print(f"✅ data.yaml: {data_config['nc']} classes - {data_config['names']}")
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False
    
    # Count files
    train_images = len(list((dataset_path / 'images' / 'train').glob('*')))
    val_images = len(list((dataset_path / 'images' / 'val').glob('*')))
    train_labels = len(list((dataset_path / 'labels' / 'train').glob('*.txt')))
    val_labels = len(list((dataset_path / 'labels' / 'val').glob('*.txt')))
    
    print(f"📊 Train: {train_images} images, {train_labels} labels")
    print(f"📊 Val: {val_images} images, {val_labels} labels")
    
    if train_images != train_labels:
        print(f"⚠️  Warning: Mismatch in train images/labels count")
    if val_images != val_labels:
        print(f"⚠️  Warning: Mismatch in val images/labels count")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert SAM2 detection batch to YOLO training format')
    parser.add_argument('source_dir', help='Path to annotated_sam2-detection_batch folder')
    parser.add_argument('output_dir', help='Path to output YOLO dataset folder')
    parser.add_argument('--classes', nargs='+', default=['scratch', 'dent'], 
                       help='Class names (default: scratch dent)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training split ratio (default: 0.8)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the converted dataset')
    
    args = parser.parse_args()
    
    # Convert dataset
    success = convert_sam2_to_yolo(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        class_names=args.classes,
        train_split=args.train_split
    )
    
    if success and args.verify:
        print("\n" + "="*50)
        verify_yolo_dataset(args.output_dir)

if __name__ == "__main__":
    main()