#!/usr/bin/env python3
"""
CVAT to YOLO Dataset Converter

This script converts a CVAT-exported dataset to proper YOLO training format
with train/val split and data.yaml configuration.
"""

import os
import sys
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Dict, Tuple


def validate_cvat_structure(cvat_path: Path) -> bool:
    """Validate CVAT dataset structure"""
    required_items = ['images', 'labels', 'classes.txt']

    for item in required_items:
        if not (cvat_path / item).exists():
            print(f"❌ Missing required item: {item}")
            return False

    return True


def read_classes(classes_file: Path) -> List[str]:
    """Read class names from classes.txt"""
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Get matching image and label file pairs"""
    pairs = []

    for img_file in images_dir.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Find corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                print(f"⚠️  No label found for image: {img_file.name}")

    return pairs


def split_data(pairs: List[Tuple[Path, Path]], train_ratio: float = 0.8) -> Tuple[List, List]:
    """Split data into train and validation sets"""
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)

    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    return train_pairs, val_pairs


def copy_files(pairs: List[Tuple[Path, Path]], output_images: Path, output_labels: Path):
    """Copy image and label files to output directory"""
    for img_file, label_file in pairs:
        # Copy image
        shutil.copy2(img_file, output_images)
        # Copy label
        shutil.copy2(label_file, output_labels)


def create_data_yaml(output_dir: Path, classes: List[str]):
    """Create data.yaml configuration file"""
    data_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }

    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)


def convert_cvat_to_yolo(cvat_path: str, output_name: str, train_ratio: float = 0.8) -> Dict[str, any]:
    """
    Convert CVAT dataset to YOLO format

    Args:
        cvat_path: Path to CVAT dataset folder
        output_name: Name for output dataset
        train_ratio: Ratio for train/val split (default: 0.8)

    Returns:
        Dictionary with conversion results
    """
    cvat_path = Path(cvat_path)
    output_dir = Path(f"yolo_{output_name}")

    print(f"🔄 Converting CVAT dataset: {cvat_path}")
    print(f"📁 Output directory: {output_dir}")

    # Validate input
    if not validate_cvat_structure(cvat_path):
        raise ValueError("Invalid CVAT dataset structure")

    # Read classes
    classes = read_classes(cvat_path / 'classes.txt')
    print(f"📋 Found {len(classes)} classes: {classes}")

    # Get image-label pairs
    pairs = get_image_label_pairs(cvat_path / 'images', cvat_path / 'labels')
    print(f"🖼️  Found {len(pairs)} image-label pairs")

    if len(pairs) == 0:
        raise ValueError("No valid image-label pairs found")

    # Split data
    train_pairs, val_pairs = split_data(pairs, train_ratio)
    print(f"📊 Split: {len(train_pairs)} train, {len(val_pairs)} val")

    # Create output directory structure
    output_dir.mkdir(exist_ok=True)

    train_images = output_dir / 'images' / 'train'
    train_labels = output_dir / 'labels' / 'train'
    val_images = output_dir / 'images' / 'val'
    val_labels = output_dir / 'labels' / 'val'

    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy files
    print("📋 Copying training files...")
    copy_files(train_pairs, train_images, train_labels)

    print("📋 Copying validation files...")
    copy_files(val_pairs, val_images, val_labels)

    # Create data.yaml
    create_data_yaml(output_dir, classes)
    print("✅ Created data.yaml configuration")

    # Create summary
    result = {
        'output_dir': str(output_dir.absolute()),
        'total_images': len(pairs),
        'train_images': len(train_pairs),
        'val_images': len(val_pairs),
        'classes': classes,
        'num_classes': len(classes)
    }

    print("✅ Conversion completed successfully!")
    print(f"📊 Summary:")
    print(f"   - Total images: {result['total_images']}")
    print(f"   - Training: {result['train_images']}")
    print(f"   - Validation: {result['val_images']}")
    print(f"   - Classes: {result['num_classes']} ({', '.join(classes)})")
    print(f"   - Output: {result['output_dir']}")

    return result


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Convert CVAT dataset to YOLO format')
    parser.add_argument('cvat_path', help='Path to CVAT dataset folder')
    parser.add_argument('output_name', help='Name for output YOLO dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')

    args = parser.parse_args()

    try:
        result = convert_cvat_to_yolo(args.cvat_path, args.output_name, args.train_ratio)

        print("\n🎉 Ready for training!")
        print(f"💡 You can now zip the dataset and upload it:")
        print(f"   cd {result['output_dir']}")
        print(f"   zip -r yolo_{args.output_name}.zip .")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()