import os
import shutil
import random
import yaml
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class RawAnnotationProcessor:
    """Processes raw annotation folders and converts them to YOLO format"""
    
    def __init__(self, output_dir: str = "ml/datasets"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_raw_folder(self, 
                          raw_folder_path: str, 
                          output_name: str,
                          train_split: float = 0.8,
                          val_split: float = 0.2) -> Dict[str, str]:
        """Process raw annotation folder into YOLO format
        
        Expected structure of raw_folder_path:
        - images/          (all image files)
        - labels/          (YOLO format .txt label files)  
        - classes.txt      (one class per line)
        
        Args:
            raw_folder_path: Path to raw annotation folder
            output_name: Name for the output dataset
            train_split: Fraction of data for training (0.0-1.0)
            val_split: Fraction of data for validation (0.0-1.0)
            
        Returns:
            dict: Information about the processed dataset
        """
        raw_path = Path(raw_folder_path)
        
        # Validate input structure
        self._validate_raw_structure(raw_path)
        
        # Read class names
        classes = self._read_classes(raw_path / "classes.txt")
        
        # Get all image and label files
        images_info = self._get_images_and_labels(raw_path / "images", raw_path / "labels")
        
        # Split data into train/val
        train_files, val_files = self._split_data(images_info, train_split, val_split)
        
        # Create output directory structure
        output_path = Path(self.output_dir) / output_name
        self._create_yolo_structure(output_path)
        
        # Copy files to appropriate directories
        self._copy_files_to_yolo_structure(train_files, output_path / "images" / "train", output_path / "labels" / "train")
        self._copy_files_to_yolo_structure(val_files, output_path / "images" / "val", output_path / "labels" / "val")
        
        # Generate data.yaml
        data_yaml_path = self._create_data_yaml(output_path, classes, len(train_files), len(val_files))
        
        return {
            "status": "success",
            "output_path": str(output_path),
            "data_yaml_path": str(data_yaml_path),
            "classes": classes,
            "train_samples": len(train_files),
            "val_samples": len(val_files),
            "total_samples": len(images_info),
            "train_split": train_split,
            "val_split": val_split
        }
    
    def _validate_raw_structure(self, raw_path: Path) -> None:
        """Validate that raw folder has required structure"""
        if not raw_path.exists():
            raise ValueError(f"Raw folder does not exist: {raw_path}")
        
        images_dir = raw_path / "images"
        labels_dir = raw_path / "labels" 
        classes_file = raw_path / "classes.txt"
        
        if not images_dir.exists():
            raise ValueError(f"Missing images/ directory in {raw_path}")
        
        if not labels_dir.exists():
            raise ValueError(f"Missing labels/ directory in {raw_path}")
            
        if not classes_file.exists():
            raise ValueError(f"Missing classes.txt file in {raw_path}")
        
        # Check if directories contain files
        image_files = list(images_dir.glob("*"))
        label_files = list(labels_dir.glob("*.txt"))
        
        if not image_files:
            raise ValueError("No image files found in images/ directory")
            
        if not label_files:
            raise ValueError("No label files found in labels/ directory")
    
    def _read_classes(self, classes_file: Path) -> List[str]:
        """Read class names from classes.txt"""
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        if not classes:
            raise ValueError("No classes found in classes.txt")
        
        return classes
    
    def _get_images_and_labels(self, images_dir: Path, labels_dir: Path) -> List[Dict[str, str]]:
        """Get matching image and label file pairs"""
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        matched_files = []
        unmatched_images = []
        
        for image_file in image_files:
            # Look for corresponding label file
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            if label_file.exists():
                matched_files.append({
                    'image': str(image_file),
                    'label': str(label_file),
                    'basename': image_file.stem
                })
            else:
                unmatched_images.append(str(image_file))
        
        if not matched_files:
            raise ValueError("No matching image-label pairs found")
        
        if unmatched_images:
            print(f"Warning: {len(unmatched_images)} images without corresponding labels will be skipped")
        
        return matched_files
    
    def _split_data(self, images_info: List[Dict[str, str]], train_split: float, val_split: float) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train and validation sets"""
        if train_split + val_split > 1.0:
            raise ValueError("train_split + val_split cannot exceed 1.0")
        
        # Shuffle data for random split
        shuffled_data = images_info.copy()
        random.shuffle(shuffled_data)
        
        total_samples = len(shuffled_data)
        train_count = int(total_samples * train_split)
        val_count = int(total_samples * val_split)
        
        # Ensure we don't exceed total samples
        if train_count + val_count > total_samples:
            val_count = total_samples - train_count
        
        train_files = shuffled_data[:train_count]
        val_files = shuffled_data[train_count:train_count + val_count]
        
        return train_files, val_files
    
    def _create_yolo_structure(self, output_path: Path) -> None:
        """Create YOLO dataset directory structure"""
        dirs_to_create = [
            output_path / "images" / "train",
            output_path / "images" / "val",
            output_path / "labels" / "train", 
            output_path / "labels" / "val"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _copy_files_to_yolo_structure(self, files_info: List[Dict], images_dest: Path, labels_dest: Path) -> None:
        """Copy image and label files to YOLO structure"""
        for file_info in files_info:
            # Copy image file
            src_image = Path(file_info['image'])
            dst_image = images_dest / src_image.name
            shutil.copy2(src_image, dst_image)
            
            # Copy label file  
            src_label = Path(file_info['label'])
            dst_label = labels_dest / src_label.name
            shutil.copy2(src_label, dst_label)
    
    def _create_data_yaml(self, output_path: Path, classes: List[str], train_count: int, val_count: int) -> Path:
        """Create data.yaml file for YOLO training"""
        data_yaml_content = {
            'path': str(output_path.absolute()),  # Dataset root directory
            'train': 'images/train',              # Train images relative to 'path'
            'val': 'images/val',                  # Validation images relative to 'path'
            'test': '',                           # Test images (optional)
            'nc': len(classes),                   # Number of classes
            'names': classes,                     # Class names
            'info': {
                'description': f'Converted raw annotation dataset',
                'train_samples': train_count,
                'val_samples': val_count,
                'total_samples': train_count + val_count,
                'created_by': 'RawAnnotationProcessor'
            }
        }
        
        data_yaml_path = output_path / 'data.yaml'
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        return data_yaml_path
    
    def validate_yolo_labels(self, labels_dir: Path, num_classes: int) -> Dict[str, any]:
        """Validate YOLO format label files"""
        validation_results = {
            'valid_files': 0,
            'invalid_files': 0,
            'errors': [],
            'class_distribution': [0] * num_classes
        }
        
        for label_file in labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                valid_file = True
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:  # class_id x_center y_center width height
                        validation_results['errors'].append(
                            f"{label_file.name}:L{line_num}: Expected 5 values, got {len(parts)}"
                        )
                        valid_file = False
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        bbox_coords = [float(x) for x in parts[1:5]]
                        
                        # Validate class_id
                        if class_id < 0 or class_id >= num_classes:
                            validation_results['errors'].append(
                                f"{label_file.name}:L{line_num}: Class ID {class_id} out of range [0, {num_classes-1}]"
                            )
                            valid_file = False
                        else:
                            validation_results['class_distribution'][class_id] += 1
                        
                        # Validate bbox coordinates (should be normalized 0-1)
                        for i, coord in enumerate(bbox_coords):
                            if coord < 0 or coord > 1:
                                coord_names = ['x_center', 'y_center', 'width', 'height']
                                validation_results['errors'].append(
                                    f"{label_file.name}:L{line_num}: {coord_names[i]} = {coord} not in range [0, 1]"
                                )
                                valid_file = False
                    
                    except ValueError as e:
                        validation_results['errors'].append(
                            f"{label_file.name}:L{line_num}: Invalid number format - {e}"
                        )
                        valid_file = False
                
                if valid_file:
                    validation_results['valid_files'] += 1
                else:
                    validation_results['invalid_files'] += 1
                    
            except Exception as e:
                validation_results['errors'].append(f"{label_file.name}: File read error - {e}")
                validation_results['invalid_files'] += 1
        
        return validation_results
    
    def get_dataset_statistics(self, dataset_path: str) -> Dict[str, any]:
        """Get statistics about a processed YOLO dataset"""
        dataset_path = Path(dataset_path)
        
        if not (dataset_path / 'data.yaml').exists():
            raise ValueError(f"Not a valid YOLO dataset: missing data.yaml in {dataset_path}")
        
        # Read data.yaml
        with open(dataset_path / 'data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        
        stats = {
            'dataset_path': str(dataset_path),
            'classes': data_config.get('names', []),
            'num_classes': data_config.get('nc', 0),
            'train_images': 0,
            'val_images': 0,
            'train_labels': 0,
            'val_labels': 0
        }
        
        # Count files
        train_images_dir = dataset_path / 'images' / 'train'
        val_images_dir = dataset_path / 'images' / 'val'
        train_labels_dir = dataset_path / 'labels' / 'train'
        val_labels_dir = dataset_path / 'labels' / 'val'
        
        if train_images_dir.exists():
            stats['train_images'] = len([f for f in train_images_dir.iterdir() if f.is_file()])
        
        if val_images_dir.exists():
            stats['val_images'] = len([f for f in val_images_dir.iterdir() if f.is_file()])
        
        if train_labels_dir.exists():
            stats['train_labels'] = len(list(train_labels_dir.glob("*.txt")))
        
        if val_labels_dir.exists():
            stats['val_labels'] = len(list(val_labels_dir.glob("*.txt")))
        
        # Validate labels
        if train_labels_dir.exists():
            train_validation = self.validate_yolo_labels(train_labels_dir, stats['num_classes'])
            stats['train_label_validation'] = train_validation
        
        if val_labels_dir.exists():
            val_validation = self.validate_yolo_labels(val_labels_dir, stats['num_classes'])
            stats['val_label_validation'] = val_validation
        
        return stats