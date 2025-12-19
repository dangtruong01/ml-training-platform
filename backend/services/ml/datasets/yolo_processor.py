import os
import yaml
import zipfile
from typing import Optional, Dict, Any
from fastapi import UploadFile
from .base_dataset_processor import BaseDatasetProcessor


class YOLODatasetProcessor(BaseDatasetProcessor):
    """YOLO-specific dataset processing"""
    
    def __init__(self, datasets_dir: str = "ml/datasets"):
        super().__init__(datasets_dir)
    
    async def process_uploaded_dataset(self, file: UploadFile, task_type: str) -> Optional[str]:
        """Process an uploaded YOLO dataset file
        
        Args:
            file: Uploaded ZIP file containing YOLO dataset
            task_type: Type of task (detection, segmentation)
            
        Returns:
            Path to data.yaml file if successful, None otherwise
        """
        try:
            print(f"🔄 Processing uploaded dataset: {file.filename}")
            
            # Save uploaded file
            zip_path = self.save_uploaded_file(file)
            print(f"📁 Saved uploaded file to: {zip_path}")
            
            # Extract ZIP file
            extract_dir = self.extract_zip(zip_path)
            print(f"📂 Extracted to: {extract_dir}")
            
            # Find data.yaml file
            data_yaml_path = self.find_data_yaml(extract_dir)
            if not data_yaml_path:
                print("❌ No data.yaml file found in dataset")
                self.clean_temp_files(zip_path)
                self.clean_temp_files(extract_dir)
                return None
            
            print(f"✅ Found data.yaml at: {data_yaml_path}")
            
            # Validate dataset structure
            if not self.validate_dataset_structure(data_yaml_path):
                print("❌ Dataset structure validation failed")
                self.clean_temp_files(zip_path)
                self.clean_temp_files(extract_dir)
                return None
            
            print("✅ Dataset structure validated successfully")
            
            # Clean up ZIP file but keep extracted data
            self.clean_temp_files(zip_path)
            
            return data_yaml_path
            
        except Exception as e:
            print(f"❌ Error processing dataset: {e}")
            return None
    
    def find_data_yaml(self, directory: str) -> Optional[str]:
        """Recursively find data.yaml file in directory
        
        Args:
            directory: Directory to search in
            
        Returns:
            Path to data.yaml file if found, None otherwise
        """
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower() in ['data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml']:
                    return os.path.join(root, file)
        return None
    
    def validate_dataset_structure(self, data_yaml_path: str) -> bool:
        """Validate that the dataset has proper YOLO structure
        
        Args:
            data_yaml_path: Path to data.yaml file
            
        Returns:
            True if dataset structure is valid, False otherwise
        """
        try:
            # Read data.yaml
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)

            print(f"Dataset config: {data_config}")

            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data_config:
                    print(f"Missing required field: {field}")
                    return False

            # Get the directory containing data.yaml
            dataset_root = os.path.dirname(data_yaml_path)

            # Check if train and val directories exist
            train_path = os.path.join(dataset_root, data_config['train'])
            val_path = os.path.join(dataset_root, data_config['val'])

            if not os.path.exists(train_path):
                print(f"Train directory not found: {train_path}")
                return False

            if not os.path.exists(val_path):
                print(f"Val directory not found: {val_path}")
                return False

            # Check if there are corresponding labels directories
            train_labels_path = train_path.replace('images', 'labels')
            val_labels_path = val_path.replace('images', 'labels')

            if not os.path.exists(train_labels_path):
                print(f"Train labels directory not found: {train_labels_path}")
                return False

            if not os.path.exists(val_labels_path):
                print(f"Val labels directory not found: {val_labels_path}")
                return False

            # Count files
            train_images = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            train_labels = len([f for f in os.listdir(train_labels_path) if f.endswith('.txt')])
            val_images = len([f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            val_labels = len([f for f in os.listdir(val_labels_path) if f.endswith('.txt')])

            print(f"Dataset statistics:")
            print(f"  Train: {train_images} images, {train_labels} labels")
            print(f"  Val: {val_images} images, {val_labels} labels")
            print(f"  Classes: {data_config['nc']} - {data_config['names']}")

            if train_images == 0:
                print("No training images found")
                return False

            if val_images == 0:
                print("No validation images found")
                return False

            return True

        except Exception as e:
            print(f"Error validating dataset: {e}")
            return False
    
    def create_data_yaml(self, dataset_path: str, class_names: list, train_dir: str = "images/train", val_dir: str = "images/val") -> str:
        """Create a data.yaml file for YOLO training
        
        Args:
            dataset_path: Path to dataset root directory
            class_names: List of class names
            train_dir: Relative path to training images
            val_dir: Relative path to validation images
            
        Returns:
            Path to created data.yaml file
        """
        data_yaml_content = {
            'path': dataset_path,
            'train': train_dir,
            'val': val_dir,
            'nc': len(class_names),
            'names': class_names
        }
        
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        return data_yaml_path
    
    def get_dataset_info(self, data_yaml_path: str) -> Dict[str, Any]:
        """Get detailed information about a YOLO dataset
        
        Args:
            data_yaml_path: Path to data.yaml file
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            dataset_root = os.path.dirname(data_yaml_path)
            train_path = os.path.join(dataset_root, data_config['train'])
            val_path = os.path.join(dataset_root, data_config['val'])
            
            # Count files
            train_images = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists(train_path) else 0
            val_images = len([f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists(val_path) else 0
            
            return {
                'config': data_config,
                'train_images': train_images,
                'val_images': val_images,
                'total_images': train_images + val_images,
                'num_classes': data_config.get('nc', 0),
                'class_names': data_config.get('names', []),
                'dataset_root': dataset_root,
                'train_path': train_path,
                'val_path': val_path
            }
        
        except Exception as e:
            return {
                'error': f"Failed to read dataset info: {e}",
                'train_images': 0,
                'val_images': 0,
                'total_images': 0,
                'num_classes': 0,
                'class_names': []
            }