import os
import zipfile
import shutil
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from fastapi import UploadFile


class BaseDatasetProcessor(ABC):
    """Abstract base class for dataset processing"""
    
    def __init__(self, datasets_dir: str = "ml/datasets"):
        self.datasets_dir = os.path.abspath(datasets_dir)
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    @abstractmethod
    def process_uploaded_dataset(self, file: UploadFile, task_type: str) -> Optional[str]:
        """Process an uploaded dataset file
        
        Args:
            file: Uploaded file
            task_type: Type of ML task (detection, segmentation, etc.)
            
        Returns:
            Path to processed dataset configuration file or None if failed
        """
        pass
    
    @abstractmethod
    def validate_dataset_structure(self, dataset_path: str) -> bool:
        """Validate the structure of a dataset
        
        Args:
            dataset_path: Path to dataset configuration file
            
        Returns:
            True if dataset structure is valid, False otherwise
        """
        pass
    
    def save_uploaded_file(self, file: UploadFile, filename: str = None) -> str:
        """Save uploaded file to datasets directory
        
        Args:
            file: Uploaded file
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = file.filename
        
        file_path = os.path.join(self.datasets_dir, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)
        
        return file_path
    
    def extract_zip(self, zip_path: str, extract_to: str = None) -> str:
        """Extract ZIP file to specified directory
        
        Args:
            zip_path: Path to ZIP file
            extract_to: Directory to extract to (optional)
            
        Returns:
            Path to extraction directory
        """
        if extract_to is None:
            extract_to = os.path.splitext(zip_path)[0]
        
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        return extract_to
    
    def find_files_by_extension(self, directory: str, extensions: list) -> list:
        """Find all files with specified extensions in directory
        
        Args:
            directory: Directory to search
            extensions: List of file extensions (e.g., ['.jpg', '.png'])
            
        Returns:
            List of file paths
        """
        found_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    found_files.append(os.path.join(root, file))
        
        return found_files
    
    def clean_temp_files(self, file_path: str) -> None:
        """Clean up temporary files
        
        Args:
            file_path: Path to file or directory to clean up
        """
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Warning: Could not clean up {file_path}: {e}")
    
    def get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """Get information about a dataset
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Dictionary containing dataset information
        """
        # Basic implementation - can be overridden by subclasses
        return {
            'path': dataset_path,
            'exists': os.path.exists(dataset_path),
            'size': self._get_directory_size(dataset_path) if os.path.exists(dataset_path) else 0
        }
    
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        return total_size