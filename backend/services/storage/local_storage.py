import os
import json
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
import shutil
import base64
from .base_storage import BaseStorageService


class LocalStorageService(BaseStorageService):
    """
    Local filesystem storage implementation.
    Maintains backward compatibility with existing file structure.
    """

    def __init__(self, base_path: str = "ml/auto_annotation"):
        self.base_path = base_path
        self.projects_path = os.path.join(base_path, "projects")
        # Ensure base directories exist
        os.makedirs(self.projects_path, exist_ok=True)

    async def upload_file(self, file_data: bytes, file_path: str, content_type: Optional[str] = None) -> str:
        """Upload file data to local filesystem"""
        full_path = os.path.join(self.base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'wb') as f:
            f.write(file_data)
        
        return full_path

    async def download_file(self, file_path: str) -> bytes:
        """Download file data from local filesystem"""
        full_path = os.path.join(self.base_path, file_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        
        with open(full_path, 'rb') as f:
            return f.read()

    async def upload_image(self, image_array, file_path: str, format: str = 'JPEG') -> str:
        """Upload OpenCV/numpy image array to local filesystem"""
        full_path = os.path.join(self.base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Convert format for cv2.imwrite
        if format.upper() == 'JPEG':
            extension = '.jpg'
        elif format.upper() == 'PNG':
            extension = '.png'
        else:
            extension = '.jpg'  # Default
        
        if not full_path.endswith(extension):
            full_path += extension
        
        success = cv2.imwrite(full_path, image_array)
        if not success:
            raise RuntimeError(f"Failed to save image to {full_path}")
        
        return full_path

    async def download_image(self, file_path: str):
        """Download image from local filesystem and return as OpenCV array"""
        full_path = os.path.join(self.base_path, file_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        image = cv2.imread(full_path)
        if image is None:
            raise RuntimeError(f"Failed to load image from {full_path}")
        
        return image

    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in local filesystem"""
        full_path = os.path.join(self.base_path, file_path)
        return os.path.exists(full_path)

    async def list_files(self, directory_path: str, extension_filter: Optional[str] = None) -> List[str]:
        """List files in directory with optional extension filter"""
        full_path = os.path.join(self.base_path, directory_path)
        
        if not os.path.exists(full_path):
            return []
        
        files = []
        for filename in os.listdir(full_path):
            file_full_path = os.path.join(full_path, filename)
            if os.path.isfile(file_full_path):
                if extension_filter is None or filename.lower().endswith(extension_filter.lower()):
                    # Return relative path from base_path
                    relative_path = os.path.join(directory_path, filename)
                    files.append(relative_path)
        
        return files

    async def delete_file(self, file_path: str) -> bool:
        """Delete file from local filesystem"""
        full_path = os.path.join(self.base_path, file_path)
        
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                return True
            return False
        except Exception as e:
            print(f"❌ Error deleting file {full_path}: {e}")
            return False

    async def create_directory(self, directory_path: str) -> bool:
        """Create directory in local filesystem"""
        full_path = os.path.join(self.base_path, directory_path)
        
        try:
            os.makedirs(full_path, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Error creating directory {full_path}: {e}")
            return False

    async def save_json(self, data: Dict[Any, Any], file_path: str) -> str:
        """Save JSON data to local filesystem"""
        full_path = os.path.join(self.base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return full_path

    async def load_json(self, file_path: str) -> Dict[Any, Any]:
        """Load JSON data from local filesystem"""
        full_path = os.path.join(self.base_path, file_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"JSON file not found: {full_path}")
        
        with open(full_path, 'r') as f:
            return json.load(f)

    def get_url(self, file_path: str) -> str:
        """Get local file URL (for development/testing)"""
        # In local mode, return relative path for local server access
        return f"/files/{file_path}"

    def get_project_directory(self, project_id: str) -> str:
        """Get base directory path for a project"""
        return os.path.join("projects", project_id)

    # Utility methods for backward compatibility
    def get_absolute_path(self, relative_path: str) -> str:
        """Convert storage-relative path to absolute filesystem path"""
        return os.path.join(self.base_path, relative_path)

    async def copy_directory(self, source_path: str, dest_path: str) -> bool:
        """Copy entire directory structure (useful for project templates)"""
        try:
            source_full = os.path.join(self.base_path, source_path)
            dest_full = os.path.join(self.base_path, dest_path)
            
            if os.path.exists(source_full):
                shutil.copytree(source_full, dest_full, dirs_exist_ok=True)
                return True
            return False
        except Exception as e:
            print(f"❌ Error copying directory {source_path} to {dest_path}: {e}")
            return False