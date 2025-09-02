from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, BinaryIO
import os
from pathlib import Path


class BaseStorageService(ABC):
    """
    Abstract base class for storage services.
    Supports both local filesystem and cloud storage backends.
    """

    @abstractmethod
    async def upload_file(self, file_data: bytes, file_path: str, content_type: Optional[str] = None) -> str:
        """Upload file data to storage and return the storage URL/path"""
        pass

    @abstractmethod
    async def download_file(self, file_path: str) -> bytes:
        """Download file data from storage"""
        pass

    @abstractmethod
    async def upload_image(self, image_array, file_path: str, format: str = 'JPEG') -> str:
        """Upload OpenCV/numpy image array to storage"""
        pass

    @abstractmethod
    async def download_image(self, file_path: str):
        """Download image from storage and return as OpenCV array"""
        pass

    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in storage"""
        pass

    @abstractmethod
    async def list_files(self, directory_path: str, extension_filter: Optional[str] = None) -> List[str]:
        """List files in directory with optional extension filter"""
        pass

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage"""
        pass

    @abstractmethod
    async def create_directory(self, directory_path: str) -> bool:
        """Create directory (if applicable for storage type)"""
        pass

    @abstractmethod
    async def save_json(self, data: Dict[Any, Any], file_path: str) -> str:
        """Save JSON data to storage"""
        pass

    @abstractmethod
    async def load_json(self, file_path: str) -> Dict[Any, Any]:
        """Load JSON data from storage"""
        pass

    @abstractmethod
    def get_url(self, file_path: str) -> str:
        """Get accessible URL for file (for downloads, previews, etc.)"""
        pass

    @abstractmethod
    def get_project_directory(self, project_id: str) -> str:
        """Get base directory path for a project"""
        pass

    def get_roi_directories(self, project_id: str) -> Dict[str, str]:
        """Get ROI cache directory paths for a project"""
        base_dir = self.get_project_directory(project_id)
        return {
            'normal': os.path.join(base_dir, 'roi_cache'),
            'defective': os.path.join(base_dir, 'defective_roi_cache')
        }

    def get_annotation_directories(self, project_id: str) -> Dict[str, str]:
        """Get annotation directory paths for a project"""
        base_dir = self.get_project_directory(project_id)
        return {
            'bounding_boxes': os.path.join(base_dir, 'generated_annotations'),
            'segmentation_masks': os.path.join(base_dir, 'segmentation_annotations'),
            'visual_boxes': os.path.join(base_dir, 'visual_bounding_boxes'),
            'visual_masks': os.path.join(base_dir, 'visual_segmentation_masks')
        }

    def get_model_directories(self, project_id: str) -> Dict[str, str]:
        """Get model storage directory paths for a project"""
        base_dir = self.get_project_directory(project_id)
        return {
            'features': os.path.join(base_dir, 'anomaly_features'),
            'defect_results': os.path.join(base_dir, 'defect_detection_results')
        }