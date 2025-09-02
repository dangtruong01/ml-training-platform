"""
Storage Adapter - Provides backward compatibility while migrating to storage abstraction.
This allows gradual migration without breaking existing code.
"""
import os
import asyncio
from typing import Optional, Dict, Any
from .base_storage import BaseStorageService
from . import storage_service


class StorageAdapter:
    """
    Adapter that provides both sync and async interfaces for storage operations.
    Allows gradual migration from direct file operations to storage service.
    """
    
    def __init__(self, storage: BaseStorageService):
        self.storage = storage
    
    # =============================================================================
    # SYNC COMPATIBILITY METHODS (for gradual migration)
    # =============================================================================
    
    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Sync wrapper for directory creation"""
        asyncio.run(self.storage.create_directory(path))
    
    def exists(self, path: str) -> bool:
        """Sync wrapper for file existence check"""
        return asyncio.run(self.storage.file_exists(path))
    
    def listdir(self, path: str) -> list:
        """Sync wrapper for directory listing"""
        files = asyncio.run(self.storage.list_files(path))
        # Return just filenames (not full paths) for compatibility
        return [os.path.basename(f) for f in files]
    
    def remove(self, path: str) -> None:
        """Sync wrapper for file deletion"""
        asyncio.run(self.storage.delete_file(path))
    
    def save_image_sync(self, image_array, path: str) -> str:
        """Sync wrapper for image upload"""
        return asyncio.run(self.storage.upload_image(image_array, path))
    
    def load_image_sync(self, path: str):
        """Sync wrapper for image download"""
        return asyncio.run(self.storage.download_image(path))
    
    def save_json_sync(self, data: Dict[Any, Any], path: str) -> str:
        """Sync wrapper for JSON save"""
        return asyncio.run(self.storage.save_json(data, path))
    
    def load_json_sync(self, path: str) -> Dict[Any, Any]:
        """Sync wrapper for JSON load"""
        return asyncio.run(self.storage.load_json(path))
    
    # =============================================================================
    # PATH TRANSLATION METHODS
    # =============================================================================
    
    def get_local_compatible_path(self, storage_path: str) -> str:
        """
        For backward compatibility, return a path that looks like local filesystem.
        This allows existing code to continue working while we migrate.
        """
        if isinstance(self.storage, LocalStorageService):
            return self.storage.get_absolute_path(storage_path)
        else:
            # For cloud storage, return the storage path as-is
            # The calling code will need to be updated to use storage service methods
            return storage_path
    
    def translate_local_to_storage_path(self, local_path: str) -> str:
        """
        Convert old hardcoded local paths to storage-relative paths.
        Example: 'ml/auto_annotation/projects/proj1/images/img.jpg' → 'projects/proj1/images/img.jpg'
        """
        # Remove the base path prefix if present
        base_prefixes = [
            'ml/auto_annotation/',
            'ml\\auto_annotation\\',
            '/ml/auto_annotation/',
            'ml/auto_annotation/projects/',
            'projects/'
        ]
        
        storage_path = local_path
        for prefix in base_prefixes:
            if storage_path.startswith(prefix):
                storage_path = storage_path[len(prefix):]
                break
        
        # Ensure forward slashes for cloud storage
        return storage_path.replace('\\', '/')
    
    # =============================================================================
    # PROJECT UTILITIES
    # =============================================================================
    
    def get_project_base_path(self, project_id: str) -> str:
        """Get project base path compatible with existing code"""
        storage_path = self.storage.get_project_directory(project_id)
        return self.get_local_compatible_path(storage_path)
    
    def get_roi_cache_paths(self, project_id: str) -> Dict[str, str]:
        """Get ROI cache paths compatible with existing code"""
        roi_dirs = self.storage.get_roi_directories(project_id)
        return {
            'normal': self.get_local_compatible_path(roi_dirs['normal']),
            'defective': self.get_local_compatible_path(roi_dirs['defective'])
        }


# Global adapter instance
storage_adapter = StorageAdapter(storage_service)


# =============================================================================
# MIGRATION HELPER FUNCTIONS
# =============================================================================

def migrate_cv2_imread(image_path: str):
    """
    Drop-in replacement for cv2.imread that works with storage service.
    
    Usage: Replace cv2.imread(path) with migrate_cv2_imread(path)
    """
    storage_path = storage_adapter.translate_local_to_storage_path(image_path)
    return storage_adapter.load_image_sync(storage_path)


def migrate_cv2_imwrite(image_path: str, image_array) -> bool:
    """
    Drop-in replacement for cv2.imwrite that works with storage service.
    
    Usage: Replace cv2.imwrite(path, img) with migrate_cv2_imwrite(path, img)
    """
    try:
        storage_path = storage_adapter.translate_local_to_storage_path(image_path)
        storage_adapter.save_image_sync(image_array, storage_path)
        return True
    except Exception as e:
        print(f"❌ Error saving image via storage service: {e}")
        return False


def migrate_json_dump(data: Dict[Any, Any], file_path: str) -> None:
    """
    Drop-in replacement for json.dump that works with storage service.
    """
    storage_path = storage_adapter.translate_local_to_storage_path(file_path)
    storage_adapter.save_json_sync(data, storage_path)


def migrate_json_load(file_path: str) -> Dict[Any, Any]:
    """
    Drop-in replacement for json.load that works with storage service.
    """
    storage_path = storage_adapter.translate_local_to_storage_path(file_path)
    return storage_adapter.load_json_sync(storage_path)


def migrate_os_path_exists(path: str) -> bool:
    """
    Drop-in replacement for os.path.exists that works with storage service.
    """
    storage_path = storage_adapter.translate_local_to_storage_path(path)
    return storage_adapter.exists(storage_path)