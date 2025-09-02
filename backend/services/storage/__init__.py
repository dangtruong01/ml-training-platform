import os
from dotenv import load_dotenv
from .base_storage import BaseStorageService
from .local_storage import LocalStorageService

# Load environment variables from .env file
load_dotenv()


def get_storage_service() -> BaseStorageService:
    """
    Factory function to get the appropriate storage service based on environment.
    
    Environment Variables:
    - STORAGE_TYPE: 'local' or 'gcs' (default: 'local')
    - GCS_BUCKET_NAME: Required if using GCS
    - GOOGLE_CLOUD_PROJECT: Required if using GCS
    """
    storage_type = os.getenv('STORAGE_TYPE', 'local').lower()
    
    if storage_type == 'gcs':
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable is required for Google Cloud Storage")
        
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required for Google Cloud Storage")
        
        from .gcs_storage import GoogleCloudStorageService
        print(f"🌥️ Using Google Cloud Storage: gs://{bucket_name}")
        return GoogleCloudStorageService(bucket_name, project_id)
    
    else:
        # Default to local storage
        base_path = os.getenv('LOCAL_STORAGE_PATH', 'ml/auto_annotation')
        print(f"💾 Using Local Storage: {base_path}")
        return LocalStorageService(base_path)


# Global storage service instance (initialized on first import)
storage_service: BaseStorageService = get_storage_service()