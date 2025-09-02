import os
import json
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
import asyncio
from .base_storage import BaseStorageService


class GoogleCloudStorageService(BaseStorageService):
    """
    Google Cloud Storage implementation.
    Handles authentication, bucket operations, and URL generation.
    """

    def __init__(self, bucket_name: str, project_id: str = None):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._client = None
        self._bucket = None
        self.base_prefix = "auto_annotation"  # Base folder in bucket

    async def _get_client(self):
        """Lazy initialization of GCS client"""
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=self.project_id)
                self._bucket = self._client.bucket(self.bucket_name)
                print(f"✅ Connected to Google Cloud Storage bucket: {self.bucket_name}")
            except Exception as e:
                print(f"❌ Failed to initialize Google Cloud Storage: {e}")
                raise
        return self._client

    def _get_blob_path(self, file_path: str) -> str:
        """Convert relative file path to GCS blob path"""
        return f"{self.base_prefix}/{file_path}"

    async def upload_file(self, file_data: bytes, file_path: str, content_type: Optional[str] = None) -> str:
        """Upload file data to Google Cloud Storage"""
        await self._get_client()
        
        blob_path = self._get_blob_path(file_path)
        blob = self._bucket.blob(blob_path)
        
        # Upload in a thread to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None, blob.upload_from_string, file_data, content_type
        )
        
        return f"gs://{self.bucket_name}/{blob_path}"

    async def download_file(self, file_path: str) -> bytes:
        """Download file data from Google Cloud Storage"""
        await self._get_client()
        
        blob_path = self._get_blob_path(file_path)
        blob = self._bucket.blob(blob_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"File not found in GCS: {blob_path}")
        
        # Download in a thread to avoid blocking
        data = await asyncio.get_event_loop().run_in_executor(
            None, blob.download_as_bytes
        )
        
        return data

    async def upload_image(self, image_array, file_path: str, format: str = 'JPEG') -> str:
        """Upload OpenCV/numpy image array to Google Cloud Storage"""
        # Encode image to bytes
        if format.upper() == 'JPEG':
            extension = '.jpg'
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif format.upper() == 'PNG':
            extension = '.png'
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        else:
            extension = '.jpg'
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
        
        # Ensure file path has correct extension
        if not file_path.endswith(extension):
            file_path += extension
        
        success, encoded_image = cv2.imencode(extension, image_array, encode_param)
        if not success:
            raise RuntimeError(f"Failed to encode image for upload: {file_path}")
        
        image_bytes = encoded_image.tobytes()
        content_type = 'image/jpeg' if format.upper() == 'JPEG' else 'image/png'
        
        return await self.upload_file(image_bytes, file_path, content_type)

    async def download_image(self, file_path: str):
        """Download image from Google Cloud Storage and return as OpenCV array"""
        image_bytes = await self.download_file(file_path)
        
        # Decode bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise RuntimeError(f"Failed to decode image from GCS: {file_path}")
        
        return image

    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in Google Cloud Storage"""
        await self._get_client()
        
        blob_path = self._get_blob_path(file_path)
        blob = self._bucket.blob(blob_path)
        
        return blob.exists()

    async def list_files(self, directory_path: str, extension_filter: Optional[str] = None) -> List[str]:
        """List files in GCS directory with optional extension filter"""
        await self._get_client()
        
        prefix = self._get_blob_path(directory_path)
        if not prefix.endswith('/'):
            prefix += '/'
        
        blobs = self._bucket.list_blobs(prefix=prefix)
        files = []
        
        for blob in blobs:
            # Get relative path from the directory
            relative_path = blob.name[len(self.base_prefix)+1:]  # Remove base prefix
            
            if extension_filter is None or relative_path.lower().endswith(extension_filter.lower()):
                files.append(relative_path)
        
        return files

    async def delete_file(self, file_path: str) -> bool:
        """Delete file from Google Cloud Storage"""
        await self._get_client()
        
        try:
            blob_path = self._get_blob_path(file_path)
            blob = self._bucket.blob(blob_path)
            
            if blob.exists():
                await asyncio.get_event_loop().run_in_executor(None, blob.delete)
                return True
            return False
        except Exception as e:
            print(f"❌ Error deleting file from GCS {file_path}: {e}")
            return False

    async def create_directory(self, directory_path: str) -> bool:
        """Create directory marker in Google Cloud Storage"""
        # GCS doesn't have true directories, but we can create a marker file
        try:
            marker_path = os.path.join(directory_path, '.keep')
            await self.upload_file(b'', marker_path)
            return True
        except Exception as e:
            print(f"❌ Error creating directory marker in GCS {directory_path}: {e}")
            return False

    async def save_json(self, data: Dict[Any, Any], file_path: str) -> str:
        """Save JSON data to Google Cloud Storage"""
        json_bytes = json.dumps(data, indent=2).encode('utf-8')
        return await self.upload_file(json_bytes, file_path, 'application/json')

    async def load_json(self, file_path: str) -> Dict[Any, Any]:
        """Load JSON data from Google Cloud Storage"""
        json_bytes = await self.download_file(file_path)
        return json.loads(json_bytes.decode('utf-8'))

    def get_url(self, file_path: str) -> str:
        """Get public URL for file in Google Cloud Storage"""
        blob_path = self._get_blob_path(file_path)
        return f"https://storage.googleapis.com/{self.bucket_name}/{blob_path}"

    def get_project_directory(self, project_id: str) -> str:
        """Get base directory path for a project in GCS"""
        return f"projects/{project_id}"

    async def generate_signed_url(self, file_path: str, expiration_minutes: int = 60) -> str:
        """Generate signed URL for secure file access"""
        await self._get_client()
        
        blob_path = self._get_blob_path(file_path)
        blob = self._bucket.blob(blob_path)
        
        from datetime import timedelta
        url = blob.generate_signed_url(expiration=timedelta(minutes=expiration_minutes))
        return url

    async def upload_from_local_file(self, local_path: str, storage_path: str) -> str:
        """Upload existing local file to Google Cloud Storage"""
        with open(local_path, 'rb') as f:
            file_data = f.read()
        
        # Detect content type based on file extension
        extension = Path(local_path).suffix.lower()
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.json': 'application/json',
            '.txt': 'text/plain'
        }
        content_type = content_type_map.get(extension, 'application/octet-stream')
        
        return await self.upload_file(file_data, storage_path, content_type)

    async def batch_upload_directory(self, local_directory: str, storage_directory: str) -> Dict[str, str]:
        """Upload entire local directory to GCS (useful for migration)"""
        results = {}
        
        if not os.path.exists(local_directory):
            return results
        
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # Calculate relative path from local_directory
                rel_path = os.path.relpath(local_file_path, local_directory)
                storage_file_path = os.path.join(storage_directory, rel_path).replace('\\', '/')
                
                try:
                    gcs_url = await self.upload_from_local_file(local_file_path, storage_file_path)
                    results[rel_path] = gcs_url
                    print(f"✅ Uploaded: {rel_path} → {gcs_url}")
                except Exception as e:
                    print(f"❌ Failed to upload {rel_path}: {e}")
                    results[rel_path] = f"ERROR: {str(e)}"
        
        return results