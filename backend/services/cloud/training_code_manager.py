"""
Training code upload manager for pre-built containers.
Uploads training code to Cloud Storage for Vertex AI to download.
"""
import os
import zipfile
import tempfile
from datetime import datetime
from typing import Dict, Any
from google.cloud import storage

from services.cloud.gcp_config import gcp_config

class TrainingCodeManager:
    """Manages training code upload to Cloud Storage"""
    
    def __init__(self):
        self.storage_client = None
        self.bucket_name = None
        
    def initialize(self):
        """Initialize the code manager"""
        if not gcp_config.initialize():
            raise RuntimeError("Failed to initialize GCP configuration")
            
        self.storage_client = gcp_config.get_storage_client()
        self.bucket_name = gcp_config.models_bucket
        print(f"✅ Training code manager initialized - Bucket: {self.bucket_name}")
    
    def _get_model_type_folder(self, model_type: str) -> str:
        """Map model type to storage folder name"""
        type_mapping = {
            'anomaly': 'anomaly',
            'anomaly_detection': 'anomaly', 
            'object_detection': 'detection',
            'classification': 'classification',
            'segmentation': 'segmentation',
            'yolo': 'detection',  # YOLO is typically object detection
            'defect_detection': 'anomaly'  # Defect detection is a type of anomaly detection
        }
        return type_mapping.get(model_type.lower(), 'other')
    
    def upload_training_code(self, task_id: str, model_type: str = 'other') -> str:
        """
        Upload training code to Cloud Storage and return the download URL.
        
        Args:
            task_id: Unique training task ID
            model_type: Type of model being trained (for organized storage)
            
        Returns:
            Cloud Storage path to the uploaded code archive
        """
        if not self.storage_client:
            self.initialize()
        
        try:
            # Create a zip file of the training code
            code_archive_path = self._create_code_archive(task_id)
            
            # Upload to Cloud Storage with organized structure
            model_folder = self._get_model_type_folder(model_type)
            blob_path = f"training-code/{model_folder}/{task_id}/training_code.zip"
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            print(f"📤 Uploading training code: {blob_path}")
            blob.upload_from_filename(code_archive_path)
            
            # Clean up local zip file
            os.unlink(code_archive_path)
            
            gs_path = f"gs://{self.bucket_name}/{blob_path}"
            print(f"✅ Training code uploaded: {gs_path}")
            
            return gs_path
            
        except Exception as e:
            print(f"❌ Failed to upload training code: {e}")
            raise
    
    def _create_code_archive(self, task_id: str) -> str:
        """Create a zip archive of the training code"""
        training_code_dir = os.path.abspath("training-code")
        
        if not os.path.exists(training_code_dir):
            raise FileNotFoundError(f"Training code directory not found: {training_code_dir}")
        
        # Create temporary zip file
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, f"training_code_{task_id}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(training_code_dir):
                for file in files:
                    if file.endswith(('.py', '.txt', '.yaml', '.yml', '.json')):
                        file_path = os.path.join(root, file)
                        # Store with relative path in the zip
                        arcname = os.path.relpath(file_path, training_code_dir)
                        zipf.write(file_path, arcname)
                        print(f"   📄 Added: {arcname}")
        
        print(f"📦 Created code archive: {zip_path}")
        return zip_path
    
    def get_startup_script(self, code_gs_path: str, task_id: str, training_config: Dict[str, Any]) -> str:
        """
        Generate startup script that downloads and runs the training code.
        
        Args:
            code_gs_path: Cloud Storage path to training code zip
            task_id: Training task ID
            training_config: Training configuration
            
        Returns:
            Startup script content
        """
        
        # Extract database connection info
        database_host = os.getenv('DATABASE_HOST', 'localhost')
        database_user = os.getenv('DATABASE_USER', os.getenv('USER'))
        database_password = os.getenv('DATABASE_PASSWORD', '')
        
        import json
        config_json = json.dumps(training_config).replace("'", "\\'")
        
        script = f'''#!/bin/bash
set -e

echo "🚀 Starting Vertex AI training job: {task_id}"

# Check hardware information
echo "🔧 Checking hardware configuration..."
nvidia-smi || echo "⚠️ nvidia-smi not available"
lscpu | head -10 || echo "⚠️ lscpu not available"
cat /proc/meminfo | head -5 || echo "⚠️ /proc/meminfo not available"

# Install additional requirements
pip install scikit-learn google-cloud-storage google-cloud-logging psycopg2-binary python-dotenv requests ultralytics opencv-python-headless

# Download training code
echo "📥 Downloading training code from {code_gs_path}"
gsutil cp {code_gs_path} /tmp/training_code.zip

# Extract training code
cd /tmp
unzip training_code.zip
chmod +x *.py

echo "🎯 Starting training..."

# Run the training script
python main.py \\
    --model-dir=$AIP_MODEL_DIR \\
    --task-id={task_id} \\
    --project-id={training_config.get('project_id', '')} \\
    --model-type=anomaly \\
    --config-json='{config_json}' \\
    --database-host={database_host} \\
    --database-user={database_user} \\
    --database-password={database_password} \\
    --models-bucket={self.bucket_name}

echo "✅ Training job completed"
'''
        
        return script

# Global instance
training_code_manager = TrainingCodeManager()