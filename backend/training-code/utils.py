"""
Utilities for training jobs running on Vertex AI.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from google.cloud import storage
from google.cloud import logging as cloud_logging
import psycopg2
from psycopg2.extras import Json

def setup_logging(task_id: str):
    """Set up logging for the training job"""
    # Set up local logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{task_id}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    try:
        # Set up Cloud Logging
        client = cloud_logging.Client()
        client.setup_logging()
        logging.info("✅ Cloud Logging initialized")
    except Exception as e:
        logging.warning(f"⚠️ Could not initialize Cloud Logging: {e}")

def download_dataset(dataset_path: str, local_dir: str) -> str:
    """Download dataset from Cloud Storage to local directory"""
    logging.info(f"📥 Downloading dataset from {dataset_path}")
    
    try:
        storage_client = storage.Client()
        
        # Parse GCS path (gs://bucket/path)
        if not dataset_path.startswith('gs://'):
            raise ValueError(f"Dataset path must start with gs://: {dataset_path}")
        
        path_parts = dataset_path[5:].split('/', 1)  # Remove 'gs://'
        bucket_name = path_parts[0]
        blob_prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        bucket = storage_client.bucket(bucket_name)
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Download all files with the prefix
        blobs = bucket.list_blobs(prefix=blob_prefix)
        downloaded_files = 0
        
        for blob in blobs:
            if blob.name.endswith('/'):  # Skip directories
                continue
                
            # Create local file path
            local_file_path = os.path.join(local_dir, os.path.basename(blob.name))
            
            # Download file
            blob.download_to_filename(local_file_path)
            downloaded_files += 1
            
            if downloaded_files % 10 == 0:  # Progress update every 10 files
                logging.info(f"📥 Downloaded {downloaded_files} files...")
        
        logging.info(f"✅ Downloaded {downloaded_files} files to {local_dir}")
        return local_dir
        
    except Exception as e:
        logging.error(f"❌ Failed to download dataset: {e}")
        raise

def _get_model_type_folder(model_type: str) -> str:
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

def upload_results(task_id: str, model_dir: str, results: Dict[str, Any], model_type: str = 'anomaly') -> List[Dict[str, Any]]:
    """Upload training results to Cloud Storage"""
    logging.info(f"📤 Uploading results for task {task_id}")
    
    try:
        storage_client = storage.Client()
        bucket_name = os.getenv('MODELS_BUCKET', 'dangtruong-mltraining-storage')
        bucket = storage_client.bucket(bucket_name)
        
        uploaded_files = []
        
        # Upload all files in model_dir
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # Create cloud storage path organized by model type and task_id
                relative_path = os.path.relpath(local_file_path, model_dir)
                model_folder = _get_model_type_folder(model_type)
                cloud_path = f"{model_folder}/{task_id}/{relative_path}"
                
                # Upload file
                blob = bucket.blob(cloud_path)
                blob.upload_from_filename(local_file_path)
                
                # Get file info
                file_info = {
                    'filename': file,
                    'filepath': f"gs://{bucket_name}/{cloud_path}",
                    'local_path': local_file_path,
                    'file_size': os.path.getsize(local_file_path),
                    'created_at': datetime.now().isoformat()
                }
                uploaded_files.append(file_info)
                
                logging.info(f"📤 Uploaded {file} → {cloud_path}")
        
        logging.info(f"✅ Uploaded {len(uploaded_files)} files")
        return uploaded_files
        
    except Exception as e:
        logging.error(f"❌ Failed to upload results: {e}")
        raise

def update_database_progress(task_id: str, status: str = None, progress: float = None, 
                           current_epoch: int = None, logs: List[str] = None):
    """Update training progress in database"""
    try:
        # Database connection from environment variables
        conn = psycopg2.connect(
            host=os.getenv('DATABASE_HOST'),
            port=os.getenv('DATABASE_PORT', '5432'),
            dbname=os.getenv('DATABASE_NAME', 'ml_training_pipeline'),
            user=os.getenv('DATABASE_USER'),
            password=os.getenv('DATABASE_PASSWORD', '')
        )
        
        with conn.cursor() as cursor:
            update_fields = []
            values = []
            
            if status:
                update_fields.append("status = %s")
                values.append(status)
            
            if progress is not None:
                update_fields.append("progress = %s")
                values.append(progress)
            
            if current_epoch is not None:
                update_fields.append("current_epoch = %s")
                values.append(current_epoch)
            
            if logs:
                update_fields.append("training_logs = %s")
                values.append(Json(logs))
            
            update_fields.append("updated_at = %s")
            values.append(datetime.now())
            
            values.append(task_id)  # For WHERE clause
            
            query = f"""
                UPDATE training_jobs 
                SET {', '.join(update_fields)}
                WHERE task_id = %s
            """
            
            cursor.execute(query, values)
            conn.commit()
            
            logging.info(f"✅ Updated database progress for {task_id}")
            
    except Exception as e:
        logging.warning(f"⚠️ Could not update database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def save_training_metrics(task_id: str, metrics: Dict[str, Any]):
    """Save training metrics to a JSON file"""
    try:
        metrics_file = f"/tmp/{task_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"✅ Saved metrics to {metrics_file}")
        return metrics_file
    except Exception as e:
        logging.error(f"❌ Failed to save metrics: {e}")
        return None