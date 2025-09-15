"""
Main entry point for Vertex AI training jobs.
This script runs inside the training container on Vertex AI.
"""
import os
import sys
import argparse
import json
import requests
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from anomaly_trainer import AnomalyTrainer
from utils import setup_logging, download_dataset, upload_results
from config import TrainingConfig

def notify_training_completion(task_id: str, results: dict, uploaded_files: list, status: str = 'completed', error_message: str = None):
    """Write completion metadata to Cloud Storage for monitoring service to pick up"""
    try:
        from google.cloud import storage
        import json
        from datetime import datetime
        
        # Write completion metadata to Cloud Storage
        storage_client = storage.Client()
        bucket_name = 'mltraining-vertex-staging'  # Use staging bucket
        
        completion_metadata = {
            'task_id': task_id,
            'status': status,
            'progress': 100.0 if status == 'completed' else None,
            'model_files_info': uploaded_files if status == 'completed' else [],
            'training_metrics': results.get('metrics', {}) if status == 'completed' else {},
            'error_message': error_message,
            'completed_at': datetime.utcnow().isoformat(),
            'notification_source': 'training_container'
        }
        
        # Write metadata file to Cloud Storage
        bucket = storage_client.bucket(bucket_name)
        metadata_blob = bucket.blob(f"job-completions/{task_id}.json")
        metadata_blob.upload_from_string(
            json.dumps(completion_metadata, indent=2),
            content_type='application/json'
        )
        
        print(f"✅ Wrote completion metadata to gs://{bucket_name}/job-completions/{task_id}.json")
            
    except Exception as e:
        print(f"❌ Error writing completion metadata: {e}")
        # Don't fail the entire training job if notification fails

def check_gpu_availability():
    """Check and log GPU availability"""
    print("\n" + "="*50)
    print("🔧 HARDWARE DETECTION")
    print("="*50)
    
    try:
        import torch
        print(f"🐍 PyTorch version: {torch.__version__}")
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"⚡ CUDA version: {torch.version.cuda}")
            print(f"🎯 Device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(i)
                print(f"🚀 GPU {i}: {device.name}")
                print(f"   💾 Memory: {device.total_memory // 1024**3} GB")
                print(f"   🏗️ Compute capability: {device.major}.{device.minor}")
                print(f"   🔧 Multi-processors: {device.multi_processor_count}")
            
            # Test GPU memory allocation
            print("🧪 Testing GPU memory allocation...")
            test_tensor = torch.ones(1000, 1000).cuda()
            print("✅ GPU memory allocation successful")
            del test_tensor
            torch.cuda.empty_cache()
            
        else:
            print("⚠️ No GPU available - training will use CPU")
            
    except ImportError:
        print("❌ PyTorch not available")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
    
    # Check system info
    try:
        import platform
        print(f"💻 System: {platform.system()} {platform.release()}")
        print(f"🏭 Machine: {platform.machine()}")
        print(f"🔧 Processor: {platform.processor()}")
    except:
        pass
    
    print("="*50 + "\n")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ML Training on Vertex AI')
    
    # Vertex AI provides these automatically
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory to save model artifacts')
    
    # Custom parameters we'll pass
    parser.add_argument('--task-id', type=str, required=True,
                       help='Training task ID')
    parser.add_argument('--project-id', type=str, required=True,
                       help='Database project ID')  
    parser.add_argument('--model-type', type=str, required=True,
                       help='Type of model to train')
    parser.add_argument('--dataset-path', type=str,
                       help='Cloud Storage path to dataset')
    parser.add_argument('--config-json', type=str, required=True,
                       help='Training configuration as JSON string')
    
    # Database connection parameters
    parser.add_argument('--database-host', type=str,
                       help='Database host')
    parser.add_argument('--database-user', type=str,
                       help='Database user')
    parser.add_argument('--database-password', type=str, default='',
                       help='Database password')
    
    # Cloud storage parameters
    parser.add_argument('--models-bucket', type=str,
                       default='dangtruong-mltraining-storage',
                       help='Cloud Storage bucket for models')
    
    return parser.parse_args()

def main():
    """Main entry point for Vertex AI training job"""
    args = parse_args()
    
    print(f"🚀 Starting training job: {args.task_id}")
    print(f"📁 Model dir: {args.model_dir}")
    print(f"🎯 Model type: {args.model_type}")
    print(f"📊 Dataset: {args.dataset_path}")
    
    # Set up environment variables
    if args.database_host:
        os.environ['DATABASE_HOST'] = args.database_host
    if args.database_user:
        os.environ['DATABASE_USER'] = args.database_user
    if args.database_password:
        os.environ['DATABASE_PASSWORD'] = args.database_password
    if args.models_bucket:
        os.environ['MODELS_BUCKET'] = args.models_bucket
    
    # Set up logging
    setup_logging(args.task_id)
    
    # Check GPU availability
    check_gpu_availability()
    
    try:
        # Parse training configuration
        training_config_dict = json.loads(args.config_json)
        
        # Add database connection info to config
        if args.database_host:
            training_config_dict['database_host'] = args.database_host
            training_config_dict['database_user'] = args.database_user
            training_config_dict['database_password'] = args.database_password
        
        training_config_dict['models_bucket'] = args.models_bucket
        training_config_dict['project_id'] = args.project_id
        
        config = TrainingConfig.from_dict(training_config_dict)
        
        print(f"⚙️ Training configuration: {config.to_dict()}")
        
        # Download dataset if needed
        local_dataset_dir = None
        if args.dataset_path and args.dataset_path.startswith('gs://'):
            local_dataset_dir = '/tmp/dataset'
            download_dataset(args.dataset_path, local_dataset_dir)
        
        # Initialize trainer based on model type
        if args.model_type == 'anomaly':
            trainer = AnomalyTrainer(
                task_id=args.task_id,
                project_id=args.project_id,
                model_dir=args.model_dir,
                dataset_path=args.dataset_path,
                config=config
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        print(f"🤖 Initialized {args.model_type} trainer")
        
        # Run training
        print("🎯 Starting training...")
        results = trainer.train()
        
        print(f"✅ Training completed: {results['status']}")
        
        # Upload results to Cloud Storage
        print("📤 Uploading results to Cloud Storage...")
        uploaded_files = upload_results(args.task_id, args.model_dir, results, args.model_type)
        
        print(f"🎉 Training job {args.task_id} completed successfully!")
        print(f"📁 Uploaded {len(uploaded_files)} files to Cloud Storage")
        
        # Notify backend of completion
        notify_training_completion(args.task_id, results, uploaded_files, 'completed')
        
        # Write success marker for Vertex AI
        success_file = os.path.join(args.model_dir, 'SUCCESS')
        with open(success_file, 'w') as f:
            f.write(f"Training completed successfully at {results['metrics']['completed_at']}")
        
    except Exception as e:
        error_msg = f"❌ Training failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # Notify backend of failure
        notify_training_completion(args.task_id, {}, [], 'failed', str(e))
        
        # Write failure marker for Vertex AI
        failure_file = os.path.join(args.model_dir, 'FAILURE')
        with open(failure_file, 'w') as f:
            f.write(f"Training failed: {str(e)}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()