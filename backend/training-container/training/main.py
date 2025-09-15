"""
Main entry point for Vertex AI training jobs.
This script runs inside the training container on Vertex AI.
"""
import os
import sys
import argparse
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from anomaly_trainer import AnomalyTrainer
from utils import setup_logging, download_dataset, upload_results
from config import TrainingConfig

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
        uploaded_files = upload_results(args.task_id, args.model_dir, results)
        
        print(f"🎉 Training job {args.task_id} completed successfully!")
        print(f"📁 Uploaded {len(uploaded_files)} files to Cloud Storage")
        
        # Write success marker for Vertex AI
        success_file = os.path.join(args.model_dir, 'SUCCESS')
        with open(success_file, 'w') as f:
            f.write(f"Training completed successfully at {results['metrics']['completed_at']}")
        
    except Exception as e:
        error_msg = f"❌ Training failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # Write failure marker for Vertex AI
        failure_file = os.path.join(args.model_dir, 'FAILURE')
        with open(failure_file, 'w') as f:
            f.write(f"Training failed: {str(e)}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()