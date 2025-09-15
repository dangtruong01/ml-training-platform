"""
Vertex AI service for submitting and managing training jobs.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomJob

from services.core.database_service import database_service
from services.cloud.gcp_config import gcp_config
from services.cloud.training_code_manager import training_code_manager

class VertexAIService:
    """Service for managing Vertex AI training jobs"""
    
    def __init__(self):
        self.project_id = None
        self.region = 'us-central1'
        self.container_uri = None
        self.staging_bucket = None
        
    def initialize(self):
        """Initialize Vertex AI service"""
        if not gcp_config.initialize():
            raise RuntimeError("Failed to initialize GCP configuration")
        
        self.project_id = gcp_config.project_id
        self.region = gcp_config.region
        # Use regional bucket for Vertex AI staging (must be same region as Vertex AI)
        self.staging_bucket = "gs://mltraining-vertex-staging"
        
        # Use official Vertex AI PyTorch GPU pre-built container (CUDA 12.x)
        self.container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
        
        print(f"✅ Vertex AI service initialized")
        print(f"   📋 Project: {self.project_id}")
        print(f"   🌍 Region: {self.region}")
        print(f"   🐳 Container: {self.container_uri}")
    
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
        
    def submit_training_job(
        self, 
        task_id: str, 
        project_id: str, 
        model_type: str,
        algorithm: str,
        training_config: Dict[str, Any],
        dataset_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit a custom training job to Vertex AI"""
        
        if not self.project_id:
            self.initialize()
        
        try:
            # Create training job record in database first
            db_result = database_service.create_training_job(
                task_id=task_id,
                project_id=project_id,
                model_type=model_type,
                algorithm=algorithm,
                training_config=training_config,
                total_epochs=training_config.get('epochs', 100)
            )
            
            if db_result['status'] != 'success':
                raise RuntimeError(f"Failed to create training job in database: {db_result['message']}")
            
            # Upload training code to Cloud Storage with model type organization
            print("📤 Uploading training code...")
            code_gs_path = training_code_manager.upload_training_code(task_id, model_type)
            
            # Add project_id to training config for the startup script
            training_config_with_project = training_config.copy()
            training_config_with_project['project_id'] = project_id
            
            # Generate startup script
            startup_script = training_code_manager.get_startup_script(
                code_gs_path=code_gs_path,
                task_id=task_id,
                training_config=training_config_with_project
            )
            
            # Select machine and GPU configuration
            machine_type, accelerator_type, accelerator_count = self._get_machine_config(training_config)
            
            # Create job spec using startup script instead of container args
            job_spec = {
                "worker_pool_specs": [
                    {
                        "machine_spec": {
                            "machine_type": machine_type,
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": self.container_uri,
                            "command": ["/bin/bash"],
                            "args": ["-c", startup_script],
                            "env": [
                                {"name": "AIP_MODEL_DIR", "value": f"{self.staging_bucket}/{self._get_model_type_folder(model_type)}/{task_id}"},
                                {"name": "PYTHONPATH", "value": "/tmp"},
                                {"name": "BACKEND_URL", "value": os.getenv('PUBLIC_BACKEND_URL', 'http://localhost:8000')}  # Set PUBLIC_BACKEND_URL for cloud deployment
                            ]
                        },
                    }
                ]
            }
            
            # Add accelerator (GPU) if needed
            if accelerator_type and accelerator_count > 0:
                job_spec["worker_pool_specs"][0]["machine_spec"]["accelerator_type"] = accelerator_type
                job_spec["worker_pool_specs"][0]["machine_spec"]["accelerator_count"] = accelerator_count
                print(f"🚀 Using GPU: {accelerator_type} (count: {accelerator_count})")
            
            # Create and submit custom job
            display_name = f"ml-training-{model_type}-{task_id}"
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.project_id,
                location=self.region,
                staging_bucket=self.staging_bucket
            )
            
            custom_job = CustomJob(
                display_name=display_name,
                worker_pool_specs=job_spec["worker_pool_specs"]
            )
            
            print(f"🚀 Submitting Vertex AI job: {display_name}")
            
            # Submit the job (non-blocking)
            custom_job.submit()
            
            # Get the job resource name
            vertex_ai_job_name = custom_job.resource_name
            vertex_ai_job_id = custom_job.name
            
            print(f"✅ Vertex AI job submitted: {vertex_ai_job_id}")
            
            # Update database with Vertex AI job info
            self._update_training_job_vertex_info(
                task_id=task_id,
                vertex_ai_job_id=vertex_ai_job_id,
                vertex_ai_job_name=vertex_ai_job_name,
                machine_type=machine_type,
                accelerator_type=accelerator_type
            )
            
            return {
                'status': 'submitted',
                'task_id': task_id,
                'vertex_ai_job_id': vertex_ai_job_id,
                'vertex_ai_job_name': vertex_ai_job_name,
                'display_name': display_name,
                'machine_type': machine_type,
                'accelerator_type': accelerator_type,
                'message': f'Training job submitted to Vertex AI: {vertex_ai_job_id}'
            }
            
        except Exception as e:
            # Update database with error
            database_service.update_training_job_status(
                task_id=task_id,
                status='failed',
                error_message=f"Failed to submit to Vertex AI: {str(e)}"
            )
            
            return {
                'status': 'error',
                'task_id': task_id,
                'message': f'Failed to submit training job: {str(e)}'
            }
    
    def get_job_status(self, vertex_ai_job_id: str) -> Dict[str, Any]:
        """Get the status of a Vertex AI job"""
        try:
            if not self.project_id:
                self.initialize()
            
            # Get job by ID
            custom_job = CustomJob.get(
                resource_name=vertex_ai_job_id,
                project=self.project_id,
                location=self.region
            )
            
            state = custom_job.state.name if custom_job.state else 'UNKNOWN'
            
            # Map Vertex AI states to our internal states
            status_mapping = {
                'JOB_STATE_QUEUED': 'pending',
                'JOB_STATE_PENDING': 'pending', 
                'JOB_STATE_RUNNING': 'running',
                'JOB_STATE_SUCCEEDED': 'completed',
                'JOB_STATE_FAILED': 'failed',
                'JOB_STATE_CANCELLING': 'cancelled',
                'JOB_STATE_CANCELLED': 'cancelled',
                'JOB_STATE_PAUSED': 'paused',
            }
            
            internal_status = status_mapping.get(state, 'unknown')
            
            return {
                'status': 'success',
                'vertex_ai_state': state,
                'internal_status': internal_status,
                'start_time': custom_job.start_time.isoformat() if custom_job.start_time else None,
                'end_time': custom_job.end_time.isoformat() if custom_job.end_time else None,
                'error': custom_job.error.message if custom_job.error else None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get job status: {str(e)}'
            }
    
    def _get_machine_config(self, training_config: Dict[str, Any]) -> tuple:
        """Get machine and accelerator configuration based on training config"""
        device = training_config.get('device', 'cpu').lower()
        model_size = training_config.get('model_size', 'n').lower()
        epochs = training_config.get('epochs', 100)
        
        # Model size based machine configuration
        machine_configs = {
            'n': {  # Nano
                'machine_type': 'n1-standard-2',
                'accelerator_type': None,
                'accelerator_count': 0
            },
            's': {  # Small
                'machine_type': 'n1-standard-4',
                'accelerator_type': None,
                'accelerator_count': 0
            },
            'm': {  # Medium
                'machine_type': 'n1-standard-8',
                'accelerator_type': 'NVIDIA_TESLA_T4',
                'accelerator_count': 1
            },
            'l': {  # Large
                'machine_type': 'n1-standard-16',
                'accelerator_type': 'NVIDIA_TESLA_T4',
                'accelerator_count': 1
            },
            'x': {  # XL
                'machine_type': 'n1-standard-32',
                'accelerator_type': 'NVIDIA_TESLA_V100',
                'accelerator_count': 1
            }
        }
        
        # Get base configuration for model size
        config = machine_configs.get(model_size, machine_configs['n'])
        machine_type = config['machine_type']
        accelerator_type = config['accelerator_type']
        accelerator_count = config['accelerator_count']
        
        # Override for GPU device requests
        if device == 'cuda':
            if accelerator_type is None:  # For nano/small, add minimal GPU
                accelerator_type = 'NVIDIA_TESLA_T4'
                accelerator_count = 1
                if machine_type in ['n1-standard-2', 'n1-standard-4']:
                    machine_type = 'n1-standard-4'  # Ensure sufficient CPU for GPU
        
        # For very long training, ensure GPU availability
        if epochs > 100 and accelerator_type is None:
            accelerator_type = 'NVIDIA_TESLA_T4'
            accelerator_count = 1
            machine_type = 'n1-standard-4'
        
        # For extremely long training, upgrade GPU
        if epochs > 300:
            accelerator_type = 'NVIDIA_TESLA_V100'
        
        return machine_type, accelerator_type, accelerator_count
    
    def _update_training_job_vertex_info(
        self, 
        task_id: str, 
        vertex_ai_job_id: str,
        vertex_ai_job_name: str, 
        machine_type: str,
        accelerator_type: Optional[str]
    ):
        """Update database with Vertex AI job information"""
        try:
            print(f"🔍 [VERTEX_DB] Updating job {task_id} with Vertex AI job ID: {vertex_ai_job_id}")
            
            # Use the database service method to ensure consistent session handling
            print(f"🔍 [VERTEX_DB] Using database service to update training config")
            
            # Get current training job to merge configs
            job_result = database_service.get_training_job(task_id)
            if job_result['status'] != 'success':
                print(f"❌ [VERTEX_DB] Could not get training job {task_id} from database")
                return
                
            current_job = job_result['training_job']
            current_config = current_job.get('training_config', {}) or {}
            
            # Merge Vertex AI info with existing config
            updated_config = current_config.copy()
            updated_config.update({
                'vertex_ai_job_id': vertex_ai_job_id,
                'vertex_ai_job_name': vertex_ai_job_name,
                'machine_type': machine_type,
                'accelerator_type': accelerator_type
            })
            
            print(f"🔍 [VERTEX_DB] Current config: {current_config}")
            print(f"🔍 [VERTEX_DB] Updated config: {updated_config}")
            
            # Use database service to update the training config  
            # We'll need to add this method to database service
            try:
                session = database_service.get_session()
                try:
                    from models.database_models import TrainingJob
                    
                    training_job = session.query(TrainingJob).filter(
                        TrainingJob.task_id == task_id
                    ).first()
                    
                    if training_job:
                        # Directly update the training_config JSON column
                        training_job.training_config = updated_config
                        
                        # Mark the column as modified for SQLAlchemy to detect the change
                        from sqlalchemy.orm.attributes import flag_modified
                        flag_modified(training_job, 'training_config')
                        
                        session.commit()
                        session.flush()  # Ensure changes are written immediately
                        print(f"✅ [VERTEX_DB] Successfully updated training_config with flag_modified")
                    else:
                        print(f"❌ [VERTEX_DB] Training job not found: {task_id}")
                        
                finally:
                    session.close()
                    
            except Exception as update_error:
                print(f"❌ [VERTEX_DB] Error during database update: {update_error}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"❌ [VERTEX_DB] Could not update Vertex AI info in database: {e}")
            import traceback
            traceback.print_exc()

# Global instance
vertex_ai_service = VertexAIService()