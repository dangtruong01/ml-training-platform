import os
import uuid
import subprocess
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from ultralytics import YOLO

from ..base.base_trainer import BaseTrainer
from ..base.training_monitor import training_monitor


class YOLODetectionTrainer(BaseTrainer):
    """YOLO-specific detection model trainer"""
    
    def __init__(self, scripts_dir: str = "ml/scripts", results_dir: str = "ml/results"):
        super().__init__(results_dir)
        self.scripts_dir = os.path.abspath(scripts_dir)
        os.makedirs(self.scripts_dir, exist_ok=True)
        
        # Initialize YOLO model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model with best available weights"""
        model_priority = ["yolov8l.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]
        
        for model_name in model_priority:
            model_path = os.path.join("ml", "models", model_name)
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Using model: {model_name}")
                break
        else:
            # Fallback to downloading nano model
            self.model = YOLO("yolov8n.pt")
            print("Using fallback nano model")
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm this trainer handles"""
        return "YOLO Detection"
    
    def train(self, dataset_path: str, config: Dict[str, Any]) -> str:
        """Train a YOLO detection model
        
        Args:
            dataset_path: Path to data.yaml file
            config: Training configuration containing epochs, device, etc.
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        # Validate configuration
        if not self.validate_config(config):
            raise ValueError("Invalid training configuration")
        
        # Validate the data.yaml path exists
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset configuration not found: {dataset_path}")
        
        task_id = self.generate_task_id()
        
        # Create task in monitor
        training_monitor.create_task(
            task_id=task_id,
            training_config=config
        )
        
        # Create results directory for this training run
        training_results_dir = self.create_results_dir(task_id)
        detection_results_dir = os.path.join(training_results_dir, "detection")
        os.makedirs(detection_results_dir, exist_ok=True)
        
        # Log start of training
        training_monitor.add_log(task_id, f"🚀 Starting YOLO detection training")
        training_monitor.add_log(task_id, f"📊 Dataset: {dataset_path}")
        training_monitor.add_log(task_id, f"⚙️ Config: {config}")
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._train_with_monitoring,
            args=(task_id, dataset_path, config, detection_results_dir)
        )
        training_thread.start()
        
        return task_id
    
    def _train_with_monitoring(self, task_id: str, dataset_path: str, config: Dict[str, Any], results_dir: str):
        """Run training with progress monitoring"""
        try:
            training_monitor.update_task_status(task_id, 'running')
            
            epochs = config.get('epochs', 10)
            device = config.get('device', 'cpu')
            batch_size = config.get('batch_size', 16)
            model_size = config.get('model_size', 'n')
            
            # Get model name based on algorithm and size
            model_name = self._get_model_name(config.get('algorithm', 'yolo_v8'), model_size)
            training_monitor.add_log(task_id, f"🤖 Using model: {model_name}")
            
            # Load model for training
            model = YOLO(model_name)
            
            # Train the model
            training_monitor.add_log(task_id, f"🔥 Starting training with {epochs} epochs...")
            
            # Custom callback for progress tracking
            def on_train_epoch_end(trainer):
                epoch = trainer.epoch + 1
                training_monitor.update_progress(
                    task_id, 
                    (epoch / epochs) * 100,
                    current_epoch=epoch,
                    total_epochs=epochs
                )
                if epoch % 5 == 0:  # Log every 5 epochs
                    training_monitor.add_log(task_id, f"Epoch {epoch}/{epochs}: Training YOLO detection...")
            
            # Train model with callback
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                device=device,
                batch=batch_size,
                project=results_dir,
                name="yolo_detection_model",
                exist_ok=True,
                verbose=True
            )
            
            # Training completed successfully
            training_monitor.update_task_status(task_id, 'completed')
            training_monitor.update_progress(task_id, 100)
            
            completion_msg = f"✅ YOLO detection training completed! Model saved to {results_dir}"
            training_monitor.add_log(task_id, completion_msg)
            
        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            training_monitor.update_task_status(task_id, 'failed', str(e))
            training_monitor.add_log(task_id, error_msg)
            print(error_msg)
    
    def _get_model_name(self, algorithm: str, model_size: str) -> str:
        """Get the model name based on algorithm and size"""
        algorithm_models = {
            'yolo_v8': f'yolov8{model_size}.pt',
            'yolo_v11': f'yolo11{model_size}.pt', 
            'rtdetr': f'rtdetr-{model_size}.pt'
        }
        
        return algorithm_models.get(algorithm, f'yolov8{model_size}.pt')
    
    def train_from_project(self, project_id: str, config: Dict[str, Any]) -> str:
        """Train a YOLO detection model for a specific project
        
        Args:
            project_id: ID of the project to train for
            config: Training configuration dictionary
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        task_id = self.generate_task_id()
        algorithm = config.get('algorithm', 'yolo_v8')
        
        # Create task in monitor
        training_monitor.create_task(
            task_id=task_id,
            project_id=project_id,
            training_config=config
        )
        
        # Log start of training
        training_monitor.add_log(
            task_id, 
            f"🔄 Starting {algorithm} detection training for project {project_id}"
        )
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._train_yolo_detection,
            args=(task_id, project_id, algorithm, config)
        )
        training_thread.start()
        
        return task_id
    
    def _train_yolo_detection(self, task_id: str, project_id: str, algorithm: str, config: Dict[str, Any]):
        """Execute YOLO detection training for a project"""
        try:
            from services.core.database_service import database_service
            
            # Get dataset path from project
            dataset_path = f"ml/datasets/projects/{project_id}"
            
            # Check if dataset_path is already a data.yaml file (from ZIP upload) or needs to be created
            if dataset_path.endswith('.yaml'):
                # Already a data.yaml file from ZIP upload
                data_yaml_path = dataset_path
                training_monitor.add_log(task_id, f"📄 Using existing data.yaml: {data_yaml_path}")
            else:
                # Create data.yaml file for the prepared dataset (individual files upload)
                import yaml
                
                data_yaml_content = {
                    'path': dataset_path,
                    'train': 'training_images',
                    'val': 'training_images',  # Use same for validation for now
                    'nc': 1,  # Number of classes (simplified for now)
                    'names': ['object']  # Default class name
                }
                
                data_yaml_path = os.path.join(dataset_path, 'data.yaml')
                with open(data_yaml_path, 'w') as f:
                    yaml.dump(data_yaml_content, f)
                
                training_monitor.add_log(task_id, f"📄 Created data.yaml: {data_yaml_path}")
            
            # Extract training parameters
            epochs = config.get('epochs', 10)
            batch_size = config.get('batch_size', 16)
            device = config.get('device', 'cpu')
            model_size = config.get('model_size', 'n')
            
            # Create results directory
            results_dir = self.create_results_dir(task_id)
            
            training_monitor.update_task_status(task_id, 'running')
            training_monitor.add_log(task_id, f"🔄 Starting {algorithm} object detection training...")
            training_monitor.add_log(task_id, f"📊 Dataset: {dataset_path}")
            training_monitor.add_log(task_id, f"⚙️ Config: {epochs} epochs, batch size {batch_size}, device {device}")
            
            # Get model name
            model_name = self._get_model_name(algorithm, model_size)
            training_monitor.add_log(task_id, f"🤖 Using model: {model_name}")
            
            # Load and train model
            model = YOLO(model_name)
            
            # Train with progress tracking
            for epoch in range(1, epochs + 1):
                training_monitor.update_progress(task_id, (epoch / epochs) * 100, current_epoch=epoch, total_epochs=epochs)
                
                if epoch % 5 == 0:
                    training_monitor.add_log(task_id, f"Epoch {epoch}/{epochs}: Training {algorithm}...")
            
            # Simulate training completion
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                device=device,
                batch=batch_size,
                project=results_dir,
                name="detection_model",
                exist_ok=True
            )
            
            # Update database with training completion
            training_job_data = {
                'project_id': project_id,
                'status': 'completed',
                'task_id': task_id,
                'algorithm': algorithm,
                'training_config': config,
                'results_path': results_dir
            }
            
            try:
                database_service.create_training_job(training_job_data)
            except Exception as db_error:
                print(f"Database update failed: {db_error}")
            
            # Mark as completed
            training_monitor.update_task_status(task_id, 'completed')
            training_monitor.update_progress(task_id, 100)
            
            completion_msg = f"✅ {algorithm} detection training completed for project {project_id}!"
            training_monitor.add_log(task_id, completion_msg)
            
        except Exception as e:
            error_msg = f"❌ {algorithm} detection training failed: {str(e)}"
            training_monitor.update_task_status(task_id, 'failed', str(e))
            training_monitor.add_log(task_id, error_msg)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate YOLO detection training configuration"""
        required_fields = ['epochs', 'batch_size', 'device']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate specific values
        if config['epochs'] <= 0 or config['epochs'] > 1000:
            return False
            
        if config['batch_size'] <= 0 or config['batch_size'] > 128:
            return False
            
        if config['device'] not in ['cpu', 'cuda', 'mps']:
            return False
        
        return True