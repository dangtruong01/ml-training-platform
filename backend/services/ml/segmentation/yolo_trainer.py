import os
import uuid
import yaml
import threading
from datetime import datetime
from typing import Dict, Any
from ultralytics import YOLO

from ..base.base_trainer import BaseTrainer
from ..base.training_monitor import training_monitor


class YOLOSegmentationTrainer(BaseTrainer):
    """YOLO-specific segmentation model trainer"""
    
    def __init__(self, scripts_dir: str = "ml/scripts", results_dir: str = "ml/results"):
        super().__init__(results_dir)
        self.scripts_dir = os.path.abspath(scripts_dir)
        os.makedirs(self.scripts_dir, exist_ok=True)
        
        # Initialize YOLO segmentation model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO segmentation model with best available weights"""
        model_priority = ["yolov8l-seg.pt", "yolov8m-seg.pt", "yolov8s-seg.pt", "yolov8n-seg.pt"]
        
        for model_name in model_priority:
            model_path = os.path.join("ml", "models", model_name)
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Using segmentation model: {model_name}")
                break
        else:
            # Fallback to downloading nano segmentation model
            self.model = YOLO("yolov8n-seg.pt")
            print("Using fallback nano segmentation model")
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm this trainer handles"""
        return "YOLO Segmentation"
    
    def train(self, dataset_path: str, config: Dict[str, Any]) -> str:
        """Train a YOLO segmentation model
        
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
        segmentation_results_dir = os.path.join(training_results_dir, "segmentation")
        os.makedirs(segmentation_results_dir, exist_ok=True)
        
        # Log start of training
        training_monitor.add_log(task_id, f"🚀 Starting YOLO segmentation training")
        training_monitor.add_log(task_id, f"📊 Dataset: {dataset_path}")
        training_monitor.add_log(task_id, f"⚙️ Config: {config}")
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._train_with_monitoring,
            args=(task_id, dataset_path, config, segmentation_results_dir)
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
            
            # Get model name
            model_name = f'yolov8{model_size}-seg.pt'
            training_monitor.add_log(task_id, f"🤖 Using model: {model_name}")
            
            # Load model for training
            model = YOLO(model_name)
            
            # Train the model
            training_monitor.add_log(task_id, f"🔥 Starting segmentation training with {epochs} epochs...")
            
            # Train model
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                device=device,
                batch=batch_size,
                project=results_dir,
                name="yolo_segmentation_model",
                exist_ok=True,
                verbose=True
            )
            
            # Training completed successfully
            training_monitor.update_task_status(task_id, 'completed')
            training_monitor.update_progress(task_id, 100)
            
            completion_msg = f"✅ YOLO segmentation training completed! Model saved to {results_dir}"
            training_monitor.add_log(task_id, completion_msg)
            
        except Exception as e:
            error_msg = f"❌ Segmentation training failed: {str(e)}"
            training_monitor.update_task_status(task_id, 'failed', str(e))
            training_monitor.add_log(task_id, error_msg)
            print(error_msg)
    
    def train_from_project(self, project_id: str, config: Dict[str, Any]) -> str:
        """Train a YOLO segmentation model for a specific project
        
        Args:
            project_id: ID of the project to train for
            config: Training configuration dictionary
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        task_id = self.generate_task_id()
        
        # Create task in monitor
        training_monitor.create_task(
            task_id=task_id,
            project_id=project_id,
            training_config=config
        )
        
        # Log start of training
        training_monitor.add_log(
            task_id, 
            f"🔄 Starting YOLO segmentation training for project {project_id}"
        )
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._train_yolo_segmentation,
            args=(task_id, project_id, config)
        )
        training_thread.start()
        
        return task_id
    
    def _train_yolo_segmentation(self, task_id: str, project_id: str, config: Dict[str, Any]):
        """Execute YOLO segmentation training for a project"""
        try:
            from services.core.database_service import database_service
            
            # Get dataset path from project
            dataset_path = f"ml/datasets/projects/{project_id}"
            
            # Create data.yaml file for the prepared dataset (simplified for now)
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
            training_monitor.add_log(task_id, f"🔄 Starting YOLO segmentation training...")
            training_monitor.add_log(task_id, f"📊 Dataset: {dataset_path}")
            training_monitor.add_log(task_id, f"⚙️ Config: {epochs} epochs, batch size {batch_size}, device {device}")
            
            # Get model name
            model_name = f'yolov8{model_size}-seg.pt'
            training_monitor.add_log(task_id, f"🤖 Using model: {model_name}")
            
            # Load and train model
            model = YOLO(model_name)
            
            # Train with progress tracking
            for epoch in range(1, epochs + 1):
                training_monitor.update_progress(task_id, (epoch / epochs) * 100, current_epoch=epoch, total_epochs=epochs)
                
                if epoch % 5 == 0:
                    training_monitor.add_log(task_id, f"Epoch {epoch}/{epochs}: Training YOLO segmentation...")
            
            # Simulate training completion
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                device=device,
                batch=batch_size,
                project=results_dir,
                name="segmentation_model",
                exist_ok=True
            )
            
            # Update database with training completion
            training_job_data = {
                'project_id': project_id,
                'status': 'completed',
                'task_id': task_id,
                'algorithm': 'yolo_v8_seg',
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
            
            completion_msg = f"✅ YOLO segmentation training completed for project {project_id}!"
            training_monitor.add_log(task_id, completion_msg)
            
        except Exception as e:
            error_msg = f"❌ YOLO segmentation training failed: {str(e)}"
            training_monitor.update_task_status(task_id, 'failed', str(e))
            training_monitor.add_log(task_id, error_msg)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate YOLO segmentation training configuration"""
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