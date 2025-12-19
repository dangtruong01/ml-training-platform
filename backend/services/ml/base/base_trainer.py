import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .training_monitor import training_monitor


class BaseTrainer(ABC):
    """Abstract base class for all ML trainers"""
    
    def __init__(self, results_dir: str = "ml/results"):
        self.results_dir = os.path.abspath(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
    
    @abstractmethod
    def train(self, dataset_path: str, config: Dict[str, Any]) -> str:
        """Train a model with the given dataset and configuration
        
        Args:
            dataset_path: Path to the training dataset
            config: Training configuration dictionary
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        pass
    
    @abstractmethod  
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm this trainer handles"""
        pass
    
    def generate_task_id(self) -> str:
        """Generate a unique task ID"""
        return str(uuid.uuid4())
    
    def create_results_dir(self, task_id: str) -> str:
        """Create a results directory for the training task"""
        results_dir = os.path.join(self.results_dir, task_id)
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def train_from_project(self, project_id: str, config: Dict[str, Any]) -> str:
        """Train a model for a specific project
        
        Args:
            project_id: ID of the project to train for
            config: Training configuration dictionary
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        # Default implementation - can be overridden by subclasses
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
            f"🔄 Starting {self.get_algorithm_name()} training for project {project_id}"
        )
        
        return task_id
    
    def get_model_files(self, task_id: str) -> list:
        """Get list of model files produced by training
        
        Args:
            task_id: Training task identifier
            
        Returns:
            List of model file paths
        """
        results_dir = os.path.join(self.results_dir, task_id)
        model_files = []
        
        if not os.path.exists(results_dir):
            return model_files
        
        # Common model file extensions
        model_extensions = ['.pt', '.pth', '.pkl', '.joblib', '.h5', '.onnx']
        
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in model_extensions):
                    model_files.append({
                        'name': file,
                        'path': os.path.join(root, file),
                        'size': os.path.getsize(os.path.join(root, file))
                    })
        
        return model_files
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate training configuration
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if config is valid, False otherwise
        """
        # Basic validation - can be overridden by subclasses
        required_fields = ['epochs', 'batch_size', 'device']
        
        for field in required_fields:
            if field not in config:
                return False
        
        return True