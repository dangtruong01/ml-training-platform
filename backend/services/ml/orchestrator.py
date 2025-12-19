"""
ML Service Orchestrator - Demonstrates how modular components work together
"""
from typing import Dict, Any, List, Optional
import os

from .base import training_monitor
from .datasets import YOLODatasetProcessor
from .detection import YOLODetectionTrainer
from .segmentation import YOLOSegmentationTrainer
from .anomaly import SklearnAnomalyTrainer, PytorchAnomalyTrainer
from .prediction import YOLOPredictor
from .pre_annotation import OpenCVAnnotator


class MLOrchestrator:
    """Orchestrator that demonstrates modular ML pipeline usage"""
    
    def __init__(self, 
                 datasets_dir: str = "ml/datasets",
                 models_dir: str = "ml/models", 
                 results_dir: str = "ml/results"):
        
        self.datasets_dir = os.path.abspath(datasets_dir)
        self.models_dir = os.path.abspath(models_dir)
        self.results_dir = os.path.abspath(results_dir)
        
        # Initialize modular components
        self._init_components()
    
    def _init_components(self):
        """Initialize all modular components"""
        # Dataset processing
        self.dataset_processor = YOLODatasetProcessor(self.datasets_dir)
        
        # Trainers
        self.yolo_detection_trainer = YOLODetectionTrainer(results_dir=self.results_dir)
        self.yolo_segmentation_trainer = YOLOSegmentationTrainer(results_dir=self.results_dir)
        self.sklearn_anomaly_trainer = SklearnAnomalyTrainer(self.results_dir)
        self.pytorch_anomaly_trainer = PytorchAnomalyTrainer(self.results_dir)
        
        # Prediction and annotation
        self.yolo_predictor = YOLOPredictor(self.models_dir)
        self.opencv_annotator = OpenCVAnnotator(self.results_dir)
    
    def get_available_trainers(self) -> Dict[str, List[str]]:
        """Get list of available training algorithms by category"""
        return {
            "detection": ["yolo_v8", "yolo_v11", "rtdetr"],
            "segmentation": ["yolo_v8_seg"],
            "anomaly_sklearn": ["isolation_forest", "one_class_svm", "local_outlier_factor"],
            "anomaly_pytorch": ["autoencoder"]
        }
    
    def train_model(self, 
                   model_type: str, 
                   algorithm: str, 
                   dataset_path: str, 
                   config: Dict[str, Any]) -> str:
        """Train a model using the appropriate modular trainer
        
        Args:
            model_type: Type of model ('detection', 'segmentation', 'anomaly')
            algorithm: Specific algorithm to use
            dataset_path: Path to dataset or project ID
            config: Training configuration
            
        Returns:
            task_id: Training task identifier
        """
        config['algorithm'] = algorithm
        
        if model_type == "detection":
            return self.yolo_detection_trainer.train(dataset_path, config)
        elif model_type == "segmentation":
            return self.yolo_segmentation_trainer.train(dataset_path, config)
        elif model_type == "anomaly":
            if algorithm in ['isolation_forest', 'one_class_svm', 'local_outlier_factor']:
                return self.sklearn_anomaly_trainer.train(dataset_path, config)
            elif algorithm == 'autoencoder':
                return self.pytorch_anomaly_trainer.train(dataset_path, config)
            else:
                raise ValueError(f"Unknown anomaly algorithm: {algorithm}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_from_project(self,
                          model_type: str,
                          algorithm: str, 
                          project_id: str,
                          config: Dict[str, Any]) -> str:
        """Train a model for a specific project"""
        config['algorithm'] = algorithm
        
        if model_type == "detection":
            return self.yolo_detection_trainer.train_from_project(project_id, config)
        elif model_type == "segmentation":
            return self.yolo_segmentation_trainer.train_from_project(project_id, config)
        elif model_type == "anomaly":
            if algorithm in ['isolation_forest', 'one_class_svm', 'local_outlier_factor']:
                return self.sklearn_anomaly_trainer.train_from_project(project_id, config)
            elif algorithm == 'autoencoder':
                return self.pytorch_anomaly_trainer.train_from_project(project_id, config)
            else:
                raise ValueError(f"Unknown anomaly algorithm: {algorithm}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict(self,
               image_path: str,
               task_type: str = "detection",
               model_path: Optional[str] = None) -> Dict[str, Any]:
        """Run prediction using YOLO predictor"""
        return self.yolo_predictor.predict(image_path, model_path, task_type)
    
    def predict_batch(self,
                     image_paths: List[str],
                     task_type: str = "detection", 
                     model_path: Optional[str] = None) -> Dict[str, Any]:
        """Run batch prediction using YOLO predictor"""
        return self.yolo_predictor.predict_batch(image_paths, model_path, task_type)
    
    def pre_annotate(self,
                    image_path: str,
                    annotation_type: str = "detection") -> str:
        """Generate pre-annotation using OpenCV"""
        if annotation_type == "detection":
            return self.opencv_annotator.annotate_detection(image_path)
        elif annotation_type == "segmentation":
            return self.opencv_annotator.annotate_segmentation(image_path)
        else:
            raise ValueError(f"Unknown annotation type: {annotation_type}")
    
    def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """Get training status using training monitor"""
        return training_monitor.get_task_status(task_id)
    
    def get_training_logs(self, task_id: str, lines: int = 50) -> Dict[str, Any]:
        """Get training logs using training monitor"""
        return training_monitor.get_task_logs(task_id, lines)
    
    def list_training_tasks(self) -> Dict[str, Any]:
        """List all training tasks"""
        return training_monitor.list_tasks()
    
    def create_ml_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, str]:
        """Example: Create a complete ML pipeline with multiple steps
        
        Args:
            pipeline_config: Configuration containing dataset_path, models to train, etc.
            
        Returns:
            dict: Task IDs for each step
        """
        pipeline_results = {}
        
        # Step 1: Process dataset if needed
        dataset_path = pipeline_config.get('dataset_path')
        
        # Step 2: Train multiple models if requested
        models_to_train = pipeline_config.get('models', [])
        
        for model_config in models_to_train:
            model_type = model_config['type']
            algorithm = model_config['algorithm'] 
            training_config = model_config.get('config', {})
            
            task_id = self.train_model(model_type, algorithm, dataset_path, training_config)
            pipeline_results[f"{model_type}_{algorithm}"] = task_id
        
        return pipeline_results
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all loaded components"""
        return {
            "dataset_processor": self.dataset_processor.__class__.__name__,
            "trainers": {
                "detection": self.yolo_detection_trainer.get_algorithm_name(),
                "segmentation": self.yolo_segmentation_trainer.get_algorithm_name(),
                "sklearn_anomaly": self.sklearn_anomaly_trainer.get_algorithm_name(),
                "pytorch_anomaly": self.pytorch_anomaly_trainer.get_algorithm_name()
            },
            "predictor": self.yolo_predictor.get_algorithm_name(),
            "annotator": self.opencv_annotator.__class__.__name__,
            "directories": {
                "datasets": self.datasets_dir,
                "models": self.models_dir,
                "results": self.results_dir
            }
        }


# Create global orchestrator instance
ml_orchestrator = MLOrchestrator()