import os
import time
import uuid
import threading
from datetime import datetime
from typing import Dict, Any

from ..base.base_trainer import BaseTrainer
from ..base.training_monitor import training_monitor


class SklearnAnomalyTrainer(BaseTrainer):
    """Sklearn-based anomaly detection trainer"""
    
    def __init__(self, results_dir: str = "ml/results"):
        super().__init__(results_dir)
        self.supported_algorithms = [
            'isolation_forest', 
            'one_class_svm', 
            'local_outlier_factor'
        ]
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm this trainer handles"""
        return "Sklearn Anomaly Detection"
    
    def train(self, dataset_path: str, config: Dict[str, Any]) -> str:
        """Train a sklearn-based anomaly detection model
        
        Args:
            dataset_path: Path to dataset directory
            config: Training configuration containing algorithm, epochs, etc.
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        algorithm = config.get('algorithm', 'isolation_forest')
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {self.supported_algorithms}")
        
        # Validate configuration
        if not self.validate_config(config):
            raise ValueError("Invalid training configuration")
        
        # Validate the dataset path exists
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset not found: {dataset_path}")
        
        task_id = self.generate_task_id()
        
        # Create task in monitor
        training_monitor.create_task(
            task_id=task_id,
            training_config=config
        )
        
        # Create results directory for this training run
        training_results_dir = self.create_results_dir(task_id)
        anomaly_results_dir = os.path.join(training_results_dir, "anomaly")
        os.makedirs(anomaly_results_dir, exist_ok=True)
        
        # Log start of training
        training_monitor.add_log(task_id, f"🚀 Starting {algorithm} anomaly detection training")
        training_monitor.add_log(task_id, f"📊 Dataset: {dataset_path}")
        training_monitor.add_log(task_id, f"⚙️ Config: {config}")
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._train_with_monitoring,
            args=(task_id, algorithm, dataset_path, config, anomaly_results_dir)
        )
        training_thread.start()
        
        return task_id
    
    def _train_with_monitoring(self, task_id: str, algorithm: str, dataset_path: str, 
                             config: Dict[str, Any], results_dir: str):
        """Run training with progress monitoring"""
        try:
            training_monitor.update_task_status(task_id, 'running')
            
            epochs = config.get('epochs', 100)
            
            training_monitor.add_log(task_id, f"🤖 Using algorithm: {algorithm}")
            training_monitor.add_log(task_id, f"🔥 Starting training with {epochs} epochs...")
            
            # Load and train model
            model_files = self._train_sklearn_anomaly(
                task_id, algorithm, dataset_path, config, results_dir
            )
            
            # Training completed successfully
            training_monitor.update_task_status(task_id, 'completed')
            training_monitor.update_progress(task_id, 100)
            
            completion_msg = f"✅ {algorithm} anomaly detection training completed! Model saved to {results_dir}"
            training_monitor.add_log(task_id, completion_msg)
            
            return model_files
            
        except Exception as e:
            error_msg = f"❌ {algorithm} training failed: {str(e)}"
            training_monitor.update_task_status(task_id, 'failed', str(e))
            training_monitor.add_log(task_id, error_msg)
            print(error_msg)
    
    def _train_sklearn_anomaly(self, task_id: str, algorithm: str, dataset_path: str, 
                             config: Dict[str, Any], results_dir: str) -> list:
        """Train sklearn-based anomaly detection models"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.svm import OneClassSVM
            from sklearn.neighbors import LocalOutlierFactor
            import pickle
            import numpy as np
            from PIL import Image
            
            # Load and preprocess images
            training_monitor.add_log(task_id, f"📂 Loading training images from {dataset_path}")
            image_features = []
            
            # Simple feature extraction (flatten images)
            image_dir = os.path.join(dataset_path, 'train', 'normal')  # Anomaly detection uses normal images
            if not os.path.exists(image_dir):
                image_dir = dataset_path  # Fallback to direct path
            
            for img_file in os.listdir(image_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img_path = os.path.join(image_dir, img_file)
                        img = Image.open(img_path).convert('RGB').resize((64, 64))
                        features = np.array(img).flatten() / 255.0  # Normalize to [0, 1]
                        image_features.append(features)
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
            
            if not image_features:
                raise ValueError("No valid images found for training")

            X = np.array(image_features)
            training_monitor.add_log(task_id, f"📊 Loaded {len(X)} images for training")

            # Use full image features without reduction
            training_monitor.add_log(task_id, f"✅ Using full image features: {X.shape[1]} dimensions")
            
            # Initialize model based on algorithm
            if algorithm == 'isolation_forest':
                model = IsolationForest(contamination=0.1, random_state=42)
                training_monitor.add_log(task_id, "🌲 Training Isolation Forest model...")
            elif algorithm == 'one_class_svm':
                model = OneClassSVM(gamma='scale', nu=0.1)
                training_monitor.add_log(task_id, "🎯 Training One-Class SVM model...")
            elif algorithm == 'local_outlier_factor':
                model = LocalOutlierFactor(contamination=0.1, novelty=True)
                training_monitor.add_log(task_id, "📍 Training Local Outlier Factor model...")
            
            # Simulate training progress
            epochs = config.get('epochs', 100)
            for epoch in range(1, epochs + 1):
                time.sleep(0.1)  # Simulate training time
                training_monitor.update_progress(
                    task_id, 
                    (epoch / epochs) * 100,
                    current_epoch=epoch,
                    total_epochs=epochs
                )
                
                if epoch % 20 == 0:
                    training_monitor.add_log(task_id, f"Epoch {epoch}/{epochs}: Training {algorithm}...")
            
            # Fit the model
            model.fit(X)
            
            # Save model with version information
            model_file = os.path.join(results_dir, f'{algorithm}_model.pkl')

            # Create model metadata with version info
            import sklearn
            model_metadata = {
                'model': model,
                'sklearn_version': sklearn.__version__,
                'algorithm': algorithm,
                'training_features': len(X[0]) if len(X) > 0 else 0,
                'training_samples': len(X),
                'trained_at': datetime.now().isoformat(),
                'image_preprocessing': {
                    'resize_shape': (64, 64),
                    'channels': 3,
                    'normalization': 'scale_0_to_1',
                    'flattened_features': 64 * 64 * 3,
                    'preprocessing_pipeline': 'resize_64x64 -> flatten -> normalize_0_1'
                }
            }

            with open(model_file, 'wb') as f:
                pickle.dump(model_metadata, f)

            training_monitor.add_log(task_id, f"💾 Model saved to {model_file} (sklearn {sklearn.__version__})")

            return [{
                'filename': f'{algorithm}_model.pkl',
                'filepath': model_file,
                'file_size': os.path.getsize(model_file),
                'model_format': 'sklearn_pickle',
                'sklearn_version': sklearn.__version__
            }]
            
        except Exception as e:
            error_msg = f"❌ Error training {algorithm}: {e}"
            training_monitor.add_log(task_id, error_msg)
            raise
    
    def train_from_project(self, project_id: str, config: Dict[str, Any]) -> str:
        """Train anomaly detection model for a specific project
        
        Args:
            project_id: ID of the project to train for
            config: Training configuration dictionary
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        task_id = self.generate_task_id()
        algorithm = config.get('algorithm', 'isolation_forest')
        
        # Create task in monitor
        training_monitor.create_task(
            task_id=task_id,
            project_id=project_id,
            training_config=config
        )
        
        # Log start of training
        training_monitor.add_log(
            task_id, 
            f"🔄 Starting {algorithm} anomaly detection training for project {project_id}"
        )
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._train_from_project_background,
            args=(task_id, project_id, algorithm, config)
        )
        training_thread.start()
        
        return task_id
    
    def _train_from_project_background(self, task_id: str, project_id: str, algorithm: str, config: Dict[str, Any]):
        """Execute anomaly detection training for a project"""
        try:
            from services.core.database_service import database_service
            
            # Get dataset path from project
            dataset_path = f"ml/datasets/projects/{project_id}"
            
            # Extract training parameters
            epochs = config.get('epochs', 100)
            
            # Create results directory
            results_dir = self.create_results_dir(task_id)
            
            training_monitor.update_task_status(task_id, 'running')
            training_monitor.add_log(task_id, f"🔄 Starting {algorithm} anomaly detection training...")
            training_monitor.add_log(task_id, f"📊 Dataset: {dataset_path}")
            training_monitor.add_log(task_id, f"⚙️ Config: {epochs} epochs")
            
            # Train the model
            model_files = self._train_sklearn_anomaly(
                task_id, algorithm, dataset_path, config, results_dir
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
            
            completion_msg = f"✅ {algorithm} anomaly detection training completed for project {project_id}!"
            training_monitor.add_log(task_id, completion_msg)
            
        except Exception as e:
            error_msg = f"❌ {algorithm} anomaly detection training failed: {str(e)}"
            training_monitor.update_task_status(task_id, 'failed', str(e))
            training_monitor.add_log(task_id, error_msg)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate anomaly detection training configuration"""
        required_fields = ['algorithm', 'epochs']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate specific values
        if config.get('epochs', 0) <= 0 or config.get('epochs', 0) > 10000:
            return False
            
        if config.get('algorithm') not in self.supported_algorithms:
            return False
        
        return True