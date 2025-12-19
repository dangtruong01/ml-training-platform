import os
import time
import uuid
import threading
from datetime import datetime
from typing import Dict, Any

from ..base.base_trainer import BaseTrainer
from ..base.training_monitor import training_monitor


class PytorchAnomalyTrainer(BaseTrainer):
    """PyTorch-based anomaly detection trainer"""
    
    def __init__(self, results_dir: str = "ml/results"):
        super().__init__(results_dir)
        self.supported_algorithms = ['autoencoder']
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm this trainer handles"""
        return "PyTorch Anomaly Detection"
    
    def train(self, dataset_path: str, config: Dict[str, Any]) -> str:
        """Train a PyTorch-based anomaly detection model
        
        Args:
            dataset_path: Path to dataset directory
            config: Training configuration containing algorithm, epochs, etc.
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        algorithm = config.get('algorithm', 'autoencoder')
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
            model_files = self._train_pytorch_anomaly(
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
    
    def _train_pytorch_anomaly(self, task_id: str, algorithm: str, dataset_path: str, 
                             config: Dict[str, Any], results_dir: str) -> list:
        """Train PyTorch-based anomaly detection models (Autoencoder)"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            import numpy as np
            from PIL import Image
            
            # Simple Autoencoder architecture
            class SimpleAutoencoder(nn.Module):
                def __init__(self, input_dim=64*64*3, hidden_dim=128):
                    super(SimpleAutoencoder, self).__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim//2),
                        nn.ReLU()
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(hidden_dim//2, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, input_dim),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
            
            # Load and preprocess images
            training_monitor.add_log(task_id, f"📂 Loading training images from {dataset_path}")
            image_features = []
            
            image_dir = os.path.join(dataset_path, 'train', 'normal')  
            if not os.path.exists(image_dir):
                image_dir = dataset_path
            
            for img_file in os.listdir(image_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img_path = os.path.join(image_dir, img_file)
                        img = Image.open(img_path).convert('RGB').resize((64, 64))
                        features = np.array(img).flatten() / 255.0  # Normalize
                        image_features.append(features)
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
            
            if not image_features:
                raise ValueError("No valid images found for training")
            
            X = torch.FloatTensor(image_features)
            dataset = TensorDataset(X, X)  # Autoencoder learns to reconstruct input
            dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 16), shuffle=True)
            
            training_monitor.add_log(task_id, f"📊 Loaded {len(X)} images for autoencoder training")
            
            # Initialize model
            model = SimpleAutoencoder()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
            
            training_monitor.add_log(task_id, "🧠 Training PyTorch Autoencoder model...")
            
            # Training loop
            epochs = config.get('epochs', 100)
            for epoch in range(1, epochs + 1):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(dataloader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Update progress
                training_monitor.update_progress(
                    task_id,
                    (epoch / epochs) * 100,
                    current_epoch=epoch,
                    total_epochs=epochs
                )
                
                if epoch % 20 == 0:
                    avg_loss = epoch_loss / len(dataloader)
                    training_monitor.add_log(task_id, f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")
                
                time.sleep(0.05)  # Simulate training time
            
            # Save model
            model_file = os.path.join(results_dir, f'{algorithm}_model.pth')
            torch.save(model.state_dict(), model_file)
            
            training_monitor.add_log(task_id, f"💾 Autoencoder model saved to {model_file}")
            
            return [{
                'filename': f'{algorithm}_model.pth',
                'filepath': model_file,
                'file_size': os.path.getsize(model_file),
                'model_format': 'pytorch_state_dict'
            }]
            
        except Exception as e:
            error_msg = f"❌ Error training autoencoder: {e}"
            training_monitor.add_log(task_id, error_msg)
            raise
    
    def train_from_project(self, project_id: str, config: Dict[str, Any]) -> str:
        """Train PyTorch anomaly detection model for a specific project
        
        Args:
            project_id: ID of the project to train for
            config: Training configuration dictionary
            
        Returns:
            task_id: Unique identifier for tracking the training process
        """
        task_id = self.generate_task_id()
        algorithm = config.get('algorithm', 'autoencoder')
        
        # Create task in monitor
        training_monitor.create_task(
            task_id=task_id,
            project_id=project_id,
            training_config=config
        )
        
        # Log start of training
        training_monitor.add_log(
            task_id, 
            f"🔄 Starting {algorithm} PyTorch anomaly detection training for project {project_id}"
        )
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._train_from_project_background,
            args=(task_id, project_id, algorithm, config)
        )
        training_thread.start()
        
        return task_id
    
    def _train_from_project_background(self, task_id: str, project_id: str, algorithm: str, config: Dict[str, Any]):
        """Execute PyTorch anomaly detection training for a project"""
        try:
            from services.core.database_service import database_service
            
            # Get dataset path from project
            dataset_path = f"ml/datasets/projects/{project_id}"
            
            # Extract training parameters
            epochs = config.get('epochs', 100)
            batch_size = config.get('batch_size', 16)
            learning_rate = config.get('learning_rate', 0.001)
            
            # Create results directory
            results_dir = self.create_results_dir(task_id)
            
            training_monitor.update_task_status(task_id, 'running')
            training_monitor.add_log(task_id, f"🔄 Starting {algorithm} PyTorch anomaly detection training...")
            training_monitor.add_log(task_id, f"📊 Dataset: {dataset_path}")
            training_monitor.add_log(task_id, f"⚙️ Config: {epochs} epochs, batch size {batch_size}, lr {learning_rate}")
            
            # Train the model
            model_files = self._train_pytorch_anomaly(
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
            
            completion_msg = f"✅ {algorithm} PyTorch anomaly detection training completed for project {project_id}!"
            training_monitor.add_log(task_id, completion_msg)
            
        except Exception as e:
            error_msg = f"❌ {algorithm} PyTorch anomaly detection training failed: {str(e)}"
            training_monitor.update_task_status(task_id, 'failed', str(e))
            training_monitor.add_log(task_id, error_msg)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate PyTorch anomaly detection training configuration"""
        required_fields = ['algorithm', 'epochs']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate specific values
        if config.get('epochs', 0) <= 0 or config.get('epochs', 0) > 10000:
            return False
            
        if config.get('batch_size', 1) <= 0 or config.get('batch_size', 1) > 1024:
            return False
            
        if config.get('learning_rate', 0) <= 0 or config.get('learning_rate', 0) > 1:
            return False
            
        if config.get('algorithm') not in self.supported_algorithms:
            return False
        
        return True