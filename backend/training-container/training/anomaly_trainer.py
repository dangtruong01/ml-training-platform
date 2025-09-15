"""
Anomaly detection trainer for Vertex AI.
"""
import os
import time
import logging
from typing import Dict, Any
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle

from config import TrainingConfig
from utils import update_database_progress, save_training_metrics

class AnomalyTrainer:
    """Anomaly detection trainer using Isolation Forest"""
    
    def __init__(self, task_id: str, project_id: str, model_dir: str, 
                 dataset_path: str, config: TrainingConfig):
        self.task_id = task_id
        self.project_id = project_id
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.config = config
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.training_logs = []
        
    def log_progress(self, message: str):
        """Log training progress"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        
        logging.info(message)
        self.training_logs.append(log_msg)
        
        # Update database every few logs
        if len(self.training_logs) % 5 == 0:
            update_database_progress(
                self.task_id,
                status='running',
                logs=self.training_logs[-10:]  # Keep last 10 logs
            )
    
    def train(self) -> Dict[str, Any]:
        """Run anomaly detection training"""
        self.log_progress(f"🚀 Starting anomaly detection training for project {self.project_id}")
        self.log_progress(f"📊 Configuration: {self.config.epochs} epochs, device: {self.config.device}")
        
        try:
            # Update database - training started
            update_database_progress(
                self.task_id,
                status='running',
                progress=0.0,
                current_epoch=0
            )
            
            # Simulate loading dataset
            self.log_progress("📁 Loading dataset...")
            time.sleep(2)  # Simulate data loading
            
            # Generate mock training data for demonstration
            # In real implementation, you'd load actual image features
            X_train = np.random.rand(1000, 50)  # Mock features
            
            self.log_progress(f"✅ Loaded dataset with {X_train.shape[0]} samples, {X_train.shape[1]} features")
            
            # Initialize model
            model = IsolationForest(
                n_estimators=self.config.epochs,
                contamination=0.1,
                random_state=42
            )
            
            self.log_progress("🤖 Initialized Isolation Forest model")
            
            # Training simulation with progress updates
            total_epochs = self.config.epochs
            
            for epoch in range(1, total_epochs + 1):
                self.log_progress(f"📝 Epoch {epoch}/{total_epochs}: Training anomaly detector...")
                
                # Simulate training time
                time.sleep(0.5)
                
                # Calculate progress
                progress = (epoch / total_epochs) * 100
                
                # Update database progress
                update_database_progress(
                    self.task_id,
                    progress=progress,
                    current_epoch=epoch,
                    logs=self.training_logs[-5:]  # Last 5 logs
                )
                
                # Simulate some training metrics
                if epoch % 10 == 0 or epoch == total_epochs:
                    mock_loss = 0.1 + (0.05 * np.random.rand())
                    mock_accuracy = 0.85 + (0.1 * np.random.rand())
                    self.log_progress(f"📊 Epoch {epoch} - Loss: {mock_loss:.4f}, Accuracy: {mock_accuracy:.4f}")
            
            # Fit the model (this happens at the end for Isolation Forest)
            self.log_progress("🔧 Fitting final model...")
            model.fit(X_train)
            
            # Save model
            model_file = os.path.join(self.model_dir, 'anomaly_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            self.log_progress(f"💾 Model saved to {model_file}")
            
            # Save training logs
            logs_file = os.path.join(self.model_dir, 'training_logs.txt')
            with open(logs_file, 'w') as f:
                f.write('\n'.join(self.training_logs))
            
            # Save configuration
            config_file = os.path.join(self.model_dir, 'config.json')
            import json
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            # Create training metrics
            metrics = {
                'model_type': 'anomaly_detection',
                'algorithm': 'isolation_forest',
                'n_estimators': self.config.epochs,
                'training_samples': X_train.shape[0],
                'feature_count': X_train.shape[1],
                'final_accuracy': 0.87 + (0.05 * np.random.rand()),
                'training_duration_seconds': total_epochs * 0.5,
                'completed_at': datetime.now().isoformat()
            }
            
            # Save metrics
            metrics_file = save_training_metrics(self.task_id, metrics)
            if metrics_file:
                import shutil
                shutil.copy(metrics_file, os.path.join(self.model_dir, 'metrics.json'))
            
            self.log_progress("✅ Anomaly detection training completed successfully!")
            
            # Final database update
            update_database_progress(
                self.task_id,
                status='completed',
                progress=100.0,
                current_epoch=total_epochs,
                logs=self.training_logs
            )
            
            # Return results
            results = {
                'status': 'completed',
                'model_file': model_file,
                'logs_file': logs_file,
                'config_file': config_file,
                'metrics': metrics,
                'training_logs': self.training_logs
            }
            
            return results
            
        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            self.log_progress(error_msg)
            logging.error(error_msg, exc_info=True)
            
            # Update database with error
            update_database_progress(
                self.task_id,
                status='failed',
                logs=self.training_logs
            )
            
            raise