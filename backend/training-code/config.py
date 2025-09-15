"""
Training configuration classes and utilities.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TrainingConfig:
    """Configuration for training jobs"""
    device: str = 'cpu'
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    model_size: str = 'n'
    
    # Database connection for progress updates
    database_host: Optional[str] = None
    database_port: str = '5432'
    database_name: str = 'ml_training_pipeline'
    database_user: Optional[str] = None
    database_password: Optional[str] = ''
    
    # Cloud Storage
    models_bucket: str = 'dangtruong-mltraining-storage'
    project_id: str = 'ml-training-pipeline-sand-jjgq'
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'device': self.device,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'model_size': self.model_size,
            'database_host': self.database_host,
            'database_port': self.database_port,
            'database_name': self.database_name,
            'database_user': self.database_user,
            'database_password': self.database_password,
            'models_bucket': self.models_bucket,
            'project_id': self.project_id
        }