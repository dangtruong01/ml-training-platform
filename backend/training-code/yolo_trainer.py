"""
YOLO trainer for object detection and segmentation in Vertex AI containers.
"""
import os
import logging
import subprocess
import json
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

class YOLOTrainer:
    def __init__(self, task_id: str, project_id: str, model_dir: str, dataset_path: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.project_id = project_id
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.config = config

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        logging.info(f"🤖 YOLOTrainer initialized - Task: {task_id}, Project: {project_id}")

    def train(self) -> Dict[str, Any]:
        """Execute YOLO training"""
        try:
            logging.info(f"🚀 Starting YOLO training for task {self.task_id}")

            # Use provided dataset path from Vertex AI service
            if not self.dataset_path:
                raise ValueError("No dataset path provided for YOLO training")

            logging.info(f"📂 Using dataset path: {self.dataset_path}")

            # Download dataset from GCS
            dataset_local_path = self._download_dataset()

            # Configure YOLO training
            model_size = getattr(self.config, 'model_size', 's')
            epochs = getattr(self.config, 'epochs', 100)
            batch_size = getattr(self.config, 'batch_size', 16)
            learning_rate = getattr(self.config, 'learning_rate', 0.001)

            # Determine if this is detection or segmentation
            model_type = getattr(self.config, 'model_type', 'detection')
            is_segmentation = model_type in ['segmentation', 'segment']
            model_name = f"yolov8{model_size}-seg.pt" if is_segmentation else f"yolov8{model_size}.pt"

            logging.info(f"📦 Using model: {model_name}")
            logging.info(f"🎯 Epochs: {epochs}, Batch size: {batch_size}")

            # Find data.yaml file
            data_yaml_path = self._find_data_yaml(dataset_local_path)
            if not data_yaml_path:
                raise FileNotFoundError("data.yaml not found in dataset")

            # Run YOLO training using ultralytics CLI
            train_cmd = [
                'yolo',
                'segment' if is_segmentation else 'detect',
                'train',
                f'model={model_name}',
                f'data={data_yaml_path}',
                f'epochs={epochs}',
                f'batch={batch_size}',
                f'lr0={learning_rate}',
                f'project={self.model_dir}',
                f'name=training_{self.task_id}',
                'save=True',
                'plots=True'
            ]

            logging.info(f"🏃 Running command: {' '.join(train_cmd)}")

            # Execute training
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                cwd=self.model_dir
            )

            if result.returncode != 0:
                logging.error(f"❌ Training failed: {result.stderr}")
                raise RuntimeError(f"YOLO training failed: {result.stderr}")

            logging.info(f"✅ Training completed successfully")
            logging.info(f"📊 Training output: {result.stdout[-500:]}")  # Last 500 chars

            # Parse training results
            results_dir = os.path.join(self.model_dir, f"training_{self.task_id}")
            metrics = self._parse_training_results(results_dir)

            return {
                'status': 'completed',
                'metrics': metrics,
                'model_path': os.path.join(results_dir, 'weights', 'best.pt'),
                'results_dir': results_dir
            }

        except Exception as e:
            logging.error(f"❌ Training failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'metrics': {}
            }

    def _download_dataset(self) -> str:
        """Download dataset from GCS to local storage"""
        local_dataset_path = f"/tmp/dataset_{self.task_id}"

        # Create the destination directory
        os.makedirs(local_dataset_path, exist_ok=True)

        # Use gsutil to download the dataset
        download_cmd = ['gsutil', '-m', 'cp', '-r', f"{self.dataset_path}*", local_dataset_path]

        logging.info(f"📥 Downloading dataset from {self.dataset_path}")
        result = subprocess.run(download_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to download dataset: {result.stderr}")

        logging.info(f"✅ Dataset downloaded to {local_dataset_path}")
        return local_dataset_path

    def _find_data_yaml(self, dataset_path: str) -> str:
        """Find data.yaml file in the dataset"""
        for root, dirs, files in os.walk(dataset_path):
            if 'data.yaml' in files:
                yaml_path = os.path.join(root, 'data.yaml')
                logging.info(f"📋 Found data.yaml at: {yaml_path}")
                return yaml_path
        return None

    def _parse_training_results(self, results_dir: str) -> Dict[str, Any]:
        """Parse YOLO training results"""
        try:
            # Look for results.csv or results.txt
            results_file = os.path.join(results_dir, 'results.csv')
            if not os.path.exists(results_file):
                results_file = os.path.join(results_dir, 'results.txt')

            metrics = {
                'final_epoch': getattr(self.config, 'epochs', 100),
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }

            if os.path.exists(results_file):
                logging.info(f"📊 Parsing results from: {results_file}")
                # Basic metrics parsing - can be enhanced
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Skip header
                        last_line = lines[-1].strip()
                        # Parse CSV values (simplified)
                        values = last_line.split(',')
                        if len(values) >= 4:
                            metrics.update({
                                'train_loss': float(values[2]) if values[2] != '' else 0.0,
                                'val_loss': float(values[3]) if values[3] != '' else 0.0,
                            })

            return metrics

        except Exception as e:
            logging.warning(f"⚠️ Could not parse training results: {e}")
            return {
                'status': 'completed',
                'final_epoch': getattr(self.config, 'epochs', 100),
                'completed_at': datetime.now().isoformat()
            }