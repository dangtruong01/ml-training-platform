import os
import torch
import numpy as np
from PIL import Image
from typing import List
from anomalib.models import Patchcore
from anomalib.post_processing import PostProcessor
from lightning import Trainer
from anomalib.data import Folder

class AnomalyService:
    def __init__(self):
        self.model_dir = os.path.abspath(os.path.join("ml", "models", "anomaly"))
        os.makedirs(self.model_dir, exist_ok=True)

    def train_patchcore(self, good_images: List[str], project_name: str) -> str:
        """
        Train a PatchCore model on a set of 'good' images.
        """
        try:
            # Create directories
            project_dir = os.path.join(self.model_dir, project_name)
            os.makedirs(project_dir, exist_ok=True)
            
            # Create data module with the good images
            # For anomaly detection, we only need 'normal' images
            
            # Create temporary directory structure for anomalib
            temp_data_dir = os.path.join(project_dir, "temp_data")
            normal_dir = os.path.join(temp_data_dir, "normal")
            os.makedirs(normal_dir, exist_ok=True)
            
            # Copy images to expected structure
            import shutil
            for i, img_path in enumerate(good_images):
                if os.path.exists(img_path):
                    dest_path = os.path.join(normal_dir, f"normal_{i}.jpg")
                    shutil.copy2(img_path, dest_path)
            
            # Create data module
            data_module = Folder(
                name=project_name,
                root=temp_data_dir,
                normal_dir="normal",
                abnormal_dir=None,  # We don't have abnormal examples
                train_batch_size=1,
                eval_batch_size=1,
            )
            
            # Configure the PatchCore model
            model = Patchcore(
                # input_size=(256, 256),
                layers=["layer1", "layer2", "layer3"],
                backbone="resnet18",
            )

            # Configure the trainer with more epochs for proper memory bank population
            trainer = Trainer(
                default_root_dir=project_dir,
                max_epochs=5,  # Increase epochs to ensure proper training
                accelerator="cpu",  # Force CPU for compatibility
                devices=1,
                logger=False,  # Disable logging
                enable_checkpointing=True,
                enable_progress_bar=False,
                num_sanity_val_steps=0,  # Skip sanity validation that causes the error
            )

            # Setup data
            data_module.setup()

            # Train the model
            trainer.fit(model, datamodule=data_module)

            # Save the trained model
            model_path = os.path.join(project_dir, "patchcore_model.ckpt")
            trainer.save_checkpoint(model_path)
            
            # Clean up temp directory
            shutil.rmtree(temp_data_dir, ignore_errors=True)

            return model_path
            
        except Exception as e:
            print(f"❌ Error training PatchCore model: {e}")
            raise

    def predict_anomaly(self, image_path: str, model_path: str) -> dict:
        """
        Predict anomalies in a new image using a trained PatchCore model.
        """
        try:
            # Load the trained model
            model = Patchcore.load_from_checkpoint(model_path)
            model.eval()
            
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            
            # Resize image to model input size
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Process image
            processed_image = transform(image_np).unsqueeze(0)  # Add batch dimension

            # Get the prediction
            with torch.no_grad():
                prediction = model(processed_image)

            # Extract results
            anomaly_map = prediction["anomaly_map"].squeeze().cpu().numpy()
            pred_score = prediction["pred_score"].cpu().item()
            
            # Create simple visualization (without using deprecated Visualizer)
            import cv2
            
            # Resize anomaly map to original image size
            original_size = image.size  # (width, height)
            anomaly_map_resized = cv2.resize(anomaly_map, original_size)
            
            # Create heatmap overlay
            heatmap = cv2.applyColorMap((anomaly_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Blend with original image
            alpha = 0.4
            annotated_image = cv2.addWeighted(image_np, 1 - alpha, heatmap_rgb, alpha, 0)

            return {
                "anomaly_map": anomaly_map,
                "pred_score": pred_score,
                "annotated_image": annotated_image,
            }
            
        except Exception as e:
            print(f"❌ Error predicting anomaly: {e}")
            raise

# Create a global instance
anomaly_service = AnomalyService()