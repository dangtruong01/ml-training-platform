import argparse
import os
import sys
from ultralytics import YOLO
import yaml

def main(data_config, epochs=10, imgsz=640, project=None, name="yolo_segmentation_model", device="cpu"):
    """
    Train a YOLO segmentation model
    
    Args:
        data_config: Path to data.yaml file
        epochs: Number of training epochs
        imgsz: Image size for training
        project: Project directory for results
        name: Name for this training run
    """
    try:
        print(f"🚀 Starting YOLO segmentation training...")
        print(f"📊 Dataset: {data_config}")
        print(f"⚙️  Epochs: {epochs}, Image size: {imgsz}")
        
        # Validate data config exists
        if not os.path.exists(data_config):
            raise FileNotFoundError(f"Data config not found: {data_config}")
        
        # Load and validate data.yaml
        with open(data_config, 'r') as f:
            data_dict = yaml.safe_load(f)
        
        print(f"📋 Dataset info:")
        print(f"   Classes: {data_dict.get('nc', 'Unknown')} - {data_dict.get('names', 'Unknown')}")
        print(f"   Train: {data_dict.get('train', 'Unknown')}")
        print(f"   Val: {data_dict.get('val', 'Unknown')}")
        
        # Initialize YOLO segmentation model with pre-trained weights
        model_path = "yolov8n-seg.pt"  # Start with nano segmentation model
        
        # Try to find the model in the models directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        local_model_path = os.path.join(project_root, "ml", "models", model_path)
        
        if os.path.exists(local_model_path):
            model_path = local_model_path
            print(f"📦 Using local model: {model_path}")
        else:
            print(f"📦 Using model: {model_path} (will download if needed)")
        
        model = YOLO(model_path)
        
        # Auto-detect best device if auto is specified
        if device == 'auto':
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 Using GPU acceleration (CUDA)")
            else:
                device = 'cpu'
                print(f"⚠️  No GPU detected, using CPU")
        
        print(f"🖥️  Training device: {device}")
        
        # Set up training parameters
        train_kwargs = {
            'data': data_config,
            'epochs': epochs,
            'imgsz': imgsz,
            'verbose': True,
            'save': True,
            'plots': True,
            'device': device,
        }
        
        # Add project and name if specified
        if project:
            train_kwargs['project'] = project
        if name:
            train_kwargs['name'] = name
        
        print(f"🔥 Starting segmentation training with parameters: {train_kwargs}")
        
        # Start training
        results = model.train(**train_kwargs)
        
        print(f"✅ Segmentation training completed successfully!")
        
        # Print results summary
        if results:
            print(f"📈 Training results:")
            print(f"   Best weights: {results.save_dir}")
            
        return results
        
    except Exception as e:
        print(f"❌ Segmentation training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model")
    parser.add_argument("--data", type=str, required=True, help="Path to the data config file (data.yaml)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--project", type=str, help="Project directory for results")
    parser.add_argument("--name", type=str, default="yolo_segmentation_model", help="Name for this training run")
    parser.add_argument("--device", type=str, default="cpu", help="Training device: 'cpu', 'cuda', 'auto'")
    
    args = parser.parse_args()
    
    main(args.data, args.epochs, args.imgsz, args.project, args.name, args.device)