import os
import shutil
from ultralytics import YOLO
from pathlib import Path
import time

def main():
    """
    Downloads the required YOLOv8 models using the ultralytics library
    and places them in the correct project directory.
    """
    # The directory to save the models in, relative to this script's location
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be saved in: {models_dir}")

    # Focus on detection models first
    detection_models = {
        "yolov8n.pt": "Nano - Fastest, smallest (6.2MB) - good for testing",
        "yolov8s.pt": "Small - Balanced speed/accuracy (21.5MB) - recommended starter", 
        "yolov8m.pt": "Medium - Better accuracy (49.7MB) - production ready",
        "yolov8l.pt": "Large - High accuracy (83.7MB) - slower but more accurate",
        "yolov8x.pt": "Extra-large - Highest accuracy (136.7MB) - best results"
    }

    print("=== Object Detection Models Available ===")
    print("Choose which model(s) to download:\n")
    
    # Display available models
    model_list = list(detection_models.keys())
    for i, (model_name, description) in enumerate(detection_models.items(), 1):
        status = "✓ Downloaded" if os.path.exists(os.path.join(models_dir, model_name)) else "  Available"
        print(f"{i}. {status} {model_name} - {description}")
    
    print("\nOptions:")
    print("• Enter numbers (e.g., '1,3' for nano and medium)")
    print("• Enter model names (e.g., 'yolov8s.pt,yolov8m.pt')")
    print("• Enter 'all' to download all detection models")
    print("• Enter 'recommended' for yolov8s.pt and yolov8m.pt")
    print("• Enter 'quit' to exit")
    
    user_input = input("\nYour choice: ").strip().lower()
    
    if user_input == 'quit':
        print("Exiting...")
        return
    
    models_to_download = []
    
    if user_input == 'all':
        models_to_download = list(detection_models.keys())
    elif user_input == 'recommended':
        models_to_download = ['yolov8s.pt', 'yolov8m.pt']
    elif user_input.replace(',', '').replace(' ', '').isdigit():
        # Handle number selection
        numbers = [int(x.strip()) for x in user_input.split(',') if x.strip().isdigit()]
        for num in numbers:
            if 1 <= num <= len(model_list):
                models_to_download.append(model_list[num-1])
            else:
                print(f"Invalid number: {num}")
    else:
        # Handle model name selection
        requested_models = [name.strip() for name in user_input.split(',')]
        for model_name in requested_models:
            if model_name in detection_models:
                models_to_download.append(model_name)
            else:
                print(f"Model '{model_name}' not found. Available: {list(detection_models.keys())}")
    
    if not models_to_download:
        print("No valid models selected.")
        return
    
    print(f"\n=== Downloading {len(models_to_download)} model(s) ===")
    successful_downloads = []
    
    for model_name in models_to_download:
        description = detection_models[model_name]
        if download_model(model_name, models_dir, description):
            successful_downloads.append(model_name)
    
    print(f"\n✅ Successfully downloaded {len(successful_downloads)}/{len(models_to_download)} models")
    if successful_downloads:
        print("Downloaded models:", ", ".join(successful_downloads))
    
    # Ask if user wants to download more
    if len(successful_downloads) > 0:
        print("\nWould you like to download more models? (y/n)")
        if input().strip().lower() == 'y':
            main()  # Restart the selection process

def clear_cache(model_name):
    """Clear cached model files that might be corrupted"""
    try:
        # Get ultralytics cache directory
        cache_dir = Path.home() / '.cache' / 'ultralytics'
        
        # Look for the specific model file in cache
        for cache_file in cache_dir.rglob(model_name):
            if cache_file.exists():
                print(f"🗑️  Clearing cached file: {cache_file}")
                cache_file.unlink()
                
        # Also check torch hub cache
        torch_cache = Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics_yolov8_main'
        if torch_cache.exists():
            for cache_file in torch_cache.rglob(model_name):
                if cache_file.exists():
                    print(f"🗑️  Clearing torch cache: {cache_file}")
                    cache_file.unlink()
                
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")

def download_model(model_name, models_dir, description="", max_retries=3):
    """Download a single model with retry logic and cleanup"""
    dest_path = os.path.join(models_dir, model_name)

    if os.path.exists(dest_path):
        print(f"✓ Model '{model_name}' already exists. Skipping.")
        return True

    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading '{model_name}'... (Attempt {attempt + 1}/{max_retries})")
            print(f"    {description}")
            
            # Clear any cached corrupted files first
            if attempt > 0:  # Only clear cache on retry attempts
                clear_cache(model_name)
                time.sleep(2)  # Wait a moment after clearing cache
            
            # Instantiating YOLO will download the model to a cache directory
            model = YOLO(model_name)
            
            # The ultralytics library saves the model in a cache.
            # We get the path to the cached model file.
            cached_model_path = model.ckpt_path
            
            # Verify the downloaded file is valid
            if not os.path.exists(cached_model_path):
                raise FileNotFoundError(f"Cached model file not found: {cached_model_path}")
            
            # Check file size (should be > 1MB for any YOLO model)
            file_size = os.path.getsize(cached_model_path)
            if file_size < 1024 * 1024:  # Less than 1MB
                raise ValueError(f"Downloaded file seems corrupted (size: {file_size} bytes)")

            # Copy the model from the cache to our project's model directory
            shutil.copy(cached_model_path, dest_path)
            
            # Verify the copied file
            if os.path.exists(dest_path) and os.path.getsize(dest_path) == file_size:
                print(f"✅ Successfully downloaded '{model_name}' ({file_size / (1024*1024):.1f} MB)")
                return True
            else:
                raise ValueError("File copy verification failed")

        except Exception as e:
            print(f"❌ Error downloading '{model_name}' (attempt {attempt + 1}): {e}")
            
            # Clean up any partial downloads
            if os.path.exists(dest_path):
                os.remove(dest_path)
            
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"💀 Failed to download '{model_name}' after {max_retries} attempts")
                return False

    return False

if __name__ == "__main__":
    print("🤖 YOLOv8 Object Detection Model Downloader")
    print("=" * 45)
    print("This script downloads YOLOv8 detection models only.")
    print("Perfect for getting started with object detection!\n")
    
    main()
    
    print("\n=== Model Usage Tips ===")
    print("• yolov8n.pt: Great for testing and development")
    print("• yolov8s.pt: Best balance of speed and accuracy")
    print("• yolov8m.pt: Recommended for production systems")
    print("• yolov8l.pt & yolov8x.pt: When you need maximum accuracy")