#!/usr/bin/env python3
"""
YOLOv8 PyTorch to ONNX Converter
This script converts a YOLOv8 PyTorch model to ONNX format for use with the Vision Software.
"""

import os
import sys
from pathlib import Path

def convert_yolo_to_onnx():
    """Convert YOLOv8 PyTorch model to ONNX format"""
    
    # Check if ultralytics is available
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO library found")
    except ImportError:
        print("❌ Error: ultralytics library not found")
        print("Please install it with: pip install ultralytics")
        return False
    
    # Path to the original PyTorch model
    pytorch_model_path = "../pytorch_package/model.pt"
    onnx_model_path = "model.onnx"
    
    if not os.path.exists(pytorch_model_path):
        print(f"❌ Error: PyTorch model not found at {pytorch_model_path}")
        return False
    
    try:
        print("🔄 Loading YOLOv8 PyTorch model...")
        model = YOLO(pytorch_model_path)
        
        print("🔄 Converting to ONNX format...")
        # Export to ONNX with standard settings
        model.export(
            format="onnx",
            imgsz=640,  # Input image size
            dynamic=False,  # Static input size for better compatibility
            simplify=True,  # Simplify the ONNX model
            opset=11  # ONNX opset version for better compatibility
        )
        
        # The export typically creates model.onnx in the same directory as the .pt file
        # We need to move it to our onnx_package directory
        original_onnx = "../pytorch_package/model.onnx"
        if os.path.exists(original_onnx):
            import shutil
            shutil.move(original_onnx, onnx_model_path)
            print(f"✅ ONNX model created successfully: {onnx_model_path}")
        else:
            print("❌ Error: ONNX export completed but file not found")
            return False
        
        # Verify the ONNX model
        print("🔍 Verifying ONNX model...")
        try:
            import onnx
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model verification passed")
            
            # Print model info
            print(f"📊 Model info:")
            print(f"   - Input shape: {[dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
            print(f"   - Output nodes: {len(onnx_model.graph.output)}")
            
        except ImportError:
            print("⚠️  Warning: onnx library not available for verification")
        except Exception as e:
            print(f"⚠️  Warning: ONNX verification failed: {e}")
        
        print("\n🎉 Conversion completed successfully!")
        print(f"   Original PyTorch model: {pytorch_model_path}")
        print(f"   New ONNX model: {onnx_model_path}")
        print(f"   ONNX package ready in: {os.getcwd()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def main():
    """Main function"""
    print("🚀 YOLOv8 PyTorch to ONNX Converter")
    print("=" * 50)
    
    # Change to the onnx_package directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📂 Working directory: {os.getcwd()}")
    
    if convert_yolo_to_onnx():
        print("\n✅ All done! You can now:")
        print("   1. Zip the onnx_package folder")
        print("   2. Upload it to your Vision Software")
        print("   3. Activate the model (should work without errors)")
    else:
        print("\n❌ Conversion failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
