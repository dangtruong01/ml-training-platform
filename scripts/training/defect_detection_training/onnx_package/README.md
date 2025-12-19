# YOLOv8 ONNX Model Package

This package contains a YOLOv8 object detection model converted to ONNX format for use with the Vision Software.

## 📁 Package Contents

- `model.onnx` - YOLOv8 model in ONNX format (to be generated)
- `config.json` - Model configuration and metadata
- `classes.txt` - List of detectable classes (dent, dirt, scratch)
- `requirements.txt` - Python dependencies for model conversion
- `model_arch.py` - Original model architecture reference
- `convert_to_onnx.py` - Conversion script
- `README.md` - This file

## 🔄 Converting the Model

To convert the PyTorch model to ONNX format:

1. **Install dependencies:**
   ```bash
   pip install ultralytics onnx onnxruntime
   ```

2. **Run the conversion script:**
   ```bash
   cd onnx_package
   python convert_to_onnx.py
   ```

3. **Verify the conversion:**
   The script will create `model.onnx` and verify it's working correctly.

## 📦 Creating the Upload Package

After conversion, create a zip file:

```bash
# From the Vision Software root directory
cd onnx_package
zip -r ../onnx_model_package.zip .
```

## 🚀 Using in Vision Software

1. Upload `onnx_model_package.zip` through the Settings → Model Tab
2. The system will automatically extract and detect it as an ONNX model
3. Activate the model - it should work without LibTorch issues

## ✅ Advantages of ONNX Format

- ✅ Better compatibility with the Vision Software
- ✅ No LibTorch dependency issues
- ✅ Faster inference on CPU
- ✅ Cross-platform support
- ✅ Smaller memory footprint

## 🔧 Model Configuration

The model is configured for:
- **Input size:** 640x640 RGB images
- **Classes:** dent, dirt, scratch (3 classes)
- **Confidence threshold:** 0.25 (adjustable in software)
- **IoU threshold:** 0.45
- **Framework:** YOLOv8s (small version)

## 📊 Performance Metrics

- **mAP@0.5:** 31.12%
- **mAP@0.5:0.95:** 13.96%

## 🔍 Troubleshooting

If conversion fails:

1. **Check PyTorch model:** Ensure `../pytorch_package/model.pt` exists
2. **Install dependencies:** Run `pip install ultralytics onnx onnxruntime`
3. **Check Python version:** Requires Python 3.8+
4. **Memory:** Ensure sufficient RAM for model loading

## 📝 Notes

- Original PyTorch model remains unchanged in `../pytorch_package/`
- ONNX conversion is one-way (PyTorch → ONNX)
- The ONNX model should be functionally identical to the PyTorch version
- Inference speed may be faster with ONNX on CPU

---

*Created: October 21, 2025*
*Converted from PyTorch YOLOv8 model*
