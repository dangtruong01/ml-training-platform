from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import os
import glob
import time
import uuid
from datetime import datetime
import requests
import torch
import tempfile
import zipfile

try:
    from backend.services.ml.yolo_service import yolo_service
    from backend.services.core.database_service import database_service
    from backend.services import get_service
except ImportError:
    from services.ml.yolo_service import yolo_service
    from services.core.database_service import database_service
    from services import get_service

router = APIRouter()


def _get_model_type_from_id(model_id: str) -> str:
    """Extract model type from task ID"""
    if model_id.startswith('anomaly_training_'):
        return 'anomaly'
    elif model_id.startswith('detection_training_'):
        return 'detection'
    elif model_id.startswith('segmentation_training_'):
        return 'segmentation'
    else:
        return 'other'

def _convert_pytorch_to_onnx(model_path: str, output_path: str, model_type: str = 'detection') -> bool:
    """
    Convert PyTorch model to ONNX format

    Args:
        model_path: Path to the PyTorch model (.pt file)
        output_path: Path where ONNX model should be saved
        model_type: Type of model (detection, segmentation, anomaly)

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        print(f"🔄 Converting {model_path} to ONNX format...")

        if model_type in ['detection', 'segmentation'] and model_path.endswith('.pt'):
            # For YOLO models, use ultralytics export capability
            from ultralytics import YOLO

            # Load the YOLO model
            model = YOLO(model_path)

            # Export to ONNX
            # Ultralytics handles the input size and other details automatically
            success = model.export(format='onnx', imgsz=640)

            if success:
                # The exported file will have .onnx extension in the same directory
                onnx_path = model_path.replace('.pt', '.onnx')
                if os.path.exists(onnx_path):
                    # Move to desired output path
                    import shutil
                    shutil.move(onnx_path, output_path)
                    print(f"✅ ONNX export successful: {output_path}")
                    return True
                else:
                    print(f"❌ ONNX file not found after export: {onnx_path}")
                    return False
            else:
                print(f"❌ YOLO export failed")
                return False

        elif model_type == 'anomaly' and model_path.endswith('.pth'):
            # For PyTorch anomaly models, use direct torch.onnx.export
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load the model
            model = torch.load(model_path, map_location=device)
            model.eval()

            # Create dummy input (adjust size based on your model's expected input)
            dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Adjust dimensions as needed

            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"✅ ONNX export successful: {output_path}")
            return True

        else:
            print(f"⚠️ Unsupported model format or type: {model_path} ({model_type})")
            return False

    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def _get_model_files_from_gcs(model_id: str) -> list:
    """Download model files from GCS with improved search logic"""
    try:
        from google.cloud import storage
        import tempfile

        storage_client = storage.Client()
        model_files = []

        # Try multiple possible GCS locations for the model
        search_locations = [
            ('mltraining-models', _get_model_type_from_id(model_id), model_id),  # Standard location
            ('mltraining-vertex-staging', _get_model_type_from_id(model_id), model_id),  # Vertex AI staging
            ('mltraining-models', 'models', model_id),  # Alternative structure
            ('mltraining-vertex-staging', 'models', model_id),  # Alternative staging structure
        ]

        for bucket_name, folder1, folder2 in search_locations:
            try:
                bucket = storage_client.bucket(bucket_name)
                gcs_prefix = f"{folder1}/{folder2}/"

                print(f"🔍 Searching gs://{bucket_name}/{gcs_prefix}")

                blobs = list(bucket.list_blobs(prefix=gcs_prefix))
                if blobs:
                    print(f"📁 Found {len(blobs)} objects in gs://{bucket_name}/{gcs_prefix}")

                    for blob in blobs:
                        if not blob.name.endswith('/') and not blob.name.endswith('.tmp'):  # Skip directories and temp files
                            filename = os.path.basename(blob.name)

                            # Skip if it's obviously not a model file
                            if filename.lower() in ['readme.txt', 'log.txt', 'config.yaml', '.gitkeep']:
                                continue

                            # Download to temporary file with proper naming
                            file_extension = os.path.splitext(filename)[1] or '.bin'
                            temp_file = tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=file_extension,
                                prefix=f"{model_id}_"
                            )

                            print(f"📥 Downloading {blob.name} ({blob.size} bytes)")
                            blob.download_to_filename(temp_file.name)

                            model_files.append({
                                'filename': filename,
                                'filepath': temp_file.name,
                                'gcs_path': f"gs://{bucket_name}/{blob.name}",
                                'file_size': blob.size,
                                'created_at': blob.time_created.isoformat() if blob.time_created else None,
                                'model_format': _detect_model_format(filename)
                            })

                            print(f"✅ Downloaded {filename} to {temp_file.name}")

                    if model_files:
                        print(f"🎯 Found {len(model_files)} model files in {bucket_name}")
                        break  # Stop searching once we find files

            except Exception as bucket_error:
                print(f"⚠️ Could not access gs://{bucket_name}: {bucket_error}")
                continue

        if not model_files:
            print(f"❌ No model files found in any GCS location for {model_id}")

        return model_files

    except Exception as e:
        print(f"❌ Error downloading from GCS: {e}")
        import traceback
        traceback.print_exc()
        return []

def _detect_model_format(filename: str) -> str:
    """Detect model format from filename"""
    ext = os.path.splitext(filename)[1].lower()

    format_mapping = {
        '.pkl': 'sklearn_pickle',
        '.pt': 'pytorch',
        '.pth': 'pytorch',
        '.onnx': 'onnx',
        '.h5': 'keras_h5',
        '.json': 'tensorflow_json',
        '.pb': 'tensorflow_pb',
        '.ckpt': 'tensorflow_checkpoint',
        '.bin': 'huggingface_transformers'
    }

    return format_mapping.get(ext, 'unknown')

def _generate_model_readme(model_id: str, algorithm: str, training_job: dict, model_files: list, files_added: int) -> str:
    """Generate a comprehensive README for the model download"""
    from datetime import datetime

    # Get algorithm-specific information
    algorithm_info = _get_algorithm_description(algorithm) if algorithm != 'unknown' else 'No algorithm information available'

    # Determine model format based on algorithm
    format_mapping = {
        'isolation_forest': 'Scikit-learn Pickle (.pkl)',
        'one_class_svm': 'Scikit-learn Pickle (.pkl)',
        'local_outlier_factor': 'Scikit-learn Pickle (.pkl)',
        'autoencoder': 'PyTorch (.pth)',
        'yolo_v8': 'Ultralytics YOLO (.pt)',
        'yolo_v11': 'Ultralytics YOLO (.pt)',
        'rtdetr': 'PyTorch (.pt)',
        'yolo_v8_seg': 'Ultralytics YOLO (.pt)',
        'sam2': 'PyTorch (.pth)',
        'unet': 'PyTorch (.pth)'
    }

    model_format = format_mapping.get(algorithm, 'Unknown format')

    readme_content = f"""# Model Download: {model_id}

## Model Information
- **Algorithm**: {algorithm}
- **Model Format**: {model_format}
- **Model ID**: {model_id}
- **Files Included**: {files_added} files

## Training Details
"""

    if training_job:
        readme_content += f"""- **Project ID**: {training_job.get('project_id', 'Unknown')}
- **Model Type**: {training_job.get('model_type', 'Unknown')}
- **Training Status**: {training_job.get('status', 'Unknown')}
- **Progress**: {training_job.get('progress', 0)}%
- **Started**: {training_job.get('started_at', 'Unknown')}
- **Completed**: {training_job.get('completed_at', 'Unknown')}

## Training Configuration
"""
        training_config = training_job.get('training_config', {})
        if training_config:
            for key, value in training_config.items():
                if key not in ['vertex_ai_job_id', 'vertex_ai_job_name']:  # Skip technical details
                    readme_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
    else:
        readme_content += "- No training information available\n"

    readme_content += f"""

## Algorithm Details
{algorithm_info}

## Files in This Archive
"""

    # List all files in the archive
    for file_info in model_files:
        filename = file_info.get('filename', 'unknown')
        file_size = file_info.get('file_size', 0)
        model_format = file_info.get('model_format', _detect_model_format(filename))

        if file_size > 0:
            size_mb = file_size / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_size} bytes"
        else:
            size_str = "unknown size"

        readme_content += f"- **{filename}** ({model_format}) - {size_str}\n"

    # Add usage instructions based on algorithm
    readme_content += f"""

## Usage Instructions
"""

    if algorithm in ['isolation_forest', 'one_class_svm', 'local_outlier_factor']:
        # Get the actual sklearn version from model files or training config
        sklearn_version = 'unknown'
        for file_info in model_files:
            if file_info.get('sklearn_version'):
                sklearn_version = file_info['sklearn_version']
                break
        if sklearn_version == 'unknown' and training_job:
            sklearn_version = training_job.get('training_config', {}).get('sklearn_version', '1.6.1 (legacy)')

        readme_content += f"""
### Scikit-learn Model Usage

**Version-Safe Loading (Recommended):**
```python
import pickle
import warnings
import sklearn
import numpy as np

# Load the model with version compatibility handling
with open('{algorithm}_model.pkl', 'rb') as f:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = pickle.load(f)

        # Handle both new and legacy model formats
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            training_version = data.get('sklearn_version', 'unknown')
            print(f"Model trained with sklearn {{training_version}}, running on {{sklearn.__version__}}")
        else:
            model = data
            print("Legacy model format detected")

        # Check for version warnings
        version_warnings = [warn for warn in w if 'InconsistentVersionWarning' in str(warn.category)]
        if version_warnings:
            print("⚠️  Version compatibility warnings detected - model should still work")

# Use for prediction
# X should be your feature array (same format as training data)
predictions = model.predict(X)  # 1 for normal, -1 for anomaly
scores = model.decision_function(X)  # Anomaly scores (lower = more anomalous)
```

**Simple Loading (if you don't care about warnings):**
```python
import pickle
import numpy as np

with open('{algorithm}_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model'] if isinstance(data, dict) and 'model' in data else data

predictions = model.predict(X)  # 1 for normal, -1 for anomaly
```

**Model Information:**
- **Training sklearn version**: {sklearn_version}
- **Compatible with**: sklearn 1.5.x, 1.6.x, 1.7.x (with warnings)
- **Model type**: {algorithm.replace('_', ' ').title()}

**Version Compatibility:**
- ✅ **sklearn 1.7.x**: Works with compatibility warnings
- ✅ **sklearn 1.6.x**: Full compatibility
- ⚠️  **sklearn 1.5.x**: May work but not tested
- ❌ **sklearn 1.4.x and below**: Not recommended

**For Production Use:**
1. Test the model thoroughly with your current sklearn version
2. Consider retraining with sklearn 1.7.1 for full compatibility
3. Use virtual environments to maintain consistent versions
"""

    elif algorithm == 'autoencoder':
        readme_content += f"""
### PyTorch Autoencoder Usage
```python
import torch

# Load the model
model = torch.load('autoencoder_model.pth')
model.eval()

# Use for anomaly detection
with torch.no_grad():
    reconstructed = model(input_tensor)
    reconstruction_error = torch.mean((input_tensor - reconstructed) ** 2, dim=1)
    # Higher error = more likely to be anomaly
```
"""

    elif algorithm.startswith('yolo'):
        readme_content += f"""
### YOLO Model Usage
```python
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')  # or your specific model file

# Run inference
results = model('path/to/image.jpg')
results[0].show()  # Display results
```

**ONNX Usage** (if ONNX files are included):
```python
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
# Prepare your input data
outputs = session.run(None, {{'input': input_data}})
```
"""

    readme_content += f"""

## Deployment Notes
- For **production deployment**, consider using ONNX models if available for better cross-platform compatibility
- For **development and fine-tuning**, use the original model format
- Ensure you have the correct dependencies installed for your chosen format

## Download Information
- **Downloaded**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Generated by**: OpenTrainer Model Management System
- **Archive Format**: ZIP with models and documentation

## Support
For questions about this model or deployment assistance, refer to the OpenTrainer documentation.
"""

    return readme_content

def _get_download_source(training_job_result: dict) -> str:
    """Determine where to download model files from"""
    if training_job_result['status'] != 'success':
        return 'local'  # Default to local filesystem

    training_job = training_job_result['training_job']
    training_config = training_job.get('training_config', {})

    # Check if this was a Vertex AI training job
    vertex_ai_job_id = training_config.get('vertex_ai_job_id')
    if vertex_ai_job_id:
        return 'gcs'  # Vertex AI jobs store models in GCS

    # Check for device preference - GPU usually means cloud training
    device = training_config.get('device', 'cpu')
    if device in ['cuda', 'gpu']:
        return 'gcs'

    return 'local'  # Default to local filesystem

@router.get("/")
async def list_models():
    """List all trained models from database"""
    try:
        # Get all training jobs from database
        training_jobs_result = database_service.list_training_jobs()
        
        if training_jobs_result['status'] != 'success':
            return JSONResponse({
                'status': 'error',
                'message': 'Failed to fetch training jobs from database',
                'models': [],
                'total_models': 0
            })
        
        training_jobs = training_jobs_result.get('training_jobs', [])
        models = []
        
        for job in training_jobs:
            task_id = job['task_id']
            project_id = job.get('project_id')
            
            # Get model file paths (from filesystem or database)
            model_files = job.get('model_files_info', [])
            if not model_files:
                # Fallback to filesystem scan for backward compatibility
                model_files = yolo_service.get_model_files(task_id)
            
            # Get project info if available
            project_info = {'project_name': 'Unknown', 'project_type': 'unknown'}
            if project_id:
                try:
                    project_result = database_service.get_project(project_id)
                    if project_result['status'] == 'success':
                        project_data = project_result['project']
                        project_info = {
                            'project_name': project_data.get('project_name', 'Unknown'),
                            'project_type': project_data.get('project_type', 'unknown')
                        }
                except Exception as e:
                    print(f"Warning: Could not fetch project info for {project_id}: {e}")
            
            model_info = {
                'model_id': task_id,
                'project_id': project_id,
                'project_name': project_info.get('project_name', 'Unknown'),
                'project_type': project_info.get('project_type', 'unknown'),
                'model_type': job.get('model_type', 'yolo'),
                'algorithm': job.get('algorithm', 'unknown'),
                'status': job.get('status', 'unknown'),
                'created_at': job.get('created_at'),
                'completed_at': job.get('completed_at'),
                'started_at': job.get('started_at'),
                'duration_seconds': job.get('duration_seconds'),
                'progress': job.get('progress', 0),
                'current_epoch': job.get('current_epoch'),
                'total_epochs': job.get('total_epochs'),
                'training_config': job.get('training_config', {}),
                'training_metrics': job.get('training_metrics', {}),
                'model_files': model_files,
                'downloadable': len(model_files) > 0
            }
            
            models.append(model_info)
        
        # Sort by completion date (newest first), handling None values
        models.sort(key=lambda x: x.get('completed_at') or x.get('created_at') or '1970-01-01T00:00:00', reverse=True)
        
        return JSONResponse({
            'status': 'success',
            'models': models,
            'total_models': len(models)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed information about a specific model from database"""
    try:
        # Get training job details from database
        training_job_result = database_service.get_training_job(model_id)
        
        if training_job_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Model not found")
        
        training_job = training_job_result['training_job']
        
        # Get model files from database or filesystem
        model_files = training_job.get('model_files_info', [])
        if not model_files:
            # Fallback to filesystem scan for backward compatibility
            model_files = yolo_service.get_model_files(model_id)
        
        # Get training logs from database
        training_logs = training_job.get('training_logs', [])
        if not training_logs:
            # Fallback to filesystem logs for backward compatibility
            training_logs = yolo_service.get_training_logs(model_id, lines=100)
        
        # Get project info if available
        project_info = {}
        project_id = training_job.get('project_id')
        if project_id:
            try:
                project_result = database_service.get_project(project_id)
                if project_result['status'] == 'success':
                    project_data = project_result['project']
                    project_info = {
                        'project_name': project_data.get('project_name', 'Unknown'),
                        'project_type': project_data.get('project_type', 'unknown'),
                        'file_counts': project_data.get('file_counts', {})
                    }
            except Exception as e:
                print(f"Warning: Could not fetch project info for {project_id}: {e}")
        
        model_details = {
            'model_id': model_id,
            'project_info': project_info,
            'training_job': training_job,
            'model_files': model_files,
            'training_logs': training_logs,
            'downloadable': len(model_files) > 0
        }
        
        return JSONResponse({
            'status': 'success',
            'model': model_details
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/download")
async def download_model(model_id: str, background_tasks: BackgroundTasks):
    """Download the trained model files as a zip archive"""
    import zipfile
    import tempfile
    from pathlib import Path
    
    try:
        # Get training job from database first (with timeout protection)
        try:
            training_job_result = database_service.get_training_job(model_id)
        except Exception as db_error:
            print(f"⚠️ Database lookup failed for {model_id}, falling back to filesystem: {db_error}")
            training_job_result = {'status': 'error'}
        
        # Step 1: Get training job details and algorithm info
        model_files = []
        algorithm = 'unknown'
        training_job = None
        
        if training_job_result['status'] == 'success':
            training_job = training_job_result['training_job']
            algorithm = training_job.get('algorithm', 'unknown')
            model_files = training_job.get('model_files_info', [])
            print(f"🧠 Algorithm: {algorithm}")
            print(f"📋 Database has {len(model_files)} files listed")
        
        # Step 2: Use database files if available (most reliable)
        if model_files:
            print(f"✅ Using model files from database (algorithm-specific)")
            # For cloud storage files (gs://), skip local existence check
            verified_files = []
            for file_info in model_files:
                filepath = file_info.get('filepath', '')
                if filepath.startswith('gs://'):
                    # Trust database records for cloud storage files
                    verified_files.append(file_info)
                    print(f"☁️ Using cloud storage file: {filepath}")
                elif os.path.exists(filepath):
                    # Check local files normally
                    verified_files.append(file_info)
                    print(f"💾 Using local file: {filepath}")
                else:
                    print(f"⚠️ Database file not found on disk: {filepath}")
            
            if verified_files:
                model_files = verified_files
            else:
                print(f"❌ No database files found, falling back to filesystem scan")
                model_files = []
        
        # Step 3: Fallback to filesystem scan if no database files
        if not model_files:
            print(f"📁 Fallback: Scanning for {model_id} (algorithm: {algorithm})")

            # Determine download source
            download_source = _get_download_source(training_job_result)
            print(f"📍 Download source: {download_source}")

            if download_source == 'gcs':
                # Download from GCS (GPU/Vertex AI training)
                print(f"☁️ Attempting GCS download for Vertex AI model {model_id}")
                model_files = _get_model_files_from_gcs(model_id)

                if not model_files:
                    print(f"⚠️ No files found in GCS, falling back to local filesystem")
                    model_files = yolo_service.get_model_files(model_id)
            else:
                # Scan local filesystem (CPU training)
                print(f"💾 Scanning local filesystem for {model_id}")
                model_files = yolo_service.get_model_files(model_id)

            # Apply algorithm-specific filtering if we know the algorithm
            if model_files and algorithm != 'unknown':
                print(f"🔍 Applying {algorithm}-specific file filtering")
                original_count = len(model_files)
                model_files = _filter_files_by_algorithm(model_files, algorithm)
                filtered_count = len(model_files)
                print(f"📊 File filtering: {original_count} → {filtered_count} files")
        
        if not model_files:
            raise HTTPException(status_code=404, detail="No model files found")
        
        print(f"📦 Final model files for download ({len(model_files)}):")
        for file_info in model_files:
            filename = file_info.get('filename', 'unknown')
            filepath = file_info.get('filepath', 'unknown') 
            print(f"  - {filename} ({filepath})")
        
        # Create a temporary zip file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            zip_path = temp_zip.name
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                files_added = 0
                
                # Determine model type for ONNX conversion
                model_type = _get_model_type_from_id(model_id)

                for file_info in model_files:
                    file_path = file_info['filepath']

                    if os.path.exists(file_path):
                        # Get just the filename for the zip archive
                        filename = file_info.get('filename', os.path.basename(file_path))

                        # Add original file to zip
                        zipf.write(file_path, filename)
                        files_added += 1
                        print(f"📦 Added {filename} to zip archive")

                        # Convert to ONNX if it's a PyTorch model
                        if (filename.endswith('.pt') or filename.endswith('.pth')) and model_type in ['detection', 'segmentation', 'anomaly']:
                            try:
                                # Create temporary ONNX file
                                onnx_filename = filename.replace('.pt', '.onnx').replace('.pth', '.onnx')

                                with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as temp_onnx:
                                    temp_onnx_path = temp_onnx.name

                                # Convert to ONNX
                                if _convert_pytorch_to_onnx(file_path, temp_onnx_path, model_type):
                                    # Add ONNX file to zip
                                    zipf.write(temp_onnx_path, onnx_filename)
                                    files_added += 1
                                    print(f"📦 Added ONNX version: {onnx_filename}")

                                    # Clean up temporary file
                                    os.unlink(temp_onnx_path)
                                else:
                                    print(f"⚠️ ONNX conversion failed for {filename}")

                            except Exception as e:
                                print(f"⚠️ ONNX conversion error for {filename}: {e}")
                                # Continue without ONNX version
                
                # Add a comprehensive README file with model information
                readme_content = _generate_model_readme(
                    model_id=model_id,
                    algorithm=algorithm,
                    training_job=training_job,
                    model_files=model_files,
                    files_added=files_added
                )

                # Add README to zip
                zipf.writestr("README.txt", readme_content)
                files_added += 1
            
            if files_added == 0:
                raise HTTPException(status_code=404, detail="No accessible model files found")
            
            print(f"✅ Created zip archive with {files_added} files for model {model_id}")
            
            # Return the zip file
            def cleanup_temp_file():
                try:
                    os.unlink(zip_path)
                except:
                    pass
            
            background_tasks.add_task(cleanup_temp_file)
            
            return FileResponse(
                path=zip_path,
                filename=f"model_{model_id}.zip",
                media_type='application/zip'
            )
            
        except Exception as zip_error:
            # Clean up temp file if zip creation failed
            try:
                os.unlink(zip_path)
            except:
                pass
            raise zip_error
        
    except Exception as e:
        print(f"❌ Error downloading model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/download-direct")
async def download_model_direct(model_id: str, background_tasks: BackgroundTasks):
    """Download model files directly from filesystem as zip (bypasses database)"""
    import zipfile
    import tempfile
    import os
    
    try:
        print(f"🚀 Direct download requested for {model_id}")
        
        # Determine model directory from task_id pattern
        model_dir = None
        if model_id.startswith('anomaly_training_'):
            model_dir = f"/Users/truonghaidang/Desktop/open-trainer/backend/ml/results/anomaly/{model_id}"
        elif model_id.startswith('detection_training_'):
            model_dir = f"/Users/truonghaidang/Desktop/open-trainer/backend/ml/results/detection/{model_id}"
        elif model_id.startswith('segmentation_training_'):
            model_dir = f"/Users/truonghaidang/Desktop/open-trainer/backend/ml/results/segmentation/{model_id}"
        
        if not model_dir or not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail=f"Model directory not found: {model_id}")
        
        print(f"📁 Found model directory: {model_dir}")
        
        # Find all files in the model directory
        model_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    model_files.append({
                        'filename': file,
                        'filepath': file_path,
                        'file_size': os.path.getsize(file_path)
                    })
        
        if not model_files:
            raise HTTPException(status_code=404, detail=f"No files found in model directory: {model_id}")
        
        print(f"📦 Found {len(model_files)} files to include in zip")
        
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            zip_path = temp_zip.name
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                files_added = 0
                
                for file_info in model_files:
                    file_path = file_info['filepath']
                    filename = file_info['filename']
                    
                    # Add file to zip
                    zipf.write(file_path, filename)
                    files_added += 1
                    print(f"📦 Added {filename} to zip archive")
                
                # Add a README file
                readme_content = f"""# Model: {model_id}

## Files Included
"""
                for file_info in model_files:
                    filename = file_info['filename']
                    file_size = file_info['file_size']
                    readme_content += f"- {filename} ({file_size} bytes)\n"
                
                readme_content += f"\n## Download Information\n- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- Archive created by OpenTrainer Direct Download\n"
                
                zipf.writestr("README.txt", readme_content)
                files_added += 1
            
            print(f"✅ Created zip archive with {files_added} files for model {model_id}")
            
            # Return the zip file
            def cleanup_temp_file():
                try:
                    os.unlink(zip_path)
                    print(f"🗑️ Cleaned up temp file: {zip_path}")
                except:
                    pass
            
            background_tasks.add_task(cleanup_temp_file)
            
            return FileResponse(
                path=zip_path,
                filename=f"model_{model_id}.zip",
                media_type='application/zip'
            )
            
        except Exception as zip_error:
            # Clean up temp file if zip creation failed
            try:
                os.unlink(zip_path)
            except:
                pass
            raise zip_error
        
    except Exception as e:
        print(f"❌ Error in direct download for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/files")
async def list_model_files(model_id: str):
    """List all files associated with a trained model"""
    try:
        # Get training job from database first
        training_job_result = database_service.get_training_job(model_id)
        
        model_files = []
        if training_job_result['status'] == 'success':
            training_job = training_job_result['training_job']
            model_files = training_job.get('model_files_info', [])
        
        # Fallback to filesystem scan if no files in database
        if not model_files:
            model_files = yolo_service.get_model_files(model_id)
        
        if not model_files:
            raise HTTPException(status_code=404, detail="No model files found")
        
        return JSONResponse({
            'status': 'success',
            'model_id': model_id,
            'files': model_files,
            'total_files': len(model_files)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/download-onnx")
async def download_model_onnx(model_id: str, background_tasks: BackgroundTasks):
    """Download only ONNX versions of trained models"""
    try:
        print(f"🔄 ONNX-only download requested for {model_id}")

        # Get training job details
        training_job_result = database_service.get_training_job(model_id)
        model_type = _get_model_type_from_id(model_id)

        if model_type not in ['detection', 'segmentation']:
            raise HTTPException(status_code=400, detail=f"ONNX conversion not supported for model type: {model_type}. ONNX conversion is only available for YOLO-based object detection and segmentation models.")

        # Get model files
        model_files = []
        if training_job_result['status'] == 'success':
            training_job = training_job_result['training_job']
            model_files = training_job.get('model_files_info', [])

        if not model_files:
            # Fallback to filesystem scan
            download_source = _get_download_source(training_job_result)
            if download_source == 'gcs':
                model_files = _get_model_files_from_gcs(model_id)
            else:
                model_files = yolo_service.get_model_files(model_id)

        if not model_files:
            raise HTTPException(status_code=404, detail="No model files found")

        # Filter for PyTorch models that can be converted
        pytorch_files = []
        for f in model_files:
            filename = f.get('filename', '')
            filepath = f.get('filepath', '')

            if filename.endswith(('.pt', '.pth')):
                if filepath.startswith('gs://'):
                    # Trust database records for cloud storage files
                    pytorch_files.append(f)
                    print(f"☁️ Found PyTorch model in cloud storage: {filepath}")
                elif os.path.exists(filepath):
                    # Check local files normally
                    pytorch_files.append(f)
                    print(f"💾 Found PyTorch model locally: {filepath}")
                else:
                    print(f"⚠️ PyTorch model file not found at path: {filepath}")

        if not pytorch_files:
            available_files = [f.get('filename', 'unknown') for f in model_files]
            print(f"📄 Available files for {model_id}: {available_files}")
            raise HTTPException(status_code=404, detail=f"No accessible PyTorch models found for ONNX conversion. Available files: {available_files}")

        # Create temporary zip file for ONNX models
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            zip_path = temp_zip.name

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                files_added = 0

                for file_info in pytorch_files:
                    file_path = file_info['filepath']
                    filename = file_info.get('filename', os.path.basename(file_path))

                    # Convert to ONNX
                    onnx_filename = filename.replace('.pt', '.onnx').replace('.pth', '.onnx')

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as temp_onnx:
                        temp_onnx_path = temp_onnx.name

                    if _convert_pytorch_to_onnx(file_path, temp_onnx_path, model_type):
                        # Add ONNX file to zip
                        zipf.write(temp_onnx_path, onnx_filename)
                        files_added += 1
                        print(f"📦 Added ONNX: {onnx_filename}")

                        # Clean up temp file
                        os.unlink(temp_onnx_path)
                    else:
                        print(f"⚠️ Failed to convert {filename} to ONNX")
                        os.unlink(temp_onnx_path)

                # Add README for ONNX-only download
                readme_content = f"""# ONNX Models: {model_id}

## About ONNX Models
These are ONNX (Open Neural Network Exchange) versions of your trained models.
ONNX provides better interoperability across different frameworks and deployment platforms.

## Model Information
- Model Type: {model_type}
- Original Model ID: {model_id}
- Converted Files: {files_added}

## Usage
- Use with ONNX Runtime: pip install onnxruntime
- Use with TensorRT for NVIDIA GPUs
- Deploy with various cloud platforms that support ONNX

## Download Information
- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Archive created by OpenTrainer (ONNX-only)
"""
                zipf.writestr("README_ONNX.txt", readme_content)
                files_added += 1

            if files_added <= 1:  # Only README was added
                raise HTTPException(status_code=500, detail="No models could be converted to ONNX")

            # Return the zip file
            def cleanup():
                try:
                    os.unlink(zip_path)
                except:
                    pass

            background_tasks.add_task(cleanup)

            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename=f"{model_id}_onnx_models.zip",
                headers={"Content-Disposition": f"attachment; filename={model_id}_onnx_models.zip"}
            )

        except Exception as zip_error:
            # Clean up zip file if error occurred
            try:
                os.unlink(zip_path)
            except:
                pass
            raise zip_error

    except Exception as e:
        print(f"❌ Error in ONNX download for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/download/{filename}")
async def download_specific_model_file(model_id: str, filename: str):
    """Download a specific model file by filename"""
    try:
        # Get training job from database first
        training_job_result = database_service.get_training_job(model_id)
        
        model_files = []
        if training_job_result['status'] == 'success':
            training_job = training_job_result['training_job']
            model_files = training_job.get('model_files_info', [])
        
        # Fallback to filesystem scan if no files in database
        if not model_files:
            model_files = yolo_service.get_model_files(model_id)
        
        # Find the specific file
        target_file = None
        for file_info in model_files:
            if file_info.get('filename') == filename:
                target_file = file_info
                break
        
        if not target_file:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        file_path = target_file['filepath']
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model from database (GCS files auto-deleted after 7 days)"""
    try:
        print(f"🗑️ Deleting model: {model_id}")
        
        # Check if model exists in database
        training_job_result = database_service.get_training_job(model_id)
        if training_job_result['status'] != 'success':
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found in database")
        
        training_job = training_job_result['training_job']
        print(f"📋 Found model in database: {training_job['model_type']} - {training_job['status']}")
        
        # Delete from database
        delete_result = database_service.delete_training_job(model_id)
        
        if delete_result['status'] != 'success':
            print(f"❌ Failed to delete from database: {delete_result.get('message')}")
            raise HTTPException(status_code=500, detail="Failed to delete model from database")
        
        print(f"✅ Successfully deleted model {model_id} from database")
        print(f"📁 GCS files will be auto-deleted after 7 days via lifecycle policy")
        
        return JSONResponse({
            'status': 'success',
            'message': f'Model {model_id} deleted successfully',
            'note': 'Model removed from database. Files in GCS will be automatically cleaned up after 7 days.',
            'model_type': training_job.get('model_type'),
            'project_id': training_job.get('project_id')
        })
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"❌ Error deleting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/{model_id}/retry")
async def retry_training(model_id: str, background_tasks: BackgroundTasks):
    """Retry a failed training job with the same configuration"""
    try:
        print(f"🔄 Retry training requested for {model_id}")

        # Get the original training job details
        training_job_result = database_service.get_training_job(model_id)
        if training_job_result['status'] != 'success':
            raise HTTPException(status_code=404, detail=f"Training job {model_id} not found")

        original_job = training_job_result['training_job']

        # Check if the job is eligible for retry (failed status)
        if original_job['status'] != 'failed':
            raise HTTPException(status_code=400, detail=f"Can only retry failed jobs. Current status: {original_job['status']}")

        # Extract necessary information for retry
        project_id = original_job['project_id']
        model_type = original_job['model_type']
        algorithm = original_job['algorithm']
        training_config = original_job.get('training_config', {})

        print(f"🔄 Retrying {model_type} training for project {project_id}")
        print(f"📋 Original config: {training_config}")

        # Import the appropriate service based on model type
        if model_type == 'anomaly':
            from services.cloud.vertex_ai_service import vertex_ai_service

            # For anomaly detection, always use Vertex AI for retries
            new_task_id = f"anomaly_training_{uuid.uuid4().hex[:8]}"
            print(f"🆔 Generated new task ID: {new_task_id}")

            # Submit to Vertex AI
            result = vertex_ai_service.submit_training_job(
                task_id=new_task_id,
                project_id=project_id,
                model_type=model_type,
                algorithm=algorithm,
                training_config=training_config
            )

            if result['status'] == 'submitted':
                return JSONResponse({
                    'status': 'success',
                    'message': f'Retry training started for {model_id}',
                    'new_task_id': new_task_id,
                    'original_task_id': model_id,
                    'vertex_ai_job_id': result.get('vertex_ai_job_id')
                })
            else:
                raise HTTPException(status_code=500, detail=f"Failed to submit retry job: {result.get('message')}")

        elif model_type in ['object_detection', 'segmentation']:
            from services.ml.yolo_service import yolo_service
            import uuid

            # Use cloud training for retries if original was cloud-based
            if training_config.get('vertex_ai_job_id'):
                print("🌥️ Retrying with Vertex AI (cloud training)")
                if model_type == 'object_detection':
                    new_task_id = yolo_service.train_detection_from_project_cloud(project_id, training_config, algorithm)
                else:  # segmentation
                    new_task_id = yolo_service.train_segmentation_from_project_cloud(project_id, training_config, algorithm)
            else:
                print("🖥️ Retrying with local training")
                if model_type == 'object_detection':
                    new_task_id = yolo_service.train_detection_from_project(project_id, training_config, algorithm)
                else:  # segmentation
                    new_task_id = yolo_service.train_segmentation_from_project(project_id, training_config, algorithm)

            return JSONResponse({
                'status': 'success',
                'message': f'Retry training started for {model_id}',
                'new_task_id': new_task_id,
                'original_task_id': model_id
            })

        else:
            raise HTTPException(status_code=400, detail=f"Retry not supported for model type: {model_type}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error retrying training for {model_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retry training: {str(e)}")


@router.get("/by-project/{project_id}")
async def list_models_by_project(project_id: str):
    """List all models trained from a specific project"""
    try:
        # Get all training jobs for this project from database
        training_jobs_result = database_service.list_training_jobs(project_id=project_id)
        
        if training_jobs_result['status'] != 'success':
            return JSONResponse({
                'status': 'error',
                'project_id': project_id,
                'models': [],
                'total_models': 0,
                'message': 'Failed to fetch training jobs for project'
            })
        
        project_jobs = training_jobs_result.get('training_jobs', [])
        
        if not project_jobs:
            return JSONResponse({
                'status': 'success',
                'project_id': project_id,
                'models': [],
                'total_models': 0,
                'message': 'No models found for this project'
            })
        
        models = []
        for job in project_jobs:
            task_id = job['task_id']
            
            # Get model files from database or filesystem
            model_files = job.get('model_files_info', [])
            if not model_files:
                # Fallback to filesystem scan for backward compatibility
                model_files = yolo_service.get_model_files(task_id)
            
            model_info = {
                'model_id': task_id,
                'project_id': project_id,
                'model_type': job.get('model_type', 'yolo'),
                'algorithm': job.get('algorithm', 'unknown'),
                'status': job.get('status'),
                'created_at': job.get('created_at'),
                'completed_at': job.get('completed_at'),
                'started_at': job.get('started_at'),
                'duration_seconds': job.get('duration_seconds'),
                'progress': job.get('progress', 0),
                'current_epoch': job.get('current_epoch'),
                'total_epochs': job.get('total_epochs'),
                'training_config': job.get('training_config', {}),
                'training_metrics': job.get('training_metrics', {}),
                'model_files': model_files,
                'downloadable': len(model_files) > 0
            }
            
            models.append(model_info)
        
        # Sort by completion date (newest first), handling None values
        models.sort(key=lambda x: x.get('completed_at') or x.get('created_at') or '1970-01-01T00:00:00', reverse=True)
        
        return JSONResponse({
            'status': 'success',
            'project_id': project_id,
            'models': models,
            'total_models': len(models)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MANUAL MODEL LOADING ENDPOINTS
# =============================================================================

@router.get("/services/status")
async def get_services_status():
    """Get the loading status of all ML services"""
    try:
        # Define all available services
        available_services = [
            'guardrail_service',
            'auto_annotation_service', 
            'grounding_dino_service',
            'grounding_dino_sam2_service',
            'sam2_service',
            'roi_extraction_service',
            'anomaly_service',
            'advanced_anomaly_service', 
            'llm_clip_anomaly_service',
            'defect_detection_service',
            'dinov2_service',
            'dinov3_service'
        ]
        
        # Check which services are currently loaded
        from services import _services
        
        services_status = []
        for service_name in available_services:
            is_loaded = service_name in _services
            service_info = {
                'name': service_name,
                'display_name': service_name.replace('_', ' ').title(),
                'loaded': is_loaded,
                'description': get_service_description(service_name)
            }
            services_status.append(service_info)
        
        return JSONResponse({
            'status': 'success',
            'services': services_status,
            'total_services': len(available_services),
            'loaded_services': len(_services)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/services/{service_name}/load")
async def load_service_manually(service_name: str):
    """Manually load a specific ML service"""
    try:
        # Validate service name
        available_services = [
            'guardrail_service', 'auto_annotation_service', 'grounding_dino_service',
            'grounding_dino_sam2_service', 'sam2_service', 'roi_extraction_service',
            'anomaly_service', 'advanced_anomaly_service', 'llm_clip_anomaly_service',
            'defect_detection_service', 'dinov2_service', 'dinov3_service'
        ]
        
        if service_name not in available_services:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        # Check if already loaded
        from services import _services
        if service_name in _services:
            return JSONResponse({
                'status': 'success',
                'service': service_name,
                'message': f'Service {service_name} is already loaded',
                'already_loaded': True
            })
        
        # Load the service
        start_time = time.time()
        service = get_service(service_name)
        load_time = time.time() - start_time
        
        return JSONResponse({
            'status': 'success',
            'service': service_name,
            'message': f'Service {service_name} loaded successfully',
            'load_time_seconds': round(load_time, 2),
            'already_loaded': False
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load service: {str(e)}")

@router.post("/services/load-all")
async def load_all_services():
    """Load all ML services (warning: may take significant time and memory)"""
    try:
        available_services = [
            'guardrail_service', 'auto_annotation_service', 'grounding_dino_service',
            'grounding_dino_sam2_service', 'sam2_service', 'roi_extraction_service',
            'anomaly_service', 'advanced_anomaly_service', 'llm_clip_anomaly_service',
            'defect_detection_service', 'dinov2_service', 'dinov3_service'
        ]
        
        start_time = time.time()
        loaded_services = []
        failed_services = []
        
        for service_name in available_services:
            try:
                service_start = time.time()
                service = get_service(service_name)
                service_time = time.time() - service_start
                
                loaded_services.append({
                    'name': service_name,
                    'load_time_seconds': round(service_time, 2),
                    'status': 'loaded'
                })
            except Exception as e:
                failed_services.append({
                    'name': service_name,
                    'error': str(e),
                    'status': 'failed'
                })
        
        total_time = time.time() - start_time
        
        return JSONResponse({
            'status': 'success',
            'message': f'Loaded {len(loaded_services)}/{len(available_services)} services',
            'total_time_seconds': round(total_time, 2),
            'loaded_services': loaded_services,
            'failed_services': failed_services,
            'success_rate': f"{len(loaded_services)}/{len(available_services)}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load services: {str(e)}")

@router.delete("/services/{service_name}")
async def unload_service(service_name: str):
    """Unload a specific service to free memory"""
    try:
        from services import _services
        
        if service_name not in _services:
            return JSONResponse({
                'status': 'success',
                'service': service_name,
                'message': f'Service {service_name} was not loaded',
                'was_loaded': False
            })
        
        # Remove from loaded services cache
        del _services[service_name]
        
        return JSONResponse({
            'status': 'success',
            'service': service_name,
            'message': f'Service {service_name} unloaded successfully',
            'was_loaded': True
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload service: {str(e)}")

@router.delete("/services/unload-all")
async def unload_all_services():
    """Unload all services to free memory"""
    try:
        from services import _services
        
        unloaded_count = len(_services)
        unloaded_services = list(_services.keys())
        
        # Clear all loaded services
        _services.clear()
        
        return JSONResponse({
            'status': 'success',
            'message': f'Unloaded {unloaded_count} services',
            'unloaded_services': unloaded_services,
            'unloaded_count': unloaded_count
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload services: {str(e)}")

def _filter_files_by_algorithm(model_files: list, algorithm: str) -> list:
    """Filter model files based on algorithm to get correct file format"""
    if not model_files or algorithm == 'unknown':
        return model_files
    
    # Algorithm to file extension mapping
    algorithm_extensions = {
        'isolation_forest': ['.pkl'],
        'one_class_svm': ['.pkl'], 
        'local_outlier_factor': ['.pkl'],
        'autoencoder': ['.pth'],
        'yolo_v8': ['.pt'],
        'yolo_v11': ['.pt'],
        'rtdetr': ['.pt'],
        'yolo_v8_seg': ['.pt'],
        'sam2': ['.pth', '.ckpt'],
        'unet': ['.pth']
    }
    
    expected_extensions = algorithm_extensions.get(algorithm, [])
    if not expected_extensions:
        print(f"⚠️ No known file extensions for algorithm: {algorithm}")
        return model_files
    
    # Filter files by expected extensions
    filtered_files = []
    for file_info in model_files:
        filename = file_info.get('filename', '')
        filepath = file_info.get('filepath', '')
        
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in expected_extensions:
            filtered_files.append(file_info)
            print(f"✅ Keeping {filename} (matches {algorithm})")
        else:
            print(f"🚫 Filtering out {filename} (doesn't match {algorithm}, expected {expected_extensions})")
    
    return filtered_files if filtered_files else model_files  # Return all if none match

def _get_algorithm_description(algorithm: str) -> str:
    """Get detailed description for algorithms"""
    descriptions = {
        'isolation_forest': '''Isolation Forest is an unsupervised anomaly detection algorithm that works by isolating anomalies.
- Uses tree structures to isolate points
- Anomalies are isolated with fewer cuts than normal points
- Fast and memory efficient
- Good for high-dimensional data''',
        'one_class_svm': '''One-Class SVM learns a decision boundary around normal data points.
- Creates a hyperplane that encapsulates normal data
- Robust to outliers in training data
- Good for complex, non-linear patterns
- Requires parameter tuning''',
        'local_outlier_factor': '''Local Outlier Factor detects anomalies based on local density.
- Compares local density of a point to its neighbors
- Good for detecting local anomalies
- Works well with irregular patterns
- Sensitive to parameter choices''',
        'autoencoder': '''Autoencoder is a neural network that learns to reconstruct input data.
- Learns compressed representation of normal data
- Anomalies have high reconstruction error
- Good for image anomaly detection
- Requires more training data and time'''
    }
    return descriptions.get(algorithm, 'Algorithm-specific anomaly detection approach')

def get_service_description(service_name: str) -> str:
    """Get human-readable description for services"""
    descriptions = {
        'guardrail_service': 'Content safety and moderation service',
        'auto_annotation_service': 'Automated image annotation with multiple ML models',
        'grounding_dino_service': 'GroundingDINO for object detection with text prompts',
        'grounding_dino_sam2_service': 'GroundingDINO + SAM2 for detection and segmentation',
        'sam2_service': 'Segment Anything Model 2 for image segmentation',
        'roi_extraction_service': 'Region of Interest extraction for anomaly detection',
        'anomaly_service': 'Basic anomaly detection for manufacturing defects',
        'advanced_anomaly_service': 'Advanced anomaly detection with multiple algorithms',
        'llm_clip_anomaly_service': 'LLM + CLIP powered anomaly detection',
        'defect_detection_service': 'Specialized defect detection for manufacturing',
        'dinov2_service': 'DINOv2 foundation model for visual features',
        'dinov3_service': 'DINOv3 foundation model for visual features'
    }
    return descriptions.get(service_name, 'ML service for computer vision tasks')