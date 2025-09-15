from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import os
import glob
import time
from datetime import datetime
import requests

try:
    from backend.services.ml.yolo_service import yolo_service
    from backend.services.core.database_service import database_service
    from backend.services import get_service
except ImportError:
    from services.ml.yolo_service import yolo_service
    from services.core.database_service import database_service
    from services import get_service

router = APIRouter()

def _get_download_source(training_job_result: dict) -> str:
    """Determine if model should be downloaded from GCS or local filesystem"""
    if training_job_result['status'] == 'success':
        training_job = training_job_result['training_job']
        training_config = training_job.get('training_config', {})
        device = training_config.get('device', 'cpu')
        
        # GPU training (cuda/mps) → GCS, CPU training → local filesystem
        return 'gcs' if device in ['cuda', 'mps'] else 'local'
    
    return 'local'  # Default fallback

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

def _get_model_files_from_gcs(model_id: str) -> list:
    """Download model files from GCS"""
    try:
        from google.cloud import storage
        import tempfile
        
        storage_client = storage.Client()
        bucket_name = 'mltraining-models'
        bucket = storage_client.bucket(bucket_name)
        
        model_type = _get_model_type_from_id(model_id)
        gcs_prefix = f"{model_type}/{model_id}/"
        
        print(f"🔍 Looking for files in gs://{bucket_name}/{gcs_prefix}")
        
        model_files = []
        blobs = bucket.list_blobs(prefix=gcs_prefix)
        
        for blob in blobs:
            if not blob.name.endswith('/'):  # Skip directories
                # Download to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{os.path.basename(blob.name)}")
                blob.download_to_filename(temp_file.name)
                
                model_files.append({
                    'filename': os.path.basename(blob.name),
                    'filepath': temp_file.name,  # Local temp file path
                    'gcs_path': f"gs://{bucket_name}/{blob.name}",
                    'file_size': blob.size,
                    'created_at': blob.time_created.isoformat() if blob.time_created else None
                })
                
                print(f"📥 Downloaded {blob.name} to {temp_file.name}")
        
        print(f"✅ Downloaded {len(model_files)} files from GCS")
        return model_files
        
    except Exception as e:
        print(f"❌ Error downloading from GCS: {e}")
        return []

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
            # Verify files still exist on disk
            verified_files = []
            for file_info in model_files:
                if os.path.exists(file_info.get('filepath', '')):
                    verified_files.append(file_info)
                else:
                    print(f"⚠️ Database file not found on disk: {file_info.get('filepath', '')}")
            
            if verified_files:
                model_files = verified_files
            else:
                print(f"❌ No database files found on disk, falling back to filesystem scan")
                model_files = []
        
        # Step 3: Fallback to filesystem scan if no database files
        if not model_files:
            print(f"📁 Fallback: Scanning filesystem for {model_id}")
            
            # Determine download source
            download_source = _get_download_source(training_job_result)
            print(f"📍 Download source: {download_source}")
            
            if download_source == 'gcs':
                # Download from GCS (GPU/Vertex AI training)
                model_files = _get_model_files_from_gcs(model_id)
            else:
                # Scan local filesystem (CPU training)
                model_files = yolo_service.get_model_files(model_id)
                
            # Apply algorithm-specific filtering if we know the algorithm
            if model_files and algorithm != 'unknown':
                model_files = _filter_files_by_algorithm(model_files, algorithm)
        
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
                
                for file_info in model_files:
                    file_path = file_info['filepath']
                    
                    if os.path.exists(file_path):
                        # Get just the filename for the zip archive
                        filename = file_info.get('filename', os.path.basename(file_path))
                        
                        # Add file to zip
                        zipf.write(file_path, filename)
                        files_added += 1
                        print(f"📦 Added {filename} to zip archive")
                
                # Add a README file with model information
                if training_job_result['status'] == 'success':
                    training_job = training_job_result['training_job']
                    
                    algorithm = training_job.get('algorithm', 'Unknown')
                    model_format = 'sklearn (.pkl)' if algorithm in ['isolation_forest', 'one_class_svm', 'local_outlier_factor'] else 'pytorch (.pth)' if algorithm == 'autoencoder' else 'Unknown'
                    
                    readme_content = f"""# Model: {model_id}

## Training Information
- Model Type: {training_job.get('model_type', 'Unknown')}
- Algorithm: {algorithm}
- Model Format: {model_format}
- Project ID: {training_job.get('project_id', 'Unknown')}
- Status: {training_job.get('status', 'Unknown')}
- Progress: {training_job.get('progress', 0)}%
- Epochs: {training_job.get('current_epoch', 0)}/{training_job.get('total_epochs', 0)}
- Started: {training_job.get('started_at', 'Unknown')}
- Completed: {training_job.get('completed_at', 'Unknown')}

## Algorithm Details
{_get_algorithm_description(algorithm)}

## Files Included
"""
                    for file_info in model_files:
                        if os.path.exists(file_info['filepath']):
                            filename = file_info.get('filename', os.path.basename(file_info['filepath']))
                            file_size = file_info.get('file_size', 0)
                            readme_content += f"- {filename} ({file_size} bytes)\n"
                    
                    readme_content += f"\n## Download Information\n- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- Archive created by OpenTrainer\n"
                    
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