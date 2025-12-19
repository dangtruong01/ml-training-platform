"""
Cloud training endpoints for Vertex AI integration.
"""
from fastapi import APIRouter, HTTPException, Form
from typing import Optional

try:
    from backend.services.ml.yolo_service import yolo_service
    from backend.services.cloud.vertex_ai_service import vertex_ai_service
    from backend.services.cloud.vertex_ai_monitor import vertex_ai_monitor
    from backend.services.core.database_service import database_service
except ImportError:
    from services.ml.yolo_service import yolo_service
    from services.cloud.vertex_ai_service import vertex_ai_service
    from services.cloud.vertex_ai_monitor import vertex_ai_monitor
    from services.core.database_service import database_service

router = APIRouter()

@router.post("/train-project-cloud/{project_id}")
async def train_project_cloud(
    project_id: str,
    algorithm: str = Form(...),
    device: str = Form(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    learning_rate: float = Form(...),
    model_size: str = Form(...)
):
    """Train a project using Vertex AI cloud training"""
    try:
        # Validate project exists
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")

        project = project_result['project']
        project_type = project.get('project_type', 'anomaly_detection')

        # Respect user device choice - if they select GPU/MPS, use cloud training
        device_lower = device.lower()
        should_use_gpu = device_lower == 'cuda' or device_lower == 'mps' or epochs > 50 or model_size in ['l', 'x']

        # Prepare training configuration
        training_config = {
            'algorithm': algorithm,
            'device': device,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_size': model_size
        }

        # Route to correct training method based on project type
        if project_type == 'object_detection':
            task_id = yolo_service.train_detection_from_project_cloud(project_id, training_config, algorithm)
        elif project_type == 'segmentation':
            task_id = yolo_service.train_segmentation_from_project_cloud(project_id, training_config, algorithm)
        elif project_type == 'anomaly_detection':
            task_id = yolo_service.train_anomaly_from_project_cloud(project_id, training_config, algorithm)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported project type: {project_type}")
        
        # Start monitoring if not already started
        if not vertex_ai_monitor.monitoring:
            vertex_ai_monitor.start_monitoring()
        
        return {
            'status': 'started',
            'task_id': task_id,
            'message': f'Cloud training started for project {project_id} {"with GPU" if should_use_gpu else "on CPU"}',
            'training_type': 'vertex_ai'
        }
        
    except Exception as e:
        print(f"❌ Cloud training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cloud-status")
async def get_cloud_status():
    """Get cloud training service status"""
    try:
        # Test GCP connection
        gcp_status = vertex_ai_service.gcp_config.test_services() if hasattr(vertex_ai_service, 'gcp_config') else {}
        
        # Get monitoring status
        monitor_status = vertex_ai_monitor.get_monitoring_status()
        
        # Get active cloud jobs count
        running_jobs = database_service.list_training_jobs(status='running')
        submitted_jobs = database_service.list_training_jobs(status='submitted')
        
        active_jobs = 0
        if running_jobs['status'] == 'success':
            active_jobs += len([job for job in running_jobs.get('training_jobs', []) 
                              if job.get('training_config', {}).get('vertex_ai_job_id')])
        if submitted_jobs['status'] == 'success':
            active_jobs += len([job for job in submitted_jobs.get('training_jobs', []) 
                               if job.get('training_config', {}).get('vertex_ai_job_id')])
        
        return {
            'status': 'success',
            'cloud_services': gcp_status,
            'monitoring': monitor_status,
            'active_cloud_jobs': active_jobs,
            'vertex_ai_available': vertex_ai_service.project_id is not None
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

@router.get("/vertex-job-status/{vertex_ai_job_id}")
async def get_vertex_job_status(vertex_ai_job_id: str):
    """Get status of a specific Vertex AI job"""
    try:
        result = vertex_ai_service.get_job_status(vertex_ai_job_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-monitoring")
async def start_monitoring():
    """Manually start the Vertex AI monitoring service"""
    try:
        if vertex_ai_monitor.monitoring:
            return {
                'status': 'already_running',
                'message': 'Monitoring service is already running'
            }
        
        vertex_ai_monitor.start_monitoring()
        return {
            'status': 'started',
            'message': 'Vertex AI monitoring service started'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-monitoring")
async def stop_monitoring():
    """Manually stop the Vertex AI monitoring service"""
    try:
        if not vertex_ai_monitor.monitoring:
            return {
                'status': 'already_stopped',
                'message': 'Monitoring service is not running'
            }
        
        vertex_ai_monitor.stop_monitoring()
        return {
            'status': 'stopped',
            'message': 'Vertex AI monitoring service stopped'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trigger-monitoring-check")
async def trigger_monitoring_check():
    """Manually trigger a monitoring check cycle for debugging"""
    try:
        if not vertex_ai_monitor.monitoring:
            return {
                'status': 'error',
                'message': 'Monitoring service is not running'
            }
        
        # Manually trigger the check
        vertex_ai_monitor._check_running_jobs()
        
        return {
            'status': 'success',
            'message': 'Monitoring check triggered successfully. Check backend logs for debug output.'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to trigger monitoring check: {str(e)}'
        }

@router.get("/debug-training-configs")
async def debug_training_configs():
    """Debug endpoint to check training configs in pending/running jobs"""
    try:
        from backend.services.core.database_service import database_service
        
        all_jobs = database_service.list_training_jobs()
        if all_jobs['status'] != 'success':
            return {'error': 'Failed to get jobs'}
        
        debug_info = []
        for job in all_jobs.get('training_jobs', []):
            # Include completed and failed jobs too to see the difference
            if job.get('status') in ['pending', 'running', 'submitted', 'completed', 'failed']:
                debug_info.append({
                    'task_id': job['task_id'],
                    'status': job['status'],
                    'training_config': job.get('training_config'),
                    'training_config_type': type(job.get('training_config')).__name__,
                    'has_vertex_ai_job_id': bool(job.get('training_config', {}).get('vertex_ai_job_id') if isinstance(job.get('training_config'), dict) else False)
                })
        
        return {
            'status': 'success',
            'debug_info': debug_info[:10]  # Limit to first 10 for readability
        }
        
    except ImportError:
        from services.core.database_service import database_service
        
        all_jobs = database_service.list_training_jobs()
        if all_jobs['status'] != 'success':
            return {'error': 'Failed to get jobs'}
        
        debug_info = []
        for job in all_jobs.get('training_jobs', []):
            # Include completed and failed jobs too to see the difference
            if job.get('status') in ['pending', 'running', 'submitted', 'completed', 'failed']:
                debug_info.append({
                    'task_id': job['task_id'],
                    'status': job['status'],
                    'training_config': job.get('training_config'),
                    'training_config_type': type(job.get('training_config')).__name__,
                    'has_vertex_ai_job_id': bool(job.get('training_config', {}).get('vertex_ai_job_id') if isinstance(job.get('training_config'), dict) else False)
                })
        
        return {
            'status': 'success',
            'debug_info': debug_info[:10]  # Limit to first 10 for readability
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Debug failed: {str(e)}'
        }

@router.get("/debug-recent-job/{task_id}")
async def debug_recent_job(task_id: str):
    """Debug a specific recent job to see its full state"""
    try:
        try:
            from backend.services.core.database_service import database_service
            from backend.services.cloud.vertex_ai_service import vertex_ai_service
        except ImportError:
            from services.core.database_service import database_service
            from services.cloud.vertex_ai_service import vertex_ai_service
        
        # Get job from database
        job_result = database_service.get_training_job(task_id)
        if job_result['status'] != 'success':
            return {'error': f'Job {task_id} not found in database'}
        
        job = job_result['training_job']
        
        # Check if it has Vertex AI job ID
        vertex_ai_job_id = None
        if job.get('training_config') and isinstance(job['training_config'], dict):
            vertex_ai_job_id = job['training_config'].get('vertex_ai_job_id')
        
        debug_info = {
            'task_id': task_id,
            'database_status': job.get('status'),
            'database_progress': job.get('progress'),
            'training_config': job.get('training_config'),
            'has_vertex_ai_job_id': bool(vertex_ai_job_id),
            'vertex_ai_job_id': vertex_ai_job_id,
            'vertex_ai_status': None,
            'completion_metadata_exists': False
        }
        
        # If has Vertex AI job ID, check its status
        if vertex_ai_job_id:
            try:
                vertex_status = vertex_ai_service.get_job_status(vertex_ai_job_id)
                debug_info['vertex_ai_status'] = vertex_status
            except Exception as e:
                debug_info['vertex_ai_status_error'] = str(e)
        
        # Check for completion metadata in Cloud Storage
        try:
            from google.cloud import storage
            storage_client = storage.Client()
            bucket = storage_client.bucket('mltraining-vertex-staging')
            metadata_blob = bucket.blob(f"job-completions/{task_id}.json")
            debug_info['completion_metadata_exists'] = metadata_blob.exists()
        except Exception as e:
            debug_info['completion_metadata_error'] = str(e)
        
        return {
            'status': 'success',
            'debug_info': debug_info
        }
        
    except ImportError:
        return {'error': 'Import error - running without backend prefix'}
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Debug failed: {str(e)}'
        }