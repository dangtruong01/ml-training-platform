from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import requests
try:
    from backend.services.ml.yolo_service import yolo_service
    from backend.services.core.database_service import database_service
except ImportError:
    from services.ml.yolo_service import yolo_service
    from services.core.database_service import database_service

router = APIRouter()

@router.post("/train-detect")
async def train_detection(file: UploadFile = File(...), device: str = Form("cpu")):
    """Train a detection model"""
    try:
        # Handle dataset upload
        dataset_path = await yolo_service.handle_dataset_upload(file, "detection")
        if not dataset_path:
            raise HTTPException(status_code=400, detail="Invalid dataset format")
        
        # Start training
        task_id = yolo_service.train_detection(dataset_path, device)
        
        return {
            "message": "Training started successfully",
            "task_id": task_id,
            "status": "started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-segment")
async def train_segmentation(file: UploadFile = File(...), device: str = Form("cpu")):
    """Train a segmentation model"""
    try:
        # Handle dataset upload
        dataset_path = await yolo_service.handle_dataset_upload(file, "segmentation")
        if not dataset_path:
            raise HTTPException(status_code=400, detail="Invalid dataset format")
        
        # Start training
        task_id = yolo_service.train_segmentation(dataset_path, device)
        
        return {
            "message": "Training started successfully",
            "task_id": task_id,
            "status": "started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-status/{task_id}")
async def get_training_status(task_id: str):
    """Get training progress and status"""
    status = yolo_service.get_training_status(task_id)
    if 'error' in status:
        raise HTTPException(status_code=404, detail=status['error'])
    return status

@router.get("/training-logs/{task_id}")
async def get_training_logs(task_id: str, lines: int = 50):
    """Get recent training logs"""
    logs = yolo_service.get_training_logs(task_id, lines)
    if 'error' in logs:
        raise HTTPException(status_code=404, detail=logs['error'])
    return logs

@router.get("/training-tasks")
async def list_training_tasks():
    """List all training tasks"""
    return yolo_service.list_training_tasks()

@router.get("/all-training-jobs")
async def list_all_training_jobs():
    """List all training jobs from database (all statuses)"""
    try:
        # Get all training jobs regardless of status
        all_jobs = database_service.list_training_jobs()
        return {"training_jobs": all_jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# PROJECT-BASED TRAINING ENDPOINTS
# =============================================================================

@router.post("/train-project/{project_id}")
async def train_project(
    project_id: str,
    algorithm: str = Form("isolation_forest"),
    device: str = Form("cpu"),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    learning_rate: float = Form(0.01),
    model_size: str = Form("n")  # n, s, m, l, x for YOLOv8 sizes
):
    """Train a model using project dataset"""
    try:
        # First validate the project dataset (call service directly instead of HTTP)
        from app.api.endpoints.projects import validate_dataset
        try:
            validation_result = await validate_dataset(project_id)
            validation_data = validation_result.body.decode('utf-8') if hasattr(validation_result, 'body') else validation_result
            if isinstance(validation_data, str):
                import json
                validation_data = json.loads(validation_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unable to validate project dataset: {str(e)}")
        if not validation_data['validation']['is_ready']:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Dataset not ready for training",
                    "validation": validation_data['validation']
                }
            )
        
        project_type = validation_data['validation']['project_type']
        
        # Prepare the dataset for training (call service directly instead of HTTP)
        from app.api.endpoints.projects import prepare_training_dataset
        try:
            preparation_result = await prepare_training_dataset(project_id)
            preparation_data = preparation_result.body.decode('utf-8') if hasattr(preparation_result, 'body') else preparation_result
            if isinstance(preparation_data, str):
                import json
                preparation_data = json.loads(preparation_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to prepare dataset for training: {str(e)}")
        dataset_path = preparation_data['preparation']['dataset_path']
        
        # Start training based on project type
        training_config = {
            'project_id': project_id,
            'dataset_path': dataset_path,
            'algorithm': algorithm,
            'device': device,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_size': model_size
        }
        
        if project_type == 'object_detection':
            # For object detection, pass the algorithm parameter
            task_id = yolo_service.train_detection_from_project(project_id, training_config, algorithm)
        elif project_type == 'segmentation':
            task_id = yolo_service.train_segmentation_from_project(project_id, training_config)
        elif project_type == 'anomaly_detection':
            # For anomaly detection, route to appropriate algorithm
            task_id = yolo_service.train_anomaly_from_project(project_id, training_config, algorithm)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported project type: {project_type}")
        
        return {
            "message": "Training started successfully",
            "task_id": task_id,
            "project_id": project_id,
            "project_type": project_type,
            "status": "started",
            "training_config": training_config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/project-training-status/{project_id}")
async def get_project_training_status(project_id: str):
    """Get training status for a specific project"""
    try:
        # Get all training tasks for this project
        all_tasks = yolo_service.list_training_tasks()
        project_tasks = [task for task in all_tasks if task.get('project_id') == project_id]
        
        if not project_tasks:
            return {
                "project_id": project_id,
                "status": "no_training",
                "message": "No training tasks found for this project"
            }
        
        # Get the latest training task
        latest_task = max(project_tasks, key=lambda x: x.get('created_at', ''))
        task_id = latest_task['task_id']
        
        # Get detailed status
        status = yolo_service.get_training_status(task_id)
        status['project_id'] = project_id
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects-with-training")
async def list_projects_with_training():
    """List all projects that have training tasks"""
    try:
        all_tasks = yolo_service.list_training_tasks()
        project_trainings = {}
        
        for task in all_tasks:
            project_id = task.get('project_id')
            if project_id:
                if project_id not in project_trainings:
                    project_trainings[project_id] = []
                project_trainings[project_id].append(task)
        
        # Sort tasks by creation date for each project
        for project_id in project_trainings:
            project_trainings[project_id].sort(
                key=lambda x: x.get('created_at', ''), 
                reverse=True
            )
        
        return {
            "projects_with_training": project_trainings,
            "total_projects": len(project_trainings)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/complete-job")
async def complete_training_job(request: dict):
    """Complete a training job (called by training container)"""
    try:
        task_id = request.get('task_id')
        status = request.get('status')
        progress = request.get('progress')
        model_files_info = request.get('model_files_info', [])
        training_metrics = request.get('training_metrics', {})
        error_message = request.get('error_message')
        
        if not task_id:
            raise HTTPException(status_code=400, detail="task_id is required")
        
        if status == 'completed':
            # Mark training as completed
            result = database_service.complete_training_job(
                task_id=task_id,
                results_dir=f"gs://mltraining-models/",  # Cloud Storage path
                model_files_info=model_files_info,
                training_metrics=training_metrics
            )
        elif status == 'failed':
            # Mark training as failed
            result = database_service.update_training_job_status(
                task_id=task_id,
                status='failed',
                error_message=error_message
            )
        else:
            # Update progress/status
            result = database_service.update_training_job_status(
                task_id=task_id,
                status=status,
                progress=progress
            )
        
        if result['status'] == 'success':
            print(f"✅ Training job {task_id} marked as {status}")
            return {
                'status': 'success',
                'message': f'Training job {task_id} updated to {status}'
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
        
    except Exception as e:
        print(f"❌ Error completing training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))