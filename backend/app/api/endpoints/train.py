from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from backend.services.yolo_service import yolo_service

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