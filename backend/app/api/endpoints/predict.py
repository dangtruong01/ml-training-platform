from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from backend.services.yolo_service import yolo_service
import os
import json
from typing import List

router = APIRouter()

@router.post("/predict-detect")
async def predict_detection(file: UploadFile = File(...), model_path: str = Form(None)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        result = yolo_service.predict_detection(temp_file_path, model_path)
        os.remove(temp_file_path)  # Clean up the temporary file
        
        if isinstance(result, dict):  # JSON result with quality assessment
            return JSONResponse(result)
        else:  # File path result
            if not os.path.exists(result):
                raise HTTPException(status_code=404, detail="Result not found")
            return FileResponse(result)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...), model_type: str = Form(...)):
    """Upload a trained model for prediction"""
    try:
        model_info = await yolo_service.upload_model(file, model_type)
        return JSONResponse(model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-models")
async def list_models():
    """List all available models"""
    try:
        models = yolo_service.list_available_models()
        return JSONResponse(models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...), model_path: str = Form(None), task_type: str = Form("detection")):
    """Predict on multiple images with quality assessment"""
    try:
        results = await yolo_service.predict_batch(files, model_path, task_type)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quality-assessment")
async def quality_assessment(file: UploadFile = File(...), model_path: str = Form(None)):
    """Assess image quality as OK/NG based on defect detection"""
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        result = yolo_service.assess_quality(temp_file_path, model_path)
        os.remove(temp_file_path)
        return JSONResponse(result)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-seg")
async def predict_segmentation(file: UploadFile = File(...), model_path: str = Form(None)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        result = yolo_service.predict_segmentation(temp_file_path, model_path)
        os.remove(temp_file_path)  # Clean up the temporary file
        
        if isinstance(result, dict):  # JSON result with quality assessment
            return JSONResponse(result)
        else:  # File path result
            if not os.path.exists(result):
                raise HTTPException(status_code=404, detail="Result not found")
            return FileResponse(result)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...), model_type: str = Form(...)):
    """Upload a trained model for prediction"""
    try:
        model_info = await yolo_service.upload_model(file, model_type)
        return JSONResponse(model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-models")
async def list_models():
    """List all available models"""
    try:
        models = yolo_service.list_available_models()
        return JSONResponse(models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...), model_path: str = Form(None), task_type: str = Form("detection")):
    """Predict on multiple images with quality assessment"""
    try:
        results = await yolo_service.predict_batch(files, model_path, task_type)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quality-assessment")
async def quality_assessment(file: UploadFile = File(...), model_path: str = Form(None)):
    """Assess image quality as OK/NG based on defect detection"""
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        result = yolo_service.assess_quality(temp_file_path, model_path)
        os.remove(temp_file_path)
        return JSONResponse(result)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))