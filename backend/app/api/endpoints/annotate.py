from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from backend.services.yolo_service import yolo_service
from backend.services.defect_detection_service import defect_detection_service
try:
    from backend.services.guardrail_service import guardrail_service
except ImportError:
    guardrail_service = None
from backend.services.grounding_dino_service import grounding_dino_service
from backend.services.grounding_dino_sam2_service import grounding_dino_sam2_service
from backend.services.llm_clip_anomaly_service import llm_clip_anomaly_service
import os
import traceback
import subprocess
import zipfile
import uuid
import json
import time
from typing import List

router = APIRouter()

# Simple in-memory cache for batch results
batch_results_cache = {}

def cache_batch_result(batch_id: str, filename: str, result_path: str, annotation_type: str):
    """Cache a processing result for later zip download"""
    if batch_id not in batch_results_cache:
        batch_results_cache[batch_id] = {
            'results': [],
            'annotation_type': annotation_type,
            'timestamp': time.time()
        }
    
    batch_results_cache[batch_id]['results'].append({
        'filename': filename,
        'result_path': result_path
    })

def get_batch_results(batch_id: str):
    """Retrieve cached batch results"""
    return batch_results_cache.get(batch_id)

def cleanup_old_cache_entries():
    """Remove cache entries older than 1 hour"""
    current_time = time.time()
    expired_keys = []
    
    for batch_id, data in batch_results_cache.items():
        if current_time - data['timestamp'] > 3600:  # 1 hour
            expired_keys.append(batch_id)
    
    for key in expired_keys:
        del batch_results_cache[key]

@router.post("/pre-annotate-detect")
async def pre_annotate_detection(file: UploadFile = File(...), batch_id: str = Form(None)):
    """Object detection with bounding boxes"""
    temp_dir = os.path.abspath("temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        result_path = yolo_service.pre_annotate_detection(temp_file_path)
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Detection result file not found after processing.")
        
        # Cache result if batch_id provided
        if batch_id:
            cache_batch_result(batch_id, file.filename, result_path, 'detection')
            print(f"Cached result for batch {batch_id}: {file.filename}")
            
        return FileResponse(result_path)
    except subprocess.CalledProcessError as e:
        error_message = f"An error occurred during detection: {e.stderr}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred during detection: {e}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

@router.post("/pre-annotate-segment")
async def pre_annotate_segmentation(file: UploadFile = File(...), batch_id: str = Form(None)):
    """Segmentation with precise contours"""
    temp_dir = os.path.abspath("temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        result_path = yolo_service.pre_annotate_segmentation(temp_file_path)
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Segmentation result file not found after processing.")
        
        # Cache result if batch_id provided
        if batch_id:
            cache_batch_result(batch_id, file.filename, result_path, 'segmentation')
            print(f"Cached result for batch {batch_id}: {file.filename}")
            
        return FileResponse(result_path)
    except subprocess.CalledProcessError as e:
        error_message = f"An error occurred during segmentation: {e.stderr}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred during segmentation: {e}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

@router.post("/pre-annotate-sam2-detect")
async def pre_annotate_sam2_detection(file: UploadFile = File(...), batch_id: str = Form(None)):
    """SAM2-based object detection with bounding boxes"""
    temp_dir = os.path.abspath("temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        result_path = yolo_service.pre_annotate_sam2_detection(temp_file_path)
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="SAM2 detection result file not found after processing.")
        
        # Cache result if batch_id provided
        if batch_id:
            cache_batch_result(batch_id, file.filename, result_path, 'sam2-detection')
            print(f"Cached result for batch {batch_id}: {file.filename}")
            
        return FileResponse(result_path)
    except RuntimeError as e:
        if "SAM2 service not available" in str(e):
            raise HTTPException(status_code=503, detail="SAM2 is not properly installed. Please install SAM2 dependencies.")
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        error_message = f"An unexpected error occurred during SAM2 detection: {e}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

@router.post("/pre-annotate-sam2-segment")
async def pre_annotate_sam2_segmentation(file: UploadFile = File(...), batch_id: str = Form(None)):
    """SAM2-based segmentation with precise masks"""
    temp_dir = os.path.abspath("temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        result_path = yolo_service.pre_annotate_sam2_segmentation(temp_file_path)
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="SAM2 segmentation result file not found after processing.")
        
        # Cache result if batch_id provided
        if batch_id:
            cache_batch_result(batch_id, file.filename, result_path, 'sam2-segmentation')
            print(f"Cached result for batch {batch_id}: {file.filename}")
            
        return FileResponse(result_path)
    except RuntimeError as e:
        if "SAM2 service not available" in str(e):
            raise HTTPException(status_code=503, detail="SAM2 is not properly installed. Please install SAM2 dependencies.")
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        error_message = f"An unexpected error occurred during SAM2 segmentation: {e}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

@router.post("/download-batch-zip/{batch_id}")
async def download_batch_zip(batch_id: str):
    """Download zip file of cached batch results (no reprocessing)"""
    print(f"Downloading batch zip for batch_id: {batch_id}")
    
    # Clean up old cache entries first
    cleanup_old_cache_entries()
    
    # Get cached results
    batch_data = get_batch_results(batch_id)
    if not batch_data:
        raise HTTPException(status_code=404, detail="Batch results not found or expired")
    
    results = batch_data['results']
    annotation_type = batch_data['annotation_type']
    
    if not results:
        raise HTTPException(status_code=404, detail="No results found in batch")
    
    # Create temporary directory for zip
    temp_dir = os.path.abspath(f"temp_zip_{uuid.uuid4()}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Create zip file
        zip_filename = f"annotated_{annotation_type}_batch.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        print(f"Creating zip with {len(results)} files...")
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for result_data in results:
                result_path = result_data['result_path']
                original_name = result_data['filename']
                
                if os.path.exists(result_path):
                    # Create annotated filename
                    base_name = os.path.splitext(original_name)[0]
                    ext = os.path.splitext(original_name)[1]
                    annotated_name = f"{annotation_type}_{base_name}{ext}"
                    
                    zip_file.write(result_path, annotated_name)
                    print(f"Added to zip: {annotated_name}")
                else:
                    print(f"Warning: Result file not found: {result_path}")
        
        if not os.path.exists(zip_path):
            raise HTTPException(status_code=404, detail="Failed to create zip file")
        
        print(f"Zip file created successfully: {zip_path}")
        
        return FileResponse(
            zip_path,
            filename=zip_filename,
            media_type='application/zip'
        )
        
    except Exception as e:
        error_message = f"An error occurred creating zip: {e}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/batch-annotate-zip") 
async def batch_annotate_zip_legacy(files: List[UploadFile] = File(...), annotationType: str = Form(...)):
    """Legacy endpoint - still processes files (for compatibility)"""
    print("⚠️  WARNING: Using legacy batch endpoint that reprocesses images")
    print("Consider using the cached download approach for better performance")
    
    # Keep the original implementation for backward compatibility
    temp_dir = os.path.abspath(f"temp_batch_{uuid.uuid4()}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        annotated_files = []
        
        for file in files:
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            with open(temp_file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            try:
                if annotationType == 'detection':
                    result_path = yolo_service.pre_annotate_detection(temp_file_path)
                elif annotationType == 'segmentation':
                    result_path = yolo_service.pre_annotate_segmentation(temp_file_path)
                elif annotationType == 'sam2-detection':
                    result_path = yolo_service.pre_annotate_sam2_detection(temp_file_path)
                elif annotationType == 'sam2-segmentation':
                    result_path = yolo_service.pre_annotate_sam2_segmentation(temp_file_path)
                else:
                    result_path = yolo_service.pre_annotate_detection(temp_file_path)
                
                if os.path.exists(result_path):
                    annotated_files.append((result_path, file.filename))
                
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                continue
        
        if not annotated_files:
            raise HTTPException(status_code=404, detail="No files were successfully processed")
        
        zip_filename = f"annotated_{annotationType}_batch.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for result_path, original_name in annotated_files:
                base_name = os.path.splitext(original_name)[0]
                ext = os.path.splitext(original_name)[1]
                annotated_name = f"{annotationType}_{base_name}{ext}"
                zip_file.write(result_path, annotated_name)
        
        return FileResponse(zip_path, filename=zip_filename, media_type='application/zip')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pre-annotate")
async def pre_annotate(file: UploadFile = File(...)):
    """Legacy endpoint for single file annotation"""
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        result_path = yolo_service.pre_annotate_detection(temp_file_path)
        
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Pre-annotation failed")
            
        return FileResponse(result_path)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# GroundingDINO Endpoints
@router.post("/grounding-dino-only-annotate")
async def grounding_dino_only_annotate(
    file: UploadFile = File(...),
    prompts: str = Form(...),
    confidence_threshold: float = Form(0.3),
    batch_id: str = Form(None)
):
    """Single image annotation with GroundingDINO only (no SAM2)"""
    temp_file_path = f"temp_grounding_dino_only_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Parse prompts
        prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
        
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No valid prompts provided")
        
        print(f"🎯 GroundingDINO-only annotation with prompts: {prompt_list}")
        
        # Run GroundingDINO annotation only (no SAM2)
        result = grounding_dino_service.annotate_with_prompts(
            temp_file_path,
            prompt_list,
            confidence_threshold
        )
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('message', 'Annotation failed'))
        
        # Cache result if batch_id provided
        if batch_id and result['status'] == 'success':
            cache_batch_result(batch_id, file.filename, result['annotated_image_path'], 'grounding-dino-only')
            print(f"Cached GroundingDINO-only result for batch {batch_id}: {file.filename}")
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"❌ GroundingDINO-only annotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/grounding-dino-annotate")
async def grounding_dino_annotate(
    file: UploadFile = File(...), 
    prompts: str = Form(...),
    confidence_threshold: float = Form(0.3),
    batch_id: str = Form(None)
):
    """Annotate image using GroundingDINO + SAM2 hybrid pipeline"""
    temp_file_path = f"temp_grounding_dino_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Parse prompts (comma-separated)
        prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
        
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No valid prompts provided")
        
        print(f"🎯 GroundingDINO annotation with prompts: {prompt_list}")
        
        # Run GroundingDINO annotation
        result = defect_detection_service.predict_with_hybrid_model(
            image_path=temp_file_path,
            prompts=prompt_list,
            confidence_threshold=confidence_threshold
        )
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('message', 'Annotation failed'))
        
        # Cache result if batch_id provided
        if batch_id and result['status'] == 'success':
            cache_batch_result(batch_id, file.filename, result['annotated_image_path'], 'grounding-dino')
            print(f"Cached GroundingDINO result for batch {batch_id}: {file.filename}")
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"❌ GroundingDINO annotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/grounding-dino-batch")
async def grounding_dino_batch_annotate(
    files: List[UploadFile] = File(...),
    prompts: str = Form(...),
    confidence_threshold: float = Form(0.3)
):
    """Batch annotate multiple images with GroundingDINO"""
    temp_files = []
    
    try:
        # Parse prompts
        prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
        
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No valid prompts provided")
        
        # Save all uploaded files
        for file in files:
            temp_path = f"temp_batch_grounding_dino_{file.filename}"
            temp_files.append(temp_path)
            
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
        
        print(f"🎯 Batch GroundingDINO annotation for {len(files)} images")
        
        # Run batch annotation
        batch_result = grounding_dino_service.batch_annotate(
            temp_files, prompt_list, confidence_threshold
        )
        
        return JSONResponse(batch_result)
        
    except Exception as e:
        print(f"❌ Batch GroundingDINO annotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@router.post("/grounding-dino-export-yolo")
async def grounding_dino_export_yolo(
    annotation_results: str = Form(...),
    dataset_name: str = Form("grounding_dino_dataset")
):
    """Export GroundingDINO annotation results to YOLO dataset format"""
    try:
        # Parse annotation results
        results = json.loads(annotation_results)
        
        # Export to YOLO format
        output_dir = os.path.abspath("ml/datasets")
        dataset_path = grounding_dino_service.export_yolo_dataset(results, output_dir)
        
        return JSONResponse({
            "status": "success",
            "message": "Dataset exported successfully",
            "dataset_path": dataset_path,
            "dataset_name": os.path.basename(dataset_path)
        })
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid annotation results format")
    except Exception as e:
        print(f"❌ Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/grounding-dino-status")
async def grounding_dino_status():
    """Check if GroundingDINO is available"""
    is_available = grounding_dino_service.model is not None
    
    return JSONResponse({
        "available": is_available,
        "status": "ready" if is_available else "not_installed",
        "message": "GroundingDINO is ready" if is_available else "GroundingDINO not available. Please install dependencies."
    })

@router.post("/train-anomaly-model")
async def train_anomaly_model(
    files: List[UploadFile] = File(...),
    project_name: str = Form(...)
):
    """
    Train an anomaly detection model on a set of 'good' images.
    """
    temp_files = []
    try:
        for file in files:
            temp_path = f"temp_train_{file.filename}"
            temp_files.append(temp_path)
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())

        if guardrail_service is None:
            raise HTTPException(status_code=501, detail="Guardrail service not available - missing dependencies")
        
        model_path = guardrail_service.train_anomaly_model(temp_files, project_name)
        
        return {"status": "success", "model_path": model_path}
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@router.post("/annotate-with-guardrail")
async def annotate_with_guardrail(
    file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    prompts: str = Form(...),
    confidence_threshold: float = Form(0.3)
):
    """
    Annotate an image using the Guardrail pipeline.
    """
    temp_file_path = f"temp_guardrail_{file.filename}"
    temp_model_path = f"temp_model_{model_file.filename}"
    
    try:
        # Save uploaded image
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Save uploaded model file
        with open(temp_model_path, "wb") as buffer:
            buffer.write(await model_file.read())

        prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
        
        if guardrail_service is None:
            raise HTTPException(status_code=501, detail="Guardrail service not available - missing dependencies")
        
        result = guardrail_service.annotate_with_guardrail(
            temp_file_path, temp_model_path, prompt_list, confidence_threshold
        )
        
        return JSONResponse(result)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

# GroundingDINO + SAM2 Integrated Endpoints
@router.post("/grounding-dino-sam2-annotate")
async def grounding_dino_sam2_annotate(
    file: UploadFile = File(...), 
    prompts: str = Form(...),
    confidence_threshold: float = Form(0.3),
    use_sam2_segmentation: bool = Form(True),
    device: str = Form("auto"),
    batch_id: str = Form(None)
):
    """Integrated GroundingDINO + SAM2 annotation pipeline"""
    temp_file_path = f"temp_grounding_dino_sam2_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Parse prompts (comma-separated)
        prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
        
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No valid prompts provided")
        
        # Set device if specified
        if device != "auto":
            grounding_dino_sam2_service.set_device(device)
        
        print(f"🎯 GroundingDINO + SAM2 annotation with prompts: {prompt_list}")
        print(f"🔧 Device: {device}, SAM2 segmentation: {use_sam2_segmentation}")
        
        # Run integrated pipeline
        result = grounding_dino_sam2_service.detect_and_segment(
            image_path=temp_file_path,
            prompts=prompt_list,
            confidence_threshold=confidence_threshold,
            use_sam2_segmentation=use_sam2_segmentation
        )
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('message', 'Annotation failed'))
        
        # Cache result if batch_id provided
        if batch_id and result['status'] == 'success':
            cache_batch_result(batch_id, file.filename, result['annotated_image_path'], 'grounding-dino-sam2')
            print(f"Cached GroundingDINO+SAM2 result for batch {batch_id}: {file.filename}")
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"❌ GroundingDINO + SAM2 annotation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/grounding-dino-sam2-batch")
async def grounding_dino_sam2_batch_annotate(
    files: List[UploadFile] = File(...),
    prompts: str = Form(...),
    confidence_threshold: float = Form(0.3),
    use_sam2_segmentation: bool = Form(True),
    device: str = Form("auto")
):
    """Batch integrated GroundingDINO + SAM2 annotation"""
    temp_files = []
    
    try:
        # Parse prompts
        prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
        
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No valid prompts provided")
        
        # Set device if specified
        if device != "auto":
            grounding_dino_sam2_service.set_device(device)
        
        # Save all uploaded files
        for file in files:
            temp_path = f"temp_batch_grounding_dino_sam2_{file.filename}"
            temp_files.append(temp_path)
            
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
        
        print(f"🎯 Batch GroundingDINO + SAM2 annotation for {len(files)} images")
        print(f"🔧 Device: {device}, SAM2 segmentation: {use_sam2_segmentation}")
        
        # Run batch annotation
        batch_result = grounding_dino_sam2_service.batch_detect_and_segment(
            temp_files, prompt_list, confidence_threshold, use_sam2_segmentation
        )
        
        return JSONResponse(batch_result)
        
    except Exception as e:
        print(f"❌ Batch GroundingDINO + SAM2 annotation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@router.get("/grounding-dino-sam2-status")
async def grounding_dino_sam2_status():
    """Check if GroundingDINO + SAM2 integrated service is available"""
    grounding_dino_available = grounding_dino_service.model is not None
    sam2_available = grounding_dino_sam2_service.sam2._ensure_model_loaded()
    
    return JSONResponse({
        "grounding_dino_available": grounding_dino_available,
        "sam2_available": sam2_available,
        "integrated_available": grounding_dino_available and sam2_available,
        "status": "ready" if (grounding_dino_available and sam2_available) else "partially_available",
        "message": "Integrated service ready" if (grounding_dino_available and sam2_available) else "Some models not available",
        "device": grounding_dino_service.device if grounding_dino_available else "unknown"
    })

# LLM + CLIP Anomaly Detection Endpoints
@router.post("/llm-clip-anomaly-detect")
async def llm_clip_anomaly_detect(
    file: UploadFile = File(...),
    component_type: str = Form(...),
    context: str = Form(""),
    confidence_threshold: float = Form(0.3),
    similarity_threshold: float = Form(0.7),
    use_llm: bool = Form(True),
    batch_id: str = Form(None)
):
    """LLM-Generated Prompts + GroundingDINO + CLIP anomaly detection"""
    temp_file_path = f"temp_llm_clip_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        print(f"🧠 LLM+CLIP anomaly detection for component: {component_type}")
        print(f"🔧 Context: {context}, Use LLM: {use_llm}")
        
        # Run LLM+CLIP analysis
        result = llm_clip_anomaly_service.detect_anomalies_with_llm_clip(
            image_path=temp_file_path,
            component_type=component_type,
            context=context,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
            use_llm=use_llm
        )
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('message', 'Analysis failed'))
        
        # Cache result if batch_id provided
        if batch_id and result['status'] == 'success' and result.get('annotated_image_path'):
            cache_batch_result(batch_id, file.filename, result['annotated_image_path'], 'llm-clip-anomaly')
            print(f"Cached LLM+CLIP result for batch {batch_id}: {file.filename}")
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"❌ LLM+CLIP anomaly detection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/llm-clip-anomaly-batch")
async def llm_clip_anomaly_batch(
    files: List[UploadFile] = File(...),
    component_type: str = Form(...),
    context: str = Form(""),
    confidence_threshold: float = Form(0.3),
    similarity_threshold: float = Form(0.7),
    use_llm: bool = Form(True)
):
    """Batch LLM+CLIP anomaly detection for multiple images"""
    temp_files = []
    
    try:
        # Save all uploaded files
        for file in files:
            temp_path = f"temp_batch_llm_clip_{file.filename}"
            temp_files.append(temp_path)
            
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
        
        print(f"🧠 Batch LLM+CLIP anomaly detection for {len(files)} images")
        print(f"📋 Component: {component_type}, Context: {context}")
        
        # Run batch analysis
        batch_result = llm_clip_anomaly_service.batch_detect_anomalies(
            temp_files, component_type, context, 
            confidence_threshold, similarity_threshold, use_llm
        )
        
        return JSONResponse(batch_result)
        
    except Exception as e:
        print(f"❌ Batch LLM+CLIP anomaly detection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@router.post("/generate-component-descriptions")
async def generate_component_descriptions(
    component_type: str = Form(...),
    context: str = Form(""),
    use_llm: bool = Form(True)
):
    """Generate normal and anomaly descriptions for a component type"""
    try:
        descriptions = llm_clip_anomaly_service.generate_component_descriptions(
            component_type, context, use_llm
        )
        
        return JSONResponse({
            "status": "success",
            "component_type": component_type,
            "context": context,
            "descriptions": descriptions,
            "generated_with_llm": use_llm
        })
        
    except Exception as e:
        print(f"❌ Description generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/llm-clip-status")
async def llm_clip_status():
    """Check if LLM+CLIP service is available"""
    grounding_dino_available = grounding_dino_service.model is not None
    clip_available = llm_clip_anomaly_service._ensure_clip_loaded()
    llm_available = llm_clip_anomaly_service._ensure_llm_client()
    
    return JSONResponse({
        "grounding_dino_available": grounding_dino_available,
        "clip_available": clip_available,
        "llm_available": llm_available,
        "service_available": grounding_dino_available and clip_available,
        "status": "ready" if (grounding_dino_available and clip_available) else "dependencies_missing",
        "message": "LLM+CLIP service ready" if (grounding_dino_available and clip_available) else "Missing dependencies",
        "recommendations": {
            "install_clip": "pip install transformers torch" if not clip_available else None,
            "install_openai": "pip install openai" if not llm_available else None,
            "set_openai_key": "Set OPENAI_API_KEY environment variable" if not llm_available else None
        }
    })

@router.get("/llm-clip-cache-info")
async def llm_clip_cache_info():
    """Get information about the LLM description cache"""
    try:
        cache_info = llm_clip_anomaly_service.get_cache_info()
        return JSONResponse({
            "status": "success",
            **cache_info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm-clip-clear-cache")
async def llm_clip_clear_cache():
    """Clear the LLM description cache"""
    try:
        llm_clip_anomaly_service.clear_description_cache()
        return JSONResponse({
            "status": "success",
            "message": "Description cache cleared successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))