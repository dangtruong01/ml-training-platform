from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import os
import json
import cv2
import numpy as np
import base64
import uuid

from backend.services.auto_annotation_service import auto_annotation_service
from backend.services.storage import storage_service
from backend.services.database_service import database_service

router = APIRouter()

# =============================================================================
# STORAGE HELPER FUNCTIONS
# =============================================================================

async def get_project_directories(project_id: str) -> Dict[str, str]:
    """Get all directory paths for a project using storage service"""
    base_dir = storage_service.get_project_directory(project_id)
    
    # Standard project directories
    directories = {
        'project': base_dir,
        'training_images': f"{base_dir}/training_images",
        'defective_images': f"{base_dir}/defective_images",
        'workflow_status': f"{base_dir}/workflow_status.json"
    }
    
    # Add ROI directories
    roi_dirs = storage_service.get_roi_directories(project_id)
    directories.update({
        'roi_cache': roi_dirs['normal'],
        'defective_roi_cache': roi_dirs['defective']
    })
    
    # Add model directories  
    model_dirs = storage_service.get_model_directories(project_id)
    directories.update({
        'anomaly_features': model_dirs['features'],
        'defect_results': model_dirs['defect_results']
    })
    
    # Add annotation directories
    annotation_dirs = storage_service.get_annotation_directories(project_id)
    directories.update(annotation_dirs)
    
    return directories

async def ensure_project_exists(project_id: str) -> str:
    """Ensure project exists in storage and return project directory"""
    project_dir = storage_service.get_project_directory(project_id)
    
    # Check if project directory exists
    if not await storage_service.file_exists(f"{project_dir}/.keep"):
        raise HTTPException(status_code=404, detail="Project not found")
    
    return project_dir

# =============================================================================
# PROJECT MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/create-project")
async def create_auto_annotation_project(
    project_name: str = Form(...),
    project_type: str = Form(...),  # 'object_detection' or 'segmentation'
    description: str = Form("")
):
    """
    Create a new auto-annotation project
    
    Two types:
    - object_detection: YOLO for defect bounding boxes
    - segmentation: SAM2 for precise defect masks
    """
    try:
        # Generate unique project ID
        project_id = f"{project_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        # Create project in database
        db_result = database_service.create_project(
            project_id=project_id,
            project_name=project_name, 
            owner="system",  # Could be extracted from authentication later
            project_type=project_type
        )
        
        if db_result['status'] != 'success':
            return JSONResponse(db_result)
        
        # Return format matching the old API
        from datetime import datetime
        metadata = {
            'project_id': project_id,
            'project_name': project_name,
            'project_type': project_type,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'training_status': 'not_started',
            'model_path': None,
            'classes': [],
            'training_images_count': 0,
            'model_metrics': {}
        }
        
        # Project structure for cloud storage paths
        if project_type == 'anomaly_detection':
            structure = {
                'training_images': f"auto_annotation/projects/{project_id}/training_images",
                'roi_cache': f"auto_annotation/projects/{project_id}/roi_cache",
                'defective_images': f"auto_annotation/projects/{project_id}/defective_images",
                'defective_roi_cache': f"auto_annotation/projects/{project_id}/defective_roi_cache",
                'anomaly_features': f"auto_annotation/projects/{project_id}/anomaly_features",
                'defect_detection_results': f"auto_annotation/projects/{project_id}/defect_detection_results"
            }
        else:
            structure = {
                'training_images': f"auto_annotation/projects/{project_id}/training_images",
                'annotations': f"auto_annotation/projects/{project_id}/annotations",
                'models': f"auto_annotation/projects/{project_id}/models",
                'inference_results': f"auto_annotation/projects/{project_id}/inference_results"
            }
        
        result = {
            'status': 'success',
            'project_id': project_id,
            'metadata': metadata,
            'structure': structure
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects")
async def get_auto_annotation_projects():
    """Get list of all auto-annotation projects from database"""
    try:
        result = database_service.list_projects()
        if result['status'] == 'success':
            return JSONResponse(result['projects'])
        else:
            raise HTTPException(status_code=500, detail=result.get('message', 'Unknown error'))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}")
async def get_project_details(project_id: str):
    """Get detailed information about a specific project from database"""
    try:
        result = database_service.get_project(project_id)
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and all its associated data"""
    try:
        result = database_service.delete_project(project_id)
        
        if result['status'] == 'success':
            return JSONResponse(result)
        else:
            raise HTTPException(status_code=404, detail=result['message'])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# DATA UPLOAD ENDPOINTS
# =============================================================================

@router.post("/projects/{project_id}/upload-training-data")
async def upload_training_data(
    project_id: str,
    training_images: List[UploadFile] = File(...),
    annotation_files: Optional[List[UploadFile]] = File(None),
    annotation_format: str = Form("auto")  # 'yolo', 'coco', or 'auto'
):
    """
    Upload training images and annotations to a project
    
    For Object Detection projects:
    - Images: JPG/PNG files
    - Annotations: Multiple YOLO format .txt files
    
    For Segmentation projects:
    - Images: JPG/PNG files  
    - Annotations: COCO format JSON file
    """
    try:
        # Check if project exists in database
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            return JSONResponse({'status': 'error', 'message': 'Project not found'})
        
        uploaded_files = []
        
        # Upload training images
        for image_file in training_images:
            content = await image_file.read()
            
            # Upload to storage
            storage_path = f"auto_annotation/projects/{project_id}/training_images/{image_file.filename}"
            storage_result = await storage_service.upload_file(
                content=content,
                filename=image_file.filename,
                storage_path=storage_path,
                content_type="image/jpeg"
            )
            
            if storage_result['status'] == 'success':
                # Track file in database
                db_result = database_service.add_uploaded_file(
                    project_id=project_id,
                    file_type='training_images',
                    filename=image_file.filename,
                    original_filename=image_file.filename,
                    storage_url=storage_result['storage_url'],
                    storage_path=storage_path,
                    file_size_bytes=len(content),
                    content_type="image/jpeg"
                )
                
                uploaded_files.append({
                    'filename': image_file.filename,
                    'type': 'training_image',
                    'storage_url': storage_result['storage_url'],
                    'tracked': db_result['status'] == 'success'
                })
        
        # Upload annotation files if provided
        if annotation_files:
            for annotation_file in annotation_files:
                content = await annotation_file.read()
                
                # Upload to storage
                storage_path = f"auto_annotation/projects/{project_id}/annotation_files/{annotation_file.filename}"
                storage_result = await storage_service.upload_file(
                    content=content,
                    filename=annotation_file.filename,
                    storage_path=storage_path,
                    content_type="application/json" if annotation_file.filename.endswith('.json') else "text/plain"
                )
                
                if storage_result['status'] == 'success':
                    # Track file in database  
                    db_result = database_service.add_uploaded_file(
                        project_id=project_id,
                        file_type='annotation_files',
                        filename=annotation_file.filename,
                        original_filename=annotation_file.filename,
                        storage_url=storage_result['storage_url'],
                        storage_path=storage_path,
                        file_size_bytes=len(content),
                        content_type="application/json" if annotation_file.filename.endswith('.json') else "text/plain"
                    )
                    
                    uploaded_files.append({
                        'filename': annotation_file.filename,
                        'type': 'annotation_file',
                        'storage_url': storage_result['storage_url'],
                        'tracked': db_result['status'] == 'success'
                    })
        
        result = {
            'status': 'success',
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'uploaded_files': uploaded_files,
            'annotation_format': annotation_format
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MODEL TRAINING ENDPOINTS  
# =============================================================================

@router.post("/projects/{project_id}/start-training")
async def start_model_training(
    project_id: str,
    epochs: int = Form(50),
    batch_size: int = Form(16),
    image_size: int = Form(640),
    patience: int = Form(10)
):
    """
    Start model training for a project
    
    For Object Detection (YOLO):
    - epochs: Number of training epochs
    - batch_size: Training batch size
    - image_size: Input image size
    - patience: Early stopping patience
    
    For Segmentation (SAM2):
    - few_shot_examples: Number of examples per class
    - similarity_threshold: Similarity threshold for matching
    - mask_threshold: Mask confidence threshold
    """
    try:
        training_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'image_size': image_size,
            'patience': patience
        }
        
        result = auto_annotation_service.start_training(project_id, training_params)
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/training-status")
async def get_training_status(project_id: str):
    """Get current training status and metrics"""
    try:
        result = auto_annotation_service.get_training_status(project_id)
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/upload-defective-images")
async def upload_defective_images(
    project_id: str,
    defective_images: List[UploadFile] = File(...)
):
    """
    Upload defective images for anomaly detection workflow
    
    These images will be processed alongside training images in the ROI extraction step
    """
    try:
        # Get project details from database
        project_details = database_service.get_project(project_id)
        if project_details['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get project directories and ensure they exist
        dirs = await get_project_directories(project_id)
        await storage_service.create_directory(dirs['defective_images'])
        
        uploaded_count = 0
        failed_uploads = []
        
        for image_file in defective_images:
            try:
                # Validate file type
                if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    failed_uploads.append(f"{image_file.filename}: Invalid file type")
                    continue
                
                # Save file using storage service
                content = await image_file.read()
                storage_path = f"{dirs['defective_images']}/{image_file.filename}"
                storage_url = await storage_service.upload_file(content, storage_path, "image/jpeg")
                
                # Track file in database
                database_service.add_uploaded_file(
                    project_id=project_id,
                    file_type='defective_images',
                    filename=image_file.filename,
                    original_filename=image_file.filename,
                    storage_url=storage_url,
                    storage_path=storage_path,
                    file_size_bytes=len(content),
                    content_type="image/jpeg"
                )
                
                uploaded_count += 1
                print(f"✅ Uploaded and tracked defective image: {image_file.filename}")
                
            except Exception as e:
                failed_uploads.append(f"{image_file.filename}: {str(e)}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Uploaded {uploaded_count} defective images",
            "uploaded_count": uploaded_count,
            "failed_uploads": failed_uploads,
            "total_attempted": len(defective_images)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ANOMALY DETECTION WORKFLOW ENDPOINTS
# =============================================================================

@router.post("/projects/{project_id}/extract-roi")
async def extract_roi_for_anomaly_detection(
    project_id: str,
    roi_method: str = Form("grounding_dino"),  # 'grounding_dino', 'manufacturing_segmentation', or 'segmentation_mask'
    # GroundingDINO parameters
    component_description: str = Form("metal plate"),
    confidence_threshold: float = Form(0.3),
    # Manufacturing segmentation parameters
    manufacturing_scenario: str = Form("general"),
    part_material: str = Form("metal"),
    fixture_type: str = Form("tray"),
    fixture_color: str = Form("blue")
):
    """
    Step 1: Extract ROI from both training and defective images for anomaly detection workflow
    
    Supports three ROI extraction methods:
    - grounding_dino: AI-based object detection (bounding boxes)
    - manufacturing_segmentation: Edge detection for manufacturing scenarios
    - segmentation_mask: GroundingDINO + SAM2 precise segmentation masks (new!)
    
    Processes both training and defective images with the same ROI method.
    """
    try:
        print(f"🎯 Starting ROI extraction using {roi_method} method")
        
        # Get project details from database
        project_details = database_service.get_project(project_id)
        if project_details['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        
        # Get training images
        training_images_dir = os.path.join(project_dir, "training_images")
        if not os.path.exists(training_images_dir):
            raise HTTPException(status_code=400, detail="No training images found")
        
        training_image_paths = []
        for filename in os.listdir(training_images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                training_image_paths.append(os.path.join(training_images_dir, filename))
        
        if not training_image_paths:
            raise HTTPException(status_code=400, detail="No valid training images found")
        
        # Get defective images
        defective_images_dir = os.path.join(project_dir, "defective_images")
        if not os.path.exists(defective_images_dir):
            raise HTTPException(status_code=400, detail="No defective images found. Upload defective images first.")
        
        defective_image_paths = []
        for filename in os.listdir(defective_images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                defective_image_paths.append(os.path.join(defective_images_dir, filename))
        
        if not defective_image_paths:
            raise HTTPException(status_code=400, detail="No valid defective images found")
        
        print(f"📊 Found {len(training_image_paths)} training images, {len(defective_image_paths)} defective images")
        
        # Extract ROI based on method
        if roi_method == "grounding_dino":
            # Extract ROI for training images using GroundingDINO
            training_roi_result = auto_annotation_service.roi_extractor.extract_roi_from_images(
                training_image_paths, component_description, confidence_threshold, 
                project_id=project_id, image_type="training"
            )
            
            # Extract ROI for defective images using GroundingDINO
            defective_roi_result = auto_annotation_service.roi_extractor.extract_roi_from_images(
                defective_image_paths, component_description, confidence_threshold, 
                project_id=project_id, image_type="defective"
            )
            
        elif roi_method == "manufacturing_segmentation":
            # Import GroundingDINO service for manufacturing ROI extraction
            from backend.services.grounding_dino_service import grounding_dino_service
            
            # Extract ROI for training images using manufacturing segmentation
            training_roi_result = grounding_dino_service.extract_manufacturing_roi(
                training_image_paths, project_id,
                manufacturing_scenario, part_material, fixture_type, fixture_color,
                image_type="training"
            )
            
            # Extract ROI for defective images using manufacturing segmentation  
            defective_roi_result = grounding_dino_service.extract_manufacturing_roi(
                defective_image_paths, project_id,
                manufacturing_scenario, part_material, fixture_type, fixture_color,
                image_type="defective"
            )
            
        elif roi_method == "segmentation_mask":
            # Import integrated GroundingDINO + SAM2 service for segmentation-based ROI extraction
            from backend.services.grounding_dino_sam2_service import grounding_dino_sam2_service
            
            print(f"🎭 Using GroundingDINO + SAM2 segmentation-based ROI extraction")
            
            # Extract ROI for training images using segmentation masks
            training_roi_result = grounding_dino_sam2_service.extract_segmentation_based_roi(
                training_image_paths, component_description, confidence_threshold, 
                project_id=project_id, image_type="training"
            )
            
            # Extract ROI for defective images using segmentation masks  
            defective_roi_result = grounding_dino_sam2_service.extract_segmentation_based_roi(
                defective_image_paths, component_description, confidence_threshold, 
                project_id=project_id, image_type="defective"
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown ROI method: {roi_method}. Supported methods: grounding_dino, manufacturing_segmentation, segmentation_mask")
        
        # Check results
        if training_roi_result['status'] != 'success':
            raise HTTPException(status_code=500, detail=f"Training ROI extraction failed: {training_roi_result.get('message', 'Unknown error')}")
        
        if defective_roi_result['status'] != 'success':
            raise HTTPException(status_code=500, detail=f"Defective ROI extraction failed: {defective_roi_result.get('message', 'Unknown error')}")
        
        # Combine results
        combined_result = {
            "status": "success",
            "roi_method": roi_method,
            "training_roi_result": training_roi_result,
            "defective_roi_result": defective_roi_result,
            "total_images": int(len(training_image_paths) + len(defective_image_paths)),
            "successful_extractions": int(training_roi_result.get('total_processed', 0) + defective_roi_result.get('total_processed', 0)),
            "failed_extractions": int(training_roi_result.get('total_failed', 0) + defective_roi_result.get('total_failed', 0)),
            "message": f"ROI extraction completed using {roi_method}"
        }
        
        # Save workflow status
        workflow_status = {
            "stage1_completed": True,
            "stage1_method": roi_method,
            "stage1_results": combined_result,
            "training_images_count": int(len(training_image_paths)),
            "defective_images_count": int(len(defective_image_paths))
        }
        
        workflow_file = os.path.join(project_dir, "workflow_status.json")
        with open(workflow_file, 'w') as f:
            json.dump(workflow_status, f, indent=2)
        
        print(f"✅ ROI extraction completed: {combined_result['successful_extractions']} success, {combined_result['failed_extractions']} failed")
        
        return JSONResponse(combined_result)
        
    except Exception as e:
        print(f"❌ ROI extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/roi-preview")
async def get_roi_preview(project_id: str, limit: int = 6):
    """
    Get preview images of extracted ROI regions
    """
    try:
        import base64
        from pathlib import Path
        
        # Get project-specific ROI cache directory
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        roi_cache_dir = os.path.join(project_dir, "roi_cache")
        
        if not os.path.exists(roi_cache_dir):
            return JSONResponse({
                'status': 'error',
                'message': 'No ROI cache found. Run ROI extraction first.'
            })
        
        # Find ROI images for this project
        roi_files = []
        for filename in os.listdir(roi_cache_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                roi_files.append(filename)
        
        # Limit number of preview images
        roi_files = sorted(roi_files)[:limit]
        
        # Convert images to base64 for preview
        preview_images = []
        for roi_filename in roi_files:
            roi_path = os.path.join(roi_cache_dir, roi_filename)
            
            try:
                with open(roi_path, 'rb') as f:
                    image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    preview_images.append({
                        'filename': roi_filename,
                        'image_base64': image_base64
                    })
            except Exception as e:
                print(f"⚠️ Error reading ROI image {roi_filename}: {e}")
                continue
        
        return JSONResponse({
            'status': 'success',
            'preview_images': preview_images,
            'total_previews': len(preview_images)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/detect-anomalies")
async def detect_anomalies_with_dinov2(
    project_id: str,
    method: str = Form("mahalanobis"),
    threshold_percentile: float = Form(95.0)
):
    """
    Step 2: Run DINOv2-based anomaly detection on extracted ROI images
    
    Uses pretrained DINOv2 features to detect anomalies by comparing against
    normal image statistics. Requires ROI extraction to be completed first.
    """
    try:
        # Import DINOv2 service
        from backend.services.dinov2_service import dinov2_service
        
        # Check if ROI extraction has been done for this project
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        roi_cache_dir = os.path.join(project_dir, "roi_cache")
        
        if not os.path.exists(roi_cache_dir):
            raise HTTPException(
                status_code=400, 
                detail="ROI extraction not found. Please run Step 1 first."
            )
        
        # Get ROI image paths for this project
        roi_image_paths = []
        for filename in os.listdir(roi_cache_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                roi_image_paths.append(os.path.join(roi_cache_dir, filename))
        
        if not roi_image_paths:
            raise HTTPException(
                status_code=400, 
                detail="No ROI images found. Please run Step 1 first."
            )
        
        print(f"🧠 Starting anomaly detection for project {project_id}")
        print(f"📸 Found {len(roi_image_paths)} ROI images")
        
        # Run anomaly detection pipeline
        result = dinov2_service.run_anomaly_detection_pipeline(
            roi_image_paths, 
            project_id, 
            method, 
            threshold_percentile
        )
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/build-normal-model")
async def build_normal_model_for_anomaly_detection(
    project_id: str,
    model_type: str = "dinov2",  # "dinov2" or "dinov3"
    detection_method: str = "statistical",  # "statistical" or "advanced"
    advanced_methods: str = "ocsvm,isolation_forest,elliptic,pca"  # comma-separated list
):
    """
    Step 2: Build normal model from extracted ROI images
    
    Extracts DINOv2 or DINOv3 features from normal ROI images and builds 
    statistical or advanced ML baseline model for anomaly detection.
    
    Args:
        project_id: Project ID
        model_type: Feature extractor model ("dinov2" or "dinov3")
        detection_method: Type of anomaly detection ("statistical" or "advanced")
        advanced_methods: Comma-separated list of methods for advanced detection
                         (ocsvm, isolation_forest, lof, elliptic, pca, autoencoder)
    """
    try:
        # Import appropriate service based on model_type
        if model_type == "dinov3":
            from backend.services.dinov3_service import dinov3_service
            # Check if DINOv3 model is actually available
            if dinov3_service.model is None:
                print("⚠️ DINOv3 model not available, falling back to DINOv2")
                from backend.services.dinov2_service import dinov2_service
                feature_service = dinov2_service
                model_type = "dinov2"  # Update model_type for consistent file naming
            else:
                feature_service = dinov3_service
                print(f"🧠 Using DINOv3 for feature extraction")
        elif model_type == "dinov2":
            from backend.services.dinov2_service import dinov2_service
            feature_service = dinov2_service
            print(f"🧠 Using DINOv2 for feature extraction")
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model_type: {model_type}. Must be 'dinov2' or 'dinov3'"
            )
        
        # Check if ROI extraction has been done for this project
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        roi_cache_dir = os.path.join(project_dir, "roi_cache")
        
        if not os.path.exists(roi_cache_dir):
            raise HTTPException(
                status_code=400, 
                detail="ROI extraction not found. Please complete Step 1 first."
            )
        
        # Get ROI image paths for this project
        roi_image_paths = []
        for filename in os.listdir(roi_cache_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                roi_image_paths.append(os.path.join(roi_cache_dir, filename))
        
        if not roi_image_paths:
            raise HTTPException(
                status_code=400, 
                detail="No ROI images found. Please complete Step 1 first."
            )
        
        print(f"🧠 Building normal model for project {project_id} using {model_type.upper()}")
        print(f"📸 Using {len(roi_image_paths)} normal ROI images")
        
        # Extract features from normal ROI images
        feature_result = feature_service.extract_features(roi_image_paths, project_id=project_id)
        if feature_result['status'] != 'success':
            raise HTTPException(status_code=500, detail=feature_result['message'])
        
        features = feature_result['features']
        
        # Build normal model using specified detection method
        if detection_method == "advanced":
            # Import advanced anomaly service
            from backend.services.advanced_anomaly_service import advanced_anomaly_service
            
            # Parse advanced methods
            methods_list = [method.strip() for method in advanced_methods.split(',') if method.strip()]
            print(f"🎯 Using advanced anomaly detection methods: {methods_list}")
            
            # Build advanced models
            advanced_result = advanced_anomaly_service.build_advanced_normal_model(
                features, project_id, methods=methods_list
            )
            
            if advanced_result['status'] != 'success':
                raise HTTPException(status_code=500, detail=advanced_result['message'])
            
            # Also build traditional statistical model as fallback
            normal_model_result = feature_service.build_normal_model(features)
            if normal_model_result['status'] != 'success':
                print("⚠️ Statistical model failed, but advanced models succeeded")
            
            # Prepare response for advanced method
            model_data = {
                'project_id': project_id,
                'model_type': model_type,
                'detection_method': 'advanced',
                'advanced_methods': methods_list,
                'advanced_results': advanced_result,
                'statistical_fallback': normal_model_result.get('status') == 'success'
            }
            
            # Save metadata
            features_dir = os.path.join(project_dir, "anomaly_features")
            model_file = os.path.join(features_dir, f"{project_id}_{model_type}_advanced_metadata.json")
            
        else:
            # Traditional statistical approach
            normal_model_result = feature_service.build_normal_model(features)
            if normal_model_result['status'] != 'success':
                raise HTTPException(status_code=500, detail=normal_model_result['message'])
            
            normal_model = normal_model_result['normal_model']
            
            # Prepare traditional model data
            model_data = {
                'project_id': project_id,
                'model_type': model_type,
                'detection_method': 'statistical',
                'normal_images_count': len(roi_image_paths),
                'feature_dimensions': len(normal_model['global_mean']),
                'global_mean': normal_model['global_mean'].tolist(),
                'global_std': normal_model['global_std'].tolist(),
                'global_cov': normal_model['global_cov'].tolist(),
                'feature_shape': normal_model['feature_shape'],
                'timestamp': feature_result.get('extraction_timestamp')
            }
            
            # Save statistical model
            features_dir = os.path.join(project_dir, "anomaly_features")
            model_file = os.path.join(features_dir, f"{project_id}_{model_type}_normal_model.json")
        
        # Save model metadata
        os.makedirs(features_dir, exist_ok=True)
        
        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"💾 Model metadata saved to: {model_file}")
        
        # Calculate statistics for display
        import numpy as np
        if detection_method == "advanced":
            stats = {
                'methods_trained': len(methods_list),
                'total_patches': int(np.prod(features.shape[:3])),  # N * H * W
                'feature_dimensions': int(features.shape[-1]),
                'advanced_models_file': advanced_result.get('models_file', 'N/A')
            }
        else:
            normal_model = normal_model_result['normal_model']
            stats = {
                'mean_norm': float(np.linalg.norm(normal_model['global_mean'])),
                'variance': float(np.mean(normal_model['global_std'] ** 2)),
                'total_patches': int(np.prod(normal_model['feature_shape'][:3]))  # N * H * W
            }
        
        return JSONResponse({
            'status': 'success',
            'message': f'Normal model built successfully using {model_type.upper()} + {detection_method.upper()}',
            'model_type': model_type,
            'detection_method': detection_method,
            'normal_images_count': len(roi_image_paths),
            'feature_dimensions': int(features.shape[-1]),
            'stats': stats,
            'model_file': model_file,
            'advanced_methods': methods_list if detection_method == 'advanced' else None,
            'note': 'Fell back to DINOv2' if model_type == 'dinov2' and 'dinov3' in locals() else None
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error building normal model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/upload-defective-images")
async def upload_defective_images_to_project(
    project_id: str,
    defective_images: List[UploadFile] = File(...)
):
    """
    Upload defective images to project directory
    
    This separates the upload process from ROI extraction for better scalability
    with large batches of defective images.
    """
    try:
        # Ensure project exists and get directories
        await ensure_project_exists(project_id)
        dirs = await get_project_directories(project_id)
        await storage_service.create_directory(dirs['defective_images'])
        
        # Clear existing defective images
        existing_files = await storage_service.list_files(dirs['defective_images'], '.jpg')
        existing_files.extend(await storage_service.list_files(dirs['defective_images'], '.jpeg'))
        existing_files.extend(await storage_service.list_files(dirs['defective_images'], '.png'))
        
        for existing_file in existing_files:
            await storage_service.delete_file(existing_file)
        
        # Save uploaded defective images
        uploaded_files = []
        failed_uploads = []
        
        for uploaded_file in defective_images:
            try:
                if uploaded_file.filename and uploaded_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    content = await uploaded_file.read()
                    storage_path = f"{dirs['defective_images']}/{uploaded_file.filename}"
                    storage_url = await storage_service.upload_file(content, storage_path, "image/jpeg")
                    
                    # Track file in database
                    database_service.add_uploaded_file(
                        project_id=project_id,
                        file_type='defective_images',
                        filename=uploaded_file.filename,
                        original_filename=uploaded_file.filename,
                        storage_url=storage_url,
                        storage_path=storage_path,
                        file_size_bytes=len(content),
                        content_type="image/jpeg"
                    )
                    
                    uploaded_files.append({
                        'filename': uploaded_file.filename,
                        'size': len(content),
                        'storage_path': storage_path,
                        'storage_url': storage_url
                    })
                else:
                    failed_uploads.append({
                        'filename': uploaded_file.filename,
                        'reason': 'Invalid file type'
                    })
            except Exception as e:
                failed_uploads.append({
                    'filename': uploaded_file.filename,
                    'reason': str(e)
                })
        
        print(f"📁 Uploaded {len(uploaded_files)} defective images to {dirs['defective_images']}")
        
        return JSONResponse({
            'status': 'success',
            'message': f'Uploaded {len(uploaded_files)} defective images',
            'uploaded_count': len(uploaded_files),
            'failed_count': len(failed_uploads),
            'uploaded_files': uploaded_files,
            'failed_uploads': failed_uploads,
            'project_id': project_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/extract-defective-roi")
async def extract_defective_roi(
    project_id: str,
    component_description: str = Form("metal plate"),
    confidence_threshold: float = Form(0.3)
):
    """
    Step 3a: Extract ROI from uploaded defective images for anomaly detection
    
    This endpoint expects defective images to already be uploaded via the 
    upload-defective-images endpoint.
    """
    try:
        # Ensure project exists and get directories
        await ensure_project_exists(project_id)
        dirs = await get_project_directories(project_id)
        
        # Check if defective images have been uploaded
        defective_image_files = await storage_service.list_files(dirs['defective_images'], '.jpg')
        defective_image_files.extend(await storage_service.list_files(dirs['defective_images'], '.jpeg'))
        defective_image_files.extend(await storage_service.list_files(dirs['defective_images'], '.png'))
        
        if not defective_image_files:
            raise HTTPException(status_code=400, detail="No defective images found. Please upload defective images first.")
        
        print(f"📂 Found {len(defective_image_files)} defective images to process")
        
        # Create defective ROI cache directory
        os.makedirs(defective_roi_dir, exist_ok=True)
        
        print(f"📸 Processing {len(defective_image_paths)} defective images for ROI extraction")
        
        # Extract ROI from defective images using separate cache
        roi_result = auto_annotation_service.roi_extractor.extract_roi_from_defective_images(
            defective_image_paths, component_description, confidence_threshold, project_id
        )
        
        return JSONResponse(roi_result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/detect-defects")
async def detect_defects_in_images(
    project_id: str,
    method: str = Form("mahalanobis"),
    threshold_percentile: float = Form(95.0),
    model_type: str = Form("auto")  # "auto", "dinov2", or "dinov3"
):
    """
    Step 3b: Detect anomalies in defective ROI images using normal model
    
    Args:
        project_id: Project ID
        method: Anomaly detection method
        threshold_percentile: Threshold percentile for anomaly detection
        model_type: Model type ("auto" to detect automatically, "dinov2", or "dinov3")
    """
    try:
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            # Check which normal model files exist
            features_dir = os.path.join(project_dir, "anomaly_features")
            dinov3_model = os.path.join(features_dir, f"{project_id}_dinov3_normal_model.json")
            dinov2_model = os.path.join(features_dir, f"{project_id}_dinov2_normal_model.json")
            legacy_model = os.path.join(features_dir, f"{project_id}_normal_model.json")
            
            if os.path.exists(dinov3_model):
                model_type = "dinov3"
                normal_model_file = dinov3_model
            elif os.path.exists(dinov2_model):
                model_type = "dinov2" 
                normal_model_file = dinov2_model
            elif os.path.exists(legacy_model):
                model_type = "dinov2"  # Legacy files are DINOv2
                normal_model_file = legacy_model
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No normal model found. Please complete Stage 2 first."
                )
        else:
            # Use specified model type
            features_dir = os.path.join(project_dir, "anomaly_features")
            normal_model_file = os.path.join(features_dir, f"{project_id}_{model_type}_normal_model.json")
            
            if not os.path.exists(normal_model_file):
                raise HTTPException(
                    status_code=400,
                    detail=f"Normal model for {model_type} not found. Please run Stage 2 with model_type={model_type} first."
                )
        
        # Import appropriate service based on detected/specified model type
        if model_type == "dinov3":
            from backend.services.dinov3_service import dinov3_service
            # Check if DINOv3 model is actually available
            if dinov3_service.model is None:
                print("⚠️ DINOv3 model not available, falling back to DINOv2")
                from backend.services.dinov2_service import dinov2_service
                feature_service = dinov2_service
                # Don't update model_type here since we're using existing model files
            else:
                feature_service = dinov3_service
                print(f"🧠 Using DINOv3 for defect detection")
        elif model_type == "dinov2":
            from backend.services.dinov2_service import dinov2_service
            feature_service = dinov2_service
            print(f"🧠 Using DINOv2 for defect detection")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model_type: {model_type}. Must be 'dinov2' or 'dinov3'"
            )
        
        # Check if defective ROI extraction has been done
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        defective_roi_dir = os.path.join(project_dir, "defective_roi_cache")
        
        if not os.path.exists(defective_roi_dir):
            raise HTTPException(
                status_code=400, 
                detail="Defective ROI extraction not found. Please extract ROI from defective images first."
            )
        
        # Get defective ROI image paths
        defective_roi_paths = []
        for filename in os.listdir(defective_roi_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                defective_roi_paths.append(os.path.join(defective_roi_dir, filename))
        
        if not defective_roi_paths:
            raise HTTPException(
                status_code=400, 
                detail="No defective ROI images found. Please extract ROI first."
            )
        
        print(f"🎯 Detecting defects in {len(defective_roi_paths)} ROI images using {model_type.upper()}")
        print(f"🧠 Using method: {method}, threshold: {threshold_percentile}%")
        print(f"📁 Normal model file: {normal_model_file}")
        
        # Load normal model
        with open(normal_model_file, 'r') as f:
            normal_model_data = json.load(f)
        
        # Run defect detection
        result = feature_service.detect_defects_with_normal_model(
            defective_roi_paths,
            normal_model_data,
            project_id,
            method, 
            threshold_percentile
        )
        
        # Add model type to result
        if result.get('status') == 'success':
            result['model_type'] = model_type
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error detecting defects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/detect-defects-advanced")
async def detect_defects_advanced(
    project_id: str,
    ensemble_method: str = Form("majority_vote"),  # "majority_vote", "average_score", "max_score"
    methods: str = Form("ocsvm,isolation_forest,elliptic,pca"),  # comma-separated
    model_type: str = Form("auto")
):
    """
    Step 3b (Advanced): Detect anomalies using advanced ML methods with ensemble
    
    Args:
        project_id: Project ID
        ensemble_method: How to combine multiple methods
        methods: Comma-separated list of methods to use
        model_type: Feature extractor model type
    """
    try:
        from backend.services.advanced_anomaly_service import advanced_anomaly_service
        
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        
        # Auto-detect model type if needed
        if model_type == "auto":
            features_dir = os.path.join(project_dir, "anomaly_features")
            dinov3_metadata = os.path.join(features_dir, f"{project_id}_dinov3_advanced_metadata.json")
            dinov2_metadata = os.path.join(features_dir, f"{project_id}_dinov2_advanced_metadata.json")
            
            if os.path.exists(dinov3_metadata):
                model_type = "dinov3"
            elif os.path.exists(dinov2_metadata):
                model_type = "dinov2"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No advanced models found. Please run Stage 2 with detection_method='advanced' first."
                )
        
        # Check if defective ROI extraction has been done
        defective_roi_dir = os.path.join(project_dir, "defective_roi_cache")
        if not os.path.exists(defective_roi_dir):
            raise HTTPException(
                status_code=400,
                detail="Defective ROI extraction not found. Please extract ROI from defective images first."
            )
        
        # Get defective ROI image paths
        defective_roi_paths = []
        for filename in os.listdir(defective_roi_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                defective_roi_paths.append(os.path.join(defective_roi_dir, filename))
        
        if not defective_roi_paths:
            raise HTTPException(
                status_code=400,
                detail="No defective ROI images found. Please extract ROI first."
            )
        
        # Import appropriate feature service
        if model_type == "dinov3":
            from backend.services.dinov3_service import dinov3_service
            feature_service = dinov3_service
        else:
            from backend.services.dinov2_service import dinov2_service
            feature_service = dinov2_service
        
        print(f"🎯 Advanced defect detection for {len(defective_roi_paths)} ROI images")
        print(f"🔬 Using {model_type.upper()} + advanced methods: {methods}")
        print(f"🎭 Ensemble method: {ensemble_method}")
        
        # Extract features from defective ROI images
        feature_result = feature_service.extract_features(defective_roi_paths, project_id=project_id)
        if feature_result['status'] != 'success':
            raise HTTPException(status_code=500, detail=feature_result['message'])
        
        defective_features = feature_result['features']
        
        # Parse methods list
        methods_list = [method.strip() for method in methods.split(',') if method.strip()]
        
        # Run advanced anomaly detection
        advanced_result = advanced_anomaly_service.detect_anomalies_advanced(
            defective_features,
            project_id,
            methods=methods_list,
            ensemble_method=ensemble_method
        )
        
        if advanced_result['status'] != 'success':
            raise HTTPException(status_code=500, detail=advanced_result['message'])
        
        # Process results for each image
        ensemble_scores = advanced_result['ensemble_scores']  # [N, H, W]
        ensemble_anomalies = advanced_result['ensemble_anomalies']  # [N, H, W]
        
        results = []
        for i, image_path in enumerate(defective_roi_paths):
            image_scores = ensemble_scores[i]  # [H, W]
            image_anomalies = ensemble_anomalies[i]  # [H, W]
            
            # Create heatmap
            heatmap = advanced_anomaly_service._create_heatmap(image_scores)
            heatmap_base64 = advanced_anomaly_service._array_to_base64(heatmap)
            
            # Save heatmap image for visual inspection
            heatmap_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_advanced_heatmap.png"
            heatmap_saved_path = advanced_anomaly_service._save_heatmap_image(heatmap, project_id, heatmap_filename, "advanced")
            
            result = {
                'image_name': os.path.basename(image_path),
                'image_path': image_path,
                'ensemble_scores': {
                    'mean': float(np.mean(image_scores)),
                    'max': float(np.max(image_scores)),
                    'anomaly_percentage': float(np.mean(image_anomalies) * 100),
                    'num_anomaly_patches': int(np.sum(image_anomalies))
                },
                'score_map_base64': heatmap_base64,
                'heatmap_file_path': heatmap_saved_path,  # New: saved heatmap file path
                'status': 'success'
            }
            results.append(result)
        
        # Save results
        results_dir = os.path.join(project_dir, "advanced_defect_detection_results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"{project_id}_advanced_defect_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'project_id': project_id,
                'model_type': model_type,
                'detection_method': 'advanced',
                'ensemble_method': ensemble_method,
                'methods_used': advanced_result['methods_used'],
                'summary_stats': advanced_result['summary_stats'],
                'total_images': len(defective_roi_paths),
                'results': results,
                'timestamp': np.datetime64('now').item().isoformat()
            }, f, indent=2)
        
        print(f"💾 Advanced detection results saved to: {results_file}")
        
        return JSONResponse({
            'status': 'success',
            'project_id': project_id,
            'model_type': model_type,
            'detection_method': 'advanced',
            'ensemble_method': ensemble_method,
            'methods_used': advanced_result['methods_used'],
            'total_images': len(defective_roi_paths),
            'summary_stats': advanced_result['summary_stats'],
            'results': results,
            'message': f'Advanced defect detection completed using {len(advanced_result["methods_used"])} methods'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in advanced defect detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/heatmaps")
async def get_project_heatmaps(project_id: str):
    """
    Get list of all saved heatmap files for a project
    
    Returns paths to all heatmap PNG files for visual inspection
    """
    try:
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        
        if not os.path.exists(project_dir):
            raise HTTPException(status_code=404, detail="Project not found")
        
        heatmap_dirs = [
            os.path.join(project_dir, "dinov2_heatmap_analysis"),
            os.path.join(project_dir, "dinov3_heatmap_analysis"),
            os.path.join(project_dir, "advanced_heatmap_analysis")
        ]
        
        all_heatmaps = []
        
        for heatmap_dir in heatmap_dirs:
            if os.path.exists(heatmap_dir):
                method_name = os.path.basename(heatmap_dir).replace('_heatmap_analysis', '')
                
                heatmap_files = [f for f in os.listdir(heatmap_dir) if f.endswith('.png')]
                
                for heatmap_file in heatmap_files:
                    heatmap_path = os.path.join(heatmap_dir, heatmap_file)
                    file_size = os.path.getsize(heatmap_path)
                    file_mtime = os.path.getmtime(heatmap_path)
                    
                    all_heatmaps.append({
                        'filename': heatmap_file,
                        'method': method_name,
                        'file_path': heatmap_path,
                        'file_size': file_size,
                        'created_at': file_mtime,
                        'image_name': heatmap_file.replace('_heatmap.png', '').replace('_advanced_heatmap.png', '')
                    })
        
        # Sort by creation time (newest first)
        all_heatmaps.sort(key=lambda x: x['created_at'], reverse=True)
        
        return JSONResponse({
            'status': 'success',
            'project_id': project_id,
            'total_heatmaps': len(all_heatmaps),
            'heatmaps': all_heatmaps,
            'message': f'Found {len(all_heatmaps)} heatmap files'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error listing heatmaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/generate-bounding-boxes")
async def generate_bounding_boxes_from_anomalies(
    project_id: str,
    min_box_size: int = Form(32),
    anomaly_threshold: float = Form(0.05),  # Select top 5% most anomalous regions
    merge_nearby_boxes: bool = Form(True),
    output_format: str = Form("yolo"),  # 'yolo' or 'coco'
    exclude_edges: bool = Form(True),   # Skip boxes touching image edges
    edge_margin: int = Form(10),        # Pixels from edge to consider as boundary
    enable_color_filtering: bool = Form(True),  # Enable color-based filtering
    center_zone_ratio: float = Form(0.6),       # Inner ratio for center zone analysis
    min_center_variance: float = Form(0.05),    # Minimum color variance for real defects
    max_border_uniformity: float = Form(0.02),  # Maximum variance for uniform edge
    min_center_border_diff: float = Form(0.1),  # Minimum color difference center vs border
    background_match_threshold: float = Form(0.7),  # Background color match threshold
    color_tolerance_hsv: int = Form(15),        # HSV tolerance for color matching
    # Manufacturing-specific segmentation
    enable_manufacturing_segmentation: bool = Form(False),  # Enable manufacturing-specific part segmentation
    manufacturing_scenario: str = Form("general"),          # Manufacturing scenario type
    part_material: str = Form("metal"),                     # Part material type
    fixture_type: str = Form("tray"),                       # Fixture/background type
    fixture_color: str = Form("blue")                       # Fixture color
):
    """
    Step 4: Generate YOLO-ready bounding boxes from anomaly detection results
    
    Converts anomaly heatmaps into bounding box annotations that can be used
    for training object detection models.
    """
    try:
        import cv2
        import numpy as np
        from pathlib import Path
        
        # Check if project exists
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        if not os.path.exists(project_dir):
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if defect detection results exist
        results_dir = os.path.join(project_dir, "defect_detection_results")
        results_file = os.path.join(results_dir, f"{project_id}_defect_results.json")
        
        if not os.path.exists(results_file):
            raise HTTPException(status_code=400, detail="No defect detection results found. Please complete Stage 3 first.")
        
        # Load anomaly detection results
        with open(results_file, 'r') as f:
            anomaly_data = json.load(f)
        
        if not anomaly_data.get('results'):
            raise HTTPException(status_code=400, detail="No anomaly results found in detection data.")
        
        print(f"📦 Generating bounding boxes from ALL {len(anomaly_data['results'])} defective ROI images")
        print(f"⚙️ Settings: min_box_size={min_box_size}, anomaly_threshold={anomaly_threshold} (for region detection within images), format={output_format}")
        
        # Create output directories
        annotations_dir = os.path.join(project_dir, "generated_annotations")
        visual_boxes_dir = os.path.join(project_dir, "visual_bounding_boxes")
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(visual_boxes_dir, exist_ok=True)
        
        generated_boxes = []
        total_boxes_count = 0
        failed_images = []
        
        for result in anomaly_data['results']:
            try:
                image_name = result['image_name']
                image_path = result['image_path']
                
                # Process ALL defective ROI images regardless of anomaly percentage
                anomaly_pct = result['image_scores']['anomaly_percentage']
                print(f"📦 Processing {image_name}: {anomaly_pct:.2f}% anomalous patches")
                
                # Decode the heatmap from base64
                import base64
                heatmap_data = base64.b64decode(result['score_map_base64'])
                heatmap_array = np.frombuffer(heatmap_data, np.uint8)
                heatmap_img = cv2.imdecode(heatmap_array, cv2.IMREAD_COLOR)
                
                if heatmap_img is None:
                    failed_images.append({'image_name': image_name, 'error': 'Failed to decode heatmap'})
                    continue
                
                # Convert to grayscale for processing
                heatmap_gray = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)
                
                # Load ROI image (which was used for anomaly detection) to get actual dimensions
                roi_image_path = os.path.join(project_dir, "defective_roi_cache", image_name)
                if os.path.exists(roi_image_path):
                    original_img = cv2.imread(roi_image_path)
                    if original_img is not None:
                        original_height, original_width = original_img.shape[:2]
                    else:
                        # Fallback dimensions
                        original_height, original_width = 640, 640
                else:
                    # Fallback dimensions
                    original_height, original_width = 640, 640
                
                # Generate bounding boxes from heatmap
                boxes = generate_boxes_from_heatmap(
                    heatmap_gray, 
                    min_box_size, 
                    anomaly_threshold,
                    merge_nearby_boxes,
                    original_width,
                    original_height,
                    exclude_edges,
                    edge_margin,
                    original_img,  # Pass original image for color analysis
                    enable_color_filtering,
                    center_zone_ratio,
                    min_center_variance,
                    max_border_uniformity,
                    min_center_border_diff,
                    background_match_threshold,
                    color_tolerance_hsv,
                    # Manufacturing segmentation parameters
                    enable_manufacturing_segmentation,
                    manufacturing_scenario,
                    part_material,
                    fixture_type,
                    fixture_color
                )
                
                if boxes:
                    # Save annotations in requested format
                    annotation_filename = Path(image_name).stem + ".txt"
                    annotation_path = os.path.join(annotations_dir, annotation_filename)
                    
                    if output_format == "yolo":
                        save_yolo_annotations(boxes, annotation_path, original_width, original_height)
                    
                    # Create visual bounding box image using ROI image
                    visual_image_path = create_visual_bounding_box_image(
                        roi_image_path, boxes, visual_boxes_dir, image_name
                    )
                    
                    generated_boxes.append({
                        'image_name': image_name,
                        'original_size': [original_width, original_height],
                        'boxes_count': len(boxes),
                        'boxes': boxes,
                        'annotation_file': annotation_filename,
                        'visual_image_path': visual_image_path,
                        'anomaly_percentage': result['image_scores']['anomaly_percentage']
                    })
                    
                    total_boxes_count += len(boxes)
                    print(f"📦 Generated {len(boxes)} boxes for {image_name}")
                else:
                    print(f"⚠️  No boxes generated for {image_name}")
                    
            except Exception as e:
                print(f"❌ Error processing {result.get('image_name', 'unknown')}: {e}")
                failed_images.append({'image_name': result.get('image_name', 'unknown'), 'error': str(e)})
        
        # Create summary file
        summary_data = {
            'project_id': project_id,
            'generation_timestamp': np.datetime64('now').item().isoformat(),
            'settings': {
                'min_box_size': min_box_size,
                'anomaly_threshold': anomaly_threshold,
                'merge_nearby_boxes': merge_nearby_boxes,
                'output_format': output_format
            },
            'statistics': {
                'total_images_processed': len(anomaly_data['results']),
                'images_with_boxes': len(generated_boxes),
                'total_boxes_generated': total_boxes_count,
                'failed_images': len(failed_images)
            },
            'generated_boxes': generated_boxes,
            'failed_images': failed_images
        }
        
        summary_file = os.path.join(annotations_dir, f"{project_id}_bounding_boxes_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"✅ Generated {total_boxes_count} bounding boxes for {len(generated_boxes)} images")
        
        return JSONResponse({
            'status': 'success',
            'message': f'Generated {total_boxes_count} bounding boxes for {len(generated_boxes)} images',
            'total_images_processed': len(anomaly_data['results']),
            'images_with_boxes': len(generated_boxes),
            'total_boxes_generated': total_boxes_count,
            'failed_images_count': len(failed_images),
            'annotations_directory': annotations_dir,
            'summary_file': summary_file,
            'generated_boxes': generated_boxes[:10],  # Return first 10 for preview
            'settings_used': summary_data['settings']
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating bounding boxes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/generate-segmentation-masks")
async def generate_segmentation_masks_from_anomalies(
    project_id: str,
    min_mask_area: int = Form(200),  # Minimum area for segmentation masks
    anomaly_threshold: float = Form(0.05),  # Select top 5% most anomalous regions
    output_format: str = Form("yolo"),  # 'yolo' or 'coco'
    exclude_edges: bool = Form(True),
    edge_margin: int = Form(10),
    simplify_polygons: bool = Form(True),  # Simplify polygon coordinates
    polygon_epsilon: float = Form(0.01)  # Polygon simplification factor
):
    """
    Step 4 Alternative: Generate segmentation masks from anomaly detection results
    Creates YOLO segmentation format with polygon coordinates for each defect
    """
    try:
        print(f"🎭 Generating segmentation masks for project: {project_id}")
        
        # Load anomaly detection results 
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        defect_results_dir = os.path.join(project_dir, "defect_detection_results")
        anomaly_results_file = os.path.join(defect_results_dir, f"{project_id}_defect_results.json")
        
        if not os.path.exists(anomaly_results_file):
            raise HTTPException(
                status_code=400, 
                detail="No anomaly detection results found. Please run Step 3 (detect defects) first."
            )
        
        with open(anomaly_results_file, 'r') as f:
            anomaly_data = json.load(f)
        
        print(f"📊 Processing {len(anomaly_data['results'])} defective images")
        
        # Create annotations directory
        annotations_dir = os.path.join(project_dir, "segmentation_annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Create visual output directory
        visual_dir = os.path.join(project_dir, "visual_segmentation_masks")
        os.makedirs(visual_dir, exist_ok=True)
        
        generated_masks = []
        total_masks_count = 0
        failed_images = []
        
        for result in anomaly_data['results']:
            try:
                image_name = result['image_name']
                print(f"🎭 Processing segmentation masks for: {image_name}")
                
                # Decode heatmap
                heatmap_data = base64.b64decode(result['score_map_base64'])
                heatmap_array = np.frombuffer(heatmap_data, np.uint8)
                heatmap_img = cv2.imdecode(heatmap_array, cv2.IMREAD_COLOR)
                heatmap_gray = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)
                
                # Load original ROI image for dimensions and visualization
                roi_dirs = get_roi_directories(project_dir)
                roi_image_path = os.path.join(roi_dirs['defective'], image_name)
                
                if not os.path.exists(roi_image_path):
                    print(f"⚠️ ROI image not found: {roi_image_path}")
                    continue
                
                original_img = cv2.imread(roi_image_path)
                original_height, original_width = original_img.shape[:2]
                
                # Generate segmentation masks from heatmap
                masks = generate_masks_from_heatmap(
                    heatmap_gray, 
                    min_mask_area,
                    anomaly_threshold,
                    original_width, 
                    original_height,
                    exclude_edges,
                    edge_margin
                )
                
                if not masks:
                    print(f"⚠️ No masks generated for {image_name}")
                    continue
                
                # Convert masks to YOLO segmentation format
                yolo_annotations = []
                for mask_idx, mask_contour in enumerate(masks):
                    # Convert contour to normalized polygon coordinates
                    polygon_points = []
                    for point in mask_contour:
                        x_norm = point[0] / original_width
                        y_norm = point[1] / original_height
                        polygon_points.extend([x_norm, y_norm])
                    
                    # Simplify polygon if requested
                    if simplify_polygons and len(polygon_points) > 12:  # More than 6 points
                        # Convert back to contour format for simplification
                        points_array = np.array([(polygon_points[i], polygon_points[i+1]) 
                                               for i in range(0, len(polygon_points), 2)])
                        points_array = points_array * np.array([original_width, original_height])
                        points_array = points_array.astype(np.int32).reshape(-1, 1, 2)
                        
                        # Simplify using Douglas-Peucker algorithm
                        epsilon = polygon_epsilon * cv2.arcLength(points_array, True)
                        simplified = cv2.approxPolyDP(points_array, epsilon, True)
                        
                        # Convert back to normalized coordinates
                        polygon_points = []
                        for point in simplified.reshape(-1, 2):
                            x_norm = point[0] / original_width
                            y_norm = point[1] / original_height
                            polygon_points.extend([x_norm, y_norm])
                    
                    # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                    class_id = 0  # Default defect class
                    yolo_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in polygon_points)
                    yolo_annotations.append(yolo_line)
                
                # Save YOLO annotation file
                annotation_filename = f"{os.path.splitext(image_name)[0]}.txt"
                annotation_path = os.path.join(annotations_dir, annotation_filename)
                
                with open(annotation_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                # Create visual overlay
                visual_img = original_img.copy()
                for mask_idx, mask_contour in enumerate(masks):
                    # Draw filled mask
                    mask_overlay = np.zeros((original_height, original_width), dtype=np.uint8)
                    cv2.fillPoly(mask_overlay, [mask_contour.astype(np.int32)], 255)
                    
                    # Add colored overlay
                    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    mask_color = color[mask_idx % len(color)]
                    visual_img[mask_overlay > 0] = np.array(mask_color) * 0.3 + visual_img[mask_overlay > 0] * 0.7
                    
                    # Draw contour outline
                    cv2.drawContours(visual_img, [mask_contour.astype(np.int32)], -1, mask_color, 2)
                
                # Add title
                cv2.putText(visual_img, f"Found {len(masks)} segmentation masks", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save visual result
                visual_filename = f"visual_{image_name}"
                visual_path = os.path.join(visual_dir, visual_filename)
                cv2.imwrite(visual_path, visual_img)
                
                # Record success
                generated_masks.append({
                    'image_name': image_name,
                    'original_size': [original_width, original_height],
                    'masks_count': len(masks),
                    'annotation_file': annotation_filename,
                    'visual_file': visual_filename
                })
                
                total_masks_count += len(masks)
                print(f"✅ Generated {len(masks)} masks for {image_name}")
                
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
                failed_images.append({
                    'image_name': image_name,
                    'error': str(e)
                })
        
        # Save summary
        summary_data = {
            'project_id': project_id,
            'annotation_type': 'segmentation_mask',
            'total_images_processed': len(anomaly_data['results']),
            'images_with_masks': len(generated_masks),
            'total_masks_generated': total_masks_count,
            'failed_images': len(failed_images),
            'settings': {
                'min_mask_area': min_mask_area,
                'anomaly_threshold': anomaly_threshold,
                'output_format': output_format,
                'exclude_edges': exclude_edges,
                'edge_margin': edge_margin,
                'simplify_polygons': simplify_polygons,
                'polygon_epsilon': polygon_epsilon
            },
            'statistics': {
                'images_with_masks': len(generated_masks),
                'total_masks_generated': total_masks_count,
                'failed_images': len(failed_images)
            },
            'generated_masks': generated_masks,
            'failed_images': failed_images
        }
        
        summary_file = os.path.join(annotations_dir, f"{project_id}_segmentation_masks_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"✅ Generated {total_masks_count} segmentation masks for {len(generated_masks)} images")
        
        return JSONResponse({
            'status': 'success',
            'message': f'Generated {total_masks_count} segmentation masks for {len(generated_masks)} images',
            'total_images_processed': len(anomaly_data['results']),
            'images_with_masks': len(generated_masks),
            'total_masks_generated': total_masks_count,
            'failed_images_count': len(failed_images),
            'annotations_directory': annotations_dir,
            'visual_directory': visual_dir,
            'summary_file': summary_file,
            'generated_masks': generated_masks[:10],  # Return first 10 for preview
            'settings_used': summary_data['settings']
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating segmentation masks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_masks_from_heatmap(heatmap_gray, min_mask_area, threshold, orig_width, orig_height, exclude_edges, edge_margin):
    """Generate segmentation masks from anomaly heatmap"""
    try:
        import cv2
        import numpy as np
        
        print(f"🎭 Generating segmentation masks from heatmap")
        print(f"   Heatmap stats: min={heatmap_gray.min()}, max={heatmap_gray.max()}, mean={heatmap_gray.mean():.1f}")
        
        # Step 1: Convert threshold to percentile
        if threshold > 1.0:
            threshold_value = np.percentile(heatmap_gray, threshold)
        else:
            percentile = (1.0 - threshold) * 100
            threshold_value = np.percentile(heatmap_gray, percentile)
        
        print(f"   Using threshold value: {threshold_value:.1f}")
        
        # Step 2: Create binary mask
        binary_mask = (heatmap_gray >= threshold_value).astype(np.uint8) * 255
        
        anomalous_pixels = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size
        print(f"   Hot pixels: {anomalous_pixels}/{total_pixels} ({anomalous_pixels/total_pixels*100:.1f}%)")
        
        # Step 3: Group nearby regions with morphological operations
        small_kernel = np.ones((3, 3), np.uint8)
        large_kernel = np.ones((15, 15), np.uint8)
        
        # Remove tiny noise
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
        
        # Connect nearby regions
        grouped_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, large_kernel, iterations=2)
        
        # Step 4: Find contours for segmentation
        contours, _ = cv2.findContours(grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   Found {len(contours)} potential mask regions")
        
        # Step 5: Filter and process contours
        masks = []
        scale_x = orig_width / heatmap_gray.shape[1]
        scale_y = orig_height / heatmap_gray.shape[0]
        
        for i, contour in enumerate(contours):
            # Filter by area
            contour_area = cv2.contourArea(contour)
            if contour_area < min_mask_area:
                print(f"   Mask {i+1}: Filtered out (area={contour_area:.0f} < {min_mask_area})")
                continue
            
            # Scale contour to original image size
            scaled_contour = contour.astype(np.float32)
            scaled_contour[:, :, 0] *= scale_x  # x coordinates
            scaled_contour[:, :, 1] *= scale_y  # y coordinates
            
            # Edge exclusion
            if exclude_edges:
                # Check if any point is near edges
                x_coords = scaled_contour[:, :, 0].flatten()
                y_coords = scaled_contour[:, :, 1].flatten()
                
                if (np.any(x_coords < edge_margin) or np.any(x_coords > orig_width - edge_margin) or
                    np.any(y_coords < edge_margin) or np.any(y_coords > orig_height - edge_margin)):
                    print(f"   Mask {i+1}: Filtered out (touches edges)")
                    continue
            
            # Reshape contour for polygon format
            mask_contour = scaled_contour.reshape(-1, 2)
            masks.append(mask_contour)
            print(f"   Mask {i+1}: ✅ Accepted - area={contour_area:.0f}, {len(mask_contour)} points")
        
        print(f"   🎯 Generated {len(masks)} segmentation masks")
        return masks
        
    except Exception as e:
        print(f"❌ Error generating masks from heatmap: {e}")
        return []

def generate_boxes_from_heatmap(heatmap_gray, min_box_size, threshold, merge_nearby, orig_width, orig_height, exclude_edges=True, edge_margin=10, original_image=None, enable_color_filtering=True, center_zone_ratio=0.6, min_center_variance=0.05, max_border_uniformity=0.02, min_center_border_diff=0.1, background_match_threshold=0.7, color_tolerance_hsv=15, enable_manufacturing_segmentation=False, manufacturing_scenario="general", part_material="metal", fixture_type="tray", fixture_color="blue", use_simple_heat_following=True):
    """Generate bounding boxes from anomaly heatmap using simple conventional approach"""
    try:
        import cv2
        import numpy as np
        
        if use_simple_heat_following:
            print(f"🔥 Simple heat-following approach")
        else:
            print(f"📦 Advanced filtering approach")
        print(f"   Heatmap stats: min={heatmap_gray.min()}, max={heatmap_gray.max()}, mean={heatmap_gray.mean():.1f}")
        
        # Simple Heat-Following Approach (Trust the Heatmap)
        if use_simple_heat_following:
            return generate_boxes_simple_heat_following(
                heatmap_gray, min_box_size, threshold, merge_nearby, 
                orig_width, orig_height, exclude_edges, edge_margin
            )
        
        # Step 1: Convert threshold to actual pixel value (more conservative)
        if threshold > 1.0:
            # If threshold > 1, treat as percentile (e.g., 95.0 = 95th percentile) 
            threshold_value = np.percentile(heatmap_gray, threshold)
        else:
            # If threshold <= 1, convert to percentile
            # threshold=0.1 means "top 10% most anomalous" -> 90th percentile
            percentile = (1.0 - threshold) * 100
            threshold_value = np.percentile(heatmap_gray, percentile)
        
        print(f"   Using threshold value: {threshold_value:.1f}")
        
        # Step 2: Create binary mask - pixels above threshold become white (255)
        binary_mask = (heatmap_gray >= threshold_value).astype(np.uint8) * 255
        
        anomalous_pixels = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size
        print(f"   Anomalous pixels: {anomalous_pixels}/{total_pixels} ({anomalous_pixels/total_pixels*100:.1f}%)")
        
        # Step 3: Aggressive morphological operations to connect nearby hot regions
        # Use larger kernels to group nearby anomalous areas together
        small_kernel = np.ones((3, 3), np.uint8)
        large_kernel = np.ones((15, 15), np.uint8)  # Much larger kernel for grouping
        
        # First: Remove tiny noise
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
        
        # Second: CONNECT nearby regions with large closing operation
        grouped_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, large_kernel, iterations=2)
        
        print(f"   After grouping: reduced regions by connecting nearby areas")
        
        # Step 4: Find contours from grouped regions
        contours, _ = cv2.findContours(grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   Found {len(contours)} grouped regions")
        
        # Step 5: Convert contours to bounding boxes (focus on main regions only)
        boxes = []
        scale_x = orig_width / heatmap_gray.shape[1]
        scale_y = orig_height / heatmap_gray.shape[0]
        
        print(f"   Processing {len(contours)} candidate regions...")
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Enhanced contour filtering for main defect regions
            contour_area = cv2.contourArea(contour)
            
            # Filter 1: Much higher minimum area for main defects only
            min_area = max(200, min_box_size * 2)  # Focus on significant defect regions only
            if contour_area < min_area:
                print(f"   Region {i+1}: Filtered out (area={contour_area:.0f} < {min_area})")
                continue
            
            # Filter 2: Aspect ratio (remove very thin/elongated shapes - likely noise)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > 8:  # Skip very elongated shapes
                continue
            
            # Filter 3: Density check - contour should fill reasonable portion of bounding box
            bbox_area = w * h
            if bbox_area > 0:
                density = contour_area / bbox_area
                if density < 0.15:  # Contour fills less than 15% of bounding box = likely noise
                    continue
            
            # Scale to original image size
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x) 
            h = int(h * scale_y)
            
            # Ensure minimum box size
            if w < min_box_size:
                expand = (min_box_size - w) // 2
                x = max(0, x - expand)
                w = min(orig_width - x, min_box_size)
            if h < min_box_size:
                expand = (min_box_size - h) // 2
                y = max(0, y - expand)
                h = min(orig_height - y, min_box_size)
            
            # Enhanced edge exclusion (more aggressive to reduce fixture noise)
            if exclude_edges:
                # Use larger edge margin to exclude more fixture/background noise
                effective_edge_margin = max(edge_margin, min_box_size)  # At least as large as min_box_size
                if (x < effective_edge_margin or y < effective_edge_margin or 
                    x + w > orig_width - effective_edge_margin or y + h > orig_height - effective_edge_margin):
                    continue
            
            boxes.append([x, y, w, h])
            print(f"   Region {i+1}: ✅ Accepted - area={contour_area:.0f}, size={w}x{h}")
        
        print(f"   🎯 Generated {len(boxes)} main defect bounding boxes (was {len(contours)} regions)")
        
        # Step 6: Merge nearby/overlapping boxes if requested
        if merge_nearby and len(boxes) > 1:
            boxes = merge_nearby_boxes(boxes)
            print(f"   After merging: {len(boxes)} final boxes")
        
        return boxes
        
    except Exception as e:
        print(f"❌ Error in generate_boxes_from_heatmap: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_boxes_simple_heat_following(heatmap_gray, min_box_size, threshold, merge_nearby, orig_width, orig_height, exclude_edges, edge_margin):
    """Simple approach: Just draw boxes around the hottest regions in the heatmap"""
    try:
        import cv2
        import numpy as np
        
        # Step 1: Convert threshold to percentile 
        if threshold > 1.0:
            threshold_value = np.percentile(heatmap_gray, threshold)
        else:
            percentile = (1.0 - threshold) * 100
            threshold_value = np.percentile(heatmap_gray, percentile)
        
        print(f"   Following heat above: {threshold_value:.1f}")
        
        # Step 2: Create binary mask - find hot spots
        binary_mask = (heatmap_gray >= threshold_value).astype(np.uint8) * 255
        
        anomalous_pixels = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size
        print(f"   Hot pixels: {anomalous_pixels}/{total_pixels} ({anomalous_pixels/total_pixels*100:.1f}%)")
        
        # Step 3: Minimal cleanup - just remove tiny noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Step 4: Find hot regions (contours)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   Found {len(contours)} hot regions")
        
        # Step 5: Draw boxes around hot regions (minimal filtering)
        boxes = []
        scale_x = orig_width / heatmap_gray.shape[1]
        scale_y = orig_height / heatmap_gray.shape[0]
        
        for contour in contours:
            # Get bounding rectangle around hot region
            x, y, w, h = cv2.boundingRect(contour)
            
            # Only filter out extremely tiny regions (trust the heatmap otherwise)
            contour_area = cv2.contourArea(contour)
            if contour_area < 4:  # Less than 2x2 pixels
                continue
            
            # Scale to original image size
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x) 
            h = int(h * scale_y)
            
            # Ensure minimum box size
            if w < min_box_size:
                expand = (min_box_size - w) // 2
                x = max(0, x - expand)
                w = min(orig_width - x, min_box_size)
            if h < min_box_size:
                expand = (min_box_size - h) // 2
                y = max(0, y - expand)
                h = min(orig_height - y, min_box_size)
            
            # Basic edge exclusion only
            if exclude_edges:
                if (x < edge_margin or y < edge_margin or 
                    x + w > orig_width - edge_margin or y + h > orig_height - edge_margin):
                    continue
            
            boxes.append([x, y, w, h])
        
        print(f"   Generated {len(boxes)} heat-following boxes")
        
        # Merge nearby boxes if requested
        if merge_nearby and len(boxes) > 1:
            boxes = merge_nearby_boxes(boxes)
            print(f"   After merging: {len(boxes)} final boxes")
        
        return boxes
        
    except Exception as e:
        print(f"❌ Error in simple heat following: {e}")
        import traceback
        traceback.print_exc()
        return []

def merge_nearby_boxes(boxes, overlap_threshold=0.3):
    """Merge overlapping/nearby bounding boxes"""
    try:
        if len(boxes) <= 1:
            return boxes
        
        import numpy as np
        
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            
            # Start with current box
            merged_box = box1[:]
            used.add(i)
            
            # Check for boxes to merge
            for j, box2 in enumerate(boxes):
                if j in used or i == j:
                    continue
                
                # Calculate IoU or overlap
                if calculate_box_overlap(box1, box2) > overlap_threshold:
                    # Merge boxes by taking min/max coordinates
                    x1_min = min(merged_box[0], box2[0])
                    y1_min = min(merged_box[1], box2[1])
                    x1_max = max(merged_box[0] + merged_box[2], box2[0] + box2[2])
                    y1_max = max(merged_box[1] + merged_box[3], box2[1] + box2[3])
                    
                    merged_box = [x1_min, y1_min, x1_max - x1_min, y1_max - y1_min]
                    used.add(j)
            
            merged.append(merged_box)
        
        return merged
        
    except Exception as e:
        print(f"❌ Error in merge_nearby_boxes: {e}")
        return boxes

def extract_background_colors(image, sample_size=10, edge_strip_ratio=0.05):
    """Extract background colors from image corners and edges"""
    try:
        if image is None:
            return []
        
        h, w = image.shape[:2]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        background_colors = []
        
        # Sample from corners (4 corners, sample_size x sample_size)
        corners = [
            (0, 0),  # Top-left
            (w - sample_size, 0),  # Top-right
            (0, h - sample_size),  # Bottom-left
            (w - sample_size, h - sample_size)  # Bottom-right
        ]
        
        for x, y in corners:
            corner_region = hsv_image[y:y+sample_size, x:x+sample_size]
            if corner_region.size > 0:
                mean_color = np.mean(corner_region.reshape(-1, 3), axis=0)
                background_colors.append(mean_color)
        
        # Sample from edge strips
        edge_width = int(min(w, h) * edge_strip_ratio)
        if edge_width > 0:
            # Top edge
            top_strip = hsv_image[0:edge_width, :]
            if top_strip.size > 0:
                mean_color = np.mean(top_strip.reshape(-1, 3), axis=0)
                background_colors.append(mean_color)
            
            # Bottom edge
            bottom_strip = hsv_image[h-edge_width:h, :]
            if bottom_strip.size > 0:
                mean_color = np.mean(bottom_strip.reshape(-1, 3), axis=0)
                background_colors.append(mean_color)
            
            # Left edge
            left_strip = hsv_image[:, 0:edge_width]
            if left_strip.size > 0:
                mean_color = np.mean(left_strip.reshape(-1, 3), axis=0)
                background_colors.append(mean_color)
            
            # Right edge
            right_strip = hsv_image[:, w-edge_width:w]
            if right_strip.size > 0:
                mean_color = np.mean(right_strip.reshape(-1, 3), axis=0)
                background_colors.append(mean_color)
        
        return background_colors
        
    except Exception as e:
        print(f"⚠️ Error extracting background colors: {e}")
        return []

def analyze_center_edge_contrast(roi_hsv, center_zone_ratio=0.6):
    """Analyze color contrast between center and edge zones of ROI"""
    try:
        if roi_hsv is None or roi_hsv.size == 0:
            return {'center_variance': 0, 'border_variance': 0, 'center_border_diff': 0}
        
        h, w = roi_hsv.shape[:2]
        
        # Define center zone (inner area)
        center_h = int(h * center_zone_ratio)
        center_w = int(w * center_zone_ratio)
        center_y = (h - center_h) // 2
        center_x = (w - center_w) // 2
        
        # Extract center and border zones
        center_zone = roi_hsv[center_y:center_y+center_h, center_x:center_x+center_w]
        
        # Create border mask (everything except center)
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[center_y:center_y+center_h, center_x:center_x+center_w] = False
        border_zone = roi_hsv[border_mask]
        
        if center_zone.size == 0 or border_zone.size == 0:
            return {'center_variance': 0, 'border_variance': 0, 'center_border_diff': 0}
        
        # Calculate color variance in each zone
        center_variance = np.mean(np.std(center_zone.reshape(-1, 3), axis=0))
        border_variance = np.mean(np.std(border_zone.reshape(-1, 3), axis=0))
        
        # Calculate color difference between center and border means
        center_mean = np.mean(center_zone.reshape(-1, 3), axis=0)
        border_mean = np.mean(border_zone.reshape(-1, 3), axis=0)
        center_border_diff = np.linalg.norm(center_mean - border_mean) / 255.0  # Normalize
        
        return {
            'center_variance': center_variance / 255.0,  # Normalize to 0-1
            'border_variance': border_variance / 255.0,
            'center_border_diff': center_border_diff
        }
        
    except Exception as e:
        print(f"⚠️ Error analyzing center-edge contrast: {e}")
        return {'center_variance': 0, 'border_variance': 0, 'center_border_diff': 0}

def check_background_color_match(roi_hsv, background_colors, color_tolerance_hsv=15, match_threshold=0.7):
    """Check if ROI matches background colors"""
    try:
        if roi_hsv is None or roi_hsv.size == 0 or not background_colors:
            return False
        
        roi_pixels = roi_hsv.reshape(-1, 3)
        total_pixels = len(roi_pixels)
        
        if total_pixels == 0:
            return False
        
        matched_pixels = 0
        tolerance = np.array([color_tolerance_hsv, 50, 50])  # H, S, V tolerances
        
        for bg_color in background_colors:
            # Check how many pixels are within tolerance of this background color
            diff = np.abs(roi_pixels - bg_color)
            # Handle hue wraparound (0-179 in OpenCV HSV)
            diff[:, 0] = np.minimum(diff[:, 0], 180 - diff[:, 0])
            
            matches = np.all(diff <= tolerance, axis=1)
            matched_pixels += np.sum(matches)
        
        match_ratio = matched_pixels / total_pixels
        return match_ratio >= match_threshold
        
    except Exception as e:
        print(f"⚠️ Error checking background color match: {e}")
        return False

def create_manufacturing_part_mask(image, scenario, part_material, fixture_type, fixture_color):
    """
    Create part mask based on manufacturing scenario configuration
    
    Args:
        image: Original BGR image
        scenario: Manufacturing scenario type
        part_material: Type of part material
        fixture_type: Type of fixture/background
        fixture_color: Color of fixture
        
    Returns:
        Binary mask where 255 = part area, 0 = background area
    """
    try:
        if image is None:
            return None
        
        print(f"   🏭 Manufacturing segmentation: {scenario} - {part_material} on {fixture_color} {fixture_type}")
        
        # Route to specific segmentation method based on configuration
        if scenario == "metal_machining" or (part_material == "metal" and fixture_color == "blue"):
            return segment_metal_on_blue_tray(image)
        elif scenario == "electronics" or (part_material == "electronic" and fixture_type == "tray"):
            return segment_electronics_on_tray(image)
        elif fixture_color in ["blue", "dark_blue", "light_blue"]:
            return segment_parts_on_blue_background(image)
        elif fixture_color in ["black", "dark"]:
            return segment_parts_on_dark_background(image)
        elif fixture_color in ["white", "light"]:
            return segment_parts_on_light_background(image)
        else:
            # Fallback: intensity-based segmentation
            return segment_parts_intensity_based(image)
            
    except Exception as e:
        print(f"⚠️ Error in manufacturing segmentation: {e}")
        return None

def segment_metal_on_blue_tray(image):
    """Segment metal parts on blue tray - optimized HSV approach"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Refined blue detection for tray
        blue_mask = cv2.inRange(hsv, np.array([95, 30, 20]), np.array([135, 255, 255]))
        
        # Dark areas (shadows, deep recesses)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_mask = (gray < 25).astype(np.uint8) * 255
        
        # Combine background areas
        background_mask = cv2.bitwise_or(blue_mask, dark_mask)
        
        # Clean background mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
        
        # Part mask is inverse
        part_mask = cv2.bitwise_not(background_mask)
        
        # Clean part mask
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_OPEN, kernel)
        
        return part_mask
        
    except Exception as e:
        print(f"⚠️ Metal-blue segmentation failed: {e}")
        return None

def segment_electronics_on_tray(image):
    """Segment electronic components on tray"""
    try:
        # Electronics are usually dark, trays are lighter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu thresholding to separate dark components from light tray
        _, part_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_CLOSE, kernel)
        
        return part_mask
        
    except Exception as e:
        print(f"⚠️ Electronics segmentation failed: {e}")
        return None

def segment_parts_on_blue_background(image):
    """General blue background segmentation"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Broader blue range
        blue_mask = cv2.inRange(hsv, np.array([90, 20, 10]), np.array([140, 255, 255]))
        
        # Invert to get part mask
        part_mask = cv2.bitwise_not(blue_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_CLOSE, kernel)
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_OPEN, kernel)
        
        return part_mask
        
    except Exception as e:
        print(f"⚠️ Blue background segmentation failed: {e}")
        return None

def segment_parts_on_dark_background(image):
    """Segment parts on dark background"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Parts are lighter than dark background
        _, part_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_OPEN, kernel)
        
        return part_mask
        
    except Exception as e:
        print(f"⚠️ Dark background segmentation failed: {e}")
        return None

def segment_parts_on_light_background(image):
    """Segment parts on light/white background"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Parts are darker than light background
        _, part_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_OPEN, kernel)
        
        return part_mask
        
    except Exception as e:
        print(f"⚠️ Light background segmentation failed: {e}")
        return None

def segment_parts_intensity_based(image):
    """Fallback: intensity-based segmentation"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding
        part_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_CLOSE, kernel)
        
        return part_mask
        
    except Exception as e:
        print(f"⚠️ Intensity-based segmentation failed: {e}")
        return None

def should_filter_box_by_color(box, original_image, heatmap_gray, background_colors, 
                              center_zone_ratio, min_center_variance, max_border_uniformity, 
                              min_center_border_diff, background_match_threshold, color_tolerance_hsv):
    """Determine if box should be filtered out based on color analysis"""
    try:
        if original_image is None:
            return False
        
        x, y, w, h = box
        
        # Scale box coordinates from heatmap to original image
        scale_x = original_image.shape[1] / heatmap_gray.shape[1]
        scale_y = original_image.shape[0] / heatmap_gray.shape[0]
        
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        orig_w = int(w * scale_x)
        orig_h = int(h * scale_y)
        
        # Ensure ROI is within image bounds
        orig_x = max(0, min(orig_x, original_image.shape[1] - 1))
        orig_y = max(0, min(orig_y, original_image.shape[0] - 1))
        orig_w = min(orig_w, original_image.shape[1] - orig_x)
        orig_h = min(orig_h, original_image.shape[0] - orig_y)
        
        if orig_w <= 0 or orig_h <= 0:
            return True  # Invalid ROI, filter out
        
        # Extract ROI from original image
        roi_bgr = original_image[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Approach B: Center vs Edge Contrast Analysis
        contrast_analysis = analyze_center_edge_contrast(roi_hsv, center_zone_ratio)
        
        # Filter if low center variance (uniform) AND low border variance (uniform edge)
        is_uniform_edge = (contrast_analysis['center_variance'] < min_center_variance and 
                          contrast_analysis['border_variance'] < max_border_uniformity and
                          contrast_analysis['center_border_diff'] < min_center_border_diff)
        
        if is_uniform_edge:
            print(f"   🎨 Filtering uniform edge box: center_var={contrast_analysis['center_variance']:.3f}, border_var={contrast_analysis['border_variance']:.3f}, diff={contrast_analysis['center_border_diff']:.3f}")
            return True
        
        # Approach C: Background Color Rejection
        is_background_match = check_background_color_match(
            roi_hsv, background_colors, color_tolerance_hsv, background_match_threshold
        )
        
        if is_background_match:
            print(f"   🎨 Filtering background color match: matches {len(background_colors)} background colors")
            return True
        
        print(f"   ✅ Keeping box: center_var={contrast_analysis['center_variance']:.3f}, border_var={contrast_analysis['border_variance']:.3f}, bg_match={is_background_match}")
        return False
        
    except Exception as e:
        print(f"⚠️ Error in color filtering: {e}")
        return False  # Don't filter if there's an error

def calculate_box_overlap(box1, box2):
    """Calculate overlap ratio between two boxes"""
    try:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    except:
        return 0

def save_yolo_annotations(boxes, annotation_path, img_width, img_height):
    """Save bounding boxes in YOLO format"""
    try:
        with open(annotation_path, 'w') as f:
            for box in boxes:
                x, y, w, h = box
                
                # Convert to YOLO format (normalized center coordinates)
                center_x = (x + w/2) / img_width
                center_y = (y + h/2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # Class ID 0 for "defect"
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
    except Exception as e:
        print(f"❌ Error saving YOLO annotations: {e}")

def create_visual_bounding_box_image(original_image_path, boxes, output_dir, image_name):
    """Create an image with bounding boxes drawn on it for visual verification"""
    try:
        import cv2
        import random
        from pathlib import Path
        
        # Load original image
        image = cv2.imread(original_image_path)
        if image is None:
            print(f"⚠️ Could not load image: {original_image_path}")
            return None
        
        # Make a copy for drawing
        visual_image = image.copy()
        
        # Colors for bounding boxes (BGR format for OpenCV)
        colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red  
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        
        # Draw bounding boxes
        for i, box in enumerate(boxes):
            x, y, w, h = box
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(visual_image, (x, y), (x + w, y + h), color, 3)
            
            # Add defect label
            label = f"Defect {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(visual_image, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), 
                         color, -1)
            
            # White text
            cv2.putText(visual_image, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add summary info
        info_text = f"Found {len(boxes)} defects"
        cv2.putText(visual_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save visual image
        visual_filename = f"visual_{Path(image_name).stem}.jpg"
        visual_path = os.path.join(output_dir, visual_filename)
        
        success = cv2.imwrite(visual_path, visual_image)
        if success:
            print(f"💾 Saved visual bounding boxes: {visual_filename}")
            return visual_path
        else:
            print(f"❌ Failed to save visual image: {visual_filename}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating visual bounding box image: {e}")
        return None

@router.get("/projects/{project_id}/download-annotations")
async def download_generated_annotations(
    project_id: str, 
    annotation_type: str = "bounding_boxes"  # "bounding_boxes" or "segmentation_masks"
):
    """
    Download generated annotations as a ZIP file
    Supports both bounding box and segmentation mask annotations
    """
    try:
        import zipfile
        import io
        from fastapi.responses import StreamingResponse
        
        # Check if project exists
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        if not os.path.exists(project_dir):
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Set directories based on annotation type
        if annotation_type == "segmentation_masks":
            annotations_dir = os.path.join(project_dir, "segmentation_annotations")
            visual_dir = os.path.join(project_dir, "visual_segmentation_masks")
            error_msg = "No segmentation mask annotations found. Please run Step 4 with mask generation first."
            filename_prefix = "segmentation_masks"
        else:  # bounding_boxes (default)
            annotations_dir = os.path.join(project_dir, "generated_annotations")
            visual_dir = os.path.join(project_dir, "visual_bounding_boxes")
            error_msg = "No bounding box annotations found. Please run Step 4 with box generation first."
            filename_prefix = "bounding_boxes"
        
        if not os.path.exists(annotations_dir):
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all annotation files
            for root, dirs, files in os.walk(annotations_dir):
                for file in files:
                    if file.endswith(('.txt', '.json')):
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('annotations', os.path.relpath(file_path, annotations_dir))
                        zip_file.write(file_path, arcname)
            
            # Add visual images if they exist
            if os.path.exists(visual_dir):
                for root, dirs, files in os.walk(visual_dir):
                    for file in files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            file_path = os.path.join(root, file)
                            arcname = os.path.join('visual_images', os.path.relpath(file_path, visual_dir))
                            zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        
        # Return ZIP file as streaming response
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={project_id}_{filename_prefix}.zip"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/visual-boxes-preview")
async def get_visual_boxes_preview(project_id: str, limit: int = 6):
    """
    Get preview of generated visual bounding box images
    """
    try:
        import base64
        from pathlib import Path
        
        # Get project visual boxes directory
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        visual_boxes_dir = os.path.join(project_dir, "visual_bounding_boxes")
        
        if not os.path.exists(visual_boxes_dir):
            return JSONResponse({
                'status': 'error',
                'message': 'No visual bounding box images found. Generate boxes first.'
            })
        
        # Find visual box images
        visual_files = []
        for filename in os.listdir(visual_boxes_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                visual_files.append(filename)
        
        # Limit number of preview images
        visual_files = sorted(visual_files)[:limit]
        
        # Convert images to base64 for preview
        preview_images = []
        for visual_filename in visual_files:
            visual_path = os.path.join(visual_boxes_dir, visual_filename)
            
            try:
                with open(visual_path, 'rb') as f:
                    image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    preview_images.append({
                        'filename': visual_filename,
                        'image_base64': image_base64,
                        'original_name': visual_filename.replace('visual_', '').replace('.jpg', '')
                    })
            except Exception as e:
                print(f"⚠️ Error reading visual image {visual_filename}: {e}")
                continue
        
        return JSONResponse({
            'status': 'success',
            'preview_images': preview_images,
            'total_previews': len(preview_images)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/workflow-status")
async def get_workflow_status(project_id: str):
    """
    Check workflow completion status by examining files on disk
    
    This allows the frontend to resume workflow from any stage
    even after page refresh or browser restart.
    """
    try:
        project_dir = os.path.join("ml/auto_annotation/projects", project_id)
        
        if not os.path.exists(project_dir):
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check each stage completion status
        workflow_status = {
            "project_id": project_id,
            "stage1_completed": check_stage1_completion(project_dir),
            "stage2_completed": check_stage2_completion(project_dir, project_id),
            "stage3_completed": check_stage3_completion(project_dir, project_id),
            "stage4_completed": check_stage4_completion(project_dir, project_id),
            "stage3_results": None,
            "normal_model_data": None
        }
        
        # Load Stage 3 results if available
        if workflow_status["stage3_completed"]:
            workflow_status["stage3_results"] = load_stage3_results(project_dir, project_id)
        
        # Load normal model data if available
        if workflow_status["stage2_completed"]:
            workflow_status["normal_model_data"] = load_normal_model_data(project_dir, project_id)
        
        print(f"📊 Workflow status for {project_id}:")
        print(f"  Stage 1 (ROI): {workflow_status['stage1_completed']}")
        print(f"  Stage 2 (Normal Model): {workflow_status['stage2_completed']}")
        print(f"  Stage 3 (Defect Detection): {workflow_status['stage3_completed']}")
        print(f"  Stage 4 (Bounding Boxes): {workflow_status['stage4_completed']}")
        
        return JSONResponse({
            'status': 'success',
            'workflow_status': workflow_status
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error checking workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_roi_directories(project_dir: str) -> Dict[str, str]:
    """
    Get ROI directories for a project
    Returns dict with 'training' and 'defective' directory paths
    """
    # All ROI methods now use the same directories
    standard_roi_dir = os.path.join(project_dir, "roi_cache")
    defective_standard_roi_dir = os.path.join(project_dir, "defective_roi_cache")
    
    # Check if segmentation masks exist to determine method used
    segmentation_masks_dir = os.path.join(project_dir, "segmentation_masks")
    
    return {
        'training': standard_roi_dir,
        'defective': defective_standard_roi_dir,
        'method': 'segmentation_mask' if os.path.exists(segmentation_masks_dir) else 'bounding_box'
    }

def check_stage1_completion(project_dir: str) -> bool:
    """Check if Stage 1 (ROI extraction) is completed"""
    try:
        roi_dirs = get_roi_directories(project_dir)
        
        # Check if training ROI directory exists
        if not os.path.exists(roi_dirs['training']):
            return False
        
        # Check if there are ROI images
        roi_files = [f for f in os.listdir(roi_dirs['training']) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        return len(roi_files) > 0
        
    except Exception as e:
        print(f"⚠️ Error checking Stage 1: {e}")
        return False

def check_stage2_completion(project_dir: str, project_id: str) -> bool:
    """Check if Stage 2 (normal model building) is completed"""
    try:
        features_dir = os.path.join(project_dir, "anomaly_features")
        
        # Check for any of the possible model files
        model_files_to_check = [
            os.path.join(features_dir, f"{project_id}_dinov3_normal_model.json"),  # DINOv3 model
            os.path.join(features_dir, f"{project_id}_dinov2_normal_model.json"),  # DINOv2 model
            os.path.join(features_dir, f"{project_id}_normal_model.json")          # Legacy model (DINOv2)
        ]
        
        # Check if any model file exists and is valid
        for model_file in model_files_to_check:
            if os.path.exists(model_file):
                try:
                    # Verify the model file is valid
                    with open(model_file, 'r') as f:
                        model_data = json.load(f)
                    
                    # Check if it has required fields
                    required_fields = ['global_mean', 'global_std', 'global_cov', 'feature_dimensions']
                    if all(field in model_data for field in required_fields):
                        print(f"✅ Found valid normal model: {os.path.basename(model_file)}")
                        return True
                    else:
                        print(f"⚠️ Invalid model file (missing fields): {model_file}")
                except Exception as e:
                    print(f"⚠️ Error reading model file {model_file}: {e}")
                    continue
        
        print(f"❌ No valid normal model found in {features_dir}")
        return False
        
    except Exception as e:
        print(f"⚠️ Error checking Stage 2: {e}")
        return False

def check_stage3_completion(project_dir: str, project_id: str) -> bool:
    """Check if Stage 3 (defect detection) is completed"""
    try:
        # Check for results from both DINOv2, DINOv3, and advanced methods
        result_files_to_check = [
            # Advanced results (check first as they're newest)
            os.path.join(project_dir, "advanced_defect_detection_results", f"{project_id}_advanced_defect_results.json"),
            # DINOv3 results
            os.path.join(project_dir, "dinov3_defect_detection_results", f"{project_id}_dinov3_defect_results.json"),
            # DINOv2 results (legacy and new)
            os.path.join(project_dir, "defect_detection_results", f"{project_id}_defect_results.json")
        ]
        
        # Check if any results file exists and is valid
        for results_file in result_files_to_check:
            if os.path.exists(results_file):
                try:
                    # Verify the results file is valid and has results
                    with open(results_file, 'r') as f:
                        results_data = json.load(f)
                    
                    # Check if we have valid defect detection results
                    has_results = len(results_data.get('results', [])) > 0
                    has_project_id = results_data.get('project_id') == project_id
                    has_method = 'method' in results_data
                    has_total_images = 'total_images' in results_data
                    
                    if has_results and has_project_id and has_method and has_total_images:
                        print(f"✅ Found valid defect detection results: {os.path.basename(results_file)}")
                        return True
                    else:
                        print(f"⚠️ Invalid results file (missing fields or no results): {results_file}")
                        
                except Exception as e:
                    print(f"⚠️ Error reading results file {results_file}: {e}")
                    continue
        
        print(f"❌ No valid defect detection results found in {project_dir}")
        return False
        
    except Exception as e:
        print(f"⚠️ Error checking Stage 3: {e}")
        return False

def check_stage4_completion(project_dir: str, project_id: str) -> bool:
    """Check if Stage 4 (bounding box generation) is completed"""
    try:
        annotations_dir = os.path.join(project_dir, "generated_annotations")
        summary_file = os.path.join(annotations_dir, f"{project_id}_bounding_boxes_summary.json")
        
        if not os.path.exists(summary_file):
            return False
        
        # Check if there are annotation files
        annotation_files = [f for f in os.listdir(annotations_dir) 
                           if f.endswith('.txt')]
        
        return len(annotation_files) > 0
        
    except Exception as e:
        print(f"⚠️ Error checking Stage 4: {e}")
        return False

def load_stage3_results(project_dir: str, project_id: str) -> dict:
    """Load Stage 3 defect detection results"""
    try:
        # Check for results from both DINOv2 and DINOv3 services
        result_files_to_check = [
            # DINOv3 results (check first as it's newer)
            os.path.join(project_dir, "dinov3_defect_detection_results", f"{project_id}_dinov3_defect_results.json"),
            # DINOv2 results (legacy and new)
            os.path.join(project_dir, "defect_detection_results", f"{project_id}_defect_results.json")
        ]
        
        # Return the first valid results file found
        for results_file in result_files_to_check:
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results_data = json.load(f)
                    
                    # Verify it has some basic structure
                    if isinstance(results_data, dict) and 'results' in results_data:
                        print(f"✅ Loaded Stage 3 results from: {os.path.basename(results_file)}")
                        return results_data
                        
                except Exception as e:
                    print(f"⚠️ Error reading results file {results_file}: {e}")
                    continue
        
        print(f"❌ No valid Stage 3 results found in {project_dir}")
        return None
        
    except Exception as e:
        print(f"⚠️ Error loading Stage 3 results: {e}")
        return None

def load_normal_model_data(project_dir: str, project_id: str) -> dict:
    """Load Stage 2 normal model data"""
    try:
        features_dir = os.path.join(project_dir, "anomaly_features")
        
        # Check for model files from both DINOv2 and DINOv3
        model_files_to_check = [
            # DINOv3 model (check first as it's newer)
            os.path.join(features_dir, f"{project_id}_dinov3_normal_model.json"),
            # DINOv2 model
            os.path.join(features_dir, f"{project_id}_dinov2_normal_model.json"),
            # Legacy model (DINOv2)
            os.path.join(features_dir, f"{project_id}_normal_model.json")
        ]
        
        # Return the first valid model file found
        for model_file in model_files_to_check:
            if os.path.exists(model_file):
                try:
                    with open(model_file, 'r') as f:
                        model_data = json.load(f)
                    
                    # Verify it has some basic structure
                    if isinstance(model_data, dict) and 'global_mean' in model_data:
                        print(f"✅ Loaded normal model from: {os.path.basename(model_file)}")
                        return model_data
                        
                except Exception as e:
                    print(f"⚠️ Error reading model file {model_file}: {e}")
                    continue
        
        print(f"❌ No valid normal model found in {features_dir}")
        return None
        
    except Exception as e:
        print(f"⚠️ Error loading normal model data: {e}")
        return None

# =============================================================================
# AUTO-ANNOTATION ENDPOINTS
# =============================================================================

@router.post("/projects/{project_id}/annotate")
async def auto_annotate_images(
    project_id: str,
    images: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5),
    batch_id: str = Form(None)
):
    """
    Automatically annotate images using trained YOLO/SAM2 model
    
    Simplified workflow:
    1. Apply trained YOLO/SAM2 model directly to full images
    2. Draw annotations and return results
    
    Args:
        project_id: Project with trained model
        images: Images to annotate
        confidence_threshold: Detection confidence threshold
        batch_id: Optional batch ID for grouping results
    """
    try:
        # Read uploaded images
        images_data = []
        image_names = []
        
        for image_file in images:
            image_data = await image_file.read()
            images_data.append(image_data)
            image_names.append(image_file.filename)
        
        result = auto_annotation_service.annotate_images(
            project_id, images_data, image_names, confidence_threshold
        )
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/batch-annotate")
async def batch_auto_annotate(
    project_id: str,
    images: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Batch auto-annotation with progress tracking
    
    Similar to single annotation but optimized for large batches
    """
    try:
        # This would handle large batch processing with progress updates
        # For now, delegate to single annotation method
        
        images_data = []
        image_names = []
        
        for image_file in images:
            image_data = await image_file.read()
            images_data.append(image_data)
            image_names.append(image_file.filename)
        
        result = auto_annotation_service.annotate_images(
            project_id, images_data, image_names, confidence_threshold
        )
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/annotation-formats")
async def get_supported_annotation_formats():
    """Get information about supported annotation formats"""
    return JSONResponse({
        "status": "success",
        "formats": {
            "object_detection": {
                "name": "YOLO Format",
                "description": "Text files with normalized bounding box coordinates",
                "file_extension": ".txt",
                "format_example": "class_id center_x center_y width height",
                "upload_format": "ZIP file containing .txt annotation files"
            },
            "segmentation": {
                "name": "COCO Format", 
                "description": "JSON file with polygon segmentation data",
                "file_extension": ".json",
                "format_example": "COCO JSON with segmentation polygons",
                "upload_format": "Single JSON file with COCO annotations"
            }
        },
        "workflow_comparison": {
            "object_detection": {
                "output": "Colored bounding boxes around defects",
                "speed": "Fast training and inference",
                "accuracy": "Good for defect location",
                "use_case": "Quick defect identification and counting"
            },
            "segmentation": {
                "output": "Precise masks following defect shapes", 
                "speed": "Slower training and inference",
                "accuracy": "Excellent for defect boundaries",
                "use_case": "Precise defect measurement and analysis"
            }
        }
    })

@router.get("/projects/{project_id}/export-results")
async def export_annotation_results(project_id: str, format: str = "json"):
    """Export annotation results in various formats"""
    try:
        # This would export results in requested format
        return JSONResponse({
            "status": "success",
            "message": f"Results exported in {format} format",
            "download_url": f"/download/results/{project_id}.{format}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STATUS AND HEALTH ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_auto_annotation_status():
    """Get overall status of auto-annotation services"""
    try:
        # Check if required dependencies are available
        dependencies = {
            "grounding_dino": auto_annotation_service.grounding_dino.model is not None,
            "yolo_available": True,  # Would check ultralytics installation
            "sam2_available": True,  # Would check SAM2 installation
            "opencv_available": True,  # Would check OpenCV
        }
        
        projects_result = database_service.list_projects()
        projects = projects_result['projects'] if projects_result['status'] == 'success' else []
        
        return JSONResponse({
            "status": "ready" if all(dependencies.values()) else "dependencies_missing",
            "dependencies": dependencies,
            "total_projects": len(projects),
            "active_projects": len([p for p in projects if p['status'] == 'active']),
            "message": "Auto-annotation service ready" if all(dependencies.values()) 
                      else "Some dependencies missing"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))