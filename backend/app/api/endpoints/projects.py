from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import os
import json
import uuid
from datetime import datetime

try:
    from backend.services.storage import storage_service
    from backend.services.core.database_service import database_service
except ImportError:
    from services.storage import storage_service
    from services.core.database_service import database_service

router = APIRouter()

# =============================================================================
# PROJECT MANAGEMENT ENDPOINTS (UNIVERSAL)
# =============================================================================

@router.post("/create")
async def create_project(
    project_name: str = Form(...),
    project_type: str = Form(...),  # 'object_detection', 'segmentation', or 'anomaly_detection'
    description: str = Form("")
):
    """
    Create a new project (universal for all workflows)
    
    Project types:
    - object_detection: YOLO for defect bounding boxes
    - segmentation: SAM2 for precise defect masks
    - anomaly_detection: ROI + normal model + defect detection
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
        
        # Create project directory structure in storage
        project_dir = storage_service.get_project_directory(project_id)
        await storage_service.create_directory(project_dir)
        
        # Initialize workflow status
        workflow_status = {
            'project_id': project_id,
            'project_type': project_type,
            'steps_completed': [],
            'current_step': None,
            'training_data_uploaded': False
        }
        
        await storage_service.save_json(
            workflow_status, 
            f"{project_dir}/workflow_status.json"
        )
        
        return JSONResponse({
            'status': 'success',
            'message': 'Project created successfully',
            'project_id': project_id,
            'metadata': {
                'project_name': project_name,
                'project_type': project_type,
                'description': description,
                'storage_directory': project_dir
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_projects():
    """List all projects"""
    try:
        projects = database_service.list_projects()
        return JSONResponse(projects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}")
async def get_project(project_id: str):
    """Get details for a specific project"""
    try:
        project = database_service.get_project(project_id)
        
        if project['status'] == 'success':
            return JSONResponse(project)
        else:
            raise HTTPException(status_code=404, detail=project['message'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and all its associated data"""
    try:
        # First delete from database
        result = database_service.delete_project(project_id)
        
        if result['status'] == 'success':
            # Also delete the project folder from Cloud Storage
            project_folder_path = f"projects/{project_id}"
            storage_deleted = await storage_service.delete_folder(project_folder_path)
            
            if storage_deleted:
                result['message'] += " (including Cloud Storage data)"
                result['storage_cleanup'] = True
            else:
                result['message'] += " (Cloud Storage cleanup failed)"
                result['storage_cleanup'] = False
            
            return JSONResponse(result)
        else:
            raise HTTPException(status_code=404, detail=result['message'])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/upload-training-data")
async def upload_training_data(
    project_id: str,
    training_images: Optional[List[UploadFile]] = File(None),
    annotation_files: Optional[List[UploadFile]] = File(None),
    dataset_zip: Optional[UploadFile] = File(None),
    annotation_format: str = Form("auto")  # 'yolo', 'coco', or 'auto'
):
    """
    Upload training data to a project (supports multiple upload methods)
    
    OPTION 1 - Individual Files:
    - training_images: List of JPG/PNG image files
    - annotation_files: List of annotation files (.txt for YOLO, .json for COCO)
    
    OPTION 2 - Complete Dataset ZIP (Recommended for Object Detection):
    - dataset_zip: ZIP file containing complete YOLO dataset structure:
      └── dataset.zip
          ├── data.yaml           # Dataset configuration
          ├── images/
          │   ├── train/          # Training images
          │   └── val/            # Validation images  
          └── labels/
              ├── train/          # Training labels (.txt files)
              └── val/            # Validation labels (.txt files)
    
    For different project types:
    - Object Detection: YOLO format (ZIP recommended)
    - Segmentation: COCO format or masks
    - Anomaly Detection: Images only (annotations optional)
    """
    try:
        # Check if project exists in database
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_data = project_result['project']
        project_type = project_data['project_type']
        
        # Determine upload method
        if dataset_zip:
            # OPTION 2: Complete YOLO dataset ZIP upload
            return await handle_dataset_zip_upload(project_id, dataset_zip, project_type)
        elif training_images:
            # OPTION 1: Individual files upload  
            return await handle_individual_files_upload(project_id, training_images, annotation_files, project_type, annotation_format)
        else:
            raise HTTPException(status_code=400, detail="Either dataset_zip or training_images must be provided")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def handle_dataset_zip_upload(project_id: str, dataset_zip: UploadFile, project_type: str):
    """Handle complete YOLO dataset ZIP upload"""
    try:
        from services.ml.yolo_service import yolo_service
        
        # Use YOLO service to handle dataset upload
        dataset_path = await yolo_service.handle_dataset_upload(dataset_zip, project_type)
        
        if not dataset_path:
            raise HTTPException(status_code=400, detail="Invalid YOLO dataset format. Please ensure your ZIP contains data.yaml and proper folder structure.")
        
        # Parse the dataset to count files for database tracking
        import yaml
        import os
        
        with open(dataset_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        dataset_root = os.path.dirname(dataset_path)
        
        # Count files
        train_images_dir = os.path.join(dataset_root, data_config.get('train', 'images/train'))
        val_images_dir = os.path.join(dataset_root, data_config.get('val', 'images/val'))
        
        train_images = len([f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(train_images_dir) else 0
        val_images = len([f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(val_images_dir) else 0
        
        # Store dataset path in database for later training use
        dataset_info = {
            'dataset_type': 'yolo_zip',
            'dataset_path': dataset_path,
            'data_yaml_path': dataset_path,
            'train_images_count': train_images,
            'val_images_count': val_images,
            'num_classes': data_config.get('nc', 0),
            'class_names': data_config.get('names', [])
        }
        
        # Store dataset info in project metadata (simplified implementation)
        # For now, we'll store this as a comment - full implementation would update project in database
        
        return JSONResponse({
            'status': 'success',
            'message': f'YOLO dataset uploaded successfully',
            'dataset_info': {
                'dataset_path': dataset_path,
                'train_images': train_images,
                'val_images': val_images,
                'num_classes': data_config.get('nc', 0),
                'class_names': data_config.get('names', [])
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process YOLO dataset: {str(e)}")

async def handle_individual_files_upload(project_id: str, training_images: List[UploadFile], annotation_files: Optional[List[UploadFile]], project_type: str, annotation_format: str):
    """Handle individual files upload (original method)"""
    try:
        # Get project directory structure
        dirs = await get_project_directories(project_id)
        
        # Ensure project directory exists in storage
        await storage_service.create_directory(dirs['project'])
        await storage_service.create_directory(dirs['training_images'])
        
        uploaded_files = []
        
        # Upload training images
        for image_file in training_images:
            content = await image_file.read()
            
            # Upload to storage
            storage_path = f"projects/{project_id}/training_images/{image_file.filename}"
            storage_url = await storage_service.upload_file(
                file_data=content,
                file_path=storage_path,
                content_type="image/jpeg"
            )
            
            # Track file in database
            db_result = database_service.add_uploaded_file(
                project_id=project_id,
                file_type='training_images',
                filename=image_file.filename,
                original_filename=image_file.filename,
                storage_url=storage_url,
                storage_path=storage_path,
                file_size_bytes=len(content),
                content_type="image/jpeg"
            )
            
            uploaded_files.append({
                'filename': image_file.filename,
                'type': 'training_image',
                'storage_url': storage_url,
                'tracked': db_result['status'] == 'success'
            })
        
        # Upload annotation files if provided
        if annotation_files:
            await storage_service.create_directory(f"{dirs['project']}/annotation_files")
            for annotation_file in annotation_files:
                content = await annotation_file.read()
                
                # Upload to storage
                storage_path = f"projects/{project_id}/annotation_files/{annotation_file.filename}"
                storage_url = await storage_service.upload_file(
                    file_data=content,
                    file_path=storage_path,
                    content_type="application/json" if annotation_file.filename.endswith('.json') else "text/plain"
                )
                
                # Track file in database  
                db_result = database_service.add_uploaded_file(
                    project_id=project_id,
                    file_type='annotation_files',
                    filename=annotation_file.filename,
                    original_filename=annotation_file.filename,
                    storage_url=storage_url,
                    storage_path=storage_path,
                    file_size_bytes=len(content),
                    content_type="application/json" if annotation_file.filename.endswith('.json') else "text/plain"
                )
                
                uploaded_files.append({
                    'filename': annotation_file.filename,
                    'type': 'annotation_file',
                    'storage_url': storage_url,
                    'tracked': db_result['status'] == 'success'
                })
        
        # Update workflow status
        try:
            workflow_status = await storage_service.load_json(dirs['workflow_status'])
            workflow_status['training_data_uploaded'] = True
            workflow_status['last_updated'] = str(datetime.now())
            await storage_service.save_json(workflow_status, dirs['workflow_status'])
        except:
            # Create new workflow status if it doesn't exist
            workflow_status = {
                'project_id': project_id,
                'project_type': project_type,
                'steps_completed': ['upload_training_data'],
                'current_step': None,
                'training_data_uploaded': True,
                'last_updated': str(datetime.now())
            }
            await storage_service.save_json(workflow_status, dirs['workflow_status'])
        
        return JSONResponse({
            'status': 'success', 
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'uploaded_files': uploaded_files
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/upload-defective-data")
async def upload_defective_data(
    project_id: str,
    defective_images: List[UploadFile] = File(...),
    annotation_files: Optional[List[UploadFile]] = File(None),
    annotation_format: str = Form("auto")  # 'yolo', 'coco', or 'auto'
):
    """
    Upload defective/anomaly images to a project
    """
    try:
        # Check if project exists in database
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_data = project_result['project']
        project_type = project_data['project_type']
        
        # Get project directory structure
        dirs = await get_project_directories(project_id)
        
        # Ensure project directory exists in storage
        await storage_service.create_directory(dirs['project'])
        await storage_service.create_directory(dirs['defective_images'])
        
        uploaded_files = []
        
        # Upload defective images
        for image_file in defective_images:
            content = await image_file.read()
            
            # Upload to storage
            storage_path = f"projects/{project_id}/defective_images/{image_file.filename}"
            storage_url = await storage_service.upload_file(
                file_data=content,
                file_path=storage_path,
                content_type="image/jpeg"
            )
            
            # Track file in database
            db_result = database_service.add_uploaded_file(
                project_id=project_id,
                file_type='defective_images',
                filename=image_file.filename,
                original_filename=image_file.filename,
                storage_url=storage_url,
                storage_path=storage_path,
                file_size_bytes=len(content),
                content_type="image/jpeg"
            )
            
            uploaded_files.append({
                'filename': image_file.filename,
                'type': 'defective_image',
                'storage_url': storage_url,
                'tracked': db_result['status'] == 'success'
            })

        # Upload annotation files if provided
        if annotation_files:
            for annotation_file in annotation_files:
                content = await annotation_file.read()
                
                # Upload to storage
                storage_path = f"projects/{project_id}/annotation_files/{annotation_file.filename}"
                storage_url = await storage_service.upload_file(
                    file_data=content,
                    file_path=storage_path,
                    content_type="application/json" if annotation_file.filename.endswith('.json') else "text/plain"
                )
                
                # Track file in database  
                db_result = database_service.add_uploaded_file(
                    project_id=project_id,
                    file_type='annotation_files',
                    filename=annotation_file.filename,
                    original_filename=annotation_file.filename,
                    storage_url=storage_url,
                    storage_path=storage_path,
                    file_size_bytes=len(content),
                    content_type="application/json" if annotation_file.filename.endswith('.json') else "text/plain"
                )
                
                uploaded_files.append({
                    'filename': annotation_file.filename,
                    'type': 'annotation_file',
                    'storage_url': storage_url,
                    'tracked': db_result['status'] == 'success'
                })
        
        return JSONResponse({
            'status': 'success', 
            'message': f'Successfully uploaded {len(uploaded_files)} defective files',
            'uploaded_files': uploaded_files
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/upload-test-data")
async def upload_test_data(
    project_id: str,
    test_images: List[UploadFile] = File(...),
    annotation_files: Optional[List[UploadFile]] = File(None),
    annotation_format: str = Form("auto")  # 'yolo', 'coco', or 'auto'
):
    """
    Upload test/validation images to a project
    """
    try:
        # Check if project exists in database
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_data = project_result['project']
        project_type = project_data['project_type']
        
        # Get project directory structure
        dirs = await get_project_directories(project_id)
        
        # Ensure project directory exists in storage
        await storage_service.create_directory(dirs['project'])
        # Create test_images directory structure
        await storage_service.create_directory(f"{dirs['project']}/test_images")
        
        uploaded_files = []
        
        # Upload test images
        for image_file in test_images:
            content = await image_file.read()
            
            # Upload to storage
            storage_path = f"projects/{project_id}/test_images/{image_file.filename}"
            storage_url = await storage_service.upload_file(
                file_data=content,
                file_path=storage_path,
                content_type="image/jpeg"
            )
            
            # Track file in database
            db_result = database_service.add_uploaded_file(
                project_id=project_id,
                file_type='test_images',
                filename=image_file.filename,
                original_filename=image_file.filename,
                storage_url=storage_url,
                storage_path=storage_path,
                file_size_bytes=len(content),
                content_type="image/jpeg"
            )
            
            uploaded_files.append({
                'filename': image_file.filename,
                'type': 'test_image',
                'storage_url': storage_url,
                'tracked': db_result['status'] == 'success'
            })

        # Upload annotation files if provided
        if annotation_files:
            for annotation_file in annotation_files:
                content = await annotation_file.read()
                
                # Upload to storage
                storage_path = f"projects/{project_id}/annotation_files/{annotation_file.filename}"
                storage_url = await storage_service.upload_file(
                    file_data=content,
                    file_path=storage_path,
                    content_type="application/json" if annotation_file.filename.endswith('.json') else "text/plain"
                )
                
                # Track file in database  
                db_result = database_service.add_uploaded_file(
                    project_id=project_id,
                    file_type='annotation_files',
                    filename=annotation_file.filename,
                    original_filename=annotation_file.filename,
                    storage_url=storage_url,
                    storage_path=storage_path,
                    file_size_bytes=len(content),
                    content_type="application/json" if annotation_file.filename.endswith('.json') else "text/plain"
                )
                
                uploaded_files.append({
                    'filename': annotation_file.filename,
                    'type': 'annotation_file',
                    'storage_url': storage_url,
                    'tracked': db_result['status'] == 'success'
                })
        
        return JSONResponse({
            'status': 'success', 
            'message': f'Successfully uploaded {len(uploaded_files)} test files',
            'uploaded_files': uploaded_files
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/validate-dataset")
async def validate_dataset(project_id: str):
    """
    Validate if project dataset is ready for training
    """
    try:
        # Get project info
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_data = project_result['project']
        project_type = project_data['project_type']
        
        # Get file counts
        file_counts = project_data.get('file_counts', {})
        training_images = file_counts.get('training_images', 0)
        defective_images = file_counts.get('defective_images', 0)
        test_images = file_counts.get('test_images', 0)
        annotation_files = file_counts.get('annotation_files', 0)
        
        # Validation rules by project type
        validation_result = {
            'project_id': project_id,
            'project_type': project_type,
            'is_ready': False,
            'requirements_met': {},
            'missing_requirements': [],
            'recommendations': []
        }
        
        if project_type == 'anomaly_detection':
            # Anomaly detection requires training images (normal samples)
            validation_result['requirements_met'] = {
                'training_images': training_images >= 10,  # Minimum 10 normal samples
                'defective_images_optional': defective_images >= 0
            }
            
            if training_images < 10:
                validation_result['missing_requirements'].append(
                    f"Need at least 10 training images (normal samples). Currently: {training_images}"
                )
            
            if defective_images == 0:
                validation_result['recommendations'].append(
                    "Consider uploading defective samples for better anomaly detection training"
                )
                
            validation_result['is_ready'] = training_images >= 10
            
        elif project_type == 'object_detection':
            # Object detection requires training images + YOLO annotations
            validation_result['requirements_met'] = {
                'training_images': training_images >= 20,  # Minimum 20 images
                'annotation_files': annotation_files >= 1   # At least 1 annotation file
            }
            
            if training_images < 20:
                validation_result['missing_requirements'].append(
                    f"Need at least 20 training images. Currently: {training_images}"
                )
            
            if annotation_files < 1:
                validation_result['missing_requirements'].append(
                    f"Need YOLO format annotation files. Currently: {annotation_files}"
                )
            
            if test_images == 0:
                validation_result['recommendations'].append(
                    "Consider uploading test images for validation"
                )
                
            validation_result['is_ready'] = training_images >= 20 and annotation_files >= 1
            
        elif project_type == 'segmentation':
            # Segmentation requires training images + COCO/mask annotations
            validation_result['requirements_met'] = {
                'training_images': training_images >= 15,  # Minimum 15 images
                'annotation_files': annotation_files >= 1   # At least 1 annotation file
            }
            
            if training_images < 15:
                validation_result['missing_requirements'].append(
                    f"Need at least 15 training images. Currently: {training_images}"
                )
            
            if annotation_files < 1:
                validation_result['missing_requirements'].append(
                    f"Need segmentation annotation files (COCO format). Currently: {annotation_files}"
                )
            
            if test_images == 0:
                validation_result['recommendations'].append(
                    "Consider uploading test images for validation"
                )
                
            validation_result['is_ready'] = training_images >= 15 and annotation_files >= 1
            
        else:
            # Unknown project type
            validation_result['missing_requirements'].append(
                f"Unknown project type: {project_type}"
            )
        
        return JSONResponse({
            'status': 'success',
            'validation': validation_result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/prepare-training")
async def prepare_training_dataset(project_id: str):
    """
    Prepare project dataset for training by organizing into proper format
    """
    try:
        # First validate the dataset
        validation_response = await validate_dataset(project_id)
        validation_data = validation_response.body.decode()
        import json
        validation_result = json.loads(validation_data)
        
        if not validation_result['validation']['is_ready']:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "Dataset not ready for training",
                    "validation": validation_result['validation']
                }
            )
        
        # Get project info
        project_result = database_service.get_project(project_id)
        project_data = project_result['project']
        project_type = project_data['project_type']
        
        # Get project directories
        dirs = await get_project_directories(project_id)
        
        # Prepare training dataset based on project type
        preparation_result = {
            'project_id': project_id,
            'project_type': project_type,
            'training_ready': False,
            'dataset_path': None,
            'format': None
        }
        
        if project_type == 'anomaly_detection':
            # For anomaly detection, organize normal/defective samples
            dataset_path = f"{dirs['project']}/prepared_dataset"
            await storage_service.create_directory(dataset_path)
            await storage_service.create_directory(f"{dataset_path}/normal")
            await storage_service.create_directory(f"{dataset_path}/defective")
            
            preparation_result.update({
                'training_ready': True,
                'dataset_path': dataset_path,
                'format': 'anomaly_detection',
                'normal_images_path': f"{dataset_path}/normal",
                'defective_images_path': f"{dataset_path}/defective"
            })
            
        elif project_type in ['object_detection', 'segmentation']:
            # For detection/segmentation, create YOLO-style dataset structure
            dataset_path = f"{dirs['project']}/prepared_dataset"
            await storage_service.create_directory(dataset_path)
            await storage_service.create_directory(f"{dataset_path}/images")
            await storage_service.create_directory(f"{dataset_path}/images/train")
            await storage_service.create_directory(f"{dataset_path}/images/val")
            await storage_service.create_directory(f"{dataset_path}/labels")
            await storage_service.create_directory(f"{dataset_path}/labels/train")
            await storage_service.create_directory(f"{dataset_path}/labels/val")
            
            preparation_result.update({
                'training_ready': True,
                'dataset_path': dataset_path,
                'format': 'yolo' if project_type == 'object_detection' else 'coco',
                'images_path': f"{dataset_path}/images",
                'labels_path': f"{dataset_path}/labels"
            })
        
        return JSONResponse({
            'status': 'success',
            'preparation': preparation_result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_project_directories(project_id: str) -> Dict[str, str]:
    """Get all directory paths for a project using storage service"""
    base_dir = storage_service.get_project_directory(project_id)
    
    # Standard project directories
    directories = {
        'project': base_dir,
        'training_images': f"{base_dir}/training_images",
        'defective_images': f"{base_dir}/defective_images", 
        'test_images': f"{base_dir}/test_images",
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