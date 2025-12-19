from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import os
import json
import uuid
import yaml
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
        
        # FALLBACK: If annotation_files is 0 but we have training images, 
        # do a direct database check to see if annotation files actually exist
        if annotation_files == 0 and training_images > 0:
            try:
                from backend.services.core.database_service import UploadedFile, DatabaseService
                from sqlalchemy import func
                
                db_service = DatabaseService()
                session = db_service.get_session()
                
                try:
                    # Check if there are actually annotation files in the database
                    annotation_count = session.query(func.count(UploadedFile.id)).filter(
                        UploadedFile.project_id == project_id,
                        UploadedFile.file_type == 'annotation_files'
                    ).scalar()
                    
                    if annotation_count > 0:
                        print(f"⚠️ Found {annotation_count} annotation files in database but file_counts shows 0. Using database count.")
                        annotation_files = annotation_count
                        
                finally:
                    session.close()
            except Exception as e:
                print(f"⚠️ Failed to check database for annotation files: {e}")
        
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
                'training_images': training_images >= 10,  # Minimum 10 images (consistent with error message)
                'annotation_files': annotation_files >= 1   # At least 1 annotation file
            }

            if training_images < 10:
                validation_result['missing_requirements'].append(
                    f"Need at least 10 training images. Currently: {training_images}"
                )

            if annotation_files < 1:
                validation_result['missing_requirements'].append(
                    f"Need YOLO format annotation files. Currently: {annotation_files}"
                )

            if test_images == 0:
                validation_result['recommendations'].append(
                    "Consider uploading test images for validation"
                )

            validation_result['is_ready'] = training_images >= 10 and annotation_files >= 1
            
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

@router.post("/{project_id}/upload-zip-dataset")
async def upload_zip_dataset(
    project_id: str,
    dataset_zip: UploadFile = File(...)
):
    """
    Upload a complete YOLO dataset as ZIP file
    
    Expected ZIP structure:
    dataset.zip
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
    """
    try:
        # Check if project exists
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_data = project_result['project']
        project_type = project_data['project_type']
        
        # Process the ZIP file directly to count files
        import tempfile
        import zipfile
        from pathlib import Path
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / "dataset.zip"

            # Save uploaded ZIP file
            with open(zip_path, 'wb') as f:
                content = await dataset_zip.read()
                f.write(content)

            # Extract and count files
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)

            # Count image files in train/val directories
            train_images = 0
            val_images = 0
            total_labels = 0
            classes = []

            # Look for standard YOLO structure
            for root, dirs, files in os.walk(temp_path):
                root_path = Path(root)

                # Count images
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                if 'train' in root_path.name.lower():
                    train_images += len(image_files)
                elif 'val' in root_path.name.lower() or 'test' in root_path.name.lower():
                    val_images += len(image_files)

                # Count labels
                label_files = [f for f in files if f.lower().endswith('.txt') and f != 'classes.txt']
                total_labels += len(label_files)

                # Read classes.txt if found
                if 'classes.txt' in files:
                    with open(root_path / 'classes.txt', 'r') as f:
                        classes = [line.strip() for line in f if line.strip()]

            # Create dataset info
            dataset_info = {
                'total_images': train_images + val_images,
                'train_images': train_images,
                'val_images': val_images,
                'total_labels': total_labels,
                'classes': classes,
                'dataset_path': f"projects/{project_id}"
            }

            # CRITICAL: Actually upload the extracted files to GCS
            # Get project directory structure
            dirs = await get_project_directories(project_id)
            
            # Ensure project directories exist in GCS
            await storage_service.create_directory(dirs['project'])
            await storage_service.create_directory(f"{dirs['project']}/images")
            await storage_service.create_directory(f"{dirs['project']}/images/train")
            await storage_service.create_directory(f"{dirs['project']}/images/val")
            await storage_service.create_directory(f"{dirs['project']}/labels")
            await storage_service.create_directory(f"{dirs['project']}/labels/train")
            await storage_service.create_directory(f"{dirs['project']}/labels/val")
            
            # Upload all extracted files to GCS
            uploaded_files = []
            print(f"🔍 [DEBUG] Starting to process extracted files from {temp_path}")
            
            for root, dirs_in_path, files in os.walk(temp_path):
                root_path = Path(root)
                relative_path = root_path.relative_to(temp_path)
                print(f"🔍 [DEBUG] Processing directory: {root_path} (relative: {relative_path})")
                print(f"🔍 [DEBUG] Found {len(files)} files in this directory: {files}")
                    
                for file in files:
                    # Skip ZIP files
                    if file.endswith('.zip'):
                        print(f"⏭️ [DEBUG] Skipping ZIP file: {file}")
                        continue
                        
                    file_path = root_path / file
                    relative_file_path = relative_path / file
                    
                    print(f"🔍 [DEBUG] Processing file: {file_path}")
                    print(f"🔍 [DEBUG] Relative file path: {relative_file_path}")
                    
                    # Check if file exists and get size
                    if not file_path.exists():
                        print(f"❌ [DEBUG] File does not exist: {file_path}")
                        continue
                        
                    file_size = file_path.stat().st_size
                    print(f"🔍 [DEBUG] File size on disk: {file_size} bytes")
                    
                    # Read file content
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    print(f"🔍 [DEBUG] Read {len(content)} bytes from file")
                    
                    # Determine content type
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        content_type = "image/jpeg"
                        file_type = 'training_images'
                    elif file.lower().endswith('.txt'):
                        content_type = "text/plain"
                        file_type = 'annotations'
                    elif file.lower().endswith('.yaml') or file.lower().endswith('.yml'):
                        content_type = "text/yaml"
                        file_type = 'config'
                    else:
                        content_type = "application/octet-stream"
                        file_type = 'other'
                    
                    # Upload to storage with proper GCS path structure
                    storage_path = f"projects/{project_id}/{relative_file_path}"
                    
                    # Debug logging
                    print(f"🔍 [DEBUG] Uploading file: {file}")
                    print(f"🔍 [DEBUG] Storage path: {storage_path}")
                    print(f"🔍 [DEBUG] Content size: {len(content)} bytes")
                    print(f"🔍 [DEBUG] Content type: {content_type}")
                    
                    try:
                        storage_url = await storage_service.upload_file(
                            file_data=content,
                            file_path=storage_path,
                            content_type=content_type
                        )
                        print(f"✅ [DEBUG] Upload successful: {storage_url}")
                    except Exception as upload_error:
                        print(f"❌ [DEBUG] Upload failed: {upload_error}")
                        print(f"❌ [DEBUG] Traceback: {str(upload_error)}")
                        # Re-raise the error so we can see what's wrong
                        raise upload_error
                    
                    # Track file in database
                    db_result = database_service.add_uploaded_file(
                        project_id=project_id,
                        file_type=file_type,
                        filename=file,
                        original_filename=file,
                        storage_url=storage_url,
                        storage_path=storage_path,
                        file_size_bytes=len(content),
                        content_type=content_type
                    )
                    
                    uploaded_files.append({
                        'filename': file,
                        'type': file_type,
                        'storage_url': storage_url,
                        'relative_path': str(relative_file_path),
                        'tracked': db_result['status'] == 'success'
                    })

            # CRITICAL: Create a proper data.yaml file for cloud training
            # The original data.yaml contains local paths which won't work in cloud
            print(f"🔧 [DEBUG] Creating proper data.yaml for cloud training")
            print(f"🔧 [DEBUG] Classes found: {classes}")
            
            # Ensure we have classes - if empty, try to read from classes.txt in the extracted files
            if not classes:
                print(f"⚠️ [DEBUG] No classes found, searching for classes.txt in extracted files")
                for root, dirs_in_path, files in os.walk(temp_path):
                    if 'classes.txt' in files:
                        classes_file_path = Path(root) / 'classes.txt'
                        print(f"🔍 [DEBUG] Found classes.txt at: {classes_file_path}")
                        with open(classes_file_path, 'r') as f:
                            classes = [line.strip() for line in f if line.strip()]
                        print(f"✅ [DEBUG] Loaded classes from classes.txt: {classes}")
                        break
            
            # If still no classes, check if there's an existing data.yaml with classes
            if not classes:
                print(f"⚠️ [DEBUG] Still no classes found, checking existing data.yaml files")
                for root, dirs_in_path, files in os.walk(temp_path):
                    for file in files:
                        if file.lower() in ['data.yaml', 'data.yml']:
                            yaml_file_path = Path(root) / file
                            print(f"🔍 [DEBUG] Found existing data.yaml at: {yaml_file_path}")
                            try:
                                with open(yaml_file_path, 'r') as f:
                                    existing_data = yaml.safe_load(f)
                                if 'names' in existing_data and existing_data['names']:
                                    classes = existing_data['names']
                                    if isinstance(classes, dict):
                                        # Convert dict format {0: 'class1', 1: 'class2'} to list
                                        classes = [classes[i] for i in sorted(classes.keys())]
                                    print(f"✅ [DEBUG] Loaded classes from existing data.yaml: {classes}")
                                    break
                            except Exception as e:
                                print(f"❌ [DEBUG] Failed to read existing data.yaml: {e}")
                    if classes:
                        break
            
            # Final validation - we must have classes
            if not classes:
                raise Exception("No classes found in dataset. Please ensure your dataset includes either classes.txt or a valid data.yaml with class definitions.")
            
            # Format classes for YAML output (as a list)
            classes_yaml = '\n'.join([f'  - {cls}' for cls in classes])
            
            # Create the correct data.yaml content for cloud environment
            data_yaml_content = f"""# YOLO dataset configuration for project {project_id}
# Generated automatically - do not edit manually

# Dataset paths (relative to this file)
path: .  # Dataset root directory
train: images/train  # Training images directory
val: images/val      # Validation images directory

# Classes
nc: {len(classes)}  # Number of classes
names:
{classes_yaml}
"""
            
            # Upload the corrected data.yaml file
            try:
                corrected_yaml_storage_path = f"projects/{project_id}/data.yaml"
                corrected_yaml_url = await storage_service.upload_file(
                    file_data=data_yaml_content.encode('utf-8'),
                    file_path=corrected_yaml_storage_path,
                    content_type="text/yaml"
                )
                
                # Track the corrected data.yaml in database
                db_result = database_service.add_uploaded_file(
                    project_id=project_id,
                    file_type='config',
                    filename='data.yaml',
                    original_filename='data.yaml',
                    storage_url=corrected_yaml_url,
                    storage_path=corrected_yaml_storage_path,
                    file_size_bytes=len(data_yaml_content.encode('utf-8')),
                    content_type="text/yaml"
                )
                
                uploaded_files.append({
                    'filename': 'data.yaml',
                    'type': 'config',
                    'storage_url': corrected_yaml_url,
                    'relative_path': 'data.yaml',
                    'tracked': db_result['status'] == 'success'
                })
                
                print(f"✅ [DEBUG] Created corrected data.yaml for cloud training")
                print(f"🔍 [DEBUG] New data.yaml content:")
                print(data_yaml_content)
                
            except Exception as yaml_error:
                print(f"❌ [DEBUG] Failed to create corrected data.yaml: {yaml_error}")

        # Create database records with actual file counts
        file_counts = {
            'total_images': dataset_info.get('total_images', 0),
            'train_images': dataset_info.get('train_images', 0),
            'val_images': dataset_info.get('val_images', 0),
            'total_labels': dataset_info.get('total_labels', 0),
            'classes': dataset_info.get('classes', []),
            'dataset_path': dataset_info.get('dataset_path', ''),
            'uploaded_files_count': len(uploaded_files)
        }

        # Update database with proper file counts
        database_service.update_project_file_counts(project_id, file_counts)

        return JSONResponse({
            'status': 'success',
            'message': 'ZIP dataset uploaded and processed successfully',
            'dataset_info': dataset_info,
            'total_images': dataset_info.get('total_images', 0),
            'total_labels': dataset_info.get('total_labels', 0),
            'classes': dataset_info.get('classes', []),
            'train_images': dataset_info.get('train_images', 0),
            'val_images': dataset_info.get('val_images', 0),
            'uploaded_files_count': len(uploaded_files),
            'uploaded_files': uploaded_files[:10]  # Show first 10 files for verification
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process ZIP dataset: {str(e)}")

@router.post("/{project_id}/upload-raw-annotations")
async def upload_raw_annotations(
    project_id: str,
    raw_files: List[UploadFile] = File(...)
):
    """
    Upload raw annotation folder with images/, labels/, and classes.txt
    
    Expected files:
    - images/ (all your images)
    - labels/ (YOLO format .txt files)
    - classes.txt (class names, one per line)
    
    The system will automatically organize into YOLO format with train/val split
    """
    try:
        # Check if project exists
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_data = project_result['project']
        
        # Create temporary directory to save uploaded files
        import tempfile
        import os
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to temporary directory
            raw_folder_path = os.path.join(temp_dir, "raw_annotations")
            os.makedirs(raw_folder_path, exist_ok=True)
            
            # Organize files by their path structure
            for file in raw_files:
                file_content = await file.read()
                
                # Extract directory structure from filename (browsers send full paths)
                file_path = file.filename
                if '/' in file_path:
                    # Extract relative path
                    rel_path = file_path
                    if rel_path.startswith('/'):
                        rel_path = rel_path[1:]
                    
                    # Create directory structure
                    full_path = os.path.join(raw_folder_path, rel_path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    
                    # Save file
                    with open(full_path, 'wb') as f:
                        f.write(file_content)
                else:
                    # File in root
                    with open(os.path.join(raw_folder_path, file.filename), 'wb') as f:
                        f.write(file_content)
            
            # Use the raw annotation processor to convert to YOLO format
            from services.ml.yolo_service import yolo_service
            
            output_name = f"project_{project_id}_{int(datetime.now().timestamp())}"
            result = yolo_service.process_raw_annotations(
                raw_folder_path=raw_folder_path,
                output_name=output_name,
                train_split=0.8,
                val_split=0.2
            )
            
            if result['status'] != 'success':
                raise HTTPException(status_code=400, detail=result.get('message', 'Failed to process raw annotations'))
            
            # Update database with file counts
            file_counts = {
                'train_images': result['train_samples'],
                'val_images': result['val_samples'],
                'total_images': result['total_samples'],
                'total_labels': result['total_samples'],  # Assume 1 label per image
                'classes': result['classes'],
                'dataset_path': result['output_path']
            }

            database_service.update_project_file_counts(project_id, file_counts)
            
            return JSONResponse({
                'status': 'success',
                'message': 'Raw annotations processed successfully',
                'output_path': result['output_path'],
                'data_yaml_path': result['data_yaml_path'],
                'classes': result['classes'],
                'train_samples': result['train_samples'],
                'val_samples': result['val_samples'],
                'total_samples': result['total_samples']
            })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process raw annotations: {str(e)}")

@router.post("/{project_id}/upload-cvat-dataset")
async def upload_cvat_dataset(
    project_id: str,
    cvat_zip: UploadFile = File(...)
):
    """
    Upload CVAT-exported ZIP file and convert to YOLO format

    Expected CVAT ZIP structure:
    - images/ (annotated images)
    - labels/ (YOLO format labels)
    - classes.txt (class definitions)
    - notes.json (CVAT metadata, optional)
    """
    try:
        # Check if project exists
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")

        if not cvat_zip:
            raise HTTPException(status_code=400, detail="No ZIP file uploaded")

        # Validate ZIP file
        if not cvat_zip.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")

        # Create temporary directory for processing
        import tempfile
        import shutil
        import zipfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / "cvat_dataset.zip"
            cvat_folder = temp_path / "cvat_export"

            # Save uploaded ZIP file
            with open(zip_path, 'wb') as f:
                content = await cvat_zip.read()
                f.write(content)

            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cvat_folder)

            # Find the actual dataset folder (might be nested)
            # Look for the folder containing images/, labels/, classes.txt
            dataset_folder = None
            for root, dirs, files in os.walk(cvat_folder):
                root_path = Path(root)
                if (root_path / 'images').exists() and (root_path / 'labels').exists() and (root_path / 'classes.txt').exists():
                    dataset_folder = root_path
                    break

            if not dataset_folder:
                raise HTTPException(status_code=400, detail="Invalid CVAT dataset structure. Expected: images/, labels/, classes.txt")

            # Convert CVAT to YOLO format using our utility
            import sys
            import importlib.util

            # Import the conversion utility
            # Get the project root directory (where the script is located)
            current_dir = os.getcwd()
            # If we're in the backend directory, go up to the project root
            if current_dir.endswith('/backend'):
                project_root = os.path.dirname(current_dir)
            else:
                project_root = current_dir

            convert_script_path = os.path.join(project_root, 'convert_cvat_to_yolo.py')
            if os.path.exists(convert_script_path):
                spec = importlib.util.spec_from_file_location("convert_cvat_to_yolo", convert_script_path)
                convert_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(convert_module)

                # Convert CVAT to YOLO
                yolo_output_name = f"cvat_{project_id}_{uuid.uuid4().hex[:8]}"
                result = convert_module.convert_cvat_to_yolo(
                    str(dataset_folder),
                    yolo_output_name,
                    train_ratio=0.8
                )

                # Move converted dataset to proper location
                yolo_dataset_path = Path(result['output_dir'])
                project_datasets_dir = Path(f"ml/datasets/{project_id}")
                project_datasets_dir.mkdir(parents=True, exist_ok=True)

                final_dataset_path = project_datasets_dir / yolo_output_name
                if final_dataset_path.exists():
                    shutil.rmtree(final_dataset_path)
                shutil.move(str(yolo_dataset_path), str(final_dataset_path))

                # Update database with file counts
                file_counts = {
                    'train_images': result['train_images'],
                    'val_images': result['val_images'],
                    'total_images': result['total_images'],
                    'total_labels': result.get('total_labels', result['train_images'] + result['val_images']),  # Assume 1 label per image
                    'classes': result['classes'],
                    'dataset_path': str(final_dataset_path)
                }

                database_service.update_project_file_counts(project_id, file_counts)

                return JSONResponse({
                    'status': 'success',
                    'message': 'CVAT dataset converted and uploaded successfully',
                    'output_path': str(final_dataset_path),
                    'data_yaml_path': str(final_dataset_path / 'data.yaml'),
                    'classes': result['classes'],
                    'train_images': result['train_images'],
                    'val_images': result['val_images'],
                    'total_images': result['total_images']
                })
            else:
                raise HTTPException(status_code=500, detail="CVAT conversion utility not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CVAT dataset: {str(e)}")

@router.get("/{project_id}/dataset-statistics")
async def get_dataset_statistics(project_id: str):
    """
    Get detailed statistics about a project's dataset
    """
    try:
        # Check if project exists
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_data = project_result['project']
        
        # Try to get dataset statistics if available
        from services.ml.yolo_service import yolo_service
        
        # This would check if the project has a processed dataset
        # For now, return basic file counts from database
        file_counts = project_data.get('file_counts', {})
        
        statistics = {
            'project_id': project_id,
            'project_type': project_data['project_type'],
            'file_counts': file_counts,
            'training_ready': (file_counts.get('training_images', 0) > 0),
            'has_annotations': (file_counts.get('annotation_files', 0) > 0),
            'last_updated': project_data.get('updated_at', project_data.get('created_at'))
        }
        
        return JSONResponse({
            'status': 'success',
            'statistics': statistics
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset statistics: {str(e)}")

@router.get("/{project_id}/debug-file-counts")
async def debug_project_file_counts(project_id: str):
    """
    Debug endpoint to check actual file counts in database vs what validation sees
    """
    try:
        # Check if project exists
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_data = project_result['project']
        
        # Get raw database records
        from backend.services.core.database_service import UploadedFile, DatabaseService
        
        db_service = DatabaseService()
        session = db_service.get_session()
        
        debug_info = {
            'project_id': project_id,
            'project_name': project_data['project_name'],
            'project_type': project_data['project_type'],
            'file_counts_from_get_project': project_data.get('file_counts', {}),
            'raw_database_records': [],
            'file_type_counts': {}
        }
        
        try:
            files = session.query(UploadedFile).filter(
                UploadedFile.project_id == project_id
            ).all()
            
            for file in files:
                debug_info['raw_database_records'].append({
                    'filename': file.filename,
                    'file_type': file.file_type,
                    'storage_path': file.storage_path,
                    'is_processed': file.is_processed
                })
                
                if file.file_type not in debug_info['file_type_counts']:
                    debug_info['file_type_counts'][file.file_type] = 0
                debug_info['file_type_counts'][file.file_type] += 1
                
        finally:
            session.close()
        
        return JSONResponse({
            'status': 'success',
            'debug_info': debug_info
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to debug file counts: {str(e)}")

@router.post("/{project_id}/refresh-file-counts")
async def refresh_project_file_counts(project_id: str):
    """
    Refresh/recalculate file counts for a project by scanning actual database records
    """
    try:
        # Check if project exists
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            raise HTTPException(status_code=404, detail="Project not found")
        
        from backend.services.core.database_service import UploadedFile, DatabaseService
        from sqlalchemy import func
        
        db_service = DatabaseService()
        session = db_service.get_session()
        
        try:
            # Count actual files in database
            count_results = session.query(
                UploadedFile.file_type, 
                func.count(UploadedFile.id)
            ).filter(
                UploadedFile.project_id == project_id
            ).group_by(UploadedFile.file_type).all()
            
            file_counts = {
                'training_images': 0,
                'defective_images': 0, 
                'test_images': 0,
                'annotation_files': 0
            }
            
            for file_type, count in count_results:
                if file_type in file_counts:
                    file_counts[file_type] = count
            
            # If we have annotation files, create a fake update to trigger the validation fix
            if file_counts['annotation_files'] > 0:
                # Create a minimal file_counts dict for update_project_file_counts
                update_data = {
                    'total_labels': file_counts['annotation_files'],
                    'train_images': file_counts['training_images'],
                    'val_images': file_counts.get('defective_images', 0),
                    'total_images': file_counts['training_images'] + file_counts.get('defective_images', 0)
                }
                
                # This will ensure annotation records exist
                database_service.update_project_file_counts(project_id, update_data)
            
            return JSONResponse({
                'status': 'success',
                'message': 'File counts refreshed successfully',
                'file_counts': file_counts
            })
            
        finally:
            session.close()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh file counts: {str(e)}")