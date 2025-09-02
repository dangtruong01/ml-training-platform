import os
import cv2
import json
import uuid
import base64
import shutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
import zipfile
import threading
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not available - YOLO training disabled")

class AutoAnnotationService:
    """
    Automated annotation service using GroundingDINO + YOLO/SAM2
    
    Two workflows:
    1. Object Detection: YOLO for defect bounding boxes
    2. Segmentation: SAM2 for precise defect masks
    
    Both workflows are independent and can be used separately.
    """
    
    def __init__(self):
        # Base directories
        self.base_dir = os.path.abspath("ml/auto_annotation")
        self.projects_dir = os.path.join(self.base_dir, "projects")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.results_dir = os.path.join(self.base_dir, "results")
        
        # Create directories
        for dir_path in [self.projects_dir, self.models_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Import existing services
        from backend.services.grounding_dino_service import grounding_dino_service
        self.grounding_dino = grounding_dino_service
        
        # Import new ROI extraction service for anomaly detection workflow
        from backend.services.roi_extraction_service import roi_extraction_service
        self.roi_extractor = roi_extraction_service
        
        print("🏗️ Auto-annotation service initialized")
    
    # =============================================================================
    # PROJECT MANAGEMENT
    # =============================================================================
    
    def create_project(self, project_name: str, project_type: str, description: str = "") -> Dict:
        """
        Create a new annotation project
        
        Args:
            project_name: Unique project name
            project_type: 'object_detection' or 'segmentation'
            description: Project description
        """
        try:
            if project_type not in ['object_detection', 'segmentation', 'anomaly_detection']:
                raise ValueError("project_type must be 'object_detection', 'segmentation', or 'anomaly_detection'")
            
            project_id = f"{project_name}_{uuid.uuid4().hex[:8]}"
            project_dir = os.path.join(self.projects_dir, project_id)
            
            # Create project structure based on project type
            if project_type == 'anomaly_detection':
                # Anomaly detection projects need different structure
                project_structure = {
                    'training_images': os.path.join(project_dir, 'training_images'),
                    'roi_cache': os.path.join(project_dir, 'roi_cache'),
                    'defective_images': os.path.join(project_dir, 'defective_images'), 
                    'defective_roi_cache': os.path.join(project_dir, 'defective_roi_cache'),
                    'anomaly_features': os.path.join(project_dir, 'anomaly_features'),
                    'defect_detection_results': os.path.join(project_dir, 'defect_detection_results')
                }
            else:
                # Regular object detection/segmentation projects
                project_structure = {
                    'training_images': os.path.join(project_dir, 'training_images'),
                    'annotations': os.path.join(project_dir, 'annotations'),
                    'models': os.path.join(project_dir, 'models'),
                    'inference_results': os.path.join(project_dir, 'inference_results')
                }
            
            for dir_path in project_structure.values():
                os.makedirs(dir_path, exist_ok=True)
            
            # Project metadata
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
            
            # Save metadata
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Created {project_type} project: {project_name} ({project_id})")
            
            return {
                'status': 'success',
                'project_id': project_id,
                'metadata': metadata,
                'structure': project_structure
            }
            
        except Exception as e:
            print(f"❌ Failed to create project: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_projects(self) -> List[Dict]:
        """Get list of all projects"""
        projects = []
        
        try:
            for project_dir in os.listdir(self.projects_dir):
                metadata_path = os.path.join(self.projects_dir, project_dir, 'project_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Update training images count
                    training_images_dir = os.path.join(self.projects_dir, project_dir, 'training_images')
                    if os.path.exists(training_images_dir):
                        metadata['training_images_count'] = len([f for f in os.listdir(training_images_dir) 
                                                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    
                    projects.append(metadata)
            
            return sorted(projects, key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            print(f"❌ Failed to get projects: {e}")
            return []
    
    def get_project_details(self, project_id: str) -> Dict:
        """Get detailed information about a specific project"""
        try:
            project_dir = os.path.join(self.projects_dir, project_id)
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            
            if not os.path.exists(metadata_path):
                return {'status': 'error', 'message': 'Project not found'}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get additional details
            training_images_dir = os.path.join(project_dir, 'training_images')
            annotations_dir = os.path.join(project_dir, 'annotations')
            
            details = {
                'metadata': metadata,
                'training_images_count': 0,
                'annotations_count': 0,
                'sample_images': [],
                'class_distribution': {}
            }
            
            # Count training images
            if os.path.exists(training_images_dir):
                image_files = [f for f in os.listdir(training_images_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                details['training_images_count'] = len(image_files)
                details['sample_images'] = image_files[:5]  # First 5 as samples
            
            # Count annotations
            if os.path.exists(annotations_dir):
                if metadata['project_type'] == 'object_detection':
                    # Count YOLO format annotations
                    details['annotations_count'] = len([f for f in os.listdir(annotations_dir) 
                                                       if f.endswith('.txt')])
                else:
                    # Count COCO format annotations
                    coco_file = os.path.join(annotations_dir, 'annotations.json')
                    if os.path.exists(coco_file):
                        with open(coco_file, 'r') as f:
                            coco_data = json.load(f)
                        details['annotations_count'] = len(coco_data.get('annotations', []))
            
            return {'status': 'success', 'details': details}
            
        except Exception as e:
            print(f"❌ Failed to get project details: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def delete_project(self, project_id: str) -> Dict:
        """
        Delete a project and all its associated data
        
        Args:
            project_id: Project ID to delete
        """
        try:
            project_dir = os.path.join(self.projects_dir, project_id)
            
            if not os.path.exists(project_dir):
                return {'status': 'error', 'message': 'Project not found'}
            
            # Load metadata to get project info
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            project_name = "Unknown"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    project_name = metadata.get('project_name', 'Unknown')
            
            # Remove entire project directory
            shutil.rmtree(project_dir)
            
            print(f"🗑️ Deleted project: {project_name} ({project_id})")
            
            return {
                'status': 'success',
                'message': f'Project "{project_name}" deleted successfully',
                'deleted_project_id': project_id
            }
            
        except Exception as e:
            print(f"❌ Failed to delete project {project_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # =============================================================================
    # DATA UPLOAD AND MANAGEMENT
    # =============================================================================
    
    def upload_training_data(
        self, 
        project_id: str, 
        images: List[bytes], 
        image_names: List[str],
        annotations_data: Optional[List[bytes]] = None,
        annotation_names: Optional[List[str]] = None,
        annotation_format: str = 'auto'
    ) -> Dict:
        """
        Upload training images and annotations to a project
        
        Args:
            project_id: Target project ID
            images: List of image bytes
            image_names: List of image filenames
            annotations_data: List of annotation file bytes
            annotation_names: List of annotation filenames
            annotation_format: 'yolo', 'coco', or 'auto'
        """
        try:
            project_dir = os.path.join(self.projects_dir, project_id)
            if not os.path.exists(project_dir):
                return {'status': 'error', 'message': 'Project not found'}
            
            # Load project metadata
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            training_images_dir = os.path.join(project_dir, 'training_images')
            annotations_dir = os.path.join(project_dir, 'annotations')
            
            # Save training images
            saved_images = []
            for image_data, image_name in zip(images, image_names):
                image_path = os.path.join(training_images_dir, image_name)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                saved_images.append(image_name)
            
            print(f"💾 Saved {len(saved_images)} training images")
            
            # Process annotations if provided
            annotation_result = {'status': 'no_annotations'}
            if annotations_data and annotation_names:
                annotation_result = self._process_annotations(
                    annotations_data, annotation_names, annotations_dir, metadata['project_type'], annotation_format
                )
            
            # Update metadata
            metadata['training_images_count'] = len(saved_images)
            metadata['status'] = 'data_uploaded'
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'status': 'success',
                'images_uploaded': len(saved_images),
                'annotation_result': annotation_result,
                'message': f'Uploaded {len(saved_images)} images successfully'
            }
            
        except Exception as e:
            print(f"❌ Failed to upload training data: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _process_annotations(self, annotations_data: List[bytes], annotation_names: List[str], 
                           annotations_dir: str, project_type: str, annotation_format: str) -> Dict:
        """Process and validate annotation data"""
        try:
            if project_type == 'object_detection':
                # Process YOLO format (individual txt files)
                return self._process_yolo_annotations(annotations_data, annotation_names, annotations_dir)
            else:
                # Process COCO format (single json file)
                if len(annotations_data) != 1:
                    return {'status': 'error', 'message': 'Segmentation projects require exactly one COCO JSON file'}
                return self._process_coco_annotations(annotations_data[0], annotations_dir)
                
        except Exception as e:
            print(f"❌ Failed to process annotations: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _process_yolo_annotations(self, annotations_data: List[bytes], annotation_names: List[str], annotations_dir: str) -> Dict:
        """Process YOLO format annotations (individual .txt files)"""
        try:
            if not annotations_data or not annotation_names:
                return {'status': 'error', 'message': 'No annotation files provided'}
            
            print(f"📝 Processing {len(annotations_data)} annotation files")
            
            # Process each annotation file
            classes_set = set()
            files_processed = 0
            annotation_stats = {}
            classes_content = None
            
            for annotation_data, annotation_name in zip(annotations_data, annotation_names):
                # Skip non-.txt files
                if not annotation_name.lower().endswith('.txt'):
                    print(f"⚠️ Skipping non-.txt file: {annotation_name}")
                    continue
                
                # Handle classes.txt specially
                if annotation_name.lower() == 'classes.txt':
                    classes_content = annotation_data.decode('utf-8')
                    classes_path = os.path.join(annotations_dir, 'classes.txt')
                    with open(classes_path, 'w') as f:
                        f.write(classes_content)
                    print("📋 Found and saved classes.txt")
                    continue
                
                # Process regular annotation file
                try:
                    annotation_content = annotation_data.decode('utf-8')
                except UnicodeDecodeError:
                    print(f"⚠️ Could not decode {annotation_name} as UTF-8")
                    continue
                
                # Save annotation file
                annotation_path = os.path.join(annotations_dir, annotation_name)
                with open(annotation_path, 'w') as f:
                    f.write(annotation_content)
                
                # Parse and validate annotation content
                lines = annotation_content.strip().split('\n')
                valid_annotations = 0
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"⚠️ Invalid annotation format in {annotation_name}:{line_num}: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate normalized coordinates (should be 0-1)
                        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                               0 <= width <= 1 and 0 <= height <= 1):
                            print(f"⚠️ Coordinates not normalized in {annotation_name}:{line_num}: {line}")
                            continue
                        
                        classes_set.add(class_id)
                        valid_annotations += 1
                        
                    except ValueError:
                        print(f"⚠️ Invalid number format in {annotation_name}:{line_num}: {line}")
                        continue
                
                annotation_stats[annotation_name] = valid_annotations
                files_processed += 1
                print(f"  ✅ {annotation_name}: {valid_annotations} annotations")
            
            # Generate classes.txt if not provided
            if not classes_content and classes_set:
                max_class_id = max(classes_set)
                class_names = []
                
                for i in range(max_class_id + 1):
                    if i in classes_set:
                        class_names.append(f"class_{i}")
                    else:
                        class_names.append(f"unused_{i}")
                
                # Save generated classes.txt
                classes_path = os.path.join(annotations_dir, 'classes.txt')
                with open(classes_path, 'w') as f:
                    f.write('\n'.join(class_names))
                
                print(f"🏷️ Generated classes.txt with {len(class_names)} classes")
            
            total_annotations = sum(annotation_stats.values())
            
            print(f"✅ Processed {files_processed} annotation files with {total_annotations} total annotations")
            print(f"🏷️ Found {len(classes_set)} unique classes: {sorted(classes_set)}")
            
            return {
                'status': 'success',
                'format': 'yolo',
                'files_processed': files_processed,
                'total_annotations': total_annotations,
                'classes_found': sorted(list(classes_set)),
                'annotation_stats': annotation_stats
            }
            
        except Exception as e:
            print(f"❌ Error processing YOLO annotations: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _process_coco_annotations(self, annotations_data: Union[str, bytes], annotations_dir: str) -> Dict:
        """Process COCO format annotations (JSON file with segmentation data)"""
        try:
            if isinstance(annotations_data, str):
                # File path provided
                with open(annotations_data, 'r') as f:
                    coco_data = json.load(f)
            else:
                # Bytes data provided (JSON file)
                coco_json = annotations_data.decode('utf-8')
                coco_data = json.loads(coco_json)
            
            # Validate COCO format
            required_fields = ['images', 'annotations', 'categories']
            for field in required_fields:
                if field not in coco_data:
                    return {'status': 'error', 'message': f'Missing required COCO field: {field}'}
            
            # Save COCO JSON file
            coco_path = os.path.join(annotations_dir, 'annotations.json')
            with open(coco_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            # Extract categories for classes.txt
            categories = coco_data['categories']
            class_names = []
            for cat in sorted(categories, key=lambda x: x['id']):
                class_names.append(cat['name'])
            
            # Save classes.txt
            classes_path = os.path.join(annotations_dir, 'classes.txt')
            with open(classes_path, 'w') as f:
                f.write('\n'.join(class_names))
            
            annotations_count = len(coco_data['annotations'])
            images_count = len(coco_data['images'])
            categories_count = len(coco_data['categories'])
            
            print(f"✅ Processed COCO annotations: {annotations_count} annotations, {images_count} images, {categories_count} categories")
            
            return {
                'status': 'success',
                'format': 'coco',
                'annotations_processed': annotations_count,
                'images_count': images_count,
                'categories_count': categories_count,
                'categories': [cat['name'] for cat in categories]
            }
            
        except json.JSONDecodeError as e:
            return {'status': 'error', 'message': f'Invalid JSON format: {str(e)}'}
        except Exception as e:
            print(f"❌ Error processing COCO annotations: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # =============================================================================
    # MODEL TRAINING
    # =============================================================================
    
    def start_training(self, project_id: str, training_params: Dict = None) -> Dict:
        """
        Start model training for a project
        
        Args:
            project_id: Project to train
            training_params: Training configuration parameters
        """
        try:
            project_dir = os.path.join(self.projects_dir, project_id)
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            
            if not os.path.exists(metadata_path):
                return {'status': 'error', 'message': 'Project not found'}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if metadata['project_type'] == 'object_detection':
                return self._train_yolo_model(project_id, training_params or {})
            else:
                return self._train_sam2_model(project_id, training_params or {})
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _train_yolo_model(self, project_id: str, training_params: Dict) -> Dict:
        """Train YOLOv8 model for object detection"""
        print(f"🚀 Starting YOLO training for project {project_id}")
        
        if not YOLO_AVAILABLE:
            return {
                'status': 'error',
                'message': 'YOLO training not available - ultralytics package required'
            }
        
        try:
            project_dir = os.path.join(self.projects_dir, project_id)
            
            # Default parameters
            default_params = {
                'epochs': 50,
                'batch_size': 16,
                'image_size': 640,
                'patience': 10
            }
            params = {**default_params, **training_params}
            
            # Update project metadata
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['training_status'] = 'running'
            metadata['training_started_at'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Start training in background thread
            training_thread = threading.Thread(
                target=self._run_yolo_training_background,
                args=(project_id, params)
            )
            training_thread.daemon = True
            training_thread.start()
            
            return {
                'status': 'success',
                'message': 'YOLO training started',
                'training_params': params,
                'model_type': 'yolov8'
            }
            
        except Exception as e:
            print(f"❌ Failed to start YOLO training: {e}")
            return {
                'status': 'error',
                'message': f'Failed to start training: {str(e)}'
            }
    
    def _run_yolo_training_background(self, project_id: str, params: Dict):
        """Run YOLO training in background thread"""
        try:
            project_dir = os.path.join(self.projects_dir, project_id)
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            
            print(f"🏋️ Running YOLO training for {project_id}")
            
            # Prepare dataset configuration
            dataset_config = self._prepare_yolo_dataset(project_id)
            if not dataset_config:
                raise Exception("Failed to prepare dataset")
            
            # Initialize YOLO model
            model = YOLO('yolov8n.pt')  # Start with nano model for faster training
            
            # Train the model
            print(f"📚 Training YOLO model with {params['epochs']} epochs...")
            results = model.train(
                data=dataset_config,
                epochs=params['epochs'],
                batch=params['batch_size'],
                imgsz=params['image_size'],
                patience=params['patience'],
                project=os.path.join(project_dir, 'models'),
                name='yolo_training',
                save=True,
                verbose=True
            )
            
            # Save model path
            model_path = os.path.join(project_dir, 'models', 'yolo_training', 'weights', 'best.pt')
            
            # Update metadata with completion
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['training_status'] = 'completed'
            metadata['training_completed_at'] = datetime.now().isoformat()
            metadata['model_path'] = model_path
            metadata['model_metrics'] = {
                'map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0))
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ YOLO training completed for {project_id}")
            
        except Exception as e:
            print(f"❌ YOLO training failed for {project_id}: {e}")
            
            # Update metadata with failure
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['training_status'] = 'failed'
                metadata['training_failed_at'] = datetime.now().isoformat()
                metadata['training_error'] = str(e)
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except:
                pass
    
    def _prepare_yolo_dataset(self, project_id: str) -> str:
        """Prepare YOLO dataset configuration file"""
        try:
            project_dir = os.path.join(self.projects_dir, project_id)
            
            # Read classes from annotations
            classes_file = None
            annotations_dir = os.path.join(project_dir, 'annotations')
            
            # Look for classes.txt or derive from annotations
            if os.path.exists(os.path.join(annotations_dir, 'classes.txt')):
                with open(os.path.join(annotations_dir, 'classes.txt'), 'r') as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
            else:
                # Extract classes from annotation files
                classes = set()
                for file in os.listdir(annotations_dir):
                    if file.endswith('.txt'):
                        with open(os.path.join(annotations_dir, file), 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    classes.add(class_id)
                
                # Create default class names
                classes = [f"class_{i}" for i in range(max(classes) + 1)]
            
            # Create dataset structure for YOLO
            dataset_dir = os.path.join(project_dir, 'yolo_dataset')
            os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)
            
            # Copy images and labels
            training_images_dir = os.path.join(project_dir, 'training_images')
            annotations_dir = os.path.join(project_dir, 'annotations')
            
            for image_file in os.listdir(training_images_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Copy image
                    src_img = os.path.join(training_images_dir, image_file)
                    dst_img = os.path.join(dataset_dir, 'images', 'train', image_file)
                    shutil.copy2(src_img, dst_img)
                    
                    # Copy corresponding label
                    label_file = os.path.splitext(image_file)[0] + '.txt'
                    src_label = os.path.join(annotations_dir, label_file)
                    if os.path.exists(src_label):
                        dst_label = os.path.join(dataset_dir, 'labels', 'train', label_file)
                        shutil.copy2(src_label, dst_label)
            
            # Create dataset.yaml
            dataset_config = {
                'path': dataset_dir,
                'train': 'images/train',
                'val': 'images/train',  # Use same for validation (small dataset)
                'names': {i: name for i, name in enumerate(classes)}
            }
            
            config_path = os.path.join(dataset_dir, 'dataset.yaml')
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(dataset_config, f, default_flow_style=False)
            
            print(f"📁 Dataset prepared at {dataset_dir}")
            return config_path
            
        except Exception as e:
            print(f"❌ Failed to prepare YOLO dataset: {e}")
            return None
    
    def get_training_status(self, project_id: str) -> Dict:
        """Get current training status and progress"""
        try:
            project_dir = os.path.join(self.projects_dir, project_id)
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            
            if not os.path.exists(metadata_path):
                return {
                    'status': 'error',
                    'message': 'Project not found'
                }
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            training_status = metadata.get('training_status', 'not_started')
            
            result = {
                'status': 'success',
                'training_status': training_status,
                'progress': 0,
                'current_epoch': 0,
                'total_epochs': 0,
                'metrics': metadata.get('model_metrics', {}),
                'logs': []
            }
            
            # If training is running, try to get more detailed progress
            if training_status == 'running':
                # Check for YOLO training logs
                models_dir = os.path.join(project_dir, 'models', 'yolo_training')
                if os.path.exists(models_dir):
                    # Look for training results
                    results_csv = os.path.join(models_dir, 'results.csv')
                    if os.path.exists(results_csv):
                        try:
                            import pandas as pd
                            df = pd.read_csv(results_csv)
                            if not df.empty:
                                current_epoch = len(df)
                                result['current_epoch'] = current_epoch
                                
                                # Calculate progress (assuming we know total epochs from training params)
                                # This is approximate since we don't store training params in metadata yet
                                total_epochs = 50  # Default value
                                result['total_epochs'] = total_epochs
                                result['progress'] = min(100, int((current_epoch / total_epochs) * 100))
                                
                                # Get latest metrics
                                if current_epoch > 0:
                                    latest = df.iloc[-1]
                                    result['metrics'] = {
                                        'train_loss': float(latest.get('train/box_loss', 0)),
                                        'val_loss': float(latest.get('val/box_loss', 0)),
                                        'map50': float(latest.get('metrics/mAP50(B)', 0)),
                                        'map50_95': float(latest.get('metrics/mAP50-95(B)', 0))
                                    }
                        except Exception as e:
                            print(f"⚠️ Error reading training progress: {e}")
            
            elif training_status == 'completed':
                result['progress'] = 100
                result['current_epoch'] = result['total_epochs'] = metadata.get('total_epochs', 50)
            
            elif training_status == 'failed':
                result['progress'] = 0
                result['error'] = metadata.get('training_error', 'Unknown error')
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting training status: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _train_sam2_model(self, project_id: str, training_params: Dict) -> Dict:
        """Train SAM2 few-shot model for segmentation"""
        # Placeholder for SAM2 few-shot training implementation
        print(f"🚀 Starting SAM2 training for project {project_id}")
        
        # Default parameters
        default_params = {
            'few_shot_examples': 5,
            'similarity_threshold': 0.8,
            'mask_threshold': 0.5
        }
        params = {**default_params, **training_params}
        
        # This would integrate with SAM2 few-shot learning
        return {
            'status': 'success',
            'message': 'SAM2 few-shot training started',
            'training_params': params,
            'model_type': 'sam2'
        }
    
    # =============================================================================
    # AUTO-ANNOTATION INFERENCE
    # =============================================================================
    
    def annotate_images(
        self, 
        project_id: str, 
        images: List[bytes], 
        image_names: List[str],
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Automatically annotate images using trained model
        
        Args:
            project_id: Project with trained model
            images: Images to annotate
            image_names: Image filenames
            component_type: Component to detect with GroundingDINO
            confidence_threshold: Detection confidence threshold
        """
        try:
            # Check if model exists
            project_dir = os.path.join(self.projects_dir, project_id)
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            
            if not os.path.exists(metadata_path):
                return {'status': 'error', 'message': 'Project not found'}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if metadata['training_status'] != 'completed':
                return {'status': 'error', 'message': 'Model not trained yet'}
            
            results = []
            for image_data, image_name in zip(images, image_names):
                
                # Save temporary image
                temp_image_path = f"/tmp/{uuid.uuid4().hex}_{image_name}"
                with open(temp_image_path, 'wb') as f:
                    f.write(image_data)
                
                try:
                    if metadata['project_type'] == 'object_detection':
                        result = self._annotate_with_yolo(
                            temp_image_path, project_id, confidence_threshold
                        )
                    else:
                        result = self._annotate_with_sam2(
                            temp_image_path, project_id, confidence_threshold
                        )
                    
                    result['filename'] = image_name
                    results.append(result)
                    
                finally:
                    # Cleanup temp file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
            
            return {
                'status': 'success',
                'project_type': metadata['project_type'],
                'results': results,
                'total_images': len(images)
            }
            
        except Exception as e:
            print(f"❌ Auto-annotation failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _annotate_with_yolo(self, image_path: str, project_id: str, confidence: float) -> Dict:
        """Annotate image using trained YOLO model (direct inference on full image)"""
        try:
            print(f"🔍 Starting YOLO annotation for project {project_id} with confidence {confidence}")
            
            # Load trained YOLO model
            project_dir = os.path.join(self.projects_dir, project_id)
            metadata_path = os.path.join(project_dir, 'project_metadata.json')
            
            print(f"📂 Looking for project at: {project_dir}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_path = metadata.get('model_path')
            print(f"🎯 Model path from metadata: {model_path}")
            
            if not model_path:
                return {
                    'status': 'error',
                    'message': 'No model path found in project metadata'
                }
                
            if not os.path.exists(model_path):
                print(f"❌ Model file does not exist at: {model_path}")
                # Let's check what files actually exist
                models_dir = os.path.dirname(model_path)
                if os.path.exists(models_dir):
                    print(f"📁 Files in models directory: {os.listdir(models_dir)}")
                return {
                    'status': 'error',
                    'message': f'Trained model file not found at: {model_path}'
                }
            
            print(f"✅ Model file exists: {model_path}")
            
            # Load YOLO model
            if not YOLO_AVAILABLE:
                return {
                    'status': 'error',
                    'message': 'YOLO not available - ultralytics package required'
                }
            
            print(f"🤖 Loading YOLO model from: {model_path}")
            model = YOLO(model_path)
            print(f"🎯 Model loaded successfully. Classes: {model.names}")
            
            # Run inference on full image
            print(f"🖼️ Running inference on image: {image_path}")
            
            # First try with very low confidence to see what model actually detects
            debug_results = model.predict(
                source=image_path,
                conf=0.01,  # Very low threshold to see all detections
                save=False,
                verbose=False
            )
            
            # Log what we find at low confidence
            if debug_results and len(debug_results) > 0 and debug_results[0].boxes is not None:
                debug_boxes = debug_results[0].boxes
                print(f"🔬 Debug: Found {len(debug_boxes)} detections at conf=0.01:")
                for i, box in enumerate(debug_boxes):
                    conf_score = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    print(f"   Detection {i+1}: class={class_id}, confidence={conf_score:.3f}")
            else:
                print(f"🔬 Debug: No detections found even at conf=0.01")
            
            # Now run with user's confidence threshold
            results = model.predict(
                source=image_path,
                conf=confidence,
                save=False,
                verbose=True  # Enable verbose to see what's happening
            )
            
            print(f"📊 Inference completed. Results count: {len(results) if results else 0}")
            
            # Process results
            detections = []
            image = cv2.imread(image_path)
            
            if results and len(results) > 0:
                result = results[0]
                print(f"🔍 Processing result - has boxes: {result.boxes is not None}")
                
                if result.boxes is not None:
                    boxes = result.boxes
                    print(f"📦 Found {len(boxes)} boxes")
                    
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates (xyxy format)
                        xyxy = box.xyxy[0].cpu().numpy()
                        confidence_score = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        print(f"🎯 Detection {i+1}: class_id={class_id}, confidence={confidence_score:.3f}, bbox={xyxy}")
                        
                        # Convert to x, y, w, h format
                        x1, y1, x2, y2 = xyxy
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Get class name from model
                        class_name = model.names.get(class_id, f'class_{class_id}')
                        
                        detection = {
                            'class': class_name,
                            'confidence': float(confidence_score),
                            'bbox': [float(x1), float(y1), float(width), float(height)],
                            'class_id': class_id
                        }
                        
                        detections.append(detection)
                        print(f"✅ Added detection: {detection}")
                else:
                    print("❌ No boxes found in result")
            else:
                print("❌ No results from model prediction")
            
            print(f"📊 Total detections found: {len(detections)}")
            
            # Draw annotations on image
            annotated_image = self._draw_yolo_detections(image, detections)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'status': 'success',
                'annotation_type': 'object_detection',
                'defect_detections': detections,
                'annotated_image_base64': image_base64,
                'total_detections': len(detections)
            }
            
        except Exception as e:
            print(f"❌ YOLO annotation failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _annotate_with_sam2(self, image_path: str, project_id: str, confidence: float) -> Dict:
        """Annotate image using SAM2 model"""
        try:
            # Similar to YOLO but with segmentation masks instead of bounding boxes
            # Placeholder implementation
            return {
                'status': 'success',
                'annotation_type': 'segmentation',
                'defect_masks': [],
                'annotated_image_base64': None
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _draw_yolo_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw YOLO detection bounding boxes on image"""
        annotated = image.copy()
        
        # Define colors for different classes
        colors = {
            'scratch': (0, 255, 255),      # Yellow
            'dent': (0, 0, 255),           # Red  
            'contamination': (255, 0, 0),   # Blue
            'crack': (255, 0, 255),        # Magenta
            'corrosion': (0, 255, 0),      # Green
            'class_0': (0, 255, 255),      # Default for class_0
            'class_1': (0, 0, 255),        # Default for class_1
            'class_2': (255, 0, 0),        # Default for class_2
            'class_3': (255, 0, 255),      # Default for class_3
            'class_4': (0, 255, 0),        # Default for class_4
        }
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Get color for this class
            class_name = detection['class']
            color = colors.get(class_name, (0, 255, 0))  # Default green if unknown class
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Add label with background
            label = f"{class_name}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated

# Create global instance
auto_annotation_service = AutoAnnotationService()