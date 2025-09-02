import os
import cv2
import json
import uuid
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class ROIExtractionService:
    """
    ROI (Region of Interest) extraction service for anomaly detection
    
    Uses GroundingDINO to identify and crop metal components from images,
    eliminating noise like trays, bolts, QR codes for better anomaly detection.
    """
    
    def __init__(self):
        # Import existing GroundingDINO service
        from backend.services.grounding_dino_service import grounding_dino_service
        self.grounding_dino = grounding_dino_service
        
        # Base directory for auto annotation projects
        self.projects_base_dir = os.path.abspath("ml/auto_annotation/projects")
        
        print("🎯 ROI Extraction Service initialized")
    
    def extract_roi_from_images(
        self,
        image_paths: List[str],
        component_description: str = "metal plate",
        confidence_threshold: float = 0.3,
        project_id: Optional[str] = None,
        image_type: str = "training"
    ) -> Dict:
        """
        Extract ROI from multiple images using GroundingDINO
        
        Args:
            image_paths: List of image file paths
            component_description: Description of component to detect
            confidence_threshold: Detection confidence threshold
            project_id: Project ID for organizing results  
            image_type: Type of images ("training" or "defective")
            
        Returns:
            Dict with extraction results and cropped image data
        """
        try:
            if not project_id:
                return {
                    'status': 'error',
                    'message': 'Project ID is required for ROI extraction'
                }
            
            # Create project-specific ROI directory
            project_dir = os.path.join(self.projects_base_dir, project_id)
            if image_type == "training":
                roi_cache_dir = os.path.join(project_dir, "roi_cache")
            else:  # defective
                roi_cache_dir = os.path.join(project_dir, "defective_roi_cache")
            os.makedirs(roi_cache_dir, exist_ok=True)
            
            print(f"🔍 Extracting ROI from {len(image_paths)} images")
            print(f"🎯 Looking for: '{component_description}'")
            print(f"📁 Saving to: {roi_cache_dir}")
            
            results = {
                'status': 'success',
                'total_images': len(image_paths),
                'successful_extractions': 0,
                'failed_extractions': 0,
                'roi_data': [],
                'average_roi_size': None,
                'component_description': component_description
            }
            
            roi_sizes = []
            
            for i, image_path in enumerate(image_paths):
                print(f"📸 Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Extract ROI from single image
                roi_result = self._extract_single_roi(
                    image_path, component_description, confidence_threshold, roi_cache_dir
                )
                
                if roi_result['status'] == 'success':
                    results['successful_extractions'] += 1
                    results['roi_data'].append(roi_result)
                    roi_sizes.append(roi_result['roi_size'])
                else:
                    results['failed_extractions'] += 1
                    results['roi_data'].append({
                        'image_path': image_path,
                        'status': 'failed',
                        'error': roi_result.get('message', 'Unknown error')
                    })
            
            # Calculate average ROI size
            if roi_sizes:
                avg_width = sum(size[0] for size in roi_sizes) / len(roi_sizes)
                avg_height = sum(size[1] for size in roi_sizes) / len(roi_sizes)
                results['average_roi_size'] = (int(avg_width), int(avg_height))
            
            success_rate = results['successful_extractions'] / results['total_images']
            print(f"✅ ROI extraction completed: {success_rate:.1%} success rate")
            
            return results
            
        except Exception as e:
            print(f"❌ ROI extraction failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _extract_single_roi(
        self, 
        image_path: str, 
        component_description: str, 
        confidence_threshold: float,
        roi_cache_dir: str
    ) -> Dict:
        """Extract ROI from a single image"""
        try:
            # Use GroundingDINO to detect component
            grounding_result = self.grounding_dino.annotate_with_prompts(
                image_path, [component_description], confidence_threshold
            )
            
            if grounding_result['status'] != 'success' or not grounding_result['detections']:
                return {
                    'status': 'failed',
                    'message': f'No {component_description} detected'
                }
            
            # Select best detection (highest confidence)
            best_detection = max(
                grounding_result['detections'], 
                key=lambda x: x['confidence']
            )
            
            # Load original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                return {
                    'status': 'failed',
                    'message': 'Could not load image'
                }
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in best_detection['bbox']]
            
            # Add padding around detection (10% of bbox size)
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            # Apply padding with image bounds checking
            height, width = original_image.shape[:2]
            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(width, x2 + padding_x)
            y2_padded = min(height, y2 + padding_y)
            
            # Crop ROI
            roi_image = original_image[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Generate unique filename for cropped ROI
            roi_filename = f"roi_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
            roi_path = os.path.join(roi_cache_dir, roi_filename)
            
            # Save cropped ROI
            cv2.imwrite(roi_path, roi_image)
            
            return {
                'status': 'success',
                'original_image_path': image_path,
                'roi_image_path': roi_path,
                'roi_bbox': [x1_padded, y1_padded, x2_padded, y2_padded],
                'roi_size': (x2_padded - x1_padded, y2_padded - y1_padded),
                'detection_confidence': best_detection['confidence'],
                'padding_applied': (padding_x, padding_y)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': str(e)
            }
    
    def get_roi_statistics(self, roi_data: List[Dict]) -> Dict:
        """Calculate statistics from ROI extraction results"""
        try:
            successful_rois = [roi for roi in roi_data if roi.get('status') == 'success']
            
            if not successful_rois:
                return {
                    'status': 'error',
                    'message': 'No successful ROI extractions to analyze'
                }
            
            # Size statistics
            sizes = [roi['roi_size'] for roi in successful_rois]
            widths = [size[0] for size in sizes]
            heights = [size[1] for size in sizes]
            
            # Confidence statistics
            confidences = [roi['detection_confidence'] for roi in successful_rois]
            
            return {
                'status': 'success',
                'total_rois': len(successful_rois),
                'size_stats': {
                    'width': {
                        'min': min(widths),
                        'max': max(widths),
                        'mean': sum(widths) / len(widths),
                        'std': np.std(widths)
                    },
                    'height': {
                        'min': min(heights),
                        'max': max(heights),
                        'mean': sum(heights) / len(heights),
                        'std': np.std(heights)
                    }
                },
                'confidence_stats': {
                    'min': min(confidences),
                    'max': max(confidences),
                    'mean': sum(confidences) / len(confidences),
                    'std': np.std(confidences)
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def cleanup_roi_cache(self, max_age_hours: int = 24) -> Dict:
        """Clean up old ROI cache files"""
        try:
            import time
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            deleted_count = 0
            
            for filename in os.listdir(self.roi_cache_dir):
                file_path = os.path.join(self.roi_cache_dir, filename)
                
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        deleted_count += 1
            
            return {
                'status': 'success',
                'deleted_files': deleted_count,
                'message': f'Cleaned up {deleted_count} old ROI cache files'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def extract_roi_from_defective_images(
        self,
        defective_image_paths: List[str],
        component_description: str = "metal plate",
        confidence_threshold: float = 0.3,
        project_id: Optional[str] = None
    ) -> Dict:
        """
        Extract ROI from defective images and save to separate cache
        
        Args:
            defective_image_paths: List of defective image file paths
            component_description: Description of component to detect
            confidence_threshold: Detection confidence threshold
            project_id: Project ID for organizing extracted ROIs
            
        Returns:
            Dict with extraction results and cropped image data
        """
        try:
            if not project_id:
                return {
                    'status': 'error',
                    'message': 'Project ID is required for defective ROI extraction'
                }
            
            # Create project-specific defective ROI directory
            project_dir = os.path.join(self.projects_base_dir, project_id)
            defective_roi_cache_dir = os.path.join(project_dir, "defective_roi_cache")
            os.makedirs(defective_roi_cache_dir, exist_ok=True)
            
            print(f"🔍 Extracting ROI from {len(defective_image_paths)} defective images")
            print(f"🎯 Looking for: '{component_description}'")
            print(f"📁 Saving to: {defective_roi_cache_dir}")
            
            results = {
                'status': 'success',
                'total_images': len(defective_image_paths),
                'successful_extractions': 0,
                'failed_extractions': 0,
                'roi_data': [],
                'average_roi_size': None,
                'component_description': component_description
            }
            
            roi_sizes = []
            
            for i, image_path in enumerate(defective_image_paths):
                print(f"📸 Processing defective image {i+1}/{len(defective_image_paths)}: {os.path.basename(image_path)}")
                
                # Extract ROI from single defective image
                roi_result = self._extract_single_roi(
                    image_path, component_description, confidence_threshold, defective_roi_cache_dir
                )
                
                if roi_result['status'] == 'success':
                    results['successful_extractions'] += 1
                    results['roi_data'].append(roi_result)
                    roi_sizes.append(roi_result['roi_size'])
                else:
                    results['failed_extractions'] += 1
                    results['roi_data'].append({
                        'image_path': image_path,
                        'status': 'failed',
                        'error': roi_result.get('message', 'Unknown error')
                    })
            
            # Calculate average ROI size
            if roi_sizes:
                avg_width = sum(size[0] for size in roi_sizes) / len(roi_sizes)
                avg_height = sum(size[1] for size in roi_sizes) / len(roi_sizes)
                results['average_roi_size'] = (int(avg_width), int(avg_height))
            
            success_rate = results['successful_extractions'] / results['total_images']
            print(f"✅ Defective ROI extraction completed: {success_rate:.1%} success rate")
            
            return results
            
        except Exception as e:
            print(f"❌ Defective ROI extraction failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Create global instance
roi_extraction_service = ROIExtractionService()