import os
import cv2
import numpy as np
import json
import uuid
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import base64

class GroundingDINOSAM2Service:
    """
    Integrated service that combines GroundingDINO for object detection 
    with SAM2 for precise segmentation
    """
    
    def __init__(self):
        self.results_dir = os.path.abspath(os.path.join("ml", "results", "grounding_dino_sam2"))
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Import services
        from backend.services.grounding_dino_service import grounding_dino_service
        from backend.services.sam2_service import sam2_service
        
        self.grounding_dino = grounding_dino_service
        self.sam2 = sam2_service
        
        print("🔧 GroundingDINO + SAM2 integrated service initialized")
    
    def extract_segmentation_based_roi(
        self, 
        image_paths: List[str], 
        component_description: str = "metal plate", 
        confidence_threshold: float = 0.3, 
        project_id: Optional[str] = None,
        image_type: str = "training"  # "training" or "defective"
    ) -> Dict:
        """
        Extract ROI using precise segmentation masks instead of bounding boxes
        
        Workflow:
        1. GroundingDINO finds component -> gives bounding box
        2. SAM2 generates precise segmentation mask within that box
        3. Extract masked region (only component pixels, no background)
        4. Save as transparent PNG or masked ROI image
        
        Args:
            image_paths: List of image paths to process
            component_description: Text description for GroundingDINO
            confidence_threshold: Minimum confidence for detection
            project_id: Project ID for saving results
            image_type: "training" or "defective" for directory naming
        """
        try:
            if not project_id:
                return {
                    'status': 'error',
                    'message': 'Project ID is required for segmentation-based ROI extraction'
                }
            
            # Create project directories
            project_dir = os.path.join("ml/auto_annotation/projects", project_id)
            
            if image_type == "training":
                roi_dir = os.path.join(project_dir, "roi_cache")
                masks_dir = os.path.join(project_dir, "segmentation_masks")
            else:  # defective
                roi_dir = os.path.join(project_dir, "defective_roi_cache")
                masks_dir = os.path.join(project_dir, "defective_segmentation_masks")
                
            os.makedirs(roi_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            print(f"🎭 Segmentation-based ROI extraction for {len(image_paths)} images")
            print(f"🎯 Looking for: '{component_description}'")
            print(f"📁 Saving ROI to: {roi_dir}")
            print(f"📁 Saving masks to: {masks_dir}")
            
            results = {
                'status': 'success',
                'total_images': len(image_paths),
                'successful_extractions': 0,
                'roi_data': [],
                'failed_images': []
            }
            
            for i, image_path in enumerate(image_paths):
                try:
                    print(f"\n🖼️  Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                    
                    # Step 1: GroundingDINO detection
                    grounding_result = self.grounding_dino.annotate_with_prompts(
                        image_path, [component_description], confidence_threshold
                    )
                    
                    if grounding_result['status'] != 'success' or not grounding_result['detections']:
                        print(f"❌ GroundingDINO failed to detect component")
                        results['failed_images'].append({
                            'image_path': image_path,
                            'error': 'Component not detected by GroundingDINO'
                        })
                        continue
                    
                    # Use the highest confidence detection
                    best_detection = max(grounding_result['detections'], key=lambda x: x['confidence'])
                    # GroundingDINO returns bbox as [x1, y1, x2, y2] list, not dict
                    x1, y1, x2, y2 = best_detection['bbox']
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    print(f"✅ GroundingDINO detected component: {best_detection['confidence']:.2f} confidence")
                    
                    # Step 2: SAM2 segmentation within bounding box
                    print(f"🎭 Generating precise segmentation mask...")
                    
                    # Load original image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"❌ Failed to load image: {image_path}")
                        continue
                    
                    # Generate segmentation mask using SAM2
                    mask, score, logits = self.sam2.segment_with_box(image, bbox)
                    
                    if mask is None:
                        print(f"❌ SAM2 failed to generate mask")
                        results['failed_images'].append({
                            'image_path': image_path,
                            'error': 'SAM2 segmentation failed'
                        })
                        continue
                    
                    print(f"✅ SAM2 generated segmentation mask: {score:.3f} score")
                    print(f"🎭 Mask stats: {np.sum(mask)} pixels segmented out of {mask.size} total ({np.sum(mask)/mask.size*100:.1f}%)")
                    
                    # Step 3: Extract masked ROI
                    masked_roi = self._extract_masked_roi(image, mask)
                    
                    if masked_roi is None:
                        print(f"❌ Failed to extract masked ROI")
                        continue
                    
                    # Step 4: Save results
                    image_filename = os.path.basename(image_path)
                    name_without_ext = os.path.splitext(image_filename)[0]
                    
                    # Save segmentation mask
                    mask_filename = f"{name_without_ext}_mask.png"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                    
                    # Save masked ROI (use consistent naming with existing ROI files)
                    roi_filename = f"{name_without_ext}_roi.jpg"
                    roi_path = os.path.join(roi_dir, roi_filename)
                    cv2.imwrite(roi_path, masked_roi)
                    
                    print(f"💾 Saved segmentation mask: {mask_filename}")
                    print(f"💾 Saved segmented ROI: {roi_filename}")
                    
                    # Record success
                    results['successful_extractions'] += 1
                    results['roi_data'].append({
                        'status': 'success',
                        'image_path': image_path,
                        'roi_path': roi_path,
                        'mask_path': mask_path,
                        'bbox': bbox,
                        'confidence': best_detection['confidence'],
                        'segmentation_score': float(score),
                        'roi_size': masked_roi.shape[:2]
                    })
                    
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {e}")
                    results['failed_images'].append({
                        'image_path': image_path,
                        'error': str(e)
                    })
            
            # Final summary
            success_rate = results['successful_extractions'] / results['total_images'] * 100
            print(f"\n✅ Segmentation-based ROI extraction completed:")
            print(f"   📊 Success rate: {success_rate:.1f}% ({results['successful_extractions']}/{results['total_images']})")
            print(f"   📁 ROI saved to: {roi_dir}")
            print(f"   🎭 Masks saved to: {masks_dir}")
            
            return results
            
        except Exception as e:
            print(f"❌ Segmentation-based ROI extraction failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _extract_masked_roi(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ROI using segmentation mask
        Returns image with background removed/transparent
        """
        try:
            # Ensure mask is binary
            if mask.dtype != bool:
                mask = mask > 0.5
            
            # Get bounding box of mask
            indices = np.where(mask)
            if len(indices[0]) == 0:
                return None
                
            y_min, y_max = np.min(indices[0]), np.max(indices[0])
            x_min, x_max = np.min(indices[1]), np.max(indices[1])
            
            # Add small padding
            padding = 10
            y_min = max(0, y_min - padding)
            y_max = min(image.shape[0], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            
            # Crop image and mask to bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            
            # Apply mask to remove background
            # Set background pixels to black (or could use transparency)
            masked_roi = cropped_image.copy()
            masked_roi[~cropped_mask] = [0, 0, 0]  # Black background
            
            return masked_roi
            
        except Exception as e:
            print(f"❌ Error extracting masked ROI: {e}")
            return None
    
    def set_device(self, device: str):
        """Set device for both GroundingDINO and SAM2 models"""
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        
        print(f"🔧 Setting device to {device} for both models...")
        
        # Set device for GroundingDINO
        self.grounding_dino.set_device(device)
        
        # SAM2 automatically detects device, but we can reload it to ensure consistency
        if hasattr(self.sam2, '_model_loaded') and self.sam2._model_loaded:
            self.sam2._model_loaded = False  # Force reload
            self.sam2._ensure_model_loaded()
    
    def detect_and_segment(
        self, 
        image_path: str, 
        prompts: List[str], 
        confidence_threshold: float = 0.3,
        use_sam2_segmentation: bool = True
    ) -> Dict:
        """
        Improved detection and segmentation pipeline:
        1. Use GroundingDINO to find the part/component of interest (ROI)
        2. Use SAM2 automatic mask generation within that ROI
        3. Filter SAM2 masks by size/shape heuristics to find defects
        
        Args:
            image_path: Path to the image
            prompts: List of text prompts for ROI detection (e.g., "metal plate", "component")
            confidence_threshold: Minimum confidence for ROI detection
            use_sam2_segmentation: Whether to use SAM2 for automatic segmentation
            
        Returns:
            Dictionary with ROI detections, defect segmentations, and annotated images
        """
        if self.grounding_dino.model is None:
            return {
                "status": "error",
                "message": "GroundingDINO model not available"
            }
        
        try:
            print(f"🎯 Starting improved GroundingDINO + SAM2 pipeline for: {os.path.basename(image_path)}")
            print(f"📝 ROI prompts: {prompts}")
            print(f"🎚️ Confidence threshold: {confidence_threshold}")
            print(f"🔍 SAM2 auto-segmentation: {'enabled' if use_sam2_segmentation else 'disabled'}")
            
            # Step 1: GroundingDINO ROI Detection
            print("🔍 Step 1: Finding ROI (part/component) with GroundingDINO...")
            grounding_result = self.grounding_dino.annotate_with_prompts(
                image_path, prompts, confidence_threshold
            )
            
            if grounding_result['status'] != 'success':
                return grounding_result
            
            roi_detections = grounding_result['detections']
            print(f"✅ GroundingDINO found {len(roi_detections)} ROI detections")
            
            if not roi_detections:
                return {
                    "status": "success",
                    "message": "No ROI detected",
                    "detections": [],
                    "segmentations": [],
                    "annotated_image_path": grounding_result['annotated_image_path'],
                    "image_base64": grounding_result['image_base64']
                }
            
            # Step 2: SAM2 Automatic Mask Generation within ROI
            defect_segmentations = []
            if use_sam2_segmentation:
                print("🎭 Step 2: Running SAM2 automatic mask generation within ROI...")
                defect_segmentations = self._generate_defect_masks_in_roi(image_path, roi_detections)
                print(f"✅ SAM2 found {len(defect_segmentations)} potential defects")
            
            # Step 3: Create combined visualization
            print("🎨 Step 3: Creating combined annotation...")
            combined_image_path, combined_base64 = self._create_roi_defect_annotation(
                image_path, roi_detections, defect_segmentations, prompts
            )
            
            # Step 4: Generate outputs
            result = {
                "status": "success",
                "image_path": image_path,
                "roi_prompts": prompts,
                "total_roi_detections": len(roi_detections),
                "total_defect_segmentations": len(defect_segmentations),
                "roi_detections": roi_detections,
                "defect_segmentations": defect_segmentations,
                "annotated_image_path": combined_image_path,
                "image_base64": combined_base64,
                "confidence_threshold": confidence_threshold,
                "sam2_enabled": use_sam2_segmentation,
                "device": self.grounding_dino.device,
                "pipeline_type": "roi_filter_defect_detection"
            }
            
            print(f"🎉 Pipeline completed successfully!")
            return result
            
        except Exception as e:
            print(f"❌ GroundingDINO + SAM2 pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Pipeline failed: {str(e)}"
            }
    
    def _generate_defect_masks_in_roi(self, image_path: str, roi_detections: List[Dict]) -> List[Dict]:
        """Generate defect masks using SAM2 automatic mask generation within ROI"""
        if not self.sam2._ensure_model_loaded():
            print("⚠️ SAM2 model not available, skipping segmentation")
            return []
        
        defect_segmentations = []
        
        # Load image for SAM2
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process each ROI separately
        for roi_idx, roi_detection in enumerate(roi_detections):
            try:
                # Extract ROI bounding box
                x1, y1, x2, y2 = [int(coord) for coord in roi_detection['bbox']]
                print(f"🎭 Processing ROI {roi_idx+1}: {roi_detection['class']} at [{x1}, {y1}, {x2}, {y2}]")
                
                # Crop the ROI from the image
                roi_image = image_rgb[y1:y2, x1:x2]
                
                if roi_image.size == 0:
                    print(f"⚠️ Invalid ROI dimensions, skipping")
                    continue
                
                # Set ROI image for SAM2 predictor
                self.sam2.predictor.set_image(roi_image)
                
                # Generate automatic masks within the ROI
                print(f"🔍 Generating automatic masks in ROI...")
                auto_masks = self._generate_automatic_masks_in_roi(roi_image)
                print(f"🎭 Generated {len(auto_masks)} raw masks")
                
                # Filter masks to find potential defects
                defect_masks = self._filter_defect_masks(auto_masks, roi_image.shape[:2])
                print(f"✅ Found {len(defect_masks)} potential defects after filtering")
                
                # Convert to global coordinates and add to results
                for defect_idx, (mask, score, area) in enumerate(defect_masks):
                    # Convert mask coordinates back to full image space
                    global_contours = self._convert_roi_contours_to_global(
                        self._extract_contours(mask), x1, y1
                    )
                    
                    defect_segmentation = {
                        "roi_index": roi_idx,
                        "defect_index": defect_idx,
                        "roi_class": roi_detection['class'],
                        "roi_bbox": roi_detection['bbox'],
                        "sam2_score": float(score),
                        "mask_area": int(area),
                        "contours": global_contours,
                        "defect_type": self._classify_defect_by_shape(mask, area),
                        "severity": self._assess_defect_severity(area, roi_image.shape[:2])
                    }
                    
                    defect_segmentations.append(defect_segmentation)
                    
            except Exception as e:
                print(f"❌ Error processing ROI {roi_idx}: {e}")
                continue
        
        return defect_segmentations
    
    def _generate_automatic_masks_in_roi(self, roi_image: np.ndarray) -> List[Tuple]:
        """Generate automatic masks using SAM2's everything mode within ROI"""
        try:
            # Use SAM2's automatic mask generation
            # Note: This is a simplified version - you might need to adjust based on SAM2 API
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            mask_generator = SAM2AutomaticMaskGenerator(self.sam2.model)
            masks = mask_generator.generate(roi_image)
            
            # Convert to our format: (mask, score, area)
            auto_masks = []
            for mask_data in masks:
                mask = mask_data['segmentation']
                score = mask_data.get('stability_score', 0.5)
                area = mask_data['area']
                auto_masks.append((mask, score, area))
            
            return auto_masks
            
        except ImportError:
            print("⚠️ SAM2AutomaticMaskGenerator not available, using fallback")
            return self._fallback_mask_generation(roi_image)
        except Exception as e:
            print(f"⚠️ Error in automatic mask generation: {e}, using fallback")
            return self._fallback_mask_generation(roi_image)
    
    def _fallback_mask_generation(self, roi_image: np.ndarray) -> List[Tuple]:
        """Fallback mask generation using grid-based sampling"""
        h, w = roi_image.shape[:2]
        masks = []
        
        # Generate masks using grid sampling
        grid_size = 32
        for y in range(0, h - grid_size, grid_size // 2):
            for x in range(0, w - grid_size, grid_size // 2):
                # Create a point prompt
                point = np.array([[x + grid_size // 2, y + grid_size // 2]])
                
                try:
                    mask, score, _ = self.sam2.predictor.predict(
                        point_coords=point,
                        point_labels=np.array([1]),
                        multimask_output=False,
                    )
                    
                    if mask is not None and len(mask) > 0:
                        mask_2d = mask[0]
                        area = np.sum(mask_2d)
                        if area > 50:  # Minimum area threshold
                            masks.append((mask_2d, float(score[0]), int(area)))
                            
                except Exception as e:
                    continue
        
        return masks
    
    def _filter_defect_masks(self, auto_masks: List[Tuple], roi_shape: Tuple[int, int]) -> List[Tuple]:
        """Filter masks to identify potential defects using size/shape heuristics"""
        defect_masks = []
        roi_area = roi_shape[0] * roi_shape[1]
        
        for mask, score, area in auto_masks:
            # Size filtering
            area_ratio = area / roi_area
            
            # Skip masks that are too large (likely background/main object)
            if area_ratio > 0.3:
                continue
                
            # Skip masks that are too small (likely noise)
            if area < 100:
                continue
            
            # Shape analysis
            irregularity = self._calculate_mask_irregularity(mask)
            compactness = self._calculate_mask_compactness(mask)
            
            # Defects tend to be irregular and less compact
            defect_score = irregularity * (1 - compactness) * score
            
            if defect_score > 0.3:  # Defect threshold
                defect_masks.append((mask, defect_score, area))
        
        # Sort by defect score and return top candidates
        defect_masks.sort(key=lambda x: x[1], reverse=True)
        return defect_masks[:10]  # Return top 10 candidates
    
    def _calculate_mask_irregularity(self, mask: np.ndarray) -> float:
        """Calculate irregularity of mask shape (higher = more irregular)"""
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if area == 0:
                return 0.0
            
            # Irregularity based on perimeter to area ratio
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return 1.0 - circularity  # Higher value = more irregular
            
        except:
            return 0.0
    
    def _calculate_mask_compactness(self, mask: np.ndarray) -> float:
        """Calculate compactness of mask (higher = more compact)"""
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area == 0:
                return 0.0
            
            return area / hull_area  # Compactness ratio
            
        except:
            return 0.0
    
    def _convert_roi_contours_to_global(self, roi_contours: List, offset_x: int, offset_y: int) -> List:
        """Convert ROI-relative contours to global image coordinates"""
        global_contours = []
        for contour in roi_contours:
            global_contour = []
            for point in contour:
                global_point = [point[0] + offset_x, point[1] + offset_y]
                global_contour.append(global_point)
            global_contours.append(global_contour)
        return global_contours
    
    def _classify_defect_by_shape(self, mask: np.ndarray, area: int) -> str:
        """Classify defect type based on shape characteristics"""
        irregularity = self._calculate_mask_irregularity(mask)
        compactness = self._calculate_mask_compactness(mask)
        
        if irregularity > 0.7:
            return "crack" if area < 1000 else "corrosion"
        elif compactness > 0.8:
            return "dent" if area > 500 else "spot"
        else:
            return "scratch"
    
    def _assess_defect_severity(self, area: int, roi_shape: Tuple[int, int]) -> str:
        """Assess defect severity based on size relative to ROI"""
        roi_area = roi_shape[0] * roi_shape[1]
        area_ratio = area / roi_area
        
        if area_ratio > 0.05:
            return "high"
        elif area_ratio > 0.01:
            return "medium"
        else:
            return "low"
    
    def _extract_contours(self, mask: np.ndarray) -> List[List[List[int]]]:
        """Extract contour points from mask for drawing"""
        try:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Convert contours to serializable format
            contour_points = []
            for contour in contours[:3]:  # Limit to 3 largest contours
                if len(contour) > 10:  # Filter small contours
                    points = contour.reshape(-1, 2).tolist()
                    contour_points.append(points[:100])  # Limit points per contour
            
            return contour_points
            
        except Exception as e:
            print(f"❌ Error extracting contours: {e}")
            return []
    
    def _create_roi_defect_annotation(
        self, 
        image_path: str, 
        roi_detections: List[Dict], 
        defect_segmentations: List[Dict],
        prompts: List[str]
    ) -> Tuple[str, str]:
        """Create annotated image showing ROI and detected defects"""
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        annotated_img = image.copy()
        
        # Define colors
        roi_color = (0, 255, 0)  # Green for ROI
        defect_colors = {
            'crack': (0, 0, 255),      # Red
            'corrosion': (0, 165, 255), # Orange
            'dent': (255, 0, 0),       # Blue
            'spot': (255, 255, 0),     # Cyan
            'scratch': (255, 0, 255),  # Magenta
            'default': (128, 128, 128) # Gray
        }
        
        # Step 1: Draw ROI bounding boxes
        for i, roi in enumerate(roi_detections):
            x1, y1, x2, y2 = [int(coord) for coord in roi['bbox']]
            
            # Draw ROI bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), roi_color, 3)
            
            # Draw ROI label
            roi_label = f"ROI: {roi['class']} ({roi['confidence']:.2f})"
            label_size = cv2.getTextSize(roi_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_img, 
                         (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), 
                         roi_color, -1)
            
            # Draw label text
            cv2.putText(annotated_img, roi_label, 
                       (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Step 2: Draw defect segmentations
        if defect_segmentations:
            overlay = annotated_img.copy()
            
            for i, defect in enumerate(defect_segmentations):
                defect_type = defect['defect_type']
                color = defect_colors.get(defect_type, defect_colors['default'])
                
                # Draw filled contours for defects
                for contour_points in defect['contours']:
                    if len(contour_points) > 2:
                        contour = np.array(contour_points, dtype=np.int32)
                        cv2.fillPoly(overlay, [contour], color)
                        cv2.polylines(annotated_img, [contour], True, color, 2)
                
                # Add defect label
                if defect['contours']:
                    # Find center of defect for label placement
                    contour = np.array(defect['contours'][0], dtype=np.int32)
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        defect_label = f"{defect_type.upper()}"
                        severity = defect['severity']
                        
                        # Severity indicator
                        severity_color = {
                            'high': (0, 0, 255),    # Red
                            'medium': (0, 165, 255), # Orange  
                            'low': (0, 255, 255)    # Yellow
                        }.get(severity, (255, 255, 255))
                        
                        cv2.putText(annotated_img, defect_label, 
                                   (cx - 20, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, severity_color, 2)
            
            # Blend defect overlay with image
            annotated_img = cv2.addWeighted(annotated_img, 0.7, overlay, 0.3, 0)
        
        # Step 3: Add summary information
        title = f"ROI Defect Detection: {len(roi_detections)} ROIs, {len(defect_segmentations)} defects found"
        cv2.putText(annotated_img, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(annotated_img, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add ROI prompts info
        prompts_text = f"ROI Search: {', '.join(prompts)}"
        cv2.putText(annotated_img, prompts_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated_img, prompts_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add defect summary
        if defect_segmentations:
            defect_types = {}
            for defect in defect_segmentations:
                defect_type = defect['defect_type']
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
            
            summary_text = f"Defects: {', '.join([f'{k}({v})' for k, v in defect_types.items()])}"
            cv2.putText(annotated_img, summary_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(annotated_img, summary_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save result
        output_filename = f"roi_defect_detection_{Path(image_path).stem}_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(self.results_dir, output_filename)
        cv2.imwrite(output_path, annotated_img)
        
        # Convert to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return output_path, image_base64
    
    def batch_detect_and_segment(
        self, 
        image_paths: List[str], 
        prompts: List[str], 
        confidence_threshold: float = 0.3,
        use_sam2_segmentation: bool = True
    ) -> Dict:
        """Run the integrated pipeline on multiple images"""
        results = []
        
        for image_path in image_paths:
            print(f"📸 Processing {os.path.basename(image_path)}...")
            result = self.detect_and_segment(
                image_path, prompts, confidence_threshold, use_sam2_segmentation
            )
            result['filename'] = os.path.basename(image_path)
            results.append(result)
        
        # Generate batch summary
        successful_results = [r for r in results if r['status'] == 'success']
        total_detections = sum(r.get('total_detections', 0) for r in successful_results)
        total_segmentations = sum(r.get('total_segmentations', 0) for r in successful_results)
        
        summary = {
            "total_images": len(image_paths),
            "successful_annotations": len(successful_results),
            "failed_annotations": len(results) - len(successful_results),
            "total_detections": total_detections,
            "total_segmentations": total_segmentations,
            "average_detections_per_image": total_detections / len(successful_results) if successful_results else 0,
            "average_segmentations_per_image": total_segmentations / len(successful_results) if successful_results else 0
        }
        
        return {
            "status": "success",
            "results": results,
            "summary": summary,
            "prompts_used": prompts,
            "confidence_threshold": confidence_threshold,
            "sam2_enabled": use_sam2_segmentation
        }


# Create global instance
grounding_dino_sam2_service = GroundingDINOSAM2Service()