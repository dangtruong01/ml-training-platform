import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
import requests
from urllib.parse import urlparse

class SAM2Service:
    def __init__(self):
        self.model = None
        self.predictor = None
        self._model_loaded = False
        
    def _ensure_model_loaded(self):
        """Load SAM2 model if not already loaded"""
        if self._model_loaded:
            return True
            
        try:
            # Try to import SAM2 dependencies
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Model configuration - you can switch between different SAM2 models
            model_configs = {
                'tiny': {
                    'config': 'sam2_hiera_t.yaml',
                    'checkpoint': 'sam2_hiera_tiny.pt',
                    'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt'
                },
                'small': {
                    'config': 'sam2_hiera_s.yaml', 
                    'checkpoint': 'sam2_hiera_small.pt',
                    'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt'
                },
                'large': {
                    'config': 'sam2_hiera_l.yaml',
                    'checkpoint': 'sam2_hiera_large.pt', 
                    'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
                }
            }
            
            # Use tiny model for CPU (much faster, still good accuracy)
            model_info = model_configs['tiny']
            
            # Ensure model directory exists
            model_dir = os.path.abspath("ml/models/sam2")
            os.makedirs(model_dir, exist_ok=True)
            
            # Download model if it doesn't exist
            checkpoint_path = os.path.join(model_dir, model_info['checkpoint'])
            if not os.path.exists(checkpoint_path):
                print(f"Downloading SAM2 model: {model_info['checkpoint']}")
                self._download_model(model_info['url'], checkpoint_path)
            
            # Auto-detect best device (GPU if available, else CPU)
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print("🚀 Using GPU acceleration (CUDA)")
            else:
                device = 'cpu'
                print("⚠️  Using CPU (slow) - consider GPU for better performance")
            
            # Load the model - use relative config path for SAM2
            try:
                # Try with just the config name (SAM2 should find it automatically)
                config_path = model_info['config']
                print(f"Loading SAM2 with config: {config_path} on {device}")
                self.model = build_sam2(config_path, checkpoint_path, device=device)
            except Exception as e:
                print(f"Failed to load with config '{config_path}': {e}")
                # Fallback: try without .yaml extension
                config_name = model_info['config'].replace('.yaml', '')
                print(f"Trying fallback config: {config_name}")
                self.model = build_sam2(config_name, checkpoint_path, device=device)
            self.predictor = SAM2ImagePredictor(self.model)
            
            self._model_loaded = True
            print("SAM2 model loaded successfully")
            return True
            
        except ImportError as e:
            print(f"SAM2 not installed: {e}")
            print("Please install SAM2: pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'")
            return False
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            return False
    
    def _download_model(self, url: str, filepath: str):
        """Download model file from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Downloaded {percent:.1f}%", end='\r')
            
            print(f"\nModel downloaded successfully: {filepath}")
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            raise
    
    def annotate_detection(self, image_path: str) -> str:
        """
        SAM2-based object detection with bounding boxes
        Returns path to annotated image
        """
        if not self._ensure_model_loaded():
            raise RuntimeError("SAM2 model could not be loaded")
        
        return self._sam2_detection_annotation(image_path)
    
    def annotate_segmentation(self, image_path: str) -> str:
        """
        SAM2-based segmentation with precise masks
        Returns path to annotated image
        """
        if not self._ensure_model_loaded():
            raise RuntimeError("SAM2 model could not be loaded")
        
        return self._sam2_segmentation_annotation(image_path)
    
    def _sam2_detection_annotation(self, image_path: str) -> str:
        """SAM2-based detection with bounding boxes"""
        print(f"Starting SAM2 detection for: {image_path}")
        results_dir = os.path.abspath(os.path.join("ml", "results", "pre_annotation"))
        os.makedirs(results_dir, exist_ok=True)
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        print(f"Image loaded: {image.shape}")
        
        # Resize large images for CPU processing (major speed improvement)
        import torch
        original_image = image.copy()
        if not torch.cuda.is_available():
            h, w = image.shape[:2]
            max_size = 800  # Limit image size for CPU
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                print(f"Resized for CPU: {original_image.shape} -> {image.shape}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_img = original_image.copy()  # Use original size for annotation
        
        # Set image for SAM2
        print("Setting image for SAM2 predictor...")
        self.predictor.set_image(image_rgb)
        
        # Generate automatic masks for all objects
        print("Generating automatic masks...")
        masks, scores, logits = self._generate_automatic_masks(image_rgb)
        print(f"Raw masks generated: {len(masks)}")
        
        # Filter and draw bounding boxes
        valid_masks = self._filter_masks_by_quality(masks, scores, image.shape[:2])
        print(f"Valid masks after filtering: {len(valid_masks)}")
        
        # Find the main object (single most significant bounding box)
        if valid_masks:
            main_object = self._find_main_object(valid_masks, image.shape[:2])
            valid_masks = main_object  # Keep only the main object
        
        # Scale masks back to original image size if we resized
        import torch
        if not torch.cuda.is_available() and image.shape != original_image.shape:
            scale_x = original_image.shape[1] / image.shape[1]
            scale_y = original_image.shape[0] / image.shape[0]
            print(f"Scaling masks back: {scale_x:.2f}x, {scale_y:.2f}x")
            
            scaled_masks = []
            for mask, score in valid_masks:
                # Resize mask to original dimensions
                scaled_mask = cv2.resize(mask.astype(np.uint8), 
                                       (original_image.shape[1], original_image.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
                scaled_masks.append((scaled_mask.astype(bool), score))
            valid_masks = scaled_masks
        
        if len(valid_masks) == 0:
            print("No valid masks found, creating fallback annotation...")
            # Add a fallback message on the image
            cv2.putText(annotated_img, "No objects detected by SAM2", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Draw the single main object with enhanced styling
            for i, (mask, score) in enumerate(valid_masks[:1]):  # Only 1 main object
                # Get bounding box from mask
                bbox = self._mask_to_bbox(mask)
                x, y, w, h = bbox
                
                print(f"Drawing main object bbox: ({x},{y},{w},{h}) with score {score:.3f}")
                
                # Use green for the main object (success/primary color)
                main_color = (0, 255, 0)  # Bright green
                
                # Draw thicker bounding rectangle for main object
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), main_color, 4)
                
                # Add a secondary border for emphasis
                cv2.rectangle(annotated_img, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 255), 2)
                
                # Add confidence score and enhanced label
                confidence = f"{score:.2f}"
                label = f"Main Object ({confidence})"
                
                # Calculate text size for better background
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Position text above the box if possible
                if y - text_h - 15 > 0:
                    text_y = y - 10
                    bg_y1, bg_y2 = y - text_h - 20, y - 5
                else:
                    text_y = y + h + text_h + 15
                    bg_y1, bg_y2 = y + h + 5, y + h + text_h + 20
                
                # White background with green border for text
                cv2.rectangle(annotated_img, (x, bg_y1), (x + text_w + 15, bg_y2), (255, 255, 255), -1)
                cv2.rectangle(annotated_img, (x, bg_y1), (x + text_w + 15, bg_y2), main_color, 2)
                
                # Black text for better readability
                cv2.putText(annotated_img, label, (x + 7, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add summary
        if len(valid_masks) > 0:
            summary = "SAM2 Detection: Main object found"
        else:
            summary = "SAM2 Detection: No main object detected"
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Save result
        output_path = os.path.join(results_dir, f"sam2_detection_{os.path.basename(image_path)}")
        success = cv2.imwrite(output_path, annotated_img)
        print(f"Result saved to: {output_path}, success: {success}")
        return output_path
    
    def _sam2_segmentation_annotation(self, image_path: str) -> str:
        """SAM2-based segmentation with precise masks"""
        print(f"Starting SAM2 segmentation for: {image_path}")
        
        # Check if CPU and warn about performance
        import torch
        if not torch.cuda.is_available():
            print("⚠️  WARNING: SAM2 segmentation on CPU is very slow (30-60+ seconds)")
            print("Consider using SAM2 detection instead, or enable GPU acceleration")
        
        results_dir = os.path.abspath(os.path.join("ml", "results", "pre_annotation"))
        os.makedirs(results_dir, exist_ok=True)
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        print(f"Image loaded: {image.shape}")
        
        # Apply same CPU optimizations as detection
        original_image = image.copy()
        if not torch.cuda.is_available():
            h, w = image.shape[:2]
            max_size = 600  # Even smaller for segmentation (more intensive)
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                print(f"Resized for CPU segmentation: {original_image.shape} -> {image.shape}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_img = original_image.copy()  # Use original size for annotation
        
        # Set image for SAM2
        print("Setting image for SAM2 predictor...")
        self.predictor.set_image(image_rgb)
        
        # Generate automatic masks (with reduced complexity for segmentation)
        print("Generating automatic masks for segmentation...")
        masks, scores, logits = self._generate_automatic_masks_segmentation(image_rgb)
        
        # Filter masks by quality
        valid_masks = self._filter_masks_by_quality(masks, scores, image.shape[:2])
        print(f"Valid segmentation masks after filtering: {len(valid_masks)}")
        
        # Scale masks back to original image size if we resized
        import torch
        if not torch.cuda.is_available() and image.shape != original_image.shape:
            scale_x = original_image.shape[1] / image.shape[1]
            scale_y = original_image.shape[0] / image.shape[0]
            print(f"Scaling segmentation masks back: {scale_x:.2f}x, {scale_y:.2f}x")
            
            scaled_masks = []
            for mask, score in valid_masks:
                # Resize mask to original dimensions
                scaled_mask = cv2.resize(mask.astype(np.uint8), 
                                       (original_image.shape[1], original_image.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
                scaled_masks.append((scaled_mask.astype(bool), score))
            valid_masks = scaled_masks
        
        if len(valid_masks) == 0:
            print("No valid segmentation masks found, creating fallback annotation...")
            cv2.putText(annotated_img, "No segments detected by SAM2", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Draw segmentation masks
            for i, (mask, score) in enumerate(valid_masks[:3]):  # Limit to top 3 for clarity
                color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i % 3]
                
                print(f"Drawing segmentation mask {i}: score={score:.3f}")
                
                # Create colored overlay for the mask
                colored_mask = np.zeros_like(annotated_img)
                colored_mask[mask] = color
                
                # Blend with original image
                annotated_img = cv2.addWeighted(annotated_img, 0.7, colored_mask, 0.3, 0)
                
                # Draw mask contours
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_img, contours, -1, color, 2)
                
                # Add label
                bbox = self._mask_to_bbox(mask)
                x, y, w, h = bbox
                confidence = f"{score:.2f}"
                label = f"SAM2-Seg{i+1} ({confidence})"
                cv2.putText(annotated_img, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add summary
        summary = f"SAM2 Segmentation: {len(valid_masks)} segments found"
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Save result
        output_path = os.path.join(results_dir, f"sam2_segmentation_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        return output_path
    
    def _generate_automatic_masks(self, image_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """Generate automatic masks using SAM2 with multiple point prompts"""
        try:
            h, w = image_rgb.shape[:2]
            all_masks = []
            all_scores = []
            all_logits = []
            
            # Strategy 1: Grid-based point prompting  
            # Use larger grid (fewer points) for CPU to speed up processing
            import torch
            if torch.cuda.is_available():
                grid_size = min(64, max(h, w) // 8)  # Dense grid for GPU
            else:
                grid_size = min(128, max(h, w) // 4)  # Sparse grid for CPU (2-4x faster)
            
            for y in range(grid_size//2, h, grid_size):
                for x in range(grid_size//2, w, grid_size):
                    try:
                        # Single point prompt
                        input_points = np.array([[x, y]])
                        input_labels = np.array([1])  # Positive prompt
                        
                        masks, scores, logits = self.predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True,
                        )
                        
                        # Add all returned masks
                        for i in range(len(masks)):
                            if scores[i] > 0.5:  # Filter by confidence
                                all_masks.append(masks[i])
                                all_scores.append(scores[i])
                                all_logits.append(logits[i])
                        
                    except Exception as e:
                        print(f"Error with point ({x}, {y}): {e}")
                        continue
            
            # Strategy 2: If no good masks found, try center points with different strategies
            if len(all_masks) == 0:
                print("No masks from grid, trying center point strategies...")
                
                center_strategies = [
                    ([w//2, h//2], [1]),  # Center positive
                    ([w//3, h//3], [1]),  # Upper left
                    ([2*w//3, h//3], [1]),  # Upper right
                    ([w//3, 2*h//3], [1]),  # Lower left
                    ([2*w//3, 2*h//3], [1]),  # Lower right
                ]
                
                for point, label in center_strategies:
                    try:
                        input_points = np.array([point])
                        input_labels = np.array(label)
                        
                        masks, scores, logits = self.predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True,
                        )
                        
                        for i in range(len(masks)):
                            if scores[i] > 0.3:  # Lower threshold for center points
                                all_masks.append(masks[i])
                                all_scores.append(scores[i])
                                all_logits.append(logits[i])
                                
                    except Exception as e:
                        print(f"Error with center strategy {point}: {e}")
                        continue
            
            print(f"Generated {len(all_masks)} masks from SAM2")
            return all_masks, all_scores, all_logits
            
        except Exception as e:
            print(f"Error in automatic mask generation: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
    
    def _filter_masks_by_quality(self, masks: List[np.ndarray], scores: List[float], image_shape: Tuple[int, int]) -> List[Tuple[np.ndarray, float]]:
        """Filter masks based on quality metrics"""
        if not masks or not scores:
            print("No masks or scores to filter")
            return []
        
        print(f"Filtering {len(masks)} masks...")
        img_area = image_shape[0] * image_shape[1]
        valid_masks = []
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_array = np.array(mask)
            
            # Calculate mask area
            mask_area = np.sum(mask_array)
            area_ratio = mask_area / img_area
            
            print(f"Mask {i}: score={score:.3f}, area={mask_area}, ratio={area_ratio:.3f}")
            
            # More lenient filtering for debugging
            min_area = 500  # Reduced from 1000
            max_area_ratio = 0.8  # Allow larger masks
            min_score = 0.3  # Lower score threshold
            
            # Filter by area (not too small, not too large)
            if mask_area < min_area:
                print(f"  Rejected: area too small ({mask_area} < {min_area})")
                continue
                
            if area_ratio > max_area_ratio:
                print(f"  Rejected: area too large ({area_ratio:.3f} > {max_area_ratio})")
                continue
            
            # Filter by score threshold
            if score < min_score:
                print(f"  Rejected: score too low ({score:.3f} < {min_score})")
                continue
            
            print(f"  Accepted: mask {i}")
            valid_masks.append((mask_array, score))
        
        # Sort by score (highest first)
        valid_masks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Final: {len(valid_masks)} valid masks after filtering")
        return valid_masks
    
    def _find_main_object(self, valid_masks: List[Tuple[np.ndarray, float]], image_shape: Tuple[int, int]) -> List[Tuple[np.ndarray, float]]:
        """Find the single most significant object based on area and score"""
        if not valid_masks:
            return []
        
        print(f"Finding main object from {len(valid_masks)} candidates...")
        img_area = image_shape[0] * image_shape[1]
        
        # Calculate composite scores combining area and confidence
        candidates = []
        for i, (mask, score) in enumerate(valid_masks):
            mask_area = np.sum(mask)
            area_ratio = mask_area / img_area
            
            # Composite score: balance between confidence score and area significance
            # Prefer larger objects (but not too large) with good confidence
            area_score = min(area_ratio * 2.0, 1.0)  # Cap at 1.0, favor larger objects
            composite_score = (score * 0.6) + (area_score * 0.4)  # 60% confidence, 40% area
            
            bbox = self._mask_to_bbox(mask)
            x, y, w, h = bbox
            
            candidates.append({
                'index': i,
                'mask': mask,
                'score': score,
                'area': mask_area,
                'area_ratio': area_ratio,
                'composite_score': composite_score,
                'bbox': bbox,
                'bbox_area': w * h
            })
            
            print(f"Candidate {i}: conf={score:.3f}, area_ratio={area_ratio:.3f}, composite={composite_score:.3f}")
        
        # Sort by composite score (highest first)
        candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Additional filtering for the main object
        best_candidate = candidates[0]
        
        # Check if the best candidate is significantly better than others
        if len(candidates) > 1:
            second_best = candidates[1]
            score_gap = best_candidate['composite_score'] - second_best['composite_score']
            print(f"Best candidate score gap: {score_gap:.3f}")
            
            # If the gap is small, prefer the larger object
            if score_gap < 0.1 and best_candidate['area'] < second_best['area'] * 1.5:
                print("Choosing second candidate due to significantly larger area")
                best_candidate = second_best
        
        print(f"Selected main object: index={best_candidate['index']}, "
              f"conf={best_candidate['score']:.3f}, "
              f"area={best_candidate['area']}, "
              f"composite={best_candidate['composite_score']:.3f}")
        
        return [(best_candidate['mask'], best_candidate['score'])]
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert mask to bounding box (x, y, w, h)"""
        indices = np.where(mask)
        if len(indices[0]) == 0:
            return (0, 0, 0, 0)
        
        y_min, y_max = indices[0].min(), indices[0].max()
        x_min, x_max = indices[1].min(), indices[1].max()
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _generate_automatic_masks_segmentation(self, image_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """Generate masks for segmentation with reduced complexity for CPU performance"""
        try:
            h, w = image_rgb.shape[:2]
            all_masks = []
            all_scores = []
            all_logits = []
            
            # For segmentation on CPU, use much sparser grid to avoid stalling
            import torch
            if torch.cuda.is_available():
                # GPU: Use moderate grid
                grid_size = min(96, max(h, w) // 6)
                max_points = 25
            else:
                # CPU: Use very sparse grid (only 4-9 points total)
                grid_size = min(max(h, w) // 2, 200)  # Much larger spacing
                max_points = 6  # Very limited for CPU
            
            print(f"Using segmentation grid size: {grid_size}, max points: {max_points}")
            
            point_count = 0
            
            for y in range(grid_size//2, h, grid_size):
                for x in range(grid_size//2, w, grid_size):
                    if point_count >= max_points:
                        print(f"Reached maximum points limit ({max_points}) for CPU performance")
                        break
                        
                    try:
                        print(f"Processing segmentation point {point_count + 1}/{max_points}: ({x}, {y})")
                        
                        # Single point prompt
                        input_points = np.array([[x, y]])
                        input_labels = np.array([1])
                        
                        masks, scores, logits = self.predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True,
                        )
                        
                        # Add best mask only
                        if len(masks) > 0:
                            best_idx = np.argmax(scores)
                            if scores[best_idx] > 0.4:  # Lower threshold for segmentation
                                all_masks.append(masks[best_idx])
                                all_scores.append(scores[best_idx])
                                all_logits.append(logits[best_idx])
                        
                        point_count += 1
                        
                    except Exception as e:
                        print(f"Error with segmentation point ({x}, {y}): {e}")
                        continue
                
                if point_count >= max_points:
                    break
            
            print(f"Generated {len(all_masks)} segmentation masks from {point_count} points")
            return all_masks, all_scores, all_logits
            
        except Exception as e:
            print(f"Error in segmentation mask generation: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []

    def segment_with_box(self, image: np.ndarray, bbox: List[int]) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """
        Generate a segmentation mask for a given bounding box.
        
        Args:
            image: The input image in BGR format.
            bbox: A list of four integers [x1, y1, x2, y2].
            
        Returns:
            A tuple of (mask, score, logits).
        """
        if not self._ensure_model_loaded():
            raise RuntimeError("SAM2 model could not be loaded")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        input_box = np.array(bbox)
        
        masks, scores, logits = self.predictor.predict(
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # The predictor returns a batch of masks, even for a single box.
        # We take the first one.
        mask = masks[0]
        score = scores[0]
        logit = logits[0]

        return mask, score, logit

# Create global instance
sam2_service = SAM2Service()