import os
import cv2
import numpy as np
import json
import uuid
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import base64

class GroundingDINOService:
    def __init__(self):
        self.results_dir = os.path.abspath(os.path.join("ml", "results", "grounding_dino"))
        os.makedirs(self.results_dir, exist_ok=True)
        self.model = None
        self.device = self._get_best_device()
        self._load_model()
    
    def _get_best_device(self):
        """Auto-detect best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def set_device(self, device: str):
        """Set device for GroundingDINO model (cpu/cuda)"""
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
            
        self.device = device
        
        # Reload model if it's already loaded
        if self.model is not None:
            print(f"🔄 Reloading GroundingDINO model on {device}...")
            self._load_model()
    
    def _load_model(self):
        """Load GroundingDINO model with flexible path detection"""
        try:
            print("🔍 Setting up GroundingDINO with local installation...")
            
            # Add the local GroundingDINO to Python path
            import sys
            grounding_dino_path = os.path.abspath("GroundingDINO")
            if grounding_dino_path not in sys.path:
                sys.path.insert(0, grounding_dino_path)
                print(f"✅ Added to Python path: {grounding_dino_path}")
            
            # Try to import and load GroundingDINO from local installation
            from groundingdino.util.inference import load_model, load_image, predict
            from groundingdino.util.slconfig import SLConfig
            
            print("✅ Successfully imported GroundingDINO from local installation")
            
            # Use the GroundingDINO config
            config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            
            # Check if weights directory exists, create if not
            weights_dir = "GroundingDINO/weights"
            os.makedirs(weights_dir, exist_ok=True)
            checkpoint_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
            
            print(f"🔍 Using GroundingDINO model files:")
            print(f"   Config: {config_path}")
            print(f"   Checkpoint: {checkpoint_path}")
            
            # Verify config exists
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            # Download checkpoint if not exists
            if not os.path.exists(checkpoint_path):
                print(f"📥 Downloading model checkpoint to {checkpoint_path}...")
                self._download_model_checkpoint(checkpoint_path)
            
            print(f"🚀 Loading model:")
            print(f"   Config: {config_path}")
            print(f"   Checkpoint: {checkpoint_path}")
            
            # Load the model
            print(f"🔌 Using device: {self.device}")
            self.model = load_model(config_path, checkpoint_path, device=self.device)
            self.predict_fn = predict
            self.load_image_fn = load_image
            
            print("✅ GroundingDINO model loaded successfully!")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("⚠️ Make sure GroundingDINO is properly installed in the local directory")
            print("   Expected structure: GroundingDINO/groundingdino/")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading GroundingDINO: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def _ensure_config_file(self, model_dir: str) -> str:
        """Download just the config file if missing"""
        config_path = os.path.join(model_dir, "GroundingDINO_SwinT_OGC.py")
        
        if not os.path.exists(config_path):
            print("📥 Downloading GroundingDINO config file...")
            try:
                import requests
                config_url = "https://github.com/IDEA-Research/GroundingDINO/raw/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
                
                response = requests.get(config_url)
                response.raise_for_status()
                
                with open(config_path, 'w') as f:
                    f.write(response.text)
                
                print(f"✅ Downloaded config to: {config_path}")
                
            except Exception as e:
                print(f"❌ Failed to download config: {e}")
                raise
        
        return config_path
    
    def _download_model_checkpoint(self, checkpoint_path: str):
        """Download just the GroundingDINO model checkpoint"""
        import requests
        
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        
        try:
            print(f"📥 Downloading model from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(checkpoint_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"⏳ Download progress: {percent:.1f}%", end='\r')
            
            print(f"\n✅ Downloaded model checkpoint: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Failed to download model checkpoint: {e}")
            raise
    
    def annotate_with_prompts(self, image_path: str, prompts: List[str], confidence_threshold: float = 0.3) -> Dict:
        """
        Annotate image using GroundingDINO with text prompts
        
        Args:
            image_path: Path to the image
            prompts: List of text prompts (e.g., ["scratch", "dent", "dirt"])
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with detections and annotated image
        """
        if self.model is None:
            return {
                "status": "error",
                "message": "GroundingDINO model not available"
            }
        
        try:
            # Load image
            image_source, image = self.load_image_fn(image_path)
            
            # Combine prompts into a single string
            text_prompt = ". ".join(prompts) + "."
            
            print(f"🔍 Analyzing image with prompts: {text_prompt}")
            
            # Run detection
            print(f"🔌 Running inference on device: {self.device}")
            boxes, logits, phrases = self.predict_fn(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=confidence_threshold,
                text_threshold=0.25,
                device=self.device
            )
            
            # Process detections
            detections = self._process_detections(
                boxes, logits, phrases, image_source.shape, prompts
            )
            
            # Create annotated image
            annotated_image_path, image_base64 = self._create_annotated_image(
                image_source, detections, image_path
            )
            
            # Generate YOLO format annotations
            yolo_annotations = self._convert_to_yolo_format(detections, image_source.shape)
            
            result = {
                "status": "success",
                "image_path": image_path,
                "prompts": prompts,
                "total_detections": len(detections),
                "detections": detections,
                "annotated_image_path": annotated_image_path,
                "image_base64": image_base64,
                "yolo_annotations": yolo_annotations,
                "confidence_threshold": confidence_threshold
            }
            
            print(f"✅ Found {len(detections)} detections")
            return result
            
        except Exception as e:
            print(f"❌ GroundingDINO annotation failed: {e}")
            return {
                "status": "error",
                "message": f"Annotation failed: {str(e)}"
            }
    
    def _process_detections(self, boxes, logits, phrases, image_shape, original_prompts) -> List[Dict]:
        """Process raw GroundingDINO output into structured detections"""
        detections = []
        
        # Convert normalized coordinates to pixel coordinates
        H, W = image_shape[:2]
        
        for box, confidence, phrase in zip(boxes, logits, phrases):
            # GroundingDINO returns normalized coordinates (0-1)
            x_center, y_center, width, height = box
            
            # Convert to pixel coordinates
            x_center_px = int(x_center * W)
            y_center_px = int(y_center * H)
            width_px = int(width * W)
            height_px = int(height * H)
            
            # Convert to top-left corner coordinates
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            
            # Map phrase to original prompt classes
            detected_class = self._map_phrase_to_class(phrase, original_prompts)
            
            detection = {
                "class": detected_class,
                "confidence": float(confidence),
                "bbox": [x1, y1, x2, y2],
                "bbox_normalized": [float(x_center), float(y_center), float(width), float(height)],
                "phrase": phrase,
                "area": (x2 - x1) * (y2 - y1)
            }
            
            detections.append(detection)
        
        return detections
    
    def _map_phrase_to_class(self, phrase: str, original_prompts: List[str]) -> str:
        """Map detected phrase back to original prompt classes"""
        phrase_lower = phrase.lower()
        
        # Try exact match first
        for prompt in original_prompts:
            if prompt.lower() in phrase_lower or phrase_lower in prompt.lower():
                return prompt
        
        # Fallback to similarity matching
        best_match = original_prompts[0] if original_prompts else "defect"
        max_similarity = 0
        
        for prompt in original_prompts:
            similarity = len(set(phrase_lower.split()) & set(prompt.lower().split()))
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = prompt
        
        return best_match
    
    def _create_annotated_image(self, image, detections, original_path) -> Tuple[str, str]:
        """Create annotated image with bounding boxes"""
        annotated_img = image.copy()
        
        # Define colors for different classes
        colors = {
            'scratch': (0, 0, 255),    # Red
            'dent': (255, 0, 0),       # Blue  
            'dirt': (0, 255, 0),       # Green
            'corrosion': (0, 165, 255), # Orange
            'crack': (255, 0, 255),    # Magenta
            'stain': (255, 255, 0),    # Cyan
            'defect': (128, 128, 128)  # Gray (default)
        }
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            # Get color for this class
            color = colors.get(class_name.lower(), colors['defect'])
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_img, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_img, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save annotated image
        output_filename = f"grounding_dino_{Path(original_path).stem}_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(self.results_dir, output_filename)
        cv2.imwrite(output_path, annotated_img)
        
        # Convert to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return output_path, image_base64
    
    def _convert_to_yolo_format(self, detections, image_shape) -> Dict:
        """Convert detections to YOLO format"""
        H, W = image_shape[:2]
        
        # Create class mapping
        classes = list(set(det['class'] for det in detections))
        class_to_id = {cls: idx for idx, cls in enumerate(classes)}
        
        yolo_lines = []
        for detection in detections:
            class_id = class_to_id[detection['class']]
            x_center, y_center, width, height = detection['bbox_normalized']
            
            # YOLO format: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        return {
            "classes": classes,
            "class_mapping": class_to_id,
            "annotations": yolo_lines,
            "data_yaml": {
                "nc": len(classes),
                "names": classes
            }
        }
    
    def batch_annotate(self, image_paths: List[str], prompts: List[str], confidence_threshold: float = 0.3) -> Dict:
        """Annotate multiple images with the same prompts"""
        results = []
        
        for image_path in image_paths:
            print(f"📸 Processing {os.path.basename(image_path)}...")
            result = self.annotate_with_prompts(image_path, prompts, confidence_threshold)
            result['filename'] = os.path.basename(image_path)
            results.append(result)
        
        # Generate batch summary
        successful_results = [r for r in results if r['status'] == 'success']
        total_detections = sum(r['total_detections'] for r in successful_results)
        
        summary = {
            "total_images": len(image_paths),
            "successful_annotations": len(successful_results),
            "failed_annotations": len(results) - len(successful_results),
            "total_detections": total_detections,
            "average_detections_per_image": total_detections / len(successful_results) if successful_results else 0
        }
        
        return {
            "status": "success",
            "results": results,
            "summary": summary,
            "prompts_used": prompts,
            "confidence_threshold": confidence_threshold
        }
    
    def export_yolo_dataset(self, results: List[Dict], output_dir: str) -> str:
        """Export annotated results as a YOLO dataset"""
        dataset_dir = os.path.join(output_dir, f"grounding_dino_dataset_{uuid.uuid4().hex[:8]}")
        
        # Create YOLO dataset structure
        os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)
        
        # Collect all unique classes
        all_classes = set()
        for result in results:
            if result['status'] == 'success':
                all_classes.update(result['yolo_annotations']['classes'])
        
        all_classes = sorted(list(all_classes))
        
        # Split into train/val (80/20)
        train_count = int(len(results) * 0.8)
        
        for idx, result in enumerate(results):
            if result['status'] != 'success':
                continue
            
            filename = Path(result['image_path']).name
            is_train = idx < train_count
            split = "train" if is_train else "val"
            
            # Copy image
            image_dst = os.path.join(dataset_dir, "images", split, filename)
            import shutil
            shutil.copy2(result['image_path'], image_dst)
            
            # Create label file
            label_filename = Path(filename).stem + ".txt"
            label_dst = os.path.join(dataset_dir, "labels", split, label_filename)
            
            # Convert class names to indices based on global class list
            with open(label_dst, 'w') as f:
                for detection in result['detections']:
                    class_idx = all_classes.index(detection['class'])
                    x_center, y_center, width, height = detection['bbox_normalized']
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Create data.yaml
        data_yaml = {
            "path": dataset_dir,
            "train": "images/train",
            "val": "images/val",
            "nc": len(all_classes),
            "names": all_classes
        }
        
        import yaml
        with open(os.path.join(dataset_dir, "data.yaml"), 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✅ YOLO dataset exported to: {dataset_dir}")
        return dataset_dir

    def extract_manufacturing_roi(self, image_paths: List[str], project_id: str, 
                                manufacturing_scenario: str = "general", 
                                part_material: str = "metal", 
                                fixture_type: str = "tray", 
                                fixture_color: str = "blue",
                                image_type: str = "training") -> Dict:
        """
        Extract ROIs using manufacturing-specific segmentation
        
        Args:
            image_paths: List of image file paths
            project_id: Project ID for organizing results
            manufacturing_scenario: Manufacturing use case
            part_material: Material of the parts
            fixture_type: Type of fixture/background
            fixture_color: Color of fixture/background
            image_type: Type of images ("training" or "defective")
            
        Returns:
            Dict with ROI extraction results
        """
        try:
            print(f"🏭 Extracting manufacturing ROIs from {len(image_paths)} images")
            print(f"📋 Scenario: {manufacturing_scenario}, Material: {part_material}, Fixture: {fixture_type} ({fixture_color})")
            
            # Create project directories
            project_dir = os.path.join("ml", "auto_annotation", "projects", project_id)
            if image_type == "training":
                roi_dir = os.path.join(project_dir, "roi_cache")
                masks_dir = os.path.join(project_dir, "manufacturing_masks_training")
            else:  # defective
                roi_dir = os.path.join(project_dir, "defective_roi_cache")
                masks_dir = os.path.join(project_dir, "manufacturing_masks_defective")
            os.makedirs(roi_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            roi_results = []
            failed_images = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    print(f"🔍 Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Could not load image: {image_path}")
                    
                    # Create part mask using manufacturing segmentation
                    part_mask = self._create_manufacturing_part_mask(
                        image, manufacturing_scenario, part_material, fixture_type, fixture_color
                    )
                    
                    # Extract ROI bounding box from part mask
                    roi_bbox = self._extract_roi_from_mask(part_mask)
                    
                    if roi_bbox is None:
                        print(f"⚠️ No valid ROI found for {os.path.basename(image_path)}")
                        failed_images.append({
                            'image_path': image_path,
                            'error': 'No valid part region detected'
                        })
                        continue
                    
                    # Extract ROI from original image
                    roi_image = self._crop_image_to_roi(image, roi_bbox)
                    
                    # Save ROI image
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    roi_filename = f"{image_name}_roi.jpg"
                    roi_path = os.path.join(roi_dir, roi_filename)
                    cv2.imwrite(roi_path, roi_image)
                    
                    # Save part mask for debugging
                    mask_filename = f"{image_name}_part_mask.png"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    cv2.imwrite(mask_path, part_mask)
                    
                    # Calculate ROI info
                    x_min, y_min, x_max, y_max = roi_bbox
                    roi_width = x_max - x_min
                    roi_height = y_max - y_min
                    
                    roi_result = {
                        'image_path': image_path,
                        'image_name': os.path.basename(image_path),
                        'roi_path': roi_path,
                        'mask_path': mask_path,
                        'roi_bbox': [int(x) for x in roi_bbox],  # Convert to Python int
                        'roi_width': int(roi_width),
                        'roi_height': int(roi_height),
                        'original_size': [int(x) for x in image.shape[:2]],  # Convert to Python int
                        'status': 'success'
                    }
                    
                    roi_results.append(roi_result)
                    
                except Exception as e:
                    print(f"❌ Failed to process {image_path}: {e}")
                    failed_images.append({
                        'image_path': image_path,
                        'error': str(e)
                    })
                    continue
            
            if not roi_results:
                return {
                    'status': 'error',
                    'message': 'No ROIs extracted from any images',
                    'failed_images': failed_images
                }
            
            # Save ROI results
            results_file = os.path.join(project_dir, f"{project_id}_manufacturing_roi_{image_type}_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    'project_id': project_id,
                    'roi_method': 'manufacturing_segmentation',
                    'manufacturing_scenario': manufacturing_scenario,
                    'part_material': part_material,
                    'fixture_type': fixture_type,
                    'fixture_color': fixture_color,
                    'total_processed': int(len(roi_results)),  # Convert to Python int
                    'total_failed': int(len(failed_images)),   # Convert to Python int
                    'roi_results': roi_results,
                    'failed_images': failed_images
                }, f, indent=2)
            
            print(f"✅ Manufacturing ROI extraction completed: {len(roi_results)} success, {len(failed_images)} failed")
            
            return {
                'status': 'success',
                'project_id': project_id,
                'roi_method': 'manufacturing_segmentation',
                'total_processed': int(len(roi_results)),  # Convert to Python int
                'total_failed': int(len(failed_images)),   # Convert to Python int
                'roi_results': roi_results,
                'failed_images': failed_images
            }
            
        except Exception as e:
            print(f"❌ Manufacturing ROI extraction failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _create_manufacturing_part_mask(self, image: np.ndarray, scenario: str, 
                                      part_material: str, fixture_type: str, fixture_color: str) -> np.ndarray:
        """Create part mask using manufacturing-specific segmentation"""
        try:
            # Route to specific segmentation method based on scenario
            if scenario == "metal_machining" or (part_material == "metal" and fixture_color == "blue"):
                return self._segment_metal_on_blue_tray(image)
            elif scenario == "electronics":
                return self._segment_electronics_on_tray(image) 
            elif scenario == "automotive":
                return self._segment_automotive_parts(image, fixture_color)
            elif scenario == "textile":
                return self._segment_textile_on_background(image, fixture_color)
            else:
                # General approach - use color-based segmentation
                return self._segment_parts_on_colored_background(image, fixture_color)
                
        except Exception as e:
            print(f"⚠️ Manufacturing segmentation failed, using fallback: {e}")
            # Fallback to simple color-based segmentation
            return self._segment_parts_on_colored_background(image, fixture_color)
    
    def _segment_metal_on_blue_tray(self, image: np.ndarray) -> np.ndarray:
        """Segment metallic parts on blue tray"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Blue tray detection (expanded range)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Invert to get part mask
        part_mask = cv2.bitwise_not(blue_mask)
        
        # Morphological operations for cleanup
        kernel = np.ones((5, 5), np.uint8)
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_CLOSE, kernel)
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_OPEN, kernel)
        
        return part_mask
    
    def _segment_electronics_on_tray(self, image: np.ndarray) -> np.ndarray:
        """Segment electronic components on tray"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect components
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours and create mask
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask from significant contours
        part_mask = np.zeros(gray.shape, dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small noise
                cv2.fillPoly(part_mask, [contour], 255)
        
        return part_mask
    
    def _segment_automotive_parts(self, image: np.ndarray, fixture_color: str) -> np.ndarray:
        """Segment automotive parts"""
        return self._segment_parts_on_colored_background(image, fixture_color)
    
    def _segment_textile_on_background(self, image: np.ndarray, fixture_color: str) -> np.ndarray:
        """Segment textile/fabric on background"""
        # Use texture-based approach for fabric
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        part_mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return part_mask
    
    def _segment_parts_on_colored_background(self, image: np.ndarray, fixture_color: str) -> np.ndarray:
        """General color-based segmentation for parts on colored background"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common fixture colors
        color_ranges = {
            "blue": ([100, 50, 50], [130, 255, 255]),
            "black": ([0, 0, 0], [180, 255, 30]),
            "white": ([0, 0, 200], [180, 30, 255]),
            "gray": ([0, 0, 50], [180, 30, 200]),
            "green": ([40, 50, 50], [80, 255, 255]),
            "red": ([0, 50, 50], [20, 255, 255])
        }
        
        if fixture_color.lower() in color_ranges:
            lower, upper = color_ranges[fixture_color.lower()]
            background_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        else:
            # Default to edge-based detection if color unknown
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            part_mask = cv2.dilate(edges, kernel, iterations=2)
            return part_mask
        
        # Invert to get part mask
        part_mask = cv2.bitwise_not(background_mask)
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_CLOSE, kernel)
        part_mask = cv2.morphologyEx(part_mask, cv2.MORPH_OPEN, kernel)
        
        return part_mask
    
    def _extract_roi_from_mask(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Extract bounding box ROI from part mask"""
        try:
            # Find all white pixels in the mask
            white_pixels = np.where(mask > 0)
            
            if len(white_pixels[0]) == 0:
                return None
            
            # Get bounding box coordinates
            y_min, y_max = np.min(white_pixels[0]), np.max(white_pixels[0])
            x_min, x_max = np.min(white_pixels[1]), np.max(white_pixels[1])
            
            # Add small padding
            padding = 10
            y_min = max(0, int(y_min) - padding)
            x_min = max(0, int(x_min) - padding)
            y_max = min(mask.shape[0], int(y_max) + padding)
            x_max = min(mask.shape[1], int(x_max) + padding)
            
            return (int(x_min), int(y_min), int(x_max), int(y_max))
            
        except Exception as e:
            print(f"⚠️ Failed to extract ROI from mask: {e}")
            return None
    
    def _crop_image_to_roi(self, image: np.ndarray, roi_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop image to ROI bounding box"""
        x_min, y_min, x_max, y_max = roi_bbox
        return image[y_min:y_max, x_min:x_max]

# Create global instance
grounding_dino_service = GroundingDINOService()