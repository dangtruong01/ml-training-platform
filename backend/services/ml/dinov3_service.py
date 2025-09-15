import os
import cv2
import numpy as np
import json
import base64
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel
    from transformers.image_utils import load_image
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch or Transformers not available - DINOv3 anomaly detection disabled")

class DINOv3AnomalyDetector:
    """
    DINOv3-based anomaly detection service for identifying defects in images
    
    Uses pretrained DINOv3 features to detect anomalies by comparing against
    normal image statistics without requiring supervised training.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.feature_cache = {}
        
        # Base directory for auto annotation projects
        self.projects_base_dir = os.path.abspath("ml/auto_annotation/projects")
        
        # Initialize model
        if TORCH_AVAILABLE:
            self._initialize_model()
        
        print("🧠 DINOv3 Anomaly Detection Service initialized")
    
    def _initialize_model(self):
        """Initialize DINOv3 model and preprocessing"""
        # List of DINOv3 models to try (from smallest to larger)
        model_candidates = [
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        ]
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Try loading models in order of preference
        for model_name in model_candidates:
            try:
                print(f"📥 Attempting to load DINOv3 model: {model_name}")
                
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map="auto" if self.device.type == 'cuda' else None
                )
                
                if self.device.type != 'cuda':
                    self.model.to(self.device)
                
                self.model.eval()
                
                print(f"✅ DINOv3 model loaded successfully: {model_name}")
                return  # Successfully loaded, exit function
                
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
                self.model = None
                self.processor = None
                continue  # Try next model
        
        # If all models failed
        print("❌ Failed to load any DINOv3 model. Please check:")
        print("   1. Internet connection")
        print("   2. HuggingFace Hub access")
        print("   3. PyTorch and Transformers installation")
        print("   Consider using DINOv2 as an alternative.")
        self.model = None
        self.processor = None
    
    def extract_features(self, image_paths: List[str], project_id: str = None) -> Dict:
        """
        Extract DINOv3 features from images
        
        Args:
            image_paths: List of image file paths
            project_id: Project ID for organizing features
            
        Returns:
            Dict with extracted features and metadata
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {
                'status': 'error',
                'message': 'DINOv3 model not available'
            }
        
        try:
            print(f"🔍 Extracting DINOv3 features from {len(image_paths)} images")
            
            all_features = []
            image_info = []
            failed_images = []
            
            with torch.inference_mode():
                for i, image_path in enumerate(image_paths):
                    try:
                        print(f"📸 Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                        
                        # Load image
                        image = Image.open(image_path).convert('RGB')
                        
                        # Process image with DINOv3 processor
                        inputs = self.processor(images=image, return_tensors="pt")
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                        
                        # Extract features
                        outputs = self.model(**inputs)
                        
                        # Get patch embeddings (excluding CLS token)
                        # DINOv3 outputs: last_hidden_state shape [1, num_patches+1, hidden_size]
                        # We exclude the first token (CLS token) to get pure patch features
                        last_hidden_state = outputs.last_hidden_state  # [1, 197, 384] for ViT-S/16
                        patch_embeddings = last_hidden_state[:, 1:, :]  # [1, 196, 384] - exclude CLS
                        
                        # Calculate grid size (14x14 = 196 patches for 224x224 input)
                        batch_size, num_patches, hidden_size = patch_embeddings.shape
                        grid_size = int(np.sqrt(num_patches))
                        
                        if grid_size * grid_size == num_patches:
                            # Reshape to spatial grid [1, H, W, C]
                            patch_features = patch_embeddings.view(1, grid_size, grid_size, hidden_size)
                        else:
                            # Fallback to flattened features
                            print(f"⚠️ Non-square patch count {num_patches}, using flattened features")
                            patch_features = patch_embeddings.view(1, 1, num_patches, hidden_size)
                        
                        # Store features and metadata
                        features_np = patch_features.cpu().numpy()[0]  # [H, W, C]
                        all_features.append(features_np)
                        
                        image_info.append({
                            'image_path': image_path,
                            'image_name': os.path.basename(image_path),
                            'original_size': image.size,
                            'feature_shape': features_np.shape,
                            'status': 'success'
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Failed to process {image_path}: {e}")
                        failed_images.append({
                            'image_path': image_path,
                            'error': str(e)
                        })
                        continue
            
            if not all_features:
                return {
                    'status': 'error',
                    'message': 'No features extracted from any images',
                    'failed_images': failed_images
                }
            
            # Stack features
            features_array = np.stack(all_features, axis=0)  # [N, H, W, C]
            
            # Cache features if project_id provided
            if project_id:
                # Create project-specific features directory
                project_dir = os.path.join(self.projects_base_dir, project_id)
                features_dir = os.path.join(project_dir, "dinov3_anomaly_features")
                os.makedirs(features_dir, exist_ok=True)
                
                cache_data = {
                    'features': features_array,
                    'image_info': image_info,
                    'extraction_timestamp': np.datetime64('now').item().isoformat()
                }
                
                cache_file = os.path.join(features_dir, f"{project_id}_dinov3_features.npz")
                np.savez_compressed(cache_file, **cache_data)
                print(f"💾 Cached DINOv3 features to: {cache_file}")
            
            return {
                'status': 'success',
                'features': features_array,
                'image_info': image_info,
                'failed_images': failed_images,
                'feature_shape': features_array.shape,
                'total_processed': len(all_features),
                'total_failed': len(failed_images)
            }
            
        except Exception as e:
            print(f"❌ DINOv3 feature extraction failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def build_normal_model(self, features: np.ndarray) -> Dict:
        """
        Build statistical normal model from normal image features
        
        Args:
            features: Normal image features [N, H, W, C]
            
        Returns:
            Dict containing normal model statistics
        """
        try:
            print(f"🧠 Building DINOv3 normal model from features shape: {features.shape}")
            
            N, H, W, C = features.shape
            
            # Reshape features to [N*H*W, C] for statistical calculations
            features_flat = features.reshape(-1, C)  # [N*H*W, C]
            
            # Calculate global statistics across all patches
            global_mean = np.mean(features_flat, axis=0)  # [C]
            global_std = np.std(features_flat, axis=0)    # [C]
            global_cov = np.cov(features_flat.T)          # [C, C]
            
            print(f"📊 DINOv3 Normal model statistics:")
            print(f"  Mean norm: {np.linalg.norm(global_mean):.3f}")
            print(f"  Std mean: {np.mean(global_std):.3f}")
            print(f"  Feature variance: {np.mean(global_std**2):.3f}")
            
            normal_model = {
                'global_mean': global_mean,
                'global_std': global_std, 
                'global_cov': global_cov,
                'feature_shape': features.shape
            }
            
            return {
                'status': 'success',
                'normal_model': normal_model,
                'message': 'DINOv3 normal model built successfully'
            }
            
        except Exception as e:
            print(f"❌ Error building DINOv3 normal model: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def detect_defects_with_normal_model(
        self, 
        defective_image_paths: List[str], 
        normal_model_data: Dict,
        project_id: str,
        method: str = "mahalanobis",
        threshold_percentile: float = 95.0
    ) -> Dict:
        """
        Detect anomalies in defective images using pre-built normal model
        
        Args:
            defective_image_paths: List of defective ROI image paths
            normal_model_data: Pre-computed normal model statistics
            project_id: Project ID for saving results
            method: Anomaly detection method
            threshold_percentile: Threshold for anomaly detection
            
        Returns:
            Dict with anomaly detection results
        """
        try:
            if not TORCH_AVAILABLE or self.model is None:
                return {
                    'status': 'error',
                    'message': 'DINOv3 model not available'
                }
            
            print(f"🎯 Detecting defects with DINOv3 in {len(defective_image_paths)} images using normal model")
            print(f"🧠 Method: {method}, Threshold: {threshold_percentile}%")
            
            # Extract features from defective images
            feature_result = self.extract_features(defective_image_paths, project_id=project_id)
            if feature_result['status'] != 'success':
                return feature_result
            
            defective_features = feature_result['features']  # [N, H, W, C]
            
            # Load normal model statistics
            global_mean = np.array(normal_model_data['global_mean'])  # [C]
            global_std = np.array(normal_model_data['global_std'])    # [C] 
            global_cov = np.array(normal_model_data['global_cov'])    # [C, C]
            
            print(f"🔍 DINOv3 normal model loaded: mean_norm={np.linalg.norm(global_mean):.3f}")
            
            # Detect anomalies using the normal model
            anomaly_result = self._detect_anomalies_with_model(
                defective_features,
                global_mean,
                global_std, 
                global_cov,
                method,
                threshold_percentile
            )
            
            if anomaly_result['status'] != 'success':
                return anomaly_result
            
            # Prepare results with heatmaps
            results = []
            for i, image_path in enumerate(defective_image_paths):
                # Generate score map for this image
                image_features = defective_features[i]  # [H, W, C]
                
                score_map = self._calculate_anomaly_scores_single_image(
                    image_features,
                    global_mean,
                    global_std,
                    global_cov,
                    method
                )
                
                # Calculate image-level statistics
                image_scores = self._calculate_image_anomaly_stats(score_map, threshold_percentile)
                
                # Convert score map to heatmap and base64
                heatmap = self._create_heatmap(score_map)
                heatmap_base64 = self._array_to_base64(heatmap)
                
                # Save heatmap image for visual inspection
                heatmap_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_heatmap.png"
                heatmap_saved_path = self._save_heatmap_image(heatmap, project_id, heatmap_filename, "dinov3")
                
                result = {
                    'image_name': os.path.basename(image_path),
                    'image_path': image_path,
                    'image_scores': image_scores,
                    'score_map_base64': heatmap_base64,
                    'heatmap_file_path': heatmap_saved_path,  # New: saved heatmap file path
                    'status': 'success'
                }
                results.append(result)
            
            # Save results to project directory
            project_dir = os.path.join(self.projects_base_dir, project_id)
            results_dir = os.path.join(project_dir, "dinov3_defect_detection_results")
            os.makedirs(results_dir, exist_ok=True)
            
            results_file = os.path.join(results_dir, f"{project_id}_dinov3_defect_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    'project_id': project_id,
                    'model': 'DINOv3',
                    'method': method,
                    'threshold_percentile': threshold_percentile,
                    'global_stats': anomaly_result['global_stats'],
                    'total_images': len(defective_image_paths),
                    'results': results,
                    'timestamp': np.datetime64('now').item().isoformat()
                }, f, indent=2)
            
            print(f"💾 DINOv3 defect detection results saved to: {results_file}")
            
            return {
                'status': 'success',
                'project_id': project_id,
                'model': 'DINOv3',
                'method': method,
                'threshold_percentile': threshold_percentile,
                'total_images': len(defective_image_paths),
                'global_stats': anomaly_result['global_stats'],
                'results': results,
                'message': 'DINOv3 defect detection completed successfully'
            }
            
        except Exception as e:
            print(f"❌ Error in DINOv3 defect detection: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    # Copy all the helper methods from DINOv2 service (they're identical)
    def _detect_anomalies_with_model(self, features, global_mean, global_std, global_cov, method, threshold_percentile):
        """Detect anomalies using pre-computed normal model statistics"""
        try:
            N, H, W, C = features.shape
            
            # Calculate anomaly scores for all patches
            all_scores = []
            
            for i in range(N):
                image_features = features[i]  # [H, W, C]
                
                scores = self._calculate_anomaly_scores_single_image(
                    image_features,
                    global_mean,
                    global_std,
                    global_cov,
                    method
                )
                
                all_scores.append(scores.flatten())
            
            # Flatten all scores for global statistics
            all_scores_flat = np.concatenate(all_scores)
            
            # Calculate threshold
            threshold = np.percentile(all_scores_flat, threshold_percentile)
            
            global_stats = {
                'min_score': float(np.min(all_scores_flat)),
                'max_score': float(np.max(all_scores_flat)),
                'mean_score': float(np.mean(all_scores_flat)),
                'std_score': float(np.std(all_scores_flat)),
                'threshold': float(threshold),
                'method': method
            }
            
            print(f"📈 Global DINOv3 anomaly statistics:")
            print(f"  Min score: {global_stats['min_score']:.3f}")
            print(f"  Max score: {global_stats['max_score']:.3f}")
            print(f"  Threshold: {global_stats['threshold']:.3f}")
            
            return {
                'status': 'success',
                'global_stats': global_stats
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _calculate_anomaly_scores_single_image(self, image_features, global_mean, global_std, global_cov, method):
        """Calculate anomaly scores for a single image"""
        H, W, C = image_features.shape
        
        # Reshape to [H*W, C] for vectorized computation
        features_flat = image_features.reshape(-1, C)
        
        if method == "mahalanobis":
            try:
                # Mahalanobis distance
                inv_cov = np.linalg.pinv(global_cov + np.eye(C) * 1e-6)
                diff = features_flat - global_mean[None, :]  # [H*W, C]
                scores = np.sum(diff @ inv_cov * diff, axis=1)  # [H*W]
            except:
                # Fallback to euclidean if covariance is singular
                diff = features_flat - global_mean[None, :]
                scores = np.sum(diff ** 2, axis=1)
        elif method == "euclidean":
            # Euclidean distance
            diff = features_flat - global_mean[None, :]
            scores = np.sum(diff ** 2, axis=1)
        elif method == "cosine":
            # Cosine distance
            norm_features = features_flat / (np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8)
            norm_mean = global_mean / (np.linalg.norm(global_mean) + 1e-8)
            scores = 1.0 - np.dot(norm_features, norm_mean)
        else:
            # Default to euclidean
            diff = features_flat - global_mean[None, :]
            scores = np.sum(diff ** 2, axis=1)
        
        # Reshape back to [H, W]
        return scores.reshape(H, W)
    
    def _calculate_image_anomaly_stats(self, score_map, threshold_percentile):
        """Calculate anomaly statistics for a single image score map"""
        # Calculate threshold from global percentile (simplified - using image-local percentile)
        threshold = np.percentile(score_map.flatten(), threshold_percentile)
        
        # Count anomalous patches
        anomaly_mask = score_map > threshold
        
        return {
            'max_score': float(np.max(score_map)),
            'mean_score': float(np.mean(score_map)),
            'anomaly_percentage': float(np.sum(anomaly_mask) / anomaly_mask.size * 100),
            'num_anomaly_patches': int(np.sum(anomaly_mask)),
            'threshold_used': float(threshold)
        }
    
    def _create_heatmap(self, score_map, target_size=(256, 256)):
        """Create a colored heatmap from anomaly scores"""
        try:
            # Normalize scores to 0-1 range
            score_normalized = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
            
            # Resize to target size
            if score_map.shape != target_size:
                score_resized = cv2.resize(score_normalized.astype(np.float32), target_size, 
                                         interpolation=cv2.INTER_LINEAR)
            else:
                score_resized = score_normalized
            
            # Convert to heatmap using colormap (0=blue/cold, 1=red/hot)
            heatmap = cv2.applyColorMap((score_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            return heatmap
            
        except Exception as e:
            print(f"⚠️ Failed to create heatmap: {e}")
            # Return a blank heatmap as fallback
            return np.zeros((*target_size, 3), dtype=np.uint8)
    
    def _array_to_base64(self, array):
        """Convert numpy array to base64 string for JSON serialization"""
        try:
            # Normalize to 0-255 range
            if array.dtype != np.uint8:
                array_norm = ((array - array.min()) / (array.max() - array.min() + 1e-8) * 255).astype(np.uint8)
            else:
                array_norm = array
            
            # Convert to PNG bytes
            _, buffer = cv2.imencode('.png', array_norm)
            
            # Encode to base64
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            print(f"⚠️ Failed to convert array to base64: {e}")
            return ""
    
    def _save_heatmap_image(self, heatmap: np.ndarray, project_id: str, filename: str, model_type: str = "dinov3") -> str:
        """
        Save heatmap as PNG image file for visual inspection
        
        Args:
            heatmap: Heatmap array from _create_heatmap()
            project_id: Project ID
            filename: Filename for the saved heatmap
            model_type: Model type for directory naming
            
        Returns:
            Path to saved heatmap file
        """
        try:
            project_dir = os.path.join(self.projects_base_dir, project_id)
            heatmaps_dir = os.path.join(project_dir, f"{model_type}_heatmap_analysis")
            os.makedirs(heatmaps_dir, exist_ok=True)
            
            heatmap_path = os.path.join(heatmaps_dir, filename)
            
            # Save heatmap as PNG
            cv2.imwrite(heatmap_path, heatmap)
            
            print(f"💾 Saved heatmap: {heatmap_path}")
            return heatmap_path
            
        except Exception as e:
            print(f"⚠️ Failed to save heatmap {filename}: {e}")
            return ""

# Create global instance
dinov3_service = DINOv3AnomalyDetector()