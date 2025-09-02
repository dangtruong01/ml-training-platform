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
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - DINOv2 anomaly detection disabled")

class DINOv2AnomalyDetector:
    """
    DINOv2-based anomaly detection service for identifying defects in images
    
    Uses pretrained DINOv2 features to detect anomalies by comparing against
    normal image statistics without requiring supervised training.
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.feature_cache = {}
        
        # Base directory for auto annotation projects
        self.projects_base_dir = os.path.abspath("ml/auto_annotation/projects")
        
        # Initialize model
        if TORCH_AVAILABLE:
            self._initialize_model()
        
        print("🧠 DINOv2 Anomaly Detection Service initialized")
    
    def _initialize_model(self):
        """Initialize DINOv2 model and preprocessing"""
        try:
            print("📥 Loading DINOv2 model...")
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")
            
            # Load DINOv2 ViT-S/14 model (good speed/quality trade-off)
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.to(self.device)
            self.model.eval()
            
            # Image preprocessing transform
            self.transform = transforms.Compose([
                transforms.Resize((518, 518)),  # DINOv2 expects multiple of 14
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ DINOv2 model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load DINOv2 model: {e}")
            self.model = None
    
    def extract_features(self, image_paths: List[str], project_id: str = None) -> Dict:
        """
        Extract DINOv2 features from images
        
        Args:
            image_paths: List of image file paths
            project_id: Project ID for organizing features
            
        Returns:
            Dict with extracted features and metadata
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {
                'status': 'error',
                'message': 'DINOv2 model not available'
            }
        
        try:
            print(f"🔍 Extracting features from {len(image_paths)} images")
            
            all_features = []
            image_info = []
            failed_images = []
            
            with torch.no_grad():
                for i, image_path in enumerate(image_paths):
                    try:
                        print(f"📸 Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                        
                        # Load and preprocess image
                        image = Image.open(image_path).convert('RGB')
                        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                        
                        # Extract patch features (not global features)
                        # Use forward_features to get patch tokens before global pooling
                        feature_dict = self.model.forward_features(input_tensor)
                        
                        # Extract the patch tokens from the feature dict
                        if isinstance(feature_dict, dict):
                            # DINOv2 returns dict with 'x_norm_patchtokens' for patch features
                            if 'x_norm_patchtokens' in feature_dict:
                                features = feature_dict['x_norm_patchtokens']
                            elif 'x_prenorm' in feature_dict:
                                features = feature_dict['x_prenorm']
                            else:
                                # Fallback to any tensor in the dict
                                features = list(feature_dict.values())[0]
                        else:
                            features = feature_dict
                        
                        # Get patch features (reshape to spatial grid)
                        # We now have pure patch tokens without CLS token: [1, 1369, 384]
                        # where 1369 = 37^2 patches for 518x518 input with patch size 14
                        if features.dim() == 3:  # [batch, num_patches, feature_dim]
                            batch_size, num_patches, feature_dim = features.shape
                            
                            # Calculate grid size
                            grid_size = int(np.sqrt(num_patches))
                            
                            if grid_size * grid_size == num_patches:
                                # Perfect square - reshape to spatial grid
                                patch_features = features.view(1, grid_size, grid_size, feature_dim)  # [1, H, W, C]
                            else:
                                # Not a perfect square - use as flattened features
                                print(f"⚠️ Non-square patch count {num_patches}, using flattened features")
                                patch_features = features.view(1, 1, num_patches, feature_dim)  # [1, 1, N, C]
                        else:
                            # Handle 2D case: [batch, feature_dim] - global features only
                            patch_features = features.unsqueeze(1).unsqueeze(1)  # [1, 1, 1, feature_dim]
                        
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
                features_dir = os.path.join(project_dir, "anomaly_features")
                os.makedirs(features_dir, exist_ok=True)
                
                cache_data = {
                    'features': features_array,
                    'image_info': image_info,
                    'extraction_timestamp': np.datetime64('now').item().isoformat()
                }
                
                cache_file = os.path.join(features_dir, f"{project_id}_features.npz")
                np.savez_compressed(cache_file, **cache_data)
                print(f"💾 Cached features to: {cache_file}")
            
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
            print(f"❌ Feature extraction failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def build_normal_model(self, features: np.ndarray) -> Dict:
        """
        Build statistical model of normal features
        
        Args:
            features: Feature array [N, H, W, C] from normal images
            
        Returns:
            Dict with normal statistics
        """
        try:
            print(f"📊 Building normal model from {features.shape[0]} images")
            
            # Reshape to [N*H*W, C] for statistics
            N, H, W, C = features.shape
            features_flat = features.reshape(-1, C)
            
            # Compute statistics
            normal_mean = np.mean(features_flat, axis=0)  # [C,]
            normal_std = np.std(features_flat, axis=0)    # [C,]
            normal_cov = np.cov(features_flat, rowvar=False)  # [C, C]
            
            # Compute per-patch statistics (optional, for spatial analysis)
            patch_means = np.mean(features, axis=0)  # [H, W, C]
            patch_stds = np.std(features, axis=0)    # [H, W, C]
            
            normal_model = {
                'global_mean': normal_mean,
                'global_std': normal_std,
                'global_cov': normal_cov,
                'patch_means': patch_means,
                'patch_stds': patch_stds,
                'feature_shape': features.shape,
                'num_normal_images': N
            }
            
            print(f"✅ Normal model built: {C}-dim features, {N} images")
            
            return {
                'status': 'success',
                'normal_model': normal_model
            }
            
        except Exception as e:
            print(f"❌ Failed to build normal model: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def detect_anomalies(
        self, 
        test_features: np.ndarray, 
        normal_model: Dict, 
        method: str = 'mahalanobis',
        threshold_percentile: float = 95.0
    ) -> Dict:
        """
        Detect anomalies in test features using normal model
        
        Args:
            test_features: Test feature array [N, H, W, C]
            normal_model: Normal statistics from build_normal_model
            method: Anomaly detection method ('mahalanobis', 'euclidean', 'cosine')
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Dict with anomaly scores and heatmaps
        """
        try:
            print(f"🔍 Detecting anomalies in {test_features.shape[0]} images")
            print(f"🎯 Method: {method}, Threshold: {threshold_percentile}%")
            
            N, H, W, C = test_features.shape
            normal_mean = normal_model['global_mean']
            normal_std = normal_model['global_std']
            normal_cov = normal_model['global_cov']
            
            # Reshape for processing
            test_flat = test_features.reshape(-1, C)  # [N*H*W, C]
            
            # Compute anomaly scores
            if method == 'mahalanobis':
                # Mahalanobis distance
                try:
                    cov_inv = np.linalg.pinv(normal_cov)
                    diff = test_flat - normal_mean
                    scores = np.sum((diff @ cov_inv) * diff, axis=1)
                except:
                    # Fallback to euclidean if covariance inversion fails
                    print("⚠️ Covariance inversion failed, using Euclidean distance")
                    scores = np.sum((test_flat - normal_mean) ** 2, axis=1)
                    
            elif method == 'euclidean':
                # Euclidean distance
                scores = np.sum((test_flat - normal_mean) ** 2, axis=1)
                
            elif method == 'cosine':
                # Cosine distance
                test_norm = test_flat / (np.linalg.norm(test_flat, axis=1, keepdims=True) + 1e-8)
                normal_norm = normal_mean / (np.linalg.norm(normal_mean) + 1e-8)
                scores = 1 - np.dot(test_norm, normal_norm)
                
            else:
                raise ValueError(f"Unknown anomaly detection method: {method}")
            
            # Reshape scores back to spatial grid
            score_maps = scores.reshape(N, H, W)  # [N, H, W]
            
            # Compute threshold
            all_scores = scores.flatten()
            threshold = np.percentile(all_scores, threshold_percentile)
            
            # Generate binary anomaly masks
            anomaly_masks = score_maps > threshold
            
            # Resize score maps to original image size (if needed)
            resized_score_maps = []
            resized_anomaly_masks = []
            
            for i in range(N):
                # Resize from 37x37 to larger size for better visualization
                score_map = score_maps[i]
                anomaly_mask = anomaly_masks[i]
                
                # Resize to 256x256 for visualization
                score_resized = cv2.resize(score_map.astype(np.float32), (256, 256), 
                                         interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(anomaly_mask.astype(np.uint8), (256, 256), 
                                        interpolation=cv2.INTER_NEAREST)
                
                resized_score_maps.append(score_resized)
                resized_anomaly_masks.append(mask_resized)
            
            # Compute per-image anomaly statistics
            image_anomaly_scores = []
            for i in range(N):
                img_scores = score_maps[i]
                image_stats = {
                    'max_score': float(np.max(img_scores)),
                    'mean_score': float(np.mean(img_scores)),
                    'anomaly_percentage': float(np.sum(anomaly_masks[i]) / (H * W) * 100),
                    'num_anomaly_patches': int(np.sum(anomaly_masks[i]))
                }
                image_anomaly_scores.append(image_stats)
            
            print(f"✅ Anomaly detection completed")
            
            return {
                'status': 'success',
                'score_maps': resized_score_maps,  # List of [256, 256] arrays
                'anomaly_masks': resized_anomaly_masks,  # List of [256, 256] binary arrays
                'raw_score_maps': score_maps,  # Original [N, 37, 37] arrays
                'image_scores': image_anomaly_scores,
                'threshold': float(threshold),
                'method': method,
                'threshold_percentile': threshold_percentile,
                'global_stats': {
                    'min_score': float(np.min(all_scores)),
                    'max_score': float(np.max(all_scores)),
                    'mean_score': float(np.mean(all_scores)),
                    'std_score': float(np.std(all_scores))
                }
            }
            
        except Exception as e:
            print(f"❌ Anomaly detection failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def run_anomaly_detection_pipeline(
        self, 
        roi_image_paths: List[str], 
        project_id: str,
        method: str = 'mahalanobis',
        threshold_percentile: float = 95.0
    ) -> Dict:
        """
        Complete anomaly detection pipeline
        
        Args:
            roi_image_paths: List of ROI image paths (normal images for training)
            project_id: Project ID for caching
            method: Anomaly detection method
            threshold_percentile: Anomaly threshold percentile
            
        Returns:
            Complete anomaly detection results
        """
        try:
            print(f"🚀 Starting anomaly detection pipeline for project {project_id}")
            
            # Step 1: Extract features
            feature_result = self.extract_features(roi_image_paths, cache_key=project_id)
            if feature_result['status'] != 'success':
                return feature_result
            
            features = feature_result['features']
            image_info = feature_result['image_info']
            
            # Step 2: Build normal model
            normal_model_result = self.build_normal_model(features)
            if normal_model_result['status'] != 'success':
                return normal_model_result
            
            normal_model = normal_model_result['normal_model']
            
            # Step 3: Detect anomalies (using same images as normal baseline for now)
            # In real use, you'd use different test images
            anomaly_result = self.detect_anomalies(features, normal_model, method, threshold_percentile)
            if anomaly_result['status'] != 'success':
                return anomaly_result
            
            # Step 4: Combine results
            results = []
            for i, info in enumerate(image_info):
                result = {
                    'image_path': info['image_path'],
                    'image_name': info['image_name'],
                    'original_size': info['original_size'],
                    'score_map_base64': self._array_to_base64(anomaly_result['score_maps'][i]),
                    'anomaly_mask_base64': self._array_to_base64(anomaly_result['anomaly_masks'][i]),
                    'image_scores': anomaly_result['image_scores'][i],
                    'status': 'success'
                }
                results.append(result)
            
            # Save results to project-specific directory
            project_dir = os.path.join(self.projects_base_dir, project_id)
            results_dir = os.path.join(project_dir, "anomaly_results")
            os.makedirs(results_dir, exist_ok=True)
            
            results_file = os.path.join(results_dir, f"{project_id}_anomaly_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    'project_id': project_id,
                    'method': method,
                    'threshold_percentile': threshold_percentile,
                    'global_stats': anomaly_result['global_stats'],
                    'results': results,
                    'timestamp': np.datetime64('now').item().isoformat()
                }, f, indent=2)
            
            print(f"💾 Results saved to: {results_file}")
            
            return {
                'status': 'success',
                'project_id': project_id,
                'method': method,
                'threshold_percentile': threshold_percentile,
                'total_images': len(results),
                'results': results,
                'global_stats': anomaly_result['global_stats'],
                'normal_model_info': {
                    'num_normal_images': normal_model['num_normal_images'],
                    'feature_dimensions': len(normal_model['global_mean'])
                }
            }
            
        except Exception as e:
            print(f"❌ Anomaly detection pipeline failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _array_to_base64(self, array: np.ndarray) -> str:
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
    
    def build_normal_model(self, features: np.ndarray) -> Dict:
        """
        Build statistical normal model from normal image features
        
        Args:
            features: Normal image features [N, H, W, C]
            
        Returns:
            Dict containing normal model statistics
        """
        try:
            print(f"🧠 Building normal model from features shape: {features.shape}")
            
            N, H, W, C = features.shape
            
            # Reshape features to [N*H*W, C] for statistical calculations
            features_flat = features.reshape(-1, C)  # [N*H*W, C]
            
            # Calculate global statistics across all patches
            global_mean = np.mean(features_flat, axis=0)  # [C]
            global_std = np.std(features_flat, axis=0)    # [C]
            global_cov = np.cov(features_flat.T)          # [C, C]
            
            print(f"📊 Normal model statistics:")
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
                'message': 'Normal model built successfully'
            }
            
        except Exception as e:
            print(f"❌ Error building normal model: {e}")
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
                    'message': 'DINOv2 model not available'
                }
            
            print(f"🎯 Detecting defects in {len(defective_image_paths)} images using normal model")
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
            
            print(f"🔍 Normal model loaded: mean_norm={np.linalg.norm(global_mean):.3f}")
            
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
                heatmap_saved_path = self._save_heatmap_image(heatmap, project_id, heatmap_filename, "dinov2")
                
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
            results_dir = os.path.join(project_dir, "defect_detection_results")
            os.makedirs(results_dir, exist_ok=True)
            
            results_file = os.path.join(results_dir, f"{project_id}_defect_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    'project_id': project_id,
                    'method': method,
                    'threshold_percentile': threshold_percentile,
                    'global_stats': anomaly_result['global_stats'],
                    'total_images': len(defective_image_paths),
                    'results': results,
                    'timestamp': np.datetime64('now').item().isoformat()
                }, f, indent=2)
            
            print(f"💾 Defect detection results saved to: {results_file}")
            
            return {
                'status': 'success',
                'project_id': project_id,
                'method': method,
                'threshold_percentile': threshold_percentile,
                'total_images': len(defective_image_paths),
                'global_stats': anomaly_result['global_stats'],
                'results': results,
                'message': 'Defect detection completed successfully'
            }
            
        except Exception as e:
            print(f"❌ Error in defect detection: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _detect_anomalies_with_model(
        self,
        features: np.ndarray,
        global_mean: np.ndarray,
        global_std: np.ndarray,
        global_cov: np.ndarray,
        method: str,
        threshold_percentile: float
    ) -> Dict:
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
            
            print(f"📈 Global anomaly statistics:")
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
    
    def _calculate_anomaly_scores_single_image(
        self,
        image_features: np.ndarray,  # [H, W, C]
        global_mean: np.ndarray,     # [C]
        global_std: np.ndarray,      # [C] 
        global_cov: np.ndarray,      # [C, C]
        method: str
    ) -> np.ndarray:
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
    
    def _calculate_image_anomaly_stats(self, score_map: np.ndarray, threshold_percentile: float) -> Dict:
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
    
    def _create_heatmap(self, score_map: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
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
    
    def _save_heatmap_image(self, heatmap: np.ndarray, project_id: str, filename: str, model_type: str = "dinov2") -> str:
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
dinov2_service = DINOv2AnomalyDetector()