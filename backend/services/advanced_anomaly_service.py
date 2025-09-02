import os
import cv2
import numpy as np
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging

try:
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.covariance import EllipticEnvelope
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - advanced anomaly detection disabled")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - deep learning anomaly detection disabled")

class AdvancedAnomalyDetector:
    """
    Advanced anomaly detection service with multiple algorithms:
    - One-Class SVM
    - Isolation Forest  
    - Local Outlier Factor
    - Robust Covariance (Elliptic Envelope)
    - PCA-based anomaly detection
    - Autoencoder (if PyTorch available)
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.pca_models = {}
        
        # Base directory for projects
        self.projects_base_dir = os.path.abspath("ml/auto_annotation/projects")
        
        print("🎯 Advanced Anomaly Detection Service initialized")
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn required for advanced anomaly detection")
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available - autoencoder methods disabled")
    
    def build_advanced_normal_model(
        self, 
        features: np.ndarray, 
        project_id: str,
        methods: List[str] = None,
        **kwargs
    ) -> Dict:
        """
        Build advanced normal models using multiple algorithms
        
        Args:
            features: Normal image features [N, H, W, C]
            project_id: Project ID for saving models
            methods: List of methods to train ['ocsvm', 'isolation_forest', 'lof', 'elliptic', 'pca', 'autoencoder']
            **kwargs: Method-specific parameters
            
        Returns:
            Dict with training results for each method
        """
        if not SKLEARN_AVAILABLE:
            return {
                'status': 'error',
                'message': 'scikit-learn not available for advanced anomaly detection'
            }
        
        if methods is None:
            methods = ['ocsvm', 'isolation_forest', 'elliptic', 'pca']
        
        try:
            print(f"🎯 Building advanced normal models for project {project_id}")
            print(f"📊 Features shape: {features.shape}")
            print(f"🔬 Methods to train: {methods}")
            
            N, H, W, C = features.shape
            
            # Reshape features to [N*H*W, C] for sklearn
            features_flat = features.reshape(-1, C)  # [N*H*W, C]
            print(f"🔄 Flattened features shape: {features_flat.shape}")
            
            # Standardize features (important for most methods)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_flat)
            
            # Optional PCA dimensionality reduction for high-dimensional features
            n_components = min(50, C, features_scaled.shape[0] // 2)  # Adaptive PCA components
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)
            
            print(f"📉 PCA reduced features to {features_pca.shape[1]} dimensions")
            print(f"📊 PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
            
            results = {}
            trained_models = {}
            
            # 1. One-Class SVM
            if 'ocsvm' in methods:
                print("🤖 Training One-Class SVM...")
                ocsvm_params = {
                    'kernel': kwargs.get('ocsvm_kernel', 'rbf'),
                    'gamma': kwargs.get('ocsvm_gamma', 'scale'),
                    'nu': kwargs.get('ocsvm_nu', 0.05),  # Expected fraction of outliers
                }
                
                ocsvm = OneClassSVM(**ocsvm_params)
                ocsvm.fit(features_pca)  # Use PCA features for efficiency
                
                # Calculate decision scores for validation
                decision_scores = ocsvm.decision_function(features_pca)
                threshold = np.percentile(decision_scores, 5)  # 5th percentile as threshold
                
                trained_models['ocsvm'] = {
                    'model': ocsvm,
                    'threshold': threshold,
                    'params': ocsvm_params,
                    'use_pca': True
                }
                
                results['ocsvm'] = {
                    'status': 'success',
                    'threshold': float(threshold),
                    'decision_scores_stats': {
                        'mean': float(np.mean(decision_scores)),
                        'std': float(np.std(decision_scores)),
                        'min': float(np.min(decision_scores)),
                        'max': float(np.max(decision_scores))
                    }
                }
                print(f"  ✅ One-Class SVM trained, threshold: {threshold:.3f}")
            
            # 2. Isolation Forest
            if 'isolation_forest' in methods:
                print("🌲 Training Isolation Forest...")
                iso_params = {
                    'n_estimators': kwargs.get('iso_n_estimators', 100),
                    'contamination': kwargs.get('iso_contamination', 0.05),
                    'random_state': kwargs.get('random_state', 42)
                }
                
                iso_forest = IsolationForest(**iso_params)
                iso_forest.fit(features_scaled)  # Use scaled features (full dimensionality is OK)
                
                # Calculate anomaly scores
                anomaly_scores = iso_forest.decision_function(features_scaled)
                threshold = iso_forest.offset_
                
                trained_models['isolation_forest'] = {
                    'model': iso_forest,
                    'threshold': threshold,
                    'params': iso_params,
                    'use_pca': False
                }
                
                results['isolation_forest'] = {
                    'status': 'success',
                    'threshold': float(threshold),
                    'anomaly_scores_stats': {
                        'mean': float(np.mean(anomaly_scores)),
                        'std': float(np.std(anomaly_scores)),
                        'min': float(np.min(anomaly_scores)),
                        'max': float(np.max(anomaly_scores))
                    }
                }
                print(f"  ✅ Isolation Forest trained, threshold: {threshold:.3f}")
            
            # 3. Elliptic Envelope (Robust Covariance)
            if 'elliptic' in methods:
                print("🔮 Training Elliptic Envelope...")
                elliptic_params = {
                    'contamination': kwargs.get('elliptic_contamination', 0.05),
                    'random_state': kwargs.get('random_state', 42)
                }
                
                elliptic = EllipticEnvelope(**elliptic_params)
                elliptic.fit(features_pca)  # Use PCA features
                
                # Calculate Mahalanobis distances
                mahal_distances = elliptic.mahalanobis(features_pca)
                threshold = np.percentile(mahal_distances, 95)  # 95th percentile
                
                trained_models['elliptic'] = {
                    'model': elliptic,
                    'threshold': threshold,
                    'params': elliptic_params,
                    'use_pca': True
                }
                
                results['elliptic'] = {
                    'status': 'success',
                    'threshold': float(threshold),
                    'mahalanobis_stats': {
                        'mean': float(np.mean(mahal_distances)),
                        'std': float(np.std(mahal_distances)),
                        'min': float(np.min(mahal_distances)),
                        'max': float(np.max(mahal_distances))
                    }
                }
                print(f"  ✅ Elliptic Envelope trained, threshold: {threshold:.3f}")
            
            # 4. PCA-based anomaly detection
            if 'pca' in methods:
                print("📊 Training PCA-based anomaly detector...")
                n_components_pca = min(20, C, features_scaled.shape[0] // 3)
                pca_anomaly = PCA(n_components=n_components_pca)
                features_pca_anomaly = pca_anomaly.fit_transform(features_scaled)
                
                # Reconstruction error as anomaly score
                features_reconstructed = pca_anomaly.inverse_transform(features_pca_anomaly)
                reconstruction_errors = np.sum((features_scaled - features_reconstructed) ** 2, axis=1)
                threshold = np.percentile(reconstruction_errors, 95)
                
                trained_models['pca'] = {
                    'model': pca_anomaly,
                    'scaler': scaler,  # Need scaler for PCA method
                    'threshold': threshold,
                    'n_components': n_components_pca,
                    'use_pca': False
                }
                
                results['pca'] = {
                    'status': 'success',
                    'threshold': float(threshold),
                    'n_components': n_components_pca,
                    'explained_variance': float(pca_anomaly.explained_variance_ratio_.sum()),
                    'reconstruction_error_stats': {
                        'mean': float(np.mean(reconstruction_errors)),
                        'std': float(np.std(reconstruction_errors)),
                        'min': float(np.min(reconstruction_errors)),
                        'max': float(np.max(reconstruction_errors))
                    }
                }
                print(f"  ✅ PCA anomaly detector trained, components: {n_components_pca}, threshold: {threshold:.3f}")
            
            # 5. Autoencoder (if PyTorch available)
            if 'autoencoder' in methods and TORCH_AVAILABLE:
                print("🧠 Training Autoencoder...")
                autoencoder_result = self._train_autoencoder(features_scaled, project_id, **kwargs)
                results['autoencoder'] = autoencoder_result
                if autoencoder_result['status'] == 'success':
                    trained_models['autoencoder'] = autoencoder_result['model_info']
            
            # Save all models and scalers
            project_dir = os.path.join(self.projects_base_dir, project_id)
            models_dir = os.path.join(project_dir, "advanced_anomaly_models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Save sklearn models using joblib
            models_file = os.path.join(models_dir, f"{project_id}_advanced_models.joblib")
            model_data = {
                'models': trained_models,
                'scaler': scaler,
                'pca': pca,
                'feature_shape': features.shape,
                'methods_trained': methods,
                'training_params': kwargs
            }
            
            joblib.dump(model_data, models_file)
            print(f"💾 Advanced models saved to: {models_file}")
            
            # Save summary as JSON for easy inspection
            summary_file = os.path.join(models_dir, f"{project_id}_training_summary.json")
            summary_data = {
                'project_id': project_id,
                'feature_shape': features.shape,
                'methods_trained': methods,
                'training_results': results,
                'feature_stats': {
                    'original_dimensions': int(C),
                    'pca_dimensions': int(n_components),
                    'explained_variance': float(pca.explained_variance_ratio_.sum()),
                    'total_patches': int(features_flat.shape[0])
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            return {
                'status': 'success',
                'message': f'Advanced normal models trained successfully using {len(methods)} methods',
                'methods_trained': methods,
                'models_file': models_file,
                'summary_file': summary_file,
                'training_results': results
            }
            
        except Exception as e:
            print(f"❌ Error training advanced models: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _train_autoencoder(self, features: np.ndarray, project_id: str, **kwargs) -> Dict:
        """Train a simple autoencoder for anomaly detection"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Parameters
            input_dim = features.shape[1]
            hidden_dims = kwargs.get('ae_hidden_dims', [128, 64, 32, 64, 128])
            epochs = kwargs.get('ae_epochs', 50)
            batch_size = kwargs.get('ae_batch_size', 256)
            learning_rate = kwargs.get('ae_learning_rate', 0.001)
            
            # Create simple autoencoder architecture
            class SimpleAutoencoder(nn.Module):
                def __init__(self, input_dim, hidden_dims):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim
                    
                    # Encoder
                    for hidden_dim in hidden_dims[:len(hidden_dims)//2 + 1]:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        prev_dim = hidden_dim
                    
                    # Decoder
                    for hidden_dim in hidden_dims[len(hidden_dims)//2 + 1:]:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        prev_dim = hidden_dim
                    
                    # Output layer
                    layers.append(nn.Linear(prev_dim, input_dim))
                    
                    self.autoencoder = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.autoencoder(x)
            
            # Prepare data
            tensor_features = torch.FloatTensor(features).to(device)
            dataset = TensorDataset(tensor_features, tensor_features)  # Input = Target for autoencoder
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            model = SimpleAutoencoder(input_dim, hidden_dims).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            model.train()
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    reconstructed = model(batch_x)
                    loss = criterion(reconstructed, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            # Calculate reconstruction errors on training data for threshold
            model.eval()
            with torch.no_grad():
                reconstructed = model(tensor_features)
                reconstruction_errors = torch.sum((tensor_features - reconstructed) ** 2, dim=1)
                reconstruction_errors = reconstruction_errors.cpu().numpy()
            
            threshold = np.percentile(reconstruction_errors, 95)
            
            # Save model
            project_dir = os.path.join(self.projects_base_dir, project_id)
            models_dir = os.path.join(project_dir, "advanced_anomaly_models")
            model_path = os.path.join(models_dir, f"{project_id}_autoencoder.pth")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dims': hidden_dims
                },
                'threshold': threshold,
                'training_losses': losses
            }, model_path)
            
            print(f"  ✅ Autoencoder trained, final loss: {losses[-1]:.6f}, threshold: {threshold:.3f}")
            
            return {
                'status': 'success',
                'threshold': float(threshold),
                'final_loss': float(losses[-1]),
                'epochs': epochs,
                'model_path': model_path,
                'model_info': {
                    'model_path': model_path,
                    'threshold': threshold,
                    'input_dim': input_dim,
                    'hidden_dims': hidden_dims
                },
                'reconstruction_error_stats': {
                    'mean': float(np.mean(reconstruction_errors)),
                    'std': float(np.std(reconstruction_errors)),
                    'min': float(np.min(reconstruction_errors)),
                    'max': float(np.max(reconstruction_errors))
                }
            }
            
        except Exception as e:
            print(f"❌ Autoencoder training failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def detect_anomalies_advanced(
        self, 
        features: np.ndarray, 
        project_id: str, 
        methods: List[str] = None,
        ensemble_method: str = 'majority_vote'
    ) -> Dict:
        """
        Detect anomalies using trained advanced models
        
        Args:
            features: Image features to analyze [N, H, W, C]
            project_id: Project ID to load models from
            methods: Methods to use for detection (None = use all available)
            ensemble_method: How to combine multiple methods ['majority_vote', 'average_score', 'max_score']
        
        Returns:
            Dict with anomaly detection results
        """
        try:
            # Load trained models
            project_dir = os.path.join(self.projects_base_dir, project_id)
            models_dir = os.path.join(project_dir, "advanced_anomaly_models")
            models_file = os.path.join(models_dir, f"{project_id}_advanced_models.joblib")
            
            if not os.path.exists(models_file):
                return {
                    'status': 'error',
                    'message': 'No trained advanced models found for this project'
                }
            
            # Load models
            model_data = joblib.load(models_file)
            trained_models = model_data['models']
            scaler = model_data['scaler']
            pca = model_data['pca']
            
            if methods is None:
                methods = list(trained_models.keys())
            
            print(f"🎯 Detecting anomalies using methods: {methods}")
            print(f"🔬 Ensemble method: {ensemble_method}")
            
            # Prepare features
            N, H, W, C = features.shape
            features_flat = features.reshape(-1, C)
            features_scaled = scaler.transform(features_flat)
            features_pca = pca.transform(features_scaled)
            
            # Collect anomaly scores from each method
            method_results = {}
            all_scores = []
            
            for method in methods:
                if method not in trained_models:
                    print(f"⚠️ Method {method} not available in trained models")
                    continue
                
                model_info = trained_models[method]
                
                if method == 'ocsvm':
                    model = model_info['model']
                    threshold = model_info['threshold']
                    input_features = features_pca if model_info['use_pca'] else features_scaled
                    
                    scores = model.decision_function(input_features)
                    anomalies = scores < threshold
                    
                elif method == 'isolation_forest':
                    model = model_info['model']
                    input_features = features_pca if model_info['use_pca'] else features_scaled
                    
                    scores = model.decision_function(input_features)
                    anomalies = model.predict(input_features) == -1
                    
                elif method == 'elliptic':
                    model = model_info['model']
                    threshold = model_info['threshold']
                    input_features = features_pca if model_info['use_pca'] else features_scaled
                    
                    scores = model.mahalanobis(input_features)
                    anomalies = scores > threshold
                    
                elif method == 'pca':
                    pca_model = model_info['model']
                    threshold = model_info['threshold']
                    method_scaler = model_info['scaler']
                    
                    features_transformed = pca_model.transform(features_scaled)
                    features_reconstructed = pca_model.inverse_transform(features_transformed)
                    scores = np.sum((features_scaled - features_reconstructed) ** 2, axis=1)
                    anomalies = scores > threshold
                
                # Convert to anomaly probability (0-1 scale)
                scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
                
                method_results[method] = {
                    'scores': scores,
                    'scores_normalized': scores_normalized,
                    'anomalies': anomalies,
                    'threshold': model_info['threshold'],
                    'anomaly_count': int(np.sum(anomalies)),
                    'anomaly_percentage': float(np.mean(anomalies) * 100)
                }
                
                all_scores.append(scores_normalized)
                print(f"  📊 {method}: {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.1f}%)")
            
            if not method_results:
                return {
                    'status': 'error',
                    'message': 'No valid methods could be applied'
                }
            
            # Ensemble the results
            all_scores = np.array(all_scores)  # [n_methods, n_patches]
            
            if ensemble_method == 'majority_vote':
                # Count how many methods detected each patch as anomaly
                anomaly_votes = np.sum([result['anomalies'] for result in method_results.values()], axis=0)
                ensemble_anomalies = anomaly_votes > (len(method_results) / 2)
                ensemble_scores = anomaly_votes / len(method_results)
                
            elif ensemble_method == 'average_score':
                # Average normalized scores
                ensemble_scores = np.mean(all_scores, axis=0)
                ensemble_anomalies = ensemble_scores > 0.5
                
            elif ensemble_method == 'max_score':
                # Take maximum score (most pessimistic)
                ensemble_scores = np.max(all_scores, axis=0)
                ensemble_anomalies = ensemble_scores > 0.5
            
            # Reshape back to image dimensions
            ensemble_scores_map = ensemble_scores.reshape(N, H, W)
            ensemble_anomalies_map = ensemble_anomalies.reshape(N, H, W)
            
            return {
                'status': 'success',
                'ensemble_method': ensemble_method,
                'methods_used': list(method_results.keys()),
                'ensemble_scores': ensemble_scores_map,  # [N, H, W]
                'ensemble_anomalies': ensemble_anomalies_map,  # [N, H, W] boolean
                'method_results': method_results,
                'summary_stats': {
                    'total_patches': int(features_flat.shape[0]),
                    'ensemble_anomaly_count': int(np.sum(ensemble_anomalies)),
                    'ensemble_anomaly_percentage': float(np.mean(ensemble_anomalies) * 100),
                    'method_agreement': float(np.std([result['anomaly_percentage'] for result in method_results.values()]))
                }
            }
            
        except Exception as e:
            print(f"❌ Advanced anomaly detection failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _create_heatmap(self, score_map, target_size=(256, 256)):
        """Create a colored heatmap from anomaly scores"""
        try:
            import cv2
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
            import cv2
            import base64
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
    
    def _save_heatmap_image(self, heatmap: np.ndarray, project_id: str, filename: str, model_type: str = "advanced") -> str:
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
            import cv2
            cv2.imwrite(heatmap_path, heatmap)
            
            print(f"💾 Saved heatmap: {heatmap_path}")
            return heatmap_path
            
        except Exception as e:
            print(f"⚠️ Failed to save heatmap {filename}: {e}")
            return ""

# Create global instance
advanced_anomaly_service = AdvancedAnomalyDetector()