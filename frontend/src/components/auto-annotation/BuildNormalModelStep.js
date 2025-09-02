import React, { useState } from 'react';
import './AnomalyDetectionWorkflow.css';

function BuildNormalModelStep({ projectId, roiData, onStepComplete }) {
  const [modelState, setModelState] = useState({
    processing: false,
    results: null
  });
  const [errors, setErrors] = useState({});
  
  // Model selection state
  const [modelSettings, setModelSettings] = useState({
    modelType: 'dinov2',      // 'dinov2' or 'dinov3'
    detectionMethod: 'statistical',  // 'statistical' or 'advanced'
    advancedMethods: 'ocsvm,isolation_forest,elliptic,pca'  // comma-separated
  });

  const buildNormalModel = async () => {
    setModelState(prev => ({ ...prev, processing: true }));
    setErrors({});

    try {
      console.log(`🧠 Building normal model for project ${projectId} using ${modelSettings.modelType.toUpperCase()} + ${modelSettings.detectionMethod}`);
      
      // Create form data for API request
      const formData = new FormData();
      formData.append('model_type', modelSettings.modelType);
      formData.append('detection_method', modelSettings.detectionMethod);
      if (modelSettings.detectionMethod === 'advanced') {
        formData.append('advanced_methods', modelSettings.advancedMethods);
      }
      
      const response = await fetch(`/api/auto-annotation/projects/${projectId}/build-normal-model`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.status === 'success') {
        setModelState(prev => ({
          ...prev,
          results: result
        }));
        console.log('✅ Normal model built successfully:', result);
      } else {
        setErrors({ model: result.message || 'Failed to build normal model' });
      }
    } catch (error) {
      console.error('Normal model building failed:', error);
      setErrors({ model: 'Network error. Please try again.' });
    } finally {
      setModelState(prev => ({ ...prev, processing: false }));
    }
  };

  const handleProceedToNextStep = () => {
    if (onStepComplete) {
      onStepComplete(modelState.results);
    }
  };

  return (
    <div className="build-normal-model-step">
      <div className="step-header">
        <h4>🧠 Step 2: Build Normal Model</h4>
        <p>Extract {modelSettings.modelType.toUpperCase()} features from your normal ROI images and create a {modelSettings.detectionMethod} baseline for anomaly detection.</p>
      </div>
      
      {/* Model Configuration */}
      <div className="model-configuration">
        <h5>⚙️ Model Configuration</h5>
        
        {/* Feature Extractor Selection */}
        <div className="config-section">
          <label className="config-label">🔧 Feature Extractor Model:</label>
          <div className="radio-group">
            <label className="radio-option">
              <input
                type="radio"
                name="modelType"
                value="dinov2"
                checked={modelSettings.modelType === 'dinov2'}
                onChange={(e) => setModelSettings(prev => ({ ...prev, modelType: e.target.value }))}
                disabled={modelState.processing}
              />
              <span className="radio-label">DINOv2 (Stable & Fast)</span>
            </label>
            <label className="radio-option">
              <input
                type="radio"
                name="modelType"
                value="dinov3"
                checked={modelSettings.modelType === 'dinov3'}
                onChange={(e) => setModelSettings(prev => ({ ...prev, modelType: e.target.value }))}
                disabled={modelState.processing}
              />
              <span className="radio-label">DINOv3 (Latest & More Accurate)</span>
            </label>
          </div>
        </div>
        
        {/* Detection Method Selection */}
        <div className="config-section">
          <label className="config-label">🧪 Anomaly Detection Method:</label>
          <div className="radio-group">
            <label className="radio-option">
              <input
                type="radio"
                name="detectionMethod"
                value="statistical"
                checked={modelSettings.detectionMethod === 'statistical'}
                onChange={(e) => setModelSettings(prev => ({ ...prev, detectionMethod: e.target.value }))}
                disabled={modelState.processing}
              />
              <span className="radio-label">Statistical (Simple & Fast)</span>
              <span className="method-desc">Uses mean and covariance for anomaly detection</span>
            </label>
            <label className="radio-option">
              <input
                type="radio"
                name="detectionMethod"
                value="advanced"
                checked={modelSettings.detectionMethod === 'advanced'}
                onChange={(e) => setModelSettings(prev => ({ ...prev, detectionMethod: e.target.value }))}
                disabled={modelState.processing}
              />
              <span className="radio-label">Advanced ML (More Accurate)</span>
              <span className="method-desc">Uses One-Class SVM, Isolation Forest, PCA, etc.</span>
            </label>
          </div>
        </div>
        
        {/* Advanced Methods Configuration */}
        {modelSettings.detectionMethod === 'advanced' && (
          <div className="config-section">
            <label className="config-label">🎯 Advanced Methods:</label>
            <div className="methods-selection">
              <div className="methods-info">
                <p>Multiple ML algorithms will be trained and combined using ensemble voting:</p>
              </div>
              <textarea
                className="methods-input"
                value={modelSettings.advancedMethods}
                onChange={(e) => setModelSettings(prev => ({ ...prev, advancedMethods: e.target.value }))}
                placeholder="ocsvm,isolation_forest,elliptic,pca,autoencoder"
                disabled={modelState.processing}
                rows={2}
              />
              <div className="methods-help">
                Available: ocsvm, isolation_forest, elliptic, pca, autoencoder (comma-separated)
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Model Building Info */}
      <div className="model-info">
        <h5>🔧 What This Step Does:</h5>
        <div className="info-points">
          <div className="info-point">
            <span className="point-icon">🔍</span>
            <span className="point-text">Extract {modelSettings.modelType.toUpperCase()} features from normal ROI images</span>
          </div>
          <div className="info-point">
            <span className="point-icon">📊</span>
            <span className="point-text">
              {modelSettings.detectionMethod === 'statistical' 
                ? 'Compute statistical model (mean, covariance) of normal patterns'
                : 'Train multiple ML models (SVM, Isolation Forest, PCA, etc.)'}
            </span>
          </div>
          <div className="info-point">
            <span className="point-icon">💾</span>
            <span className="point-text">Cache the model for anomaly detection in Stage 3</span>
          </div>
          <div className="info-point">
            <span className="point-icon">⚡</span>
            <span className="point-text">No training required - uses pretrained {modelSettings.modelType.toUpperCase()} features</span>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="model-actions">
        {errors.model && (
          <div className="error-banner">
            ❌ {errors.model}
          </div>
        )}
        
        <button
          className="start-model-building-button"
          onClick={buildNormalModel}
          disabled={modelState.processing}
        >
          {modelState.processing 
            ? '🧠 Building Normal Model...' 
            : '🚀 Build Normal Model'
          }
        </button>
        
        {modelState.processing && (
          <div className="processing-info">
            <p>🔄 Extracting features and computing statistics from your normal ROI images...</p>
            <div className="progress-indicator">
              <div className="progress-bar">
                <div className="progress-fill indeterminate"></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {modelState.results && (
        <div className="model-results">
          <div className="results-header">
            <h5>✅ Normal Model Built Successfully</h5>
          </div>
          
          <div className="results-summary">
            <div className="summary-item">
              <span className="summary-label">Normal Images:</span>
              <span className="summary-value">{modelState.results.normal_images_count}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Feature Dimensions:</span>
              <span className="summary-value">{modelState.results.feature_dimensions}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Model Type:</span>
              <span className="summary-value">{modelState.results.model_type?.toUpperCase() || 'DINOv2'} {modelState.results.detection_method?.charAt(0).toUpperCase() + modelState.results.detection_method?.slice(1) || 'Statistical'}</span>
            </div>
            {modelState.results.advanced_methods && (
              <div className="summary-item">
                <span className="summary-label">ML Methods:</span>
                <span className="summary-value">{modelState.results.advanced_methods.join(', ')}</span>
              </div>
            )}
            <div className="summary-item">
              <span className="summary-label">Status:</span>
              <span className="summary-value success">Ready</span>
            </div>
          </div>

          {/* Model Statistics */}
          <div className="model-stats">
            <h6>📈 Model Statistics</h6>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Mean Feature Norm:</span>
                <span className="stat-value">{modelState.results.stats?.mean_norm?.toFixed(3) || 'N/A'}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Feature Variance:</span>
                <span className="stat-value">{modelState.results.stats?.variance?.toFixed(3) || 'N/A'}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Total Patches:</span>
                <span className="stat-value">{modelState.results.stats?.total_patches || 'N/A'}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Cache Status:</span>
                <span className="stat-value success">Saved</span>
              </div>
            </div>
          </div>

          {/* Next Step Button */}
          <div className="next-step-section">
            <div className="success-banner">
              🎉 Normal model is ready! You can now proceed to detect defects in Stage 3.
              <button
                className="next-step-button"
                onClick={handleProceedToNextStep}
                style={{ 
                  marginLeft: '15px', 
                  padding: '8px 16px', 
                  background: '#28a745', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: '4px', 
                  cursor: 'pointer' 
                }}
              >
                Proceed to Stage 3 →
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Usage Instructions */}
      <div className="step-info">
        <h5>ℹ️ About Normal Model Building</h5>
        <div className="info-points">
          <div className="info-point">
            <span className="point-icon">📚</span>
            <span className="point-text">Uses your normal ROI images as the foundation for anomaly detection</span>
          </div>
          <div className="info-point">
            <span className="point-icon">🧠</span>
            <span className="point-text">{modelSettings.modelType.toUpperCase()} extracts semantic features without requiring training</span>
          </div>
          <div className="info-point">
            <span className="point-icon">📊</span>
            <span className="point-text">{modelSettings.detectionMethod === 'statistical' ? 'Statistical model captures what "normal" looks like' : 'Advanced ML models learn complex normal patterns'}</span>
          </div>
          <div className="info-point">
            <span className="point-icon">🎯</span>
            <span className="point-text">Ready to detect deviations in defective images (Stage 3)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default BuildNormalModelStep;