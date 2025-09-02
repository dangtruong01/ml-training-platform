import React, { useState } from 'react';

function AnomalyDetectionStep({ projectId, roiData, onStepComplete }) {
  const [detectionState, setDetectionState] = useState({
    method: 'mahalanobis',
    thresholdPercentile: 95.0,
    processing: false,
    results: null
  });
  const [errors, setErrors] = useState({});

  const handleSettingChange = (setting, value) => {
    setDetectionState(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const startAnomalyDetection = async () => {
    setDetectionState(prev => ({ ...prev, processing: true }));
    setErrors({});

    try {
      const formData = new FormData();
      formData.append('method', detectionState.method);
      formData.append('threshold_percentile', detectionState.thresholdPercentile);

      console.log(`🧠 Starting anomaly detection with ${detectionState.method} method`);
      
      const response = await fetch(`/api/auto-annotation/projects/${projectId}/detect-anomalies`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.status === 'success') {
        setDetectionState(prev => ({
          ...prev,
          results: result
        }));
        console.log('✅ Anomaly detection completed:', result);
      } else {
        setErrors({ detection: result.message || 'Anomaly detection failed' });
      }
    } catch (error) {
      console.error('Anomaly detection failed:', error);
      setErrors({ detection: 'Network error. Please try again.' });
    } finally {
      setDetectionState(prev => ({ ...prev, processing: false }));
    }
  };

  const handleProceedToNextStep = () => {
    if (onStepComplete) {
      onStepComplete(detectionState.results);
    }
  };

  const getAnomalyCount = () => {
    if (!detectionState.results) return 0;
    return detectionState.results.results.filter(r => 
      r.image_scores.anomaly_percentage > 5.0 // Consider >5% anomalous patches as anomalous
    ).length;
  };

  return (
    <div className="anomaly-detection-step">
      <div className="step-header">
        <h4>🧠 Step 2: DINOv2 Anomaly Detection</h4>
        <p>Use advanced AI to detect anomalous patterns in your ROI images without manual labeling.</p>
      </div>

      {/* Detection Settings */}
      <div className="detection-settings">
        <h5>⚙️ Detection Settings</h5>
        
        <div className="settings-grid">
          <div className="setting-item">
            <label>Detection Method</label>
            <select
              value={detectionState.method}
              onChange={(e) => handleSettingChange('method', e.target.value)}
              disabled={detectionState.processing}
            >
              <option value="mahalanobis">Mahalanobis Distance (Recommended)</option>
              <option value="euclidean">Euclidean Distance</option>
              <option value="cosine">Cosine Distance</option>
            </select>
            <span className="setting-help">
              {detectionState.method === 'mahalanobis' && "Best for correlated features"}
              {detectionState.method === 'euclidean' && "Simple distance in feature space"}
              {detectionState.method === 'cosine' && "Good for texture-based anomalies"}
            </span>
          </div>
          
          <div className="setting-item">
            <label>Anomaly Threshold</label>
            <div className="threshold-control">
              <input
                type="range"
                value={detectionState.thresholdPercentile}
                onChange={(e) => handleSettingChange('thresholdPercentile', parseFloat(e.target.value))}
                min="90.0"
                max="99.9"
                step="0.1"
                disabled={detectionState.processing}
              />
              <span className="threshold-value">{detectionState.thresholdPercentile}%</span>
            </div>
            <span className="setting-help">
              Percentile threshold - higher = fewer anomalies detected
            </span>
          </div>
        </div>
        
        <div className="method-explanation">
          <p>
            <strong>How it works:</strong> DINOv2 extracts features from your ROI images, 
            builds a model of "normal" patterns, then identifies regions that deviate significantly from normal.
          </p>
        </div>
      </div>

      {/* Action Button */}
      <div className="detection-actions">
        {errors.detection && (
          <div className="error-banner">
            ❌ {errors.detection}
          </div>
        )}
        
        <button
          className="start-detection-button"
          onClick={startAnomalyDetection}
          disabled={detectionState.processing}
        >
          {detectionState.processing 
            ? '🧠 Analyzing patterns...' 
            : '🔍 Start Anomaly Detection'
          }
        </button>
        
        {detectionState.processing && (
          <div className="processing-info">
            <p>🔄 This may take 30-60 seconds depending on the number of ROI images...</p>
            <div className="progress-indicator">
              <div className="progress-bar">
                <div className="progress-fill indeterminate"></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {detectionState.results && (
        <div className="detection-results">
          <div className="results-header">
            <h5>📊 Anomaly Detection Results</h5>
          </div>
          
          <div className="results-summary">
            <div className="summary-item">
              <span className="summary-label">Images Processed:</span>
              <span className="summary-value">{detectionState.results.total_images}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Anomalies Found:</span>
              <span className="summary-value warning">{getAnomalyCount()}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Detection Method:</span>
              <span className="summary-value">{detectionState.results.method}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Threshold Used:</span>
              <span className="summary-value">{detectionState.results.threshold_percentile}%</span>
            </div>
          </div>

          {/* Heatmap Grid */}
          <div className="heatmap-section">
            <h6>🔥 Anomaly Heatmaps</h6>
            <div className="heatmap-grid">
              {detectionState.results.results.slice(0, 8).map((result, index) => (
                <div key={index} className="heatmap-item">
                  <div className="heatmap-header">
                    <span className="image-name">{result.image_name.substring(13)}</span>
                    <span className={`anomaly-score ${result.image_scores.anomaly_percentage > 5.0 ? 'high' : 'low'}`}>
                      {result.image_scores.anomaly_percentage.toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="heatmap-container">
                    <img
                      src={`data:image/png;base64,${result.score_map_base64}`}
                      alt={`Heatmap ${index + 1}`}
                      className="heatmap-image"
                      title={`Max Score: ${result.image_scores.max_score.toFixed(3)}`}
                    />
                  </div>
                  
                  <div className="heatmap-stats">
                    <div className="stat">Max: {result.image_scores.max_score.toFixed(3)}</div>
                    <div className="stat">Patches: {result.image_scores.num_anomaly_patches}</div>
                  </div>
                </div>
              ))}
            </div>
            
            {detectionState.results.results.length > 8 && (
              <p className="more-results">
                ... and {detectionState.results.results.length - 8} more images
              </p>
            )}
          </div>

          {/* Global Statistics */}
          <div className="global-stats">
            <h6>📈 Global Statistics</h6>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Min Score:</span>
                <span className="stat-value">{detectionState.results.global_stats.min_score.toFixed(3)}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Max Score:</span>
                <span className="stat-value">{detectionState.results.global_stats.max_score.toFixed(3)}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Mean Score:</span>
                <span className="stat-value">{detectionState.results.global_stats.mean_score.toFixed(3)}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Std Dev:</span>
                <span className="stat-value">{detectionState.results.global_stats.std_score.toFixed(3)}</span>
              </div>
            </div>
          </div>

          {/* Next Step Button */}
          <div className="next-step-section">
            <div className="success-banner">
              ✅ Anomaly detection completed! Ready to generate annotation candidates.
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
                Proceed to Step 3 →
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Info Section */}
      <div className="step-info">
        <h5>ℹ️ About DINOv2 Anomaly Detection</h5>
        <div className="info-points">
          <div className="info-point">
            <span className="point-icon">🤖</span>
            <span className="point-text">Uses pretrained vision transformer features (no training required)</span>
          </div>
          <div className="info-point">
            <span className="point-icon">📊</span>
            <span className="point-text">Builds statistical model from your normal ROI images</span>
          </div>
          <div className="info-point">
            <span className="point-icon">🔥</span>
            <span className="point-text">Generates heatmaps showing anomalous regions</span>
          </div>
          <div className="info-point">
            <span className="point-icon">🎯</span>
            <span className="point-text">Works without manual defect labeling</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnomalyDetectionStep;