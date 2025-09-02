import React, { useState, useEffect } from 'react';
import './AnomalyDetectionWorkflow.css';

function DetectDefectsStep({ projectId, normalModelData, onStepComplete }) {
  const [detectionState, setDetectionState] = useState({
    loadingWorkflowStatus: true,
    workflowStatus: null,
    detectingAnomalies: false,
    anomalyResults: null,
    method: 'mahalanobis',
    thresholdPercentile: 95.0,
    detectionMethod: 'statistical',  // 'statistical' or 'advanced'
    ensembleMethod: 'majority_vote',  // 'majority_vote', 'average_score', 'max_score'
    advancedMethods: 'ocsvm,isolation_forest,elliptic,pca'
  });
  const [errors, setErrors] = useState({});

  // Load workflow status on component mount
  useEffect(() => {
    checkWorkflowStatus();
  }, [projectId]);

  const checkWorkflowStatus = async () => {
    if (!projectId) return;

    console.log('🔍 Checking workflow status for Stage 3...');
    setDetectionState(prev => ({ ...prev, loadingWorkflowStatus: true }));

    try {
      const response = await fetch(`/api/auto-annotation/projects/${projectId}/workflow-status`);
      const result = await response.json();

      if (result.status === 'success') {
        const workflowStatus = result.workflow_status;
        
        console.log('📊 Workflow status loaded:', workflowStatus);
        
        setDetectionState(prev => ({
          ...prev,
          workflowStatus: workflowStatus,
          loadingWorkflowStatus: false
        }));

        // If Stage 3 is already completed, load existing results
        if (workflowStatus.stage3_completed) {
          console.log('✅ Stage 3 already completed, loading existing results');
          // Could load existing Stage 3 results here if needed
        }
      } else {
        console.log('⚠️ Could not load workflow status:', result.message);
        setDetectionState(prev => ({ ...prev, loadingWorkflowStatus: false }));
      }
    } catch (error) {
      console.error('❌ Error checking workflow status:', error);
      setDetectionState(prev => ({ ...prev, loadingWorkflowStatus: false }));
    }
  };

  const handleSettingChange = (setting, value) => {
    setDetectionState(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const detectDefects = async () => {
    console.log('🎯 Starting defect detection using Stage 1 defective ROI results');
    setDetectionState(prev => ({ ...prev, detectingAnomalies: true }));
    setErrors({});

    try {
      const formData = new FormData();
      formData.append('method', detectionState.method);
      formData.append('threshold_percentile', detectionState.thresholdPercentile);
      formData.append('detection_method', detectionState.detectionMethod);
      
      if (detectionState.detectionMethod === 'advanced') {
        formData.append('ensemble_method', detectionState.ensembleMethod);
        formData.append('advanced_methods', detectionState.advancedMethods);
      }

      console.log(`🚀 Sending defect detection request with method: ${detectionState.method}`);

      const response = await fetch(`/api/auto-annotation/projects/${projectId}/detect-defects`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      console.log('📡 Defect detection response:', result);

      if (result.status === 'success') {
        setDetectionState(prev => ({
          ...prev,
          anomalyResults: result
        }));
        console.log(`✅ Defect detection completed: ${result.total_images} images processed`);
      } else {
        setErrors({ detection: result.message || 'Defect detection failed' });
      }
    } catch (error) {
      console.error('❌ Defect detection failed:', error);
      setErrors({ detection: `Detection error: ${error.message}` });
    } finally {
      setDetectionState(prev => ({ ...prev, detectingAnomalies: false }));
    }
  };

  const handleProceedToNextStep = () => {
    if (onStepComplete) {
      onStepComplete(detectionState.anomalyResults);
    }
  };

  const isReady = () => {
    return (
      !detectionState.loadingWorkflowStatus &&
      detectionState.workflowStatus?.stage1_completed &&
      normalModelData
    );
  };

  const getDefectiveImagesCount = () => {
    return detectionState.workflowStatus?.defective_images_count || 0;
  };

  return (
    <div className="detect-defects-step">
      <div className="step-header">
        <h4>🔍 Step 3: Detect Defects in Images</h4>
        <p>Apply the normal model to detect anomalies in defective images from Stage 1.</p>
      </div>

      {/* Workflow Status Banner */}
      {detectionState.loadingWorkflowStatus && (
        <div className="status-banner" style={{ padding: '15px', background: '#fff3cd', border: '1px solid #ffeaa7', borderRadius: '8px', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ fontSize: '16px' }}>🔍</div>
            <div>
              <div style={{ fontWeight: 'bold' }}>Loading Workflow Status...</div>
              <div style={{ fontSize: '14px', color: '#856404' }}>Checking for Stage 1 results and defective images</div>
            </div>
          </div>
        </div>
      )}

      {detectionState.workflowStatus && (
        <div className="workflow-status-banner" style={{ 
          padding: '15px', 
          background: isReady() ? '#d4edda' : '#f8d7da', 
          border: `1px solid ${isReady() ? '#c3e6cb' : '#f5c6cb'}`, 
          borderRadius: '8px', 
          marginBottom: '20px' 
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ fontSize: '16px' }}>{isReady() ? '✅' : '⚠️'}</div>
            <div>
              {isReady() ? (
                <>
                  <div style={{ fontWeight: 'bold', color: '#155724' }}>Ready for Defect Detection!</div>
                  <div style={{ fontSize: '14px', color: '#155724' }}>
                    Found {getDefectiveImagesCount()} defective images from Stage 1 
                    ({detectionState.workflowStatus.stage1_method}) and normal model from Stage 2
                  </div>
                </>
              ) : (
                <>
                  <div style={{ fontWeight: 'bold', color: '#721c24' }}>Prerequisites Not Met</div>
                  <div style={{ fontSize: '14px', color: '#721c24' }}>
                    {!detectionState.workflowStatus?.stage1_completed && 'Complete Stage 1 (ROI Extraction) first. '}
                    {!normalModelData && 'Complete Stage 2 (Build Normal Model) first.'}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Detection Settings */}
      <div className="detection-settings">
        <h5>⚙️ Anomaly Detection Settings</h5>
        <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
          
          <div className="setting-item">
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Detection Method
            </label>
            <select
              value={detectionState.detectionMethod}
              onChange={(e) => handleSettingChange('detectionMethod', e.target.value)}
              disabled={detectionState.detectingAnomalies}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            >
              <option value="statistical">Statistical (Fast)</option>
              <option value="advanced">Advanced Ensemble (Slower)</option>
            </select>
            <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              Statistical uses single method, Advanced combines multiple algorithms
            </p>
          </div>

          <div className="setting-item">
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Statistical Method
            </label>
            <select
              value={detectionState.method}
              onChange={(e) => handleSettingChange('method', e.target.value)}
              disabled={detectionState.detectingAnomalies || detectionState.detectionMethod === 'advanced'}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                opacity: detectionState.detectionMethod === 'advanced' ? 0.5 : 1
              }}
            >
              <option value="mahalanobis">Mahalanobis Distance</option>
              <option value="euclidean">Euclidean Distance</option>
              <option value="cosine">Cosine Distance</option>
            </select>
            <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              Distance metric for comparing against normal features
            </p>
          </div>

          <div className="setting-item">
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Anomaly Threshold
            </label>
            <div className="threshold-control">
              <input
                type="range"
                value={detectionState.thresholdPercentile}
                onChange={(e) => handleSettingChange('thresholdPercentile', parseFloat(e.target.value))}
                min="85"
                max="99"
                step="0.5"
                disabled={detectionState.detectingAnomalies}
                style={{ width: '70%' }}
              />
              <span style={{ marginLeft: '10px', fontWeight: 'bold' }}>
                {detectionState.thresholdPercentile}%
              </span>
            </div>
            <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              Percentile threshold for anomaly detection (higher = stricter)
            </p>
          </div>

          {detectionState.detectionMethod === 'advanced' && (
            <div className="setting-item">
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Ensemble Method
              </label>
              <select
                value={detectionState.ensembleMethod}
                onChange={(e) => handleSettingChange('ensembleMethod', e.target.value)}
                disabled={detectionState.detectingAnomalies}
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px'
                }}
              >
                <option value="majority_vote">Majority Vote</option>
                <option value="average_score">Average Score</option>
                <option value="max_score">Maximum Score</option>
              </select>
              <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                How to combine results from multiple detection algorithms
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Detection Button */}
      <div className="detection-actions" style={{ textAlign: 'center', marginBottom: '30px' }}>
        {errors.detection && (
          <div className="error-banner" style={{ marginBottom: '15px', padding: '10px', background: '#fee', border: '1px solid #fcc', borderRadius: '4px', color: '#c33' }}>
            ❌ {errors.detection}
          </div>
        )}

        <button
          className="detect-defects-button"
          onClick={detectDefects}
          disabled={detectionState.detectingAnomalies || !isReady()}
          style={{
            padding: '15px 30px',
            background: (detectionState.detectingAnomalies || !isReady()) ? '#6c757d' : '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: (detectionState.detectingAnomalies || !isReady()) ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
            boxShadow: '0 4px 8px rgba(220,53,69,0.3)',
            opacity: (!isReady() || detectionState.detectingAnomalies) ? 0.6 : 1
          }}
        >
          {detectionState.detectingAnomalies 
            ? '🔍 Detecting Defects...' 
            : `🚀 Detect Defects (${detectionState.detectionMethod})`
          }
        </button>

        {!isReady() && !detectionState.loadingWorkflowStatus && (
          <p style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
            Complete Stage 1 and Stage 2 first
          </p>
        )}

        {isReady() && (
          <p style={{ fontSize: '12px', color: '#dc3545', marginTop: '8px' }}>
            ✅ Ready to detect defects in {getDefectiveImagesCount()} images from Stage 1
          </p>
        )}

        {detectionState.detectingAnomalies && (
          <div className="detection-progress" style={{ marginTop: '15px' }}>
            <div className="progress-indicator">
              <div className="progress-bar" style={{ width: '100%', height: '4px', background: '#e9ecef', borderRadius: '2px', overflow: 'hidden' }}>
                <div className="progress-fill indeterminate" style={{ height: '100%', background: '#dc3545', animation: 'indeterminate 1.5s infinite linear' }}></div>
              </div>
            </div>
            <p style={{ marginTop: '8px', fontSize: '14px', color: '#666' }}>
              🧠 Analyzing defective images using normal model...
            </p>
          </div>
        )}
      </div>

      {/* Results Section */}
      {detectionState.anomalyResults && (
        <div className="detection-results">
          <div className="results-header">
            <h5>🎯 Defect Detection Results</h5>
          </div>
          
          <div className="results-summary" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '20px' }}>
            <div className="summary-card" style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#dc3545' }}>
                {detectionState.anomalyResults.total_images}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Total Images Processed</div>
            </div>
            
            <div className="summary-card" style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#28a745' }}>
                {detectionState.anomalyResults.results ? detectionState.anomalyResults.results.filter(r => r.status === 'success').length : 0}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Successful Detections</div>
            </div>
            
            <div className="summary-card" style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#6f42c1' }}>
                {detectionState.anomalyResults.method}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Detection Method</div>
            </div>
            
            <div className="summary-card" style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fd7e14' }}>
                {detectionState.anomalyResults.threshold_percentile}%
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Threshold Used</div>
            </div>
          </div>

          {/* Sample Results */}
          {detectionState.anomalyResults.results && detectionState.anomalyResults.results.length > 0 && (
            <div className="sample-results" style={{ marginBottom: '20px' }}>
              <h6>🔍 Sample Detection Results (First {Math.min(5, detectionState.anomalyResults.results.length)} Images)</h6>
              <div className="sample-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '10px' }}>
                {detectionState.anomalyResults.results.slice(0, 5).map((result, index) => (
                  <div key={index} className="sample-item" style={{ padding: '10px', background: '#fff', border: '1px solid #dee2e6', borderRadius: '4px' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                      {result.image_name && result.image_name.substring(0, 20)}...
                    </div>
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      Max Score: {result.image_scores?.max_score?.toFixed(3)} | 
                      Anomaly: {result.image_scores?.anomaly_percentage?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '10px', color: result.status === 'success' ? '#28a745' : '#dc3545', marginTop: '5px' }}>
                      Status: {result.status}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="action-buttons" style={{ display: 'flex', gap: '15px', justifyContent: 'center', alignItems: 'center' }}>
            <button
              className="complete-button"
              onClick={handleProceedToNextStep}
              style={{
                padding: '12px 24px',
                background: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 'bold'
              }}
            >
              ✅ Proceed to Step 4 →
            </button>
          </div>

          <div className="success-message" style={{ textAlign: 'center', marginTop: '15px', padding: '15px', background: '#d4edda', border: '1px solid #c3e6cb', borderRadius: '8px', color: '#155724' }}>
            <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '8px' }}>
              ✅ Defect Detection Complete!
            </div>
            <div style={{ fontSize: '14px' }}>
              🧠 Applied normal model to detect anomalies<br/>
              🎯 Generated heatmaps for defective regions<br/>
              📊 Ready for bounding box generation in Step 4
            </div>
          </div>
        </div>
      )}

      {/* Info Section */}
      <div className="step-info" style={{ marginTop: '30px' }}>
        <h5>ℹ️ About Defect Detection</h5>
        <div className="info-points">
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>🧠</span>
            <span style={{ fontSize: '14px' }}>Uses defective image ROIs from Stage 1 (no upload needed)</span>
          </div>
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>📊</span>
            <span style={{ fontSize: '14px' }}>Applies normal model from Stage 2 to detect anomalies</span>
          </div>
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>🎯</span>
            <span style={{ fontSize: '14px' }}>Generates heatmaps showing defective regions</span>
          </div>
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>⚙️</span>
            <span style={{ fontSize: '14px' }}>Statistical or advanced ensemble detection methods</span>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes indeterminate {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(300%); }
        }
        
        .progress-fill.indeterminate {
          animation: indeterminate 1.5s infinite linear;
        }
      `}</style>
    </div>
  );
}

export default DetectDefectsStep;