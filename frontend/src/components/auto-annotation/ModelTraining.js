import React, { useState, useEffect, useCallback } from 'react';
import './ModelTraining.css';

function ModelTraining({ projectId, projectType, trainingStatus, onTrainingStatusChange }) {
  const [trainingParams, setTrainingParams] = useState({
    epochs: 50,
    batch_size: 16,
    image_size: 640,
    patience: 10,
    // SAM2 specific params
    few_shot_examples: 5,
    similarity_threshold: 0.8,
    mask_threshold: 0.5
  });
  
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState({
    status: trainingStatus || 'not_started',
    progress: 0,
    current_epoch: 0,
    total_epochs: 0,
    metrics: {},
    logs: []
  });
  const [errors, setErrors] = useState({});

  const fetchTrainingStatus = useCallback(async () => {
    try {
      const response = await fetch(`/api/auto-annotation/projects/${projectId}/training-status`);
      const data = await response.json();
      
      if (data.status === 'success') {
        console.log('Training status update:', data);
        const newStatus = data.training_status;
        setTrainingProgress(prev => ({
          ...prev,
          status: newStatus,
          progress: data.progress,
          current_epoch: data.current_epoch,
          total_epochs: data.total_epochs,
          metrics: data.metrics,
          logs: data.logs
        }));
        
        // Notify parent when training completes
        if (newStatus === 'completed' && onTrainingStatusChange) {
          onTrainingStatusChange();
        }
      }
    } catch (error) {
      console.error('Failed to fetch training status:', error);
    }
  }, [projectId]);

  // Poll training status
  useEffect(() => {
    let interval;
    
    if (trainingProgress.status === 'running') {
      interval = setInterval(() => {
        fetchTrainingStatus();
      }, 2000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [trainingProgress.status, projectId, fetchTrainingStatus]);

  const handleParamChange = (param, value) => {
    setTrainingParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const startTraining = async () => {
    setTraining(true);
    setErrors({});
    
    try {
      const formData = new FormData();
      
      if (projectType === 'object_detection') {
        formData.append('epochs', trainingParams.epochs);
        formData.append('batch_size', trainingParams.batch_size);
        formData.append('image_size', trainingParams.image_size);
        formData.append('patience', trainingParams.patience);
      } else {
        formData.append('few_shot_examples', trainingParams.few_shot_examples);
        formData.append('similarity_threshold', trainingParams.similarity_threshold);
        formData.append('mask_threshold', trainingParams.mask_threshold);
      }

      const response = await fetch(`/api/auto-annotation/projects/${projectId}/start-training`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        console.log('Training started successfully:', result);
        setTrainingProgress(prev => ({
          ...prev,
          status: 'running',
          total_epochs: trainingParams.epochs
        }));
      } else {
        console.log('Training start failed:', result);
        setErrors({ training: result.message || 'Failed to start training' });
      }
    } catch (error) {
      console.error('Training failed:', error);
      setErrors({ training: 'Network error. Please try again.' });
    } finally {
      setTraining(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'not_started': return '⏳';
      case 'running': return '🚀';
      case 'completed': return '✅';
      case 'failed': return '❌';
      default: return '⚪';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'not_started': return '#6c757d';
      case 'running': return '#007bff';
      case 'completed': return '#28a745';
      case 'failed': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const canStartTraining = trainingProgress.status === 'not_started' || trainingProgress.status === 'failed';

  return (
    <div className="model-training">
      {/* Training Status */}
      <div className="training-status">
        <div className="status-header">
          <span 
            className="status-indicator" 
            style={{ color: getStatusColor(trainingProgress.status) }}
          >
            {getStatusIcon(trainingProgress.status)} 
            {trainingProgress.status.replace('_', ' ').toUpperCase()}
          </span>
          
          {trainingProgress.status === 'running' && (
            <span className="epoch-info">
              Epoch {trainingProgress.current_epoch} / {trainingProgress.total_epochs}
            </span>
          )}
        </div>
        
        {trainingProgress.status === 'running' && (
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${Math.max(trainingProgress.progress, 2)}%` }}
              />
            </div>
            <span className="progress-text">
              {trainingProgress.progress > 0 ? `${trainingProgress.progress}%` : 'Starting...'}
            </span>
          </div>
        )}
      </div>

      {/* Training Parameters */}
      {canStartTraining && (
        <div className="training-params">
          <h5>🔧 Training Parameters</h5>
          
          {projectType === 'object_detection' ? (
            <div className="params-grid">
              <div className="param-item">
                <label>Epochs</label>
                <input
                  type="number"
                  value={trainingParams.epochs}
                  onChange={(e) => handleParamChange('epochs', parseInt(e.target.value))}
                  min="1"
                  max="1000"
                />
                <span className="param-help">Number of training iterations</span>
              </div>
              
              <div className="param-item">
                <label>Batch Size</label>
                <input
                  type="number"
                  value={trainingParams.batch_size}
                  onChange={(e) => handleParamChange('batch_size', parseInt(e.target.value))}
                  min="1"
                  max="64"
                />
                <span className="param-help">Images processed together</span>
              </div>
              
              <div className="param-item">
                <label>Image Size</label>
                <select
                  value={trainingParams.image_size}
                  onChange={(e) => handleParamChange('image_size', parseInt(e.target.value))}
                >
                  <option value={416}>416px</option>
                  <option value={512}>512px</option>
                  <option value={640}>640px (recommended)</option>
                  <option value={800}>800px</option>
                </select>
                <span className="param-help">Input image resolution</span>
              </div>
              
              <div className="param-item">
                <label>Patience</label>
                <input
                  type="number"
                  value={trainingParams.patience}
                  onChange={(e) => handleParamChange('patience', parseInt(e.target.value))}
                  min="1"
                  max="50"
                />
                <span className="param-help">Early stopping patience</span>
              </div>
            </div>
          ) : (
            <div className="params-grid">
              <div className="param-item">
                <label>Few-Shot Examples</label>
                <input
                  type="number"
                  value={trainingParams.few_shot_examples}
                  onChange={(e) => handleParamChange('few_shot_examples', parseInt(e.target.value))}
                  min="1"
                  max="20"
                />
                <span className="param-help">Examples per defect class</span>
              </div>
              
              <div className="param-item">
                <label>Similarity Threshold</label>
                <input
                  type="range"
                  value={trainingParams.similarity_threshold}
                  onChange={(e) => handleParamChange('similarity_threshold', parseFloat(e.target.value))}
                  min="0.1"
                  max="1.0"
                  step="0.1"
                />
                <span className="param-value">{trainingParams.similarity_threshold}</span>
                <span className="param-help">Feature similarity threshold</span>
              </div>
              
              <div className="param-item">
                <label>Mask Threshold</label>
                <input
                  type="range"
                  value={trainingParams.mask_threshold}
                  onChange={(e) => handleParamChange('mask_threshold', parseFloat(e.target.value))}
                  min="0.1"
                  max="1.0"
                  step="0.1"
                />
                <span className="param-value">{trainingParams.mask_threshold}</span>
                <span className="param-help">Mask confidence threshold</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Training Metrics */}
      {trainingProgress.status === 'running' && Object.keys(trainingProgress.metrics).length > 0 && (
        <div className="training-metrics">
          <h5>📊 Training Metrics</h5>
          <div className="metrics-grid">
            {Object.entries(trainingProgress.metrics).map(([key, value]) => (
              <div key={key} className="metric-item">
                <span className="metric-label">{key}</span>
                <span className="metric-value">
                  {typeof value === 'number' ? value.toFixed(4) : value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training Logs */}
      {trainingProgress.logs && trainingProgress.logs.length > 0 && (
        <div className="training-logs">
          <h5>📝 Training Logs</h5>
          <div className="logs-container">
            {trainingProgress.logs.slice(-10).map((log, index) => (
              <div key={index} className="log-entry">
                {log}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="training-actions">
        {errors.training && (
          <div className="error-banner">
            ❌ {errors.training}
          </div>
        )}
        
        {canStartTraining && (
          <button
            className="start-training-button"
            onClick={startTraining}
            disabled={training}
          >
            {training ? '⏳ Starting...' : `🚀 Start ${projectType === 'object_detection' ? 'YOLO' : 'SAM2'} Training`}
          </button>
        )}
        
        {trainingProgress.status === 'completed' && (
          <div className="success-banner">
            ✅ Model training completed successfully! Ready for auto-annotation.
          </div>
        )}
        
        {trainingProgress.status === 'failed' && (
          <div className="error-banner">
            ❌ Training failed. Please check your data and try again.
          </div>
        )}
      </div>

      {/* Training Info */}
      <div className="training-info">
        <h5>ℹ️ Training Process</h5>
        <div className="info-steps">
          <div className="info-step">
            <span className="step-number">1</span>
            <span className="step-text">
              Data preparation and validation
            </span>
          </div>
          <div className="info-step">
            <span className="step-number">2</span>
            <span className="step-text">
              {projectType === 'object_detection' 
                ? 'YOLO model training with your annotations'
                : 'SAM2 few-shot learning setup'
              }
            </span>
          </div>
          <div className="info-step">
            <span className="step-number">3</span>
            <span className="step-text">
              Model evaluation and optimization
            </span>
          </div>
          <div className="info-step">
            <span className="step-number">4</span>
            <span className="step-text">
              Model ready for automatic annotation
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelTraining;