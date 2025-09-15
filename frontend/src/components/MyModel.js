import React, { useState, useEffect } from 'react';
import './MyModel.css';

function MyModel() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [testImages, setTestImages] = useState([]);
  const [inferenceResults, setInferenceResults] = useState(null);
  const [inferenceLoading, setInferenceLoading] = useState(false);

  useEffect(() => {
    loadModels();
    
    // Auto-refresh every 10 seconds to show real-time progress
    const interval = setInterval(loadModels, 10000);
    
    // Cleanup interval on unmount
    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = (status) => {
    const statusColors = {
      'pending': 'bg-yellow-100 text-yellow-800',
      'submitted': 'bg-blue-100 text-blue-800',
      'running': 'bg-blue-100 text-blue-800 animate-pulse',
      'completed': 'bg-green-100 text-green-800',
      'failed': 'bg-red-100 text-red-800'
    };
    
    const statusText = {
      'pending': 'Pending',
      'submitted': 'Submitted',
      'running': 'Training',
      'completed': 'Completed',
      'failed': 'Failed'
    };
    
    return (
      <span className={`px-2 py-1 text-xs font-medium rounded-full ${statusColors[status] || 'bg-gray-100 text-gray-800'}`}>
        {statusText[status] || status}
      </span>
    );
  };

  const getAlgorithmDisplayName = (algorithm) => {
    const algorithmNames = {
      'isolation_forest': '🌲 Isolation Forest',
      'one_class_svm': '🎯 One-Class SVM',
      'local_outlier_factor': '📍 Local Outlier Factor',
      'autoencoder': '🧠 Autoencoder',
      'yolo_v8': '🚀 YOLO v8',
      'yolo_v11': '✨ YOLO v11',
      'rtdetr': '⚡ RT-DETR',
      'yolo_v8_seg': '🎭 YOLO v8 Seg',
      'sam2': '🔥 SAM 2.0',
      'unet': '🏥 U-Net'
    };
    
    return algorithmNames[algorithm] || `🤖 ${algorithm}`;
  };
  
  const formatDuration = (seconds) => {
    if (!seconds) return '-';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };
  
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '-';
    return new Date(timestamp).toLocaleString();
  };

  const loadModels = async () => {
    try {
      setLoading(true);
      // Load all models with algorithm information
      const response = await fetch('/api/models/');
      const data = await response.json();
      
      if (data.status === 'success' && data.models) {
        // Sort by most recent first
        const sortedModels = data.models.sort(
          (a, b) => new Date(b.created_at) - new Date(a.created_at)
        );
        setModels(sortedModels);
      } else {
        console.error('Failed to load models:', data);
        setModels([]);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
      setModels([]);
    } finally {
      setLoading(false);
    }
  };

  const handleTestImages = (event) => {
    const files = Array.from(event.target.files);
    setTestImages(files);
    setInferenceResults(null);
  };

  const runInference = async () => {
    if (!selectedModel || testImages.length === 0) {
      alert('Please select a model and upload test images');
      return;
    }

    try {
      setInferenceLoading(true);
      const formData = new FormData();
      formData.append('model_id', selectedModel.id);
      testImages.forEach(image => {
        formData.append('images', image);
      });

      const response = await fetch('/api/inference', {
        method: 'POST',
        body: formData
      });

      const results = await response.json();
      setInferenceResults(results);
    } catch (error) {
      console.error('Inference failed:', error);
      alert('Inference failed. Please try again.');
    } finally {
      setInferenceLoading(false);
    }
  };

  const downloadModel = async (modelId) => {
    try {
      const response = await fetch(`/api/models/${modelId}/download`);
      const blob = await response.blob();
      
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `model_${modelId}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    }
  };

  const deleteModel = async (modelId) => {
    if (!window.confirm('Are you sure you want to delete this model?')) {
      return;
    }

    try {
      const response = await fetch(`/api/models/${modelId}`, { method: 'DELETE' });
      const data = await response.json();
      
      if (data.status === 'success') {
        setModels(prev => prev.filter(m => m.model_id !== modelId));
        if (selectedModel && selectedModel.model_id === modelId) {
          setSelectedModel(null);
        }
        alert('Model deleted successfully');
      } else {
        alert('Failed to delete model');
      }
    } catch (error) {
      console.error('Delete failed:', error);
      alert('Delete failed. Please try again.');
    }
  };

  return (
    <div className="my-model-container">
      <div className="my-model-header">
        <h1>🎯 My Model</h1>
        <p>Manage, test, and deploy your trained models</p>
      </div>

      <div className="model-content">
        <div className="models-section">
          <h2>🤖 My Models</h2>
          {loading ? (
            <div className="loading">Loading models...</div>
          ) : models.length > 0 ? (
            <div className="models-grid">
              {models.map(model => (
                <div 
                  key={model.model_id} 
                  className={`model-card ${selectedModel?.model_id === model.model_id ? 'selected' : ''} status-${model.status}`}
                  onClick={() => setSelectedModel(model)}
                >
                  <div className="model-header">
                    <div className="model-title">
                      <h3>{model.project_name} ({model.model_id})</h3>
                      <span className="model-type-badge">
                        {model.model_type === 'anomaly' ? '🔍 Anomaly Detection' : 
                         model.model_type === 'detection' ? '📦 Object Detection' : 
                         model.model_type === 'classification' ? '🏷️ Classification' :
                         '🎯 Segmentation'}
                      </span>
                    </div>
                    <div className="status-section">
                      {getStatusBadge(model.status)}
                    </div>
                  </div>
                  
                  {/* Progress Bar */}
                  {(model.status === 'running' || model.status === 'submitted') && (
                    <div className="progress-section">
                      <div className="progress-info">
                        <span>Progress: {model.current_epoch || 0}/{model.total_epochs || 0} epochs</span>
                        <span>{model.progress || 0}%</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill"
                          style={{ width: `${model.progress || 0}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                  
                  <div className="model-info">
                    <div className="info-grid">
                      <div className="info-item">
                        <span className="label">Project ID:</span>
                        <span className="value">{model.project_id}</span>
                      </div>
                      <div className="info-item">
                        <span className="label">Algorithm:</span>
                        <span className="value algorithm-badge">{getAlgorithmDisplayName(model.algorithm)}</span>
                      </div>
                      <div className="info-item">
                        <span className="label">Started:</span>
                        <span className="value">{formatTimestamp(model.started_at)}</span>
                      </div>
                      {model.completed_at && (
                        <div className="info-item">
                          <span className="label">Completed:</span>
                          <span className="value">{formatTimestamp(model.completed_at)}</span>
                        </div>
                      )}
                      <div className="info-item">
                        <span className="label">Duration:</span>
                        <span className="value">{formatDuration(model.duration_seconds)}</span>
                      </div>
                      <div className="info-item">
                        <span className="label">Epochs:</span>
                        <span className="value">{model.total_epochs}</span>
                      </div>
                      <div className="info-item">
                        <span className="label">Files:</span>
                        <span className="value">{model.model_files?.length || 0} files</span>
                      </div>
                    </div>
                  </div>

                  <div className="model-actions">
                    {model.status === 'completed' && (
                      <button 
                        className="action-btn primary"
                        onClick={(e) => {
                          e.stopPropagation();
                          downloadModel(model.model_id);
                        }}
                      >
                        📥 Download
                      </button>
                    )}
                    {model.status === 'failed' && (
                      <button 
                        className="action-btn secondary"
                        onClick={(e) => {
                          e.stopPropagation();
                          // TODO: Implement retry training
                          alert('Retry training feature coming soon!');
                        }}
                      >
                        🔄 Retry
                      </button>
                    )}
                    <button 
                      className="action-btn danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteModel(model.model_id);
                      }}
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <p>No models found. Train your first model in the "My Training" tab!</p>
            </div>
          )}
        </div>

        <div className="testing-section">
          <h2>🧪 Model Testing</h2>
          {selectedModel ? (
            <div className="testing-interface">
              <div className="selected-model-info">
                <h3>Testing: {selectedModel.project_name} Model</h3>
                <p>Type: {selectedModel.project_type} | Algorithm: {getAlgorithmDisplayName(selectedModel.algorithm)} | Status: {selectedModel.status}</p>
              </div>

              <div className="test-upload">
                <label htmlFor="test-images">Upload Test Images:</label>
                <input
                  type="file"
                  id="test-images"
                  accept="image/*"
                  multiple
                  onChange={handleTestImages}
                />
                {testImages.length > 0 && (
                  <p>{testImages.length} images selected</p>
                )}
              </div>

              <button 
                className="run-inference-btn"
                onClick={runInference}
                disabled={testImages.length === 0 || inferenceLoading}
              >
                {inferenceLoading ? '🔄 Running...' : '🚀 Run Inference'}
              </button>

              {inferenceResults && (
                <div className="inference-results">
                  <h3>🎯 Inference Results</h3>
                  <div className="results-grid">
                    {inferenceResults.results?.map((result, index) => (
                      <div key={index} className="result-item">
                        <img 
                          src={result.image_url} 
                          alt={`Result ${index + 1}`}
                          className="result-image"
                        />
                        <div className="result-info">
                          <p>Detections: {result.detections?.length || 0}</p>
                          <p>Confidence: {result.confidence?.toFixed(2) || 'N/A'}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="no-model-selected">
              <p>Select a model from the list above to start testing</p>
            </div>
          )}
        </div>

        <div className="deployment-section">
          <h2>🚀 Model Deployment</h2>
          {selectedModel ? (
            <div className="deployment-options">
              <h3>Deploy: {selectedModel.project_name} Model</h3>
              <div className="deployment-buttons">
                <button className="deploy-btn">
                  🐳 Deploy to Docker
                </button>
                <button className="deploy-btn">
                  ☁️ Deploy to Cloud
                </button>
                <button className="deploy-btn">
                  📱 Generate Mobile SDK
                </button>
                <button className="deploy-btn">
                  🔗 Create API Endpoint
                </button>
              </div>
            </div>
          ) : (
            <div className="no-deployment">
              <p>Select a model to see deployment options</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default MyModel;