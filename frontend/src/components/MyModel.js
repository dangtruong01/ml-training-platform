import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import './MyModel.css';

function MyModel() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [testImages, setTestImages] = useState([]);
  const [inferenceResults, setInferenceResults] = useState(null);
  const [inferenceLoading, setInferenceLoading] = useState(false);
  const [openDropdown, setOpenDropdown] = useState(null);
  const [downloadingModel, setDownloadingModel] = useState(null);
  const [retryingModel, setRetryingModel] = useState(null);

  useEffect(() => {
    loadModels();

    // Auto-refresh every 10 seconds to show real-time progress
    const interval = setInterval(loadModels, 10000);

    // Close dropdown when clicking outside
    const handleClickOutside = (event) => {
      if (!event.target.closest('.download-dropdown')) {
        setOpenDropdown(null);
      }
    };

    document.addEventListener('click', handleClickOutside);

    // Cleanup interval and event listener on unmount
    return () => {
      clearInterval(interval);
      document.removeEventListener('click', handleClickOutside);
    };
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

  const getDownloadOptions = (model) => {
    const baseOptions = [
      {
        id: 'all',
        label: '📦 Complete Package (PyTorch + ONNX)',
        description: 'All model files including both formats',
        icon: '📦'
      }
    ];

    // Add ONNX-specific option for compatible model types (only YOLO-based models)
    const supportedForOnnx = ['object_detection', 'segmentation'];
    if (supportedForOnnx.includes(model.project_type)) {
      baseOptions.push({
        id: 'onnx',
        label: '🚀 ONNX Models Only',
        description: 'Optimized for deployment and cross-platform use',
        icon: '🚀'
      });
    }

    return baseOptions;
  };

  const downloadModel = async (modelId, format = 'all') => {
    try {
      setDownloadingModel(`${modelId}-${format}`);
      let endpoint;
      let filename;

      switch (format) {
        case 'onnx':
          endpoint = `/api/models/${modelId}/download-onnx`;
          filename = `model_${modelId}_onnx.zip`;
          break;
        case 'pytorch':
        case 'all':
        default:
          endpoint = `/api/models/${modelId}/download`;
          filename = `model_${modelId}.zip`;
          break;
      }

      const response = await fetch(endpoint);

      if (!response.ok) {
        throw new Error(`Download failed: ${response.status} ${response.statusText}`);
      }

      const blob = await response.blob();

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      alert(`Download failed: ${error.message}`);
    } finally {
      setDownloadingModel(null);
    }
  };

  const DownloadDropdown = ({ model }) => {
    const options = getDownloadOptions(model);
    const isOpen = openDropdown === model.model_id;
    const isDownloading = downloadingModel && downloadingModel.startsWith(model.model_id);
    const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0, flip: false });
    const dropdownRef = useRef(null);

    const handleDropdownClick = (e) => {
      e.stopPropagation();

      if (!isOpen) {
        // Calculate dropdown position relative to viewport
        const button = e.currentTarget;
        const rect = button.getBoundingClientRect();
        const windowHeight = window.innerHeight;
        const windowWidth = window.innerWidth;
        const spaceBelow = windowHeight - rect.bottom;
        const dropdownHeight = 200; // Approximate dropdown height
        const dropdownWidth = 240; // Dropdown min width

        // Determine if we should flip up
        const shouldFlip = spaceBelow < dropdownHeight && rect.top > dropdownHeight;

        // Calculate position
        let top = shouldFlip ? rect.top - dropdownHeight - 4 : rect.bottom + 4;
        let left = rect.left;

        // Ensure dropdown doesn't go off-screen horizontally
        if (left + dropdownWidth > windowWidth) {
          left = windowWidth - dropdownWidth - 10;
        }
        if (left < 10) {
          left = 10;
        }

        // Ensure dropdown doesn't go off-screen vertically
        if (top < 10) {
          top = rect.bottom + 4;
        }
        if (top + dropdownHeight > windowHeight - 10) {
          top = windowHeight - dropdownHeight - 10;
        }

        setDropdownPosition({ top, left, flip: shouldFlip });
      }

      setOpenDropdown(isOpen ? null : model.model_id);
    };

    const handleOptionClick = (e, format) => {
      e.stopPropagation();
      downloadModel(model.model_id, format);
      setOpenDropdown(null);
    };

    return (
      <>
        <div className="download-dropdown">
          <button
            className="action-btn primary dropdown-trigger"
            onClick={handleDropdownClick}
            disabled={isDownloading}
          >
            {isDownloading ? '⏳ Downloading...' : `📥 Download ${isOpen ? '▲' : '▼'}`}
          </button>
        </div>

        {isOpen && createPortal(
          <div
            className="dropdown-menu-portal"
            style={{
              position: 'fixed',
              top: `${dropdownPosition.top}px`,
              left: `${dropdownPosition.left}px`,
              zIndex: 10000,
              background: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 15px 35px rgba(0, 0, 0, 0.25)',
              overflow: 'hidden',
              animation: 'dropdown-appear 0.2s ease',
              minWidth: '240px',
            }}
          >
            {options.map(option => {
              const isThisOptionDownloading = downloadingModel === `${model.model_id}-${option.id}`;
              return (
                <button
                  key={option.id}
                  className="dropdown-option"
                  data-format={option.id}
                  onClick={(e) => handleOptionClick(e, option.id)}
                  disabled={isDownloading}
                >
                  <div className="option-header">
                    <span className="option-icon">
                      {isThisOptionDownloading ? '⏳' : option.icon}
                    </span>
                    <span className="option-label">
                      {isThisOptionDownloading ? 'Downloading...' : option.label}
                    </span>
                  </div>
                  <div className="option-description">{option.description}</div>
                </button>
              );
            })}
          </div>,
          document.body
        )}
      </>
    );
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

  const retryTraining = async (modelId) => {
    if (!window.confirm('Are you sure you want to retry training for this model?')) {
      return;
    }

    setRetryingModel(modelId);

    try {
      const response = await fetch(`/api/models/${modelId}/retry`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      const data = await response.json();

      if (data.status === 'success') {
        alert(`Training restarted successfully! New task ID: ${data.task_id}`);
        // Refresh models to show the new training job
        loadModels();
      } else {
        alert(`Failed to retry training: ${data.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Retry failed:', error);
      alert('Retry failed. Please try again.');
    } finally {
      setRetryingModel(null);
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
                      <span className={`model-type-badge ${model.project_type || 'other'}`}>
                        {model.project_type === 'anomaly_detection' ? '🔍 Anomaly Detection' : 
                         model.project_type === 'object_detection' ? '📦 Object Detection' : 
                         model.project_type === 'segmentation' ? '🎯 Segmentation' :
                         model.model_type === 'classification' ? '🏷️ Classification' :
                         '🤖 Other'}
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
                      <DownloadDropdown model={model} />
                    )}
                    {model.status === 'failed' && (
                      <button
                        className="action-btn secondary"
                        disabled={retryingModel === model.model_id}
                        onClick={(e) => {
                          e.stopPropagation();
                          retryTraining(model.model_id);
                        }}
                      >
                        {retryingModel === model.model_id ? '⏳ Retrying...' : '🔄 Retry'}
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