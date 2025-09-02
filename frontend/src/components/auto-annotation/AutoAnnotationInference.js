import React, { useState } from 'react';
import './AutoAnnotationInference.css';

function AutoAnnotationInference({ projectId, projectType, trainingStatus }) {
  const [inferenceState, setInferenceState] = useState({
    images: [],
    confidenceThreshold: 0.5,
    processing: false,
    results: [],
    dragActive: false
  });
  const [errors, setErrors] = useState({});

  const acceptedImageTypes = ['image/jpeg', 'image/jpg', 'image/png'];

  const handleImagesDrop = (e) => {
    e.preventDefault();
    setInferenceState(prev => ({ ...prev, dragActive: false }));
    
    const files = Array.from(e.dataTransfer.files);
    const imageFiles = files.filter(file => acceptedImageTypes.includes(file.type));
    
    if (imageFiles.length !== files.length) {
      setErrors({ images: 'Some files were not images and were ignored' });
    } else {
      setErrors(prev => ({ ...prev, images: '' }));
    }
    
    setInferenceState(prev => ({ ...prev, images: imageFiles }));
  };

  const handleImagesSelect = (e) => {
    const files = Array.from(e.target.files);
    setInferenceState(prev => ({ ...prev, images: files }));
    setErrors(prev => ({ ...prev, images: '' }));
  };

  const handleSettingChange = (setting, value) => {
    setInferenceState(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const removeImage = (index) => {
    setInferenceState(prev => ({
      ...prev,
      images: prev.images.filter((_, i) => i !== index)
    }));
  };

  const runAutoAnnotation = async () => {
    if (inferenceState.images.length === 0) {
      setErrors({ images: 'Please select at least one image' });
      return;
    }

    setInferenceState(prev => ({ ...prev, processing: true }));
    setErrors({});
    
    try {
      const formData = new FormData();
      
      // Add images
      inferenceState.images.forEach(image => {
        formData.append('images', image);
      });
      
      // Add settings
      formData.append('confidence_threshold', inferenceState.confidenceThreshold);

      const response = await fetch(`/api/auto-annotation/projects/${projectId}/annotate`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setInferenceState(prev => ({
          ...prev,
          results: result.results,
          images: [] // Clear images after successful processing
        }));
      } else {
        setErrors({ annotation: result.message || 'Auto-annotation failed' });
      }
    } catch (error) {
      console.error('Auto-annotation failed:', error);
      setErrors({ annotation: 'Network error. Please try again.' });
    } finally {
      setInferenceState(prev => ({ ...prev, processing: false }));
    }
  };

  const downloadResults = () => {
    // Create downloadable results
    const resultsData = {
      project_id: projectId,
      project_type: projectType,
      timestamp: new Date().toISOString(),
      results: inferenceState.results
    };
    
    const dataStr = JSON.stringify(resultsData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `auto_annotation_results_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const canRunInference = trainingStatus === 'completed';

  if (!canRunInference) {
    return (
      <div className="auto-annotation-inference">
        <div className="not-ready-state">
          <div className="not-ready-icon">⚠️</div>
          <h3>Model Not Ready</h3>
          <p>Please complete model training before running auto-annotation</p>
          <div className="status-info">
            Current training status: <strong>{trainingStatus}</strong>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="auto-annotation-inference">
      {/* Settings Section */}
      <div className="inference-settings">
        <h4>⚙️ Annotation Settings</h4>
        
        <div className="settings-grid">
          <div className="setting-item">
            <label>Confidence Threshold</label>
            <div className="threshold-control">
              <input
                type="range"
                value={inferenceState.confidenceThreshold}
                onChange={(e) => handleSettingChange('confidenceThreshold', parseFloat(e.target.value))}
                min="0.1"
                max="1.0"
                step="0.1"
              />
              <span className="threshold-value">{inferenceState.confidenceThreshold}</span>
            </div>
            <span className="setting-help">Minimum confidence for defect detection</span>
          </div>
        </div>
      </div>

      {/* Image Upload Section */}
      <div className="image-upload-section">
        <h4>📸 Upload Images for Annotation</h4>
        
        <div 
          className={`drop-zone ${inferenceState.dragActive ? 'active' : ''}`}
          onDrop={handleImagesDrop}
          onDragOver={(e) => e.preventDefault()}
          onDragEnter={() => setInferenceState(prev => ({ ...prev, dragActive: true }))}
          onDragLeave={() => setInferenceState(prev => ({ ...prev, dragActive: false }))}
        >
          <div className="drop-zone-content">
            <div className="drop-icon">🖼️</div>
            <p>Drag and drop images here, or click to select</p>
            <p className="file-types">Supported: JPG, JPEG, PNG</p>
            <input
              type="file"
              multiple
              accept=".jpg,.jpeg,.png"
              onChange={handleImagesSelect}
              className="hidden-file-input"
            />
          </div>
        </div>
        
        {errors.images && (
          <div className="error-message">{errors.images}</div>
        )}
        
        {/* Image Preview */}
        {inferenceState.images.length > 0 && (
          <div className="image-preview">
            <h5>Selected Images ({inferenceState.images.length})</h5>
            <div className="image-grid">
              {inferenceState.images.slice(0, 8).map((file, index) => (
                <div key={index} className="image-preview-item">
                  <img 
                    src={URL.createObjectURL(file)} 
                    alt={`Preview ${index}`}
                    onLoad={(e) => URL.revokeObjectURL(e.target.src)}
                  />
                  <div className="image-info">
                    <span className="image-name">{file.name}</span>
                    <span className="image-size">{formatFileSize(file.size)}</span>
                  </div>
                  <button 
                    className="remove-button"
                    onClick={() => removeImage(index)}
                  >
                    ×
                  </button>
                </div>
              ))}
              {inferenceState.images.length > 8 && (
                <div className="more-images">
                  +{inferenceState.images.length - 8} more
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Process Button */}
      <div className="process-section">
        {errors.annotation && (
          <div className="error-banner">
            ❌ {errors.annotation}
          </div>
        )}
        
        <button
          className="process-button"
          onClick={runAutoAnnotation}
          disabled={inferenceState.processing || inferenceState.images.length === 0}
        >
          {inferenceState.processing 
            ? '⏳ Processing...' 
            : `🤖 Run ${projectType === 'object_detection' ? 'Object Detection' : 'Segmentation'}`
          }
        </button>
        
        <div className="process-info">
          <p>
            Will process {inferenceState.images.length} images using trained {projectType === 'object_detection' ? 'YOLO' : 'SAM2'} model
          </p>
        </div>
      </div>

      {/* Results Section */}
      {inferenceState.results.length > 0 && (
        <div className="results-section">
          <div className="results-header">
            <h4>📊 Annotation Results</h4>
            <button className="download-button" onClick={downloadResults}>
              💾 Download Results
            </button>
          </div>
          
          <div className="results-grid">
            {inferenceState.results.map((result, index) => (
              <div key={index} className="result-item">
                <div className="result-header">
                  <h5>{result.filename}</h5>
                  <span className={`result-status ${result.status}`}>
                    {result.status === 'success' ? '✅' : '❌'} {result.status}
                  </span>
                </div>
                
                {result.status === 'success' ? (
                  <div className="result-content">
                    {result.annotated_image_base64 && (
                      <div className="annotated-image">
                        <img 
                          src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                          alt={`Annotated ${result.filename}`}
                        />
                      </div>
                    )}
                    
                    <div className="detection-summary">
                      {projectType === 'object_detection' ? (
                        <div>
                          <strong>Defects Found:</strong> {result.defect_detections?.length || 0}
                          {result.defect_detections?.map((detection, idx) => (
                            <div key={idx} className="detection-item">
                              {detection.class}: {(detection.confidence * 100).toFixed(1)}%
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div>
                          <strong>Masks Generated:</strong> {result.defect_masks?.length || 0}
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="result-error">
                    {result.message || 'Processing failed'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Workflow Info */}
      <div className="workflow-info">
        <h5>🔄 Simplified Auto-Annotation Workflow</h5>
        <div className="workflow-steps">
          <div className="workflow-step">
            <span className="step-number">1</span>
            <span className="step-text">Load trained {projectType === 'object_detection' ? 'YOLO' : 'SAM2'} model</span>
          </div>
          <div className="workflow-step">
            <span className="step-number">2</span>
            <span className="step-text">Apply model directly to full images</span>
          </div>
          <div className="workflow-step">
            <span className="step-number">3</span>
            <span className="step-text">
              Generate {projectType === 'object_detection' ? 'bounding boxes' : 'precise masks'} for detected defects
            </span>
          </div>
          <div className="workflow-step">
            <span className="step-number">4</span>
            <span className="step-text">Return annotated images with confidence scores</span>
          </div>
        </div>
        <div className="workflow-note">
          <p>📝 <strong>Note:</strong> This simplified pipeline matches training and inference data exactly, ensuring optimal model performance.</p>
        </div>
      </div>
    </div>
  );
}

export default AutoAnnotationInference;