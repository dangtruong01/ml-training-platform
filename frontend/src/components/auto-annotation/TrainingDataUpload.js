import React, { useState } from 'react';
import './TrainingDataUpload.css';

function TrainingDataUpload({ projectId, projectType, onDataUploaded }) {
  const [uploadState, setUploadState] = useState({
    images: [],
    annotations: [],
    uploading: false,
    dragActive: false
  });
  const [errors, setErrors] = useState({});

  const acceptedImageTypes = ['image/jpeg', 'image/jpg', 'image/png'];

  const handleImagesDrop = (e) => {
    e.preventDefault();
    setUploadState(prev => ({ ...prev, dragActive: false }));
    
    const files = Array.from(e.dataTransfer.files);
    const imageFiles = files.filter(file => acceptedImageTypes.includes(file.type));
    
    if (imageFiles.length !== files.length) {
      setErrors({ images: 'Some files were not images and were ignored' });
    } else {
      setErrors(prev => ({ ...prev, images: '' }));
    }
    
    setUploadState(prev => ({ ...prev, images: imageFiles }));
  };

  const handleImagesSelect = (e) => {
    const files = Array.from(e.target.files);
    setUploadState(prev => ({ ...prev, images: files }));
    setErrors(prev => ({ ...prev, images: '' }));
  };

  const handleAnnotationsSelect = (e) => {
    const files = Array.from(e.target.files);
    const txtFiles = files.filter(file => file.name.toLowerCase().endsWith('.txt'));
    
    if (txtFiles.length !== files.length) {
      setErrors({ annotations: 'Some files were not .txt files and were ignored' });
    } else {
      setErrors(prev => ({ ...prev, annotations: '' }));
    }
    
    setUploadState(prev => ({ ...prev, annotations: txtFiles }));
  };

  const validateUpload = () => {
    const newErrors = {};
    
    if (uploadState.images.length === 0) {
      newErrors.images = 'Please select at least one training image';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleUpload = async () => {
    if (!validateUpload()) {
      return;
    }

    setUploadState(prev => ({ ...prev, uploading: true }));
    
    try {
      const formData = new FormData();
      
      // Add images
      uploadState.images.forEach(image => {
        formData.append('training_images', image);
      });
      
      // Add annotation files if provided
      if (uploadState.annotations && uploadState.annotations.length > 0) {
        uploadState.annotations.forEach(annotation => {
          formData.append('annotation_files', annotation);
        });
      }
      
      // Add format
      formData.append('annotation_format', 'auto');

      const response = await fetch(`/api/auto-annotation/projects/${projectId}/upload-training-data`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        // Reset form
        setUploadState({
          images: [],
          annotations: [],
          uploading: false,
          dragActive: false
        });
        
        if (onDataUploaded) {
          onDataUploaded();
        }
      } else {
        setErrors({ upload: result.message || 'Upload failed' });
      }
    } catch (error) {
      console.error('Upload failed:', error);
      setErrors({ upload: 'Network error. Please try again.' });
    } finally {
      setUploadState(prev => ({ ...prev, uploading: false }));
    }
  };

  const removeImage = (index) => {
    setUploadState(prev => ({
      ...prev,
      images: prev.images.filter((_, i) => i !== index)
    }));
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="training-data-upload">
      {/* Image Upload Section */}
      <div className="upload-section">
        <h4>📸 Training Images</h4>
        <p className="section-description">
          Upload images that will be used to train your {projectType === 'object_detection' ? 'YOLO' : 'SAM2'} model
        </p>
        
        <div 
          className={`drop-zone ${uploadState.dragActive ? 'active' : ''}`}
          onDrop={handleImagesDrop}
          onDragOver={(e) => e.preventDefault()}
          onDragEnter={() => setUploadState(prev => ({ ...prev, dragActive: true }))}
          onDragLeave={() => setUploadState(prev => ({ ...prev, dragActive: false }))}
        >
          <div className="drop-zone-content">
            <div className="drop-icon">📁</div>
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
        {uploadState.images.length > 0 && (
          <div className="file-preview">
            <h5>Selected Images ({uploadState.images.length})</h5>
            <div className="image-grid">
              {uploadState.images.slice(0, 6).map((file, index) => (
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
              {uploadState.images.length > 6 && (
                <div className="more-images">
                  +{uploadState.images.length - 6} more
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Annotations Upload Section */}
      <div className="upload-section">
        <h4>🏷️ Annotations (Optional)</h4>
        <p className="section-description">
          {projectType === 'object_detection' 
            ? 'Upload YOLO format annotations (multiple .txt files)'
            : 'Upload COCO format annotations (JSON file with segmentation data)'
          }
        </p>
        
        <div className="file-input-container">
          <input
            type="file"
            accept={projectType === 'object_detection' ? '.txt' : '.json'}
            multiple={projectType === 'object_detection'}
            onChange={handleAnnotationsSelect}
            className="file-input"
            id="annotations-input"
          />
          <label htmlFor="annotations-input" className="file-input-label">
            {uploadState.annotations.length > 0 
              ? `${uploadState.annotations.length} annotation file${uploadState.annotations.length > 1 ? 's' : ''} selected`
              : `Choose ${projectType === 'object_detection' ? 'Annotation Files (.txt)' : 'Annotation File (.json)'}`
            }
          </label>
        </div>
        
        {errors.annotations && (
          <div className="error-message">{errors.annotations}</div>
        )}
        
        {/* Annotation Format Info */}
        <div className="format-info">
          <h5>📋 Expected Format:</h5>
          {projectType === 'object_detection' ? (
            <div className="format-example">
              <p><strong>YOLO Format (multiple .txt files):</strong></p>
              <code>class_id center_x center_y width height</code>
              <p className="format-note">
                Example: <code>0 0.5 0.3 0.2 0.1</code> (normalized coordinates)
              </p>
              <p className="format-note">
                Select all .txt annotation files at once. Each file should correspond to an image.
              </p>
            </div>
          ) : (
            <div className="format-example">
              <p><strong>COCO Format (JSON file):</strong></p>
              <code>{`{"annotations": [{"segmentation": [[x1,y1,x2,y2,...]], "bbox": [x,y,w,h]}]}`}</code>
              <p className="format-note">
                Standard COCO JSON with polygon segmentation data
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Upload Actions */}
      <div className="upload-actions">
        {errors.upload && (
          <div className="error-banner">
            ❌ {errors.upload}
          </div>
        )}
        
        <div className="action-buttons">
          <button
            className="upload-button"
            onClick={handleUpload}
            disabled={uploadState.uploading || uploadState.images.length === 0}
          >
            {uploadState.uploading ? '⏳ Uploading...' : '📤 Upload Training Data'}
          </button>
        </div>
        
        <div className="upload-summary">
          <span>Ready to upload: {uploadState.images.length} images</span>
          {uploadState.annotations.length > 0 && (
            <span>, {uploadState.annotations.length} annotation file{uploadState.annotations.length > 1 ? 's' : ''}</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default TrainingDataUpload;