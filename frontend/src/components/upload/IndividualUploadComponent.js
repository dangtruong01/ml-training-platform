import React, { useState, useCallback } from 'react';
import './Upload.css';

const IndividualUploadComponent = ({ 
  projectId, 
  algorithm, 
  projectType,
  onUploadSuccess, 
  onUploadError,
  disabled = false 
}) => {
  const [trainingImages, setTrainingImages] = useState([]);
  const [annotationFiles, setAnnotationFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const getExpectedAnnotationFormat = () => {
    const formats = {
      'object_detection': {
        extension: '.txt',
        description: 'YOLO format text files',
        example: 'image1.txt (one per image)'
      },
      'segmentation': {
        extension: '.txt,.json',
        description: 'Segmentation masks or polygon files',
        example: 'masks or COCO JSON'
      },
      'anomaly_detection': {
        extension: 'none',
        description: 'No annotation files needed',
        example: 'Just upload normal images'
      }
    };
    
    return formats[projectType] || formats['object_detection'];
  };

  const annotationFormat = getExpectedAnnotationFormat();

  const validateFiles = useCallback((files, type) => {
    const errors = [];
    const validFiles = [];

    for (const file of files) {
      // Check file size (50MB per file limit)
      if (file.size > 50 * 1024 * 1024) {
        errors.push(`${file.name}: File too large (max 50MB)`);
        continue;
      }

      if (type === 'images') {
        // Validate image files
        if (!file.type.startsWith('image/')) {
          errors.push(`${file.name}: Not a valid image file`);
          continue;
        }
        
        const validExtensions = ['.jpg', '.jpeg', '.png', '.bmp'];
        const hasValidExtension = validExtensions.some(ext => 
          file.name.toLowerCase().endsWith(ext)
        );
        
        if (!hasValidExtension) {
          errors.push(`${file.name}: Invalid image format. Use JPG, PNG, or BMP`);
          continue;
        }
      } else if (type === 'annotations') {
        // Validate annotation files
        if (projectType === 'object_detection') {
          if (!file.name.toLowerCase().endsWith('.txt')) {
            errors.push(`${file.name}: YOLO annotations must be .txt files`);
            continue;
          }
        } else if (projectType === 'segmentation') {
          const validAnnotations = ['.txt', '.json', '.xml'];
          const hasValidAnnotation = validAnnotations.some(ext => 
            file.name.toLowerCase().endsWith(ext)
          );
          
          if (!hasValidAnnotation) {
            errors.push(`${file.name}: Invalid annotation format`);
            continue;
          }
        }
      }

      validFiles.push(file);
    }

    return { validFiles, errors };
  }, [projectType]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e, type) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (disabled) return;

    const files = [...e.dataTransfer.files];
    handleFileSelection(files, type);
  }, [disabled, validateFiles]);

  const handleFileSelection = (files, type) => {
    const { validFiles, errors } = validateFiles(files, type);
    
    if (errors.length > 0) {
      onUploadError(errors.join('\n'));
    }

    if (validFiles.length > 0) {
      if (type === 'images') {
        setTrainingImages(prev => [...prev, ...validFiles]);
      } else if (type === 'annotations') {
        setAnnotationFiles(prev => [...prev, ...validFiles]);
      }
    }
  };

  const removeFile = (fileName, type) => {
    if (type === 'images') {
      setTrainingImages(prev => prev.filter(f => f.name !== fileName));
    } else if (type === 'annotations') {
      setAnnotationFiles(prev => prev.filter(f => f.name !== fileName));
    }
  };

  const handleUpload = async () => {
    if (uploading) return;

    // Validation
    if (trainingImages.length === 0) {
      onUploadError('Please select at least one training image');
      return;
    }

    if (projectType === 'object_detection' && annotationFiles.length === 0) {
      onUploadError('Object detection requires annotation files (.txt)');
      return;
    }

    setUploading(true);
    
    try {
      const formData = new FormData();
      
      // Add training images
      trainingImages.forEach(file => {
        formData.append('training_images', file);
      });
      
      // Add annotation files if any
      if (annotationFiles.length > 0) {
        annotationFiles.forEach(file => {
          formData.append('annotation_files', file);
        });
      }

      const response = await fetch(`/api/projects/${projectId}/upload-training-data`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (response.ok && result.status === 'success') {
        onUploadSuccess({
          method: 'individual',
          trainingImages: trainingImages.length,
          annotationFiles: annotationFiles.length,
          files: result.uploaded_files || []
        });
      } else {
        throw new Error(result.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      onUploadError(error.message);
    } finally {
      setUploading(false);
    }
  };

  const clearAllFiles = () => {
    setTrainingImages([]);
    setAnnotationFiles([]);
  };

  return (
    <div className="individual-upload-component">
      <div className="upload-section">
        <h4>📁 Upload Individual Files</h4>
        <p>Upload your images and annotation files separately - we'll organize them for training</p>

        {/* Training Images Upload */}
        <div className="file-upload-group">
          <h5>📷 Training Images</h5>
          <div 
            className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={(e) => handleDrop(e, 'images')}
          >
            <div className="drop-zone-content">
              <span className="upload-icon">📷</span>
              <p>Drop training images here or</p>
              <label className="file-input-label">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => handleFileSelection([...e.target.files], 'images')}
                  disabled={disabled}
                  style={{ display: 'none' }}
                />
                <span className="browse-button">Browse Images</span>
              </label>
            </div>
            <p className="format-hint">Accepts: JPG, PNG, BMP • Max 50MB per file</p>
          </div>

          {trainingImages.length > 0 && (
            <div className="selected-files">
              <h6>Selected Images ({trainingImages.length})</h6>
              <div className="files-list">
                {trainingImages.map((file, index) => (
                  <div key={index} className="file-item">
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">{(file.size / 1024).toFixed(1)}KB</span>
                    <button 
                      className="remove-file"
                      onClick={() => removeFile(file.name, 'images')}
                      disabled={uploading}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Annotation Files Upload (if needed) */}
        {projectType !== 'anomaly_detection' && (
          <div className="file-upload-group">
            <h5>📄 Annotation Files</h5>
            <div 
              className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={(e) => handleDrop(e, 'annotations')}
            >
              <div className="drop-zone-content">
                <span className="upload-icon">📄</span>
                <p>Drop annotation files here or</p>
                <label className="file-input-label">
                  <input
                    type="file"
                    accept={annotationFormat.extension}
                    multiple
                    onChange={(e) => handleFileSelection([...e.target.files], 'annotations')}
                    disabled={disabled}
                    style={{ display: 'none' }}
                  />
                  <span className="browse-button">Browse Annotations</span>
                </label>
              </div>
              <p className="format-hint">
                Format: {annotationFormat.description} • Example: {annotationFormat.example}
              </p>
            </div>

            {annotationFiles.length > 0 && (
              <div className="selected-files">
                <h6>Selected Annotations ({annotationFiles.length})</h6>
                <div className="files-list">
                  {annotationFiles.map((file, index) => (
                    <div key={index} className="file-item">
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">{(file.size / 1024).toFixed(1)}KB</span>
                      <button 
                        className="remove-file"
                        onClick={() => removeFile(file.name, 'annotations')}
                        disabled={uploading}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Upload Actions */}
        <div className="upload-actions">
          <div className="files-summary">
            <span>Images: {trainingImages.length}</span>
            {projectType !== 'anomaly_detection' && (
              <span>Annotations: {annotationFiles.length}</span>
            )}
          </div>
          
          <div className="action-buttons">
            <button
              className="clear-button"
              onClick={clearAllFiles}
              disabled={uploading || (trainingImages.length === 0 && annotationFiles.length === 0)}
            >
              🗑️ Clear All
            </button>
            <button
              className="upload-button primary"
              onClick={handleUpload}
              disabled={uploading || trainingImages.length === 0 || disabled}
            >
              {uploading ? '📤 Uploading...' : '🚀 Upload Files'}
            </button>
          </div>
        </div>

        {uploading && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div className="progress-fill uploading"></div>
            </div>
            <p>Uploading files and organizing dataset structure...</p>
          </div>
        )}
      </div>

      {/* Format Guidelines */}
      {projectType === 'object_detection' && (
        <div className="format-guidelines">
          <h4>📖 YOLO Annotation Format</h4>
          <div className="guidelines-content">
            <p>Each image needs a corresponding .txt file with the same name:</p>
            <div className="example">
              <strong>Files:</strong> image1.jpg → image1.txt
              <br />
              <strong>Format:</strong> class_id center_x center_y width height
              <br />
              <strong>Example:</strong> 0 0.5 0.5 0.3 0.4
            </div>
            <ul>
              <li>Coordinates are normalized (0-1)</li>
              <li>class_id starts from 0</li>
              <li>One line per bounding box</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default IndividualUploadComponent;