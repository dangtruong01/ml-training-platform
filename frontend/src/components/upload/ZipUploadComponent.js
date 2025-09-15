import React, { useState, useCallback } from 'react';
import './Upload.css';

const ZipUploadComponent = ({ 
  projectId, 
  algorithm, 
  onUploadSuccess, 
  onUploadError,
  disabled = false 
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [validationInfo, setValidationInfo] = useState(null);

  const validateZipFile = useCallback((file) => {
    // Basic validation
    if (!file.name.toLowerCase().endsWith('.zip')) {
      return {
        valid: false,
        error: 'Please select a ZIP file'
      };
    }

    if (file.size > 500 * 1024 * 1024) { // 500MB limit
      return {
        valid: false,
        error: 'ZIP file too large. Maximum size is 500MB'
      };
    }

    return {
      valid: true,
      info: {
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        type: 'ZIP Dataset'
      }
    };
  }, []);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (disabled) return;

    const files = [...e.dataTransfer.files];
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  }, [disabled]);

  const handleFileSelect = (file) => {
    const validation = validateZipFile(file);
    
    if (validation.valid) {
      setSelectedFile(file);
      setValidationInfo(validation.info);
    } else {
      setValidationInfo(null);
      onUploadError(validation.error);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || uploading) return;

    setUploading(true);
    
    try {
      const formData = new FormData();
      formData.append('dataset_zip', selectedFile);

      const response = await fetch(`/api/projects/${projectId}/upload-training-data`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (response.ok && result.status === 'success') {
        onUploadSuccess({
          method: 'zip',
          file: selectedFile.name,
          info: result.dataset_info || {}
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

  const clearSelection = () => {
    setSelectedFile(null);
    setValidationInfo(null);
  };

  return (
    <div className="zip-upload-component">
      <div className="upload-section">
        <h4>📦 Upload Complete Dataset</h4>
        <p>Upload your prepared {algorithm} dataset as a ZIP file</p>

        <div 
          className={`drop-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {!selectedFile ? (
            <>
              <div className="drop-zone-content">
                <div className="upload-icon">📦</div>
                <h3>Drop your dataset ZIP file here</h3>
                <p>or</p>
                <label className="file-input-label">
                  <input
                    type="file"
                    accept=".zip"
                    onChange={handleFileInput}
                    disabled={disabled}
                    style={{ display: 'none' }}
                  />
                  <span className="browse-button">Browse Files</span>
                </label>
              </div>
              <div className="file-requirements">
                <p><strong>Requirements:</strong></p>
                <ul>
                  <li>ZIP file containing complete dataset</li>
                  <li>Must include data.yaml configuration</li>
                  <li>Organized folder structure (images/, labels/)</li>
                  <li>Maximum size: 500MB</li>
                </ul>
              </div>
            </>
          ) : (
            <div className="selected-file-info">
              <div className="file-preview">
                <span className="file-icon">📦</span>
                <div className="file-details">
                  <h4>{validationInfo.name}</h4>
                  <p>Size: {validationInfo.size}</p>
                  <p>Type: {validationInfo.type}</p>
                </div>
                <button 
                  className="clear-selection"
                  onClick={clearSelection}
                  disabled={uploading}
                >
                  ✕
                </button>
              </div>
              
              <div className="upload-actions">
                <button
                  className="upload-button primary"
                  onClick={handleUpload}
                  disabled={uploading || disabled}
                >
                  {uploading ? '📤 Uploading...' : '🚀 Upload Dataset'}
                </button>
              </div>
            </div>
          )}
        </div>

        {uploading && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div className="progress-fill uploading"></div>
            </div>
            <p>Uploading and validating dataset structure...</p>
          </div>
        )}
      </div>

      <div className="dataset-structure-guide">
        <h4>📋 Expected Dataset Structure</h4>
        <div className="structure-example">
          <pre>{`your_dataset.zip
├── data.yaml
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── image3.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        └── image3.txt`}</pre>
        </div>
        
        <div className="structure-tips">
          <h5>💡 Tips for Success:</h5>
          <ul>
            <li>Use annotation tools like LabelImg, Roboflow, or CVAT</li>
            <li>Export in YOLO format</li>
            <li>Ensure image and label filenames match</li>
            <li>Include both train and val splits</li>
            <li>Verify data.yaml has correct paths and class names</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ZipUploadComponent;