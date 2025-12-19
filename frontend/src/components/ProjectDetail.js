import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import TrainingDataUpload from './auto-annotation/TrainingDataUpload';
import './ProjectDetail.css';

function ProjectDetail() {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [uploadStates, setUploadStates] = useState({
    zipUploading: false,
    rawUploading: false,
    cvatUploading: false
  });
  const [uploadResults, setUploadResults] = useState({});
  const [errors, setErrors] = useState({});
  const [selectedUploadMethod, setSelectedUploadMethod] = useState('manual');

  useEffect(() => {
    loadProject();
  }, [projectId]);

  const loadProject = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/projects/${projectId}`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setProject(data.project);
      } else {
        console.error('Failed to load project:', data.message);
        navigate('/my-data');
      }
    } catch (error) {
      console.error('Error loading project:', error);
      navigate('/my-data');
    } finally {
      setLoading(false);
    }
  };

  const handleDataUploaded = () => {
    loadProject(); // Refresh project data after upload
  };

  // Handle ZIP file upload for YOLO datasets
  const handleZipUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.zip')) {
      setErrors({ zip: 'Please select a ZIP file' });
      return;
    }

    setUploadStates(prev => ({ ...prev, zipUploading: true }));
    setErrors(prev => ({ ...prev, zip: '' }));

    try {
      const formData = new FormData();
      formData.append('dataset_zip', file);

      const response = await fetch(`/api/projects/${projectId}/upload-zip-dataset`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setUploadResults(prev => ({ ...prev, zip: result }));
        handleDataUploaded();
      } else {
        setErrors({ zip: result.message || 'ZIP upload failed' });
      }
    } catch (error) {
      console.error('ZIP upload failed:', error);
      setErrors({ zip: 'Network error. Please try again.' });
    } finally {
      setUploadStates(prev => ({ ...prev, zipUploading: false }));
    }
  };

  // Handle raw annotation folder upload
  const handleRawAnnotationUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    setUploadStates(prev => ({ ...prev, rawUploading: true }));
    setErrors(prev => ({ ...prev, raw: '' }));

    try {
      const formData = new FormData();
      
      // Add all files to FormData
      files.forEach(file => {
        formData.append('raw_files', file);
      });

      const response = await fetch(`/api/projects/${projectId}/upload-raw-annotations`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setUploadResults(prev => ({ ...prev, raw: result }));
        handleDataUploaded();
      } else {
        setErrors({ raw: result.message || 'Raw annotation upload failed' });
      }
    } catch (error) {
      console.error('Raw annotation upload failed:', error);
      setErrors({ raw: 'Network error. Please try again.' });
    } finally {
      setUploadStates(prev => ({ ...prev, rawUploading: false }));
    }
  };

  // Handle CVAT ZIP file upload and conversion
  const handleCvatUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.zip')) {
      setErrors({ cvat: 'Please select a ZIP file' });
      return;
    }

    setUploadStates(prev => ({ ...prev, cvatUploading: true }));
    setErrors(prev => ({ ...prev, cvat: '' }));

    try {
      const formData = new FormData();
      formData.append('cvat_zip', file);

      const response = await fetch(`/api/projects/${projectId}/upload-cvat-dataset`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.status === 'success') {
        setUploadResults(prev => ({ ...prev, cvat: result }));
        handleDataUploaded();
      } else {
        setErrors({ cvat: result.message || 'CVAT upload failed' });
      }
    } catch (error) {
      console.error('CVAT upload failed:', error);
      setErrors({ cvat: 'Network error. Please try again.' });
    } finally {
      setUploadStates(prev => ({ ...prev, cvatUploading: false }));
    }
  };

  const getDataTabContent = () => {
    switch (project?.project_type) {
      case 'object_detection':
        return {
          title: '📦 Object Detection Data',
          description: 'Upload training images and YOLO format annotations',
          uploadOptions: [
            {
              type: 'manual',
              title: '📁 Manual Upload',
              description: 'Upload images and annotation files separately',
              component: (
                <TrainingDataUpload
                  projectId={projectId}
                  projectType={project?.project_type}
                  onDataUploaded={handleDataUploaded}
                  uploadType="training"
                />
              )
            },
            {
              type: 'zip',
              title: '📦 YOLO ZIP Dataset',
              description: 'Upload a complete YOLO dataset as ZIP file (includes images/, labels/, data.yaml)',
              component: (
                <div className="zip-upload-section">
                  <div className="upload-instructions">
                    <h5>📋 ZIP Structure Expected:</h5>
                    <div className="file-structure">
                      <code>
                        dataset.zip<br/>
                        ├── images/<br/>
                        │   ├── train/<br/>
                        │   └── val/<br/>
                        ├── labels/<br/>
                        │   ├── train/<br/>
                        │   └── val/<br/>
                        └── data.yaml
                      </code>
                    </div>
                  </div>
                  
                  <div className="file-input-container">
                    <input
                      type="file"
                      accept=".zip"
                      onChange={handleZipUpload}
                      disabled={uploadStates.zipUploading}
                      className="file-input"
                      id="zip-upload"
                    />
                    <label htmlFor="zip-upload" className="file-input-label">
                      {uploadStates.zipUploading ? '⏳ Uploading...' : '📦 Choose ZIP File'}
                    </label>
                  </div>
                  
                  {errors.zip && (
                    <div className="error-message">{errors.zip}</div>
                  )}
                  
                  {uploadResults.zip && (
                    <div className="success-message">
                      ✅ ZIP dataset uploaded successfully! Found {uploadResults.zip.total_images} images and {uploadResults.zip.total_labels} labels.
                    </div>
                  )}
                </div>
              )
            },
            {
              type: 'raw',
              title: '🗂️ Raw Annotations',
              description: 'Upload raw annotation folder with images/, labels/, and classes.txt',
              component: (
                <div className="raw-upload-section">
                  <div className="upload-instructions">
                    <h5>📋 Raw Folder Structure Expected:</h5>
                    <div className="file-structure">
                      <code>
                        Select all files from:<br/>
                        ├── images/ (all your images)<br/>
                        ├── labels/ (YOLO .txt files)<br/>
                        └── classes.txt (class names)
                      </code>
                    </div>
                  </div>
                  
                  <div className="file-input-container">
                    <input
                      type="file"
                      multiple
                      webkitdirectory=""
                      directory=""
                      onChange={handleRawAnnotationUpload}
                      disabled={uploadStates.rawUploading}
                      className="file-input"
                      id="raw-upload"
                    />
                    <label htmlFor="raw-upload" className="file-input-label">
                      {uploadStates.rawUploading ? '⏳ Processing...' : '🗂️ Select Raw Annotation Folder'}
                    </label>
                  </div>
                  
                  {errors.raw && (
                    <div className="error-message">{errors.raw}</div>
                  )}
                  
                  {uploadResults.raw && (
                    <div className="success-message">
                      ✅ Raw annotations processed successfully!
                      <br/>📊 {uploadResults.raw.train_samples} training samples, {uploadResults.raw.val_samples} validation samples
                      <br/>🏷️ Classes: {uploadResults.raw.classes.join(', ')}
                    </div>
                  )}
                </div>
              )
            },
            {
              type: 'cvat',
              title: '🏷️ CVAT Dataset',
              description: 'Upload CVAT-exported folder with annotations',
              component: (
                <div className="cvat-upload-section">
                  <div className="upload-instructions">
                    <h5>📋 CVAT ZIP Structure Expected:</h5>
                    <div className="file-structure">
                      <code>
                        ZIP file containing:<br/>
                        ├── images/ (annotated images)<br/>
                        ├── labels/ (YOLO format labels)<br/>
                        ├── classes.txt (class definitions)<br/>
                        └── notes.json (CVAT metadata)
                      </code>
                    </div>
                    <p className="info-note">
                      💡 CVAT datasets will be automatically converted to YOLO training format with train/val split
                    </p>
                  </div>

                  <div className="upload-area">
                    <input
                      type="file"
                      accept=".zip"
                      onChange={handleCvatUpload}
                      disabled={uploadStates.cvatUploading}
                      className="file-input"
                      id="cvat-upload"
                    />
                    <label htmlFor="cvat-upload" className="file-input-label">
                      {uploadStates.cvatUploading ? '⏳ Converting CVAT to YOLO...' : '🏷️ Choose CVAT ZIP File'}
                    </label>
                  </div>

                  {errors.cvat && (
                    <div className="error-message">{errors.cvat}</div>
                  )}

                  {uploadResults.cvat && (
                    <div className="success-message">
                      ✅ CVAT dataset converted and uploaded successfully!
                      <br/>📊 {uploadResults.cvat.train_images} training, {uploadResults.cvat.val_images} validation images
                      <br/>🏷️ Classes: {uploadResults.cvat.classes.join(', ')}
                    </div>
                  )}
                </div>
              )
            }
          ]
        };

      case 'segmentation':
        return {
          title: '🎯 Segmentation Data',
          description: 'Upload images and segmentation masks',
          uploadOptions: [
            {
              type: 'manual',
              title: '📁 Manual Upload',
              description: 'Upload images and segmentation annotations separately',
              component: (
                <TrainingDataUpload
                  projectId={projectId}
                  projectType={project?.project_type}
                  onDataUploaded={handleDataUploaded}
                  uploadType="training"
                />
              )
            },
            {
              type: 'zip',
              title: '📦 Segmentation ZIP Dataset',
              description: 'Upload a complete segmentation dataset as ZIP file',
              component: (
                <div className="zip-upload-section">
                  <div className="upload-instructions">
                    <h5>📋 ZIP Structure Expected:</h5>
                    <div className="file-structure">
                      <code>
                        dataset.zip<br/>
                        ├── images/<br/>
                        ├── masks/ (or labels/)<br/>
                        └── data.yaml
                      </code>
                    </div>
                  </div>
                  
                  <div className="file-input-container">
                    <input
                      type="file"
                      accept=".zip"
                      onChange={handleZipUpload}
                      disabled={uploadStates.zipUploading}
                      className="file-input"
                      id="seg-zip-upload"
                    />
                    <label htmlFor="seg-zip-upload" className="file-input-label">
                      {uploadStates.zipUploading ? '⏳ Uploading...' : '📦 Choose ZIP File'}
                    </label>
                  </div>
                  
                  {errors.zip && (
                    <div className="error-message">{errors.zip}</div>
                  )}
                </div>
              )
            }
          ]
        };

      case 'anomaly_detection':
        return {
          title: '🔍 Anomaly Detection Data',
          description: 'Upload normal images for anomaly detection training',
          uploadOptions: [
            {
              type: 'manual',
              title: '📁 Normal Images',
              description: 'Upload normal/good images (no defects) for training',
              component: (
                <TrainingDataUpload
                  projectId={projectId}
                  projectType={project?.project_type}
                  onDataUploaded={handleDataUploaded}
                  uploadType="training"
                />
              )
            },
            {
              type: 'defective',
              title: '🚨 Defective Images (Optional)',
              description: 'Upload defective/anomalous images for validation',
              component: (
                <TrainingDataUpload
                  projectId={projectId}
                  projectType={project?.project_type}
                  onDataUploaded={handleDataUploaded}
                  uploadType="defective"
                />
              )
            }
          ]
        };

      default:
        return {
          title: '📁 Data Upload',
          description: 'Upload your training data',
          uploadOptions: []
        };
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="tab-content">
            <div className="project-overview">
              <div className="project-info-grid">
                <div className="info-card">
                  <h3>📋 Project Information</h3>
                  <div className="info-item">
                    <span className="label">Name:</span>
                    <span className="value">{project?.project_name}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">Type:</span>
                    <span className={`project-type-badge ${project?.project_type}`}>
                      {project?.project_type === 'object_detection' ? '📦 Object Detection' : 
                       project?.project_type === 'segmentation' ? '🎯 Segmentation' : 
                       '🔍 Anomaly Detection'}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="label">Status:</span>
                    <span className={`status-badge ${project?.status || 'no_data'}`}>
                      {project?.status || 'No Data'}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="label">Created:</span>
                    <span className="value">{new Date(project?.created_at).toLocaleDateString()}</span>
                  </div>
                  {project?.description && (
                    <div className="info-item">
                      <span className="label">Description:</span>
                      <span className="value">{project?.description}</span>
                    </div>
                  )}
                </div>

                <div className="info-card">
                  <h3>📊 Dataset Statistics</h3>
                  <div className="stats-grid">
                    <div className="stat-item">
                      <span className="stat-number">{project?.file_counts?.training_images || 0}</span>
                      <span className="stat-label">Training Images</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-number">{project?.file_counts?.defective_images || 0}</span>
                      <span className="stat-label">
                        {project?.project_type === 'anomaly_detection' ? 'Defective Images' : 'Validation Images'}
                      </span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-number">{project?.file_counts?.test_images || 0}</span>
                      <span className="stat-label">Test Images</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-number">{project?.file_counts?.annotation_files || 0}</span>
                      <span className="stat-label">Annotations</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="project-actions">
                <button 
                  className="action-button secondary"
                  onClick={() => setActiveTab('data')}
                  disabled={false}
                >
                  📤 Manage Data
                </button>
                <button 
                  className="action-button secondary"
                  onClick={() => window.open(`/api/projects/${projectId}/export-data`, '_blank')}
                  disabled={(project?.file_counts?.training_images || 0) === 0}
                >
                  📥 Export Dataset
                </button>
                <button 
                  className="action-button primary"
                  onClick={() => navigate('/my-training', { state: { projectId, projectType: project?.project_type } })}
                  disabled={(project?.file_counts?.training_images || 0) === 0}
                >
                  🚀 Start Training
                </button>
              </div>

              {(project?.file_counts?.training_images || 0) === 0 && (
                <div className="no-data-notice">
                  <h4>🎯 Ready to Get Started?</h4>
                  <p>Upload your training data to begin using this project.</p>
                  <button 
                    className="action-button primary"
                    onClick={() => setActiveTab('data')}
                  >
                    📤 Upload Data Now
                  </button>
                </div>
              )}
            </div>
          </div>
        );

      case 'data':
        const dataConfig = getDataTabContent();
        const selectedOption = dataConfig.uploadOptions.find(option => option.type === selectedUploadMethod) || dataConfig.uploadOptions[0];

        return (
          <div className="tab-content">
            <div className="data-management-hub">
              <div className="data-header">
                <h3>{dataConfig.title}</h3>
                <p>{dataConfig.description}</p>
              </div>

              {/* Upload Method Toggle */}
              {dataConfig.uploadOptions.length > 1 && (
                <div className="upload-method-toggle">
                  <div className="toggle-header">
                    <h4>📋 Upload Method</h4>
                    <p>Choose how you want to upload your data</p>
                  </div>
                  <div className="toggle-options">
                    {dataConfig.uploadOptions.map((option) => (
                      <button
                        key={option.type}
                        className={`toggle-button ${selectedUploadMethod === option.type ? 'active' : ''}`}
                        onClick={() => setSelectedUploadMethod(option.type)}
                      >
                        <span className="toggle-icon">
                          {option.type === 'manual' ? '📁' :
                           option.type === 'zip' ? '📦' :
                           option.type === 'raw' ? '🗂️' :
                           option.type === 'cvat' ? '🏷️' :
                           option.type === 'defective' ? '🚨' : '📄'}
                        </span>
                        <div className="toggle-text">
                          <span className="toggle-title">{option.title.replace(/^[^\s]+\s/, '')}</span>
                          <span className="toggle-desc">{option.description}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Selected Upload Option */}
              {selectedOption && (
                <div className="active-upload-section">
                  <div className="upload-section-header">
                    <h4>{selectedOption.title}</h4>
                    <p>{selectedOption.description}</p>
                  </div>
                  <div className="upload-content">
                    {selectedOption.component}
                  </div>
                </div>
              )}
            </div>
          </div>
        );

      case 'training':
        return (
          <div className="tab-content">
            <div className="training-section">
              <h3>🚀 Training Configuration</h3>
              <p>Configure and start training your {project?.project_type} model.</p>
              
              {(project?.file_counts?.training_images || 0) === 0 ? (
                <div className="requirement-notice">
                  <h4>⚠️ Training Data Required</h4>
                  <p>Please upload training data before configuring training.</p>
                  <button 
                    className="action-button primary"
                    onClick={() => setActiveTab('data')}
                  >
                    📤 Upload Data
                  </button>
                </div>
              ) : (
                <div className="training-ready">
                  <h4>✅ Ready for Training</h4>
                  <p>Your project has {project?.file_counts?.training_images} training images.</p>
                  <button 
                    className="action-button primary"
                    onClick={() => navigate('/my-training', { state: { projectId, projectType: project?.project_type } })}
                  >
                    🚀 Go to Training Dashboard
                  </button>
                </div>
              )}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="project-detail-container">
        <div className="loading-spinner">Loading project details...</div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="project-detail-container">
        <div className="error-message">Project not found</div>
      </div>
    );
  }

  return (
    <div className="project-detail-container">
      <div className="project-header">
        <button 
          className="back-button"
          onClick={() => navigate('/my-data')}
        >
          ← Back to My Projects
        </button>
        <div className="project-title">
          <h1>{project.project_name}</h1>
          <span className={`project-type-badge ${project.project_type}`}>
            {project.project_type === 'object_detection' ? '📦 Object Detection' : 
             project.project_type === 'segmentation' ? '🎯 Segmentation' : 
             '🔍 Anomaly Detection'}
          </span>
        </div>
      </div>

      <div className="project-tabs">
        <button
          className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📊 Overview
        </button>
        <button
          className={`tab-button ${activeTab === 'data' ? 'active' : ''}`}
          onClick={() => setActiveTab('data')}
        >
          📁 Data
        </button>
        <button
          className={`tab-button ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          🚀 Training
        </button>
      </div>

      {renderTabContent()}
    </div>
  );
}

export default ProjectDetail;