import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import TrainingDataUpload from './upload/TrainingDataUpload';

function ProjectDetail() {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

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
                    <span className={`status-badge ${project?.status}`}>{project?.status}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">Created:</span>
                    <span className="value">{new Date(project?.created_at).toLocaleDateString()}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">Owner:</span>
                    <span className="value">{project?.owner}</span>
                  </div>
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
                      <span className="stat-label">Defective Images</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-number">{project?.file_counts?.test_images || 0}</span>
                      <span className="stat-label">Test Images</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-number">{project?.file_counts?.annotation_files || 0}</span>
                      <span className="stat-label">Annotation Files</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="project-actions">
                <button 
                  className="action-button secondary"
                  onClick={() => window.open(`/api/projects/${projectId}/export-data`, '_blank')}
                >
                  📥 Export Dataset
                </button>
                <button 
                  className="action-button primary"
                  onClick={() => navigate('/auto-annotation', { state: { projectId } })}
                  disabled={(project?.file_counts?.training_images || 0) === 0}
                >
                  🤖 Start Auto-Annotation
                </button>
                <button 
                  className="action-button primary"
                  onClick={() => navigate('/my-training', { state: { projectId } })}
                  disabled={(project?.file_counts?.training_images || 0) === 0}
                >
                  🚀 Start Training
                </button>
              </div>
            </div>
          </div>
        );

      case 'training-data':
        return (
          <div className="tab-content">
            <div className="upload-section">
              <h3>📁 Training Images</h3>
              <p>Upload normal/good images for training your model</p>
              <TrainingDataUpload
                projectId={projectId}
                algorithm={project?.algorithm}
                projectType={project?.project_type}
                onUploadComplete={handleDataUploaded}
                onError={(error) => console.error('Upload error:', error)}
              />
            </div>
          </div>
        );

      case 'defective-data':
        return (
          <div className="tab-content">
            <div className="upload-section">
              <h3>🚨 Defective Images</h3>
              <p>Upload defective/anomaly images for testing or validation</p>
              <div className="simple-upload-note">
                <p><strong>Note:</strong> For defective data upload, please use the simple file upload endpoint directly or the legacy interface.</p>
                <p>The algorithm-aware upload system is designed for training data with annotations.</p>
              </div>
            </div>
          </div>
        );

      case 'test-data':
        return (
          <div className="tab-content">
            <div className="upload-section">
              <h3>🧪 Test Images</h3>
              <p>Upload images for validation and testing</p>
              <TrainingDataUpload
                projectId={projectId}
                projectType={project?.project_type}
                onDataUploaded={handleDataUploaded}
                uploadType="test"
              />
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
          ← Back to My Data
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
          className={`tab-button ${activeTab === 'training-data' ? 'active' : ''}`}
          onClick={() => setActiveTab('training-data')}
        >
          📁 Training Data
        </button>
        <button
          className={`tab-button ${activeTab === 'defective-data' ? 'active' : ''}`}
          onClick={() => setActiveTab('defective-data')}
        >
          🚨 Defective Data
        </button>
        <button
          className={`tab-button ${activeTab === 'test-data' ? 'active' : ''}`}
          onClick={() => setActiveTab('test-data')}
        >
          🧪 Test Data
        </button>
      </div>

      {renderTabContent()}
    </div>
  );
}

export default ProjectDetail;