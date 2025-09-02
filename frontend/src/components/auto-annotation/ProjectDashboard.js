import React, { useState } from 'react';
import './ProjectDashboard.css';

function ProjectDashboard({ projects, onProjectSelect, onCreateProject, onRefresh, loading, onProjectDeleted }) {
  const [deleteConfirm, setDeleteConfirm] = useState(null); // {projectId, projectName}
  const [deleting, setDeleting] = useState(null); // projectId being deleted

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'created': return '📝';
      case 'data_uploaded': return '📊';
      case 'training': return '🚀';
      case 'completed': return '✅';
      case 'failed': return '❌';
      default: return '⚪';
    }
  };

  const getTrainingStatusIcon = (trainingStatus) => {
    switch (trainingStatus) {
      case 'not_started': return '⏳';
      case 'running': return '🚀';
      case 'completed': return '✅';
      case 'failed': return '❌';
      default: return '⚪';
    }
  };

  const handleDeleteClick = (e, project) => {
    e.stopPropagation(); // Prevent project selection
    setDeleteConfirm({
      projectId: project.project_id,
      projectName: project.project_name
    });
  };

  const handleDeleteConfirm = async () => {
    if (!deleteConfirm) return;
    
    setDeleting(deleteConfirm.projectId);
    
    try {
      const response = await fetch(`/api/auto-annotation/projects/${deleteConfirm.projectId}`, {
        method: 'DELETE'
      });
      
      const result = await response.json();
      
      if (result.status === 'success') {
        // Notify parent component
        if (onProjectDeleted) {
          onProjectDeleted(deleteConfirm.projectId);
        }
        // Refresh the project list
        if (onRefresh) {
          onRefresh();
        }
      } else {
        console.error('Failed to delete project:', result.message);
        alert('Failed to delete project: ' + result.message);
      }
    } catch (error) {
      console.error('Error deleting project:', error);
      alert('Error deleting project. Please try again.');
    } finally {
      setDeleting(null);
      setDeleteConfirm(null);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteConfirm(null);
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading-spinner">
          Loading projects...
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="dashboard-title">
          <h2>📊 Project Dashboard</h2>
          <p>Manage your auto-annotation projects</p>
        </div>
        
        <div className="dashboard-actions">
          <button className="refresh-button" onClick={onRefresh}>
            🔄 Refresh
          </button>
          <button className="create-button" onClick={onCreateProject}>
            ➕ New Project
          </button>
        </div>
      </div>

      {projects.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📋</div>
          <h3>No Projects Yet</h3>
          <p>Create your first auto-annotation project to get started</p>
          <button className="create-button-large" onClick={onCreateProject}>
            ➕ Create Your First Project
          </button>
        </div>
      ) : (
        <div className="projects-grid">
          {projects.map((project) => (
            <div
              key={project.project_id}
              className="project-card"
              onClick={() => onProjectSelect(project)}
            >
              <div className="project-header">
                <div className="project-title">
                  <h3>{project.project_name}</h3>
                  <span className={`project-type-badge ${project.project_type}`}>
                    {project.project_type === 'object_detection' ? '📦 Object Detection' : 
                     project.project_type === 'segmentation' ? '🎯 Segmentation' :
                     '🧠 Anomaly Detection'}
                  </span>
                </div>
                <div className="project-status">
                  <span className="status-badge">
                    {getStatusIcon(project.status)} {project.status}
                  </span>
                </div>
              </div>

              <div className="project-description">
                {project.description || 'No description provided'}
              </div>

              <div className="project-stats">
                <div className="stat-item">
                  <span className="stat-label">Training Images:</span>
                  <span className="stat-value">{project.training_images_count || 0}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Training Status:</span>
                  <span className="stat-value">
                    {getTrainingStatusIcon(project.training_status)} {project.training_status}
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Created:</span>
                  <span className="stat-value">{formatDate(project.created_at)}</span>
                </div>
              </div>

              <div className="project-actions">
                <button className="action-button primary">
                  Open Project →
                </button>
                <button 
                  className="action-button delete"
                  onClick={(e) => handleDeleteClick(e, project)}
                  disabled={deleting === project.project_id}
                  title="Delete Project"
                >
                  {deleting === project.project_id ? '⏳' : '🗑️'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="dashboard-footer">
        <div className="stats-summary">
          <div className="summary-item">
            <span className="summary-number">{projects.length}</span>
            <span className="summary-label">Total Projects</span>
          </div>
          <div className="summary-item">
            <span className="summary-number">
              {projects.filter(p => p.project_type === 'object_detection').length}
            </span>
            <span className="summary-label">Object Detection</span>
          </div>
          <div className="summary-item">
            <span className="summary-number">
              {projects.filter(p => p.project_type === 'segmentation').length}
            </span>
            <span className="summary-label">Segmentation</span>
          </div>
          <div className="summary-item">
            <span className="summary-number">
              {projects.filter(p => p.project_type === 'anomaly_detection').length}
            </span>
            <span className="summary-label">Anomaly Detection</span>
          </div>
          <div className="summary-item">
            <span className="summary-number">
              {projects.filter(p => p.training_status === 'completed' || p.project_type === 'anomaly_detection').length}
            </span>
            <span className="summary-label">Ready for Use</span>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div className="modal-overlay">
          <div className="delete-modal">
            <div className="modal-header">
              <h3>🗑️ Delete Project</h3>
            </div>
            <div className="modal-content">
              <p>Are you sure you want to delete the project:</p>
              <p className="project-name-highlight">"{deleteConfirm.projectName}"</p>
              <p className="warning-text">
                ⚠️ This action cannot be undone. All training data, models, and results will be permanently deleted.
              </p>
            </div>
            <div className="modal-actions">
              <button 
                className="modal-button cancel"
                onClick={handleDeleteCancel}
                disabled={deleting}
              >
                Cancel
              </button>
              <button 
                className="modal-button delete"
                onClick={handleDeleteConfirm}
                disabled={deleting}
              >
                {deleting ? '⏳ Deleting...' : '🗑️ Delete Project'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProjectDashboard;