import React, { useState, useEffect } from 'react';
import ProjectDashboard from './auto-annotation/ProjectDashboard';
import ProjectCreation from './auto-annotation/ProjectCreation';
import TrainingDataUpload from './auto-annotation/TrainingDataUpload';
import ModelTraining from './auto-annotation/ModelTraining';
import AutoAnnotationInference from './auto-annotation/AutoAnnotationInference';
import AnomalyDetectionWorkflow from './auto-annotation/AnomalyDetectionWorkflow';
import './AutoAnnotation.css';

function AutoAnnotation() {
  const [currentView, setCurrentView] = useState('dashboard');
  const [selectedProject, setSelectedProject] = useState(null);
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/auto-annotation/projects');
      const data = await response.json();
      if (data.status === 'success') {
        setProjects(data.projects);
      }
    } catch (error) {
      console.error('Failed to load projects:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleProjectCreated = (newProject) => {
    setProjects(prev => [newProject.metadata, ...prev]);
    setCurrentView('dashboard');
  };

  const handleProjectSelected = (project) => {
    setSelectedProject(project);
    setCurrentView('project-details');
  };

  const handleProjectUpdated = async () => {
    // Reload projects to get updated training status
    await loadProjects();
    
    // Update selected project if one is selected
    if (selectedProject) {
      const response = await fetch(`/api/auto-annotation/projects/${selectedProject.project_id}`);
      const data = await response.json();
      if (data.status === 'success') {
        setSelectedProject(data.project);
      }
    }
  };

  const handleProjectDeleted = (deletedProjectId) => {
    // Remove project from local state
    setProjects(prev => prev.filter(p => p.project_id !== deletedProjectId));
    
    // If the deleted project was currently selected, go back to dashboard
    if (selectedProject && selectedProject.project_id === deletedProjectId) {
      setSelectedProject(null);
      setCurrentView('dashboard');
    }
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'dashboard':
        return (
          <ProjectDashboard
            projects={projects}
            onProjectSelect={handleProjectSelected}
            onCreateProject={() => setCurrentView('create-project')}
            onRefresh={loadProjects}
            onProjectDeleted={handleProjectDeleted}
            loading={loading}
          />
        );
      case 'create-project':
        return (
          <ProjectCreation
            onProjectCreated={handleProjectCreated}
            onCancel={() => setCurrentView('dashboard')}
          />
        );
      case 'project-details':
        return (
          <div className="project-details-container">
            <div className="project-header">
              <button 
                className="back-button"
                onClick={() => setCurrentView('dashboard')}
              >
                ← Back to Dashboard
              </button>
              <h2>{selectedProject?.project_name}</h2>
              <span className={`project-type-badge ${selectedProject?.project_type}`}>
                {selectedProject?.project_type === 'object_detection' ? '📦 Object Detection' : '🎯 Segmentation'}
              </span>
            </div>
            
            <div className="project-workflow">
              <div className="workflow-steps">
                <div className="workflow-step">
                  <h3>1. Upload Training Data</h3>
                  <TrainingDataUpload
                    projectId={selectedProject?.project_id}
                    projectType={selectedProject?.project_type}
                    onDataUploaded={loadProjects}
                  />
                </div>
                
                {selectedProject?.project_type !== 'anomaly_detection' && (
                  <div className="workflow-step">
                    <h3>2. Train Model</h3>
                    <ModelTraining
                      projectId={selectedProject?.project_id}
                      projectType={selectedProject?.project_type}
                      trainingStatus={selectedProject?.training_status}
                      onTrainingStatusChange={handleProjectUpdated}
                    />
                  </div>
                )}
                
                {selectedProject?.project_type === 'anomaly_detection' ? (
                  <div className="workflow-step">
                    <h3>2. Anomaly Detection Workflow</h3>
                    <button
                      className="start-anomaly-workflow-button"
                      onClick={() => setCurrentView('anomaly-workflow')}
                      disabled={selectedProject?.training_images_count === 0}
                    >
                      🔍 Start Anomaly Detection
                    </button>
                    <p className="workflow-note">
                      Smart detection using only normal images - no manual labeling required
                    </p>
                  </div>
                ) : (
                  <div className="workflow-step">
                    <h3>3. Auto-Annotate Images</h3>
                    <AutoAnnotationInference
                      projectId={selectedProject?.project_id}
                      projectType={selectedProject?.project_type}
                      trainingStatus={selectedProject?.training_status}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      case 'anomaly-workflow':
        return (
          <AnomalyDetectionWorkflow
            selectedProject={selectedProject}
            onBack={() => setCurrentView('project-details')}
          />
        );
      default:
        return <ProjectDashboard projects={projects} onProjectSelect={handleProjectSelected} />;
    }
  };

  return (
    <div className="auto-annotation-container">
      <div className="auto-annotation-header">
        <h1>🤖 Auto-Annotation System</h1>
        <p className="subtitle">
          Create training projects, upload manual annotations, and automatically annotate new images
        </p>
      </div>
      
      <div className="workflow-info">
        <div className="workflow-card">
          <div className="workflow-icon">📦</div>
          <h3>Object Detection</h3>
          <p>YOLO for fast defect bounding boxes</p>
          <ul>
            <li>Quick defect identification</li>
            <li>Fast training and inference</li>
            <li>Good for defect counting</li>
          </ul>
          <div className="card-requirement">Requires manual annotations</div>
        </div>
        
        <div className="workflow-card">
          <div className="workflow-icon">🎯</div>
          <h3>Segmentation</h3>
          <p>SAM2 for precise defect masks</p>
          <ul>
            <li>Precise defect boundaries</li>
            <li>Few-shot learning capability</li>
            <li>Excellent for measurement</li>
          </ul>
          <div className="card-requirement">Requires manual annotations</div>
        </div>

        <div className="workflow-card">
          <div className="workflow-icon">🧠</div>
          <h3>Anomaly Detection</h3>
          <p>Smart detection without manual labels</p>
          <ul>
            <li>No manual labeling required</li>
            <li>Uses only normal images</li>
            <li>Human-in-the-loop workflow</li>
          </ul>
          <div className="card-requirement">Requires normal images only</div>
        </div>
      </div>

      <div className="auto-annotation-content">
        {renderCurrentView()}
      </div>
    </div>
  );
}

export default AutoAnnotation;