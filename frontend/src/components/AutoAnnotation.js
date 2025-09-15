import React, { useState, useEffect } from 'react';
import AnomalyDetectionWorkflow from './auto-annotation/AnomalyDetectionWorkflow';
import AutoAnnotationInference from './auto-annotation/AutoAnnotationInference';
import Annotate from './Annotate';
import './AutoAnnotation.css';

function AutoAnnotation() {
  const [annotationMode, setAnnotationMode] = useState('tools'); // 'tools' or 'workflows'
  const [currentView, setCurrentView] = useState('methods');
  const [selectedProject, setSelectedProject] = useState(null);
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/projects/');
      const data = await response.json();
      setProjects(data.projects || []);
    } catch (error) {
      console.error('Failed to load projects:', error);
    } finally {
      setLoading(false);
    }
  };

  const projectsWithData = projects.filter(p => 
    (p.file_counts?.training_images || 0) > 0 || 
    (p.file_counts?.defective_images || 0) > 0
  );

  const renderMainContent = () => {
    if (annotationMode === 'tools') {
      // Original annotate page with different annotation tools  
      return <Annotate />;
    } else {
      // Workflow-based auto-annotation (requires projects and training)
      switch (currentView) {
        case 'methods':
          return (
            <div className="annotation-workflows">
              <div className="workflows-header">
                <h2>🤖 Workflow-based Auto-Annotation</h2>
                <p>Use trained models and pipelines to generate annotations</p>
              </div>

              <div className="workflows-grid">
                <div className="workflow-card">
                  <div className="workflow-header">
                    <h3>🔍 Anomaly Detection Pipeline</h3>
                    <span className="workflow-badge advanced">4-Step Process</span>
                  </div>
                  <div className="workflow-description">
                    <p>Smart defect detection using only normal training images. No manual labeling required!</p>
                    <ul>
                      <li>Extract ROI from normal images</li>
                      <li>Build statistical normal model</li>
                      <li>Detect anomalies in defective images</li>
                      <li>Generate bounding box annotations</li>
                    </ul>
                  </div>
                  <div className="workflow-projects">
                    <h4>Available Projects:</h4>
                    {projectsWithData.filter(p => p.project_type === 'anomaly_detection').length > 0 ? (
                      <div className="project-list">
                        {projectsWithData.filter(p => p.project_type === 'anomaly_detection').map(project => (
                          <button
                            key={project.project_id}
                            className="project-button"
                            onClick={() => {
                              setSelectedProject(project);
                              setCurrentView('anomaly-workflow');
                            }}
                          >
                            📁 {project.project_name}
                            <span className="file-info">
                              {project.file_counts?.training_images || 0} training, {project.file_counts?.defective_images || 0} defective
                            </span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="no-projects">No anomaly detection projects with data. Create one in "My Data" tab.</p>
                    )}
                  </div>
                </div>

                <div className="workflow-card">
                  <div className="workflow-header">
                    <h3>📦 Object Detection</h3>
                    <span className="workflow-badge trained">Trained Model</span>
                  </div>
                  <div className="workflow-description">
                    <p>Generate bounding box annotations using trained YOLO models</p>
                    <ul>
                      <li>Upload images for annotation</li>
                      <li>Run trained model inference</li>
                      <li>Review and export annotations</li>
                    </ul>
                  </div>
                  <div className="workflow-projects">
                    <h4>Available Projects:</h4>
                    {projectsWithData.filter(p => p.project_type === 'object_detection').length > 0 ? (
                      <div className="project-list">
                        {projectsWithData.filter(p => p.project_type === 'object_detection').map(project => (
                          <button
                            key={project.project_id}
                            className="project-button"
                            onClick={() => {
                              setSelectedProject(project);
                              setCurrentView('detection-annotation');
                            }}
                          >
                            📁 {project.project_name}
                            <span className="file-info">
                              {project.file_counts?.training_images || 0} training images
                            </span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="no-projects">No object detection projects with data. Create one in "My Data" tab.</p>
                    )}
                  </div>
                </div>

                <div className="workflow-card">
                  <div className="workflow-header">
                    <h3>🎯 Segmentation</h3>
                    <span className="workflow-badge trained">Trained Model</span>
                  </div>
                  <div className="workflow-description">
                    <p>Generate precise mask annotations using trained segmentation models</p>
                    <ul>
                      <li>Upload images for annotation</li>
                      <li>Run trained model inference</li>
                      <li>Generate precise masks</li>
                    </ul>
                  </div>
                  <div className="workflow-projects">
                    <h4>Available Projects:</h4>
                    {projectsWithData.filter(p => p.project_type === 'segmentation').length > 0 ? (
                      <div className="project-list">
                        {projectsWithData.filter(p => p.project_type === 'segmentation').map(project => (
                          <button
                            key={project.project_id}
                            className="project-button"
                            onClick={() => {
                              setSelectedProject(project);
                              setCurrentView('segmentation-annotation');
                            }}
                          >
                            📁 {project.project_name}
                            <span className="file-info">
                              {project.file_counts?.training_images || 0} training images
                            </span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="no-projects">No segmentation projects with data. Create one in "My Data" tab.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );

        case 'anomaly-workflow':
          return (
            <AnomalyDetectionWorkflow
              selectedProject={selectedProject}
              onBack={() => setCurrentView('methods')}
            />
          );

        case 'detection-annotation':
          return (
            <AutoAnnotationInference
              projectId={selectedProject?.project_id}
              projectType="object_detection"
              onBack={() => setCurrentView('methods')}
            />
          );

        case 'segmentation-annotation':
          return (
            <AutoAnnotationInference
              projectId={selectedProject?.project_id}
              projectType="segmentation"
              onBack={() => setCurrentView('methods')}
            />
          );

        default:
          return null;
      }
    }
  };

  return (
    <div className="auto-annotation-container">
      <div className="auto-annotation-header">
        <h1>🤖 Auto-Annotation</h1>
        <p>Generate annotations using AI-powered tools and workflows</p>
        
        <div className="annotation-mode-toggle">
          <button
            className={`toggle-btn ${annotationMode === 'tools' ? 'active' : ''}`}
            onClick={() => {
              setAnnotationMode('tools');
              setCurrentView('methods');
            }}
          >
            🛠️ Annotation Tools
          </button>
          <button
            className={`toggle-btn ${annotationMode === 'workflows' ? 'active' : ''}`}
            onClick={() => {
              setAnnotationMode('workflows');
              setCurrentView('methods');
            }}
          >
            🤖 Workflow-based
          </button>
        </div>
        
        <div className="mode-description">
          {annotationMode === 'tools' ? (
            <p>📝 Use pre-trained models and direct tools (GroundingDINO, CLIP, SAM2)</p>
          ) : (
            <p>🔄 Use project-based workflows that train custom models first</p>
          )}
        </div>
      </div>
      
      {renderMainContent()}
    </div>
  );
}

export default AutoAnnotation;