import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ProjectDashboard from './auto-annotation/ProjectDashboard';
import EnhancedProjectCreation from './project/EnhancedProjectCreation';

function MyData() {
  const navigate = useNavigate();
  const [currentView, setCurrentView] = useState('dashboard');
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

  const handleProjectCreated = (newProject) => {
    // Enhanced project creation includes more data (project + algorithm + upload result)
    const projectForList = {
      project_id: newProject.project_id,
      project_name: newProject.metadata?.project_name || newProject.project_name,
      project_type: newProject.metadata?.project_type || newProject.project_type,
      algorithm: newProject.algorithm, // New field from enhanced creation
      owner: newProject.metadata?.owner || 'system',
      status: newProject.metadata?.status || 'active',
      created_at: newProject.metadata?.created_at || new Date().toISOString(),
      updated_at: newProject.metadata?.created_at || new Date().toISOString(),
      file_counts: newProject.uploadResult ? {
        training_images: newProject.uploadResult.trainingImages || newProject.uploadResult.info?.train_images || 0,
        annotation_files: newProject.uploadResult.annotationFiles || 0
      } : { training_images: 0, defective_images: 0 }
    };
    setProjects(prev => [projectForList, ...prev]);
    setCurrentView('dashboard');
    
    // Show success message
    console.log('✅ Project created successfully with algorithm:', newProject.algorithm);
  };

  const handleProjectSelected = (project) => {
    navigate(`/my-data/project/${project.project_id}`);
  };

  const handleProjectDeleted = (deletedProjectId) => {
    // Remove project from local state
    setProjects(prev => prev.filter(p => p.project_id !== deletedProjectId));
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
          <EnhancedProjectCreation
            onProjectCreated={handleProjectCreated}
            onCancel={() => setCurrentView('dashboard')}
          />
        );
      default:
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
    }
  };

  return (
    <div className="my-data-container">
      <div className="my-data-header">
        <h1>📊 My Data</h1>
        <p>Manage your projects and datasets</p>
      </div>
      {renderCurrentView()}
    </div>
  );
}

export default MyData;