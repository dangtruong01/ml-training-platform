import React, { useState } from 'react';
import './ProjectCreation.css';

function ProjectCreation({ onProjectCreated, onCancel }) {
  const [formData, setFormData] = useState({
    projectName: '',
    projectType: 'object_detection',
    description: ''
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.projectName.trim()) {
      newErrors.projectName = 'Project name is required';
    } else if (formData.projectName.length < 3) {
      newErrors.projectName = 'Project name must be at least 3 characters';
    }
    
    if (!formData.projectType) {
      newErrors.projectType = 'Project type is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    try {
      const formDataToSend = new FormData();
      formDataToSend.append('project_name', formData.projectName);
      formDataToSend.append('project_type', formData.projectType);
      formDataToSend.append('description', formData.description);

      const response = await fetch('/api/auto-annotation/create-project', {
        method: 'POST',
        body: formDataToSend
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        onProjectCreated(result);
      } else {
        setErrors({ submit: result.message || 'Failed to create project' });
      }
    } catch (error) {
      console.error('Failed to create project:', error);
      setErrors({ submit: 'Network error. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  return (
    <div className="project-creation-container">
      <div className="creation-header">
        <h2>🆕 Create New Project</h2>
        <p>Set up a new auto-annotation project</p>
      </div>

      <form onSubmit={handleSubmit} className="creation-form">
        <div className="form-group">
          <label htmlFor="projectName">
            Project Name <span className="required">*</span>
          </label>
          <input
            type="text"
            id="projectName"
            name="projectName"
            value={formData.projectName}
            onChange={handleInputChange}
            placeholder="Enter a descriptive project name"
            className={errors.projectName ? 'error' : ''}
          />
          {errors.projectName && (
            <span className="error-message">{errors.projectName}</span>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="projectType">
            Project Type <span className="required">*</span>
          </label>
          <div className="project-type-selection">
            <div 
              className={`type-option ${formData.projectType === 'object_detection' ? 'selected' : ''}`}
              onClick={() => setFormData(prev => ({ ...prev, projectType: 'object_detection' }))}
            >
              <div className="type-icon">📦</div>
              <div className="type-info">
                <h4>Object Detection</h4>
                <p>YOLO for fast defect bounding boxes</p>
                <ul>
                  <li>✅ Quick training and inference</li>
                  <li>✅ Good for defect counting</li>
                  <li>✅ Fast defect identification</li>
                </ul>
                <div className="requirement">Requires: Manual annotations</div>
              </div>
              <input
                type="radio"
                name="projectType"
                value="object_detection"
                checked={formData.projectType === 'object_detection'}
                onChange={handleInputChange}
              />
            </div>

            <div 
              className={`type-option ${formData.projectType === 'segmentation' ? 'selected' : ''}`}
              onClick={() => setFormData(prev => ({ ...prev, projectType: 'segmentation' }))}
            >
              <div className="type-icon">🎯</div>
              <div className="type-info">
                <h4>Segmentation</h4>
                <p>SAM2 for precise defect masks</p>
                <ul>
                  <li>✅ Precise defect boundaries</li>
                  <li>✅ Few-shot learning capability</li>
                  <li>✅ Excellent for measurement</li>
                </ul>
                <div className="requirement">Requires: Manual annotations</div>
              </div>
              <input
                type="radio"
                name="projectType"
                value="segmentation"
                checked={formData.projectType === 'segmentation'}
                onChange={handleInputChange}
              />
            </div>

            <div 
              className={`type-option ${formData.projectType === 'anomaly_detection' ? 'selected' : ''}`}
              onClick={() => setFormData(prev => ({ ...prev, projectType: 'anomaly_detection' }))}
            >
              <div className="type-icon">🧠</div>
              <div className="type-info">
                <h4>Anomaly Detection</h4>
                <p>Smart detection without manual labels</p>
                <ul>
                  <li>✅ No manual labeling required</li>
                  <li>✅ Uses only normal images</li>
                  <li>✅ Human-in-the-loop workflow</li>
                </ul>
                <div className="requirement">Requires: Normal images only</div>
              </div>
              <input
                type="radio"
                name="projectType"
                value="anomaly_detection"
                checked={formData.projectType === 'anomaly_detection'}
                onChange={handleInputChange}
              />
            </div>
          </div>
          {errors.projectType && (
            <span className="error-message">{errors.projectType}</span>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="description">
            Description
          </label>
          <textarea
            id="description"
            name="description"
            value={formData.description}
            onChange={handleInputChange}
            placeholder="Describe your project (optional)"
            rows="3"
          />
        </div>

        {errors.submit && (
          <div className="error-banner">
            ❌ {errors.submit}
          </div>
        )}

        <div className="form-actions">
          <button
            type="button"
            className="cancel-button"
            onClick={onCancel}
            disabled={loading}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="create-button"
            disabled={loading}
          >
            {loading ? '⏳ Creating...' : '✨ Create Project'}
          </button>
        </div>
      </form>

      <div className="workflow-preview">
        <h3>📋 Project Workflow</h3>
        <div className="workflow-steps">
          {formData.projectType === 'anomaly_detection' ? (
            <>
              <div className="workflow-step">
                <span className="step-number">1</span>
                <span className="step-text">Upload normal images (no defects)</span>
              </div>
              <div className="workflow-step">
                <span className="step-number">2</span>
                <span className="step-text">Extract component regions using AI</span>
              </div>
              <div className="workflow-step">
                <span className="step-number">3</span>
                <span className="step-text">Detect anomalies with DINOv2</span>
              </div>
              <div className="workflow-step">
                <span className="step-number">4</span>
                <span className="step-text">Review and correct AI suggestions</span>
              </div>
            </>
          ) : (
            <>
              <div className="workflow-step">
                <span className="step-number">1</span>
                <span className="step-text">Upload training images and annotations</span>
              </div>
              <div className="workflow-step">
                <span className="step-number">2</span>
                <span className="step-text">Train your {formData.projectType === 'object_detection' ? 'YOLO' : 'SAM2'} model</span>
              </div>
              <div className="workflow-step">
                <span className="step-number">3</span>
                <span className="step-text">Automatically annotate new images</span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default ProjectCreation;