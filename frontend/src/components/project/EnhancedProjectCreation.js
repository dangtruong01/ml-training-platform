import React, { useState } from 'react';
import './Project.css';

function EnhancedProjectCreation({ onProjectCreated, onCancel }) {
  const [currentStep, setCurrentStep] = useState('project'); // project, algorithm, complete
  const [projectData, setProjectData] = useState(null);
  const [formData, setFormData] = useState({
    projectName: '',
    projectType: 'object_detection',
    algorithm: '',
    description: ''
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  // Get available algorithms based on project type
  const getAvailableAlgorithms = (projectType) => {
    const algorithmsByType = {
      'object_detection': [
        { value: 'yolo_v8', label: 'YOLO v8', description: '🚀 State-of-the-art object detection. Fast and accurate.', recommended: true },
        { value: 'yolo_v11', label: 'YOLO v11', description: '✨ Latest YOLO version. Improved accuracy and efficiency.' },
        { value: 'rtdetr', label: 'RT-DETR', description: '⚡ Real-time transformer detection. High accuracy.' }
      ],
      'anomaly_detection': [
        { value: 'isolation_forest', label: 'Isolation Forest', description: '🌲 Fast, unsupervised anomaly detection. Good for manufacturing defects.', recommended: true },
        { value: 'one_class_svm', label: 'One-Class SVM', description: '🎯 Robust to outliers. Better for complex patterns.' },
        { value: 'local_outlier_factor', label: 'Local Outlier Factor', description: '📍 Local density-based detection. Good for irregular patterns.' },
        { value: 'autoencoder', label: 'Autoencoder', description: '🧠 Deep learning approach. Best for image anomalies.' }
      ],
      'segmentation': [
        { value: 'yolo_v8_seg', label: 'YOLO v8 Segmentation', description: '🎭 Pixel-level segmentation. Precise boundary detection.', recommended: true },
        { value: 'sam2', label: 'SAM 2.0', description: '🔥 Meta\'s Segment Anything v2. Universal segmentation.' },
        { value: 'unet', label: 'U-Net', description: '🏥 Medical-grade segmentation. High precision boundaries.' }
      ]
    };

    return algorithmsByType[projectType] || [];
  };

  // Auto-select recommended algorithm when project type changes
  React.useEffect(() => {
    const algorithms = getAvailableAlgorithms(formData.projectType);
    const recommended = algorithms.find(alg => alg.recommended);
    if (recommended && !formData.algorithm) {
      setFormData(prev => ({ ...prev, algorithm: recommended.value }));
    }
  }, [formData.projectType]);

  const validateProjectForm = () => {
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

  const validateAlgorithmForm = () => {
    const newErrors = {};
    
    if (!formData.algorithm) {
      newErrors.algorithm = 'Algorithm selection is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleProjectNext = () => {
    if (validateProjectForm()) {
      setCurrentStep('algorithm');
    }
  };

  const handleAlgorithmNext = () => {
    if (validateAlgorithmForm()) {
      createProject();
    }
  };

  const createProject = async () => {
    setLoading(true);
    try {
      const formDataToSend = new FormData();
      formDataToSend.append('project_name', formData.projectName);
      formDataToSend.append('project_type', formData.projectType);
      formDataToSend.append('description', formData.description);

      const response = await fetch('/api/projects/create', {
        method: 'POST',
        body: formDataToSend
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setProjectData({
          ...result,
          algorithm: formData.algorithm
        });
        setCurrentStep('complete');
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

  const getStepIndicator = () => {
    const steps = [
      { id: 'project', label: 'Project Details', icon: '📋' },
      { id: 'algorithm', label: 'Algorithm Selection', icon: '🤖' },
      { id: 'complete', label: 'Complete', icon: '✅' }
    ];

    const currentIndex = steps.findIndex(step => step.id === currentStep);

    return (
      <div className="step-indicator">
        {steps.map((step, index) => {
          const isActive = step.id === currentStep;
          const isCompleted = index < currentIndex;
          const isAccessible = index <= currentIndex;

          return (
            <div 
              key={step.id} 
              className={`step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''} ${isAccessible ? 'accessible' : ''}`}
            >
              <div className="step-icon">
                {isCompleted ? '✅' : step.icon}
              </div>
              <span className="step-label">{step.label}</span>
            </div>
          );
        })}
      </div>
    );
  };

  const renderProjectStep = () => (
    <div className="project-step">
      <div className="step-header">
        <h2>🆕 Create New Project</h2>
        <p>Set up your machine learning project with the right configuration</p>
      </div>

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
              <p>Detect and locate objects with bounding boxes</p>
              <ul>
                <li>✅ Fast training and inference</li>
                <li>✅ Good for defect counting</li>
                <li>✅ YOLO algorithms available</li>
              </ul>
            </div>
          </div>

          <div 
            className={`type-option ${formData.projectType === 'segmentation' ? 'selected' : ''}`}
            onClick={() => setFormData(prev => ({ ...prev, projectType: 'segmentation' }))}
          >
            <div className="type-icon">🎯</div>
            <div className="type-info">
              <h4>Segmentation</h4>
              <p>Precise pixel-level object boundaries</p>
              <ul>
                <li>✅ Precise defect boundaries</li>
                <li>✅ Measurement capabilities</li>
                <li>✅ Advanced algorithms</li>
              </ul>
            </div>
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
                <li>✅ Multiple algorithms</li>
              </ul>
            </div>
          </div>
        </div>
        {errors.projectType && (
          <span className="error-message">{errors.projectType}</span>
        )}
      </div>

      <div className="form-group">
        <label htmlFor="description">Description</label>
        <textarea
          id="description"
          name="description"
          value={formData.description}
          onChange={handleInputChange}
          placeholder="Describe your project (optional)"
          rows="3"
        />
      </div>

      <div className="step-actions">
        <button
          type="button"
          className="cancel-button"
          onClick={onCancel}
          disabled={loading}
        >
          Cancel
        </button>
        <button
          type="button"
          className="next-button primary"
          onClick={handleProjectNext}
          disabled={loading}
        >
          Next: Choose Algorithm →
        </button>
      </div>
    </div>
  );

  const renderAlgorithmStep = () => {
    const algorithms = getAvailableAlgorithms(formData.projectType);

    return (
      <div className="algorithm-step">
        <div className="step-header">
          <h2>🤖 Choose Algorithm</h2>
          <p>Select the best algorithm for your {formData.projectType.replace('_', ' ')} project</p>
        </div>

        <div className="algorithm-selection">
          {algorithms.map((algorithm) => (
            <div
              key={algorithm.value}
              className={`algorithm-option ${formData.algorithm === algorithm.value ? 'selected' : ''} ${algorithm.recommended ? 'recommended' : ''}`}
              onClick={() => setFormData(prev => ({ ...prev, algorithm: algorithm.value }))}
            >
              {algorithm.recommended && (
                <div className="recommended-badge">⭐ Recommended</div>
              )}
              <div className="algorithm-header">
                <h4>{algorithm.label}</h4>
                <p>{algorithm.description}</p>
              </div>
              <input
                type="radio"
                name="algorithm"
                value={algorithm.value}
                checked={formData.algorithm === algorithm.value}
                onChange={handleInputChange}
              />
            </div>
          ))}
        </div>
        {errors.algorithm && (
          <span className="error-message">{errors.algorithm}</span>
        )}

        {errors.submit && (
          <div className="error-banner">
            ❌ {errors.submit}
          </div>
        )}

        <div className="step-actions">
          <button
            type="button"
            className="back-button"
            onClick={() => setCurrentStep('project')}
            disabled={loading}
          >
            ← Back
          </button>
          <button
            type="button"
            className="next-button primary"
            onClick={handleAlgorithmNext}
            disabled={loading || !formData.algorithm}
          >
            {loading ? '⏳ Creating Project...' : 'Create Project →'}
          </button>
        </div>
      </div>
    );
  };


  const renderCompleteStep = () => (
    <div className="complete-step">
      <div className="success-message">
        <div className="success-icon">🎉</div>
        <h2>Project Created Successfully!</h2>
        <p>Your project <strong>{formData.projectName}</strong> has been created. You can now upload data and start training.</p>
      </div>

      <div className="project-summary">
        <h3>📊 Project Summary</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <strong>Project Name:</strong> {formData.projectName}
          </div>
          <div className="summary-item">
            <strong>Type:</strong> {formData.projectType.replace('_', ' ')}
          </div>
          <div className="summary-item">
            <strong>Algorithm:</strong> {formData.algorithm}
          </div>
          <div className="summary-item">
            <strong>Project ID:</strong> {projectData?.project_id}
          </div>
        </div>
      </div>

      <div className="next-steps">
        <h3>🚀 What's Next?</h3>
        <div className="steps-list">
          <div className="next-step">
            <span className="step-number">1</span>
            <div className="step-content">
              <strong>Upload Training Data</strong>
              <p>Add your training images and annotations to the project</p>
            </div>
          </div>
          <div className="next-step">
            <span className="step-number">2</span>
            <div className="step-content">
              <strong>Train Your Model</strong>
              <p>Start training your {formData.algorithm} model with uploaded data</p>
            </div>
          </div>
          <div className="next-step">
            <span className="step-number">3</span>
            <div className="step-content">
              <strong>Monitor & Deploy</strong>
              <p>Track progress and deploy your trained model</p>
            </div>
          </div>
        </div>
      </div>

      <div className="step-actions">
        <button
          type="button"
          className="secondary-button"
          onClick={() => window.location.href = '/my-data'}
        >
          📋 View All Projects
        </button>
        <button
          type="button"
          className="primary-button"
          onClick={() => {
            if (onProjectCreated) {
              onProjectCreated({
                ...projectData,
                algorithm: formData.algorithm
              });
            }
          }}
        >
          📤 Upload Data Now
        </button>
      </div>
    </div>
  );

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'project':
        return renderProjectStep();
      case 'algorithm':
        return renderAlgorithmStep();
      case 'complete':
        return renderCompleteStep();
      default:
        return null;
    }
  };

  return (
    <div className="enhanced-project-creation">
      {getStepIndicator()}
      <div className="step-content">
        {renderCurrentStep()}
      </div>
    </div>
  );
}

export default EnhancedProjectCreation;