import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './MyTraining.css';

function MyTraining() {
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null);
  const [validation, setValidation] = useState(null);
  const [algorithm, setAlgorithm] = useState('');
  const [device, setDevice] = useState('cpu');
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.01);
  const [modelSize, setModelSize] = useState('n');
  const [message, setMessage] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [allTasks, setAllTasks] = useState([]);
  const logsEndRef = useRef(null);
  const pollingInterval = useRef(null);

  // Auto-scroll logs to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [trainingLogs]);

  // Fetch projects and training tasks on component mount
  useEffect(() => {
    fetchProjects();
    fetchAllTasks();
    
    // Cleanup polling on unmount
    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
    };
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await fetch('/api/projects/');
      const data = await response.json();
      setProjects(data.projects || []);
    } catch (error) {
      console.error('Error fetching projects:', error);
    }
  };

  const fetchAllTasks = async () => {
    try {
      const response = await axios.get('/api/training/training-tasks');
      setAllTasks(response.data.tasks || []);
    } catch (error) {
      console.error('Error fetching training tasks:', error);
    }
  };

  const validateProject = async (projectId) => {
    try {
      const response = await fetch(`/api/projects/${projectId}/validate-dataset`);
      const data = await response.json();
      setValidation(data.validation);
      return data.validation;
    } catch (error) {
      console.error('Error validating project:', error);
      setValidation(null);
      return null;
    }
  };

  const handleProjectSelect = async (event) => {
    const projectId = event.target.value;
    if (projectId) {
      const project = projects.find(p => p.project_id === projectId);
      setSelectedProject(project);
      
      // Get algorithm from project (set during project creation)
      if (project.algorithm) {
        setAlgorithm(project.algorithm);
      } else {
        // Fallback for older projects without algorithm field
        const defaultAlgorithms = {
          'anomaly_detection': 'isolation_forest',
          'object_detection': 'yolo_v8',
          'segmentation': 'yolo_v8_seg'
        };
        setAlgorithm(defaultAlgorithms[project.project_type] || 'isolation_forest');
      }
      
      await validateProject(projectId);
    } else {
      setSelectedProject(null);
      setValidation(null);
      setAlgorithm('');
    }
  };


  const fetchTrainingStatus = async (taskId) => {
    try {
      const response = await axios.get(`/api/training/training-status/${taskId}`);
      const status = response.data;
      setTrainingStatus(status);
      setTrainingLogs(status.recent_logs || status.logs || []);
      
      if (status.status === 'completed' || status.status === 'failed') {
        setIsTraining(false);
        setCurrentTaskId(null);
        if (pollingInterval.current) {
          clearInterval(pollingInterval.current);
        }
        fetchAllTasks(); // Refresh task list
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };

  const startPolling = (taskId) => {
    // Clear any existing polling
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
    }
    
    // Start polling every 2 seconds
    pollingInterval.current = setInterval(() => {
      fetchTrainingStatus(taskId);
    }, 2000);
    
    // Initial fetch
    fetchTrainingStatus(taskId);
  };

  const shouldUseCloudTraining = (device, epochs, modelSize) => {
    return device === 'cuda' || device === 'mps' || epochs > 50 || ['l', 'x'].includes(modelSize);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!selectedProject) {
      setMessage('Please select a project');
      return;
    }

    if (!validation || !validation.is_ready) {
      setMessage('Project dataset is not ready for training');
      return;
    }

    const useCloudTraining = shouldUseCloudTraining(device, epochs, modelSize);

    const formData = new FormData();
    formData.append('algorithm', algorithm);
    formData.append('device', device);
    formData.append('epochs', epochs);
    formData.append('batch_size', batchSize);
    formData.append('learning_rate', learningRate);
    formData.append('model_size', modelSize);

    try {
      setIsTraining(true);
      const trainingType = useCloudTraining ? 'cloud (Vertex AI)' : 'local';
      setMessage(`Starting ${trainingType} training...`);
      setTrainingLogs([`📡 Preparing dataset and starting ${trainingType} training...`]);
      
      // Choose endpoint based on training requirements
      const endpoint = useCloudTraining 
        ? `/api/cloud-training/train-project-cloud/${selectedProject.project_id}`
        : `/api/training/train-project/${selectedProject.project_id}`;
      
      const response = await axios.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'started') {
        const taskId = response.data.task_id;
        setCurrentTaskId(taskId);
        const trainingInfo = response.data.training_type === 'vertex_ai' ? ' (Vertex AI)' : '';
        setMessage(`Training started${trainingInfo}! Task ID: ${taskId}`);
        startPolling(taskId);
        fetchAllTasks(); // Refresh task list
      } else {
        setMessage(`Error: ${response.data.message}`);
        setIsTraining(false);
      }
    } catch (error) {
      console.error('Training error:', error);
      const errorMsg = error.response?.data?.detail || error.message;
      
      // If cloud training fails and accelerated device requested, show specific error
      if (useCloudTraining && (device === 'cuda' || device === 'mps')) {
        setMessage(`Cloud training failed: ${errorMsg}. GPU/Accelerated training requires cloud infrastructure.`);
      } else {
        setMessage(`Training failed: ${errorMsg}`);
      }
      setIsTraining(false);
    }
  };

  const stopTraining = async () => {
    if (!currentTaskId) return;
    
    try {
      await axios.post(`/stop-training/${currentTaskId}`);
      setMessage('Training stopped');
      setIsTraining(false);
      setCurrentTaskId(null);
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
      fetchAllTasks();
    } catch (error) {
      console.error('Error stopping training:', error);
      setMessage(`Error stopping training: ${error.message}`);
    }
  };

  const downloadModel = async (taskId) => {
    try {
      const response = await axios.get(`/api/models/${taskId}/download`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `model_${taskId}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading model:', error);
      alert('Error downloading model');
    }
  };

  return (
    <div className="my-training-container">
      <div className="my-training-header">
        <h1>🚀 My Training</h1>
        <p>Train custom models on your annotated datasets</p>
      </div>

      <div className="training-content">
        {/* Step 1 & 2: Project Selection and Dataset Validation */}
        <div className="training-steps-row">
          <div className="step-section">
            <div className="step-header">
              <span className="step-number">1</span>
              <h3>📂 Select Project</h3>
            </div>
            
            <div className="project-selection">
              <select
                id="project"
                value={selectedProject?.project_id || ''}
                onChange={handleProjectSelect}
                disabled={isTraining}
                required
                className="project-dropdown"
              >
                <option value="">-- Choose a project --</option>
                {projects.map((project) => (
                  <option key={project.project_id} value={project.project_id}>
                    {project.project_name}
                  </option>
                ))}
              </select>
              
              {selectedProject && (
                <div className="project-card">
                  <div className="project-info">
                    <h4>{selectedProject.project_name}</h4>
                    <span className={`project-type-badge ${selectedProject.project_type}`}>
                      {selectedProject.project_type === 'object_detection' ? '📦 Object Detection' : 
                       selectedProject.project_type === 'segmentation' ? '🎯 Segmentation' : 
                       '🔍 Anomaly Detection'}
                    </span>
                    <div className="project-stats">
                      <div className="stat">
                        <span className="label">Training Images:</span>
                        <span className="value">{selectedProject.file_counts?.training_images || 0}</span>
                      </div>
                      <div className="stat">
                        <span className="label">Created:</span>
                        <span className="value">{new Date(selectedProject.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="step-section">
            <div className="step-header">
              <span className="step-number">2</span>
              <h3>✅ Dataset Validation</h3>
            </div>
            
            {selectedProject && validation ? (
              <div className="dataset-validation">
                <div className={`validation-status ${validation.is_ready ? 'ready' : 'not-ready'}`}>
                  <div className="status-indicator">
                    <span className="status-icon">
                      {validation.is_ready ? '✅' : '❌'}
                    </span>
                    <span className="status-text">
                      {validation.is_ready ? 'Ready for Training' : 'Dataset Not Ready'}
                    </span>
                  </div>
                </div>
                
                {validation.missing_requirements.length > 0 && (
                  <div className="requirements-box">
                    <h5>⚠️ Missing Requirements:</h5>
                    <ul>
                      {validation.missing_requirements.map((req, index) => (
                        <li key={index}>{req}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {validation.recommendations.length > 0 && (
                  <div className="recommendations-box">
                    <h5>💡 Recommendations:</h5>
                    <ul>
                      {validation.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : selectedProject ? (
              <div className="validation-loading">
                <span>🔄 Validating dataset...</span>
              </div>
            ) : (
              <div className="validation-placeholder">
                <span>Select a project to validate dataset</span>
              </div>
            )}
          </div>
        </div>

        {/* Step 3: Training Configuration */}
        <div className="step-section full-width">
          <div className="step-header">
            <span className="step-number">3</span>
            <h3>⚙️ Training Configuration</h3>
          </div>
          
          <form onSubmit={handleSubmit} className="training-form">
            <div className="config-horizontal">
              <div className="config-group">
                <label>Selected Algorithm</label>
                {selectedProject && algorithm ? (
                  <div className="algorithm-display">
                    <div className="algorithm-badge">
                      {(() => {
                        const algorithmNames = {
                          'isolation_forest': '🌲 Isolation Forest',
                          'one_class_svm': '🎯 One-Class SVM',
                          'local_outlier_factor': '📍 Local Outlier Factor',
                          'autoencoder': '🧠 Autoencoder',
                          'yolo_v8': '🚀 YOLO v8',
                          'yolo_v11': '✨ YOLO v11',
                          'rtdetr': '⚡ RT-DETR',
                          'yolo_v8_seg': '🎭 YOLO v8 Seg',
                          'sam2': '🔥 SAM 2.0',
                          'unet': '🏥 U-Net'
                        };
                        return algorithmNames[algorithm] || `🤖 ${algorithm}`;
                      })()}
                    </div>
                    <div className="algorithm-note">
                      <small>✅ Algorithm was selected during project creation</small>
                    </div>
                  </div>
                ) : (
                  <div className="algorithm-placeholder">
                    <span>Select a project to see its algorithm</span>
                  </div>
                )}
              </div>

              <div className="horizontal-config-row">
                <div className="config-group half-width">
                  <label>Training Device</label>
                  <div className="radio-group horizontal">
                    <label className="radio-option">
                      <input
                        type="radio"
                        name="device"
                        value="cpu"
                        checked={device === 'cpu'}
                        onChange={(e) => setDevice(e.target.value)}
                        disabled={isTraining}
                      />
                      <span>💻 CPU</span>
                    </label>
                    <label className="radio-option">
                      <input
                        type="radio"
                        name="device"
                        value="cuda"
                        checked={device === 'cuda'}
                        onChange={(e) => setDevice(e.target.value)}
                        disabled={isTraining}
                      />
                      <span>🚀 GPU (CUDA)</span>
                    </label>
                    <label className="radio-option">
                      <input
                        type="radio"
                        name="device"
                        value="mps"
                        checked={device === 'mps'}
                        onChange={(e) => setDevice(e.target.value)}
                        disabled={isTraining}
                      />
                      <span>🍎 Apple Silicon</span>
                    </label>
                  </div>
                </div>

                <div className="config-group half-width">
                  <label>Model Size</label>
                  <div className="radio-group horizontal">
                    {['n', 's', 'm', 'l', 'x'].map((size) => (
                      <label key={size} className="radio-option">
                        <input
                          type="radio"
                          name="modelSize"
                          value={size}
                          checked={modelSize === size}
                          onChange={(e) => setModelSize(e.target.value)}
                          disabled={isTraining}
                        />
                        <span>
                          {size === 'n' ? '⚡ Nano' :
                           size === 's' ? '🔹 Small' :
                           size === 'm' ? '🔸 Medium' :
                           size === 'l' ? '🔶 Large' :
                           '🔥 XL'}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <div className="config-group">
                <label>Training Parameters</label>
                <div className="params-grid">
                  <div className="param-item">
                    <label htmlFor="epochs">Epochs</label>
                    <input
                      type="number"
                      id="epochs"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value))}
                      disabled={isTraining}
                      min="1"
                      max="1000"
                    />
                  </div>
                  <div className="param-item">
                    <label htmlFor="batchSize">Batch Size</label>
                    <input
                      type="number"
                      id="batchSize"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value))}
                      disabled={isTraining}
                      min="1"
                      max="128"
                    />
                  </div>
                  <div className="param-item">
                    <label htmlFor="learningRate">Learning Rate</label>
                    <input
                      type="number"
                      id="learningRate"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                      disabled={isTraining}
                      min="0.0001"
                      max="1"
                      step="0.001"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="form-actions">
              <button
                type="submit"
                disabled={isTraining || !selectedProject || !validation?.is_ready}
                className="start-training-btn"
              >
                {isTraining ? '🔄 Training in Progress...' : '🚀 Start Training'}
              </button>
              
              {isTraining && (
                <button
                  type="button"
                  onClick={stopTraining}
                  className="stop-training-btn"
                >
                  ⏹️ Stop Training
                </button>
              )}
            </div>
          </form>

          {message && (
            <div className={`message ${isTraining ? 'info' : 'success'}`}>
              {message}
            </div>
          )}
        </div>

        <div className="training-monitor">
          <h2>📊 Training Progress</h2>
          {trainingLogs.length > 0 && (
            <div className="logs-container">
              <div className="logs-header">
                <h3>Training Logs</h3>
                {trainingStatus && (
                  <span className={`status-badge ${trainingStatus.status}`}>
                    {trainingStatus.status}
                  </span>
                )}
              </div>
              
              {/* Progress Bar */}
              {trainingStatus && (trainingStatus.status === 'in_progress' || trainingStatus.status === 'running') && (
                <div className="progress-section">
                  <div className="progress-info">
                    <span>Epoch {trainingStatus.current_epoch || 0}/{trainingStatus.total_epochs || 0}</span>
                    <span>{Math.round(trainingStatus.progress || 0)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${trainingStatus.progress || 0}%` }}
                    ></div>
                  </div>
                </div>
              )}
              <div className="logs-content">
                {trainingLogs.map((log, index) => (
                  <div key={index} className="log-line">
                    {log}
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          )}
        </div>

        <div className="training-history">
          <h2>📜 Training History</h2>
          {allTasks.length > 0 ? (
            <div className="tasks-list">
              {allTasks.map((task) => (
                <div key={task.task_id} className="task-item">
                  <div className="task-info">
                    <strong>Task ID:</strong> {task.task_id}<br/>
                    <strong>Status:</strong> 
                    <span className={`status-badge ${task.status}`}>
                      {task.status}
                    </span><br/>
                    <strong>Started:</strong> {new Date(task.created_at).toLocaleString()}<br/>
                    {task.completed_at && (
                      <>
                        <strong>Completed:</strong> {new Date(task.completed_at).toLocaleString()}<br/>
                      </>
                    )}
                  </div>
                  <div className="task-actions">
                    {task.status === 'completed' && (
                      <button
                        onClick={() => downloadModel(task.task_id)}
                        className="download-btn"
                      >
                        📥 Download Model
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p>No training tasks found. Start your first training above!</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default MyTraining;