import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

function Train() {
  const [dataset, setDataset] = useState(null);
  const [task, setTask] = useState('detection');
  const [device, setDevice] = useState('cpu');
  const [message, setMessage] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [allTasks, setAllTasks] = useState([]);
  const logsEndRef = useRef(null);
  const pollingInterval = useRef(null);

  // Anomaly model training state
  const [anomalyTrainingFiles, setAnomalyTrainingFiles] = useState([]);
  const [anomalyProjectName, setAnomalyProjectName] = useState('');
  const [isTrainingAnomaly, setIsTrainingAnomaly] = useState(false);
  const [anomalyMessage, setAnomalyMessage] = useState('');

  // Auto-scroll logs to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [trainingLogs]);

  // Fetch all training tasks on component mount
  useEffect(() => {
    fetchAllTasks();
    
    // Cleanup polling on unmount
    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
    };
  }, []);

  const fetchAllTasks = async () => {
    try {
      const response = await axios.get('/training-tasks');
      setAllTasks(response.data.tasks || []);
    } catch (error) {
      console.error('Error fetching training tasks:', error);
    }
  };

  const fetchTrainingStatus = async (taskId) => {
    try {
      const [statusResponse, logsResponse] = await Promise.all([
        axios.get(`/training-status/${taskId}`),
        axios.get(`/training-logs/${taskId}?lines=50`)
      ]);
      
      setTrainingStatus(statusResponse.data);
      setTrainingLogs(logsResponse.data.logs || []);
      
      // Stop polling if training completed or failed
      if (statusResponse.data.status === 'completed' || statusResponse.data.status === 'failed') {
        setIsTraining(false);
        if (pollingInterval.current) {
          clearInterval(pollingInterval.current);
          pollingInterval.current = null;
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
    
    // Start new polling
    pollingInterval.current = setInterval(() => {
      fetchTrainingStatus(taskId);
    }, 2000); // Poll every 2 seconds
    
    // Fetch immediately
    fetchTrainingStatus(taskId);
  };

  const handleTrain = async () => {
    if (!dataset) {
      setMessage('Please select a dataset zip file.');
      return;
    }

    const endpoint = task === 'detection' ? '/train-detect' : '/train-segment';
    const formData = new FormData();
    formData.append('file', dataset);
    formData.append('device', device);

    setIsTraining(true);
    setMessage('Starting training...');
    setTrainingStatus(null);
    setTrainingLogs([]);

    try {
      const response = await axios.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setMessage(response.data.message);
      setCurrentTaskId(response.data.task_id);
      
      // Start polling for progress
      startPolling(response.data.task_id);
      
    } catch (error) {
      setMessage('Error starting training.');
      setIsTraining(false);
      console.error(error);
    }
  };

  const monitorExistingTask = (taskId) => {
    setCurrentTaskId(taskId);
    setIsTraining(true);
    startPolling(taskId);
  };

  const handleAnomalyModelTraining = async () => {
    if (anomalyTrainingFiles.length === 0) {
      setAnomalyMessage("Please select 'good' images to train the anomaly model.");
      return;
    }

    if (!anomalyProjectName.trim()) {
      setAnomalyMessage("Please provide a project name for the anomaly model.");
      return;
    }

    setIsTrainingAnomaly(true);
    setAnomalyMessage('Training anomaly detection model...');

    try {
      const formData = new FormData();
      anomalyTrainingFiles.forEach(file => formData.append('files', file));
      formData.append('project_name', anomalyProjectName.trim());

      const response = await axios.post('/train-anomaly-model', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setAnomalyMessage(`✅ Anomaly model trained successfully! Model saved at: ${response.data.model_path}`);
      setAnomalyTrainingFiles([]);
      setAnomalyProjectName('');
    } catch (error) {
      setAnomalyMessage(`❌ Failed to train anomaly model: ${error.response?.data?.detail || error.message}`);
      console.error('Error training anomaly model:', error);
    } finally {
      setIsTrainingAnomaly(false);
    }
  };

  const handleAnomalyTrainingFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setAnomalyTrainingFiles(selectedFiles);
    setAnomalyMessage('');
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px' }}>
      <h2>🏋️ Model Training</h2>
      
      {/* Training Setup */}
      <div style={{ 
        backgroundColor: '#f8fafc', 
        border: '1px solid #e2e8f0', 
        borderRadius: '8px',
        padding: '20px',
        marginBottom: '20px'
      }}>
        <h3>Start New Training</h3>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Dataset (.zip): </label>
          <input 
            type="file" 
            accept=".zip"
            onChange={(e) => setDataset(e.target.files[0])}
            disabled={isTraining}
            style={{ padding: '8px', width: '300px' }}
          />
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            Upload a zip file containing images/ and labels/ folders with data.yaml
          </div>
        </div>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Task Type: </label>
          <select 
            value={task} 
            onChange={(e) => setTask(e.target.value)}
            disabled={isTraining}
            style={{ padding: '8px', width: '200px' }}
          >
            <option value="detection">Object Detection</option>
            <option value="segmentation">Image Segmentation</option>
          </select>
        </div>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Training Device: </label>
          <select 
            value={device} 
            onChange={(e) => setDevice(e.target.value)}
            disabled={isTraining}
            style={{ padding: '8px', width: '200px' }}
          >
            <option value="cpu">🖥️ CPU (Compatible, Slower)</option>
            <option value="auto">🚀 Auto-detect (GPU if available)</option>
            <option value="cuda">⚡ GPU/CUDA (Faster, requires NVIDIA GPU)</option>
          </select>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            {device === 'cpu' && '⚠️ CPU training is slower but works on all systems'}
            {device === 'auto' && '🔍 Will automatically use GPU if available, fallback to CPU'}
            {device === 'cuda' && '⚡ Requires NVIDIA GPU with CUDA support'}
          </div>
        </div>
        <button 
          onClick={handleTrain}
          disabled={!dataset || isTraining}
          style={{
            backgroundColor: !dataset || isTraining ? '#ccc' : '#1e3a8a',
            color: 'white',
            padding: '12px 24px',
            border: 'none',
            borderRadius: '6px',
            cursor: !dataset || isTraining ? 'not-allowed' : 'pointer',
            fontSize: '16px'
          }}
        >
          {isTraining ? '🔄 Training in Progress...' : '🚀 Start Training'}
        </button>
        {message && (
          <div style={{ 
            marginTop: '10px', 
            padding: '10px',
            backgroundColor: message.includes('Error') ? '#fee2e2' : '#dcfce7',
            border: `1px solid ${message.includes('Error') ? '#fca5a5' : '#86efac'}`,
            borderRadius: '4px',
            color: message.includes('Error') ? '#dc2626' : '#059669'
          }}>
            {message}
          </div>
        )}
      </div>

      {/* Anomaly Model Training */}
      <div style={{ 
        backgroundColor: '#fef3c7', 
        border: '1px solid #fbbf24', 
        borderRadius: '8px',
        padding: '20px',
        marginBottom: '20px'
      }}>
        <h3>🛡️ Train Anomaly Detection Model</h3>
        <div style={{ 
          fontSize: '14px', 
          color: '#92400e', 
          marginBottom: '15px',
          padding: '10px',
          backgroundColor: '#fef3c7',
          borderRadius: '4px'
        }}>
          <strong>📚 How it works:</strong> Upload 10-50 images of defect-free "good" products. 
          The PatchCore model will learn what "normal" looks like and detect any deviations as anomalies.
        </div>
        
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Project Name:
          </label>
          <input
            type="text"
            value={anomalyProjectName}
            onChange={(e) => setAnomalyProjectName(e.target.value)}
            placeholder="e.g., metal_parts_project"
            disabled={isTrainingAnomaly}
            style={{
              width: '300px',
              padding: '8px',
              border: '1px solid #d1d5db',
              borderRadius: '4px',
              fontSize: '14px'
            }}
          />
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            A unique name for your anomaly detection project
          </div>
        </div>

        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Select "Good" Images (Defect-Free Examples):
          </label>
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleAnomalyTrainingFileChange}
            disabled={isTrainingAnomaly}
            style={{
              width: '400px',
              padding: '8px',
              border: '1px solid #d1d5db',
              borderRadius: '4px',
              fontSize: '14px'
            }}
          />
          {anomalyTrainingFiles.length > 0 && (
            <div style={{ fontSize: '12px', color: '#059669', marginTop: '5px' }}>
              ✅ {anomalyTrainingFiles.length} good image(s) selected
            </div>
          )}
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            Upload 10-50 images of defect-free products. The model will learn what "normal" looks like.
          </div>
        </div>

        <button
          onClick={handleAnomalyModelTraining}
          disabled={isTrainingAnomaly || anomalyTrainingFiles.length === 0 || !anomalyProjectName.trim()}
          style={{
            backgroundColor: isTrainingAnomaly ? '#9ca3af' : '#f59e0b',
            color: 'white',
            padding: '12px 24px',
            border: 'none',
            borderRadius: '6px',
            cursor: isTrainingAnomaly ? 'not-allowed' : 'pointer',
            fontSize: '16px'
          }}
        >
          {isTrainingAnomaly ? '🔄 Training Anomaly Model...' : '🛡️ Train Anomaly Model'}
        </button>

        {anomalyMessage && (
          <div style={{ 
            marginTop: '15px', 
            padding: '10px',
            backgroundColor: anomalyMessage.includes('❌') ? '#fee2e2' : '#dcfce7',
            border: `1px solid ${anomalyMessage.includes('❌') ? '#fca5a5' : '#86efac'}`,
            borderRadius: '4px',
            color: anomalyMessage.includes('❌') ? '#dc2626' : '#059669',
            fontSize: '14px'
          }}>
            {anomalyMessage}
          </div>
        )}
      </div>

      {/* Training Progress */}
      {isTraining && trainingStatus && (
        <div style={{ 
          backgroundColor: '#f0fdf4', 
          border: '1px solid #059669', 
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '20px'
        }}>
          <h3>🔥 Training Progress</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '15px' }}>
            <div>
              <strong>Task ID:</strong> {currentTaskId}
            </div>
            <div>
              <strong>Status:</strong> 
              <span style={{ 
                marginLeft: '8px',
                padding: '2px 8px',
                borderRadius: '4px',
                backgroundColor: trainingStatus.status === 'running' ? '#059669' : 
                                 trainingStatus.status === 'completed' ? '#16a34a' :
                                 trainingStatus.status === 'failed' ? '#dc2626' : '#6b7280',
                color: 'white',
                fontSize: '12px'
              }}>
                {trainingStatus.status?.toUpperCase()}
              </span>
            </div>
            <div>
              <strong>Epoch:</strong> {trainingStatus.current_epoch || 0} / {trainingStatus.total_epochs || 10}
            </div>
            <div>
              <strong>Overall Progress:</strong> {Math.round(trainingStatus.progress || 0)}%
            </div>
            <div>
              <strong>Device:</strong> {device === 'cpu' ? '🖥️ CPU' : device === 'auto' ? '🚀 Auto' : '⚡ GPU'}
            </div>
          </div>
          
          {/* Progress Bar */}
          <div style={{ marginBottom: '15px' }}>
            <div style={{ 
              width: '100%', 
              backgroundColor: '#e5e7eb', 
              borderRadius: '8px',
              height: '12px',
              overflow: 'hidden'
            }}>
              <div style={{ 
                width: `${trainingStatus.progress || 0}%`,
                backgroundColor: '#059669',
                height: '12px',
                borderRadius: '8px',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
        </div>
      )}

      {/* Training Logs */}
      {isTraining && trainingLogs.length > 0 && (
        <div style={{ 
          backgroundColor: '#1f2937', 
          border: '1px solid #374151', 
          borderRadius: '8px',
          padding: '15px',
          marginBottom: '20px'
        }}>
          <h3 style={{ color: '#f9fafb', marginTop: '0' }}>📋 Training Logs</h3>
          <div style={{
            backgroundColor: '#000',
            color: '#10b981',
            padding: '10px',
            borderRadius: '4px',
            height: '300px',
            overflowY: 'auto',
            fontSize: '12px',
            fontFamily: 'monospace'
          }}>
            {trainingLogs.map((log, index) => (
              <div key={index} style={{ marginBottom: '2px' }}>
                {log}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}

      {/* All Training Tasks */}
      {allTasks.length > 0 && (
        <div style={{ 
          backgroundColor: '#f8fafc', 
          border: '1px solid #e2e8f0', 
          borderRadius: '8px',
          padding: '20px'
        }}>
          <h3>📈 Training History</h3>
          <div style={{ display: 'grid', gap: '10px' }}>
            {allTasks.map((taskInfo, index) => (
              <div key={index} style={{
                backgroundColor: 'white',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                padding: '15px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <div>
                  <strong>{taskInfo.task_id}</strong>
                  <div style={{ fontSize: '14px', color: '#6b7280' }}>
                    Status: <span style={{
                      color: taskInfo.status === 'completed' ? '#059669' :
                             taskInfo.status === 'running' ? '#2563eb' :
                             taskInfo.status === 'failed' ? '#dc2626' : '#6b7280'
                    }}>{taskInfo.status}</span> |
                    Progress: {Math.round(taskInfo.progress || 0)}% |
                    Epoch: {taskInfo.current_epoch || 0}/{taskInfo.total_epochs || 0}
                  </div>
                </div>
                {(taskInfo.status === 'running' || taskInfo.status === 'starting') && (
                  <button
                    onClick={() => monitorExistingTask(taskInfo.task_id)}
                    style={{
                      backgroundColor: '#2563eb',
                      color: 'white',
                      padding: '6px 12px',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    📊 Monitor
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default Train;