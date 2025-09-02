import React, { useState, useEffect } from 'react';

const TrainingProgress = ({ taskId }) => {
  const [status, setStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!taskId) return;

    const fetchProgress = async () => {
      try {
        const response = await fetch(`/training-status/${taskId}`);
        const data = await response.json();
        setStatus(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching training status:', error);
        setLoading(false);
      }
    };

    const fetchLogs = async () => {
      try {
        const response = await fetch(`/training-logs/${taskId}?lines=20`);
        const data = await response.json();
        setLogs(data.logs || []);
      } catch (error) {
        console.error('Error fetching logs:', error);
      }
    };

    // Initial fetch
    fetchProgress();
    fetchLogs();

    // Poll for updates every 2 seconds if training is running
    const interval = setInterval(() => {
      fetchProgress();
      fetchLogs();
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId]);

  if (loading) {
    return <div className="training-progress">Loading training status...</div>;
  }

  if (!status) {
    return <div className="training-progress">Training status not found</div>;
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return '#2196F3';
      case 'completed': return '#4CAF50';
      case 'failed': return '#F44336';
      default: return '#FF9800';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return '🔄';
      case 'completed': return '✅';
      case 'failed': return '❌';
      default: return '⏳';
    }
  };

  return (
    <div className="training-progress" style={{ padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
      <h3>Training Progress</h3>
      
      {/* Status Header */}
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '15px' }}>
        <span style={{ fontSize: '24px', marginRight: '10px' }}>
          {getStatusIcon(status.status)}
        </span>
        <div>
          <strong style={{ color: getStatusColor(status.status), fontSize: '18px' }}>
            {status.status.toUpperCase()}
          </strong>
          <div style={{ fontSize: '14px', color: '#666' }}>
            Task ID: {taskId}
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      {status.status === 'running' && (
        <div style={{ marginBottom: '15px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
            <span>Epoch {status.current_epoch || 0} / {status.total_epochs || 10}</span>
            <span>{Math.round(status.progress || 0)}%</span>
          </div>
          <div style={{ 
            width: '100%', 
            height: '20px', 
            backgroundColor: '#e0e0e0', 
            borderRadius: '10px',
            overflow: 'hidden'
          }}>
            <div style={{ 
              width: `${status.progress || 0}%`, 
              height: '100%', 
              backgroundColor: '#2196F3',
              transition: 'width 0.3s ease'
            }}></div>
          </div>
        </div>
      )}

      {/* Recent Logs */}
      <div>
        <h4>Recent Logs:</h4>
        <div style={{ 
          backgroundColor: '#f5f5f5', 
          padding: '10px', 
          borderRadius: '4px',
          maxHeight: '300px',
          overflowY: 'auto',
          fontFamily: 'monospace',
          fontSize: '12px'
        }}>
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <div key={index} style={{ marginBottom: '2px' }}>
                {log}
              </div>
            ))
          ) : (
            <div style={{ color: '#666', fontStyle: 'italic' }}>
              No logs available yet...
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ marginTop: '15px', display: 'flex', gap: '10px' }}>
        <button 
          onClick={() => window.location.reload()} 
          style={{ 
            padding: '8px 16px', 
            backgroundColor: '#2196F3', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Refresh
        </button>
        {status.status === 'completed' && (
          <button 
            onClick={() => {
              // Add logic to download results
              window.open(`/training-results/${taskId}`, '_blank');
            }}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: '#4CAF50', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Download Results
          </button>
        )}
      </div>
    </div>
  );
};

export default TrainingProgress;