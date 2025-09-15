import React, { useState, useEffect } from 'react';
import AlgorithmRequirements from './AlgorithmRequirements';
import UploadMethodSelector from './UploadMethodSelector';
import ZipUploadComponent from './ZipUploadComponent';
import IndividualUploadComponent from './IndividualUploadComponent';
import './Upload.css';

const TrainingDataUpload = ({ 
  projectId, 
  algorithm, 
  projectType,
  onUploadComplete,
  onError 
}) => {
  const [selectedMethod, setSelectedMethod] = useState('');
  const [uploadComplete, setUploadComplete] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [currentStep, setCurrentStep] = useState('requirements'); // requirements, method, upload, complete

  // Auto-advance through steps based on algorithm capabilities
  useEffect(() => {
    // For algorithms with only one upload method, skip method selection
    const singleMethodAlgorithms = [
      'isolation_forest', 'one_class_svm', 'local_outlier_factor', 'autoencoder'
    ];
    
    if (singleMethodAlgorithms.includes(algorithm)) {
      setSelectedMethod('individual');
      setCurrentStep('upload');
    } else {
      setCurrentStep('requirements');
    }
  }, [algorithm]);

  const handleMethodSelect = (method) => {
    setSelectedMethod(method);
    setCurrentStep('upload');
  };

  const handleUploadSuccess = (result) => {
    setUploadResult(result);
    setUploadComplete(true);
    setCurrentStep('complete');
    
    if (onUploadComplete) {
      onUploadComplete(result);
    }
  };

  const handleUploadError = (error) => {
    console.error('Upload error:', error);
    if (onError) {
      onError(error);
    }
  };

  const restartUpload = () => {
    setUploadComplete(false);
    setUploadResult(null);
    setSelectedMethod('');
    setCurrentStep('requirements');
  };

  const getStepIndicator = () => {
    const steps = [
      { id: 'requirements', label: 'Requirements', icon: '📋' },
      { id: 'method', label: 'Upload Method', icon: '📤' },
      { id: 'upload', label: 'Upload Data', icon: '📁' },
      { id: 'complete', label: 'Complete', icon: '✅' }
    ];

    // Filter out method step for single-method algorithms
    const filteredSteps = algorithm && ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'autoencoder'].includes(algorithm)
      ? steps.filter(step => step.id !== 'method')
      : steps;

    const currentIndex = filteredSteps.findIndex(step => step.id === currentStep);

    return (
      <div className="step-indicator">
        {filteredSteps.map((step, index) => {
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

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'requirements':
        return (
          <div className="requirements-step">
            <AlgorithmRequirements 
              algorithm={algorithm} 
              projectType={projectType} 
            />
            <div className="step-actions">
              <button 
                className="next-button primary"
                onClick={() => setCurrentStep('method')}
              >
                📤 Continue to Upload Method
              </button>
            </div>
          </div>
        );

      case 'method':
        return (
          <div className="method-step">
            <UploadMethodSelector
              algorithm={algorithm}
              projectType={projectType}
              selectedMethod={selectedMethod}
              onMethodChange={handleMethodSelect}
            />
          </div>
        );

      case 'upload':
        return (
          <div className="upload-step">
            <div className="upload-header">
              <h3>
                {selectedMethod === 'zip' ? '📦 ZIP Dataset Upload' : '📁 Individual Files Upload'}
              </h3>
              <p>
                Algorithm: <strong>{algorithm}</strong> • 
                Method: <strong>{selectedMethod === 'zip' ? 'ZIP Dataset' : 'Individual Files'}</strong>
              </p>
            </div>

            {selectedMethod === 'zip' ? (
              <ZipUploadComponent
                projectId={projectId}
                algorithm={algorithm}
                onUploadSuccess={handleUploadSuccess}
                onUploadError={handleUploadError}
              />
            ) : (
              <IndividualUploadComponent
                projectId={projectId}
                algorithm={algorithm}
                projectType={projectType}
                onUploadSuccess={handleUploadSuccess}
                onUploadError={handleUploadError}
              />
            )}

            <div className="step-actions">
              <button 
                className="back-button"
                onClick={() => setCurrentStep(algorithm && ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'autoencoder'].includes(algorithm) ? 'requirements' : 'method')}
              >
                ← Back
              </button>
            </div>
          </div>
        );

      case 'complete':
        return (
          <div className="complete-step">
            <div className="success-message">
              <div className="success-icon">🎉</div>
              <h3>Upload Complete!</h3>
              <p>Your training data has been successfully uploaded and is ready for training.</p>
            </div>

            <div className="upload-summary">
              <h4>📊 Upload Summary</h4>
              <div className="summary-details">
                <div className="summary-item">
                  <strong>Method:</strong> {uploadResult?.method === 'zip' ? 'ZIP Dataset' : 'Individual Files'}
                </div>
                <div className="summary-item">
                  <strong>Algorithm:</strong> {algorithm}
                </div>
                {uploadResult?.method === 'zip' ? (
                  <>
                    <div className="summary-item">
                      <strong>Dataset:</strong> {uploadResult?.file}
                    </div>
                    {uploadResult?.info && (
                      <>
                        <div className="summary-item">
                          <strong>Training Images:</strong> {uploadResult.info.train_images || 0}
                        </div>
                        <div className="summary-item">
                          <strong>Validation Images:</strong> {uploadResult.info.val_images || 0}
                        </div>
                        <div className="summary-item">
                          <strong>Classes:</strong> {uploadResult.info.num_classes || 0}
                        </div>
                      </>
                    )}
                  </>
                ) : (
                  <>
                    <div className="summary-item">
                      <strong>Training Images:</strong> {uploadResult?.trainingImages || 0}
                    </div>
                    <div className="summary-item">
                      <strong>Annotation Files:</strong> {uploadResult?.annotationFiles || 0}
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="next-steps">
              <h4>🚀 Next Steps</h4>
              <div className="steps-list">
                <div className="next-step">
                  <span className="step-number">1</span>
                  <div className="step-content">
                    <strong>Validate Dataset</strong>
                    <p>Go to My Training to validate your dataset structure</p>
                  </div>
                </div>
                <div className="next-step">
                  <span className="step-number">2</span>
                  <div className="step-content">
                    <strong>Configure Training</strong>
                    <p>Set training parameters (epochs, batch size, device)</p>
                  </div>
                </div>
                <div className="next-step">
                  <span className="step-number">3</span>
                  <div className="step-content">
                    <strong>Start Training</strong>
                    <p>Begin training your {algorithm} model</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="step-actions">
              <button 
                className="restart-button"
                onClick={restartUpload}
              >
                📁 Upload More Data
              </button>
              <button 
                className="continue-button primary"
                onClick={() => window.location.href = '/training'}
              >
                🚀 Go to Training
              </button>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (!algorithm || !projectType) {
    return (
      <div className="training-data-upload error">
        <div className="error-message">
          <h3>⚠️ Missing Information</h3>
          <p>Algorithm and project type are required for data upload.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="training-data-upload">
      <div className="upload-header">
        <h2>📊 Upload Training Data</h2>
        <p>
          Project Type: <strong>{projectType.replace('_', ' ')}</strong> • 
          Algorithm: <strong>{algorithm}</strong>
        </p>
      </div>

      {getStepIndicator()}

      <div className="upload-content">
        {renderCurrentStep()}
      </div>
    </div>
  );
};

export default TrainingDataUpload;