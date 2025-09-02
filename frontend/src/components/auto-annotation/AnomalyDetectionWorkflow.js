import React, { useState } from 'react';
import ROIExtractionStep from './ROIExtractionStep';
import BuildNormalModelStep from './BuildNormalModelStep';
import DetectDefectsStep from './DetectDefectsStep';
import GenerateBoxesStep from './GenerateBoxesStep';
import AnomalyDetectionStep from './AnomalyDetectionStep';
import './AnomalyDetectionWorkflow.css';

function AnomalyDetectionWorkflow({ selectedProject, onBack }) {
  const [currentStep, setCurrentStep] = useState(1);
  const [stepData, setStepData] = useState({
    step1: null, // Normal ROI extraction results
    step2: null, // Normal model building results
    step3: null, // Defective images anomaly detection results
    step4: null  // Bounding box generation results
  });

  const steps = [
    {
      id: 1,
      title: "Extract Normal ROI",
      description: "Process normal images to build baseline",
      status: currentStep >= 1 ? (currentStep > 1 ? 'completed' : 'active') : 'pending'
    },
    {
      id: 2,
      title: "Build Normal Model",
      description: "Create statistical model of normal patterns",
      status: currentStep >= 2 ? (currentStep > 2 ? 'completed' : 'active') : 'pending'
    },
    {
      id: 3,
      title: "Detect Defects",
      description: "Analyze defective images for anomalies",
      status: currentStep >= 3 ? (currentStep > 3 ? 'completed' : 'active') : 'pending'
    },
    {
      id: 4,
      title: "Generate Boxes",
      description: "Convert anomalies to YOLO annotations",
      status: currentStep >= 4 ? 'active' : 'pending'
    }
  ];

  const getStepIcon = (status) => {
    switch (status) {
      case 'completed': return '✅';
      case 'active': return '🔄';
      case 'pending': return '⏳';
      default: return '⚪';
    }
  };

  const handleStepComplete = (step, data) => {
    console.log(`✅ Step ${step} completed:`, data);
    
    setStepData(prev => ({
      ...prev,
      [`step${step}`]: data
    }));
    
    // Move to next step
    setCurrentStep(step + 1);
  };

  const canProceedToStep = (stepNumber) => {
    if (stepNumber === 1) return true;
    if (stepNumber === 2) {
      // Allow Step 2 if Step 1 data exists OR if ROI extraction was previously completed
      return stepData.step1 !== null || checkROIExtractionCompleted();
    }
    if (stepNumber === 3) {
      // Allow Step 3 if Step 2 data exists OR if normal model was previously built
      return stepData.step2 !== null || checkNormalModelCompleted();
    }
    if (stepNumber === 4) return stepData.step3 !== null;
    return false;
  };

  const checkROIExtractionCompleted = () => {
    // This could be enhanced to check the backend for ROI cache existence
    // For now, we'll allow proceeding to Step 2 if user wants to retry
    return true; // Always allow access to Step 2 for now
  };

  const checkNormalModelCompleted = () => {
    // This could be enhanced to check the backend for normal model existence
    // For now, we'll allow proceeding to Step 3 if user wants to retry
    return true; // Always allow access to Step 3 for now
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <ROIExtractionStep
            projectId={selectedProject?.project_id}
            projectName={selectedProject?.project_name}
            trainingImagesCount={selectedProject?.training_images_count || 0}
            onStepComplete={(data) => handleStepComplete(1, data)}
          />
        );
      case 2:
        return (
          <BuildNormalModelStep
            projectId={selectedProject?.project_id}
            roiData={stepData.step1}
            onStepComplete={(data) => handleStepComplete(2, data)}
          />
        );
      case 3:
        return (
          <DetectDefectsStep
            projectId={selectedProject?.project_id}
            normalModelData={stepData.step2}
            onStepComplete={(data) => handleStepComplete(3, data)}
          />
        );
      case 4:
        return (
          <GenerateBoxesStep
            projectId={selectedProject?.project_id}
            anomalyResults={stepData.step3}
            onStepComplete={(data) => handleStepComplete(4, data)}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="anomaly-detection-workflow">
      {/* Header */}
      <div className="workflow-header">
        <button className="back-button" onClick={onBack}>
          ← Back to Project
        </button>
        <div className="workflow-title">
          <h2>🧠 Anomaly Detection Workflow</h2>
          <p>Project: <strong>{selectedProject?.project_name}</strong></p>
        </div>
      </div>

      {/* Workflow Steps Progress */}
      <div className="workflow-progress">
        <div className="steps-container">
          {steps.map((step, index) => (
            <div 
              key={step.id} 
              className={`step-item ${step.status} ${canProceedToStep(step.id) ? 'clickable' : ''}`}
              onClick={() => canProceedToStep(step.id) && setCurrentStep(step.id)}
            >
              <div className="step-number">
                {getStepIcon(step.status)} {step.id}
              </div>
              <div className="step-content">
                <h4 className="step-title">{step.title}</h4>
                <p className="step-description">{step.description}</p>
              </div>
              {index < steps.length - 1 && <div className="step-connector" />}
            </div>
          ))}
        </div>
        
        {/* Quick Navigation */}
        {currentStep === 1 && (
          <div className="quick-navigation">
            <p className="nav-helper">💡 Already completed ROI extraction?</p>
            <button 
              className="skip-button"
              onClick={() => setCurrentStep(2)}
            >
              Skip to Step 2: Build Normal Model →
            </button>
          </div>
        )}
        
        {currentStep === 2 && (
          <div className="quick-navigation">
            <p className="nav-helper">💡 Already built your normal model?</p>
            <button 
              className="skip-button"
              onClick={() => setCurrentStep(3)}
            >
              Skip to Step 3: Detect Defects →
            </button>
          </div>
        )}
        
        {currentStep === 3 && (
          <div className="quick-navigation">
            <p className="nav-helper">💡 Already completed defect detection?</p>
            <button 
              className="skip-button"
              onClick={() => setCurrentStep(4)}
            >
              Skip to Step 4: Generate Boxes →
            </button>
          </div>
        )}
      </div>

      {/* Current Step Content */}
      <div className="current-step-content">
        {renderCurrentStep()}
      </div>

      {/* Workflow Info */}
      <div className="workflow-info">
        <h5>🎯 4-Stage Anomaly Detection → YOLO Training</h5>
        <div className="info-comparison">
          <div className="approach-comparison">
            <div className="approach-item" style={{ background: '#f0f8ff', borderLeft: '4px solid #007bff' }}>
              <h6>🔄 How It Works</h6>
              <ul>
                <li><strong>Stage 1:</strong> Extract ROI from normal/good images → build baseline</li>
                <li><strong>Stage 2:</strong> Build statistical model using DINOv2 features</li>
                <li><strong>Stage 3:</strong> Upload defective images → detect anomalies</li>
                <li><strong>Stage 4:</strong> Convert anomaly heatmaps to YOLO annotations</li>
              </ul>
            </div>
            <div className="approach-item anomaly">
              <h6>✅ End Result</h6>
              <ul>
                <li>Automatic bounding box generation for defects</li>
                <li>YOLO-ready annotation files (.txt format)</li>
                <li>No manual labeling of defective images needed</li>
                <li>Ready for YOLO model training!</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnomalyDetectionWorkflow;