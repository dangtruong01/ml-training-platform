import React from 'react';
import './Upload.css';

const UploadMethodSelector = ({ 
  algorithm, 
  projectType, 
  selectedMethod, 
  onMethodChange, 
  disabled = false 
}) => {
  const getAvailableMethods = () => {
    // Define which upload methods are available for each algorithm
    const methodSupport = {
      // Object Detection - both methods supported, ZIP preferred
      'yolo_v8': ['zip', 'individual'],
      'yolo_v11': ['zip', 'individual'],
      'rtdetr': ['zip', 'individual'],
      
      // Anomaly Detection - individual only (simpler workflow)
      'isolation_forest': ['individual'],
      'one_class_svm': ['individual'],
      'local_outlier_factor': ['individual'],
      'autoencoder': ['individual'],
      
      // Segmentation - both methods supported
      'yolo_v8_seg': ['zip', 'individual'],
      'sam2': ['zip', 'individual'],
      'unet': ['individual']
    };

    return methodSupport[algorithm] || ['individual'];
  };

  const getPreferredMethod = () => {
    const preferences = {
      'yolo_v8': 'zip',
      'yolo_v11': 'zip', 
      'rtdetr': 'zip',
      'yolo_v8_seg': 'zip',
      'sam2': 'zip'
    };
    
    return preferences[algorithm] || 'individual';
  };

  const getMethodInfo = (method) => {
    const methodDetails = {
      zip: {
        icon: '📦',
        title: 'ZIP Dataset Upload',
        description: 'Upload a complete, structured dataset as a ZIP file',
        pros: [
          'Professional workflow',
          'Maintains folder structure', 
          'Faster for large datasets',
          'Works with annotation tools output'
        ],
        cons: [
          'Requires understanding of dataset structure',
          'Need to prepare ZIP file first'
        ],
        bestFor: 'Experienced users with pre-structured datasets'
      },
      individual: {
        icon: '📁',
        title: 'Individual Files Upload',
        description: 'Upload images and annotation files separately',
        pros: [
          'Simple and intuitive',
          'No preparation needed',
          'Good for beginners',
          'Upload files as you have them'
        ],
        cons: [
          'Slower for many files',
          'We organize structure for you'
        ],
        bestFor: 'Beginners or small datasets'
      }
    };

    return methodDetails[method];
  };

  const availableMethods = getAvailableMethods();
  const preferredMethod = getPreferredMethod();

  // If only one method available, auto-select it
  React.useEffect(() => {
    if (availableMethods.length === 1 && selectedMethod !== availableMethods[0]) {
      onMethodChange(availableMethods[0]);
    }
  }, [availableMethods, selectedMethod, onMethodChange]);

  if (availableMethods.length === 1) {
    // If only one method available, show it as selected but not as a choice
    const method = availableMethods[0];
    const info = getMethodInfo(method);
    
    return (
      <div className="upload-method-selector single-method">
        <h3>📤 Upload Method</h3>
        <div className="method-card selected single">
          <div className="method-header">
            <span className="method-icon">{info.icon}</span>
            <div className="method-title">
              <h4>{info.title}</h4>
              <p>{info.description}</p>
            </div>
          </div>
          <div className="method-benefits">
            <strong>Perfect for {algorithm}:</strong>
            <ul>
              {info.pros.slice(0, 2).map((pro, index) => (
                <li key={index}>{pro}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="upload-method-selector">
      <h3>📤 Choose Upload Method</h3>
      <p className="method-intro">
        {algorithm} supports multiple upload methods. Choose the one that works best for you:
      </p>
      
      <div className="methods-grid">
        {availableMethods.map((method) => {
          const info = getMethodInfo(method);
          const isSelected = selectedMethod === method;
          const isPreferred = method === preferredMethod;
          
          return (
            <div
              key={method}
              className={`method-card ${isSelected ? 'selected' : ''} ${isPreferred ? 'preferred' : ''}`}
              onClick={() => !disabled && onMethodChange(method)}
              role="button"
              tabIndex={0}
            >
              {isPreferred && (
                <div className="preferred-badge">
                  ⭐ Recommended
                </div>
              )}
              
              <div className="method-header">
                <span className="method-icon">{info.icon}</span>
                <div className="method-title">
                  <h4>{info.title}</h4>
                  <p>{info.description}</p>
                </div>
              </div>
              
              <div className="method-details">
                <div className="pros-cons">
                  <div className="pros">
                    <strong>✅ Advantages:</strong>
                    <ul>
                      {info.pros.map((pro, index) => (
                        <li key={index}>{pro}</li>
                      ))}
                    </ul>
                  </div>
                  
                  {info.cons.length > 0 && (
                    <div className="cons">
                      <strong>⚠️ Considerations:</strong>
                      <ul>
                        {info.cons.map((con, index) => (
                          <li key={index}>{con}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                
                <div className="best-for">
                  <strong>👥 Best for:</strong> {info.bestFor}
                </div>
              </div>
              
              {isSelected && (
                <div className="selected-indicator">
                  ✅ Selected
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      {preferredMethod && availableMethods.includes(preferredMethod) && selectedMethod !== preferredMethod && (
        <div className="recommendation-notice">
          💡 <strong>Tip:</strong> We recommend <strong>{getMethodInfo(preferredMethod).title}</strong> for {algorithm} 
          as it provides the best training results and workflow efficiency.
        </div>
      )}
    </div>
  );
};

export default UploadMethodSelector;