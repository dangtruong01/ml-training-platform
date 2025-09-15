import React from 'react';
import './Upload.css';

const AlgorithmRequirements = ({ algorithm, projectType }) => {
  const getAlgorithmInfo = () => {
    const algorithmSpecs = {
      // Object Detection Algorithms
      'yolo_v8': {
        name: '🚀 YOLO v8',
        description: 'State-of-the-art object detection. Fast and accurate.',
        preferredMethod: 'zip',
        fileFormat: '.pt',
        requirements: {
          zip: {
            structure: 'Complete YOLO dataset',
            files: ['data.yaml', 'images/train/', 'images/val/', 'labels/train/', 'labels/val/'],
            example: 'yolo_dataset.zip'
          },
          individual: {
            structure: 'Individual files (we\'ll organize for you)',
            files: ['Training images (.jpg, .png)', 'Label files (.txt, YOLO format)'],
            note: 'One .txt file per image with same filename'
          }
        }
      },
      'yolo_v11': {
        name: '✨ YOLO v11',
        description: 'Latest YOLO version. Improved accuracy and efficiency.',
        preferredMethod: 'zip',
        fileFormat: '.pt',
        requirements: {
          zip: {
            structure: 'Complete YOLO dataset',
            files: ['data.yaml', 'images/train/', 'images/val/', 'labels/train/', 'labels/val/'],
            example: 'yolo_dataset.zip'
          },
          individual: {
            structure: 'Individual files (we\'ll organize for you)',
            files: ['Training images (.jpg, .png)', 'Label files (.txt, YOLO format)'],
            note: 'One .txt file per image with same filename'
          }
        }
      },
      'rtdetr': {
        name: '⚡ RT-DETR',
        description: 'Real-time transformer detection. High accuracy.',
        preferredMethod: 'zip',
        fileFormat: '.pt',
        requirements: {
          zip: {
            structure: 'Complete YOLO dataset',
            files: ['data.yaml', 'images/train/', 'images/val/', 'labels/train/', 'labels/val/'],
            example: 'yolo_dataset.zip'
          },
          individual: {
            structure: 'Individual files (we\'ll organize for you)',
            files: ['Training images (.jpg, .png)', 'Label files (.txt, YOLO format)'],
            note: 'One .txt file per image with same filename'
          }
        }
      },
      // Anomaly Detection Algorithms
      'isolation_forest': {
        name: '🌲 Isolation Forest',
        description: 'Fast, unsupervised anomaly detection. Good for manufacturing defects.',
        preferredMethod: 'individual',
        fileFormat: '.pkl',
        requirements: {
          individual: {
            structure: 'Normal training images',
            files: ['Training images (.jpg, .png) - normal samples only'],
            note: 'Minimum 10 images recommended'
          }
        }
      },
      'one_class_svm': {
        name: '🎯 One-Class SVM',
        description: 'Robust to outliers. Better for complex patterns.',
        preferredMethod: 'individual',
        fileFormat: '.pkl',
        requirements: {
          individual: {
            structure: 'Normal training images',
            files: ['Training images (.jpg, .png) - normal samples only'],
            note: 'Minimum 15 images recommended for better accuracy'
          }
        }
      },
      'local_outlier_factor': {
        name: '📍 Local Outlier Factor',
        description: 'Local density-based detection. Good for irregular patterns.',
        preferredMethod: 'individual',
        fileFormat: '.pkl',
        requirements: {
          individual: {
            structure: 'Normal training images',
            files: ['Training images (.jpg, .png) - normal samples only'],
            note: 'Works well with 20+ diverse normal samples'
          }
        }
      },
      'autoencoder': {
        name: '🧠 Autoencoder',
        description: 'Deep learning approach. Best for image anomalies.',
        preferredMethod: 'individual',
        fileFormat: '.pth',
        requirements: {
          individual: {
            structure: 'Normal training images',
            files: ['Training images (.jpg, .png) - normal samples only'],
            note: 'Requires 50+ images for good reconstruction learning'
          }
        }
      },
      // Segmentation Algorithms
      'yolo_v8_seg': {
        name: '🎭 YOLO v8 Segmentation',
        description: 'Pixel-level segmentation. Precise boundary detection.',
        preferredMethod: 'zip',
        fileFormat: '.pt',
        requirements: {
          zip: {
            structure: 'Complete YOLO segmentation dataset',
            files: ['data.yaml', 'images/train/', 'images/val/', 'labels/train/', 'labels/val/'],
            note: 'Label files contain polygon coordinates'
          },
          individual: {
            structure: 'Individual files',
            files: ['Training images (.jpg, .png)', 'Segmentation masks or polygon files'],
            note: 'Masks should match image names'
          }
        }
      }
    };

    return algorithmSpecs[algorithm] || {
      name: `🤖 ${algorithm}`,
      description: 'Custom algorithm',
      preferredMethod: 'individual',
      fileFormat: '.pt',
      requirements: {
        individual: {
          structure: 'Training data',
          files: ['Training files'],
          note: 'Check algorithm documentation'
        }
      }
    };
  };

  const algorithmInfo = getAlgorithmInfo();

  const getMethodRequirements = (method) => {
    return algorithmInfo.requirements[method];
  };

  return (
    <div className="algorithm-requirements">
      <div className="algorithm-header">
        <h3>{algorithmInfo.name}</h3>
        <p className="algorithm-description">{algorithmInfo.description}</p>
        <div className="algorithm-meta">
          <span className="file-format">Output: {algorithmInfo.fileFormat}</span>
          <span className="preferred-method">
            Recommended: {algorithmInfo.preferredMethod === 'zip' ? '📦 ZIP Upload' : '📁 Individual Files'}
          </span>
        </div>
      </div>

      <div className="requirements-section">
        <h4>📋 Data Requirements</h4>
        
        {Object.entries(algorithmInfo.requirements).map(([method, req]) => (
          <div key={method} className="method-requirements">
            <h5>
              {method === 'zip' ? '📦 ZIP Dataset Method' : '📁 Individual Files Method'}
              {algorithmInfo.preferredMethod === method && (
                <span className="recommended-badge">Recommended</span>
              )}
            </h5>
            
            <div className="requirement-details">
              <p className="structure">{req.structure}</p>
              
              <div className="files-list">
                <strong>Required files:</strong>
                <ul>
                  {req.files.map((file, index) => (
                    <li key={index}>{file}</li>
                  ))}
                </ul>
              </div>
              
              {req.note && (
                <div className="requirement-note">
                  <strong>Note:</strong> {req.note}
                </div>
              )}
              
              {req.example && (
                <div className="example">
                  <strong>Example:</strong> {req.example}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {projectType === 'object_detection' && (
        <div className="yolo-format-info">
          <h4>📖 YOLO Format Guide</h4>
          <div className="format-example">
            <strong>data.yaml example:</strong>
            <pre>{`path: /path/to/dataset
train: images/train
val: images/val
nc: 2
names: ['defect', 'scratch']`}</pre>
            
            <strong>Label file (.txt) example:</strong>
            <pre>{`0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2`}</pre>
            <small>Format: class_id center_x center_y width height (normalized 0-1)</small>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlgorithmRequirements;