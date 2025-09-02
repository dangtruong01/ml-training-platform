import React, { useState } from 'react';

function ROIExtractionStep({ projectId, projectName, trainingImagesCount, onStepComplete }) {
  const [extractionState, setExtractionState] = useState({
    // ROI Method Selection
    roiMethod: 'grounding_dino',
    // GroundingDINO settings
    componentDescription: 'metal plate',
    confidenceThreshold: 0.3,
    // Manufacturing segmentation settings
    manufacturingScenario: 'general',
    partMaterial: 'metal',
    fixtureType: 'tray',
    fixtureColor: 'blue',
    // Processing state
    processing: false,
    results: null,
    previewImages: [],
    loadingPreview: false,
    // Defective images
    uploadingDefectiveImages: false,
    defectiveImagesCount: 0
  });
  const [errors, setErrors] = useState({});
  const [defectiveFiles, setDefectiveFiles] = useState([]);

  const handleSettingChange = (setting, value) => {
    setExtractionState(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const handleDefectiveImagesUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    setDefectiveFiles(files);
    setExtractionState(prev => ({ 
      ...prev, 
      uploadingDefectiveImages: true,
      defectiveImagesCount: files.length 
    }));
    setErrors({});

    try {
      const formData = new FormData();
      files.forEach((file, index) => {
        formData.append(`defective_images`, file);
      });

      console.log(`📤 Uploading ${files.length} defective images...`);

      const response = await fetch(`/api/auto-annotation/projects/${projectId}/upload-defective-images`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.status === 'success') {
        console.log('✅ Defective images uploaded successfully');
        setExtractionState(prev => ({ 
          ...prev, 
          defectiveImagesCount: result.uploaded_count 
        }));
      } else {
        setErrors({ defectiveUpload: result.message || 'Failed to upload defective images' });
      }
    } catch (error) {
      console.error('❌ Defective images upload failed:', error);
      setErrors({ defectiveUpload: 'Network error during upload' });
    } finally {
      setExtractionState(prev => ({ ...prev, uploadingDefectiveImages: false }));
    }
  };

  const startROIExtraction = async () => {
    setExtractionState(prev => ({ ...prev, processing: true }));
    setErrors({});

    try {
      const formData = new FormData();
      formData.append('roi_method', extractionState.roiMethod);

      if (extractionState.roiMethod === 'grounding_dino') {
        formData.append('component_description', extractionState.componentDescription);
        formData.append('confidence_threshold', extractionState.confidenceThreshold);
      } else if (extractionState.roiMethod === 'manufacturing_segmentation') {
        formData.append('manufacturing_scenario', extractionState.manufacturingScenario);
        formData.append('part_material', extractionState.partMaterial);
        formData.append('fixture_type', extractionState.fixtureType);
        formData.append('fixture_color', extractionState.fixtureColor);
      } else if (extractionState.roiMethod === 'segmentation_mask') {
        formData.append('component_description', extractionState.componentDescription);
        formData.append('confidence_threshold', extractionState.confidenceThreshold);
      }

      const response = await fetch(`/api/auto-annotation/projects/${projectId}/extract-roi`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.status === 'success') {
        setExtractionState(prev => ({
          ...prev,
          results: result
        }));
        console.log('✅ ROI extraction completed:', result);
        
        // Auto-load preview after successful extraction
        setTimeout(() => {
          loadROIPreview();
        }, 500);
      } else {
        setErrors({ extraction: result.message || 'ROI extraction failed' });
      }
    } catch (error) {
      console.error('ROI extraction failed:', error);
      setErrors({ extraction: 'Network error. Please try again.' });
    } finally {
      setExtractionState(prev => ({ ...prev, processing: false }));
    }
  };

  const getSuccessRate = () => {
    if (!extractionState.results) return 0;
    return (extractionState.results.successful_extractions / extractionState.results.total_images) * 100;
  };

  const loadROIPreview = async () => {
    setExtractionState(prev => ({ ...prev, loadingPreview: true }));

    try {
      const response = await fetch(`/api/auto-annotation/projects/${projectId}/roi-preview`);
      const result = await response.json();

      if (result.status === 'success') {
        setExtractionState(prev => ({
          ...prev,
          previewImages: result.preview_images
        }));
      } else {
        console.warn('Failed to load ROI preview:', result.message);
      }
    } catch (error) {
      console.error('Error loading ROI preview:', error);
    } finally {
      setExtractionState(prev => ({ ...prev, loadingPreview: false }));
    }
  };

  const handleProceedToNextStep = () => {
    if (onStepComplete) {
      onStepComplete(extractionState.results);
    }
  };

  return (
    <div className="roi-extraction-step">
      <div className="step-header">
        <h4>🎯 Step 1: Upload Defective Images & Extract ROI</h4>
        <p>Upload defective images and extract ROI from both training ({trainingImagesCount}) and defective images using your chosen method.</p>
      </div>

      {/* Defective Images Upload */}
      <div className="defective-images-section" style={{ marginBottom: '30px', padding: '20px', border: '2px dashed #ddd', borderRadius: '8px', background: '#fafafa' }}>
        <h5>📤 Upload Defective Images</h5>
        <p style={{ color: '#666', marginBottom: '15px' }}>
          Upload images containing defects that you want to detect and annotate.
        </p>

        <div className="defective-upload-area">
          <input
            type="file"
            id="defective-images-upload"
            multiple
            accept="image/*"
            onChange={handleDefectiveImagesUpload}
            disabled={extractionState.uploadingDefectiveImages}
            style={{ display: 'none' }}
          />
          <label 
            htmlFor="defective-images-upload" 
            className="upload-button"
            style={{
              display: 'inline-block',
              padding: '12px 24px',
              background: extractionState.uploadingDefectiveImages ? '#6c757d' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: extractionState.uploadingDefectiveImages ? 'wait' : 'pointer',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
          >
            {extractionState.uploadingDefectiveImages 
              ? '⏳ Uploading...' 
              : '📁 Select Defective Images'
            }
          </label>

          {extractionState.defectiveImagesCount > 0 && (
            <div style={{ marginTop: '10px', color: '#28a745', fontWeight: 'bold' }}>
              ✅ {extractionState.defectiveImagesCount} defective images uploaded
            </div>
          )}

          {defectiveFiles.length > 0 && (
            <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
              Selected: {defectiveFiles.map(f => f.name).join(', ').substring(0, 100)}
              {defectiveFiles.map(f => f.name).join(', ').length > 100 && '...'}
            </div>
          )}
        </div>

        {errors.defectiveUpload && (
          <div style={{ marginTop: '10px', padding: '8px', background: '#fee', border: '1px solid #fcc', borderRadius: '4px', color: '#c33' }}>
            ❌ {errors.defectiveUpload}
          </div>
        )}
      </div>

      {/* Settings */}
      <div className="extraction-settings">
        <h5>⚙️ ROI Extraction Settings</h5>
        
        {/* ROI Method Selection */}
        <div className="roi-method-selection" style={{ marginBottom: '25px' }}>
          <div className="setting-item">
            <label style={{ display: 'block', marginBottom: '10px', fontWeight: 'bold' }}>ROI Extraction Method</label>
            <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="radio"
                  name="roiMethod"
                  value="grounding_dino"
                  checked={extractionState.roiMethod === 'grounding_dino'}
                  onChange={(e) => handleSettingChange('roiMethod', e.target.value)}
                  disabled={extractionState.processing}
                />
                <span>🎯 GroundingDINO</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="radio"
                  name="roiMethod"
                  value="manufacturing_segmentation"
                  checked={extractionState.roiMethod === 'manufacturing_segmentation'}
                  onChange={(e) => handleSettingChange('roiMethod', e.target.value)}
                  disabled={extractionState.processing}
                />
                <span>🏭 Manufacturing Segmentation</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="radio"
                  name="roiMethod"
                  value="segmentation_mask"
                  checked={extractionState.roiMethod === 'segmentation_mask'}
                  onChange={(e) => handleSettingChange('roiMethod', e.target.value)}
                  disabled={extractionState.processing}
                />
                <span>🎭 Segmentation Mask (GroundingDINO + SAM2)</span>
              </label>
            </div>
            <span className="setting-help" style={{ display: 'block', marginTop: '5px', fontSize: '12px', color: '#666' }}>
              Choose ROI extraction method: AI detection (boxes), manufacturing edges, or precise segmentation masks
            </span>
          </div>
        </div>
        
        {/* GroundingDINO Settings */}
        {extractionState.roiMethod === 'grounding_dino' && (
          <div className="grounding-dino-settings">
            <h6 style={{ marginBottom: '15px', color: '#495057' }}>🎯 GroundingDINO Configuration</h6>
            <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div className="setting-item">
                <label>Component Description</label>
                <input
                  type="text"
                  value={extractionState.componentDescription}
                  onChange={(e) => handleSettingChange('componentDescription', e.target.value)}
                  placeholder="e.g., metal plate, circuit board, component"
                  disabled={extractionState.processing}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                />
                <span className="setting-help">Describe what GroundingDINO should detect</span>
              </div>
              
              <div className="setting-item">
                <label>Confidence Threshold</label>
                <div className="threshold-control">
                  <input
                    type="range"
                    value={extractionState.confidenceThreshold}
                    onChange={(e) => handleSettingChange('confidenceThreshold', parseFloat(e.target.value))}
                    min="0.1"
                    max="0.8"
                    step="0.1"
                    disabled={extractionState.processing}
                    style={{ width: '70%' }}
                  />
                  <span className="threshold-value" style={{ marginLeft: '10px', fontWeight: 'bold' }}>
                    {extractionState.confidenceThreshold}
                  </span>
                </div>
                <span className="setting-help">Detection confidence threshold (lower = more detections)</span>
              </div>
            </div>
          </div>
        )}

        {/* Manufacturing Segmentation Settings */}
        {extractionState.roiMethod === 'manufacturing_segmentation' && (
          <div className="manufacturing-settings">
            <h6 style={{ marginBottom: '15px', color: '#495057' }}>🏭 Manufacturing Segmentation Configuration</h6>
            <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              
              <div className="setting-item">
                <label>Manufacturing Scenario</label>
                <select
                  value={extractionState.manufacturingScenario}
                  onChange={(e) => handleSettingChange('manufacturingScenario', e.target.value)}
                  disabled={extractionState.processing}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="general">General Manufacturing</option>
                  <option value="metal_machining">Metal Machining</option>
                  <option value="electronics">Electronics Assembly</option>
                  <option value="automotive">Automotive Parts</option>
                  <option value="textile">Textile/Fabric</option>
                </select>
                <span className="setting-help">Select your manufacturing use case</span>
              </div>

              <div className="setting-item">
                <label>Part Material</label>
                <select
                  value={extractionState.partMaterial}
                  onChange={(e) => handleSettingChange('partMaterial', e.target.value)}
                  disabled={extractionState.processing}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="metal">Metal</option>
                  <option value="plastic">Plastic</option>
                  <option value="ceramic">Ceramic</option>
                  <option value="fabric">Fabric</option>
                  <option value="glass">Glass</option>
                  <option value="mixed">Mixed Materials</option>
                </select>
                <span className="setting-help">Primary material of the parts</span>
              </div>

              <div className="setting-item">
                <label>Fixture Type</label>
                <select
                  value={extractionState.fixtureType}
                  onChange={(e) => handleSettingChange('fixtureType', e.target.value)}
                  disabled={extractionState.processing}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="tray">Tray/Platform</option>
                  <option value="conveyor">Conveyor Belt</option>
                  <option value="fixture">Custom Fixture</option>
                  <option value="none">No Fixture</option>
                </select>
                <span className="setting-help">Type of fixture or background</span>
              </div>

              <div className="setting-item">
                <label>Fixture Color</label>
                <select
                  value={extractionState.fixtureColor}
                  onChange={(e) => handleSettingChange('fixtureColor', e.target.value)}
                  disabled={extractionState.processing}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="blue">Blue</option>
                  <option value="black">Black</option>
                  <option value="white">White</option>
                  <option value="gray">Gray</option>
                  <option value="green">Green</option>
                  <option value="red">Red</option>
                  <option value="mixed">Mixed/Variable</option>
                </select>
                <span className="setting-help">Dominant color of the fixture</span>
              </div>
            </div>
          </div>
        )}

        {/* Segmentation Mask Settings */}
        {extractionState.roiMethod === 'segmentation_mask' && (
          <div className="segmentation-settings">
            <h6 style={{ marginBottom: '15px', color: '#495057' }}>🎭 Segmentation Mask Configuration</h6>
            <p style={{ fontSize: '14px', color: '#666', marginBottom: '15px', padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '5px' }}>
              <strong>GroundingDINO + SAM2 Pipeline:</strong><br/>
              1. GroundingDINO detects component using text prompt<br/>
              2. SAM2 generates precise segmentation mask<br/>
              3. Extracts only component pixels (background removed)
            </p>
            <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div className="setting-item">
                <label className="setting-label">🎯 Component Description</label>
                <input
                  type="text"
                  value={extractionState.componentDescription}
                  onChange={(e) => handleSettingChange('componentDescription', e.target.value)}
                  disabled={extractionState.processing}
                  placeholder="e.g., metal plate, circuit board"
                  className="setting-input"
                />
                <span className="setting-help">Describe the component to segment</span>
              </div>
              <div className="setting-item">
                <label className="setting-label">🎚️ Confidence Threshold</label>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <input
                    type="range"
                    value={extractionState.confidenceThreshold}
                    onChange={(e) => handleSettingChange('confidenceThreshold', parseFloat(e.target.value))}
                    min="0.1"
                    max="0.8"
                    step="0.1"
                    disabled={extractionState.processing}
                    style={{ width: '70%' }}
                  />
                  <span className="threshold-value" style={{ marginLeft: '10px', fontWeight: 'bold' }}>
                    {extractionState.confidenceThreshold}
                  </span>
                </div>
                <span className="setting-help">Detection confidence for GroundingDINO</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Action Button */}
      <div className="extraction-actions">
        {errors.extraction && (
          <div className="error-banner">
            ❌ {errors.extraction}
          </div>
        )}
        
        <button
          className="start-extraction-button"
          onClick={startROIExtraction}
          disabled={extractionState.processing || extractionState.defectiveImagesCount === 0}
          style={{
            padding: '15px 30px',
            background: (extractionState.processing || extractionState.defectiveImagesCount === 0) ? '#6c757d' : '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: (extractionState.processing || extractionState.defectiveImagesCount === 0) ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
            opacity: (extractionState.processing || extractionState.defectiveImagesCount === 0) ? 0.6 : 1
          }}
        >
          {extractionState.processing 
            ? '⏳ Extracting ROI from both image sets...' 
            : `🚀 Extract ROI (${
                extractionState.roiMethod === 'grounding_dino' ? 'GroundingDINO' :
                extractionState.roiMethod === 'segmentation_mask' ? 'Segmentation Mask' :
                'Manufacturing'
              })`
          }
        </button>
        
        {extractionState.defectiveImagesCount === 0 && !extractionState.processing && (
          <p style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
            Upload defective images first
          </p>
        )}
        
        {extractionState.defectiveImagesCount > 0 && (
          <p style={{ fontSize: '12px', color: '#28a745', marginTop: '8px' }}>
            ✅ Ready to extract ROI from training ({trainingImagesCount}) + defective ({extractionState.defectiveImagesCount}) images
          </p>
        )}
      </div>

      {/* Results */}
      {extractionState.results && (
        <div className="extraction-results">
          <div className="results-header">
            <h5>📊 ROI Extraction Results</h5>
          </div>
          
          <div className="results-summary">
            <div className="summary-item">
              <span className="summary-label">Total Images:</span>
              <span className="summary-value">{extractionState.results.total_images}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Successful:</span>
              <span className="summary-value success">{extractionState.results.successful_extractions}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Failed:</span>
              <span className="summary-value error">{extractionState.results.failed_extractions}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Success Rate:</span>
              <span className={`summary-value ${getSuccessRate() > 80 ? 'success' : getSuccessRate() > 50 ? 'warning' : 'error'}`}>
                {getSuccessRate().toFixed(1)}%
              </span>
            </div>
          </div>

          {extractionState.results.average_roi_size && (
            <div className="roi-stats">
              <h6>📏 Average ROI Size</h6>
              <p>{extractionState.results.average_roi_size[0]} × {extractionState.results.average_roi_size[1]} pixels</p>
            </div>
          )}

          {getSuccessRate() < 70 && (
            <div className="warning-banner">
              ⚠️ Low success rate detected. Consider:
              <ul>
                <li>Lowering the confidence threshold</li>
                <li>Adjusting the component description</li>
                <li>Checking if images contain the specified component</li>
              </ul>
            </div>
          )}

          {getSuccessRate() >= 70 && (
            <div className="success-banner">
              ✅ Good ROI extraction! Ready for anomaly detection analysis.
              <button
                className="next-step-button"
                onClick={handleProceedToNextStep}
                style={{ marginLeft: '15px', padding: '8px 16px', background: '#28a745', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
              >
                Proceed to Step 2 →
              </button>
            </div>
          )}
        </div>
      )}

      {/* ROI Preview */}
      {extractionState.results && extractionState.results.successful_extractions > 0 && (
        <div className="roi-preview-section">
          <div className="preview-header">
            <h5>🖼️ Extracted ROI Preview</h5>
            {!extractionState.previewImages.length && (
              <button
                className="load-preview-button"
                onClick={loadROIPreview}
                disabled={extractionState.loadingPreview}
              >
                {extractionState.loadingPreview ? '⏳ Loading...' : '📸 Load Preview'}
              </button>
            )}
          </div>
          
          {extractionState.previewImages.length > 0 && (
            <div className="preview-grid">
              {extractionState.previewImages.map((preview, index) => (
                <div key={index} className="preview-item">
                  <img
                    src={`data:image/jpeg;base64,${preview.image_base64}`}
                    alt={`ROI ${index + 1}`}
                    className="preview-image"
                  />
                  <div className="preview-filename">{preview.filename.substring(13)}</div>
                </div>
              ))}
            </div>
          )}
          
          <p className="preview-note">
            📝 Showing sample extracted ROI regions. These cropped areas will be used for anomaly detection.
          </p>
        </div>
      )}

      {/* Info */}
      <div className="step-info">
        <h5>ℹ️ About ROI Extraction</h5>
        <div className="info-points">
          <div className="info-point">
            <span className="point-icon">🎯</span>
            <span className="point-text">Identifies component regions using GroundingDINO</span>
          </div>
          <div className="info-point">
            <span className="point-icon">✂️</span>
            <span className="point-text">Crops out noise (trays, bolts, QR codes)</span>
          </div>
          <div className="info-point">
            <span className="point-icon">📏</span>
            <span className="point-text">Adds padding for better anomaly detection</span>
          </div>
          <div className="info-point">
            <span className="point-icon">🎯</span>
            <span className="point-text">Prepares clean data for next steps</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ROIExtractionStep;