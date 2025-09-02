import React, { useState, useEffect } from 'react';

function GenerateBoxesStep({ projectId, anomalyResults, onStepComplete }) {
  const [boxGenerationState, setBoxGenerationState] = useState({
    // Annotation type selection
    annotationType: 'bounding_boxes', // 'bounding_boxes' or 'segmentation_masks'
    generating: false,
    boxResults: null,
    visualPreview: null,
    loadingPreviousResults: false,
    workflowStatus: null,
    settings: {
      minBoxSize: 32,
      anomalyThreshold: 0.1,
      mergeNearbyBoxes: true,
      outputFormat: 'yolo',
      // Edge filtering settings
      excludeEdges: true,
      edgeMargin: 10,
      // Color filtering settings
      enableColorFiltering: true,
      centerZoneRatio: 0.6,
      minCenterVariance: 0.05,
      maxBorderUniformity: 0.02,
      minCenterBorderDiff: 0.1,
      backgroundMatchThreshold: 0.7,
      colorToleranceHsv: 15,
      // Manufacturing segmentation settings
      enableManufacturingSegmentation: false,
      manufacturingScenario: 'general',
      partMaterial: 'metal',
      fixtureType: 'tray',
      fixtureColor: 'blue'
    }
  });
  const [errors, setErrors] = useState({});
  const [effectiveAnomalyResults, setEffectiveAnomalyResults] = useState(anomalyResults);

  // Check for previous results on component mount
  useEffect(() => {
    checkWorkflowStatus();
  }, [projectId]);

  // Update effective anomaly results when prop changes
  useEffect(() => {
    setEffectiveAnomalyResults(anomalyResults);
  }, [anomalyResults]);

  const checkWorkflowStatus = async () => {
    if (!projectId) return;

    console.log('🔍 Checking workflow status from server...');
    setBoxGenerationState(prev => ({ ...prev, loadingPreviousResults: true }));

    try {
      const response = await fetch(`/api/auto-annotation/projects/${projectId}/workflow-status`);
      const result = await response.json();

      if (result.status === 'success') {
        const workflowStatus = result.workflow_status;
        
        console.log('📊 Workflow status loaded:', workflowStatus);
        
        setBoxGenerationState(prev => ({
          ...prev,
          workflowStatus: workflowStatus,
          loadingPreviousResults: false
        }));

        // If Stage 3 is completed but we don't have anomalyResults prop, use server data
        if (workflowStatus.stage3_completed && !anomalyResults && workflowStatus.stage3_results) {
          console.log('✅ Found existing Stage 3 results, enabling Stage 4');
          setEffectiveAnomalyResults(workflowStatus.stage3_results);
        }

        // If Stage 4 is completed, load existing results
        if (workflowStatus.stage4_completed) {
          console.log('✅ Stage 4 already completed, loading existing results');
          // Could load existing Stage 4 results here if needed
        }
      } else {
        console.log('⚠️ Could not load workflow status:', result.message);
        setBoxGenerationState(prev => ({ ...prev, loadingPreviousResults: false }));
      }
    } catch (error) {
      console.error('❌ Error checking workflow status:', error);
      setBoxGenerationState(prev => ({ ...prev, loadingPreviousResults: false }));
    }
  };

  const handleSettingChange = (setting, value) => {
    setBoxGenerationState(prev => ({
      ...prev,
      settings: {
        ...prev.settings,
        [setting]: value
      }
    }));
  };

  const fetchVisualPreview = async () => {
    try {
      console.log('🖼️ Fetching visual bounding box preview...');
      const response = await fetch(`/api/auto-annotation/projects/${projectId}/visual-boxes-preview?limit=6`);
      const result = await response.json();
      
      if (result.status === 'success') {
        setBoxGenerationState(prev => ({
          ...prev,
          visualPreview: result
        }));
        console.log(`✅ Loaded ${result.total_previews} visual preview images`);
      } else {
        console.log('⚠️ No visual preview available:', result.message);
      }
    } catch (error) {
      console.error('❌ Failed to fetch visual preview:', error);
    }
  };

  const generateAnnotations = async () => {
    if (!effectiveAnomalyResults) {
      setErrors({ generation: 'No anomaly detection results available. Please complete Stage 3 first.' });
      return;
    }

    const isSegmentationMode = boxGenerationState.annotationType === 'segmentation_masks';
    console.log(`📦 Starting ${isSegmentationMode ? 'segmentation mask' : 'bounding box'} generation...`);
    setBoxGenerationState(prev => ({ ...prev, generating: true }));
    setErrors({});

    try {
      const formData = new FormData();
      
      if (isSegmentationMode) {
        // Segmentation mask parameters
        formData.append('min_mask_area', boxGenerationState.settings.minBoxSize * 4); // Convert box size to area
        formData.append('anomaly_threshold', boxGenerationState.settings.anomalyThreshold);
        formData.append('output_format', boxGenerationState.settings.outputFormat);
        formData.append('exclude_edges', boxGenerationState.settings.excludeEdges);
        formData.append('edge_margin', boxGenerationState.settings.edgeMargin);
        formData.append('simplify_polygons', true);
        formData.append('polygon_epsilon', 0.01);
      } else {
        // Bounding box parameters
        formData.append('min_box_size', boxGenerationState.settings.minBoxSize);
        formData.append('anomaly_threshold', boxGenerationState.settings.anomalyThreshold);
        formData.append('merge_nearby_boxes', boxGenerationState.settings.mergeNearbyBoxes);
        formData.append('output_format', boxGenerationState.settings.outputFormat);
        
        // Edge filtering parameters
        formData.append('exclude_edges', boxGenerationState.settings.excludeEdges);
        formData.append('edge_margin', boxGenerationState.settings.edgeMargin);
        
        // Color filtering parameters
        formData.append('enable_color_filtering', boxGenerationState.settings.enableColorFiltering);
        formData.append('center_zone_ratio', boxGenerationState.settings.centerZoneRatio);
        formData.append('min_center_variance', boxGenerationState.settings.minCenterVariance);
        formData.append('max_border_uniformity', boxGenerationState.settings.maxBorderUniformity);
        formData.append('min_center_border_diff', boxGenerationState.settings.minCenterBorderDiff);
        formData.append('background_match_threshold', boxGenerationState.settings.backgroundMatchThreshold);
        formData.append('color_tolerance_hsv', boxGenerationState.settings.colorToleranceHsv);
        
        // Manufacturing segmentation parameters
        formData.append('enable_manufacturing_segmentation', boxGenerationState.settings.enableManufacturingSegmentation);
        formData.append('manufacturing_scenario', boxGenerationState.settings.manufacturingScenario);
        formData.append('part_material', boxGenerationState.settings.partMaterial);
        formData.append('fixture_type', boxGenerationState.settings.fixtureType);
        formData.append('fixture_color', boxGenerationState.settings.fixtureColor);
      }

      const endpoint = isSegmentationMode ? 'generate-segmentation-masks' : 'generate-bounding-boxes';
      console.log(`🌐 Request URL: /api/auto-annotation/projects/${projectId}/${endpoint}`);

      const response = await fetch(`/api/auto-annotation/projects/${projectId}/${endpoint}`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      console.log(`📡 ${isSegmentationMode ? 'Mask' : 'Box'} generation response:`, result);

      if (result.status === 'success') {
        setBoxGenerationState(prev => ({
          ...prev,
          boxResults: result
        }));
        
        if (isSegmentationMode) {
          console.log(`✅ Successfully generated ${result.total_masks_generated} segmentation masks`);
        } else {
          console.log(`✅ Successfully generated ${result.total_boxes_generated} bounding boxes`);
          // Fetch visual preview images for boxes
          fetchVisualPreview();
        }
      } else {
        setErrors({ generation: result.message || `${isSegmentationMode ? 'Mask' : 'Box'} generation failed` });
      }
    } catch (error) {
      console.error('❌ Box generation failed:', error);
      setErrors({ generation: `Generation error: ${error.message}` });
    } finally {
      setBoxGenerationState(prev => ({ ...prev, generating: false }));
    }
  };

  const handleProceedToNextStep = () => {
    if (onStepComplete) {
      onStepComplete(boxGenerationState.boxResults);
    }
  };

  const downloadAnnotations = () => {
    // Create download functionality for the generated annotations
    if (boxGenerationState.boxResults) {
      const annotationType = boxGenerationState.annotationType;
      const downloadUrl = `/api/auto-annotation/projects/${projectId}/download-annotations?annotation_type=${annotationType}`;
      window.open(downloadUrl, '_blank');
    }
  };

  return (
    <div className="generate-boxes-step">
      <div className="step-header">
        <h4>📦 Step 4: Generate Annotations</h4>
        <p>Convert anomaly heatmaps into YOLO-ready annotations for training. Choose between bounding boxes or segmentation masks.</p>
      </div>

      {/* Annotation Type Selection */}
      <div className="annotation-type-selection" style={{ marginBottom: '30px', padding: '20px', background: '#f8f9fa', borderRadius: '8px', border: '2px solid #dee2e6' }}>
        <h5 style={{ marginBottom: '15px', color: '#495057' }}>🎯 Annotation Type</h5>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', padding: '10px 15px', background: boxGenerationState.annotationType === 'bounding_boxes' ? '#e3f2fd' : 'transparent', borderRadius: '6px', border: boxGenerationState.annotationType === 'bounding_boxes' ? '2px solid #2196f3' : '2px solid transparent' }}>
            <input
              type="radio"
              name="annotationType"
              value="bounding_boxes"
              checked={boxGenerationState.annotationType === 'bounding_boxes'}
              onChange={(e) => setBoxGenerationState(prev => ({ ...prev, annotationType: e.target.value }))}
              disabled={boxGenerationState.generating}
            />
            <span style={{ fontWeight: 'bold' }}>📦 Bounding Boxes</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', padding: '10px 15px', background: boxGenerationState.annotationType === 'segmentation_masks' ? '#e8f5e8' : 'transparent', borderRadius: '6px', border: boxGenerationState.annotationType === 'segmentation_masks' ? '2px solid #4caf50' : '2px solid transparent' }}>
            <input
              type="radio"
              name="annotationType"
              value="segmentation_masks"
              checked={boxGenerationState.annotationType === 'segmentation_masks'}
              onChange={(e) => setBoxGenerationState(prev => ({ ...prev, annotationType: e.target.value }))}
              disabled={boxGenerationState.generating}
            />
            <span style={{ fontWeight: 'bold' }}>🎭 Segmentation Masks</span>
          </label>
        </div>
        <div style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
          {boxGenerationState.annotationType === 'bounding_boxes' ? 
            '📦 Creates rectangular bounding boxes around defect regions - standard YOLO format' : 
            '🎭 Creates precise polygon masks following defect contours - YOLO segmentation format'
          }
        </div>
      </div>

      {/* Workflow Status Banner */}
      {boxGenerationState.loadingPreviousResults && (
        <div className="status-banner" style={{ padding: '15px', background: '#fff3cd', border: '1px solid #ffeaa7', borderRadius: '8px', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ fontSize: '16px' }}>🔍</div>
            <div>
              <div style={{ fontWeight: 'bold' }}>Checking Previous Results...</div>
              <div style={{ fontSize: '14px', color: '#856404' }}>Looking for completed stages on the server</div>
            </div>
          </div>
        </div>
      )}

      {boxGenerationState.workflowStatus && (
        <div className="workflow-status-banner" style={{ padding: '15px', background: effectiveAnomalyResults ? '#d4edda' : '#f8d7da', border: `1px solid ${effectiveAnomalyResults ? '#c3e6cb' : '#f5c6cb'}`, borderRadius: '8px', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ fontSize: '16px' }}>{effectiveAnomalyResults ? '✅' : '⚠️'}</div>
            <div>
              {effectiveAnomalyResults ? (
                <>
                  <div style={{ fontWeight: 'bold', color: '#155724' }}>Stage 3 Results Found!</div>
                  <div style={{ fontSize: '14px', color: '#155724' }}>
                    Loaded existing defect detection results with {effectiveAnomalyResults.total_images || 0} images processed
                  </div>
                </>
              ) : (
                <>
                  <div style={{ fontWeight: 'bold', color: '#721c24' }}>Stage 3 Not Completed</div>
                  <div style={{ fontSize: '14px', color: '#721c24' }}>
                    Please complete Stage 3 (Detect Defects) first, or use the skip button to navigate back
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Settings Section */}
      <div className="box-generation-settings">
        <h5>⚙️ Generation Settings</h5>
        <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
          
          <div className="setting-item">
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Minimum Box Size (pixels)
            </label>
            <input
              type="number"
              value={boxGenerationState.settings.minBoxSize}
              onChange={(e) => handleSettingChange('minBoxSize', parseInt(e.target.value))}
              min="16"
              max="128"
              disabled={boxGenerationState.generating}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            />
            <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              Minimum size for generated bounding boxes
            </p>
          </div>

          <div className="setting-item">
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Anomaly Threshold
            </label>
            <div className="threshold-control">
              <input
                type="range"
                value={boxGenerationState.settings.anomalyThreshold}
                onChange={(e) => handleSettingChange('anomalyThreshold', parseFloat(e.target.value))}
                min="0.05"
                max="0.5"
                step="0.01"
                disabled={boxGenerationState.generating}
                style={{ width: '70%' }}
              />
              <span style={{ marginLeft: '10px', fontWeight: 'bold' }}>
                {(boxGenerationState.settings.anomalyThreshold * 100).toFixed(1)}%
              </span>
            </div>
            <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              Controls region sensitivity within each image (all defective images are processed)
            </p>
          </div>

          <div className="setting-item">
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input
                type="checkbox"
                checked={boxGenerationState.settings.mergeNearbyBoxes}
                onChange={(e) => handleSettingChange('mergeNearbyBoxes', e.target.checked)}
                disabled={boxGenerationState.generating}
              />
              <span style={{ fontWeight: 'bold' }}>Merge Nearby Boxes</span>
            </label>
            <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              Combine overlapping bounding boxes
            </p>
          </div>

          <div className="setting-item">
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Output Format
            </label>
            <select
              value={boxGenerationState.settings.outputFormat}
              onChange={(e) => handleSettingChange('outputFormat', e.target.value)}
              disabled={boxGenerationState.generating}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            >
              <option value="yolo">YOLO Format (.txt)</option>
              <option value="coco">COCO Format (.json)</option>
            </select>
            <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
              Annotation format for training
            </p>
          </div>
        </div>

        {/* Advanced Filtering Settings - Only for Bounding Boxes */}
        {boxGenerationState.annotationType === 'bounding_boxes' && (
          <div className="advanced-filtering-section" style={{ marginTop: '25px' }}>
            <h6>🔧 Advanced Filtering Settings</h6>
            <p style={{ fontSize: '14px', color: '#666', marginBottom: '15px' }}>
              Advanced options to filter out edge artifacts and background detections (Bounding Box mode only)
            </p>

          {/* Edge Filtering */}
          <div className="filtering-subsection" style={{ marginBottom: '20px', padding: '15px', background: '#f8f9fa', borderRadius: '8px' }}>
            <h6 style={{ marginBottom: '15px', color: '#495057' }}>🚫 Edge Filtering</h6>
            <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              
              <div className="setting-item">
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={boxGenerationState.settings.excludeEdges}
                    onChange={(e) => handleSettingChange('excludeEdges', e.target.checked)}
                    disabled={boxGenerationState.generating}
                  />
                  <span style={{ fontWeight: 'bold' }}>Exclude Edge Boxes</span>
                </label>
                <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                  Skip bounding boxes that touch image edges (prevents tray/fixture detection)
                </p>
              </div>

              <div className="setting-item">
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                  Edge Margin (pixels)
                </label>
                <input
                  type="number"
                  value={boxGenerationState.settings.edgeMargin}
                  onChange={(e) => handleSettingChange('edgeMargin', parseInt(e.target.value))}
                  min="5"
                  max="50"
                  disabled={boxGenerationState.generating || !boxGenerationState.settings.excludeEdges}
                  style={{
                    width: '100%',
                    padding: '8px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    opacity: !boxGenerationState.settings.excludeEdges ? 0.5 : 1
                  }}
                />
                <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                  Distance from edge to consider as boundary
                </p>
              </div>
            </div>
          </div>

          {/* Color Filtering */}
          <div className="filtering-subsection" style={{ marginBottom: '20px', padding: '15px', background: '#f0f8ff', borderRadius: '8px' }}>
            <h6 style={{ marginBottom: '15px', color: '#495057' }}>🎨 Color-Based Filtering</h6>
            
            <div className="setting-item" style={{ marginBottom: '15px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  checked={boxGenerationState.settings.enableColorFiltering}
                  onChange={(e) => handleSettingChange('enableColorFiltering', e.target.checked)}
                  disabled={boxGenerationState.generating}
                />
                <span style={{ fontWeight: 'bold' }}>Enable Color Filtering</span>
              </label>
              <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                Filter out boxes based on color analysis (center vs edge contrast + background matching)
              </p>
            </div>

            {boxGenerationState.settings.enableColorFiltering && (
              <div className="color-settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                
                <div className="setting-item">
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Center Zone Ratio
                  </label>
                  <div className="threshold-control">
                    <input
                      type="range"
                      value={boxGenerationState.settings.centerZoneRatio}
                      onChange={(e) => handleSettingChange('centerZoneRatio', parseFloat(e.target.value))}
                      min="0.4"
                      max="0.8"
                      step="0.05"
                      disabled={boxGenerationState.generating}
                      style={{ width: '70%' }}
                    />
                    <span style={{ marginLeft: '10px', fontWeight: 'bold' }}>
                      {(boxGenerationState.settings.centerZoneRatio * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                    Inner area percentage for center vs edge analysis
                  </p>
                </div>

                <div className="setting-item">
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Min Center Variance
                  </label>
                  <input
                    type="number"
                    value={boxGenerationState.settings.minCenterVariance}
                    onChange={(e) => handleSettingChange('minCenterVariance', parseFloat(e.target.value))}
                    min="0.01"
                    max="0.2"
                    step="0.01"
                    disabled={boxGenerationState.generating}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  />
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                    Minimum color variance required for real defects
                  </p>
                </div>

                <div className="setting-item">
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Background Match Threshold
                  </label>
                  <div className="threshold-control">
                    <input
                      type="range"
                      value={boxGenerationState.settings.backgroundMatchThreshold}
                      onChange={(e) => handleSettingChange('backgroundMatchThreshold', parseFloat(e.target.value))}
                      min="0.5"
                      max="0.9"
                      step="0.05"
                      disabled={boxGenerationState.generating}
                      style={{ width: '70%' }}
                    />
                    <span style={{ marginLeft: '10px', fontWeight: 'bold' }}>
                      {(boxGenerationState.settings.backgroundMatchThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                    Reject boxes if this % of pixels match background colors
                  </p>
                </div>

                <div className="setting-item">
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Color Tolerance (HSV)
                  </label>
                  <input
                    type="number"
                    value={boxGenerationState.settings.colorToleranceHsv}
                    onChange={(e) => handleSettingChange('colorToleranceHsv', parseInt(e.target.value))}
                    min="5"
                    max="30"
                    disabled={boxGenerationState.generating}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  />
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                    HSV hue tolerance for color matching (degrees)
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Manufacturing Segmentation */}
          <div className="filtering-subsection" style={{ marginBottom: '20px', padding: '15px', background: '#f0f8f0', borderRadius: '8px' }}> 
            <h6 style={{ marginBottom: '15px', color: '#495057' }}>🏭 Manufacturing-Specific Segmentation</h6>
            
            <div className="setting-item" style={{ marginBottom: '15px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  checked={boxGenerationState.settings.enableManufacturingSegmentation}
                  onChange={(e) => handleSettingChange('enableManufacturingSegmentation', e.target.checked)}
                  disabled={boxGenerationState.generating}
                />
                <span style={{ fontWeight: 'bold' }}>Enable Manufacturing Segmentation</span>
              </label>
              <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                Use specialized segmentation to separate parts from fixtures/backgrounds
              </p>
            </div>

            {boxGenerationState.settings.enableManufacturingSegmentation && (
              <div className="manufacturing-settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                
                <div className="setting-item">
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Manufacturing Scenario
                  </label>
                  <select
                    value={boxGenerationState.settings.manufacturingScenario}
                    onChange={(e) => handleSettingChange('manufacturingScenario', e.target.value)}
                    disabled={boxGenerationState.generating}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="general">General Manufacturing</option>
                    <option value="metal_machining">Metal Machining</option>
                    <option value="electronics">Electronics Assembly</option>
                    <option value="automotive">Automotive Parts</option>
                    <option value="textile">Textile/Fabric</option>
                  </select>
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                    Select your manufacturing use case for optimized segmentation
                  </p>
                </div>

                <div className="setting-item">
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Part Material
                  </label>
                  <select
                    value={boxGenerationState.settings.partMaterial}
                    onChange={(e) => handleSettingChange('partMaterial', e.target.value)}
                    disabled={boxGenerationState.generating}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="metal">Metal</option>
                    <option value="plastic">Plastic</option>
                    <option value="ceramic">Ceramic</option>
                    <option value="fabric">Fabric</option>
                    <option value="glass">Glass</option>
                    <option value="mixed">Mixed Materials</option>
                  </select>
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                    Primary material of the parts being inspected
                  </p>
                </div>

                <div className="setting-item">
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Fixture Type
                  </label>
                  <select
                    value={boxGenerationState.settings.fixtureType}
                    onChange={(e) => handleSettingChange('fixtureType', e.target.value)}
                    disabled={boxGenerationState.generating}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="tray">Tray/Platform</option>
                    <option value="conveyor">Conveyor Belt</option>
                    <option value="fixture">Custom Fixture</option>
                    <option value="none">No Fixture</option>
                  </select>
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                    Type of fixture or background holding the parts
                  </p>
                </div>

                <div className="setting-item">
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Fixture Color
                  </label>
                  <select
                    value={boxGenerationState.settings.fixtureColor}
                    onChange={(e) => handleSettingChange('fixtureColor', e.target.value)}
                    disabled={boxGenerationState.generating}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="blue">Blue</option>
                    <option value="black">Black</option>
                    <option value="white">White</option>
                    <option value="gray">Gray</option>
                    <option value="green">Green</option>
                    <option value="red">Red</option>
                    <option value="mixed">Mixed/Variable</option>
                  </select>
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                    Dominant color of the fixture or background
                  </p>
                </div>
              </div>
            )}
          </div>
          </div>
        )}
      </div>

      {/* Generate Button */}
      <div className="generate-section" style={{ textAlign: 'center', marginBottom: '30px' }}>
        {errors.generation && (
          <div className="error-banner" style={{ marginBottom: '15px', padding: '10px', background: '#fee', border: '1px solid #fcc', borderRadius: '4px', color: '#c33' }}>
            ❌ {errors.generation}
          </div>
        )}

        <button
          className="generate-annotations-button"
          onClick={generateAnnotations}
          disabled={boxGenerationState.generating || boxGenerationState.loadingPreviousResults || !effectiveAnomalyResults}
          style={{
            padding: '15px 30px',
            background: boxGenerationState.generating || boxGenerationState.loadingPreviousResults ? '#6c757d' : '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: (boxGenerationState.generating || boxGenerationState.loadingPreviousResults) ? 'wait' : 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
            boxShadow: '0 4px 8px rgba(40,167,69,0.3)',
            opacity: (!effectiveAnomalyResults || boxGenerationState.generating || boxGenerationState.loadingPreviousResults) ? 0.6 : 1
          }}
        >
          {boxGenerationState.generating 
            ? `${boxGenerationState.annotationType === 'segmentation_masks' ? '🎭 Generating Segmentation Masks...' : '📦 Generating Bounding Boxes...'}`
            : boxGenerationState.loadingPreviousResults
            ? '🔍 Loading Previous Results...'
            : `🚀 Generate YOLO ${boxGenerationState.annotationType === 'segmentation_masks' ? 'Segmentation Masks' : 'Bounding Boxes'}`
          }
        </button>

        {!effectiveAnomalyResults && !boxGenerationState.loadingPreviousResults && (
          <p style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
            Complete Stage 3 anomaly detection first
          </p>
        )}

        {effectiveAnomalyResults && (
          <p style={{ fontSize: '12px', color: '#28a745', marginTop: '8px' }}>
            ✅ Ready to generate boxes from {effectiveAnomalyResults.total_images || 0} defective images
          </p>
        )}

        {boxGenerationState.generating && (
          <div className="generation-progress" style={{ marginTop: '15px' }}>
            <div className="progress-indicator">
              <div className="progress-bar" style={{ width: '100%', height: '4px', background: '#e9ecef', borderRadius: '2px', overflow: 'hidden' }}>
                <div className="progress-fill indeterminate" style={{ height: '100%', background: '#28a745', animation: 'indeterminate 1.5s infinite linear' }}></div>
              </div>
            </div>
            <p style={{ marginTop: '8px', fontSize: '14px', color: '#666' }}>
              🔄 Processing anomaly heatmaps and generating bounding boxes...
            </p>
          </div>
        )}
      </div>

      {/* Results Section */}
      {boxGenerationState.boxResults && (
        <div className="box-results">
          <div className="results-header">
            <h5>{boxGenerationState.annotationType === 'segmentation_masks' ? '🎭 Segmentation Mask Generation Results' : '📦 Bounding Box Generation Results'}</h5>
          </div>
          
          <div className="results-summary" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '20px' }}>
            <div className="summary-card" style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#28a745' }}>
                {boxGenerationState.boxResults.total_boxes_generated || boxGenerationState.boxResults.total_masks_generated}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                {boxGenerationState.annotationType === 'segmentation_masks' ? 'Total Masks Generated' : 'Total Boxes Generated'}
              </div>
            </div>
            
            <div className="summary-card" style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#007bff' }}>
                {boxGenerationState.boxResults.images_with_boxes || boxGenerationState.boxResults.images_with_masks}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                {boxGenerationState.annotationType === 'segmentation_masks' ? 'Images with Masks' : 'Images with Boxes'}
              </div>
            </div>
            
            <div className="summary-card" style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#6f42c1' }}>
                {boxGenerationState.boxResults.total_images_processed}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Images Processed</div>
            </div>
            
            <div className="summary-card" style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#dc3545' }}>
                {boxGenerationState.boxResults.failed_images_count}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Failed Images</div>
            </div>
          </div>

          {/* Settings Used */}
          <div className="settings-used" style={{ marginBottom: '20px' }}>
            <h6>⚙️ Settings Used</h6>
            <div style={{ background: '#f8f9fa', padding: '10px', borderRadius: '4px', fontSize: '12px' }}>
              Min Box Size: {boxGenerationState.boxResults.settings_used.min_box_size}px | 
              Threshold: {(boxGenerationState.boxResults.settings_used.anomaly_threshold * 100).toFixed(1)}% | 
              Merge Boxes: {boxGenerationState.boxResults.settings_used.merge_nearby_boxes ? 'Yes' : 'No'} | 
              Format: {boxGenerationState.boxResults.settings_used.output_format.toUpperCase()}
            </div>
          </div>

          {/* Visual Bounding Box Preview */}
          {boxGenerationState.visualPreview && boxGenerationState.visualPreview.preview_images && (
            <div className="visual-preview-section" style={{ marginBottom: '25px' }}>
              <h6>🖼️ Visual Bounding Box Preview (First {boxGenerationState.visualPreview.total_previews} Images)</h6>
              <p style={{ fontSize: '14px', color: '#666', marginBottom: '15px' }}>
                These images show the actual bounding boxes drawn on your defective images for visual verification.
              </p>
              
              <div className="visual-preview-grid" style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', 
                gap: '15px',
                marginBottom: '20px'
              }}>
                {boxGenerationState.visualPreview.preview_images.map((preview, index) => (
                  <div key={index} className="visual-preview-item" style={{ 
                    border: '2px solid #dee2e6', 
                    borderRadius: '8px', 
                    overflow: 'hidden',
                    backgroundColor: '#fff',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                  }}>
                    <img
                      src={`data:image/jpeg;base64,${preview.image_base64}`}
                      alt={`Visual boxes for ${preview.original_name}`}
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'contain',
                        background: '#f8f9fa'
                      }}
                    />
                    <div style={{ padding: '10px' }}>
                      <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '5px' }}>
                        {preview.original_name}
                      </div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Visual bounding boxes overlaid on original defective image
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div style={{ textAlign: 'center', padding: '10px', background: '#e7f3ff', borderRadius: '6px' }}>
                <span style={{ fontSize: '14px', color: '#0066cc' }}>
                  💡 These visual images are saved on the server and included in your download
                </span>
              </div>
            </div>
          )}

          {/* Sample Results */}
          {boxGenerationState.boxResults.generated_boxes && boxGenerationState.boxResults.generated_boxes.length > 0 && (
            <div className="sample-results" style={{ marginBottom: '20px' }}>
              <h6>🔍 Sample Results (First {Math.min(5, boxGenerationState.boxResults.generated_boxes.length)} Images)</h6>
              <div className="sample-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '10px' }}>
                {boxGenerationState.boxResults.generated_boxes.slice(0, 5).map((result, index) => (
                  <div key={index} className="sample-item" style={{ padding: '10px', background: '#fff', border: '1px solid #dee2e6', borderRadius: '4px' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                      {result.image_name.substring(0, 20)}...
                    </div>
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      Boxes: {result.boxes_count} | 
                      Size: {result.original_size[0]}×{result.original_size[1]} | 
                      Anomaly: {result.anomaly_percentage.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '10px', color: '#999', marginTop: '5px' }}>
                      File: {result.annotation_file}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="action-buttons" style={{ display: 'flex', gap: '15px', justifyContent: 'center', alignItems: 'center' }}>
            <button
              className="download-button"
              onClick={downloadAnnotations}
              style={{
                padding: '12px 24px',
                background: '#6f42c1',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 'bold'
              }}
            >
              📥 Download Annotations
            </button>

            <button
              className="complete-button"
              onClick={handleProceedToNextStep}
              style={{
                padding: '12px 24px',
                background: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 'bold'
              }}
            >
              ✅ Complete Workflow
            </button>
          </div>

          <div className="success-message" style={{ textAlign: 'center', marginTop: '15px', padding: '15px', background: '#d4edda', border: '1px solid #c3e6cb', borderRadius: '8px', color: '#155724' }}>
            <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '8px' }}>
              ✅ {boxGenerationState.annotationType === 'segmentation_masks' ? 'Segmentation Mask Generation Complete!' : 'Bounding Box Generation Complete!'}
            </div>
            <div style={{ fontSize: '14px' }}>
              {boxGenerationState.annotationType === 'segmentation_masks' ? (
                <>
                  🎭 YOLO segmentation annotations (.txt files) ready for training<br/>
                  🖼️ Visual images with colored masks saved for verification<br/>
                  📥 Download includes both polygon annotations and visual preview images
                </>
              ) : (
                <>
                  📦 YOLO annotations (.txt files) ready for training<br/>
                  🖼️ Visual images with boxes saved for verification<br/>
                  📥 Download includes both annotations and visual preview images
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Info Section */}
      <div className="step-info" style={{ marginTop: '30px' }}>
        <h5>ℹ️ About Annotation Generation</h5>
        <div className="info-points">
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>🎯</span>
            <span style={{ fontSize: '14px' }}>
              {boxGenerationState.annotationType === 'segmentation_masks' 
                ? 'Converts anomaly heatmaps into precise segmentation masks' 
                : 'Converts anomaly heatmaps into precise bounding boxes'}
            </span>
          </div>
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>📏</span>
            <span style={{ fontSize: '14px' }}>
              {boxGenerationState.annotationType === 'segmentation_masks' 
                ? 'Generates YOLO segmentation format with polygon coordinates' 
                : 'Generates YOLO format annotations ready for training'}
            </span>
          </div>
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>🔧</span>
            <span style={{ fontSize: '14px' }}>
              {boxGenerationState.annotationType === 'segmentation_masks' 
                ? 'Configurable mask area thresholds and polygon simplification' 
                : 'Adjustable settings for optimal box generation'}
            </span>
          </div>
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>🖼️</span>
            <span style={{ fontSize: '14px' }}>
              {boxGenerationState.annotationType === 'segmentation_masks' 
                ? 'Creates visual images with colored masks overlaid for verification' 
                : 'Creates visual images with boxes drawn for verification'}
            </span>
          </div>
          <div className="info-point" style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '16px' }}>💾</span>
            <span style={{ fontSize: '14px' }}>Downloads include annotations, visual images, and summary statistics</span>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes indeterminate {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(300%); }
        }
        
        .progress-fill.indeterminate {
          animation: indeterminate 1.5s infinite linear;
        }
      `}</style>
    </div>
  );
}

export default GenerateBoxesStep;