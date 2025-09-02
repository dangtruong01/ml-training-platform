import React, { useState, useEffect } from 'react';
import axios from 'axios';

function Predict() {
  const [images, setImages] = useState([]);
  const [task, setTask] = useState('detection');
  const [selectedModel, setSelectedModel] = useState('default');
  const [uploadedModel, setUploadedModel] = useState(null);
  const [results, setResults] = useState([]);
  const [availableModels, setAvailableModels] = useState({ pretrained: [], trained: [], uploaded: [] });
  const [message, setMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchSummary, setBatchSummary] = useState(null);

  // Fetch available models on component mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get('/list-models');
      setAvailableModels(response.data);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    setImages(files);
    setResults([]); // Clear previous results
    setBatchSummary(null);
  };

  const handleModelUpload = async () => {
    if (!uploadedModel) {
      setMessage('Please select a model file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', uploadedModel);
    formData.append('model_type', task);

    try {
      const response = await axios.post('/upload-model', formData);
      setMessage(`Model uploaded successfully: ${response.data.filename}`);
      fetchAvailableModels(); // Refresh model list
      setUploadedModel(null);
    } catch (error) {
      setMessage('Error uploading model.');
      console.error(error);
    }
  };

  const handlePredict = async () => {
    if (images.length === 0) {
      setMessage('Please select one or more images.');
      return;
    }

    setIsProcessing(true);
    setMessage('Processing images...');
    setResults([]);

    try {
      if (images.length === 1) {
        // Single image prediction
        await handleSinglePrediction();
      } else {
        // Batch prediction
        await handleBatchPrediction();
      }
    } catch (error) {
      setMessage('Error making prediction.');
      console.error(error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSinglePrediction = async () => {
    const formData = new FormData();
    formData.append('file', images[0]);
    
    if (selectedModel !== 'default') {
      formData.append('model_path', selectedModel);
    }

    const endpoint = task === 'detection' ? '/predict-detect' : '/predict-seg';
    
    try {
      const response = await axios.post(endpoint, formData);
      
      if (response.headers['content-type']?.includes('application/json')) {
        // JSON response with quality assessment
        const result = response.data;
        result.filename = images[0].name;
        setResults([result]);
        setMessage(`Prediction complete. Quality: ${result.quality.status}`);
      } else {
        // File response (legacy mode)
        const imageUrl = URL.createObjectURL(response.data);
        setResults([{ 
          filename: images[0].name, 
          image_url: imageUrl, 
          quality: { status: 'Unknown', reason: 'Legacy prediction mode' }
        }]);
        setMessage('Prediction complete (legacy mode).');
      }
    } catch (error) {
      throw error;
    }
  };

  const handleBatchPrediction = async () => {
    const formData = new FormData();
    
    images.forEach(image => {
      formData.append('files', image);
    });
    
    if (selectedModel !== 'default') {
      formData.append('model_path', selectedModel);
    }
    formData.append('task_type', task);

    try {
      const response = await axios.post('/predict-batch', formData);
      const data = response.data;
      
      setResults(data.results);
      setBatchSummary(data.summary);
      setMessage(`Batch prediction complete. ${data.summary.ok_count} OK, ${data.summary.ng_count} NG`);
    } catch (error) {
      throw error;
    }
  };

  const renderModelSelection = () => {
    const allModels = [
      ...availableModels.pretrained.map(m => ({ ...m, category: 'Pre-trained' })),
      ...availableModels.trained.map(m => ({ ...m, category: 'Trained' })),
      ...availableModels.uploaded.map(m => ({ ...m, category: 'Uploaded' }))
    ];

    return (
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Model Selection:</label>
        <select 
          value={selectedModel} 
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={isProcessing}
          style={{ padding: '8px', width: '300px' }}
        >
          <option value="default">🤖 Default Model</option>
          {allModels.map((model, index) => (
            <option key={index} value={model.path}>
              [{model.category}] {model.name}
            </option>
          ))}
        </select>
        <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
          Select a model or use default. Upload new models below.
        </div>
      </div>
    );
  };

  const renderQualityBadge = (quality) => {
    const isOK = quality.status === 'OK';
    return (
      <span style={{
        padding: '4px 8px',
        borderRadius: '12px',
        fontSize: '12px',
        fontWeight: 'bold',
        backgroundColor: isOK ? '#dcfce7' : '#fee2e2',
        color: isOK ? '#059669' : '#dc2626',
        border: `1px solid ${isOK ? '#86efac' : '#fca5a5'}`
      }}>
        {quality.status} ({Math.round(quality.confidence * 100)}%)
      </span>
    );
  };

  const renderResults = () => {
    if (results.length === 0) return null;

    return (
      <div style={{ marginTop: '20px' }}>
        <h3>🔍 Prediction Results</h3>
        
        {/* Batch Summary */}
        {batchSummary && (
          <div style={{
            backgroundColor: '#f0fdf4',
            border: '1px solid #059669',
            borderRadius: '8px',
            padding: '15px',
            marginBottom: '20px'
          }}>
            <h4>📊 Batch Summary</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
              <div><strong>Total Images:</strong> {batchSummary.total_images}</div>
              <div><strong>✅ OK:</strong> {batchSummary.ok_count} ({batchSummary.ok_percentage.toFixed(1)}%)</div>
              <div><strong>❌ NG:</strong> {batchSummary.ng_count} ({batchSummary.ng_percentage.toFixed(1)}%)</div>
            </div>
          </div>
        )}

        {/* Individual Results */}
        <div style={{ display: 'grid', gap: '20px' }}>
          {results.map((result, index) => (
            <div key={index} style={{
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '15px',
              backgroundColor: 'white'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <h4 style={{ margin: 0 }}>{result.filename}</h4>
                {result.quality && renderQualityBadge(result.quality)}
              </div>

              {/* Image Display */}
              {result.image_base64 && (
                <div style={{ marginBottom: '15px' }}>
                  <img 
                    src={`data:image/jpeg;base64,${result.image_base64}`}
                    alt="Prediction result"
                    style={{ 
                      maxWidth: '100%', 
                      maxHeight: '400px', 
                      borderRadius: '4px',
                      border: '1px solid #e2e8f0'
                    }}
                  />
                </div>
              )}

              {result.image_url && (
                <div style={{ marginBottom: '15px' }}>
                  <img 
                    src={result.image_url}
                    alt="Prediction result"
                    style={{ 
                      maxWidth: '100%', 
                      maxHeight: '400px', 
                      borderRadius: '4px',
                      border: '1px solid #e2e8f0'
                    }}
                  />
                </div>
              )}

              {/* Quality Details */}
              {result.quality && (
                <div style={{ marginBottom: '10px' }}>
                  <strong>Quality Assessment:</strong> {result.quality.reason}
                </div>
              )}

              {/* Detection Details */}
              {result.detections && result.detections.length > 0 && (
                <div>
                  <strong>Detections Found:</strong>
                  <div style={{ marginTop: '5px' }}>
                    {result.detections.map((detection, idx) => (
                      <div key={idx} style={{ 
                        padding: '5px', 
                        backgroundColor: '#f8fafc', 
                        margin: '2px 0',
                        borderRadius: '4px',
                        fontSize: '14px'
                      }}>
                        {detection.class}: {(detection.confidence * 100).toFixed(1)}%
                        {detection.area && ` (Area: ${detection.area}px)`}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.total_defects !== undefined && (
                <div style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
                  Total defects detected: {result.total_defects}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px' }}>
      <h2>🔮 Model Prediction & Quality Assessment</h2>
      
      {/* Main Prediction Interface */}
      <div style={{ 
        backgroundColor: '#f8fafc', 
        border: '1px solid #e2e8f0', 
        borderRadius: '8px',
        padding: '20px',
        marginBottom: '20px'
      }}>
        <h3>Upload & Predict</h3>
        
        {/* Image Upload */}
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Images: 
          </label>
          <input 
            type="file" 
            multiple
            accept="image/*"
            onChange={handleImageUpload}
            disabled={isProcessing}
            style={{ padding: '8px', width: '300px' }}
          />
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            Select one or multiple images for prediction. Supports batch processing.
          </div>
          {images.length > 0 && (
            <div style={{ marginTop: '5px', fontSize: '14px', color: '#059669' }}>
              ✅ {images.length} image(s) selected
            </div>
          )}
        </div>

        {/* Task Selection */}
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Task Type:</label>
          <select 
            value={task} 
            onChange={(e) => setTask(e.target.value)}
            disabled={isProcessing}
            style={{ padding: '8px', width: '200px' }}
          >
            <option value="detection">🎯 Object Detection</option>
            <option value="segmentation">✂️ Image Segmentation</option>
          </select>
        </div>

        {/* Model Selection */}
        {renderModelSelection()}

        {/* Predict Button */}
        <button 
          onClick={handlePredict}
          disabled={!images.length || isProcessing}
          style={{
            backgroundColor: !images.length || isProcessing ? '#ccc' : '#1e3a8a',
            color: 'white',
            padding: '12px 24px',
            border: 'none',
            borderRadius: '6px',
            cursor: !images.length || isProcessing ? 'not-allowed' : 'pointer',
            fontSize: '16px'
          }}
        >
          {isProcessing ? '🔄 Processing...' : '🚀 Run Prediction'}
        </button>

        {/* Status Message */}
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

      {/* Model Upload Section */}
      <div style={{ 
        backgroundColor: '#fffbeb', 
        border: '1px solid #f59e0b', 
        borderRadius: '8px',
        padding: '20px',
        marginBottom: '20px'
      }}>
        <h3>📤 Upload Custom Model</h3>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Model File (.pt):
          </label>
          <input 
            type="file" 
            accept=".pt"
            onChange={(e) => setUploadedModel(e.target.files[0])}
            style={{ padding: '8px', width: '300px' }}
          />
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            Upload your trained YOLO model (.pt file)
          </div>
        </div>
        <button 
          onClick={handleModelUpload}
          disabled={!uploadedModel}
          style={{
            backgroundColor: !uploadedModel ? '#ccc' : '#f59e0b',
            color: 'white',
            padding: '8px 16px',
            border: 'none',
            borderRadius: '4px',
            cursor: !uploadedModel ? 'not-allowed' : 'pointer'
          }}
        >
          📤 Upload Model
        </button>
      </div>

      {/* Results Display */}
      {renderResults()}
    </div>
  );
}

export default Predict;