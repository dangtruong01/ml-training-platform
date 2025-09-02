import React, { useState, useEffect } from 'react';

function Annotate() {
    const [files, setFiles] = useState([]);
    // const [previews, setPreviews] = useState([]);
    const [annotatedResults, setAnnotatedResults] = useState([]);
    const [annotationType, setAnnotationType] = useState('detection');
    const [isLoading, setIsLoading] = useState(false);
    const [processingProgress, setProcessingProgress] = useState({ current: 0, total: 0 });
    const [batchId, setBatchId] = useState(null);
    
    // GroundingDINO specific state
    const [groundingDinoPrompts, setGroundingDinoPrompts] = useState('scratch, dent, dirt');
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.3);
    const [groundingDinoAvailable, setGroundingDinoAvailable] = useState(false);

    // Guardrail (Anomaly Detection + Classification) specific state
    const [guardrailModelFile, setGuardrailModelFile] = useState(null);
    const [guardrailPrompts, setGuardrailPrompts] = useState('scratch, dent, dirt');
    const [guardrailConfidenceThreshold, setGuardrailConfidenceThreshold] = useState(0.3);

    // LLM+CLIP Anomaly Detection specific state
    const [llmClipAvailable, setLlmClipAvailable] = useState(false);
    const [componentType, setComponentType] = useState('metal plate');
    const [componentContext, setComponentContext] = useState('automotive');
    const [useLLM, setUseLLM] = useState(true);
    const [similarityThreshold, setSimilarityThreshold] = useState(0.7);

    // Check GroundingDINO and LLM+CLIP availability on component mount
    useEffect(() => {
        checkGroundingDinoStatus();
        checkLlmClipStatus();
    }, []);

    const checkGroundingDinoStatus = async () => {
        try {
            console.log('🔍 Checking GroundingDINO status...');
            const response = await fetch('/grounding-dino-status');
            console.log('📡 Response status:', response.status);
            const data = await response.json();
            console.log('📊 Response data:', data);
            setGroundingDinoAvailable(data.available);
            console.log('✅ GroundingDINO available set to:', data.available);
        } catch (error) {
            console.error('❌ Error checking GroundingDINO status:', error);
            setGroundingDinoAvailable(false);
        }
    };

    const checkLlmClipStatus = async () => {
        try {
            console.log('🧠 Checking LLM+CLIP status...');
            const response = await fetch('/llm-clip-status');
            const data = await response.json();
            console.log('📊 LLM+CLIP status:', data);
            setLlmClipAvailable(data.service_available);
            console.log('✅ LLM+CLIP available set to:', data.service_available);
        } catch (error) {
            console.error('❌ Error checking LLM+CLIP status:', error);
            setLlmClipAvailable(false);
        }
    };

    const handleFileChange = (e) => {
        const selectedFiles = Array.from(e.target.files);
        setFiles(selectedFiles);
        
        // Create previews for all selected files (functionality disabled)
        // const newPreviews = selectedFiles.map(file => ({
        //     file,
        //     url: URL.createObjectURL(file),
        //     name: file.name
        // }));
        // setPreviews(newPreviews);
        setAnnotatedResults([]); // Reset results when new files are selected
    };

    const handleAnnotate = async () => {
        if (files.length === 0) {
            alert("Please select at least one file first.");
            return;
        }

        setIsLoading(true);
        setProcessingProgress({ current: 0, total: files.length });
        setAnnotatedResults([]);

        // Generate unique batch ID
        const newBatchId = `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        setBatchId(newBatchId);

        if (annotationType === 'grounding-dino-only') {
            await handleGroundingDinoOnlyAnnotation(newBatchId);
        } else if (annotationType === 'grounding-dino-sam2') {
            await handleGroundingDinoSAM2Annotation(newBatchId);
        } else if (annotationType === 'guardrail') {
            await handleGuardrailAnnotation(newBatchId);
        } else if (annotationType === 'llm-clip-anomaly') {
            await handleLlmClipAnomalyAnnotation(newBatchId);
        } else {
            await handleTraditionalAnnotation(newBatchId);
        }
    };

    const handleGroundingDinoSAM2Annotation = async (newBatchId) => {
        const results = [];
        
        try {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                setProcessingProgress({ current: i + 1, total: files.length });

                const formData = new FormData();
                formData.append('file', file);
                formData.append('prompts', groundingDinoPrompts);
                formData.append('confidence_threshold', confidenceThreshold);
                formData.append('batch_id', newBatchId);

                try {
                    const response = await fetch('/grounding-dino-annotate', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const result = await response.json();
                        results.push({
                            originalName: file.name,
                            success: result.status === 'success',
                            annotatedUrl: result.status === 'success' ? `data:image/jpeg;base64,${result.image_base64}` : null,
                            error: result.status !== 'success' ? result.message : null,
                            detections: result.detections || [],
                            total_detections: result.total_detections || 0,
                            yolo_annotations: result.yolo_annotations
                        });
                    } else {
                        results.push({
                            originalName: file.name,
                            success: false,
                            error: "GroundingDINO annotation failed"
                        });
                    }
                } catch (error) {
                    console.error(`Error processing ${file.name}:`, error);
                    results.push({
                        originalName: file.name,
                        success: false,
                        error: "Processing error"
                    });
                }
            }

            setAnnotatedResults(results);
        } catch (error) {
            console.error('GroundingDINO annotation error:', error);
            alert('GroundingDINO annotation failed. Please try again.');
        } finally {
            setIsLoading(false);
            setProcessingProgress({ current: 0, total: 0 });
        }
    };

    const handleGroundingDinoOnlyAnnotation = async (newBatchId) => {
        const results = [];
        
        try {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                setProcessingProgress({ current: i + 1, total: files.length });

                const formData = new FormData();
                formData.append('file', file);
                formData.append('prompts', groundingDinoPrompts);
                formData.append('confidence_threshold', confidenceThreshold);
                formData.append('batch_id', newBatchId);

                try {
                    const response = await fetch('/grounding-dino-only-annotate', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const result = await response.json();
                        results.push({
                            originalName: file.name,
                            success: result.status === 'success',
                            annotatedUrl: result.status === 'success' ? `data:image/jpeg;base64,${result.image_base64}` : null,
                            error: result.status !== 'success' ? result.message : null,
                            detections: result.detections || [],
                            total_detections: result.total_detections || 0,
                            yolo_annotations: result.yolo_annotations
                        });
                    } else {
                        results.push({
                            originalName: file.name,
                            success: false,
                            error: "GroundingDINO annotation failed"
                        });
                    }
                } catch (error) {
                    console.error(`Error processing ${file.name}:`, error);
                    results.push({
                        originalName: file.name,
                        success: false,
                        error: "Processing error"
                    });
                }
            }

            setAnnotatedResults(results);
        } catch (error) {
            console.error('GroundingDINO annotation error:', error);
            alert('GroundingDINO annotation failed. Please try again.');
        } finally {
            setIsLoading(false);
            setProcessingProgress({ current: 0, total: 0 });
        }
    };

    const handleGuardrailAnnotation = async (newBatchId) => {
        if (!guardrailModelFile) {
            alert("Please upload your trained anomaly model file (.ckpt).");
            setIsLoading(false);
            return;
        }

        const results = [];

        try {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                setProcessingProgress({ current: i + 1, total: files.length });

                const formData = new FormData();
                formData.append("file", file);
                formData.append("model_file", guardrailModelFile);
                formData.append("prompts", guardrailPrompts);
                formData.append("confidence_threshold", guardrailConfidenceThreshold);

                try {
                    const response = await fetch('/annotate-with-guardrail', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const data = await response.json();
                        results.push({
                            originalName: file.name,
                            success: data.status === 'success',
                            annotatedUrl: data.status === 'success' && data.annotated_image ? 
                                `data:image/jpeg;base64,${btoa(String.fromCharCode(...new Uint8Array(data.annotated_image)))}` : null,
                            error: data.status !== 'success' ? data.message : null,
                            detections: data.detections || [],
                            total_detections: data.detections ? data.detections.length : 0
                        });
                    } else {
                        results.push({
                            originalName: file.name,
                            success: false,
                            error: "Guardrail annotation failed"
                        });
                    }
                } catch (error) {
                    console.error(`Error processing ${file.name}:`, error);
                    results.push({
                        originalName: file.name,
                        success: false,
                        error: "Processing error"
                    });
                }
            }

            setAnnotatedResults(results);
        } catch (error) {
            console.error('Guardrail annotation error:', error);
            alert('Guardrail annotation failed. Please try again.');
        } finally {
            setIsLoading(false);
            setProcessingProgress({ current: 0, total: 0 });
        }
    };

    const handleLlmClipAnomalyAnnotation = async (newBatchId) => {
        const results = [];

        try {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                setProcessingProgress({ current: i + 1, total: files.length });

                const formData = new FormData();
                formData.append('file', file);
                formData.append('component_type', componentType);
                formData.append('context', componentContext);
                formData.append('confidence_threshold', confidenceThreshold);
                formData.append('similarity_threshold', similarityThreshold);
                formData.append('use_llm', useLLM);
                formData.append('batch_id', newBatchId);

                try {
                    const response = await fetch('/llm-clip-anomaly-detect', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const result = await response.json();
                        results.push({
                            originalName: file.name,
                            success: result.status === 'success',
                            annotatedUrl: result.status === 'success' ? `data:image/jpeg;base64,${result.image_base64}` : null,
                            error: result.status !== 'success' ? result.message : null,
                            anomaly_regions: result.anomaly_regions_detected || 0,
                            total_regions: result.total_regions_detected || 0,
                            anomaly_rate: result.anomaly_rate || 0,
                            detections: result.detections || [],
                            component_type: result.component_type,
                            descriptions_used: result.descriptions_used
                        });
                    } else {
                        results.push({
                            originalName: file.name,
                            success: false,
                            error: "LLM+CLIP anomaly detection failed"
                        });
                    }
                } catch (error) {
                    console.error(`Error processing ${file.name}:`, error);
                    results.push({
                        originalName: file.name,
                        success: false,
                        error: "Processing error"
                    });
                }
            }

            setAnnotatedResults(results);
        } catch (error) {
            console.error('LLM+CLIP annotation error:', error);
            alert('LLM+CLIP annotation failed. Please try again.');
        } finally {
            setIsLoading(false);
            setProcessingProgress({ current: 0, total: 0 });
        }
    };

    const handleTraditionalAnnotation = async (newBatchId) => {
        const endpointMap = {
            'detection': '/pre-annotate-detect',
            'segmentation': '/pre-annotate-segment',
            'sam2-detection': '/pre-annotate-sam2-detect',
            'sam2-segmentation': '/pre-annotate-sam2-segment'
        };
        const endpoint = endpointMap[annotationType] || '/pre-annotate-detect';

        const results = [];

        try {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                setProcessingProgress({ current: i + 1, total: files.length });

                const formData = new FormData();
                formData.append("file", file);
                formData.append("batch_id", newBatchId); // Add batch ID for caching

                try {
                    const response = await fetch(endpoint, {
                        method: "POST",
                        body: formData,
                    });

                    if (response.ok) {
                        const imageBlob = await response.blob();
                        results.push({
                            originalName: file.name,
                            annotatedUrl: URL.createObjectURL(imageBlob),
                            success: true
                        });
                    } else {
                        results.push({
                            originalName: file.name,
                            success: false,
                            error: `${annotationType} annotation failed`
                        });
                    }
                } catch (error) {
                    console.error(`Error processing ${file.name}:`, error);
                    results.push({
                        originalName: file.name,
                        success: false,
                        error: "Processing error"
                    });
                }
            }

            setAnnotatedResults(results);
        } catch (error) {
            console.error("Error during batch annotation:", error);
            alert("An error occurred during batch annotation.");
        } finally {
            setIsLoading(false);
            setProcessingProgress({ current: 0, total: 0 });
        }
    };

    const handleDownloadZip = async () => {
        const successfulResults = annotatedResults.filter(result => result.success);
        
        if (successfulResults.length === 0) {
            alert("No successful annotations to download.");
            return;
        }

        try {
            const response = await fetch(`/download-batch-zip/${batchId}`, {
                method: 'POST',
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `annotated_${annotationType}_batch.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                throw new Error('Failed to download zip file');
            }
        } catch (error) {
            console.error('Error downloading zip:', error);
            alert('Failed to download zip file. Please try again.');
        }
    };

    const handleExportYoloDataset = async () => {
        const groundingDinoResults = annotatedResults.filter(result => 
            result.success && result.yolo_annotations
        );

        if (groundingDinoResults.length === 0) {
            alert("No GroundingDINO results to export.");
            return;
        }

        try {
            const formData = new FormData();
            formData.append('annotation_results', JSON.stringify(groundingDinoResults));
            formData.append('dataset_name', `grounding_dino_${Date.now()}`);

            const response = await fetch('/grounding-dino-export-yolo', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                alert(`YOLO dataset exported successfully: ${data.dataset_name}`);
            } else {
                throw new Error('Failed to export dataset');
            }
        } catch (error) {
            console.error('Error exporting dataset:', error);
            alert('Failed to export YOLO dataset.');
        }
    };

    const renderAnnotationTypeSelector = () => (
        <div style={{ marginBottom: '20px' }}>
            <h3>🎯 Annotation Method</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                <label style={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    padding: '10px',
                    border: '2px solid #e2e8f0',
                    borderRadius: '8px',
                    backgroundColor: annotationType === 'detection' ? '#dbeafe' : 'white',
                    borderColor: annotationType === 'detection' ? '#3b82f6' : '#e2e8f0',
                    cursor: 'pointer'
                }}>
                    <input
                        type="radio"
                        value="detection"
                        checked={annotationType === 'detection'}
                        onChange={(e) => setAnnotationType(e.target.value)}
                        style={{ marginRight: '8px' }}
                    />
                    <span>📦 OpenCV Detection</span>
                </label>

                <label style={{
                    display: 'flex', 
                    alignItems: 'center',
                    padding: '10px',
                    border: '2px solid #e2e8f0',
                    borderRadius: '8px',
                    backgroundColor: annotationType === 'segmentation' ? '#dbeafe' : 'white',
                    borderColor: annotationType === 'segmentation' ? '#3b82f6' : '#e2e8f0',
                    cursor: 'pointer'
                }}>
                    <input
                        type="radio"
                        value="segmentation"
                        checked={annotationType === 'segmentation'}
                        onChange={(e) => setAnnotationType(e.target.value)}
                        style={{ marginRight: '8px' }}
                    />
                    <span>✂️ OpenCV Segmentation</span>
                </label>

                <label style={{
                    display: 'flex', 
                    alignItems: 'center',
                    padding: '10px',
                    border: '2px solid #e2e8f0',
                    borderRadius: '8px',
                    backgroundColor: annotationType === 'sam2-detection' ? '#dbeafe' : 'white',
                    borderColor: annotationType === 'sam2-detection' ? '#3b82f6' : '#e2e8f0',
                    cursor: 'pointer'
                }}>
                    <input
                        type="radio"
                        value="sam2-detection"
                        checked={annotationType === 'sam2-detection'}
                        onChange={(e) => setAnnotationType(e.target.value)}
                        style={{ marginRight: '8px' }}
                    />
                    <span>🎯 SAM2 Detection</span>
                </label>

                <label style={{
                    display: 'flex', 
                    alignItems: 'center',
                    padding: '10px',
                    border: '2px solid #e2e8f0',
                    borderRadius: '8px',
                    backgroundColor: annotationType === 'sam2-segmentation' ? '#dbeafe' : 'white',
                    borderColor: annotationType === 'sam2-segmentation' ? '#3b82f6' : '#e2e8f0',
                    cursor: 'pointer'
                }}>
                    <input
                        type="radio"
                        value="sam2-segmentation"
                        checked={annotationType === 'sam2-segmentation'}
                        onChange={(e) => setAnnotationType(e.target.value)}
                        style={{ marginRight: '8px' }}
                    />
                    <span>🔪 SAM2 Segmentation</span>
                </label>

                {groundingDinoAvailable && (
                    <label style={{
                        display: 'flex', 
                        alignItems: 'center',
                        padding: '10px',
                        border: '2px solid #10b981',
                        borderRadius: '8px',
                        backgroundColor: annotationType === 'grounding-dino-only' ? '#d1fae5' : 'white',
                        borderColor: annotationType === 'grounding-dino-only' ? '#10b981' : '#10b981',
                        cursor: 'pointer'
                    }}>
                        <input
                            type="radio"
                            value="grounding-dino-only"
                            checked={annotationType === 'grounding-dino-only'}
                            onChange={(e) => setAnnotationType(e.target.value)}
                            style={{ marginRight: '8px' }}
                        />
                        <span>✨ GroundingDINO Only (Detection)</span>
                    </label>
                )}

                {groundingDinoAvailable && (
                    <label style={{
                        display: 'flex', 
                        alignItems: 'center',
                        padding: '10px',
                        border: '2px solid #7c2d12',
                        borderRadius: '8px',
                        backgroundColor: annotationType === 'grounding-dino-sam2' ? '#fed7aa' : 'white',
                        borderColor: annotationType === 'grounding-dino-sam2' ? '#7c2d12' : '#7c2d12',
                        cursor: 'pointer'
                    }}>
                        <input
                            type="radio"
                            value="grounding-dino-sam2"
                            checked={annotationType === 'grounding-dino-sam2'}
                            onChange={(e) => setAnnotationType(e.target.value)}
                            style={{ marginRight: '8px' }}
                        />
                        <span>🎭 GroundingDINO + SAM2 (Detection + Segmentation)</span>
                    </label>
                )}

                {groundingDinoAvailable && (
                    <label style={{
                        display: 'flex', 
                        alignItems: 'center',
                        padding: '10px',
                        border: '2px solid #7c3aed',
                        borderRadius: '8px',
                        backgroundColor: annotationType === 'guardrail' ? '#ede9fe' : 'white',
                        borderColor: annotationType === 'guardrail' ? '#7c3aed' : '#7c3aed',
                        cursor: 'pointer'
                    }}>
                        <input
                            type="radio"
                            value="guardrail"
                            checked={annotationType === 'guardrail'}
                            onChange={(e) => setAnnotationType(e.target.value)}
                            style={{ marginRight: '8px' }}
                        />
                        <span>🛡️ Guardrail (Anomaly + AI)</span>
                    </label>
                )}

                {llmClipAvailable && (
                    <label style={{
                        display: 'flex', 
                        alignItems: 'center',
                        padding: '10px',
                        border: '2px solid #8b5cf6',
                        borderRadius: '8px',
                        backgroundColor: annotationType === 'llm-clip-anomaly' ? '#f3e8ff' : 'white',
                        borderColor: annotationType === 'llm-clip-anomaly' ? '#8b5cf6' : '#8b5cf6',
                        cursor: 'pointer'
                    }}>
                        <input
                            type="radio"
                            value="llm-clip-anomaly"
                            checked={annotationType === 'llm-clip-anomaly'}
                            onChange={(e) => setAnnotationType(e.target.value)}
                            style={{ marginRight: '8px' }}
                        />
                        <span>🧠 LLM+CLIP Anomaly Detection</span>
                    </label>
                )}
            </div>

            {(!groundingDinoAvailable || !llmClipAvailable) && (
                <div style={{ 
                    marginTop: '10px', 
                    padding: '10px', 
                    backgroundColor: '#fef3c7', 
                    border: '1px solid #f59e0b',
                    borderRadius: '4px',
                    fontSize: '14px'
                }}>
                    {!groundingDinoAvailable && (
                        <>⚠️ GroundingDINO not available. Please install dependencies to use AI-powered annotation.<br /></>
                    )}
                    {!llmClipAvailable && (
                        <>🧠 LLM+CLIP not available. Install transformers and torch, and set OpenAI API key.<br /></>
                    )}
                    <div style={{ marginTop: '8px', display: 'flex', gap: '5px' }}>
                        <button 
                            onClick={checkGroundingDinoStatus}
                            style={{
                                backgroundColor: '#3b82f6',
                                color: 'white',
                                border: 'none',
                                padding: '5px 10px',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '12px'
                            }}
                        >
                            🔄 Check GroundingDINO
                        </button>
                        <button 
                            onClick={checkLlmClipStatus}
                            style={{
                                backgroundColor: '#8b5cf6',
                                color: 'white',
                                border: 'none',
                                padding: '5px 10px',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '12px'
                            }}
                        >
                            🧠 Check LLM+CLIP
                        </button>
                    </div>
                </div>
            )}
        </div>
    );

    const renderGroundingDinoOptions = () => {
        if (annotationType !== 'grounding-dino-only' && annotationType !== 'grounding-dino-sam2') return null;

        return (
            <div style={{
                marginBottom: '20px',
                padding: '20px',
                backgroundColor: '#f0fdf4',
                border: '1px solid #10b981',
                borderRadius: '8px'
            }}>
                <h4 style={{ margin: '0 0 15px 0', color: '#059669' }}>
                    {annotationType === 'grounding-dino-only' ? '✨ GroundingDINO Only Settings' : '🎭 GroundingDINO + SAM2 Settings'}
                </h4>
                
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Defect Prompts (comma-separated):
                    </label>
                    <input
                        type="text"
                        value={groundingDinoPrompts}
                        onChange={(e) => setGroundingDinoPrompts(e.target.value)}
                        placeholder="e.g., scratch, dent, dirt, corrosion, crack"
                        style={{ 
                            width: '100%', 
                            padding: '8px', 
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '14px'
                        }}
                    />
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '5px' }}>
                        Describe the defects you want to detect. Be specific (e.g., "scratch on metal surface").
                    </div>
                </div>

                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Confidence Threshold: {confidenceThreshold}
                    </label>
                    <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.1"
                        value={confidenceThreshold}
                        onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                        style={{ width: '200px' }}
                    />
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '5px' }}>
                        Lower values detect more objects but may include false positives.
                    </div>
                </div>
            </div>
        );
    };

    const renderGuardrailOptions = () => {
        if (annotationType !== 'guardrail') return null;

        return (
            <div style={{
                marginBottom: '20px',
                padding: '20px',
                backgroundColor: '#faf5ff',
                border: '1px solid #7c3aed',
                borderRadius: '8px'
            }}>
                <h4 style={{ margin: '0 0 15px 0', color: '#7c3aed' }}>🛡️ Guardrail Settings (Anomaly Detection + AI Classification)</h4>
                
                {/* Model File Upload */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Upload Anomaly Model File:
                    </label>
                    <input
                        type="file"
                        accept=".ckpt,.pth,.pt"
                        onChange={(e) => setGuardrailModelFile(e.target.files[0])}
                        style={{ 
                            width: '100%', 
                            padding: '8px', 
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '14px'
                        }}
                    />
                    {guardrailModelFile && (
                        <div style={{ fontSize: '12px', color: '#059669', marginTop: '5px' }}>
                            ✅ Selected: {guardrailModelFile.name}
                        </div>
                    )}
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '5px' }}>
                        Upload your trained PatchCore anomaly detection model (.ckpt file). 
                        <br />💡 Train one in the <strong>Train</strong> page first!
                    </div>
                </div>

                {/* Training Link */}
                <div style={{ 
                    marginBottom: '15px',
                    padding: '10px',
                    backgroundColor: '#e0e7ff',
                    border: '1px solid #6366f1',
                    borderRadius: '6px'
                }}>
                    <div style={{ fontSize: '14px', color: '#4f46e5' }}>
                        <strong>📚 Need to train an anomaly model?</strong>
                        <br />Go to the <strong>Train</strong> page to create a new anomaly detection model using defect-free images.
                    </div>
                </div>

                {/* Classification Prompts */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Classification Prompts (comma-separated):
                    </label>
                    <input
                        type="text"
                        value={guardrailPrompts}
                        onChange={(e) => setGuardrailPrompts(e.target.value)}
                        placeholder="e.g., scratch, dent, dirt, corrosion, crack"
                        style={{ 
                            width: '100%', 
                            padding: '8px', 
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '14px'
                        }}
                    />
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '5px' }}>
                        After detecting anomalies, the system will use GroundingDINO to classify them with these prompts.
                    </div>
                </div>

                {/* Confidence Threshold */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Classification Confidence Threshold: {guardrailConfidenceThreshold}
                    </label>
                    <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.1"
                        value={guardrailConfidenceThreshold}
                        onChange={(e) => setGuardrailConfidenceThreshold(parseFloat(e.target.value))}
                        style={{ width: '200px' }}
                    />
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '5px' }}>
                        Confidence threshold for GroundingDINO classification of detected anomalies.
                    </div>
                </div>

                {/* How it works */}
                <div style={{
                    padding: '10px',
                    backgroundColor: '#e0e7ff',
                    border: '1px solid #3b82f6',
                    borderRadius: '4px',
                    fontSize: '12px'
                }}>
                    <strong>How Guardrail Works:</strong>
                    <ol style={{ margin: '5px 0', paddingLeft: '20px' }}>
                        <li><strong>Anomaly Detection:</strong> PatchCore detects deviations from "normal" (high sensitivity)</li>
                        <li><strong>Classification:</strong> GroundingDINO classifies detected anomalies using your prompts</li>
                        <li><strong>Result:</strong> Only real defects are labeled, with minimal false positives</li>
                    </ol>
                </div>
            </div>
        );
    };

    const renderLlmClipOptions = () => {
        if (annotationType !== 'llm-clip-anomaly') return null;

        return (
            <div style={{
                marginBottom: '20px',
                padding: '20px',
                backgroundColor: '#f8fafc',
                border: '1px solid #8b5cf6',
                borderRadius: '8px'
            }}>
                <h4 style={{ margin: '0 0 15px 0', color: '#8b5cf6' }}>🧠 LLM+CLIP Anomaly Detection Settings</h4>
                
                {/* Component Type */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Component Type:
                    </label>
                    <input
                        type="text"
                        value={componentType}
                        onChange={(e) => setComponentType(e.target.value)}
                        placeholder="e.g., metal plate, circuit board, automotive part"
                        style={{ 
                            width: '100%', 
                            padding: '8px', 
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '14px'
                        }}
                    />
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '5px' }}>
                        What type of component/object are you analyzing for defects?
                    </div>
                </div>

                {/* Context */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Context/Industry:
                    </label>
                    <input
                        type="text"
                        value={componentContext}
                        onChange={(e) => setComponentContext(e.target.value)}
                        placeholder="e.g., automotive, electronics, manufacturing"
                        style={{ 
                            width: '100%', 
                            padding: '8px', 
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            fontSize: '14px'
                        }}
                    />
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '5px' }}>
                        Additional context to help generate better anomaly descriptions.
                    </div>
                </div>

                {/* Use LLM Toggle */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'flex', alignItems: 'center' }}>
                        <input
                            type="checkbox"
                            checked={useLLM}
                            onChange={(e) => setUseLLM(e.target.checked)}
                            style={{ marginRight: '8px' }}
                        />
                        <span style={{ fontWeight: 'bold' }}>Use LLM for Description Generation</span>
                    </label>
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '5px' }}>
                        {useLLM 
                            ? "🤖 LLM will generate detailed anomaly descriptions (requires OpenAI API key)"
                            : "📋 Use predefined template descriptions (faster, no API key needed)"
                        }
                    </div>
                </div>

                {/* Thresholds */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                    <div>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                            Detection Confidence: {confidenceThreshold}
                        </label>
                        <input
                            type="range"
                            min="0.1"
                            max="0.9"
                            step="0.1"
                            value={confidenceThreshold}
                            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                            style={{ width: '100%' }}
                        />
                        <div style={{ fontSize: '12px', color: '#6b7280' }}>
                            GroundingDINO component detection threshold
                        </div>
                    </div>
                    
                    <div>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                            Similarity Threshold: {similarityThreshold}
                        </label>
                        <input
                            type="range"
                            min="0.3"
                            max="0.9"
                            step="0.1"
                            value={similarityThreshold}
                            onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
                            style={{ width: '100%' }}
                        />
                        <div style={{ fontSize: '12px', color: '#6b7280' }}>
                            CLIP similarity threshold for anomaly detection
                        </div>
                    </div>
                </div>

                {/* How it works */}
                <div style={{
                    padding: '10px',
                    backgroundColor: '#e0e7ff',
                    border: '1px solid #8b5cf6',
                    borderRadius: '4px',
                    fontSize: '12px'
                }}>
                    <strong>How LLM+CLIP Works:</strong>
                    <ol style={{ margin: '5px 0', paddingLeft: '20px' }}>
                        <li><strong>LLM Generation:</strong> Creates detailed normal/anomaly descriptions for your component</li>
                        <li><strong>Component Localization:</strong> GroundingDINO finds component instances in the image</li>
                        <li><strong>CLIP Analysis:</strong> Measures semantic similarity between image regions and descriptions</li>
                        <li><strong>Anomaly Detection:</strong> Regions with low similarity to normal descriptions are flagged as anomalies</li>
                    </ol>
                </div>

                {/* API Key Warning */}
                {useLLM && (
                    <div style={{
                        marginTop: '10px',
                        padding: '8px',
                        backgroundColor: '#fef3c7',
                        border: '1px solid #f59e0b',
                        borderRadius: '4px',
                        fontSize: '12px'
                    }}>
                        💡 <strong>Note:</strong> LLM mode requires an OpenAI API key to be set in the backend environment.
                    </div>
                )}
            </div>
        );
    };

    const renderResults = () => {
        if (annotatedResults.length === 0) return null;

        const successfulResults = annotatedResults.filter(result => result.success);
        const failedResults = annotatedResults.filter(result => !result.success);

        return (
            <div style={{ marginTop: '30px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <h3>📊 Annotation Results</h3>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        {successfulResults.length > 0 && (
                            <button 
                                onClick={handleDownloadZip}
                                style={{
                                    backgroundColor: '#059669',
                                    color: 'white',
                                    padding: '8px 16px',
                                    border: 'none',
                                    borderRadius: '4px',
                                    cursor: 'pointer'
                                }}
                            >
                                📥 Download ZIP
                            </button>
                        )}
                        {(annotationType === 'grounding-dino-only' || annotationType === 'grounding-dino-sam2') && successfulResults.some(r => r.yolo_annotations) && (
                            <button 
                                onClick={handleExportYoloDataset}
                                style={{
                                    backgroundColor: '#7c3aed',
                                    color: 'white',
                                    padding: '8px 16px',
                                    border: 'none',
                                    borderRadius: '4px',
                                    cursor: 'pointer'
                                }}
                            >
                                📦 Export YOLO Dataset
                            </button>
                        )}
                    </div>
                </div>

                <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
                    gap: '20px',
                    marginBottom: '20px'
                }}>
                    <div style={{
                        padding: '15px',
                        backgroundColor: '#dcfce7',
                        border: '1px solid #86efac',
                        borderRadius: '8px'
                    }}>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#059669' }}>
                            ✅ Successful: {successfulResults.length}
                        </div>
                    </div>
                    <div style={{
                        padding: '15px',
                        backgroundColor: '#fee2e2',
                        border: '1px solid #fca5a5',
                        borderRadius: '8px'
                    }}>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#dc2626' }}>
                            ❌ Failed: {failedResults.length}
                        </div>
                    </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
                    {annotatedResults.map((result, index) => (
                        <div key={index} style={{ 
                            border: '1px solid #e2e8f0', 
                            borderRadius: '8px', 
                            padding: '15px',
                            backgroundColor: result.success ? '#f9fafb' : '#fef2f2'
                        }}>
                            <h4 style={{ margin: '0 0 10px 0' }}>{result.originalName}</h4>
                            
                            {result.success ? (
                                <div>
                                    <img 
                                        src={result.annotatedUrl} 
                                        alt={`Annotated ${result.originalName}`} 
                                        style={{ width: '100%', borderRadius: '4px', marginBottom: '10px' }}
                                    />
                                    
                                    {/* LLM+CLIP Results */}
                                    {result.anomaly_regions !== undefined && (
                                        <div style={{ fontSize: '14px', color: '#6b7280', marginBottom: '10px' }}>
                                            <strong>Anomaly Analysis:</strong>
                                            <div style={{ marginLeft: '10px', fontSize: '12px' }}>
                                                • Component: {result.component_type}
                                            </div>
                                            <div style={{ marginLeft: '10px', fontSize: '12px' }}>
                                                • Regions analyzed: {result.total_regions}
                                            </div>
                                            <div style={{ marginLeft: '10px', fontSize: '12px' }}>
                                                • Anomalies found: {result.anomaly_regions}
                                            </div>
                                            <div style={{ marginLeft: '10px', fontSize: '12px' }}>
                                                • Anomaly rate: {(result.anomaly_rate * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    )}
                                    
                                    {/* Traditional Detection Results */}
                                    {result.detections && result.anomaly_regions === undefined && (
                                        <div style={{ fontSize: '14px', color: '#6b7280' }}>
                                            <strong>Detections:</strong> {result.total_detections || result.detections.length}
                                            {result.detections.map((detection, idx) => (
                                                <div key={idx} style={{ marginLeft: '10px', fontSize: '12px' }}>
                                                    • {detection.class}: {(detection.confidence * 100).toFixed(1)}%
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                    
                                    {/* LLM+CLIP Detection Details */}
                                    {result.detections && result.anomaly_regions !== undefined && result.detections.length > 0 && (
                                        <div style={{ fontSize: '14px', color: '#6b7280' }}>
                                            <strong>Anomaly Details:</strong>
                                            {result.detections.map((detection, idx) => (
                                                <div key={idx} style={{ marginLeft: '10px', fontSize: '12px' }}>
                                                    • Region {detection.region_index + 1}: {detection.is_anomaly ? '🔴 ANOMALY' : '🟢 NORMAL'} 
                                                    {detection.is_anomaly && ` (${detection.anomaly_type})`}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div style={{ color: '#dc2626' }}>
                                    Error: {result.error}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
            <h2>🏷️ Image Annotation</h2>
            
            <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '10px', fontSize: '16px', fontWeight: 'bold' }}>
                    📁 Select Images:
                </label>
                <input 
                    type="file" 
                    multiple 
                    accept="image/*"
                    onChange={handleFileChange}
                    style={{ 
                        padding: '10px',
                        border: '2px dashed #d1d5db',
                        borderRadius: '8px',
                        width: '100%',
                        cursor: 'pointer'
                    }}
                />
                {files.length > 0 && (
                    <div style={{ marginTop: '10px', fontSize: '14px', color: '#059669' }}>
                        ✅ {files.length} image(s) selected
                    </div>
                )}
            </div>

            {renderAnnotationTypeSelector()}
            {renderGroundingDinoOptions()}
            {renderGuardrailOptions()}
            {renderLlmClipOptions()}

            <div style={{ marginBottom: '30px' }}>
                <button 
                    onClick={handleAnnotate}
                    disabled={files.length === 0 || isLoading}
                    style={{
                        backgroundColor: files.length === 0 || isLoading ? '#9ca3af' : '#1e3a8a',
                        color: 'white',
                        padding: '12px 24px',
                        border: 'none',
                        borderRadius: '8px',
                        fontSize: '16px',
                        cursor: files.length === 0 || isLoading ? 'not-allowed' : 'pointer'
                    }}
                >
                    {isLoading ? `🔄 Processing... (${processingProgress.current}/${processingProgress.total})` : '🚀 Start Annotation'}
                </button>
            </div>

            {renderResults()}
        </div>
    );
}

export default Annotate;