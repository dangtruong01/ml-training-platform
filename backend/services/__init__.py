# Lazy loading for services - only import when needed to avoid startup model loading
# Essential services (no heavy models) - import immediately
from .core.database_service import database_service

# Lightweight services - import immediately  
from .ml.yolo_service import yolo_service

# Global service instances for lazy loading
_services = {}

def get_service(service_name: str):
    """Get service instance with lazy loading"""
    if service_name in _services:
        return _services[service_name]
    
    if service_name == 'guardrail_service':
        from .core.guardrail_service import guardrail_service
        _services[service_name] = guardrail_service
        return guardrail_service
    
    elif service_name == 'auto_annotation_service':
        from .annotation.auto_annotation_service import auto_annotation_service
        _services[service_name] = auto_annotation_service
        return auto_annotation_service
    
    elif service_name == 'grounding_dino_service':
        from .annotation.grounding_dino_service import grounding_dino_service
        _services[service_name] = grounding_dino_service
        return grounding_dino_service
    
    elif service_name == 'grounding_dino_sam2_service':
        from .annotation.grounding_dino_sam2_service import grounding_dino_sam2_service
        _services[service_name] = grounding_dino_sam2_service
        return grounding_dino_sam2_service
    
    elif service_name == 'sam2_service':
        from .annotation.sam2_service import sam2_service
        _services[service_name] = sam2_service
        return sam2_service
    
    elif service_name == 'roi_extraction_service':
        from .annotation.roi_extraction_service import roi_extraction_service
        _services[service_name] = roi_extraction_service
        return roi_extraction_service
    
    elif service_name == 'anomaly_service':
        from .ml.anomaly_service import anomaly_service
        _services[service_name] = anomaly_service
        return anomaly_service
    
    elif service_name == 'advanced_anomaly_service':
        from .ml.advanced_anomaly_service import advanced_anomaly_service
        _services[service_name] = advanced_anomaly_service
        return advanced_anomaly_service
    
    elif service_name == 'llm_clip_anomaly_service':
        from .ml.llm_clip_anomaly_service import llm_clip_anomaly_service
        _services[service_name] = llm_clip_anomaly_service
        return llm_clip_anomaly_service
    
    elif service_name == 'defect_detection_service':
        from .ml.defect_detection_service import defect_detection_service
        _services[service_name] = defect_detection_service
        return defect_detection_service
    
    elif service_name == 'dinov2_service':
        from .ml.dinov2_service import dinov2_service
        _services[service_name] = dinov2_service
        return dinov2_service
    
    elif service_name == 'dinov3_service':
        from .ml.dinov3_service import dinov3_service
        _services[service_name] = dinov3_service
        return dinov3_service
    
    else:
        raise ValueError(f"Unknown service: {service_name}")

# Maintain backward compatibility by providing lazy-loaded service references
class LazyService:
    def __init__(self, service_name):
        self._service_name = service_name
        self._service = None
    
    def __getattr__(self, name):
        if self._service is None:
            self._service = get_service(self._service_name)
        return getattr(self._service, name)

# Create lazy service proxies for backward compatibility
guardrail_service = LazyService('guardrail_service')
auto_annotation_service = LazyService('auto_annotation_service')
grounding_dino_service = LazyService('grounding_dino_service')
grounding_dino_sam2_service = LazyService('grounding_dino_sam2_service')
sam2_service = LazyService('sam2_service')
roi_extraction_service = LazyService('roi_extraction_service')
anomaly_service = LazyService('anomaly_service')
advanced_anomaly_service = LazyService('advanced_anomaly_service')
llm_clip_anomaly_service = LazyService('llm_clip_anomaly_service')
defect_detection_service = LazyService('defect_detection_service')
dinov2_service = LazyService('dinov2_service')
dinov3_service = LazyService('dinov3_service')