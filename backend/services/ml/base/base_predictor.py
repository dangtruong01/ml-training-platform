import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BasePredictor(ABC):
    """Abstract base class for all ML predictors"""
    
    def __init__(self, models_dir: str = "ml/models"):
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
    
    @abstractmethod
    def predict(self, image_path: str, model_path: str = None, **kwargs) -> Dict[str, Any]:
        """Make prediction on a single image
        
        Args:
            image_path: Path to the input image
            model_path: Path to the model file (optional, use default if None)
            **kwargs: Additional parameters for prediction
            
        Returns:
            Dictionary containing prediction results
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm this predictor handles"""
        pass
    
    def predict_batch(self, image_paths: List[str], model_path: str = None, **kwargs) -> List[Dict[str, Any]]:
        """Make predictions on multiple images
        
        Args:
            image_paths: List of paths to input images
            model_path: Path to the model file (optional)
            **kwargs: Additional parameters for prediction
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path, model_path, **kwargs)
                result['image_path'] = image_path
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def validate_image(self, image_path: str) -> bool:
        """Validate if the image file exists and is readable
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image is valid, False otherwise
        """
        if not os.path.exists(image_path):
            return False
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        file_ext = os.path.splitext(image_path)[1].lower()
        
        return file_ext in valid_extensions
    
    def validate_model(self, model_path: str) -> bool:
        """Validate if the model file exists and is readable
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model is valid, False otherwise
        """
        return os.path.exists(model_path) and os.path.isfile(model_path)
    
    def get_default_model_path(self) -> Optional[str]:
        """Get the default model path for this predictor
        
        Returns:
            Path to default model, or None if no default available
        """
        # Can be overridden by subclasses
        return None
    
    def preprocess_image(self, image_path: str) -> Any:
        """Preprocess image for prediction
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image data
        """
        # Default implementation - can be overridden by subclasses
        import cv2
        return cv2.imread(image_path)
    
    def postprocess_results(self, raw_results: Any, image_path: str) -> Dict[str, Any]:
        """Postprocess raw prediction results
        
        Args:
            raw_results: Raw results from the model
            image_path: Path to the input image
            
        Returns:
            Processed results dictionary
        """
        # Default implementation - can be overridden by subclasses
        return {
            'raw_results': raw_results,
            'processed': True
        }