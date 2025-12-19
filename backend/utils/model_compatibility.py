"""
Model compatibility utilities for handling sklearn version mismatches
"""
import pickle
import warnings
import sklearn
from typing import Any, Dict, Union

def load_sklearn_model_safely(model_path: str, suppress_warnings: bool = False) -> Dict[str, Any]:
    """
    Safely load a scikit-learn model with version compatibility handling

    Args:
        model_path: Path to the pickle file
        suppress_warnings: Whether to suppress sklearn version warnings

    Returns:
        Dictionary containing model info and compatibility status
    """
    result = {
        'model': None,
        'loaded_successfully': False,
        'version_mismatch': False,
        'training_version': 'unknown',
        'current_version': sklearn.__version__,
        'warnings': [],
        'metadata': {}
    }

    try:
        # Temporarily capture sklearn warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            # Handle both old format (direct model) and new format (metadata dict)
            if isinstance(data, dict) and 'model' in data:
                # New format with metadata
                result['model'] = data['model']
                result['metadata'] = data
                result['training_version'] = data.get('sklearn_version', 'unknown')
            else:
                # Old format - direct model
                result['model'] = data
                result['training_version'] = 'unknown (legacy format)'

            result['loaded_successfully'] = True

            # Check for version warnings
            sklearn_warnings = [warning for warning in w if 'InconsistentVersionWarning' in str(warning.category)]
            if sklearn_warnings:
                result['version_mismatch'] = True
                result['warnings'] = [str(warning.message) for warning in sklearn_warnings]

                if not suppress_warnings:
                    print(f"⚠️  Version mismatch detected:")
                    print(f"   Training version: {result['training_version']}")
                    print(f"   Current version: {result['current_version']}")
                    print(f"   Model should still work but may have subtle differences")

    except Exception as e:
        result['error'] = str(e)
        print(f"❌ Error loading model: {e}")

    return result

def check_version_compatibility(training_version: str, current_version: str = None) -> Dict[str, Any]:
    """
    Check compatibility between sklearn versions

    Args:
        training_version: Version used for training
        current_version: Current sklearn version (defaults to current)

    Returns:
        Compatibility analysis
    """
    if current_version is None:
        current_version = sklearn.__version__

    def parse_version(version_str):
        # Remove any non-numeric suffixes and parse major.minor.patch
        clean_version = version_str.split(' ')[0]  # Remove " (legacy)" etc.
        try:
            parts = clean_version.split('.')
            return tuple(int(x) for x in parts[:3])  # Take only major.minor.patch
        except:
            return (0, 0, 0)

    training_ver = parse_version(training_version)
    current_ver = parse_version(current_version)

    # Calculate compatibility
    major_diff = abs(training_ver[0] - current_ver[0])
    minor_diff = abs(training_ver[1] - current_ver[1])

    if major_diff > 0:
        compatibility = 'incompatible'
        risk_level = 'high'
    elif minor_diff > 1:
        compatibility = 'risky'
        risk_level = 'medium'
    elif minor_diff == 1:
        compatibility = 'compatible_with_warnings'
        risk_level = 'low'
    else:
        compatibility = 'fully_compatible'
        risk_level = 'none'

    return {
        'training_version': training_version,
        'current_version': current_version,
        'compatibility': compatibility,
        'risk_level': risk_level,
        'recommendation': get_compatibility_recommendation(compatibility)
    }

def get_compatibility_recommendation(compatibility: str) -> str:
    """Get recommendation based on compatibility level"""
    recommendations = {
        'fully_compatible': 'Safe to use. No issues expected.',
        'compatible_with_warnings': 'Safe to use but may show warnings. Consider upgrading training environment.',
        'risky': 'May work but could have behavioral differences. Test thoroughly before production use.',
        'incompatible': 'High risk of failure. Strongly recommend retraining with current sklearn version.'
    }
    return recommendations.get(compatibility, 'Unknown compatibility level.')

def create_model_loading_code(algorithm: str, include_compatibility_check: bool = True) -> str:
    """Generate model loading code with compatibility handling"""

    base_code = f"""
import pickle
import warnings
import sklearn

# Load the {algorithm} model
with open('{algorithm}_model.pkl', 'rb') as f:
    """

    if include_compatibility_check:
        base_code += f"""    # Capture version warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = pickle.load(f)

        # Handle both new and legacy model formats
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            training_version = data.get('sklearn_version', 'unknown')
            print(f"Model trained with sklearn {{training_version}}, running on {{sklearn.__version__}}")
        else:
            model = data
            print("Legacy model format detected")

        # Check for version warnings
        version_warnings = [w for w in w if 'InconsistentVersionWarning' in str(w.category)]
        if version_warnings:
            print("⚠️  Version compatibility warnings detected - model should still work")

# Use the model for inference
predictions = model.predict(X)  # 1 for normal, -1 for anomaly
scores = model.decision_function(X)  # Anomaly scores (lower = more anomalous)
"""
    else:
        base_code += f"""    data = pickle.load(f)

    # Handle both new and legacy model formats
    if isinstance(data, dict) and 'model' in data:
        model = data['model']
    else:
        model = data

# Use the model for inference
predictions = model.predict(X)  # 1 for normal, -1 for anomaly
scores = model.decision_function(X)  # Anomaly scores
"""

    return base_code