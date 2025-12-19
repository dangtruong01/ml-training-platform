# C++ Inference Specification for Isolation Forest Models

## Root Cause Analysis

Your C++ inference was crashing because of a **feature dimension mismatch**:

- **Your C++ code**: Sending 150,528 features (224×224×3 flattened image pixels)
- **Your models**: Expecting 50 features (after PCA feature reduction)

This wasn't a sklearn version issue - it was an input preprocessing mismatch!

## Required Image Preprocessing

### For Isolation Forest Anomaly Detection Models

**Complete Preprocessing Pipeline:**
1. **Resize**: Image to exactly 64×64 pixels
2. **Convert**: To RGB (3 channels)
3. **Flatten**: Convert to 1D array → 12,288 features (64 × 64 × 3)
4. **Normalize**: Scale pixel values to [0, 1] range
5. **Result**: 12,288 features for model input

### C++ Implementation Required

```cpp
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<float> preprocessImageForAnomalyDetection(const cv::Mat& input_image) {
    cv::Mat processed_image;

    // Step 1: Resize to 64x64
    cv::resize(input_image, processed_image, cv::Size(64, 64));

    // Step 2: Ensure RGB format (if needed)
    if (processed_image.channels() == 4) {
        cv::cvtColor(processed_image, processed_image, cv::COLOR_BGRA2RGB);
    } else if (processed_image.channels() == 1) {
        cv::cvtColor(processed_image, processed_image, cv::COLOR_GRAY2RGB);
    } else if (processed_image.channels() == 3) {
        cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);
    }

    // Step 3: Convert to float and normalize to [0, 1]
    processed_image.convertTo(processed_image, CV_32F, 1.0/255.0);

    // Step 4: Flatten to 1D vector (64 * 64 * 3 = 12,288 features)
    std::vector<float> features;
    features.reserve(64 * 64 * 3);

    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            cv::Vec3f pixel = processed_image.at<cv::Vec3f>(y, x);
            features.push_back(pixel[0]); // R
            features.push_back(pixel[1]); // G
            features.push_back(pixel[2]); // B
        }
    }

    // Verify correct size
    assert(features.size() == 12288);

    return features;
}
```

### Python Equivalent (for reference)

```python
import numpy as np
from PIL import Image

def preprocess_image_for_anomaly_detection(image_path):
    # Load and resize to 64x64
    img = Image.open(image_path).convert('RGB').resize((64, 64))

    # Convert to numpy array and flatten
    features = np.array(img).flatten()  # Shape: (12288,)

    # Normalize to [0, 1]
    features = features / 255.0

    return features
```

## Model Loading (Enhanced Format)

Your new models include metadata. Load them like this:

```python
import pickle

# Load model with metadata
with open('anomaly_model.pkl', 'rb') as f:
    data = pickle.load(f)

    if isinstance(data, dict) and 'model' in data:
        # New format with metadata
        model = data['model']
        sklearn_version = data['sklearn_version']
        preprocessing_info = data['image_preprocessing']

        print(f"Model trained with sklearn {sklearn_version}")
        print(f"Expected features: {preprocessing_info['flattened_features']}")
        print(f"Image size: {preprocessing_info['resize_shape']}")
    else:
        # Legacy format
        model = data

# Use model
predictions = model.predict([features])  # features must be 12,288 length
```

## Error Handling Recommendations

Add this to your C++ code:

```cpp
// Before calling sklearn predict
if (features.size() != 12288) {
    throw std::runtime_error(
        "Feature size mismatch: expected 12288, got " +
        std::to_string(features.size())
    );
}

// Verify feature range
for (const auto& f : features) {
    if (f < 0.0f || f > 1.0f) {
        throw std::runtime_error(
            "Feature out of range: expected [0, 1], got " +
            std::to_string(f)
        );
    }
}
```

## What Changed

### ✅ **Fixed in Training Code:**
1. **Vertex AI trainer** now uses real image features instead of mock data
2. **Local trainer** now matches Vertex AI preprocessing exactly
3. **Both trainers** save metadata with preprocessing information
4. **sklearn version** fixed to 1.7.1 across all environments

### ✅ **New Model Format:**
```python
{
    'model': <IsolationForest object>,
    'sklearn_version': '1.7.1',
    'algorithm': 'isolation_forest',
    'training_features': 12288,
    'training_samples': 1000,
    'trained_at': '2025-10-14T16:45:00',
    'image_preprocessing': {
        'resize_shape': (64, 64),
        'channels': 3,
        'normalization': 'scale_0_to_1',
        'flattened_features': 12288
    }
}
```

## Next Steps

1. **Update your C++ preprocessing** to match the specification above
2. **Train a new model** - it will now use real image features (12,288 dimensions)
3. **Test with the new model** - no more version warnings or segfaults
4. **Verify feature extraction** by comparing C++ output with Python preprocessing

## Testing Your Fix

```bash
# Train a new model (will use real 64x64x3 features)
curl -X POST http://localhost:8000/api/train-anomaly

# Download the new model
curl -O http://localhost:8000/api/models/{model_id}/download

# Test the preprocessing in Python first
python -c "
import numpy as np
from PIL import Image

# Test your image
img = Image.open('test_image.jpg').convert('RGB').resize((64, 64))
features = np.array(img).flatten() / 255.0
print(f'Features shape: {features.shape}')  # Should be (12288,)
print(f'Features range: [{features.min():.3f}, {features.max():.3f}]')  # Should be [0, 1]
"
```

Your models will now work correctly with the proper 12,288-feature input!