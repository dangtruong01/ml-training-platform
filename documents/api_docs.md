
# API Documentation

The backend provides a RESTful API for interacting with the platform.

**Base URL**: `http://localhost:8000/api`

---

## Projects

### `POST /projects/create`
Create a new project.
- **Body (form-data)**:
  - `project_name` (str): The name of the project.
  - `project_type` (str): `object_detection`, `segmentation`, or `anomaly_detection`.
  - `description` (str, optional): A description for the project.
- **Response**:
  ```json
  {
    "status": "success",
    "project_id": "My_Project_1234abcd"
  }
  ```

### `POST /projects/{project_id}/upload-zip-dataset`
Upload a complete YOLO dataset in a ZIP file.
- **Parameters**:
  - `project_id` (str): The ID of the project.
- **Body (form-data)**:
  - `dataset_zip` (file): The ZIP file containing the dataset.
- **Response**:
  ```json
  {
    "status": "success",
    "message": "ZIP dataset uploaded and processed successfully",
    "dataset_info": { ... }
  }
  ```

### `GET /projects/{project_id}/validate-dataset`
Validate if a project's dataset is ready for training.
- **Response**:
  ```json
  {
    "status": "success",
    "validation": {
      "is_ready": true,
      "missing_requirements": []
    }
  }
  ```

---

## Cloud Training

### `POST /cloud-training/train-project-cloud/{project_id}`
Start a new training job on Vertex AI.
- **Parameters**:
  - `project_id` (str): The ID of the project.
- **Body (form-data)**:
  - `algorithm` (str): e.g., `yolov8`.
  - `model_size` (str): `n`, `s`, `m`, `l`, `x`.
  - `epochs` (int): Number of training epochs.
  - `batch_size` (int): Batch size.
  - `learning_rate` (float): Learning rate.
  - `device` (str): `cpu` or `cuda`.
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Cloud training job started successfully",
    "job_id": "vertex-job-12345",
    "job_url": "https://console.cloud.google.com/..."
  }
  ```

### `GET /cloud-training/job-status/{job_id}`
Check the status of a specific training job.
- **Response**:
  ```json
  {
    "status": "success",
    "job_id": "vertex-job-12345",
    "job_status": "SUCCEEDED",
    "output_files": [ ... ]
  }
  ```

---

## Models

### `GET /models/`
List all available trained models.
- **Response**:
  ```json
  {
    "status": "success",
    "models": [
      {
        "model_id": "detection_training_abcdef12",
        "model_name": "yolov8m.pt",
        ...
      }
    ]
  }
  ```

### `GET /models/{model_id}/download`
Download the PyTorch (`.pt`) model file.
- **Response**: A file download of `best.pt`.

### `GET /models/{model_id}/download-onnx`
Download the model in ONNX format. This triggers an on-the-fly conversion if the ONNX version doesn't exist.
- **Response**: A file download of `model.onnx`.

---
