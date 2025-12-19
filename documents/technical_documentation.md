# Technical Documentation (Comprehensive)

**Audience**: Software Engineers, DevOps, and future maintainers of the project.
**Purpose**: This document provides a deep, technical overview of the system's architecture, components, data models, and development practices, intended to facilitate a smooth project handover.

---

## 1. Core Architectural Principles

The platform is designed around a set of core principles to ensure scalability, maintainability, and flexibility:

1.  **Separation of Concerns**: The system is divided into three distinct parts: a **Frontend** for user interaction, a **Backend** for business logic and orchestration, and containerized **ML Services** for heavy computation.
2.  **Stateless Backend**: The FastAPI backend is designed to be stateless. All application state is externalized to a PostgreSQL database (for metadata) and Google Cloud Storage (for files), allowing the backend to be scaled horizontally without issue.
3.  **Abstraction of Services**: Critical services like storage and ML training are abstracted. For example, the `StorageService` can be configured to use local disk for development or GCS for production, without changing the application code.
4.  **Containerization for Consistency**: All training and inference tasks are executed within Docker containers. This guarantees a reproducible environment, eliminating "it works on my machine" problems, especially when moving from local development to cloud-based training on Vertex AI.

---

## 2. System Components

### 2.1. Frontend
- **Framework**: React with Material-UI.
- **Responsibilities**:
  - Provides the user interface for project management, data upload, training configuration, and results visualization.
  - Communicates with the backend via REST API calls.
  - Handles client-side state management and user interaction.

### 2.2. Backend (FastAPI)
- **Framework**: Python with FastAPI.
- **Responsibilities**:
  - **API Server**: Exposes a REST API for the frontend to consume.
  - **Orchestrator**: Manages the entire ML lifecycle. It does not perform heavy computation itself but instead delegates tasks to other services (e.g., submitting a job to Vertex AI).
  - **Business Logic**: Handles project creation, user management (future), data validation, and access control.
  - **Database Interaction**: Manages all communication with the PostgreSQL database via SQLAlchemy.

### 2.3. Training Container
- **Technology**: Docker.
- **Location**: `backend/training-container/`
- **Responsibilities**:
  - A self-contained environment with all necessary dependencies (`ultralytics`, `torch`, etc.) to run a training script.
  - The container is designed to be generic. It accepts arguments (like dataset path, epochs, model size) to execute a specific training run.
  - This same container image is used for both local debugging and for running jobs on Vertex AI, ensuring consistency.

---

## 3. Backend Deep Dive

The backend is the core of the system. Its internal structure is organized by function.

### 3.1. Directory Structure
```
backend/
├── app/
│   ├── api/
│   │   └── endpoints/  # FastAPI routers for each resource (projects, models, etc.)
│   └── main.py         # Main application entry point
├── services/
│   ├── core/           # Foundational services (database, storage)
│   ├── ml/             # Machine learning logic (YOLO, auto-annotation)
│   └── cloud/          # Cloud integration (Vertex AI)
├── training-container/ # Dockerfile and scripts for the training job
└── .env                # Environment variable configuration
```

### 3.2. Core Services (`/services/core`)
- **`database_service.py`**:
  - **Purpose**: The single source of truth for all database interactions.
  - **Technology**: SQLAlchemy Core (for query building) and ORM (for table definitions).
  - **Key Functions**: `create_project`, `add_uploaded_file`, `create_training_job`, `update_job_status`, `get_model_by_id`.
- **`storage_service.py`**:
  - **Purpose**: An abstraction layer for file storage. This is a critical design choice that allows the application to be environment-agnostic.
  - **How it works**: It checks the `STORAGE_TYPE` environment variable.
    - If `local`, it performs file operations on the local disk.
    - If `gcs`, it uses the `google-cloud-storage` library to interact with a GCS bucket.
  - **Key Functions**: `upload_file`, `download_file`, `delete_folder`, `get_project_directory`.

### 3.3. Machine Learning Services (`/services/ml`)
- **`yolo_service.py`**:
  - **Purpose**: Orchestrates all YOLO-related tasks.
  - **Key Functions**:
    - `train_detection_from_project_cloud()`: Gathers all necessary data (config, dataset path) and calls the `vertex_ai_service` to launch a cloud training job.
    - `convert_pytorch_to_onnx()`: Loads a trained `.pt` model and exports it to the ONNX format for production deployment.
- **`auto_annotation_service.py`**:
  - **Purpose**: Encapsulates the logic for the auto-annotation pipeline.
  - **How it works**: It follows a two-stage process:
    1.  **Open-Set Detection**: Uses GroundingDINO to find objects in an image based on a free-text prompt.
    2.  **Segmentation**: Passes the detected bounding boxes to the Segment Anything Model (SAM) to generate precise pixel-level masks.
    3.  **Conversion**: Transforms the SAM masks into YOLO's polygon format.

### 3.4. Cloud Services (`/services/cloud`)
- **`vertex_ai_service.py`**:
  - **Purpose**: Handles all direct communication with the Google Vertex AI API.
  - **Key Functions**:
    - `create_custom_job()`: Constructs the detailed job specification required by Vertex AI. This includes defining the worker pool (machine type, GPU type), the path to the training container image in GCR, and the command-line arguments to pass to the container.
    - `get_job_status()`: Queries the Vertex AI API for the current state of a job.
- **`vertex_ai_monitor.py`**:
  - **Purpose**: A background service that periodically checks the status of active training jobs.
  - **How it works**: It runs in a separate thread (`threading.Thread`). Every 60 seconds, it queries the database for jobs in a "running" or "submitted" state, calls `vertex_ai_service` to get their latest status from Google Cloud, and updates the database accordingly. This is crucial for automatically marking jobs as "completed" or "failed".

---

## 4. Database Schema

The database (PostgreSQL) stores all metadata. The key tables are:

- **`projects`**: Stores information about each project.
  - `id` (PK), `project_id` (str), `project_name` (str), `project_type` (str), `created_at`.
- **`uploaded_files`**: Tracks every file uploaded to the system.
  - `id` (PK), `project_id` (FK), `file_type` (str, e.g., 'image', 'label'), `storage_url` (str), `relative_path` (str).
- **`training_jobs`**: A record of every training job initiated.
  - `id` (PK), `task_id` (str), `project_id` (FK), `status` (str, e.g., 'pending', 'running', 'completed'), `training_config` (JSON), `vertex_ai_job_id` (str).
- **`models`**: Stores metadata about trained models.
  - `id` (PK), `model_id` (str), `project_id` (FK), `training_job_id` (FK), `model_path` (str), `performance_metrics` (JSON).

---

## 5. Configuration Management

- **`.env` file**: All environment-specific configuration is managed through a `.env` file in the `backend` directory. This file is **not** checked into source control.
- **Loading**: The `start_server.sh` script is responsible for loading these variables into the environment before starting the Uvicorn server.
- **Key Variables**:
  - `DATABASE_URL`: The connection string for the PostgreSQL database.
  - `STORAGE_TYPE`: `local` or `gcs`.
  - `GCS_BUCKET_NAME`: The name of the Google Cloud Storage bucket to use.
  - `GCP_PROJECT_ID`, `GCP_REGION`: Google Cloud project details for Vertex AI.

---

## 6. Local Development & Debugging

- **Running the System**: The most reliable way to run the system locally is to use the `start_server.sh` script, as it ensures all environment variables and Python paths are set correctly.
- **Database**: For local development, it's recommended to run PostgreSQL in a Docker container.
- **Cloud Services**: When `STORAGE_TYPE` is `local`, the application does not need cloud access for most features. However, to test cloud training, you must have `gcloud` authenticated and the correct GCP variables set in your `.env` file.
- **Debugging Tips**:
  - The backend includes several `/debug-*` endpoints (e.g., `/api/projects/{id}/debug-file-counts`) that provide raw data from the database and are useful for troubleshooting state issues.
  - Set a breakpoint in the `train_project_cloud` function in `cloud_training.py` to inspect the configuration being sent to Vertex AI just before a job is submitted.
  - Check the logs in the GCP console for Vertex AI jobs to see the output from the training container itself.

---

## 7. Feature Catalog (Detailed)

This section enumerates all major features implemented, their purpose, and technical details.

### 7.1 Project & Dataset Management
- **Create Project**: Generates a unique `project_id`, persists metadata, and initializes storage directories.
- **Upload Training Data (files)**: Accepts individual images and annotations. Files are stored via `storage_service` and tracked in `uploaded_files`.
- **Upload YOLO Dataset (ZIP)**: Validates ZIP, extracts locally, counts images/labels, regenerates a cloud-safe `data.yaml` (relative paths), uploads all contents to `projects/{project_id}/` on GCS, and records metadata.
- **Upload CVAT Dataset (ZIP)**: Detects expected structure (`images/`, `labels/`, `classes.txt`), converts to YOLO using `convert_cvat_to_yolo.py`, persists results and `data.yaml`.
- **Validation**: `validate_dataset` checks readiness rules per project type, with database fallbacks for annotation counts.
- **Dataset Statistics**: Summarizes counts and readiness state from DB.

### 7.2 Training (Local & Cloud)
- **Local Training**: Standalone scripts (e.g., `scripts/training/defect_detection_training/train_defect_model.py`) using Ultralytics YOLO APIs.
- **Cloud Training (Vertex AI)**:
  - `cloud_training.py` endpoint accepts hyperparameters and project ID.
  - `yolo_service` resolves dataset, constructs training args, and calls `vertex_ai_service.create_custom_job()`.
  - Vertex AI runs the container, reads dataset from GCS, writes outputs back to `mltraining-models` bucket.
  - `vertex_ai_monitor` watches job status and updates DB.

### 7.3 Model Management
- **Model Listing**: Aggregates trained models from DB and/or storage.
- **Downloads**: Provide PyTorch `.pt` and dynamic ONNX conversion via Ultralytics `model.export(format='onnx')`, with environment-aware dependency checks.
- **Metadata**: Stores mAP and other metrics (when available) in `models` table.

### 7.4 Auto-Annotation
- **Text-Guided Detection**: GroundingDINO locates objects based on prompts.
- **Segmentation**: SAM generates precise masks from detected regions.
- **Conversion to YOLO**: Masks converted to polygons, saved as YOLO labels. Integrates into project dataset.
- **Services**: Implemented under `ml/auto_annotation` with endpoints to trigger and manage workflows.

### 7.5 Storage Abstraction
- **Local vs GCS**: Unified interface for upload/download/path generation. Ensures portability across environments.
- **Project Directories**: Standardized layout returned by `get_project_directories(project_id)`.

### 7.6 Monitoring & Debugging
- **Vertex AI Monitoring**: Background thread, periodic status polling, and DB updates.
- **Debug Endpoints**: `/api/projects/{id}/debug-file-counts`, `/api/cloud-training/debug-*` provide internal state snapshots.

---

## 8. Technology Stack (Detailed Versions & Rationale)

- **Language**: Python 3.10–3.13 (tested), Node.js (Frontend)
- **Backend Framework**: FastAPI (async-first, OpenAPI generation)
- **DB Layer**: SQLAlchemy (ORM + Core) over PostgreSQL
- **ML**:
  - Ultralytics YOLOv8 (detection & segmentation)
  - PyTorch (training runtime)
  - GroundingDINO (open-vocabulary detection)
  - SAM (segmentation)
  - Optional: Anomalib (advanced anomaly detection)
- **Cloud**:
  - Google Cloud Storage (datasets, artifacts)
  - Vertex AI (custom training jobs)
  - Firestore (optional) and Cloud SQL (PostgreSQL)
- **Containers**: Docker; images stored in GCR
- **Frontend**: React + Material-UI, build output deployable to GCS static hosting
- **Reasoning**:
  - FastAPI for speed and maintainability; Ultralytics for a stable high-level API; Vertex AI for managed training at scale.

---

## 9. Design Structures & Patterns

- **Router-per-domain**: `projects`, `models`, `cloud_training` maintain clear API boundaries.
- **Service Abstractions**: `database_service`, `storage_service`, `vertex_ai_service` enforce separation from framework specifics.
- **Configuration via Environment**: No hardcoded secrets; `.env` + `start_server.sh` ensure consistent runtime.
- **Idempotent Data Processing**: Cloud-safe `data.yaml` regeneration to avoid path errors.
- **Background Workers**: Monitoring implemented as a managed thread with safe start/stop endpoints.

---

## 10. Operational Workflows (Step-by-Step)

### 10.1 End-to-End Training (Cloud)
1. User uploads ZIP → Backend extracts, validates, regenerates `data.yaml`.
2. Files uploaded to `gs://{GCS_BUCKET}/projects/{project_id}/`.
3. User triggers training → Backend constructs job spec → Vertex AI runs container.
4. Container saves outputs to `gs://mltraining-models/...`.
5. Monitor updates DB → Frontend shows job completion & model availability.

### 10.2 ONNX Export
1. Request `/models/{model_id}/download-onnx`.
2. Backend locates `.pt` (local or GCS), loads via Ultralytics.
3. Checks dependencies (`onnx`, `onnxruntime`), installs if needed.
4. Exports to ONNX; returns file for download.

### 10.3 Auto-Annotation Flow
1. Frontend sends image + prompt.
2. Backend runs GroundingDINO → bounding boxes.
3. SAM refines to masks.
4. Convert masks → YOLO polygons; store labels and update dataset.

---

## 11. Environments & Configuration

- **Development**: `STORAGE_TYPE=local`, local PostgreSQL, optional GCS.
- **Staging**: `STORAGE_TYPE=gcs`, Cloud SQL, Vertex AI; limited quotas.
- **Production**: Hardened secrets, IAM roles, VPC, Vertex AI production quotas.
- **Key Env Vars**: `DATABASE_URL`, `STORAGE_TYPE`, `GCS_BUCKET_NAME`, `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_MODELS_BUCKET`.

---

## 12. Security & Access Control

- **Secrets**: Stored in `.env` or Secret Manager; never committed.
- **IAM**: Service accounts with least privilege for Storage, Vertex AI, SQL.
- **Network**: Prefer private access between GKE and Cloud SQL.
- **Uploads**: Validate file types and sizes; sanitize ZIP contents.

---

## 13. Logging, Metrics, and Monitoring

- **Backend Logs**: Uvicorn/FASTAPI logs; structured prints in critical paths.
- **Vertex AI Logs**: Access via GCP console for container output.
- **Metrics (Future)**: Integrate Prometheus/Grafana or Cloud Monitoring for API latency, job counts, failures.

---

## 14. CI/CD & Release

- **Build**: Docker images for backend and training container built via CI.
- **Test**: Unit tests for services; integration tests against local stack.
- **Deploy**: GKE deployments with rolling updates; frontend build to GCS.
- **Versioning**: Tag models; store metrics; maintain changelog.

---

## 15. Scalability & Performance

- **Horizontal Scale**: Stateless backend allows scaling API replicas.
- **Data Scale**: Large datasets stored on GCS; streaming reads in containers.
- **GPU Utilization**: Vertex AI worker pool sizing by job type; autoscaling via job specs.

---

## 16. Known Limitations & Workarounds

- **ONNX Export Dependencies**: On some macOS/ARM environments, protobuf/onnx versions may conflict; prefer exporting in Linux training container.
- **GroundingDINO Local Install**: Ensure `third_party/GroundingDINO` has all Python deps (e.g., `addict`) installed in backend environment.
- **Anomalib Optionality**: Advanced anomaly detection disabled if library missing; base ROI + heuristics still work.

---

## 17. Handover Checklist

- [ ] Confirm `.env` variables for target environment.
- [ ] Verify GCS buckets: datasets (`projects/`), models (`mltraining-models`).
- [ ] Build/push latest backend and training container images to GCR.
- [ ] Run smoke tests: dataset upload, cloud training start, model listing, ONNX export.
- [ ] Ensure monitoring thread starts and reports statuses.

---

## 18. Appendices

- **Key Paths**:
  - Datasets: `gs://{GCS_BUCKET}/projects/{project_id}/`
  - Models: `gs://mltraining-models/{type}/{training_id}/`
- **Important Files**:
  - `backend/app/api/endpoints/*`
  - `backend/services/*`
  - `backend/training-container/*`
  - `documents/*`
