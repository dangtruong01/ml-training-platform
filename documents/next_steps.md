
# Next Steps & Future Work

This document outlines planned features, areas for improvement, and the roadmap for future development.

## 1. Uncompleted Features & Immediate Priorities

### a. Full Cloud Integration for Auto-Annotation
- **Current State**: The auto-annotation pipeline (GroundingDINO + SAM) runs within the backend container. This is not scalable for large datasets or high-resolution images.
- **Next Step**: Move the auto-annotation logic into its own containerized service.
- **Future Goal**: Create a dedicated Vertex AI pipeline for auto-annotation. A user uploads images, and a multi-step pipeline (e.g., run GroundingDINO, then run SAM) processes them in the cloud, similar to how training jobs work.

### b. Real-Time Job Monitoring and Websockets
- **Current State**: The frontend polls the backend periodically to check the status of training jobs.
- **Next Step**: Implement a WebSocket connection between the frontend and backend.
- **Future Goal**: The backend's `vertex_ai_monitor` service will push real-time status updates (e.g., "Epoch 5/50 complete", "mAP: 0.75") to the frontend, providing a much better user experience.

### c. Enhanced Model Management & Versioning
- **Current State**: Models are stored and identified by a training job ID. There is no explicit versioning system.
- **Next Step**: Add a versioning system to models (e.g., v1, v2). Allow users to "promote" a specific model version to a "production" or "staging" status.
- **Future Goal**: Integrate with a formal model registry like Vertex AI Model Registry. This would provide a centralized place to manage, version, and deploy models.

## 2. Future Features Roadmap

### a. Interactive Data Exploration and Annotation
- **Vision**: An in-browser tool that allows users to view their uploaded images, draw bounding boxes, and correct annotations.
- **Technology**: Integrate an open-source labeling tool like `react-image-annotate` or a simplified version built with Konva.js.
- **Benefit**: Creates a complete, end-to-end workflow within the platform, from data upload to annotation to training.

### b. Advanced Experiment Tracking
- **Vision**: A dashboard similar to MLflow or Weights & Biases, allowing users to compare different training runs.
- **Features**:
  - Plotting metrics (loss, mAP) from multiple runs on the same chart.
  - Comparing hyperparameters side-by-side.
  - Saving and viewing generated plots (e.g., confusion matrix) for each run.
- **Benefit**: Empowers users to make data-driven decisions about which model is best.

### c. Multi-Cloud and On-Premise Support
- **Vision**: Abstract the cloud services further to support other providers (like AWS SageMaker) or even on-premise GPU clusters.
- **Implementation**:
  - Create a generic `CloudTrainingService` interface.
  - Implement concrete classes for different providers (`VertexAIService`, `SageMakerService`).
  - The backend would choose the provider based on configuration.
- **Benefit**: Makes the platform more flexible and adaptable to different enterprise environments.

### d. User Authentication and Multi-Tenancy
- **Vision**: Add user accounts, roles, and permissions.
- **Features**:
  - Users can only see and manage their own projects.
  - "Admin" role for managing users and system-wide settings.
- **Technology**: Integrate a service like Firebase Authentication, Auth0, or a simple JWT-based system.
- **Benefit**: Makes the platform secure and ready for use by teams.

---
