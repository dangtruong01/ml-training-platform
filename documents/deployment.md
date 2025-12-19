
# Deployment Guide

This guide covers the steps for deploying the application to a production environment on Google Cloud Platform.

## 1. Prerequisites

- A Google Cloud Project with billing enabled.
- APIs enabled: Vertex AI, Google Cloud Storage, Google Kubernetes Engine (GKE), Firestore.
- A service account with appropriate permissions (Storage Admin, Vertex AI User, Kubernetes Engine Admin).

## 2. Backend Deployment (GKE)

The FastAPI backend is best deployed as a containerized application on GKE for scalability and reliability.

### a. Build and Push the Docker Image
1.  **Navigate to the `backend` directory.**
2.  **Build the Docker image:**
    ```bash
    docker build -t gcr.io/YOUR_GCP_PROJECT_ID/open-trainer-backend:latest .
    ```
3.  **Configure Docker to use gcloud credentials:**
    ```bash
    gcloud auth configure-docker
    ```
4.  **Push the image to Google Container Registry (GCR):**
    ```bash
    docker push gcr.io/YOUR_GCP_PROJECT_ID/open-trainer-backend:latest
    ```

### b. Deploy to GKE
1.  **Create a GKE cluster** through the GCP console or `gcloud`.
2.  **Create a Kubernetes Deployment file (`deployment.yaml`):**
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: open-trainer-backend
    spec:
      replicas: 2
      selector:
        matchLabels:
          app: backend
      template:
        metadata:
          labels:
            app: backend
        spec:
          containers:
          - name: backend
            image: gcr.io/YOUR_GCP_PROJECT_ID/open-trainer-backend:latest
            ports:
            - containerPort: 8000
            env:
            - name: DATABASE_URL
              value: "postgresql://user:password@db-host:5432/dbname"
            - name: GCS_BUCKET_NAME
              value: "your-gcs-bucket-name"
            # Add other environment variables
    ```
3.  **Create a Kubernetes Service file (`service.yaml`) to expose the deployment:**
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: backend-service
    spec:
      type: LoadBalancer
      selector:
        app: backend
      ports:
      - protocol: TCP
        port: 80
        targetPort: 8000
    ```
4.  **Apply the configurations:**
    ```bash
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    ```

## 3. Frontend Deployment (Cloud Storage)

The React frontend is a static application and can be deployed easily using a GCS bucket.

1.  **Build the production version of the frontend:**
    ```bash
    cd frontend
    npm run build
    ```
2.  **Create a GCS bucket** to host the static files.
3.  **Sync the build directory with the bucket:**
    ```bash
    gsutil -m rsync -r build/ gs://your-frontend-bucket-name
    ```
4.  **Set public access** on the bucket objects to make the site visible.
5.  **Configure a Load Balancer** (like Google Cloud Armor) in front of the bucket to provide an HTTPS URL and CDN caching.

## 4. Training Container Deployment

The training code needs to be containerized and pushed to GCR so Vertex AI can use it.

1.  **Navigate to the `backend/training-container` directory.**
2.  **Build and push the Docker image:**
    ```bash
    docker build -t gcr.io/YOUR_GCP_PROJECT_ID/training-container:latest .
    docker push gcr.io/YOUR_GCP_PROJECT_ID/training-container:latest
    ```
3.  When a training job is submitted, the backend will reference this image path in the Vertex AI job request.

## 5. Database Deployment (Cloud SQL)

For a production environment, use a managed PostgreSQL database like Google Cloud SQL.

1.  **Create a Cloud SQL for PostgreSQL instance.**
2.  **Create a database and a user.**
3.  **Configure networking** to allow connections from your GKE cluster (e.g., using a Cloud SQL Auth Proxy sidecar container in your backend deployment).
4.  Update the `DATABASE_URL` environment variable in your GKE deployment to point to the Cloud SQL instance.

---
