
# Quick Guide & Walkthrough

This guide provides a step-by-step walkthrough for setting up and using the platform to train a new defect detection model.

## 1. Prerequisites

- **Docker**: Ensure Docker is installed and running on your system.
- **Python 3.9+**: A recent version of Python is required.
- **Google Cloud SDK**: If using cloud features, ensure `gcloud` is configured.

## 2. Setup and Installation

### a. Clone the Repository
```bash
git clone <your-repository-url>
cd open-trainer
```

### b. Set Up the Backend
1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables:**
    - Copy the `.env.example` file to `.env`.
    - Fill in your database credentials and Google Cloud project details.
4.  **Start the Backend Server:**
    - Use the provided script to ensure all environment variables are loaded correctly.
    ```bash
    ./start_server.sh
    ```
    - The API will be available at `http://127.0.0.1:8000`.

### c. Set Up the Frontend
1.  **Navigate to the frontend directory:**
    ```bash
    cd ../frontend
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
3.  **Start the frontend development server:**
    ```bash
    npm start
    ```
    - The web interface will be available at `http://localhost:3000`.

## 3. Walkthrough: Training Your First Model

### Step 1: Create a New Project
1.  Open the web interface in your browser.
2.  Click on the "New Project" button.
3.  Fill in the project details:
    - **Project Name**: `My Defect Detector`
    - **Project Type**: `Object Detection`
4.  Click "Create Project". You will be redirected to the project's dashboard.

### Step 2: Upload Your Dataset
1.  Prepare your dataset in the YOLO format and compress it into a single **ZIP file**. The structure should be:
    ```
    my_dataset.zip
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── classes.txt  (or data.yaml)
    ```
2.  On the project dashboard, find the "Upload Dataset" section.
3.  Drag and drop your ZIP file or click to browse.
4.  The system will automatically process the ZIP file, validate its contents, and upload it to the appropriate storage location (local or GCS).

### Step 3: Start a Cloud Training Job
1.  Navigate to the "Training" tab for your project.
2.  Click "Start New Training Job".
3.  Configure the training parameters:
    - **Model Size**: `yolov8s` (a good starting point)
    - **Epochs**: `50`
    - **Batch Size**: `16`
    - **Device**: `cuda` (to ensure it runs on a GPU in the cloud)
4.  Click "Start Training".
5.  The system will submit the job to Vertex AI. You can monitor its status directly from the web interface.

### Step 4: View and Use Your Trained Model
1.  Once the training job is complete, a new model will appear in the "Models" section of your project.
2.  From here, you can:
    - **View Performance**: See the mAP scores and other metrics.
    - **Download the Model**: Download the `best.pt` weights for local use.
    - **Download as ONNX**: Get a production-ready ONNX version of the model for deployment in other systems.
    - **Test Inference**: Upload a sample image to see the model's predictions in real-time.

---
