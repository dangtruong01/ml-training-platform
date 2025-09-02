# Project Structure

This document outlines the proposed directory structure for the Object Detection and Segmentation POC.

```
/
|-- backend/
|   |-- app/
|   |   |-- __init__.py
|   |   |-- main.py             # FastAPI application
|   |   |-- api/
|   |   |   |-- __init__.py
|   |   |   |-- endpoints/
|   |   |   |   |-- __init__.py
|   |   |   |   |-- train.py        # /train-detect, /train-seg endpoints
|   |   |   |   |-- predict.py      # /predict-detect, /predict-seg endpoints
|   |-- core/
|   |   |-- __init__.py
|   |   |-- config.py           # Configuration settings
|   |-- services/
|   |   |-- __init__.py
|   |   |-- yolo_service.py     # Logic for running YOLOv8 scripts
|   |-- requirements.txt
|
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |   |-- Train.js
|   |   |   |-- Predict.js
|   |   |-- App.js
|   |   |-- index.js
|   |-- public/
|   |   |-- index.html
|   |-- package.json
|
|-- ml/
|   |-- datasets/
|   |   |-- detection/
|   |   |   |-- data.yaml
|   |   |   |-- images/
|   |   |   |-- labels/
|   |   |-- segmentation/
|   |   |   |-- data.yaml
|   |   |   |-- images/
|   |   |   |-- labels/
|   |-- models/
|   |   |-- yolov8n.pt
|   |   |-- yolov8n-seg.pt
|   |-- scripts/
|   |   |-- train_detection.py
|   |   |-- train_segmentation.py
|   |   |-- predict_detection.py
|   |   |-- predict_segmentation.py
|   |-- results/
|       |-- detection/
|       |-- segmentation/
|
|-- .gitignore
|-- README.md

## How to Run the Application

There are two main components to this application: the **backend server** (which handles all the machine learning tasks) and the **frontend server** (which provides the user interface). You'll need to run both simultaneously.

### 1. Starting the Backend Server

The backend is a Python application built with FastAPI.

*   **Step 1: Install Dependencies**
    First, you need to install all the required Python packages. Open a terminal in the project's root directory and run:
    ```bash
    pip install -r backend/requirements.txt
    ```

*   **Step 2: Start the Server**
    Once the dependencies are installed, you can start the backend server using `uvicorn`. In the same terminal, run:
    ```bash
    uvicorn backend.app.main:app --reload
    ```
    This command tells `uvicorn` to run the `app` object from the `backend.app.main` module. The `--reload` flag will automatically restart the server whenever you make changes to the code.

The backend server will now be running at `http://127.0.0.1:8000`.

### 2. Starting the Frontend Server

The frontend is a React application.

*   **Step 1: Navigate to the Frontend Directory**
    Open a **new terminal** and navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

*   **Step 2: Install Dependencies**
    If you haven't already, install the necessary Node.js packages:
    ```bash
    npm install
    ```

*   **Step 3: Start the Development Server**
    Now, you can start the frontend development server:
    ```bash
    npm start
    ```
    This will automatically open a new tab in your web browser with the application running at `http://localhost:3000`.

## The Machine Learning Workflow

Here’s a step-by-step guide to the machine learning workflow that this application now supports:

1.  **Pre-annotation (New Feature):**
    *   **Purpose:** To speed up the process of labeling your images.
    *   **How it works:**
        1.  Go to the **"Annotate"** page in the web interface.
        2.  Upload an image you want to label.
        3.  The application will use the pre-trained `yolov8n.pt` model to automatically detect objects in the image and draw bounding boxes around them.
        4.  You can then download this pre-annotated image. The next step would be to import these annotations into a labeling tool like CVAT for review and refinement. This is much faster than drawing every bounding box from scratch.

2.  **Training:**
    *   **Purpose:** To train your own custom object detection or segmentation model using your labeled dataset.
    *   **How it works:**
        1.  Once you have a complete dataset (images and their corresponding labels in YOLO format), zip them into a single file. This zip file must contain a `data.yaml` file that describes the dataset structure.
        2.  Go to the **"Train"** page.
        3.  Upload your zipped dataset.
        4.  The backend will start a training process using your data.

3.  **Prediction:**
    *   **Purpose:** To use your newly trained model (or the default one) to make predictions on new images.
    *   **How it works:**
        1.  Go to the **"Predict"** page.
        2.  Upload an image.
        3.  The application will run the model and display the image with the detected objects or segmentation masks.

This workflow creates a powerful loop: you can quickly pre-annotate your data, use it to train a custom model, and then use that model for predictions, all within a single, integrated application.