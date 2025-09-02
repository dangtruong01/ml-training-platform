import subprocess
import os
import zipfile
import uuid
import cv2
import numpy as np
import json
import yaml
import shutil
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from fastapi import UploadFile
from ultralytics import YOLO

class YoloService:
    def __init__(self, scripts_dir: str = "ml/scripts", datasets_dir: str = "ml/datasets"):
        self.scripts_dir = os.path.abspath(scripts_dir)
        self.datasets_dir = os.path.abspath(datasets_dir)
        self.results_dir = os.path.abspath("ml/results")
        
        # Training progress tracking
        self.training_processes: Dict[str, dict] = {}
        self.training_logs: Dict[str, list] = {}
        
        # Ensure directories exist
        os.makedirs(self.scripts_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Try to use the best available model
        model_priority = ["yolov8l.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]
        
        for model_name in model_priority:
            model_path = os.path.join("ml", "models", model_name)
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Using model: {model_name}")
                break
        else:
            # Fallback to downloading nano model
            self.model = YOLO("yolov8n.pt")
            print("Using fallback nano model")

    async def handle_dataset_upload(self, file: UploadFile, task_type: str) -> str | None:
        """Saves and unzips an uploaded dataset, returning the path to data.yaml."""
        try:
            # Create a unique directory for this dataset
            dataset_id = str(uuid.uuid4())
            dataset_dir = os.path.join(self.datasets_dir, task_type, dataset_id)
            os.makedirs(dataset_dir, exist_ok=True)

            # Save the zip file
            zip_path = os.path.join(dataset_dir, file.filename)
            with open(zip_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            print(f"Saved dataset zip to: {zip_path}")

            # Extract the zip file
            extract_dir = os.path.join(dataset_dir, "extracted")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                print("❌ Invalid zip file")
                return None

            print(f"Extracted dataset to: {extract_dir}")

            # Clean up the zip file
            os.remove(zip_path)

            # Find data.yaml file
            data_yaml_path = self._find_data_yaml(extract_dir)
            if not data_yaml_path:
                print("No data.yaml found, attempting to find dataset structure...")
                # Try to find dataset in subdirectories
                for item in os.listdir(extract_dir):
                    item_path = os.path.join(extract_dir, item)
                    if os.path.isdir(item_path):
                        nested_yaml = self._find_data_yaml(item_path)
                        if nested_yaml:
                            data_yaml_path = nested_yaml
                            break

            if not data_yaml_path:
                print(f"❌ No data.yaml file found in dataset")
                return None

            print(f"✅ Found data.yaml at: {data_yaml_path}")

            # Validate dataset structure
            if self._validate_dataset_structure(data_yaml_path):
                print(f"✅ Dataset structure validated")
                return data_yaml_path
            else:
                print(f"❌ Invalid dataset structure")
                return None

        except Exception as e:
            print(f"Error handling dataset upload: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_data_yaml(self, directory: str) -> Optional[str]:
        """Recursively find data.yaml file in directory"""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower() in ['data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml']:
                    return os.path.join(root, file)
        return None

    def _validate_dataset_structure(self, data_yaml_path: str) -> bool:
        """Validate that the dataset has proper YOLO structure"""
        try:
            # Read data.yaml
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)

            print(f"Dataset config: {data_config}")

            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data_config:
                    print(f"Missing required field: {field}")
                    return False

            # Get the directory containing data.yaml
            dataset_root = os.path.dirname(data_yaml_path)

            # Check if train and val directories exist
            train_path = os.path.join(dataset_root, data_config['train'])
            val_path = os.path.join(dataset_root, data_config['val'])

            if not os.path.exists(train_path):
                print(f"Train directory not found: {train_path}")
                return False

            if not os.path.exists(val_path):
                print(f"Val directory not found: {val_path}")
                return False

            # Check if there are corresponding labels directories
            train_labels_path = train_path.replace('images', 'labels')
            val_labels_path = val_path.replace('images', 'labels')

            if not os.path.exists(train_labels_path):
                print(f"Train labels directory not found: {train_labels_path}")
                return False

            if not os.path.exists(val_labels_path):
                print(f"Val labels directory not found: {val_labels_path}")
                return False

            # Count files
            train_images = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            train_labels = len([f for f in os.listdir(train_labels_path) if f.endswith('.txt')])
            val_images = len([f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            val_labels = len([f for f in os.listdir(val_labels_path) if f.endswith('.txt')])

            print(f"Dataset statistics:")
            print(f"  Train: {train_images} images, {train_labels} labels")
            print(f"  Val: {val_images} images, {val_labels} labels")
            print(f"  Classes: {data_config['nc']} - {data_config['names']}")

            if train_images == 0:
                print("No training images found")
                return False

            if val_images == 0:
                print("No validation images found")
                return False

            return True

        except Exception as e:
            print(f"Error validating dataset: {e}")
            return False

    def train_detection(self, dataset_path: str, device: str = "cpu") -> str:
        """
        Train a YOLO detection model using the provided dataset
        Returns a task ID for tracking
        """
        try:
            task_id = f"detection_training_{uuid.uuid4().hex[:8]}"
            print(f"🚀 Starting detection training with task ID: {task_id}")

            # Validate the data.yaml path exists
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset configuration not found: {dataset_path}")

            # Create results directory for this training run
            training_results_dir = os.path.join(self.results_dir, "detection", task_id)
            os.makedirs(training_results_dir, exist_ok=True)

            # Use the train_detection.py script
            script_path = os.path.join(self.scripts_dir, "train_detection.py")

            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Training script not found: {script_path}")

            # Prepare training command
            training_command = [
                "python", script_path,
                "--data", dataset_path,
                "--epochs", "10",
                "--imgsz", "640",
                "--project", training_results_dir,
                "--name", "yolo_detection_model",
                "--device", device
            ]

            print(f"🔥 Training command: {' '.join(training_command)}")

            # Initialize tracking
            self.training_processes[task_id] = {
                'status': 'starting',
                'progress': 0,
                'current_epoch': 0,
                'total_epochs': 10,
                'dataset_path': dataset_path,
                'results_dir': training_results_dir,
                'started_at': datetime.now().isoformat(),
                'log_file': os.path.join(training_results_dir, 'training.log')
            }
            self.training_logs[task_id] = []

            # Start training in background thread with real-time monitoring
            training_thread = threading.Thread(
                target=self._run_training_with_monitoring,
                args=(task_id, training_command, script_path)
            )
            training_thread.daemon = True
            training_thread.start()

            print(f"✅ Detection training started successfully!")
            print(f"📁 Results will be saved to: {training_results_dir}")

            return task_id

        except Exception as e:
            error_msg = f"Failed to start detection training: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

    def train_segmentation(self, dataset_path: str, device: str = "cpu") -> str:
        """Similar to train_detection but for segmentation"""
        try:
            task_id = f"segmentation_training_{uuid.uuid4().hex[:8]}"
            print(f"🚀 Starting segmentation training with task ID: {task_id}")

            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset configuration not found: {dataset_path}")

            training_results_dir = os.path.join(self.results_dir, "segmentation", task_id)
            os.makedirs(training_results_dir, exist_ok=True)

            script_path = os.path.join(self.scripts_dir, "train_segmentation.py")

            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Training script not found: {script_path}")

            training_command = [
                "python", script_path,
                "--data", dataset_path,
                "--epochs", "10",
                "--imgsz", "640",
                "--project", training_results_dir,
                "--name", "yolo_segmentation_model",
                "--device", device
            ]

            print(f"🔥 Training command: {' '.join(training_command)}")

            # Initialize tracking
            self.training_processes[task_id] = {
                'status': 'starting',
                'progress': 0,
                'current_epoch': 0,
                'total_epochs': 10,
                'dataset_path': dataset_path,
                'results_dir': training_results_dir,
                'started_at': datetime.now().isoformat(),
                'log_file': os.path.join(training_results_dir, 'training.log')
            }
            self.training_logs[task_id] = []

            # Start training in background thread
            training_thread = threading.Thread(
                target=self._run_training_with_monitoring,
                args=(task_id, training_command, script_path)
            )
            training_thread.daemon = True
            training_thread.start()

            print(f"✅ Segmentation training started successfully!")
            print(f"📁 Results will be saved to: {training_results_dir}")

            return task_id

        except Exception as e:
            error_msg = f"Failed to start segmentation training: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

    def _run_training_with_monitoring(self, task_id: str, command: list, script_dir: str):
        """Run training with real-time output monitoring"""
        try:
            self.training_processes[task_id]['status'] = 'running'
            
            # Start the training process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=os.path.dirname(script_dir)
            )

            # Store process info
            self.training_processes[task_id]['pid'] = process.pid
            
            # Create log file
            log_file = self.training_processes[task_id]['log_file']
            
            print(f"📝 Training started - Task ID: {task_id}, PID: {process.pid}")
            print(f"📄 Log file: {log_file}")

            # Read output line by line in real-time
            with open(log_file, 'w') as log_f:
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    
                    # Remove newline and clean up
                    clean_line = line.rstrip()
                    
                    if clean_line:
                        # Write to log file
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"[{timestamp}] {clean_line}"
                        log_f.write(log_entry + '\n')
                        log_f.flush()
                        
                        # Store in memory for API access
                        self.training_logs[task_id].append({
                            'timestamp': timestamp,
                            'message': clean_line
                        })
                        
                        # Keep only last 100 log entries in memory
                        if len(self.training_logs[task_id]) > 100:
                            self.training_logs[task_id] = self.training_logs[task_id][-100:]
                        
                        # Print to console for real-time viewing
                        print(f"🔥 [{task_id}] {clean_line}")
                        
                        # Parse progress information
                        self._parse_training_progress(task_id, clean_line)

            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                self.training_processes[task_id]['status'] = 'completed'
                self.training_processes[task_id]['progress'] = 100
                print(f"✅ Training completed successfully - Task ID: {task_id}")
            else:
                self.training_processes[task_id]['status'] = 'failed'
                print(f"❌ Training failed - Task ID: {task_id}, Return code: {return_code}")

        except Exception as e:
            self.training_processes[task_id]['status'] = 'failed'
            self.training_processes[task_id]['error'] = str(e)
            print(f"❌ Training error - Task ID: {task_id}, Error: {e}")
            import traceback
            traceback.print_exc()

    def _parse_training_progress(self, task_id: str, log_line: str):
        """Parse training output to extract progress information"""
        try:
            # Look for epoch information in YOLO output
            if "Epoch" in log_line and "/" in log_line:
                # Example: "Epoch 1/10: 45%|████▌     | 45/100 [00:30<00:22,  2.45it/s]"
                parts = log_line.split()
                for i, part in enumerate(parts):
                    if "Epoch" in part and i + 1 < len(parts):
                        epoch_info = parts[i + 1]
                        if "/" in epoch_info:
                            try:
                                current, total = epoch_info.split("/")
                                current_epoch = int(current)
                                total_epochs = int(total.split(":")[0])  # Remove any trailing colon
                                
                                self.training_processes[task_id]['current_epoch'] = current_epoch
                                self.training_processes[task_id]['total_epochs'] = total_epochs
                                
                                # Calculate overall progress
                                progress = (current_epoch / total_epochs) * 100
                                self.training_processes[task_id]['progress'] = min(100, progress)
                                
                            except (ValueError, IndexError):
                                pass
                        break
            
            # Look for other progress indicators
            elif "%" in log_line and "|" in log_line:
                # Progress bar format: "45%|████▌     |"
                try:
                    percent_start = log_line.find("%")
                    if percent_start > 0:
                        # Look backwards for the percentage number
                        i = percent_start - 1
                        while i >= 0 and (log_line[i].isdigit() or log_line[i] == '.'):
                            i -= 1
                        percent_str = log_line[i+1:percent_start]
                        if percent_str:
                            current_progress = float(percent_str)
                            # Update epoch progress, not overall progress
                            self.training_processes[task_id]['epoch_progress'] = current_progress
                except (ValueError, IndexError):
                    pass

        except Exception as e:
            # Silently handle parsing errors
            pass

    def get_training_status(self, task_id: str) -> dict:
        """Get the current status of a training task"""
        if task_id not in self.training_processes:
            return {'error': 'Task not found'}
        
        status_info = self.training_processes[task_id].copy()
        
        # Add recent logs
        if task_id in self.training_logs:
            status_info['recent_logs'] = self.training_logs[task_id][-10:]  # Last 10 entries
        
        return status_info

    def get_training_logs(self, task_id: str, lines: int = 50) -> dict:
        """Get recent training logs"""
        if task_id not in self.training_processes:
            return {'error': 'Task not found'}
        
        # Try to read from log file if it exists
        log_file = self.training_processes[task_id].get('log_file')
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()
                    recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    return {
                        'task_id': task_id,
                        'logs': [line.strip() for line in recent_lines],
                        'total_lines': len(all_lines)
                    }
            except Exception as e:
                pass
        
        # Fallback to in-memory logs
        if task_id in self.training_logs:
            recent_logs = self.training_logs[task_id][-lines:]
            return {
                'task_id': task_id,
                'logs': [f"[{log['timestamp']}] {log['message']}" for log in recent_logs],
                'total_lines': len(self.training_logs[task_id])
            }
        
        return {'task_id': task_id, 'logs': [], 'total_lines': 0}

    def list_training_tasks(self) -> dict:
        """List all training tasks and their status"""
        return {
            'tasks': [
                {
                    'task_id': task_id,
                    'status': info['status'],
                    'progress': info.get('progress', 0),
                    'started_at': info.get('started_at', ''),
                    'current_epoch': info.get('current_epoch', 0),
                    'total_epochs': info.get('total_epochs', 0)
                }
                for task_id, info in self.training_processes.items()
            ]
        }

    def _store_training_process(self, task_id: str, process, dataset_path: str, results_dir: str):
        """Store training process information for tracking"""
        # This method is now handled by the monitoring thread
        pass

    # Keep all your existing annotation methods unchanged...
    def predict_detection(self, image_path: str, model_path: str = None) -> dict:
        """Run detection prediction with quality assessment"""
        try:
            # Load model
            if model_path and os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"Using custom model: {model_path}")
            else:
                model = self.model
                print(f"Using default model")
            
            # Run prediction
            results = model(image_path)
            
            # Process results for quality assessment
            return self._process_detection_results(results[0], image_path)
            
        except Exception as e:
            print(f"Detection prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_segmentation(self, image_path: str, model_path: str = None) -> dict:
        """Run segmentation prediction with quality assessment"""
        try:
            # Load segmentation model
            if model_path and os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"Using custom segmentation model: {model_path}")
            else:
                # Use default segmentation model
                seg_model_path = os.path.join("ml", "models", "yolov8n-seg.pt")
                if os.path.exists(seg_model_path):
                    model = YOLO(seg_model_path)
                else:
                    model = YOLO("yolov8n-seg.pt")  # Download if needed
                print(f"Using default segmentation model")
            
            # Run prediction
            results = model(image_path)
            
            # Process results for quality assessment
            return self._process_segmentation_results(results[0], image_path)
            
        except Exception as e:
            print(f"Segmentation prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def pre_annotate_detection(self, image_path: str) -> str:
        return self._opencv_detection_annotation(image_path)
    
    def pre_annotate_segmentation(self, image_path: str) -> str:
        return self._opencv_segmentation_annotation(image_path)
    
    def pre_annotate_sam2_detection(self, image_path: str) -> str:
        try:
            from backend.services.sam2_service import sam2_service
            return sam2_service.annotate_detection(image_path)
        except ImportError:
            raise RuntimeError("SAM2 service not available. Please install SAM2 dependencies.")
        except Exception as e:
            print(f"SAM2 detection failed: {e}. Falling back to OpenCV.")
            return self._opencv_detection_annotation(image_path)
    
    def pre_annotate_sam2_segmentation(self, image_path: str) -> str:
        try:
            from backend.services.sam2_service import sam2_service
            return sam2_service.annotate_segmentation(image_path)
        except ImportError:
            raise RuntimeError("SAM2 service not available. Please install SAM2 dependencies.")
        except Exception as e:
            print(f"SAM2 segmentation failed: {e}. Falling back to OpenCV.")
            return self._opencv_segmentation_annotation(image_path)

    # Keep all your existing OpenCV annotation methods...
    def _opencv_detection_annotation(self, image_path: str) -> str:
        """Enhanced OpenCV-based object detection with edge detection for metallic objects"""
        results_dir = os.path.abspath(os.path.join("ml", "results", "pre_annotation"))
        os.makedirs(results_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        annotated_img = img.copy()
        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width
        
        # Save original for debugging
        original_debug = os.path.join(results_dir, f"debug_original_{os.path.basename(image_path)}")
        cv2.imwrite(original_debug, img)
        
        # METHOD 1: Enhanced edge-based detection
        valid_contours = self._edge_based_detection(img, results_dir, image_path)
        
        # METHOD 2: If insufficient contours, try adaptive thresholding
        if len(valid_contours) < 1:
            valid_contours.extend(self._adaptive_threshold_detection(img, img_area, results_dir, image_path))
        
        # METHOD 3: If still insufficient, try color-based detection
        if len(valid_contours) < 1:
            valid_contours.extend(self._color_based_detection(img, img_area))
        
        # Remove duplicates and filter by area
        valid_contours = self._filter_and_merge_contours(valid_contours, img_area)
        
        # Draw bounding boxes (detection style)
        for i, c in enumerate(valid_contours[:5]):
            x, y, w, h = cv2.boundingRect(c)
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)][i % 5]
            
            # Draw bounding rectangle
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
            
            # Add area and confidence info
            area = cv2.contourArea(c)
            label = f"Object {i+1} ({int(area)}px)"
            cv2.putText(annotated_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add detection summary
        summary = f"Detection: {len(valid_contours)} objects found"
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        output_path = os.path.join(results_dir, f"detection_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        return output_path
    
    def _opencv_segmentation_annotation(self, image_path: str) -> str:
        """Enhanced OpenCV-based segmentation with edge detection for metallic objects"""
        results_dir = os.path.abspath(os.path.join("ml", "results", "pre_annotation"))
        os.makedirs(results_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        annotated_img = img.copy()
        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width
        
        # Use enhanced edge detection for segmentation too
        valid_contours = self._edge_based_detection(img, results_dir, image_path)
        
        # If insufficient contours, try adaptive thresholding
        if len(valid_contours) < 1:
            valid_contours.extend(self._adaptive_threshold_detection(img, img_area, results_dir, image_path))
        
        # If still insufficient, try color-based segmentation
        if len(valid_contours) < 1:
            valid_contours.extend(self._color_based_segmentation(img, img_area))
        
        # Filter and merge for segmentation
        valid_contours = self._filter_and_merge_contours(valid_contours, img_area)
        
        # Draw detailed contours (segmentation style) with more precise contours
        for i, c in enumerate(valid_contours[:3]):
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i % 3]
            
            # For segmentation, we want more precise contours
            # Approximate contour to reduce noise while preserving shape
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx_contour = cv2.approxPolyDP(c, epsilon, True)
            
            # Draw filled contour with transparency
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [approx_contour], 255)
            
            # Create colored overlay
            colored_mask = np.zeros_like(img)
            colored_mask[mask == 255] = color
            annotated_img = cv2.addWeighted(annotated_img, 0.7, colored_mask, 0.3, 0)
            
            # Draw precise contour outline
            cv2.drawContours(annotated_img, [approx_contour], -1, color, 2)
            
            # Add label with area information
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            label = f"Segment {i+1} ({int(area)}px)"
            cv2.putText(annotated_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add segmentation summary
        summary = f"Segmentation: {len(valid_contours)} segments found"
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        output_path = os.path.join(results_dir, f"segmentation_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        return output_path
    
    def _color_based_detection(self, img, img_area):
        """Helper method for color-based object detection"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-blue areas (metallic objects typically not blue)
        lower_blue = np.array([90, 70, 70])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        non_blue_mask = cv2.bitwise_not(blue_mask)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(non_blue_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if 1000 < cv2.contourArea(c) < 0.8 * img_area]
    
    def _color_based_segmentation(self, img, img_area):
        """Helper method for color-based segmentation"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # More refined color segmentation
        lower_blue = np.array([90, 70, 70])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        non_blue_mask = cv2.bitwise_not(blue_mask)
        
        # Finer morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(non_blue_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours with better approximation
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return [c for c in contours if 1000 < cv2.contourArea(c) < 0.8 * img_area]
    
    def _edge_based_detection(self, img, results_dir, image_path):
        """Enhanced edge-based detection for metallic objects with varying backgrounds"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle different lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Save enhanced image for debugging
        enhanced_debug = os.path.join(results_dir, f"debug_enhanced_{os.path.basename(image_path)}")
        cv2.imwrite(enhanced_debug, enhanced)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Use Canny edge detection with adaptive thresholds
        # Calculate dynamic thresholds based on image statistics
        v = np.median(bilateral)
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        
        edges = cv2.Canny(bilateral, lower_thresh, upper_thresh)
        
        # Save edges for debugging
        edges_debug = os.path.join(results_dir, f"debug_edges_{os.path.basename(image_path)}")
        cv2.imwrite(edges_debug, edges)
        
        # Dilate edges to connect nearby edge pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Close gaps in edges
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close)
        
        # Save processed edges for debugging
        edges_processed_debug = os.path.join(results_dir, f"debug_edges_processed_{os.path.basename(image_path)}")
        cv2.imwrite(edges_processed_debug, edges_closed)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = img.shape[0] * img.shape[1]
        valid_contours = []
        
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            # Filter by area (not too small, not too large)
            if area < 1500 or area > 0.7 * img_area:
                continue
                
            # Filter by aspect ratio and solidity for metallic objects
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Metallic objects usually have reasonable aspect ratios and decent solidity
            if 0.1 < aspect_ratio < 10 and solidity > 0.3:
                valid_contours.append(c)
        
        return valid_contours
    
    def _adaptive_threshold_detection(self, img, img_area, results_dir, image_path):
        """Adaptive thresholding for different lighting conditions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Use adaptive threshold to handle varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Save adaptive threshold for debugging
        adaptive_debug = os.path.join(results_dir, f"debug_adaptive_{os.path.basename(image_path)}")
        cv2.imwrite(adaptive_debug, adaptive_thresh)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if 1200 < area < 0.6 * img_area:
                # Additional filtering for shape characteristics
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                if 0.1 < aspect_ratio < 8:  # Reasonable aspect ratio
                    valid_contours.append(c)
        
        return valid_contours
    
    def _filter_and_merge_contours(self, contours, img_area):
        """Filter and merge overlapping contours"""
        if not contours:
            return []
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        merged_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip if too small or too large
            if area < 800 or area > 0.8 * img_area:
                continue
            
            # Check if this contour overlaps significantly with existing ones
            is_duplicate = False
            current_rect = cv2.boundingRect(contour)
            
            for existing in merged_contours:
                existing_rect = cv2.boundingRect(existing)
                
                # Calculate intersection over union (IoU)
                iou = self._calculate_bbox_iou(current_rect, existing_rect)
                if iou > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_contours.append(contour)
                
                # Limit to avoid too many detections
                if len(merged_contours) >= 5:
                    break
        
        return merged_contours
    
    def _calculate_bbox_iou(self, box1, box2):
        """Calculate Intersection over Union for bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _process_detection_results(self, result, image_path: str) -> dict:
        """Process YOLO detection results and assess quality"""
        import cv2
        import base64
        from pathlib import Path
        
        # Load original image
        img = cv2.imread(image_path)
        annotated_img = img.copy()
        
        detections = []
        total_defects = 0
        confidence_scores = []
        
        if result.boxes is not None:
            for box in result.boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(annotated_img, f"{class_name}: {confidence:.2f}", 
                           (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                detections.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
                
                # Count defects (assuming scratch/dent classes indicate defects)
                if class_name.lower() in ['scratch', 'dent', 'defect', 'damage']:
                    total_defects += 1
                    confidence_scores.append(confidence)
        
        # Quality assessment logic
        quality_status = self._assess_quality_from_detections(total_defects, confidence_scores)
        
        # Save annotated image
        results_dir = os.path.join("ml", "results", "predictions")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"prediction_{Path(image_path).stem}.jpg")
        cv2.imwrite(output_path, annotated_img)
        
        # Convert image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "quality": quality_status,
            "detections": detections,
            "total_defects": total_defects,
            "image_base64": img_base64,
            "output_path": output_path
        }

    def _process_segmentation_results(self, result, image_path: str) -> dict:
        """Process YOLO segmentation results and assess quality"""
        import cv2
        import base64
        import numpy as np
        from pathlib import Path
        
        # Load original image
        img = cv2.imread(image_path)
        annotated_img = img.copy()
        
        detections = []
        total_defects = 0
        confidence_scores = []
        total_defect_area = 0
        
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            
            for i, mask in enumerate(masks):
                if boxes is not None:
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    
                    # Resize mask to image size
                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Calculate defect area
                    defect_area = np.sum(mask_binary)
                    
                    # Create colored overlay
                    color = (0, 0, 255) if class_name.lower() in ['scratch', 'dent', 'defect'] else (0, 255, 0)
                    annotated_img[mask_binary == 1] = annotated_img[mask_binary == 1] * 0.6 + np.array(color) * 0.4
                    
                    detections.append({
                        "class": class_name,
                        "confidence": float(confidence),
                        "area": int(defect_area)
                    })
                    
                    # Count defects
                    if class_name.lower() in ['scratch', 'dent', 'defect', 'damage']:
                        total_defects += 1
                        confidence_scores.append(confidence)
                        total_defect_area += defect_area
        
        # Quality assessment for segmentation
        quality_status = self._assess_quality_from_segmentation(total_defects, confidence_scores, total_defect_area, img.shape[0] * img.shape[1])
        
        # Save result
        results_dir = os.path.join("ml", "results", "predictions")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"segmentation_{Path(image_path).stem}.jpg")
        cv2.imwrite(output_path, annotated_img)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "quality": quality_status,
            "detections": detections,
            "total_defects": total_defects,
            "total_defect_area": total_defect_area,
            "image_base64": img_base64,
            "output_path": output_path
        }

    def _assess_quality_from_detections(self, total_defects: int, confidence_scores: list) -> dict:
        """Assess quality based on detection results"""
        # Quality rules for detection
        if total_defects == 0:
            return {
                "status": "OK",
                "confidence": 0.95,
                "reason": "No defects detected"
            }
        elif total_defects <= 2 and all(conf < 0.6 for conf in confidence_scores):
            return {
                "status": "OK",
                "confidence": 0.75,
                "reason": f"Minor defects detected ({total_defects}) with low confidence"
            }
        else:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            return {
                "status": "NG",
                "confidence": float(avg_confidence),
                "reason": f"Significant defects detected ({total_defects})"
            }

    def _assess_quality_from_segmentation(self, total_defects: int, confidence_scores: list, total_area: int, image_area: int) -> dict:
        """Assess quality based on segmentation results"""
        area_percentage = (total_area / image_area) * 100 if image_area > 0 else 0
        
        if total_defects == 0:
            return {
                "status": "OK",
                "confidence": 0.95,
                "reason": "No defects detected"
            }
        elif area_percentage < 1.0 and all(conf < 0.7 for conf in confidence_scores):
            return {
                "status": "OK",
                "confidence": 0.70,
                "reason": f"Minor defects (<1% area)"
            }
        else:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            return {
                "status": "NG",
                "confidence": float(avg_confidence),
                "reason": f"Significant defects ({area_percentage:.1f}% area)"
            }

    async def upload_model(self, file, model_type: str) -> dict:
        """Upload and save a trained model"""
        models_dir = os.path.join("ml", "models", "uploaded")
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate unique filename
        import uuid
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{model_type}_{uuid.uuid4().hex[:8]}.{file_extension}"
        model_path = os.path.join(models_dir, unique_filename)
        
        # Save file
        with open(model_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Test load the model to validate
        try:
            test_model = YOLO(model_path)
            model_info = {
                "status": "success",
                "model_path": model_path,
                "model_type": model_type,
                "filename": unique_filename,
                "original_name": file.filename,
                "classes": test_model.names if hasattr(test_model, 'names') else None
            }
            print(f"✅ Model uploaded successfully: {model_path}")
            return model_info
        except Exception as e:
            # Remove invalid model file
            if os.path.exists(model_path):
                os.remove(model_path)
            raise RuntimeError(f"Invalid model file: {e}")

    def list_available_models(self) -> dict:
        """List all available models"""
        models = {
            "pretrained": [],
            "trained": [],
            "uploaded": []
        }
        
        # Pretrained models
        pretrained_dir = os.path.join("ml", "models")
        if os.path.exists(pretrained_dir):
            for file in os.listdir(pretrained_dir):
                if file.endswith('.pt'):
                    models["pretrained"].append({
                        "name": file,
                        "path": os.path.join(pretrained_dir, file),
                        "type": "segmentation" if "seg" in file else "detection"
                    })
        
        # Trained models from training results
        results_dir = os.path.join("ml", "results", "detection")
        if os.path.exists(results_dir):
            for training_folder in os.listdir(results_dir):
                weights_dir = os.path.join(results_dir, training_folder, "yolo_detection_model", "weights")
                if os.path.exists(weights_dir):
                    best_path = os.path.join(weights_dir, "best.pt")
                    if os.path.exists(best_path):
                        models["trained"].append({
                            "name": f"{training_folder}_best.pt",
                            "path": best_path,
                            "type": "detection",
                            "training_id": training_folder
                        })
        
        # Uploaded models
        uploaded_dir = os.path.join("ml", "models", "uploaded")
        if os.path.exists(uploaded_dir):
            for file in os.listdir(uploaded_dir):
                if file.endswith('.pt'):
                    models["uploaded"].append({
                        "name": file,
                        "path": os.path.join(uploaded_dir, file),
                        "type": "detection" if "detection" in file else "segmentation"
                    })
        
        return models

    async def predict_batch(self, files, model_path: str = None, task_type: str = "detection"):
        """Predict on multiple images"""
        results = []
        temp_files = []
        
        try:
            for file in files:
                # Save temp file
                temp_path = f"temp_batch_{file.filename}"
                temp_files.append(temp_path)
                
                with open(temp_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Run prediction
                if task_type == "detection":
                    result = self.predict_detection(temp_path, model_path)
                else:
                    result = self.predict_segmentation(temp_path, model_path)
                
                result["filename"] = file.filename
                results.append(result)
        
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return {
            "status": "success",
            "total_images": len(files),
            "results": results,
            "summary": self._generate_batch_summary(results)
        }

    def assess_quality(self, image_path: str, model_path: str = None) -> dict:
        """Standalone quality assessment"""
        result = self.predict_detection(image_path, model_path)
        return {
            "status": "success",
            "filename": os.path.basename(image_path),
            "quality": result["quality"],
            "defects_found": result["total_defects"],
            "detections": result["detections"]
        }

    def _generate_batch_summary(self, results: list) -> dict:
        """Generate summary statistics for batch prediction"""
        total = len(results)
        ok_count = sum(1 for r in results if r["quality"]["status"] == "OK")
        ng_count = total - ok_count
        
        return {
            "total_images": total,
            "ok_count": ok_count,
            "ng_count": ng_count,
            "ok_percentage": (ok_count / total * 100) if total > 0 else 0,
            "ng_percentage": (ng_count / total * 100) if total > 0 else 0
        }

# Create global instance
yolo_service = YoloService()