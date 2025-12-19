import os
import threading
import queue
import time
from datetime import datetime
from typing import Dict, List, Optional


class TrainingMonitor:
    """Centralized training progress monitoring and logging"""
    
    def __init__(self):
        self.training_processes: Dict[str, dict] = {}
        self.training_logs: Dict[str, list] = {}
    
    def create_task(self, task_id: str, project_id: str = None, training_config: dict = None) -> None:
        """Create a new training task entry"""
        self.training_processes[task_id] = {
            'task_id': task_id,
            'status': 'pending',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'project_id': project_id,
            'training_config': training_config,
        }
        self.training_logs[task_id] = []
    
    def update_task_status(self, task_id: str, status: str, error: str = None) -> None:
        """Update task status"""
        if task_id in self.training_processes:
            self.training_processes[task_id]['status'] = status
            if error:
                self.training_processes[task_id]['error'] = error
            if status == 'completed':
                self.training_processes[task_id]['completed_at'] = datetime.now().isoformat()
                self.training_processes[task_id]['progress'] = 100
    
    def update_progress(self, task_id: str, progress: float, current_epoch: int = None, total_epochs: int = None) -> None:
        """Update training progress"""
        if task_id in self.training_processes:
            self.training_processes[task_id]['progress'] = min(100, progress)
            if current_epoch is not None:
                self.training_processes[task_id]['current_epoch'] = current_epoch
            if total_epochs is not None:
                self.training_processes[task_id]['total_epochs'] = total_epochs
    
    def add_log(self, task_id: str, message: str, log_type: str = 'info') -> None:
        """Add a log message to the task"""
        if task_id not in self.training_logs:
            self.training_logs[task_id] = []
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'type': log_type
        }
        
        self.training_logs[task_id].append(log_entry)
        
        # Keep only last 100 log entries to prevent memory issues
        if len(self.training_logs[task_id]) > 100:
            self.training_logs[task_id] = self.training_logs[task_id][-100:]
    
    def get_task_status(self, task_id: str) -> dict:
        """Get current status of a training task"""
        if task_id not in self.training_processes:
            return {'status': 'not_found', 'error': 'Task not found'}
        
        status_info = self.training_processes[task_id].copy()
        
        # Add recent logs
        if task_id in self.training_logs:
            status_info['recent_logs'] = self.training_logs[task_id][-10:]  # Last 10 entries
        
        return status_info
    
    def get_task_logs(self, task_id: str, lines: int = 50) -> dict:
        """Get logs for a specific task"""
        if task_id not in self.training_processes:
            return {'status': 'not_found', 'error': 'Task not found'}
        
        log_file = self.training_processes[task_id].get('log_file')
        file_logs = []
        
        # Try to read from log file if available
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    file_logs = f.readlines()[-lines:]
            except Exception as e:
                print(f"Error reading log file: {e}")
        
        # Get in-memory logs
        memory_logs = []
        if task_id in self.training_logs:
            recent_logs = self.training_logs[task_id][-lines:]
            memory_logs = [log['message'] if isinstance(log, dict) else str(log) for log in recent_logs]
        
        return {
            'task_id': task_id,
            'logs': file_logs + memory_logs,
            'total_lines': len(self.training_logs.get(task_id, []))
        }
    
    def list_tasks(self) -> dict:
        """List all training tasks"""
        return {
            'status': 'success',
            'tasks': [
                {
                    **info,
                    'recent_logs': self.training_logs.get(task_id, [])[-3:] if task_id in self.training_logs else []
                }
                for task_id, info in self.training_processes.items()
            ]
        }
    
    def delete_task(self, task_id: str) -> dict:
        """Delete a training task and its associated data"""
        if task_id not in self.training_processes:
            return {'status': 'error', 'error': 'Task not found'}
        
        # Remove from tracking
        del self.training_processes[task_id]
        if task_id in self.training_logs:
            del self.training_logs[task_id]
        
        return {'status': 'success', 'message': f'Task {task_id} deleted successfully'}
    
    def store_process_info(self, task_id: str, process, dataset_path: str, results_dir: str, log_file: str = None) -> None:
        """Store additional process information"""
        if task_id in self.training_processes:
            self.training_processes[task_id].update({
                'pid': process.pid if process else None,
                'dataset_path': dataset_path,
                'results_dir': results_dir,
                'log_file': log_file
            })
    
    def parse_training_progress(self, task_id: str, log_line: str) -> None:
        """Parse training progress from log line (YOLO-specific)"""
        if task_id not in self.training_processes:
            return
            
        try:
            # Parse YOLO training output
            if "Epoch" in log_line and "/" in log_line:
                parts = log_line.strip().split()
                for i, part in enumerate(parts):
                    if part == "Epoch" and i + 1 < len(parts):
                        epoch_info = parts[i + 1]
                        if "/" in epoch_info:
                            current_epoch, total_epochs = map(int, epoch_info.split("/"))
                            self.training_processes[task_id]['current_epoch'] = current_epoch
                            self.training_processes[task_id]['total_epochs'] = total_epochs
                            
                            # Calculate progress
                            if total_epochs > 0:
                                progress = (current_epoch / total_epochs) * 100
                                self.training_processes[task_id]['progress'] = min(100, progress)
                        break
                        
            # Parse batch progress within epoch
            if "%" in log_line and ("loss" in log_line.lower() or "val" in log_line.lower()):
                import re
                progress_match = re.search(r'(\d+)%', log_line)
                if progress_match:
                    batch_progress = int(progress_match.group(1))
                    current_epoch = self.training_processes[task_id].get('current_epoch', 0)
                    total_epochs = self.training_processes[task_id].get('total_epochs', 1)
                    
                    # Calculate overall progress including batch progress
                    if total_epochs > 0:
                        epoch_progress = (current_epoch - 1) / total_epochs * 100
                        current_progress = epoch_progress + (batch_progress / total_epochs)
                        self.training_processes[task_id]['epoch_progress'] = current_progress
                        
        except Exception as e:
            # Don't fail on parsing errors, just skip
            pass


# Global instance to be used across services
training_monitor = TrainingMonitor()