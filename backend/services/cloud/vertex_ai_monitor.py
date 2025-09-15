"""
Vertex AI job monitoring service.
Monitors running Vertex AI jobs and updates database with progress.
"""
import time
import threading
from typing import Dict, Any, List
from datetime import datetime, timedelta

from services.core.database_service import database_service
from services.cloud.vertex_ai_service import vertex_ai_service

class VertexAIMonitor:
    """Monitor Vertex AI training jobs and update database"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.check_interval = 30  # Check every 30 seconds
        
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ Started Vertex AI monitoring service")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⏹️ Stopped Vertex AI monitoring service")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._check_running_jobs()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(self.check_interval)  # Continue monitoring even if there's an error
    
    def _check_running_jobs(self):
        """Check status of all running Vertex AI jobs"""
        try:
            print(f"🔍 [MONITOR] Starting job check cycle...")
            
            # Get all jobs that might be running on Vertex AI
            running_jobs = database_service.list_training_jobs(status='running')
            submitted_jobs = database_service.list_training_jobs(status='submitted')
            pending_jobs = database_service.list_training_jobs(status='pending')
            
            print(f"🔍 [MONITOR] Found jobs - Running: {len(running_jobs.get('training_jobs', []))}, Submitted: {len(submitted_jobs.get('training_jobs', []))}, Pending: {len(pending_jobs.get('training_jobs', []))}")
            
            if running_jobs['status'] != 'success' or submitted_jobs['status'] != 'success':
                print(f"❌ [MONITOR] Failed to fetch jobs from database")
                return
            
            # Include pending jobs too (they might have been submitted to Vertex AI)
            jobs_to_check = running_jobs.get('training_jobs', []) + submitted_jobs.get('training_jobs', []) + pending_jobs.get('training_jobs', [])
            
            print(f"🔍 [MONITOR] Total jobs to check: {len(jobs_to_check)}")
            
            for job in jobs_to_check:
                task_id = job['task_id']
                current_status = job.get('status')
                print(f"🔍 [MONITOR] Checking job: {task_id} (status: {current_status})")
                
                # Get Vertex AI job ID
                vertex_ai_job_id = None
                if job.get('training_config') and isinstance(job['training_config'], dict):
                    vertex_ai_job_id = job['training_config'].get('vertex_ai_job_id')
                
                if not vertex_ai_job_id:
                    print(f"⚠️ [MONITOR] Job {task_id} has no Vertex AI job ID - skipping")
                    continue
                
                print(f"🔍 [MONITOR] Job {task_id} has Vertex AI job ID: {vertex_ai_job_id}")
                
                # STAGED MONITORING LOGIC
                if current_status == 'pending':
                    # For pending jobs: check if they started running in Vertex AI
                    self._check_pending_job(task_id, vertex_ai_job_id)
                    
                elif current_status == 'running':
                    # For running jobs: check if they completed
                    # First check completion metadata (faster)
                    if self._check_completion_metadata(task_id):
                        print(f"✅ [MONITOR] Running job {task_id} completed via metadata")
                        continue
                    
                    # Then check Vertex AI status
                    self._check_running_job(task_id, vertex_ai_job_id)
                    
                else:
                    # For other statuses (submitted), treat as pending
                    print(f"🔍 [MONITOR] Job {task_id} status '{current_status}' treated as pending")
                    self._check_pending_job(task_id, vertex_ai_job_id)
                
        except Exception as e:
            print(f"❌ [MONITOR] Error checking running jobs: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_completion_metadata(self, task_id: str) -> bool:
        """Check for completion metadata from training container in Cloud Storage"""
        try:
            from google.cloud import storage
            import json
            
            print(f"🔍 [METADATA] Checking completion metadata for {task_id}")
            
            storage_client = storage.Client()
            bucket_name = 'mltraining-vertex-staging'
            
            # Check for completion metadata file
            bucket = storage_client.bucket(bucket_name)
            metadata_blob = bucket.blob(f"job-completions/{task_id}.json")
            
            if metadata_blob.exists():
                print(f"📋 [METADATA] Found completion metadata file for {task_id}")
                
                # Download and parse metadata
                metadata_content = metadata_blob.download_as_text()
                completion_data = json.loads(metadata_content)
                
                print(f"📋 [METADATA] Completion data for {task_id}: {completion_data['status']}")
                
                # Update database based on metadata
                if completion_data['status'] == 'completed':
                    result = database_service.complete_training_job(
                        task_id=task_id,
                        results_dir="gs://mltraining-models/",
                        model_files_info=completion_data.get('model_files_info', []),
                        training_metrics=completion_data.get('training_metrics', {})
                    )
                elif completion_data['status'] == 'failed':
                    result = database_service.update_training_job_status(
                        task_id=task_id,
                        status='failed',
                        error_message=completion_data.get('error_message')
                    )
                else:
                    result = database_service.update_training_job_status(
                        task_id=task_id,
                        status=completion_data['status'],
                        progress=completion_data.get('progress')
                    )
                
                if result['status'] == 'success':
                    print(f"✅ [METADATA] Updated {task_id} from completion metadata")
                    
                    # Delete the metadata file after processing
                    metadata_blob.delete()
                    print(f"🗑️ [METADATA] Cleaned up metadata file for {task_id}")
                    
                    return True
                else:
                    print(f"❌ [METADATA] Failed to update job from metadata: {result['message']}")
            else:
                print(f"⚠️ [METADATA] No completion metadata file found for {task_id}")
            
            return False
            
        except Exception as e:
            print(f"❌ [METADATA] Error checking completion metadata for {task_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_pending_job(self, task_id: str, vertex_ai_job_id: str):
        """Check if a pending job has started running on Vertex AI"""
        try:
            print(f"🔍 [PENDING] Checking if pending job {task_id} started running")
            
            # Get job status from Vertex AI
            status_result = vertex_ai_service.get_job_status(vertex_ai_job_id)
            
            if status_result['status'] != 'success':
                print(f"⚠️ [PENDING] Could not get Vertex AI status for {task_id}: {status_result.get('message')}")
                return
            
            vertex_ai_state = status_result['vertex_ai_state']
            internal_status = status_result['internal_status']
            
            print(f"🔍 [PENDING] Vertex AI job {vertex_ai_job_id} state: {vertex_ai_state} -> internal: {internal_status}")
            
            # If job is now running, update to running status
            if internal_status == 'running':
                print(f"📊 [PENDING] Job {task_id}: pending → running")
                
                result = database_service.update_training_job_status(
                    task_id=task_id,
                    status='running',
                    progress=5.0  # Small initial progress to show it started
                )
                
                if result['status'] == 'success':
                    print(f"✅ [PENDING] Successfully updated {task_id} to running")
                else:
                    print(f"❌ [PENDING] Failed to update {task_id}: {result.get('message')}")
                    
            elif internal_status == 'completed':
                # Job completed very quickly, jump directly to completed
                print(f"📊 [PENDING] Job {task_id}: pending → completed (fast job)")
                self._handle_quick_completion(task_id, vertex_ai_job_id)
                
            else:
                print(f"⚠️ [PENDING] Job {task_id} still in state: {internal_status}")
                
        except Exception as e:
            print(f"❌ [PENDING] Error checking pending job {task_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_running_job(self, task_id: str, vertex_ai_job_id: str):
        """Check progress and completion status for a running job"""
        try:
            print(f"🔍 [RUNNING] Checking progress and status for running job {task_id}")
            
            # First, get current job state from database to check if training container updated progress
            job_result = database_service.get_training_job(task_id)
            if job_result['status'] == 'success':
                current_job = job_result['training_job']
                current_progress = current_job.get('progress', 0)
                current_epoch = current_job.get('current_epoch', 0)
                total_epochs = current_job.get('total_epochs', 0)
                
                print(f"📊 [RUNNING] Job {task_id} progress: {current_progress}% (epoch {current_epoch}/{total_epochs})")
                
                # If the training container has updated progress significantly, job is actively running
                if current_progress > 0 and current_progress < 100:
                    print(f"🔄 [RUNNING] Job {task_id} actively training with {current_progress}% progress")
                    # Don't need to do anything - training container is updating progress automatically
                    return
                
                # If progress is 100% from training container, it completed through database update
                if current_progress >= 100:
                    print(f"✅ [RUNNING] Job {task_id} completed via training container (100% progress)")
                    return
            
            # If no progress updates from training container, check Vertex AI status for completion
            print(f"🔍 [RUNNING] Checking Vertex AI status for {task_id}")
            status_result = vertex_ai_service.get_job_status(vertex_ai_job_id)
            
            if status_result['status'] != 'success':
                print(f"⚠️ [RUNNING] Could not get Vertex AI status for {task_id}: {status_result.get('message')}")
                return
            
            vertex_ai_state = status_result['vertex_ai_state']
            internal_status = status_result['internal_status']
            error_message = status_result.get('error')
            
            print(f"🔍 [RUNNING] Vertex AI job {vertex_ai_job_id} state: {vertex_ai_state} -> internal: {internal_status}")
            
            # If job completed on Vertex AI, update to completed status
            if internal_status == 'completed':
                print(f"📊 [RUNNING] Job {task_id}: running → completed (via Vertex AI)")
                
                result = database_service.update_training_job_status(
                    task_id=task_id,
                    status='completed',
                    progress=100.0
                )
                
                if result['status'] == 'success':
                    print(f"✅ [RUNNING] Successfully updated {task_id} to completed")
                    self._handle_job_completion(task_id, {})
                else:
                    print(f"❌ [RUNNING] Failed to update {task_id}: {result.get('message')}")
                    
            elif internal_status == 'failed':
                # Job failed, update status
                print(f"📊 [RUNNING] Job {task_id}: running → failed")
                
                result = database_service.update_training_job_status(
                    task_id=task_id,
                    status='failed',
                    error_message=error_message or 'Training failed on Vertex AI'
                )
                
                if result['status'] == 'success':
                    print(f"✅ [RUNNING] Successfully updated {task_id} to failed")
                    self._handle_job_failure(task_id, {}, error_message)
                else:
                    print(f"❌ [RUNNING] Failed to update {task_id}: {result.get('message')}")
                    
            else:
                print(f"⚠️ [RUNNING] Job {task_id} still running in state: {internal_status}")
                
        except Exception as e:
            print(f"❌ [RUNNING] Error checking running job {task_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_quick_completion(self, task_id: str, vertex_ai_job_id: str):
        """Handle jobs that complete very quickly (pending → completed)"""
        try:
            print(f"🚀 [QUICK] Handling quick completion for {task_id}")
            
            result = database_service.update_training_job_status(
                task_id=task_id,
                status='completed',
                progress=100.0
            )
            
            if result['status'] == 'success':
                print(f"✅ [QUICK] Successfully updated {task_id} to completed")
                self._handle_job_completion(task_id, {})
            else:
                print(f"❌ [QUICK] Failed to update {task_id}: {result.get('message')}")
                
        except Exception as e:
            print(f"❌ [QUICK] Error handling quick completion for {task_id}: {e}")
    
    def _update_job_status(self, task_id: str, vertex_ai_job_id: str):
        """Update status of a specific job"""
        try:
            print(f"🔍 [VERTEX] Getting status for Vertex AI job {vertex_ai_job_id} (task: {task_id})")
            
            # Get job status from Vertex AI
            status_result = vertex_ai_service.get_job_status(vertex_ai_job_id)
            
            if status_result['status'] != 'success':
                print(f"⚠️ [VERTEX] Could not get status for job {task_id}: {status_result.get('message')}")
                return
            
            vertex_ai_state = status_result['vertex_ai_state']
            internal_status = status_result['internal_status']
            error_message = status_result.get('error')
            
            print(f"🔍 [VERTEX] Vertex AI job {vertex_ai_job_id} state: {vertex_ai_state} -> internal status: {internal_status}")
            
            # Get current job from database
            job_result = database_service.get_training_job(task_id)
            if job_result['status'] != 'success':
                print(f"⚠️ [VERTEX] Could not get job {task_id} from database")
                return
            
            current_job = job_result['training_job']
            current_status = current_job.get('status')
            
            print(f"🔍 [VERTEX] Job {task_id} current status: {current_status}")
            
            # Only update if status has changed
            if current_status != internal_status:
                print(f"📊 [VERTEX] Job {task_id}: {current_status} → {internal_status}")
                
                # Prepare update parameters
                update_params = {
                    'task_id': task_id,
                    'status': internal_status
                }
                
                # Add error message if job failed
                if error_message:
                    update_params['error_message'] = error_message
                
                # Set progress based on status
                if internal_status == 'running':
                    # For running jobs, we can't get real progress from Vertex AI easily
                    # The training container updates progress directly to database
                    pass
                elif internal_status == 'completed':
                    update_params['progress'] = 100.0
                elif internal_status == 'failed':
                    # Keep current progress for failed jobs
                    pass
                
                print(f"🔍 [VERTEX] Updating database with params: {update_params}")
                
                # Update database
                result = database_service.update_training_job_status(**update_params)
                
                if result['status'] == 'success':
                    print(f"✅ [VERTEX] Successfully updated job {task_id} status to {internal_status}")
                    
                    # If job completed, handle post-completion tasks
                    if internal_status == 'completed':
                        self._handle_job_completion(task_id, current_job)
                    elif internal_status == 'failed':
                        self._handle_job_failure(task_id, current_job, error_message)
                else:
                    print(f"❌ [VERTEX] Failed to update job {task_id}: {result.get('message')}")
            else:
                print(f"⚠️ [VERTEX] Job {task_id} status unchanged: {current_status}")
                
        except Exception as e:
            print(f"❌ [VERTEX] Error updating job status for {task_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_job_completion(self, task_id: str, job_info: Dict[str, Any]):
        """Handle job completion tasks"""
        try:
            print(f"🎉 Vertex AI job completed: {task_id}")
            
            # The training container should have already updated the database
            # with model file information via complete_training_job()
            # But we can add additional post-processing here if needed
            
            # Log completion
            completion_msg = f"✅ Vertex AI training completed for task {task_id}"
            print(completion_msg)
            
        except Exception as e:
            print(f"❌ Error handling job completion for {task_id}: {e}")
    
    def _handle_job_failure(self, task_id: str, job_info: Dict[str, Any], error_message: str):
        """Handle job failure tasks"""
        try:
            print(f"💥 Vertex AI job failed: {task_id}")
            print(f"   Error: {error_message}")
            
            # Could add cleanup tasks here
            # Could send notifications to users
            # Could attempt retries in some cases
            
        except Exception as e:
            print(f"❌ Error handling job failure for {task_id}: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring': self.monitoring,
            'check_interval': self.check_interval,
            'thread_active': self.monitor_thread.is_alive() if self.monitor_thread else False
        }

# Global monitor instance
vertex_ai_monitor = VertexAIMonitor()