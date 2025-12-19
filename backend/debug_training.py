#!/usr/bin/env python3
"""
Debug script to check why training models aren't being uploaded to GCS.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.core.database_service import database_service
from google.cloud import storage

def check_recent_training_jobs(limit=5):
    """Check recent training jobs and their status"""
    print("🔍 Checking recent training jobs...")
    
    result = database_service.list_training_jobs()
    if result['status'] != 'success':
        print(f"❌ Failed to get training jobs: {result['message']}")
        return
    
    jobs = result['training_jobs'][:limit]
    print(f"📊 Found {len(jobs)} recent jobs")
    
    for job in jobs:
        task_id = job['task_id']
        status = job['status']
        model_type = job['model_type']
        model_files = job.get('model_files_info', []) or []
        results_dir = job.get('results_dir', 'N/A')
        
        print(f"\n🔍 Job: {task_id}")
        print(f"   Status: {status}")
        print(f"   Model Type: {model_type}")
        print(f"   Results Dir: {results_dir}")
        print(f"   Model Files: {len(model_files)} files")
        
        if len(model_files) == 0:
            print(f"   ⚠️ No model files recorded!")
        else:
            for file_info in model_files[:3]:  # Show first 3 files
                print(f"   📄 {file_info.get('filename', 'Unknown')}")
        
        # Check if files exist in GCS
        check_gcs_files_for_job(task_id, model_type)

def check_gcs_files_for_job(task_id: str, model_type: str):
    """Check if files exist in GCS for a specific job"""
    try:
        storage_client = storage.Client()
        
        # Check both possible buckets and locations
        buckets_to_check = [
            ('mltraining-models', get_model_folder(model_type), task_id),
            ('mltraining-vertex-staging', get_model_folder(model_type), task_id),
            ('mltraining-models', 'models', task_id),
            ('mltraining-vertex-staging', 'models', task_id),
        ]
        
        files_found = False
        
        for bucket_name, folder1, folder2 in buckets_to_check:
            try:
                bucket = storage_client.bucket(bucket_name)
                prefix = f"{folder1}/{folder2}/"
                
                blobs = list(bucket.list_blobs(prefix=prefix))
                if blobs:
                    print(f"   ✅ Found {len(blobs)} files in gs://{bucket_name}/{prefix}")
                    files_found = True
                    for blob in blobs[:3]:  # Show first 3
                        print(f"      📄 {os.path.basename(blob.name)} ({blob.size} bytes)")
                    if len(blobs) > 3:
                        print(f"      ... and {len(blobs) - 3} more files")
                    break
                        
            except Exception as e:
                print(f"   ⚠️ Could not check gs://{bucket_name}/{prefix}: {e}")
        
        if not files_found:
            print(f"   ❌ No files found in any GCS location")
            
    except Exception as e:
        print(f"   ❌ Error checking GCS: {e}")

def get_model_folder(model_type: str) -> str:
    """Map model type to folder name"""
    mapping = {
        'anomaly': 'anomaly',
        'object_detection': 'detection',
        'segmentation': 'segmentation',
    }
    return mapping.get(model_type, 'other')

def check_vertex_job_logs(job_id: str):
    """Check Vertex AI job logs"""
    print(f"\n🔍 Checking Vertex AI job logs for {job_id}...")
    try:
        import subprocess
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'describe', job_id, 
            '--region=us-central1', '--format=json'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            job_data = json.loads(result.stdout)
            state = job_data.get('state', 'UNKNOWN')
            print(f"   Job State: {state}")
            
            if 'endTime' in job_data:
                print(f"   End Time: {job_data['endTime']}")
        else:
            print(f"   ❌ Could not get job info: {result.stderr}")
            
    except Exception as e:
        print(f"   ❌ Error checking job: {e}")

def main():
    print("🐛 TRAINING DEBUG TOOL")
    print("=" * 50)
    
    check_recent_training_jobs()
    
    # Get the latest detection training job for detailed check
    result = database_service.list_training_jobs()
    if result['status'] == 'success':
        detection_jobs = [job for job in result['training_jobs'] 
                         if job['model_type'] == 'object_detection']
        
        if detection_jobs:
            latest_job = detection_jobs[0]
            print(f"\n🎯 Detailed check for latest detection job: {latest_job['task_id']}")
            
            # Check for Vertex AI job ID
            config = latest_job.get('training_config', {})
            vertex_job_id = config.get('vertex_ai_job_id')
            if vertex_job_id:
                check_vertex_job_logs(vertex_job_id)

if __name__ == "__main__":
    main()
