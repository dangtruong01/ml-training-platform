import os
import asyncio
from typing import Dict, List, Any
from .local_storage import LocalStorageService
from .gcs_storage import GoogleCloudStorageService


async def migrate_project_to_gcs(
    project_id: str, 
    local_storage: LocalStorageService, 
    gcs_storage: GoogleCloudStorageService
) -> Dict[str, Any]:
    """
    Migrate a single project from local storage to Google Cloud Storage.
    
    Returns migration report with success/failure details.
    """
    print(f"🚀 Starting migration of project {project_id} to Google Cloud Storage...")
    
    migration_report = {
        'project_id': project_id,
        'files_migrated': 0,
        'files_failed': 0,
        'directories_created': 0,
        'success_files': [],
        'failed_files': [],
        'total_size_mb': 0
    }
    
    # Get local project directory
    local_project_dir = local_storage.get_absolute_path(f"projects/{project_id}")
    
    if not os.path.exists(local_project_dir):
        print(f"⚠️ Project {project_id} not found locally")
        return migration_report
    
    try:
        # Create project structure in GCS
        project_dirs = [
            f"projects/{project_id}/roi_cache",
            f"projects/{project_id}/defective_roi_cache", 
            f"projects/{project_id}/anomaly_features",
            f"projects/{project_id}/defect_detection_results",
            f"projects/{project_id}/generated_annotations",
            f"projects/{project_id}/segmentation_annotations",
            f"projects/{project_id}/visual_bounding_boxes",
            f"projects/{project_id}/visual_segmentation_masks"
        ]
        
        for directory in project_dirs:
            await gcs_storage.create_directory(directory)
            migration_report['directories_created'] += 1
        
        # Migrate all files
        for root, dirs, files in os.walk(local_project_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # Calculate relative path from project directory
                rel_path = os.path.relpath(local_file_path, local_project_dir)
                storage_path = f"projects/{project_id}/{rel_path}".replace('\\', '/')
                
                try:
                    # Get file size for reporting
                    file_size_mb = os.path.getsize(local_file_path) / (1024 * 1024)
                    migration_report['total_size_mb'] += file_size_mb
                    
                    # Upload to GCS
                    gcs_url = await gcs_storage.upload_from_local_file(local_file_path, storage_path)
                    
                    migration_report['files_migrated'] += 1
                    migration_report['success_files'].append({
                        'local_path': local_file_path,
                        'gcs_path': storage_path,
                        'gcs_url': gcs_url,
                        'size_mb': round(file_size_mb, 2)
                    })
                    
                    print(f"✅ Migrated: {rel_path} ({file_size_mb:.1f}MB)")
                    
                except Exception as e:
                    migration_report['files_failed'] += 1
                    migration_report['failed_files'].append({
                        'local_path': local_file_path,
                        'storage_path': storage_path,
                        'error': str(e)
                    })
                    print(f"❌ Failed to migrate {rel_path}: {e}")
        
        print(f"✅ Migration completed for project {project_id}")
        print(f"   📊 Files migrated: {migration_report['files_migrated']}")
        print(f"   ❌ Files failed: {migration_report['files_failed']}")
        print(f"   💾 Total size: {migration_report['total_size_mb']:.1f}MB")
        
    except Exception as e:
        print(f"❌ Migration failed for project {project_id}: {e}")
        migration_report['migration_error'] = str(e)
    
    return migration_report


async def migrate_all_projects_to_gcs(
    local_base_path: str = "ml/auto_annotation",
    bucket_name: str = None,
    gcs_project_id: str = None
) -> Dict[str, Any]:
    """
    Migrate all existing projects from local storage to Google Cloud Storage.
    
    Returns comprehensive migration report.
    """
    if not bucket_name or not gcs_project_id:
        raise ValueError("bucket_name and gcs_project_id are required for GCS migration")
    
    print(f"🌥️ Starting full migration to Google Cloud Storage...")
    print(f"   Source: {local_base_path}")
    print(f"   Destination: gs://{bucket_name}")
    
    # Initialize storage services
    local_storage = LocalStorageService(local_base_path)
    gcs_storage = GoogleCloudStorageService(bucket_name, gcs_project_id)
    
    # Find all local projects
    projects_dir = os.path.join(local_base_path, "projects")
    if not os.path.exists(projects_dir):
        print("⚠️ No projects directory found locally")
        return {'total_projects': 0, 'migrated_projects': 0, 'project_reports': []}
    
    project_ids = [d for d in os.listdir(projects_dir) 
                   if os.path.isdir(os.path.join(projects_dir, d))]
    
    print(f"📂 Found {len(project_ids)} projects to migrate: {project_ids}")
    
    # Migration report
    full_report = {
        'migration_timestamp': None,
        'total_projects': len(project_ids),
        'migrated_projects': 0,
        'failed_projects': 0,
        'total_files_migrated': 0,
        'total_files_failed': 0,
        'total_size_mb': 0,
        'project_reports': []
    }
    
    # Migrate each project
    for project_id in project_ids:
        try:
            report = await migrate_project_to_gcs(project_id, local_storage, gcs_storage)
            
            full_report['project_reports'].append(report)
            full_report['total_files_migrated'] += report['files_migrated']
            full_report['total_files_failed'] += report['files_failed']
            full_report['total_size_mb'] += report['total_size_mb']
            
            if report['files_migrated'] > 0 and report.get('migration_error') is None:
                full_report['migrated_projects'] += 1
            else:
                full_report['failed_projects'] += 1
                
        except Exception as e:
            print(f"❌ Critical error migrating project {project_id}: {e}")
            full_report['failed_projects'] += 1
            full_report['project_reports'].append({
                'project_id': project_id,
                'migration_error': str(e),
                'files_migrated': 0,
                'files_failed': 0
            })
    
    # Save migration report
    import json
    from datetime import datetime
    full_report['migration_timestamp'] = datetime.now().isoformat()
    
    print(f"\n🎯 Migration Summary:")
    print(f"   ✅ Projects migrated: {full_report['migrated_projects']}/{full_report['total_projects']}")
    print(f"   📁 Total files migrated: {full_report['total_files_migrated']}")
    print(f"   💾 Total data: {full_report['total_size_mb']:.1f}MB")
    
    return full_report


async def verify_migration(
    project_id: str,
    local_storage: LocalStorageService,
    gcs_storage: GoogleCloudStorageService
) -> Dict[str, Any]:
    """
    Verify that a migrated project has all files correctly transferred.
    """
    verification_report = {
        'project_id': project_id,
        'verified_files': 0,
        'missing_files': 0,
        'mismatched_files': 0,
        'missing_file_list': [],
        'verification_passed': False
    }
    
    print(f"🔍 Verifying migration for project {project_id}...")
    
    # Get all local files
    local_project_dir = local_storage.get_absolute_path(f"projects/{project_id}")
    
    if not os.path.exists(local_project_dir):
        verification_report['verification_error'] = "Local project directory not found"
        return verification_report
    
    local_files = []
    for root, dirs, files in os.walk(local_project_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), local_project_dir)
            local_files.append(rel_path.replace('\\', '/'))
    
    # Check each file exists in GCS
    for rel_file_path in local_files:
        gcs_file_path = f"projects/{project_id}/{rel_file_path}"
        
        try:
            exists = await gcs_storage.file_exists(gcs_file_path)
            if exists:
                verification_report['verified_files'] += 1
            else:
                verification_report['missing_files'] += 1
                verification_report['missing_file_list'].append(gcs_file_path)
                
        except Exception as e:
            verification_report['missing_files'] += 1
            verification_report['missing_file_list'].append(f"{gcs_file_path} (Error: {e})")
    
    verification_report['verification_passed'] = verification_report['missing_files'] == 0
    
    print(f"   ✅ Verified files: {verification_report['verified_files']}")
    print(f"   ❌ Missing files: {verification_report['missing_files']}")
    
    return verification_report