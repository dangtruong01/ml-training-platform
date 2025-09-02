"""
Database service for managing project data with PostgreSQL.
Provides high-level operations for the ML training pipeline.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

try:
    from backend.models.database_models import Base, Project, UploadedFile, ProcessingJob, Annotation
except ImportError:
    from models.database_models import Base, Project, UploadedFile, ProcessingJob, Annotation

load_dotenv()


class DatabaseService:
    """Service for managing project data in PostgreSQL"""
    
    def __init__(self):
        # Database connection configuration
        self.db_host = os.getenv('DATABASE_HOST', 'localhost')
        self.db_port = os.getenv('DATABASE_PORT', '5432')
        self.db_name = os.getenv('DATABASE_NAME', 'ml_training_pipeline')
        self.db_user = os.getenv('DATABASE_USER', os.getenv('USER'))  # Default to Mac username
        self.db_password = os.getenv('DATABASE_PASSWORD', '')
        
        # Create connection string
        if self.db_password:
            connection_string = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        else:
            connection_string = f"postgresql://{self.db_user}@{self.db_host}:{self.db_port}/{self.db_name}"
        
        # Create engine and session
        self.engine = create_engine(connection_string, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        print(f"🐘 Connected to PostgreSQL: {self.db_name} at {self.db_host}:{self.db_port}")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    # =============================================================================
    # PROJECT MANAGEMENT
    # =============================================================================
    
    def create_project(
        self, 
        project_id: str, 
        project_name: str, 
        owner: str,
        project_type: str = "auto_annotation"
    ) -> Dict[str, Any]:
        """Create a new project"""
        session = self.get_session()
        try:
            project = Project(
                project_id=project_id,
                project_name=project_name,
                project_type=project_type,
                owner=owner,
                storage_bucket=os.getenv('GCS_BUCKET_NAME'),
                storage_prefix=f"auto_annotation/projects/{project_id}"
            )
            
            session.add(project)
            session.commit()
            
            print(f"✅ Created project: {project_id}")
            return {'status': 'success', 'project_id': project_id}
            
        except SQLAlchemyError as e:
            session.rollback()
            print(f"❌ Error creating project: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()
    
    def get_project(self, project_id: str) -> Dict[str, Any]:
        """Get project with all related data"""
        session = self.get_session()
        try:
            project = session.query(Project).filter(Project.project_id == project_id).first()
            
            if project:
                # Get file counts
                file_counts = {}
                for file_type in ['training_images', 'defective_images', 'annotation_files']:
                    count = session.query(UploadedFile).filter(
                        UploadedFile.project_id == project_id,
                        UploadedFile.file_type == file_type
                    ).count()
                    file_counts[file_type] = count
                
                # Get latest processing status
                latest_jobs = {}
                for job_type in ['roi_extraction', 'model_building', 'defect_detection', 'annotation_generation']:
                    job = session.query(ProcessingJob).filter(
                        ProcessingJob.project_id == project_id,
                        ProcessingJob.job_type == job_type
                    ).order_by(desc(ProcessingJob.created_at)).first()
                    
                    if job:
                        latest_jobs[job_type] = {
                            'status': job.status,
                            'started_at': job.started_at.isoformat() if job.started_at else None,
                            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                            'results_summary': job.results_summary
                        }
                    else:
                        latest_jobs[job_type] = {'status': 'pending'}
                
                project_data = {
                    'project_id': project.project_id,
                    'project_name': project.project_name,
                    'project_type': project.project_type,
                    'owner': project.owner,
                    'status': project.status,
                    'created_at': project.created_at.isoformat(),
                    'updated_at': project.updated_at.isoformat(),
                    'file_counts': file_counts,
                    'processing_status': latest_jobs,
                    'settings': {
                        'roi_component_description': project.roi_component_description,
                        'roi_confidence_threshold': project.roi_confidence_threshold,
                        'anomaly_model_type': project.anomaly_model_type,
                        'anomaly_threshold': project.anomaly_threshold
                    }
                }
                
                return {'status': 'success', 'project': project_data}
            else:
                return {'status': 'error', 'message': 'Project not found'}
                
        except SQLAlchemyError as e:
            print(f"❌ Error getting project: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()
    
    def list_projects(self, owner: Optional[str] = None) -> Dict[str, Any]:
        """List all projects with summary data"""
        session = self.get_session()
        try:
            query = session.query(Project)
            if owner:
                query = query.filter(Project.owner == owner)
            
            projects = query.order_by(desc(Project.updated_at)).all()
            
            project_list = []
            for project in projects:
                # Get file counts
                file_counts = {}
                for file_type in ['training_images', 'defective_images']:
                    count = session.query(UploadedFile).filter(
                        UploadedFile.project_id == project.project_id,
                        UploadedFile.file_type == file_type
                    ).count()
                    file_counts[file_type] = count
                
                project_list.append({
                    'project_id': project.project_id,
                    'project_name': project.project_name,
                    'owner': project.owner,
                    'status': project.status,
                    'created_at': project.created_at.isoformat(),
                    'updated_at': project.updated_at.isoformat(),
                    'file_counts': file_counts
                })
            
            return {'status': 'success', 'projects': project_list}
            
        except SQLAlchemyError as e:
            print(f"❌ Error listing projects: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()
    
    def delete_project(self, project_id: str) -> Dict[str, Any]:
        """Delete a project and all its associated data"""
        session = self.get_session()
        try:
            project = session.query(Project).filter(Project.project_id == project_id).first()
            
            if project:
                # SQLAlchemy cascade will automatically delete related records
                session.delete(project)
                session.commit()
                
                print(f"✅ Deleted project and all associated data: {project_id}")
                return {'status': 'success', 'message': f'Project {project_id} deleted successfully'}
            else:
                return {'status': 'error', 'message': 'Project not found'}
                
        except SQLAlchemyError as e:
            session.rollback()
            print(f"❌ Error deleting project: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()
    
    # =============================================================================
    # FILE TRACKING
    # =============================================================================
    
    def add_uploaded_file(
        self,
        project_id: str,
        file_type: str,
        filename: str,
        original_filename: str,
        storage_url: str,
        storage_path: str,
        file_size_bytes: int,
        content_type: str = None,
        image_width: int = None,
        image_height: int = None
    ) -> Dict[str, Any]:
        """Track uploaded file in database"""
        session = self.get_session()
        try:
            uploaded_file = UploadedFile(
                project_id=project_id,
                file_type=file_type,
                filename=filename,
                original_filename=original_filename,
                storage_url=storage_url,
                storage_path=storage_path,
                file_size_bytes=file_size_bytes,
                content_type=content_type,
                image_width=image_width,
                image_height=image_height
            )
            
            session.add(uploaded_file)
            session.commit()
            
            print(f"✅ Tracked uploaded file: {filename}")
            return {'status': 'success', 'file_id': uploaded_file.id}
            
        except SQLAlchemyError as e:
            session.rollback()
            print(f"❌ Error tracking uploaded file: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()
    
    def get_uploaded_files(self, project_id: str, file_type: str = None) -> List[Dict[str, Any]]:
        """Get uploaded files for a project"""
        session = self.get_session()
        try:
            query = session.query(UploadedFile).filter(UploadedFile.project_id == project_id)
            
            if file_type:
                query = query.filter(UploadedFile.file_type == file_type)
            
            files = query.order_by(desc(UploadedFile.upload_date)).all()
            
            file_list = []
            for file in files:
                file_list.append({
                    'id': file.id,
                    'filename': file.filename,
                    'original_filename': file.original_filename,
                    'file_type': file.file_type,
                    'storage_url': file.storage_url,
                    'storage_path': file.storage_path,
                    'file_size_bytes': file.file_size_bytes,
                    'upload_date': file.upload_date.isoformat(),
                    'image_dimensions': [file.image_width, file.image_height] if file.image_width else None,
                    'is_processed': file.is_processed
                })
            
            return file_list
            
        except SQLAlchemyError as e:
            print(f"❌ Error getting uploaded files: {e}")
            return []
        finally:
            session.close()
    
    # =============================================================================
    # PROCESSING STATUS TRACKING
    # =============================================================================
    
    def start_processing_job(
        self,
        project_id: str,
        job_type: str,
        job_settings: Dict[str, Any] = None,
        input_files_count: int = 0
    ) -> int:
        """Start a new processing job and return job ID"""
        session = self.get_session()
        try:
            job = ProcessingJob(
                project_id=project_id,
                job_type=job_type,
                status='in_progress',
                started_at=func.current_timestamp(),
                job_settings=job_settings,
                input_files_count=input_files_count
            )
            
            session.add(job)
            session.commit()
            
            print(f"✅ Started processing job: {job_type} (ID: {job.id})")
            return job.id
            
        except SQLAlchemyError as e:
            session.rollback()
            print(f"❌ Error starting processing job: {e}")
            return None
        finally:
            session.close()
    
    def complete_processing_job(
        self,
        job_id: int,
        results_summary: Dict[str, Any] = None,
        output_files_count: int = 0,
        error_message: str = None
    ) -> bool:
        """Complete a processing job with results"""
        session = self.get_session()
        try:
            job = session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            
            if job:
                job.status = 'failed' if error_message else 'completed'
                job.completed_at = func.current_timestamp()
                job.results_summary = results_summary
                job.output_files_count = output_files_count
                job.error_message = error_message
                
                if job.started_at:
                    duration = datetime.utcnow() - job.started_at
                    job.duration_seconds = int(duration.total_seconds())
                
                session.commit()
                print(f"✅ Completed processing job: {job_id}")
                return True
            else:
                print(f"❌ Processing job not found: {job_id}")
                return False
                
        except SQLAlchemyError as e:
            session.rollback()
            print(f"❌ Error completing processing job: {e}")
            return False
        finally:
            session.close()


# Global database service instance
database_service = DatabaseService()