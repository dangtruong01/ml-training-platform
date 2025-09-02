"""
SQLAlchemy models for ML Training Pipeline database.
Maps to PostgreSQL tables created in setup script.
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, TIMESTAMP, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Project(Base):
    """Core project metadata and configuration"""
    __tablename__ = 'projects'
    
    project_id = Column(String(255), primary_key=True)
    project_name = Column(String(255), nullable=False)
    project_type = Column(String(50), default='auto_annotation')
    owner = Column(String(255), nullable=False)
    status = Column(String(50), default='active')
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Storage configuration
    storage_bucket = Column(String(255))
    storage_prefix = Column(String(255))
    
    # Workflow settings
    roi_component_description = Column(Text, default='metal plate')
    roi_confidence_threshold = Column(Float, default=0.3)
    anomaly_model_type = Column(String(50), default='dinov2')
    anomaly_threshold = Column(Float, default=0.05)
    
    # Relationships
    uploaded_files = relationship("UploadedFile", back_populates="project", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="project", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="project", cascade="all, delete-orphan")


class UploadedFile(Base):
    """Track uploaded files with storage URLs and metadata"""
    __tablename__ = 'uploaded_files'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(255), ForeignKey('projects.project_id', ondelete='CASCADE'), nullable=False)
    file_type = Column(String(50), nullable=False)  # 'training_images', 'defective_images', 'annotation_files'
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    storage_url = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    file_size_bytes = Column(BigInteger)
    content_type = Column(String(100))
    upload_date = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    # Image metadata
    image_width = Column(Integer)
    image_height = Column(Integer)
    is_processed = Column(Boolean, default=False)
    
    # Relationships
    project = relationship("Project", back_populates="uploaded_files")


class ProcessingJob(Base):
    """Track workflow step execution and results"""
    __tablename__ = 'processing_jobs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(255), ForeignKey('projects.project_id', ondelete='CASCADE'), nullable=False)
    job_type = Column(String(50), nullable=False)  # 'roi_extraction', 'model_building', etc.
    status = Column(String(50), default='pending')  # 'pending', 'in_progress', 'completed', 'failed'
    
    # Timing
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    duration_seconds = Column(Integer)
    
    # Configuration and results
    job_settings = Column(JSONB)
    results_summary = Column(JSONB)
    error_message = Column(Text)
    
    # File tracking
    input_files_count = Column(Integer, default=0)
    output_files_count = Column(Integer, default=0)
    
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    # Relationships
    project = relationship("Project", back_populates="processing_jobs")
    annotations = relationship("Annotation", back_populates="processing_job")


class Annotation(Base):
    """Track generated annotations and their metadata"""
    __tablename__ = 'annotations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(255), ForeignKey('projects.project_id', ondelete='CASCADE'), nullable=False)
    annotation_type = Column(String(50), nullable=False)  # 'bounding_boxes', 'segmentation_masks'
    source_image_filename = Column(String(255), nullable=False)
    
    # Annotation data
    annotation_format = Column(String(50))  # 'yolo', 'coco'
    annotation_file_url = Column(Text)
    visual_preview_url = Column(Text)
    
    # Statistics
    objects_count = Column(Integer, default=0)
    annotation_area_pixels = Column(Integer)
    confidence_score = Column(Float)
    
    # Processing info
    generated_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    processing_job_id = Column(Integer, ForeignKey('processing_jobs.id'))
    
    # Relationships
    project = relationship("Project", back_populates="annotations")
    processing_job = relationship("ProcessingJob", back_populates="annotations")