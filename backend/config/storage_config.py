"""
Storage configuration management for cloud migration.
Handles environment variables and service initialization.
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class StorageConfig:
    """Configuration manager for storage services"""
    
    def __init__(self):
        self.storage_type = os.getenv('STORAGE_TYPE', 'local').lower()
        self.local_storage_path = os.getenv('LOCAL_STORAGE_PATH', 'ml/auto_annotation')
        self.gcs_bucket_name = os.getenv('GCS_BUCKET_NAME')
        self.gcs_project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.gcs_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate storage configuration and return status"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'config_summary': {}
        }
        
        if self.storage_type == 'gcs':
            # Validate GCS configuration
            if not self.gcs_bucket_name:
                validation['is_valid'] = False
                validation['errors'].append("GCS_BUCKET_NAME environment variable is required for Google Cloud Storage")
            
            if not self.gcs_project_id:
                validation['is_valid'] = False  
                validation['errors'].append("GOOGLE_CLOUD_PROJECT environment variable is required for Google Cloud Storage")
            
            if not self.gcs_credentials_path:
                validation['warnings'].append("GOOGLE_APPLICATION_CREDENTIALS not set - will use default credentials")
            elif not os.path.exists(self.gcs_credentials_path):
                validation['errors'].append(f"Service account credentials file not found: {self.gcs_credentials_path}")
                validation['is_valid'] = False
            
            validation['config_summary'] = {
                'storage_type': 'Google Cloud Storage',
                'bucket': self.gcs_bucket_name,
                'project': self.gcs_project_id,
                'credentials': self.gcs_credentials_path or 'default'
            }
            
        elif self.storage_type == 'local':
            # Validate local storage configuration
            if not os.path.exists(self.local_storage_path):
                try:
                    os.makedirs(self.local_storage_path, exist_ok=True)
                    validation['warnings'].append(f"Created local storage directory: {self.local_storage_path}")
                except Exception as e:
                    validation['is_valid'] = False
                    validation['errors'].append(f"Cannot create local storage directory {self.local_storage_path}: {e}")
            
            validation['config_summary'] = {
                'storage_type': 'Local Filesystem',
                'base_path': os.path.abspath(self.local_storage_path)
            }
            
        else:
            validation['is_valid'] = False
            validation['errors'].append(f"Invalid STORAGE_TYPE: {self.storage_type}. Must be 'local' or 'gcs'")
        
        return validation
    
    def get_environment_template(self) -> str:
        """Generate environment variable template for deployment"""
        return """
# Storage Configuration for Claude Code Auto-Annotation Pipeline
# =============================================================================

# Storage Backend Selection
STORAGE_TYPE=local                        # 'local' or 'gcs'

# Local Storage Configuration (when STORAGE_TYPE=local)
LOCAL_STORAGE_PATH=ml/auto_annotation     # Base directory for local storage

# Google Cloud Storage Configuration (when STORAGE_TYPE=gcs)  
GCS_BUCKET_NAME=your-bucket-name          # GCS bucket for storing all project data
GOOGLE_CLOUD_PROJECT=your-project-id     # Google Cloud project ID
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json  # Optional: path to service account JSON

# Cloud Run / Container Configuration
PORT=8000                                 # Port for the API server
ENVIRONMENT=production                    # 'development', 'staging', or 'production'

# =============================================================================
# Usage Instructions:
# 
# For Local Development:
#   STORAGE_TYPE=local
#   LOCAL_STORAGE_PATH=ml/auto_annotation
#
# For Google Cloud Deployment:
#   STORAGE_TYPE=gcs
#   GCS_BUCKET_NAME=my-ml-pipeline-bucket
#   GOOGLE_CLOUD_PROJECT=my-gcp-project
#   GOOGLE_APPLICATION_CREDENTIALS=/secrets/service-account.json
# =============================================================================
"""


# Global configuration instance
config = StorageConfig()


def print_storage_status():
    """Print current storage configuration status"""
    validation = config.validate_config()
    
    print("🔧 Storage Configuration Status")
    print("-" * 40)
    
    if validation['is_valid']:
        print("✅ Configuration is valid")
        print(f"   Storage Type: {validation['config_summary']['storage_type']}")
        for key, value in validation['config_summary'].items():
            if key != 'storage_type':
                print(f"   {key.title()}: {value}")
    else:
        print("❌ Configuration has errors:")
        for error in validation['errors']:
            print(f"   • {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"   • {warning}")
    
    print("-" * 40)
    return validation['is_valid']


if __name__ == "__main__":
    print_storage_status()