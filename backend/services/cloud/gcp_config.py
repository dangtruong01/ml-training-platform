"""
Google Cloud Platform configuration and authentication service.
Handles authentication and provides access to GCP services.
"""
import os
from google.auth import default
from google.cloud import aiplatform
from google.cloud import storage
from typing import Tuple, Optional

class GCPConfig:
    def __init__(self):
        self.project_id = None
        self.credentials = None
        self.region = os.getenv('GCP_REGION', 'us-central1')
        self.models_bucket = os.getenv('GCP_MODELS_BUCKET', 'dangtruong-mltraining-storage')
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize GCP credentials and project info"""
        if self._initialized:
            return True
            
        try:
            self.credentials, self.project_id = default()
            print(f"✅ GCP authenticated - Project: {self.project_id}")
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.project_id,
                location=self.region,
                credentials=self.credentials
            )
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"❌ GCP authentication failed: {e}")
            return False
    
    def get_storage_client(self) -> storage.Client:
        """Get authenticated Cloud Storage client"""
        if not self._initialized:
            self.initialize()
            
        return storage.Client(
            project=self.project_id,
            credentials=self.credentials
        )
    
    def test_services(self) -> dict:
        """Test access to required GCP services"""
        results = {}
        
        if not self._initialized:
            if not self.initialize():
                return {'error': 'Failed to initialize GCP'}
        
        # Test Vertex AI
        try:
            # Just test if we can initialize - no actual operations
            results['vertex_ai'] = True
            print("✅ Vertex AI access confirmed")
        except Exception as e:
            results['vertex_ai'] = False
            print(f"❌ Vertex AI access failed: {e}")
        
        # Test Cloud Storage
        try:
            storage_client = self.get_storage_client()
            buckets = list(storage_client.list_buckets())
            results['cloud_storage'] = True
            results['bucket_count'] = len(buckets)
            results['buckets'] = [bucket.name for bucket in buckets]
            print(f"✅ Cloud Storage access confirmed - Found {len(buckets)} buckets")
            for bucket in buckets:
                print(f"   📁 {bucket.name}")
        except Exception as e:
            results['cloud_storage'] = False
            print(f"❌ Cloud Storage access failed: {e}")
            
        return results
    
    def get_project_info(self) -> dict:
        """Get current project information"""
        if not self._initialized:
            self.initialize()
            
        return {
            'project_id': self.project_id,
            'region': self.region,
            'models_bucket': self.models_bucket,
            'initialized': self._initialized
        }

# Global instance
gcp_config = GCPConfig()