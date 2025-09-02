#!/usr/bin/env python3
"""
Setup script for Firestore database initialization.
Creates project documents and initial data structure.
"""
import os
import sys
from datetime import datetime
from google.cloud import firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def get_firestore_client():
    """Get Firestore client with sync operations"""
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    database_id = os.getenv('FIRESTORE_DATABASE_ID', '(default)')
    
    print(f"🔥 Connecting to Firestore: {project_id}/{database_id}")
    return firestore.Client(project=project_id, database=database_id)


def create_project_document(db, project_id: str, project_name: str, owner: str = "system"):
    """Create a project document in Firestore"""
    try:
        current_time = datetime.utcnow().isoformat() + 'Z'
        
        project_data = {
            'metadata': {
                'project_name': project_name,
                'project_type': 'auto_annotation',
                'owner': owner,
                'status': 'active',
                'created_at': current_time,
                'updated_at': current_time
            },
            'uploaded_files': {
                'training_images': [],
                'defective_images': [],
                'annotation_files': []
            },
            'processing_status': {
                'step1_roi_extraction': {'status': 'pending'},
                'step2_model_building': {'status': 'pending'},
                'step3_defect_detection': {'status': 'pending'},
                'step4_annotation_generation': {'status': 'pending'}
            },
            'results': {},
            'settings': {}
        }
        
        # Create the document
        doc_ref = db.collection('projects').document(project_id)
        doc_ref.set(project_data)
        
        print(f"✅ Created project document: {project_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating project {project_id}: {e}")
        return False


def scan_existing_projects():
    """Scan local filesystem for existing projects"""
    projects = []
    projects_dir = "ml/auto_annotation/projects"
    
    if os.path.exists(projects_dir):
        for item in os.listdir(projects_dir):
            project_path = os.path.join(projects_dir, item)
            if os.path.isdir(project_path):
                # Try to infer project name from directory structure
                project_name = item.replace('_', ' ').title()
                projects.append({
                    'project_id': item,
                    'project_name': project_name,
                    'local_path': project_path
                })
    
    return projects


def setup_firestore_database():
    """Main setup function"""
    print("🚀 Setting up Firestore database for ML Training Pipeline")
    print("=" * 60)
    
    try:
        # Connect to Firestore
        db = get_firestore_client()
        
        # Test connection with a simple read
        print("🔍 Testing Firestore connection...")
        collections = list(db.collections())
        print(f"✅ Connection successful. Found {len(collections)} existing collections.")
        
        # Scan for existing local projects
        print("\n📂 Scanning for existing projects...")
        existing_projects = scan_existing_projects()
        
        if existing_projects:
            print(f"Found {len(existing_projects)} existing projects:")
            for proj in existing_projects:
                print(f"   - {proj['project_id']} ({proj['project_name']})")
            
            # Create Firestore documents for existing projects
            print("\n📝 Creating Firestore documents...")
            success_count = 0
            
            for proj in existing_projects:
                if create_project_document(
                    db, 
                    proj['project_id'], 
                    proj['project_name'],
                    "hai_dang_truong@apple.com"
                ):
                    success_count += 1
            
            print(f"\n✅ Successfully created {success_count}/{len(existing_projects)} project documents")
            
        else:
            print("No existing projects found in local filesystem")
            
            # Create a sample project for testing
            print("\n🧪 Creating sample test project...")
            create_project_document(
                db,
                "sample_test_project",
                "Sample Test Project",
                "hai_dang_truong@apple.com"
            )
        
        # Verify setup
        print("\n🔍 Verifying Firestore setup...")
        projects_ref = db.collection('projects')
        docs = list(projects_ref.stream())
        
        print(f"✅ Firestore setup complete!")
        print(f"   Total projects in database: {len(docs)}")
        for doc in docs:
            data = doc.to_dict()
            print(f"   - {doc.id}: {data['metadata']['project_name']}")
        
        print("\n🎯 Next Steps:")
        print("   1. Restart your backend server")
        print("   2. Test project listing via API")
        print("   3. Upload files and verify Firestore tracking")
        
        return True
        
    except Exception as e:
        print(f"❌ Firestore setup failed: {e}")
        return False


if __name__ == "__main__":
    success = setup_firestore_database()
    sys.exit(0 if success else 1)