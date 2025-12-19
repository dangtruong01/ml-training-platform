#!/usr/bin/env python3
"""
Debug script to check project file counts in the database
"""
import sys
import os

# Add the backend path to sys.path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

try:
    from services.core.database_service import database_service
    
    def debug_project_files(project_id):
        """Debug project file counts"""
        print(f"🔍 Debugging project: {project_id}")
        
        # Get project data
        project_result = database_service.get_project(project_id)
        if project_result['status'] != 'success':
            print(f"❌ Project not found: {project_id}")
            return
        
        project_data = project_result['project']
        
        print(f"📊 Project Data:")
        print(f"   - Name: {project_data['project_name']}")
        print(f"   - Type: {project_data['project_type']}")
        print(f"   - Status: {project_data['status']}")
        
        print(f"📁 File Counts:")
        file_counts = project_data.get('file_counts', {})
        for key, value in file_counts.items():
            print(f"   - {key}: {value}")
        
        # Check raw database records
        print(f"📋 Raw Database Records:")
        from services.core.database_service import UploadedFile, DatabaseService
        
        db_service = DatabaseService()
        session = db_service.get_session()
        
        try:
            files = session.query(UploadedFile).filter(
                UploadedFile.project_id == project_id
            ).all()
            
            file_type_counts = {}
            for file in files:
                if file.file_type not in file_type_counts:
                    file_type_counts[file.file_type] = 0
                file_type_counts[file.file_type] += 1
                print(f"   - {file.file_type}: {file.filename}")
            
            print(f"📈 File Type Counts:")
            for file_type, count in file_type_counts.items():
                print(f"   - {file_type}: {count}")
                
        finally:
            session.close()
    
    if len(sys.argv) < 2:
        print("Usage: python debug_project.py PROJECT_ID")
        
        # List all projects
        projects_result = database_service.list_projects()
        if projects_result['status'] == 'success':
            print("\n📋 Available Projects:")
            for project in projects_result['projects']:
                print(f"   - {project['project_id']} ({project['project_name']})")
    else:
        project_id = sys.argv[1]
        debug_project_files(project_id)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
