#!/usr/bin/env python3
"""
Fix annotation file counts for ODv3 training validation
"""
import requests
import json
import sys

def fix_project_annotations(base_url, project_id):
    """Fix annotation counts for a specific project"""
    
    print(f"🔧 Fixing annotation counts for project: {project_id}")
    
    # First, debug the current state
    debug_url = f"{base_url}/api/{project_id}/debug-file-counts"
    try:
        response = requests.get(debug_url)
        if response.status_code == 200:
            debug_data = response.json()
            print(f"📊 Current state:")
            print(f"   Project: {debug_data['debug_info']['project_name']}")
            print(f"   Type: {debug_data['debug_info']['project_type']}")
            print(f"   File counts: {debug_data['debug_info']['file_counts_from_get_project']}")
            print(f"   Raw DB counts: {debug_data['debug_info']['file_type_counts']}")
        else:
            print(f"❌ Could not debug project: {response.status_code}")
    except Exception as e:
        print(f"❌ Debug request failed: {e}")
    
    # Try to refresh file counts
    refresh_url = f"{base_url}/api/{project_id}/refresh-file-counts"
    try:
        response = requests.post(refresh_url)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Refreshed file counts: {result['file_counts']}")
        else:
            print(f"❌ Could not refresh counts: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Refresh request failed: {e}")
    
    # Check validation after fix
    validate_url = f"{base_url}/api/{project_id}/validate-dataset"
    try:
        response = requests.get(validate_url)
        if response.status_code == 200:
            validation = response.json()['validation']
            print(f"📋 Validation after fix:")
            print(f"   Ready for training: {validation['is_ready']}")
            print(f"   Requirements met: {validation['requirements_met']}")
            if validation['missing_requirements']:
                print(f"   Missing: {validation['missing_requirements']}")
        else:
            print(f"❌ Could not validate: {response.status_code}")
    except Exception as e:
        print(f"❌ Validation request failed: {e}")

def list_projects(base_url):
    """List all projects"""
    try:
        response = requests.get(f"{base_url}/api/")
        if response.status_code == 200:
            projects = response.json()['projects']
            print(f"📋 Available projects:")
            for project in projects:
                print(f"   - {project['project_id']} ({project['project_name']}) - {project['project_type']}")
            return projects
        else:
            print(f"❌ Could not list projects: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Failed to list projects: {e}")
        return []

if __name__ == "__main__":
    # Default to local backend
    base_url = "http://localhost:8000"
    
    if len(sys.argv) < 2:
        print("Usage: python fix_annotations.py PROJECT_ID")
        print("       python fix_annotations.py --list")
        print("")
        projects = list_projects(base_url)
    elif sys.argv[1] == "--list":
        projects = list_projects(base_url)
    else:
        project_id = sys.argv[1]
        fix_project_annotations(base_url, project_id)
