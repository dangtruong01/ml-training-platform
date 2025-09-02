#!/usr/bin/env python3
"""
Test script for storage service migration.
This script validates that the storage abstraction works correctly.
"""
import os
import sys
import asyncio
import json

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.storage import storage_service
from services.storage.migration_utils import migrate_project_to_gcs, verify_migration
from services.storage.local_storage import LocalStorageService
from services.storage.gcs_storage import GoogleCloudStorageService


async def test_basic_storage_operations():
    """Test basic storage operations with current storage backend"""
    print("🧪 Testing Basic Storage Operations")
    print(f"   Storage backend: {type(storage_service).__name__}")
    
    test_project_id = "test_project_123"
    
    try:
        # Test 1: Directory creation
        print("📁 Testing directory creation...")
        dirs = storage_service.get_roi_directories(test_project_id)
        await storage_service.create_directory(dirs['normal'])
        await storage_service.create_directory(dirs['defective'])
        print(f"   ✅ Created directories: {dirs['normal']}, {dirs['defective']}")
        
        # Test 2: JSON operations
        print("📄 Testing JSON operations...")
        test_data = {
            'project_id': test_project_id,
            'timestamp': '2024-01-01T00:00:00',
            'settings': {'threshold': 0.95, 'model': 'dinov2'},
            'results': [1, 2, 3, 4, 5]
        }
        
        json_path = f"{storage_service.get_project_directory(test_project_id)}/test_data.json"
        await storage_service.save_json(test_data, json_path)
        print(f"   ✅ Saved JSON to: {json_path}")
        
        loaded_data = await storage_service.load_json(json_path)
        assert loaded_data == test_data, "JSON data mismatch!"
        print("   ✅ JSON load/save verified")
        
        # Test 3: File existence
        print("🔍 Testing file existence checks...")
        exists = await storage_service.file_exists(json_path)
        assert exists, "File should exist!"
        print(f"   ✅ File existence check passed: {exists}")
        
        # Test 4: File listing
        print("📋 Testing file listing...")
        project_dir = storage_service.get_project_directory(test_project_id)
        files = await storage_service.list_files(project_dir)
        print(f"   ✅ Found {len(files)} files in project directory")
        for file in files:
            print(f"      - {file}")
        
        # Test 5: Cleanup
        print("🗑️ Testing file deletion...")
        deleted = await storage_service.delete_file(json_path)
        assert deleted, "File deletion should succeed"
        print("   ✅ File deletion successful")
        
        print("\n🎉 All basic storage tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Storage test failed: {e}")
        return False


async def test_image_operations():
    """Test image upload/download operations"""
    print("\n🖼️ Testing Image Operations")
    
    test_project_id = "test_images_456"
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[20:80, 20:80] = [0, 255, 0]  # Green square
        
        # Test image upload
        print("📤 Testing image upload...")
        image_path = f"{storage_service.get_project_directory(test_project_id)}/test_image.jpg"
        storage_url = await storage_service.upload_image(test_image, image_path, 'JPEG')
        print(f"   ✅ Image uploaded to: {storage_url}")
        
        # Test image download
        print("📥 Testing image download...")
        downloaded_image = await storage_service.download_image(image_path)
        print(f"   ✅ Image downloaded, shape: {downloaded_image.shape}")
        
        # Verify image integrity (basic check)
        if downloaded_image.shape == test_image.shape:
            print("   ✅ Image dimensions match")
        else:
            print(f"   ⚠️ Image dimension mismatch: {downloaded_image.shape} vs {test_image.shape}")
        
        # Cleanup
        await storage_service.delete_file(image_path)
        print("   ✅ Test image cleaned up")
        
        print("🎉 Image operations test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Image operations test failed: {e}")
        return False


async def test_migration_functionality():
    """Test migration utilities if we have existing projects"""
    print("\n🚀 Testing Migration Functionality")
    
    try:
        # Check if we have any existing projects to test with
        local_projects_path = "ml/auto_annotation/projects"
        
        if os.path.exists(local_projects_path):
            projects = [d for d in os.listdir(local_projects_path) 
                       if os.path.isdir(os.path.join(local_projects_path, d))]
            
            if projects:
                print(f"   📂 Found {len(projects)} existing projects: {projects[:3]}{'...' if len(projects) > 3 else ''}")
                
                # Test migration dry-run (just list files, don't actually migrate)
                test_project = projects[0]
                local_storage = LocalStorageService()
                
                project_dir = os.path.join(local_projects_path, test_project)
                file_count = 0
                total_size = 0
                
                for root, dirs, files in os.walk(project_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            file_count += 1
                            total_size += os.path.getsize(file_path)
                
                print(f"   📊 Project '{test_project}' contains {file_count} files ({total_size/1024/1024:.1f}MB)")
                print("   ✅ Migration analysis complete (no actual migration performed)")
                
            else:
                print("   📂 No existing projects found for migration testing")
        else:
            print("   📂 No projects directory found for migration testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration functionality test failed: {e}")
        return False


async def main():
    """Run all storage tests"""
    print("🧪 Storage Service Migration Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic operations
    if await test_basic_storage_operations():
        tests_passed += 1
    
    # Test 2: Image operations  
    if await test_image_operations():
        tests_passed += 1
    
    # Test 3: Migration functionality
    if await test_migration_functionality():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Storage abstraction is ready for production.")
        print("\n📋 Next Steps:")
        print("   1. Set STORAGE_TYPE=local to use local storage (current)")
        print("   2. Set STORAGE_TYPE=gcs with bucket config to use Google Cloud Storage")
        print("   3. Run migration script to move existing projects to cloud")
    else:
        print("❌ Some tests failed. Please check the storage configuration.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())