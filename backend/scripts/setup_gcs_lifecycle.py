#!/usr/bin/env python3
"""
Script to set up GCS lifecycle policy for automatic cleanup of model files.
This sets up a 7-day automatic deletion policy on the mltraining-models bucket.
"""

from google.cloud import storage
import json

def setup_gcs_lifecycle_policy():
    """Set up lifecycle policy to delete objects after 7 days"""
    
    bucket_name = 'mltraining-models'
    
    try:
        print(f"🌥️ Setting up GCS lifecycle policy for bucket: {bucket_name}")
        
        # Initialize the storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Define lifecycle policy
        lifecycle_policy = {
            "rule": [
                {
                    "action": {"type": "Delete"},
                    "condition": {
                        "age": 7,  # Delete objects older than 7 days
                        "matchesStorageClass": ["STANDARD", "NEARLINE", "COLDLINE"]
                    }
                }
            ]
        }
        
        # Apply the lifecycle policy
        bucket.lifecycle_management_policy = lifecycle_policy
        bucket.patch()
        
        print(f"✅ Successfully applied lifecycle policy to {bucket_name}")
        print(f"📋 Policy details:")
        print(f"   - Action: Delete objects")
        print(f"   - Condition: Age > 7 days")
        print(f"   - Applies to: All storage classes")
        
        # Verify the policy was applied
        updated_bucket = storage_client.bucket(bucket_name)
        updated_bucket.reload()
        
        if updated_bucket.lifecycle_management_policy:
            print(f"🔍 Verification: Lifecycle policy is active")
            policy_rules = updated_bucket.lifecycle_management_policy.get('rule', [])
            for i, rule in enumerate(policy_rules):
                action = rule.get('action', {}).get('type')
                age = rule.get('condition', {}).get('age')
                print(f"   Rule {i+1}: {action} objects after {age} days")
        else:
            print(f"⚠️ Warning: Could not verify lifecycle policy")
            
        return True
        
    except Exception as e:
        print(f"❌ Error setting up lifecycle policy: {e}")
        print(f"   Make sure you have the required permissions on the bucket")
        print(f"   Required roles: Storage Admin or Storage Object Admin")
        return False

def check_existing_policy():
    """Check if there's already a lifecycle policy on the bucket"""
    
    bucket_name = 'mltraining-models'
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        bucket.reload()
        
        if bucket.lifecycle_management_policy:
            print(f"📋 Existing lifecycle policy found on {bucket_name}:")
            policy_rules = bucket.lifecycle_management_policy.get('rule', [])
            for i, rule in enumerate(policy_rules):
                action = rule.get('action', {}).get('type')
                age = rule.get('condition', {}).get('age')
                storage_class = rule.get('condition', {}).get('matchesStorageClass', [])
                print(f"   Rule {i+1}: {action} objects after {age} days (classes: {storage_class})")
            return True
        else:
            print(f"ℹ️ No existing lifecycle policy found on {bucket_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking existing policy: {e}")
        return False

def main():
    """Main function to set up GCS lifecycle policy"""
    
    print("🚀 GCS Lifecycle Policy Setup")
    print("=" * 50)
    
    # Check if bucket exists and is accessible
    bucket_name = 'mltraining-models'
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        bucket.reload()
        print(f"✅ Successfully connected to bucket: {bucket_name}")
    except Exception as e:
        print(f"❌ Cannot access bucket {bucket_name}: {e}")
        print(f"   Make sure the bucket exists and you have proper permissions")
        return False
    
    # Check existing policy
    print(f"\n1. Checking existing lifecycle policy...")
    has_existing = check_existing_policy()
    
    # Ask user if they want to proceed
    if has_existing:
        response = input(f"\nOverwrite existing policy? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled by user")
            return False
    
    # Apply new policy
    print(f"\n2. Applying 7-day deletion policy...")
    success = setup_gcs_lifecycle_policy()
    
    if success:
        print(f"\n🎉 GCS Lifecycle policy setup complete!")
        print(f"📁 Model files in {bucket_name} will be automatically deleted after 7 days")
        print(f"💡 This supports the delete model functionality in the app")
    else:
        print(f"\n❌ Failed to set up lifecycle policy")
        print(f"💡 You may need to set this up manually in the GCS console")
    
    return success

if __name__ == "__main__":
    main()