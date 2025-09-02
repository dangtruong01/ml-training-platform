#!/usr/bin/env python3
"""
Script to create a YOLO-format ZIP file from the annotated_dataset folder
for use with the auto-annotation system.
"""
import os
import zipfile
import shutil
from pathlib import Path

def create_yolo_zip():
    """Create ZIP file with YOLO annotations for auto-annotation upload"""
    
    # Paths
    annotated_dataset_dir = "annotated_dataset"
    labels_dir = os.path.join(annotated_dataset_dir, "labels")
    classes_file = os.path.join(annotated_dataset_dir, "classes.txt")
    output_zip = "yolo_annotations.zip"
    
    # Check if directories exist
    if not os.path.exists(annotated_dataset_dir):
        print(f"❌ Directory {annotated_dataset_dir} not found!")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory {labels_dir} not found!")
        return False
    
    # Create ZIP file
    print(f"📦 Creating {output_zip}...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all .txt files from labels directory
        txt_files = 0
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(labels_dir, filename)
                zip_file.write(file_path, filename)  # Store in root of ZIP
                txt_files += 1
                print(f"  ✅ Added {filename}")
        
        # Add classes.txt if it exists
        if os.path.exists(classes_file):
            zip_file.write(classes_file, "classes.txt")
            print(f"  ✅ Added classes.txt")
        else:
            print(f"  ⚠️ classes.txt not found, will be auto-generated")
    
    print(f"🎉 Successfully created {output_zip} with {txt_files} annotation files")
    print(f"📁 File size: {os.path.getsize(output_zip)} bytes")
    
    # Show instructions
    print("\n📋 Next steps:")
    print("1. Go to the auto-annotation page in the frontend")
    print("2. Create a new 'Object Detection' project")
    print("3. Upload your training images (from annotated_dataset/images/)")
    print(f"4. Upload the annotations ZIP file: {output_zip}")
    print("5. Start training!")
    
    return True

def verify_annotations():
    """Verify annotation format before creating ZIP"""
    labels_dir = os.path.join("annotated_dataset", "labels")
    
    if not os.path.exists(labels_dir):
        return False
    
    print("🔍 Verifying annotation format...")
    
    valid_files = 0
    total_annotations = 0
    classes_found = set()
    
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'):
            continue
        
        file_path = os.path.join(labels_dir, filename)
        valid_annotations = 0
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"  ⚠️ {filename}:{line_num} - Invalid format (expected 5 values): {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check if coordinates are normalized (0-1)
                    if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        print(f"  ⚠️ {filename}:{line_num} - Coordinates not normalized: {line}")
                        continue
                    
                    classes_found.add(class_id)
                    valid_annotations += 1
                    
                except ValueError:
                    print(f"  ⚠️ {filename}:{line_num} - Invalid number format: {line}")
                    continue
        
        if valid_annotations > 0:
            valid_files += 1
            total_annotations += valid_annotations
            print(f"  ✅ {filename}: {valid_annotations} annotations")
    
    print(f"\n📊 Verification Summary:")
    print(f"  Valid files: {valid_files}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Classes found: {sorted(classes_found)}")
    
    # Check classes.txt
    classes_file = os.path.join("annotated_dataset", "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"  Class names: {class_names}")
    
    return valid_files > 0

if __name__ == "__main__":
    print("🚀 YOLO Annotation ZIP Creator")
    print("=" * 40)
    
    # Verify annotations first
    if verify_annotations():
        print("\n" + "=" * 40)
        create_yolo_zip()
    else:
        print("❌ No valid annotations found!")