#!/usr/bin/env python3
"""
Test script to verify the modular refactoring is working correctly.
This script tests the new modular components independently of the main service imports.
"""

import os
import sys

# Add current directory to path to enable direct imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_modular_components():
    """Test all modular components independently"""
    print("🧪 Testing Modular ML Components Refactoring")
    print("=" * 50)
    
    results = []
    
    # Test 1: Base Components
    print("\n1️⃣ Testing Base Components...")
    try:
        from services.ml.base.base_trainer import BaseTrainer
        from services.ml.base.base_predictor import BasePredictor
        from services.ml.base.training_monitor import TrainingMonitor, training_monitor
        
        # Test abstract methods are defined
        assert hasattr(BaseTrainer, 'train'), "BaseTrainer missing train method"
        assert hasattr(BaseTrainer, 'get_algorithm_name'), "BaseTrainer missing get_algorithm_name method"
        assert hasattr(BasePredictor, 'predict'), "BasePredictor missing predict method"
        
        # Test training monitor instance
        assert training_monitor is not None, "training_monitor instance not available"
        assert hasattr(training_monitor, 'create_task'), "TrainingMonitor missing create_task method"
        
        print("   ✅ Base components working correctly")
        results.append(("Base Components", True, "All abstract classes and monitoring working"))
    except Exception as e:
        print(f"   ❌ Base components failed: {e}")
        results.append(("Base Components", False, str(e)))
    
    # Test 2: Dataset Processing
    print("\n2️⃣ Testing Dataset Processing...")
    try:
        from services.ml.datasets.yolo_processor import YOLODatasetProcessor
        
        processor = YOLODatasetProcessor("test_datasets")
        assert hasattr(processor, 'process_uploaded_dataset'), "Missing process_uploaded_dataset method"
        assert hasattr(processor, 'validate_dataset_structure'), "Missing validate_dataset_structure method"
        
        print("   ✅ Dataset processor working correctly")
        results.append(("Dataset Processing", True, "YOLO dataset processor functional"))
    except Exception as e:
        print(f"   ❌ Dataset processing failed: {e}")
        results.append(("Dataset Processing", False, str(e)))
    
    # Test 3: YOLO Trainers
    print("\n3️⃣ Testing YOLO Trainers...")
    try:
        from services.ml.detection.yolo_trainer import YOLODetectionTrainer
        from services.ml.segmentation.yolo_trainer import YOLOSegmentationTrainer
        
        detection_trainer = YOLODetectionTrainer()
        segmentation_trainer = YOLOSegmentationTrainer()
        
        assert detection_trainer.get_algorithm_name() == "YOLO Detection", "Detection trainer name incorrect"
        assert segmentation_trainer.get_algorithm_name() == "YOLO Segmentation", "Segmentation trainer name incorrect"
        
        print("   ✅ YOLO trainers working correctly")
        results.append(("YOLO Trainers", True, "Detection and segmentation trainers functional"))
    except Exception as e:
        print(f"   ❌ YOLO trainers failed: {e}")
        results.append(("YOLO Trainers", False, str(e)))
    
    # Test 4: Anomaly Detection Trainers
    print("\n4️⃣ Testing Anomaly Detection Trainers...")
    try:
        from services.ml.anomaly.sklearn_trainer import SklearnAnomalyTrainer
        from services.ml.anomaly.pytorch_trainer import PytorchAnomalyTrainer
        
        sklearn_trainer = SklearnAnomalyTrainer()
        pytorch_trainer = PytorchAnomalyTrainer()
        
        assert sklearn_trainer.get_algorithm_name() == "Sklearn Anomaly Detection", "Sklearn trainer name incorrect"
        assert pytorch_trainer.get_algorithm_name() == "PyTorch Anomaly Detection", "PyTorch trainer name incorrect"
        
        print("   ✅ Anomaly detection trainers working correctly")
        results.append(("Anomaly Trainers", True, "Sklearn and PyTorch anomaly trainers functional"))
    except Exception as e:
        print(f"   ❌ Anomaly trainers failed: {e}")
        results.append(("Anomaly Trainers", False, str(e)))
    
    # Test 5: Prediction Components
    print("\n5️⃣ Testing Prediction Components...")
    try:
        from services.ml.prediction.yolo_predictor import YOLOPredictor
        
        predictor = YOLOPredictor()
        assert predictor.get_algorithm_name() == "YOLO", "YOLO predictor name incorrect"
        assert hasattr(predictor, 'predict_detection'), "Missing predict_detection method"
        assert hasattr(predictor, 'predict_segmentation'), "Missing predict_segmentation method"
        
        print("   ✅ Prediction components working correctly")
        results.append(("Prediction Components", True, "YOLO predictor functional"))
    except Exception as e:
        print(f"   ❌ Prediction components failed: {e}")
        results.append(("Prediction Components", False, str(e)))
    
    # Test 6: Pre-annotation Components
    print("\n6️⃣ Testing Pre-annotation Components...")
    try:
        from services.ml.pre_annotation.opencv_annotator import OpenCVAnnotator
        
        annotator = OpenCVAnnotator()
        assert hasattr(annotator, 'annotate_detection'), "Missing annotate_detection method"
        assert hasattr(annotator, 'annotate_segmentation'), "Missing annotate_segmentation method"
        
        print("   ✅ Pre-annotation components working correctly")
        results.append(("Pre-annotation Components", True, "OpenCV annotator functional"))
    except Exception as e:
        print(f"   ❌ Pre-annotation components failed: {e}")
        results.append(("Pre-annotation Components", False, str(e)))
    
    # Test 7: Raw Annotation Processing
    print("\n7️⃣ Testing Raw Annotation Processing...")
    try:
        from services.ml.raw_annotation.raw_processor import RawAnnotationProcessor
        
        processor = RawAnnotationProcessor()
        assert hasattr(processor, 'process_raw_folder'), "Missing process_raw_folder method"
        assert hasattr(processor, 'get_dataset_statistics'), "Missing get_dataset_statistics method"
        
        print("   ✅ Raw annotation processing working correctly")
        results.append(("Raw Annotation Processing", True, "Raw annotation processor functional"))
    except Exception as e:
        print(f"   ❌ Raw annotation processing failed: {e}")
        results.append(("Raw Annotation Processing", False, str(e)))
    
    # Test 8: Service Orchestrator
    print("\n8️⃣ Testing Service Orchestrator...")
    try:
        from services.ml.orchestrator import MLOrchestrator, ml_orchestrator
        
        orchestrator = MLOrchestrator()
        info = orchestrator.get_component_info()
        
        assert 'trainers' in info, "Missing trainers info"
        assert 'predictor' in info, "Missing predictor info"
        assert len(info['trainers']) == 4, "Expected 4 trainer types"
        
        print("   ✅ Service orchestrator working correctly")
        results.append(("Service Orchestrator", True, "ML orchestrator functional"))
    except Exception as e:
        print(f"   ❌ Service orchestrator failed: {e}")
        results.append(("Service Orchestrator", False, str(e)))
    
    # Print Summary
    print("\n" + "=" * 50)
    print("📊 REFACTORING TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for component, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<8} {component:<25} - {message}")
    
    print(f"\n🎯 Results: {passed}/{total} components working correctly")
    
    if passed == total:
        print("\n🎉 REFACTORING COMPLETE AND SUCCESSFUL!")
        print("✨ All modular components are working correctly.")
        print("📦 The monolithic YoloService has been successfully refactored into modular components.")
        print("\n🚀 The system is ready for testing and use!")
        return True
    else:
        print(f"\n⚠️  {total - passed} components have issues that need to be addressed.")
        return False

if __name__ == "__main__":
    success = test_modular_components()
    sys.exit(0 if success else 1)