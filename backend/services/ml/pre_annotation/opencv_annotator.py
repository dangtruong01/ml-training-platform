import os
import cv2
import numpy as np
from typing import List


class OpenCVAnnotator:
    """OpenCV-based pre-annotation for detection and segmentation"""
    
    def __init__(self, results_dir: str = "ml/results"):
        self.results_dir = os.path.abspath(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def annotate_detection(self, image_path: str) -> str:
        """Enhanced OpenCV-based object detection with edge detection for metallic objects"""
        results_dir = os.path.abspath(os.path.join(self.results_dir, "pre_annotation"))
        os.makedirs(results_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        annotated_img = img.copy()
        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width
        
        # Save original for debugging
        original_debug = os.path.join(results_dir, f"debug_original_{os.path.basename(image_path)}")
        cv2.imwrite(original_debug, img)
        
        # METHOD 1: Enhanced edge-based detection
        valid_contours = self._edge_based_detection(img, results_dir, image_path)
        
        # METHOD 2: If insufficient contours, try adaptive thresholding
        if len(valid_contours) < 1:
            valid_contours.extend(self._adaptive_threshold_detection(img, img_area, results_dir, image_path))
        
        # METHOD 3: If still insufficient, try color-based detection
        if len(valid_contours) < 1:
            valid_contours.extend(self._color_based_detection(img, img_area))
        
        # Remove duplicates and filter by area
        valid_contours = self._filter_and_merge_contours(valid_contours, img_area)
        
        # Draw bounding boxes (detection style)
        for i, c in enumerate(valid_contours[:5]):
            x, y, w, h = cv2.boundingRect(c)
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)][i % 5]
            
            # Draw bounding rectangle
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
            
            # Add area and confidence info
            area = cv2.contourArea(c)
            label = f"Object {i+1} ({int(area)}px)"
            cv2.putText(annotated_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add detection summary
        summary = f"Detection: {len(valid_contours)} objects found"
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        output_path = os.path.join(results_dir, f"detection_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        return output_path
    
    def annotate_segmentation(self, image_path: str) -> str:
        """Enhanced OpenCV-based segmentation with edge detection for metallic objects"""
        results_dir = os.path.abspath(os.path.join(self.results_dir, "pre_annotation"))
        os.makedirs(results_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        annotated_img = img.copy()
        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width
        
        # Use enhanced edge detection for segmentation too
        valid_contours = self._edge_based_detection(img, results_dir, image_path)
        
        # If insufficient contours, try adaptive thresholding
        if len(valid_contours) < 1:
            valid_contours.extend(self._adaptive_threshold_detection(img, img_area, results_dir, image_path))
        
        # If still insufficient, try color-based segmentation
        if len(valid_contours) < 1:
            valid_contours.extend(self._color_based_segmentation(img, img_area))
        
        # Filter and merge for segmentation
        valid_contours = self._filter_and_merge_contours(valid_contours, img_area)
        
        # Draw detailed contours (segmentation style) with more precise contours
        for i, c in enumerate(valid_contours[:3]):
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i % 3]
            
            # For segmentation, we want more precise contours
            # Approximate contour to reduce noise while preserving shape
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx_contour = cv2.approxPolyDP(c, epsilon, True)
            
            # Draw filled contour with transparency
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [approx_contour], 255)
            
            # Create colored overlay
            colored_mask = np.zeros_like(img)
            colored_mask[mask == 255] = color
            annotated_img = cv2.addWeighted(annotated_img, 0.7, colored_mask, 0.3, 0)
            
            # Draw precise contour outline
            cv2.drawContours(annotated_img, [approx_contour], -1, color, 2)
            
            # Add label with area information
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            label = f"Segment {i+1} ({int(area)}px)"
            cv2.putText(annotated_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add segmentation summary
        summary = f"Segmentation: {len(valid_contours)} segments found"
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        output_path = os.path.join(results_dir, f"segmentation_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        return output_path
    
    def _color_based_detection(self, img, img_area):
        """Helper method for color-based object detection"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-blue areas (metallic objects typically not blue)
        lower_blue = np.array([90, 70, 70])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        non_blue_mask = cv2.bitwise_not(blue_mask)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(non_blue_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if 1000 < cv2.contourArea(c) < 0.8 * img_area]
    
    def _color_based_segmentation(self, img, img_area):
        """Helper method for color-based segmentation"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # More refined color segmentation
        lower_blue = np.array([90, 70, 70])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        non_blue_mask = cv2.bitwise_not(blue_mask)
        
        # Finer morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(non_blue_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours with better approximation
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return [c for c in contours if 1000 < cv2.contourArea(c) < 0.8 * img_area]
    
    def _edge_based_detection(self, img, results_dir, image_path):
        """Enhanced edge-based detection for metallic objects with varying backgrounds"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle different lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Save enhanced image for debugging
        enhanced_debug = os.path.join(results_dir, f"debug_enhanced_{os.path.basename(image_path)}")
        cv2.imwrite(enhanced_debug, enhanced)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Use Canny edge detection with adaptive thresholds
        # Calculate dynamic thresholds based on image statistics
        v = np.median(bilateral)
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        
        edges = cv2.Canny(bilateral, lower_thresh, upper_thresh)
        
        # Save edges for debugging
        edges_debug = os.path.join(results_dir, f"debug_edges_{os.path.basename(image_path)}")
        cv2.imwrite(edges_debug, edges)
        
        # Dilate edges to connect nearby edge pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Close gaps in edges
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close)
        
        # Save processed edges for debugging
        edges_processed_debug = os.path.join(results_dir, f"debug_edges_processed_{os.path.basename(image_path)}")
        cv2.imwrite(edges_processed_debug, edges_closed)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = img.shape[0] * img.shape[1]
        valid_contours = []
        
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            # Filter by area (not too small, not too large)
            if area < 1500 or area > 0.7 * img_area:
                continue
                
            # Filter by aspect ratio and solidity for metallic objects
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Metallic objects usually have reasonable aspect ratios and decent solidity
            if 0.1 < aspect_ratio < 10 and solidity > 0.3:
                valid_contours.append(c)
        
        return valid_contours
    
    def _adaptive_threshold_detection(self, img, img_area, results_dir, image_path):
        """Adaptive thresholding for different lighting conditions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Use adaptive threshold to handle varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Save adaptive threshold for debugging
        adaptive_debug = os.path.join(results_dir, f"debug_adaptive_{os.path.basename(image_path)}")
        cv2.imwrite(adaptive_debug, adaptive_thresh)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if 1200 < area < 0.6 * img_area:
                # Additional filtering for shape characteristics
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                if 0.1 < aspect_ratio < 8:  # Reasonable aspect ratio
                    valid_contours.append(c)
        
        return valid_contours
    
    def _filter_and_merge_contours(self, contours, img_area):
        """Filter and merge overlapping contours"""
        if not contours:
            return []
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        merged_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip if too small or too large
            if area < 800 or area > 0.8 * img_area:
                continue
            
            # Check if this contour overlaps significantly with existing ones
            is_duplicate = False
            current_rect = cv2.boundingRect(contour)
            
            for existing in merged_contours:
                existing_rect = cv2.boundingRect(existing)
                
                # Calculate intersection over union (IoU)
                iou = self._calculate_bbox_iou(current_rect, existing_rect)
                if iou > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_contours.append(contour)
                
                # Limit to avoid too many detections
                if len(merged_contours) >= 5:
                    break
        
        return merged_contours
    
    def _calculate_bbox_iou(self, box1, box2):
        """Calculate Intersection over Union for bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0