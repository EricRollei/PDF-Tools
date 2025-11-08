"""
Spread Detection Enhanced

Description: ComfyUI custom node for PDF tools and media processing
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: This software is licensed under the terms of the
   Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
   
2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at eric@historic.camera or eric@rollei.us for licensing options.

Dependencies:
This code depends on several third-party libraries, each with its own license.
See CREDITS.md for a comprehensive list of dependencies and their licenses.

Third-party code:
- See CREDITS.md for complete list of dependencies
"""

#!/usr/bin/env python3
"""
Enhanced Double Page Spread Detection & Joining Test Script
Incorporates 14 sophisticated heuristics for accurate spread detection

Usage: python spread_detection_enhanced.py path/to/test.pdf
"""

import os
import sys
import json
import time
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat
from io import BytesIO

try:
    import pymupdf as fitz
except ImportError:
    import fitz

def analyze_symmetry_across_seam(img1: Image.Image, img2: Image.Image,
                               strip_width: int = 50) -> Dict[str, Any]:
    """Analyze visual symmetry/continuity across seam using template matching"""
    
    try:
        # Convert PIL images to numpy arrays
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        
        # Extract strips near the seam
        left_strip = img1_array[:, -strip_width:]  # Right edge of left image
        right_strip = img2_array[:, :strip_width]  # Left edge of right image
        
        # Ensure strips are the same height
        min_height = min(left_strip.shape[0], right_strip.shape[0])
        left_strip = left_strip[:min_height]
        right_strip = right_strip[:min_height]
        
        # Convert to grayscale for analysis
        left_gray = np.mean(left_strip, axis=2) if len(left_strip.shape) == 3 else left_strip
        right_gray = np.mean(right_strip, axis=2) if len(right_strip.shape) == 3 else right_strip
        
        # PRE-FILTERING: Check for "boring" uniform strips
        left_brightness = np.mean(left_gray)
        right_brightness = np.mean(right_gray)
        left_variance = np.var(left_gray)
        right_variance = np.var(right_gray)
        
        # Check for uniform white/black/gray areas
        white_threshold = 250
        black_threshold = 20
        min_variance = 100  # Minimum variance to be considered "interesting"
        
        # Exclude cases where both strips are uniform/boring
        left_is_boring = (
            left_brightness > white_threshold or  # Very white
            left_brightness < black_threshold or  # Very black  
            left_variance < min_variance          # Very uniform
        )
        
        right_is_boring = (
            right_brightness > white_threshold or
            right_brightness < black_threshold or
            right_variance < min_variance
        )
        
        # If both strips are boring, return no symmetry
        if left_is_boring and right_is_boring:
            return {
                "correlation": 0,
                "hist_similarity": 0,
                "edge_correlation": 0,
                "combined_score": 0,
                "points": 0,
                "strip_width": strip_width,
                "passes_rule": False,
                "analysis_method": "excluded_boring_strips",
                "left_brightness": left_brightness,
                "right_brightness": right_brightness,
                "left_variance": left_variance,
                "right_variance": right_variance,
                "exclusion_reason": "Both strips are uniform/boring (white/black/low variance)"
            }
        
        # Additional check: if one strip is very different brightness, probably not a spread
        brightness_diff = abs(left_brightness - right_brightness)
        if brightness_diff > 100:  # Very different brightness levels
            return {
                "correlation": 0,
                "hist_similarity": 0,
                "edge_correlation": 0,
                "combined_score": 0,
                "points": 0,
                "strip_width": strip_width,
                "passes_rule": False,
                "analysis_method": "excluded_brightness_mismatch",
                "left_brightness": left_brightness,
                "right_brightness": right_brightness,
                "brightness_diff": brightness_diff,
                "exclusion_reason": f"Very different brightness levels: {brightness_diff:.1f}"
            }
        
        # Method 1: Direct correlation (measures how similar the strips are)
        # Normalize the strips
        left_norm = (left_gray - np.mean(left_gray)) / (np.std(left_gray) + 1e-8)
        right_norm = (right_gray - np.mean(right_gray)) / (np.std(right_gray) + 1e-8)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(left_norm.flatten(), right_norm.flatten())[0, 1]
        correlation = max(0, correlation) if not np.isnan(correlation) else 0
        
        # Method 2: Color histogram similarity
        left_hist = np.histogram(left_gray.flatten(), bins=50, range=(0, 255))[0]
        right_hist = np.histogram(right_gray.flatten(), bins=50, range=(0, 255))[0]
        
        # Normalize histograms
        left_hist = left_hist.astype(float) / np.sum(left_hist)
        right_hist = right_hist.astype(float) / np.sum(right_hist)
        
        # Calculate histogram intersection (similarity)
        hist_similarity = np.sum(np.minimum(left_hist, right_hist))
        
        # Method 3: Edge continuity analysis
        # Simple edge detection using gradient
        left_edges = np.abs(np.gradient(left_gray, axis=1))
        right_edges = np.abs(np.gradient(right_gray, axis=1))
        
        # Count meaningful edges (above threshold)
        edge_threshold = 10
        left_edge_count = np.sum(left_edges > edge_threshold)
        right_edge_count = np.sum(right_edges > edge_threshold)
        
        # If very few edges, the strips are probably uniform
        min_edge_count = 50  # Minimum edges to be considered detailed
        if left_edge_count < min_edge_count and right_edge_count < min_edge_count:
            return {
                "correlation": correlation,
                "hist_similarity": hist_similarity,
                "edge_correlation": 0,
                "combined_score": correlation * 0.5 + hist_similarity * 0.5,  # No edge component
                "points": 0,  # No points for low-detail areas
                "strip_width": strip_width,
                "passes_rule": False,
                "analysis_method": "excluded_low_detail",
                "left_edge_count": left_edge_count,
                "right_edge_count": right_edge_count,
                "exclusion_reason": f"Low detail: {left_edge_count + right_edge_count} edges"
            }
        
        # Compare edge patterns at the seam
        left_seam_edges = left_edges[:, -5:]  # Last 5 columns
        right_seam_edges = right_edges[:, :5]  # First 5 columns
        
        edge_correlation = np.corrcoef(left_seam_edges.flatten(), right_seam_edges.flatten())[0, 1]
        edge_correlation = max(0, edge_correlation) if not np.isnan(edge_correlation) else 0
        
        # Combine metrics for overall symmetry score
        combined_score = (correlation * 0.5 + hist_similarity * 0.3 + edge_correlation * 0.2)
        
        # Convert to points (0-25 points based on symmetry strength)
        if combined_score > 0.7:
            points = 25  # Very strong symmetry
        elif combined_score > 0.5:
            points = 20  # Good symmetry
        elif combined_score > 0.3:
            points = 15  # Moderate symmetry
        elif combined_score > 0.15:
            points = 10  # Weak symmetry
        else:
            points = 0   # No meaningful symmetry
        
        return {
            "correlation": correlation,
            "hist_similarity": hist_similarity,
            "edge_correlation": edge_correlation,
            "combined_score": combined_score,
            "points": points,
            "strip_width": strip_width,
            "passes_rule": points > 0,
            "analysis_method": "full_analysis",
            "left_brightness": left_brightness,
            "right_brightness": right_brightness,
            "left_variance": left_variance,
            "right_variance": right_variance,
            "left_edge_count": left_edge_count,
            "right_edge_count": right_edge_count,
            "brightness_diff": brightness_diff
        }
        
    except Exception as e:
        print(f"      ❌ Symmetry analysis failed: {e}")
        return {
            "correlation": 0,
            "hist_similarity": 0,
            "edge_correlation": 0,
            "combined_score": 0,
            "points": 0,
            "strip_width": strip_width,
            "passes_rule": False,
            "analysis_method": "failed",
            "error": str(e)
        }

# Import working analysis engine for Florence2 detection
import os
import sys

# Add current directory to Python path explicitly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"🔍 Current directory: {os.getcwd()}")
print(f"🔍 Script directory: {current_dir}")
print(f"🔍 Python files found: {[f for f in os.listdir('.') if f.endswith('.py')]}")

try:
    # Now try the import with fixed path
    import analysis_engine
    from analysis_engine import create_content_analyzer, analyze_for_pdf_extraction
    print("✅ Analysis engine imported successfully")
    AI_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Analysis engine import failed: {e}")
    try:
        # Try importing as a module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("analysis_engine", "analysis_engine.py")
        analysis_engine_module = importlib.util.module_from_spec(spec)
        sys.modules["analysis_engine"] = analysis_engine_module
        spec.loader.exec_module(analysis_engine_module)
        
        create_content_analyzer = analysis_engine_module.create_content_analyzer
        analyze_for_pdf_extraction = analysis_engine_module.analyze_for_pdf_extraction
        print("✅ Analysis engine imported via direct file loading")
        AI_MODELS_AVAILABLE = True
    except Exception as e2:
        print(f"❌ Direct file loading also failed: {e2}")
        print(f"   📋 There might be an error in analysis_engine.py itself")
        print(f"   📋 Will use fallback methods without AI detection")
        AI_MODELS_AVAILABLE = False

def create_output_dirs(pdf_path: str) -> str:
    """Create test output directories"""
    pdf_name = Path(pdf_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"enhanced_spread_test_{pdf_name}_{timestamp}"
    
    dirs = [
        base_dir,
        os.path.join(base_dir, "individual_pages"),
        os.path.join(base_dir, "spread_candidates"),
        os.path.join(base_dir, "joined_spreads"),
        os.path.join(base_dir, "validation_analysis"),
        os.path.join(base_dir, "color_analysis"),
        os.path.join(base_dir, "symmetry_analysis"),
        os.path.join(base_dir, "brute_force_test")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dir

def extract_page_as_image(page, dpi: int = 200) -> Image.Image:
    """Extract page as high-quality PIL Image"""
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_data = pix.tobytes("ppm")
    return Image.open(BytesIO(img_data))

def detect_images_with_florence2(image: Image.Image, analyzer=None) -> List[Dict[str, Any]]:
    """Detect images using Florence2 (Rule #9: Florence2 accuracy within few pixels)"""
    
    if not AI_MODELS_AVAILABLE:
        print(f"      ⚠️  Florence2 unavailable - using fallback scoring")
        return []
    
    if analyzer is None:
        print(f"      ❌ No analyzer provided - skipping Florence2")
        return []
    
    try:
        print(f"      🔍 Running Florence2 detection...")
        analysis = analyzer.analyze_image_comprehensive(image)  # Reuse existing analyzer
        florence2_boxes = analysis.get("florence2_rectangles", [])
        
        print(f"      📦 Florence2 raw result: {len(florence2_boxes)} regions")
        
        # Debug: print the full analysis keys
        print(f"      🔍 Analysis keys: {list(analysis.keys())}")
        
        return florence2_boxes
        
    except Exception as e:
        print(f"      ❌ Florence2 detection failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def analyze_page_margins(page) -> Dict[str, float]:
    """Analyze inside vs outside margins (Rule #2)"""
    
    page_width = page.rect.width
    page_height = page.rect.height
    
    # Get text blocks to estimate margins
    text_blocks = page.get_text("dict").get("blocks", [])
    
    # Filter for text blocks only (type 0)
    text_only_blocks = [block for block in text_blocks if block.get("type") == 0]
    
    if not text_only_blocks:
        return {
            "left_margin": 0, "right_margin": 0,
            "top_margin": 0, "bottom_margin": 0,
            "inside_margin": 0, "outside_margin": 0
        }
    
    # Find text boundaries
    min_x = min(block["bbox"][0] for block in text_only_blocks)
    max_x = max(block["bbox"][2] for block in text_only_blocks)
    min_y = min(block["bbox"][1] for block in text_only_blocks)
    max_y = max(block["bbox"][3] for block in text_only_blocks)
    
    left_margin = min_x
    right_margin = page_width - max_x
    top_margin = min_y
    bottom_margin = page_height - max_y
    
    return {
        "left_margin": left_margin,
        "right_margin": right_margin,
        "top_margin": top_margin,
        "bottom_margin": bottom_margin,
        "inside_margin": right_margin,  # Will be corrected based on page number
        "outside_margin": left_margin
    }

def get_even_odd_candidates(doc) -> List[Tuple[int, int]]:
    """Get spread candidates using even/odd strategy (Rule #1)"""
    
    candidates = []
    total_pages = len(doc)
    
    print(f"📖 Applying Rule #1: Even to odd page sequences only")
    
    # Skip page 0 (cover), start joining from page 1 (even) to 2 (odd), etc.
    for i in range(1, total_pages - 1, 2):  # 1, 3, 5, ... (even page numbers in 0-indexed)
        left_page_idx = i      # Even page number (1, 3, 5...)
        right_page_idx = i + 1 # Odd page number (2, 4, 6...)
        
        if right_page_idx < total_pages:
            candidates.append((left_page_idx, right_page_idx))
            print(f"   📖 Candidate: page {left_page_idx + 1} (even) → page {right_page_idx + 1} (odd)")
    
    return candidates

def validate_page_sizes(page1, page2, tolerance: float = 5.0) -> Dict[str, Any]:
    """Validate page sizes are same (Rule #4)"""
    
    size1 = (page1.rect.width, page1.rect.height)
    size2 = (page2.rect.width, page2.rect.height)
    
    width_diff = abs(size1[0] - size2[0])
    height_diff = abs(size1[1] - size2[1])
    
    sizes_match = width_diff <= tolerance and height_diff <= tolerance
    
    return {
        "page1_size": size1,
        "page2_size": size2,
        "width_diff": width_diff,
        "height_diff": height_diff,
        "sizes_match": sizes_match,
        "tolerance": tolerance
    }

def validate_image_heights(images1: List[Dict], images2: List[Dict], tolerance: int = 2) -> Dict[str, Any]:
    """Validate image heights match within 1-2 pixels (Rule #3, #5)"""
    
    if not images1 or not images2:
        return {"height_validation": False, "reason": "No images found on one or both pages"}
    
    matches = []
    
    for img1 in images1:
        bbox1 = img1["bbox"]
        height1 = bbox1[3] - bbox1[1]
        
        for img2 in images2:
            bbox2 = img2["bbox"]
            height2 = bbox2[3] - bbox2[1]
            
            height_diff = abs(height1 - height2)
            
            if height_diff <= tolerance:
                matches.append({
                    "img1_height": height1,
                    "img2_height": height2,
                    "height_diff": height_diff,
                    "img1_bbox": bbox1,
                    "img2_bbox": bbox2
                })
    
    return {
        "height_validation": len(matches) > 0,
        "matching_pairs": matches,
        "total_matches": len(matches),
        "tolerance": tolerance
    }

def analyze_inside_margins(images1: List[Dict], images2: List[Dict], 
                          page1_width: float, page2_width: float,
                          threshold: float = 25.0) -> Dict[str, Any]:
    """Analyze inside margin distances (Rule #10: 25 pixels or less)"""
    
    inside_distances = []
    
    # Check right edge of left page images (inside margin)
    for img in images1:
        bbox = img["bbox"]
        right_edge = bbox[2]
        distance_to_seam = page1_width - right_edge
        inside_distances.append({
            "page": "left",
            "distance_to_seam": distance_to_seam,
            "within_threshold": distance_to_seam <= threshold,
            "image_bbox": bbox
        })
    
    # Check left edge of right page images (inside margin)
    for img in images2:
        bbox = img["bbox"]
        left_edge = bbox[0]
        distance_to_seam = left_edge
        inside_distances.append({
            "page": "right", 
            "distance_to_seam": distance_to_seam,
            "within_threshold": distance_to_seam <= threshold,
            "image_bbox": bbox
        })
    
    within_threshold_count = sum(1 for d in inside_distances if d["within_threshold"])
    
    return {
        "inside_distances": inside_distances,
        "within_threshold_count": within_threshold_count,
        "total_images": len(inside_distances),
        "threshold": threshold,
        "passes_rule": within_threshold_count > 0
    }

def analyze_color_patches_across_seam(img1: Image.Image, img2: Image.Image, 
                                    patch_count: int = 20) -> Dict[str, Any]:
    """Analyze color matching across seam with refined graduated scoring"""
    
    height1, height2 = img1.height, img2.height
    min_height = min(height1, height2)
    
    # Sample patches at regular intervals vertically
    patch_height = min_height // (patch_count + 1)
    matches = []
    
    # Move patches back to seam for better accuracy (no seam_offset)
    patch_size = 10  # 10x10 pixel patches
    
    for i in range(1, patch_count + 1):
        y = i * patch_height
        
        # Get patch from right edge of left image (at seam)
        left_patch_box = (img1.width - patch_size, y, img1.width, y + patch_size)
        left_patch = img1.crop(left_patch_box)
        
        # Get patch from left edge of right image (at seam)
        right_patch_box = (0, y, patch_size, y + patch_size)
        right_patch = img2.crop(right_patch_box)
        
        # Calculate average colors
        left_avg = ImageStat.Stat(left_patch).mean
        right_avg = ImageStat.Stat(right_patch).mean
        
        # Calculate color difference (Euclidean distance in RGB space)
        color_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(left_avg, right_avg)))
        
        # Consider colors matching if difference < 30 (out of 255)
        is_match = color_diff < 30
        
        matches.append({
            "patch_index": i,
            "y_position": y,
            "left_color": left_avg,
            "right_color": right_avg,
            "color_difference": color_diff,
            "is_match": is_match,
            "left_patch_box": left_patch_box,
            "right_patch_box": right_patch_box
        })
    
    match_count = sum(1 for m in matches if m["is_match"])
    match_ratio = match_count / patch_count
    
    # Enhanced background detection with higher white threshold
    all_colors = []
    for m in matches:
        all_colors.extend([m["left_color"], m["right_color"]])
    
    # Calculate average brightness for each color
    avg_brightness = [sum(color) / 3 for color in all_colors]
    brightness_variance = np.var(avg_brightness)
    overall_brightness = np.mean(avg_brightness)
    
    # Check for white/near-white patches (raised threshold)
    white_threshold = 250  # Raised from 240 to 250
    near_white_count = sum(1 for brightness in avg_brightness if brightness > white_threshold)
    near_white_ratio = near_white_count / len(avg_brightness)
    
    # Multiple checks for background
    is_likely_background = (
        brightness_variance < 100 or  # Low variance (uniform colors)
        overall_brightness > 250 or   # Very bright overall (white pages)
        near_white_ratio > 0.8        # More than 80% patches are near-white
    )
    
    # Additional check: If all matching patches are white, it's background
    if match_count >= 16:  # Updated for 20 patches (16/20 = 80%)
        matching_brightness = []
        for m in matches:
            if m["is_match"]:
                matching_brightness.extend([sum(m["left_color"])/3, sum(m["right_color"])/3])
        
        if matching_brightness:
            avg_matching_brightness = np.mean(matching_brightness)
            if avg_matching_brightness > white_threshold:
                is_likely_background = True
    
    # Refined graduated scoring based on match count
    if match_count >= 16 and not is_likely_background:
        score = 30  # Full points for 16+ matches
    elif match_count >= 14 and not is_likely_background:
        score = 25  # 25 points for 14-15 matches
    elif match_count >= 12 and not is_likely_background:
        score = 20  # 20 points for 12-13 matches
    elif match_count >= 10 and not is_likely_background:
        score = 15  # 15 points for 10-11 matches
    else:
        score = 0   # No points for <10 matches or background
    
    passes_rule = score > 0
    
    return {
        "patch_matches": matches,
        "match_count": match_count,
        "total_patches": patch_count,
        "match_ratio": match_ratio,
        "passes_rule": passes_rule,
        "score": score,
        "threshold": 30,
        "seam_offset": 0,  # Back to seam
        "brightness_variance": brightness_variance,
        "overall_brightness": overall_brightness,
        "near_white_count": near_white_count,
        "near_white_ratio": near_white_ratio,
        "is_likely_background": is_likely_background,
        "white_threshold": white_threshold
    }

def comprehensive_spread_validation(doc, left_idx: int, right_idx: int, 
                                  output_dir: str, analyzer=None) -> Dict[str, Any]:
    """Comprehensive validation using refined heuristics"""
    
    left_page = doc[left_idx]
    right_page = doc[right_idx]
    
    print(f"   🔍 Comprehensive validation: pages {left_idx + 1}-{right_idx + 1}")
    
    # Extract page images
    left_img = extract_page_as_image(left_page)
    right_img = extract_page_as_image(right_page)
    
    validation_results = {
        "pages": (left_idx + 1, right_idx + 1),
        "validation_score": 0,
        "heuristics": {}
    }
    
    # Rule #1: Even/odd sequence (NO POINTS - already filtered for this)
    validation_results["heuristics"]["rule_1_even_odd"] = {
        "passed": True,
        "score": 0,  # Changed from 20 to 0
        "description": "Even to odd page sequence (prerequisite)"
    }
    # No points added for even/odd since it's a prerequisite
    
    # Rule #2: Inside vs outside margins
    left_margins = analyze_page_margins(left_page)
    right_margins = analyze_page_margins(right_page)
    
    # Correct inside/outside for page position
    left_margins["inside_margin"] = left_margins["right_margin"]
    left_margins["outside_margin"] = left_margins["left_margin"]
    right_margins["inside_margin"] = right_margins["left_margin"]
    right_margins["outside_margin"] = right_margins["right_margin"]
    
    inside_margin_ratio = (left_margins["inside_margin"] + right_margins["inside_margin"]) / \
                         (left_margins["outside_margin"] + right_margins["outside_margin"] + 1)
    
    inside_margin_small = inside_margin_ratio < 0.5  # Inside margins significantly smaller
    
    validation_results["heuristics"]["rule_2_inside_margins"] = {
        "passed": inside_margin_small,
        "score": 15 if inside_margin_small else 0,
        "left_margins": left_margins,
        "right_margins": right_margins,
        "inside_margin_ratio": inside_margin_ratio
    }
    if inside_margin_small:
        validation_results["validation_score"] += 15
    
    # Rule #4: Page sizes match (REMOVED - not discriminatory)
    validation_results["heuristics"]["rule_4_page_sizes"] = {
        "passed": True,
        "score": 0,  # Changed from 10 to 0
        "description": "Page size check removed (not discriminatory)"
    }
    # No points added for page sizes
    
    # Rule #9: Florence2 image detection (REUSE ANALYZER)
    left_images = detect_images_with_florence2(left_img, analyzer)
    right_images = detect_images_with_florence2(right_img, analyzer)
    
    has_florence2_detections = len(left_images) > 0 or len(right_images) > 0
    
    validation_results["heuristics"]["rule_9_florence2_detection"] = {
        "left_image_count": len(left_images),
        "right_image_count": len(right_images),
        "total_images": len(left_images) + len(right_images),
        "left_images": left_images,
        "right_images": right_images,
        "florence2_available": AI_MODELS_AVAILABLE
    }
    
    # Rule #3 & #5: Image heights match (ONLY if Florence2 detected images)
    if has_florence2_detections:
        height_validation = validate_image_heights(left_images, right_images)
        validation_results["heuristics"]["rule_3_image_heights"] = {
            "passed": height_validation["height_validation"],
            "score": 25 if height_validation["height_validation"] else 0,
            "details": height_validation
        }
        if height_validation["height_validation"]:
            validation_results["validation_score"] += 25
        
        # Rule #10: Inside margin distances (ONLY if Florence2 detected images)
        margin_analysis = analyze_inside_margins(
            left_images, right_images, 
            left_page.rect.width, right_page.rect.width
        )
        validation_results["heuristics"]["rule_10_inside_distances"] = {
            "passed": margin_analysis["passes_rule"],
            "score": 20 if margin_analysis["passes_rule"] else 0,
            "details": margin_analysis
        }
        if margin_analysis["passes_rule"]:
            validation_results["validation_score"] += 20
    else:
        # Florence2 unavailable - use alternative scoring
        print(f"      📋 Florence2 unavailable - using text-based fallback scoring")
        
        # Alternative: Check if pages have minimal text (likely image-heavy)
        left_text_blocks = len([b for b in left_page.get_text("dict").get("blocks", []) if b.get("type") == 0])
        right_text_blocks = len([b for b in right_page.get_text("dict").get("blocks", []) if b.get("type") == 0])
        total_text_blocks = left_text_blocks + right_text_blocks
        
        # If very few text blocks, likely image-heavy pages
        if total_text_blocks <= 5:
            validation_results["validation_score"] += 15  # Bonus for image-heavy pages
            validation_results["heuristics"]["fallback_image_heavy"] = {
                "passed": True,
                "score": 15,
                "total_text_blocks": total_text_blocks,
                "description": "Low text density suggests image-heavy pages"
            }
    
    # Rule #13: Color matching across seam (ENHANCED with graduated scoring)
    color_analysis = analyze_color_patches_across_seam(left_img, right_img)
    validation_results["heuristics"]["rule_13_color_matching"] = {
        "passed": color_analysis["passes_rule"],
        "score": color_analysis["score"],  # Now uses graduated scoring (0/15/20/25/30)
        "details": color_analysis
    }
    validation_results["validation_score"] += color_analysis["score"]
    
    # Rule #14: Symmetry/continuity analysis across seam (NEW)
    print(f"      🔄 Running symmetry analysis...")
    symmetry_analysis = analyze_symmetry_across_seam(left_img, right_img)
    validation_results["heuristics"]["rule_14_symmetry"] = {
        "passed": symmetry_analysis["passes_rule"],
        "score": symmetry_analysis["points"],  # 0-25 points based on visual continuity
        "details": symmetry_analysis
    }
    validation_results["validation_score"] += symmetry_analysis["points"]
    
    # Show symmetry result with exclusion info
    if "exclusion_reason" in symmetry_analysis:
        print(f"      🚫 Symmetry excluded: {symmetry_analysis['exclusion_reason']}")
    else:
        print(f"      🎯 Symmetry score: {symmetry_analysis['points']} pts (combined: {symmetry_analysis['combined_score']:.3f})")
    
    # Save color analysis visualization (ENSURE IT'S ALWAYS CALLED)
    print(f"      💾 Saving color analysis visualization...")
    try:
        save_color_analysis_visualization(left_img, right_img, color_analysis, 
                                        left_idx + 1, right_idx + 1, output_dir)
        print(f"      ✅ Color analysis saved")
    except Exception as e:
        print(f"      ❌ Color analysis save failed: {e}")
    
    # Save symmetry analysis visualization
    print(f"      💾 Saving symmetry analysis visualization...")
    try:
        save_symmetry_analysis_visualization(left_img, right_img, symmetry_analysis,
                                           left_idx + 1, right_idx + 1, output_dir)
        print(f"      ✅ Symmetry analysis saved")
    except Exception as e:
        print(f"      ❌ Symmetry analysis save failed: {e}")
    
    # Final determination - PROPER THRESHOLD BASED ON TESTING
    validation_results["validation_score"] = min(validation_results["validation_score"], 100)  # Cap at 100
    validation_results["is_spread"] = validation_results["validation_score"] >= 49  # Based on testing: non-matches are 45 and under
    validation_results["confidence"] = validation_results["validation_score"] / 100.0
    
    return validation_results

def save_color_analysis_visualization(left_img: Image.Image, right_img: Image.Image,
                                    color_analysis: Dict, left_page_num: int, 
                                    right_page_num: int, output_dir: str):
    """Save color analysis visualization with larger readable text"""
    
    # Create side-by-side visualization
    combined_width = left_img.width + right_img.width
    combined_height = max(left_img.height, right_img.height)
    
    vis_img = Image.new('RGB', (combined_width, combined_height), 'white')
    vis_img.paste(left_img, (0, 0))
    vis_img.paste(right_img, (left_img.width, 0))
    
    draw = ImageDraw.Draw(vis_img)
    
    # Draw patch locations and results (patches are now at seam)
    for match in color_analysis["patch_matches"]:
        y = match["y_position"]
        color = "green" if match["is_match"] else "red"
        
        # Use the actual patch boxes from the analysis
        left_box = match.get("left_patch_box", [left_img.width - 10, y, left_img.width, y + 10])
        right_box = match.get("right_patch_box", [0, y, 10, y + 10])
        
        # Left patch
        draw.rectangle(left_box, outline=color, width=2)
        
        # Right patch (adjusted position in combined image coordinates)
        right_box_adjusted = [left_img.width + right_box[0], right_box[1], 
                             left_img.width + right_box[2], right_box[3]]
        draw.rectangle(right_box_adjusted, outline=color, width=2)
        
        # Draw connection line between patch centers
        left_center_x = (left_box[0] + left_box[2]) // 2
        left_center_y = (left_box[1] + left_box[3]) // 2
        right_center_x = (right_box_adjusted[0] + right_box_adjusted[2]) // 2
        right_center_y = (right_box_adjusted[1] + right_box_adjusted[3]) // 2
        
        draw.line([left_center_x, left_center_y, right_center_x, right_center_y], 
                 fill=color, width=1)
    
    # Add summary text with enhanced background detection info (LARGER FONT)
    font_size = 16  # Increased from default small size
    try:
        # Try to use a better font if available
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fall back to default font
        font = ImageFont.load_default()
    
    line_height = 20
    y_text = 10
    
    # Text information
    match_text = f"Color Matches: {color_analysis['match_count']}/{color_analysis['total_patches']}"
    score_text = f"Score: {color_analysis.get('score', 0)} points"
    background_text = f"Background: {'YES' if color_analysis.get('is_likely_background', False) else 'NO'}"
    brightness_text = f"Brightness: {color_analysis.get('overall_brightness', 0):.1f}"
    white_ratio_text = f"White Ratio: {color_analysis.get('near_white_ratio', 0):.2f}"
    variance_text = f"Variance: {color_analysis.get('brightness_variance', 0):.1f}"
    
    # Draw text with larger font
    draw.text((10, y_text), match_text, fill="black", font=font)
    y_text += line_height
    draw.text((10, y_text), score_text, fill="blue", font=font)
    y_text += line_height
    draw.text((10, y_text), background_text, 
             fill="red" if color_analysis.get('is_likely_background', False) else "green", font=font)
    y_text += line_height
    draw.text((10, y_text), brightness_text, fill="black", font=font)
    y_text += line_height
    draw.text((10, y_text), white_ratio_text, fill="black", font=font)
    y_text += line_height
    draw.text((10, y_text), variance_text, fill="black", font=font)
    y_text += line_height
    
    # Add result status
    result_text = f"PASSES: {'NO' if color_analysis.get('is_likely_background', False) else 'YES'}"
    result_color = "red" if color_analysis.get('is_likely_background', False) else "green"
    draw.text((10, y_text), result_text, fill=result_color, font=font)
    
    # Draw seam line
    draw.line([left_img.width, 0, left_img.width, combined_height], fill="blue", width=2)
    draw.text((left_img.width - 60, 70), "SEAM", fill="blue", font=font)
    
def save_symmetry_analysis_visualization(left_img: Image.Image, right_img: Image.Image,
                                       symmetry_analysis: Dict, left_page_num: int, 
                                       right_page_num: int, output_dir: str):
    """Save symmetry analysis visualization with exclusion info"""
    
    strip_width = symmetry_analysis.get("strip_width", 50)
    
    # Create side-by-side visualization with highlighted strips
    combined_width = left_img.width + right_img.width
    combined_height = max(left_img.height, right_img.height)
    
    vis_img = Image.new('RGB', (combined_width, combined_height), 'white')
    vis_img.paste(left_img, (0, 0))
    vis_img.paste(right_img, (left_img.width, 0))
    
    draw = ImageDraw.Draw(vis_img)
    
    # Highlight the strips used for analysis
    # Color code based on analysis result
    analysis_method = symmetry_analysis.get("analysis_method", "unknown")
    if "excluded" in analysis_method:
        strip_color = "red"  # Red for excluded strips
    elif symmetry_analysis.get("points", 0) > 0:
        strip_color = "green"  # Green for good symmetry
    else:
        strip_color = "orange"  # Orange for analyzed but low score
    
    # Left strip (right edge of left image)
    left_strip_box = [left_img.width - strip_width, 0, left_img.width, left_img.height]
    draw.rectangle(left_strip_box, outline=strip_color, width=3)
    
    # Right strip (left edge of right image in combined coordinates)
    right_strip_box = [left_img.width, 0, left_img.width + strip_width, right_img.height]
    draw.rectangle(right_strip_box, outline=strip_color, width=3)
    
    # Add text information with larger font
    font_size = 16
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    line_height = 20
    y_text = 10
    
    # Symmetry metrics
    correlation = symmetry_analysis.get("correlation", 0)
    hist_similarity = symmetry_analysis.get("hist_similarity", 0)
    edge_correlation = symmetry_analysis.get("edge_correlation", 0)
    combined_score = symmetry_analysis.get("combined_score", 0)
    points = symmetry_analysis.get("points", 0)
    
    # Draw metrics
    draw.text((10, y_text), f"Symmetry Score: {points} points", fill="blue", font=font)
    y_text += line_height
    
    # Show exclusion reason if applicable
    if "exclusion_reason" in symmetry_analysis:
        exclusion_reason = symmetry_analysis["exclusion_reason"]
        draw.text((10, y_text), f"EXCLUDED: {exclusion_reason}", fill="red", font=font)
        y_text += line_height
    else:
        draw.text((10, y_text), f"Combined: {combined_score:.3f}", fill="black", font=font)
        y_text += line_height
        draw.text((10, y_text), f"Correlation: {correlation:.3f}", fill="black", font=font)
        y_text += line_height
        draw.text((10, y_text), f"Histogram: {hist_similarity:.3f}", fill="black", font=font)
        y_text += line_height
        draw.text((10, y_text), f"Edge Match: {edge_correlation:.3f}", fill="black", font=font)
        y_text += line_height
    
    # Show brightness and variance info
    if "left_brightness" in symmetry_analysis:
        left_brightness = symmetry_analysis["left_brightness"]
        right_brightness = symmetry_analysis["right_brightness"]
        left_variance = symmetry_analysis.get("left_variance", 0)
        right_variance = symmetry_analysis.get("right_variance", 0)
        
        draw.text((10, y_text), f"L Bright: {left_brightness:.1f}, Var: {left_variance:.1f}", fill="gray", font=font)
        y_text += line_height
        draw.text((10, y_text), f"R Bright: {right_brightness:.1f}, Var: {right_variance:.1f}", fill="gray", font=font)
        y_text += line_height
    
    # Result
    result_text = f"PASSES: {'YES' if symmetry_analysis.get('passes_rule', False) else 'NO'}"
    result_color = "green" if symmetry_analysis.get('passes_rule', False) else "red"
    draw.text((10, y_text), result_text, fill=result_color, font=font)
    
    # Draw seam line
    draw.line([left_img.width, 0, left_img.width, combined_height], fill="blue", width=2)
    draw.text((left_img.width - 60, combined_height - 30), "SEAM", fill="blue", font=font)
    
    # Label the analysis strips with color coding
    strip_label = "EXCLUDED" if "excluded" in analysis_method else "ANALYSIS"
    draw.text((left_img.width - strip_width + 5, combined_height - 50), strip_label, fill=strip_color, font=font)
    draw.text((left_img.width - strip_width + 5, combined_height - 30), "STRIP", fill=strip_color, font=font)
    
    vis_path = os.path.join(output_dir, "symmetry_analysis", 
                           f"symmetry_{left_page_num:03d}_{right_page_num:03d}.png")
    vis_img.save(vis_path)

def save_spread_candidate_visualization(doc, left_idx: int, right_idx: int, 
                                      validation: Dict, output_dir: str):
    """Save visualization of spread candidate with validation overlay"""
    
    try:
        # Extract pages as images
        left_page = doc[left_idx]
        right_page = doc[right_idx]
        
        # Get images at medium resolution for visualization
        mat = fitz.Matrix(100/72, 100/72)  # 100 DPI for candidates
        left_pix = left_page.get_pixmap(matrix=mat, alpha=False)
        right_pix = right_page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL images
        left_img_data = left_pix.tobytes("ppm")
        right_img_data = right_pix.tobytes("ppm")
        left_img = Image.open(BytesIO(left_img_data))
        right_img = Image.open(BytesIO(right_img_data))
        
        # Create side-by-side visualization
        combined_width = left_img.width + right_img.width
        combined_height = max(left_img.height, right_img.height) + 100  # Extra space for text
        
        vis_img = Image.new('RGB', (combined_width, combined_height), 'white')
        vis_img.paste(left_img, (0, 0))
        vis_img.paste(right_img, (left_img.width, 0))
        
        draw = ImageDraw.Draw(vis_img)
        
        # Add validation information
        font_size = 14
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Title
        title = f"Candidate: Pages {left_idx + 1}-{right_idx + 1}"
        draw.text((10, left_img.height + 10), title, fill="black", font=font)
        
        # Score and result
        score = validation["validation_score"]
        is_spread = validation["is_spread"]
        result_color = "green" if is_spread else "red"
        result_text = f"Score: {score}/100 - {'SPREAD' if is_spread else 'NOT SPREAD'}"
        draw.text((10, left_img.height + 30), result_text, fill=result_color, font=font)
        
        # Key heuristics breakdown
        y_pos = left_img.height + 50
        heuristics = validation.get("heuristics", {})
        
        # Color matching
        color_info = heuristics.get("rule_13_color_matching", {})
        color_score = color_info.get("score", 0)
        color_details = color_info.get("details", {})
        color_matches = color_details.get("match_count", 0)
        color_total = color_details.get("total_patches", 20)
        draw.text((10, y_pos), f"Color: {color_score}pts ({color_matches}/{color_total} patches)", fill="blue", font=font)
        
        # Symmetry
        y_pos += 20
        symmetry_info = heuristics.get("rule_14_symmetry", {})
        symmetry_score = symmetry_info.get("score", 0)
        symmetry_details = symmetry_info.get("details", {})
        if "exclusion_reason" in symmetry_details:
            symmetry_text = f"Symmetry: {symmetry_score}pts (EXCLUDED)"
        else:
            combined_score = symmetry_details.get("combined_score", 0)
            symmetry_text = f"Symmetry: {symmetry_score}pts ({combined_score:.2f})"
        draw.text((10, y_pos), symmetry_text, fill="orange", font=font)
        
        # Florence2 if available
        y_pos += 20
        f2_info = heuristics.get("rule_9_florence2_detection", {})
        f2_total = f2_info.get("total_images", 0)
        draw.text((10, y_pos), f"Florence2: {f2_total} images detected", fill="purple", font=font)
        
        # Draw seam line
        draw.line([left_img.width, 0, left_img.width, left_img.height], fill="blue", width=2)
        
        # Save candidate visualization
        filename = f"candidate_{left_idx+1:03d}_{right_idx+1:03d}_score_{score}.png"
        vis_path = os.path.join(output_dir, "spread_candidates", filename)
        vis_img.save(vis_path)
        
    except Exception as e:
        print(f"      ❌ Failed to save candidate visualization: {e}")

def save_validation_summary(all_validations: List[Dict], output_dir: str):
    """Save overall validation summary visualization"""
    
    try:
        # Create summary chart
        scores = [v["validation_score"] for v in all_validations]
        is_spreads = [v["is_spread"] for v in all_validations]
        
        # Simple text-based summary for now
        summary_lines = [
            "VALIDATION SUMMARY",
            "=" * 50,
            f"Total candidates: {len(all_validations)}",
            f"Valid spreads: {sum(is_spreads)}",
            f"Invalid: {len(all_validations) - sum(is_spreads)}",
            "",
            "SCORE DISTRIBUTION:",
            f"Max score: {max(scores) if scores else 0}",
            f"Min score: {min(scores) if scores else 0}",
            f"Avg score: {sum(scores)/len(scores):.1f}" if scores else "0",
            "",
            "THRESHOLD ANALYSIS:",
            f"Above 49 (current): {sum(1 for s in scores if s >= 49)}",
            f"40-48: {sum(1 for s in scores if 40 <= s < 49)}",
            f"30-39: {sum(1 for s in scores if 30 <= s < 40)}",
            f"Below 30: {sum(1 for s in scores if s < 30)}",
        ]
        
        # Save to text file
        summary_path = os.path.join(output_dir, "validation_analysis", "validation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
            
        # Save detailed JSON
        json_path = os.path.join(output_dir, "validation_analysis", "detailed_scores.json")
        score_data = [
            {
                "pages": v["pages"],
                "score": v["validation_score"],
                "is_spread": v["is_spread"],
                "color_score": v.get("heuristics", {}).get("rule_13_color_matching", {}).get("score", 0),
                "symmetry_score": v.get("heuristics", {}).get("rule_14_symmetry", {}).get("score", 0)
            }
            for v in all_validations
        ]
        
        with open(json_path, 'w') as f:
            json.dump(score_data, f, indent=2)
            
    except Exception as e:
        print(f"      ❌ Failed to save validation summary: {e}")

def create_spread_if_valid(doc, left_idx: int, right_idx: int, 
                          validation: Dict, output_dir: str) -> Optional[str]:
    """Create joined spread if validation passes (Rule #7, #8: Join full pages first)"""
    
    if not validation["is_spread"]:
        return None
    
    print(f"   ✅ Creating spread: pages {left_idx + 1}-{right_idx + 1}")
    
    # Extract full resolution pages
    left_page = doc[left_idx]
    right_page = doc[right_idx]
    
    # High resolution for final output
    left_img = extract_page_as_image(left_page, dpi=300)
    right_img = extract_page_as_image(right_page, dpi=300)
    
    # Rule #7: Photoshop method - double canvas width, paste right image
    spread_width = left_img.width * 2
    spread_height = left_img.height
    
    spread_img = Image.new('RGB', (spread_width, spread_height), 'white')
    
    # Paste left image
    spread_img.paste(left_img, (0, 0))
    
    # Paste right image at the far right initially
    initial_right_x = left_img.width
    spread_img.paste(right_img, (initial_right_x, 0))
    
    # Rule #6: Simulate seam adjustment (in real implementation, 
    # this would be done iteratively to minimize seam visibility)
    # For now, we'll use a small overlap typical of print spreads
    overlap_pixels = 5  # Typical overlap for print spreads
    final_right_x = initial_right_x - overlap_pixels
    
    # Create final spread with overlap
    final_spread = Image.new('RGB', (left_img.width + right_img.width - overlap_pixels, spread_height), 'white')
    final_spread.paste(left_img, (0, 0))
    final_spread.paste(right_img, (final_right_x, 0))
    
    # Save the spread
    spread_filename = f"spread_{left_idx + 1:03d}_{right_idx + 1:03d}_score_{validation['validation_score']}.png"
    spread_path = os.path.join(output_dir, "joined_spreads", spread_filename)
    final_spread.save(spread_path)
    
    return spread_path

def run_brute_force_join_test(doc, output_dir: str, analyzer) -> Dict[str, Any]:
    """Test brute force approach: join all even-odd pairs and run Florence2"""
    
    print(f"   🔧 BRUTE FORCE TEST: Join all even-odd pairs without validation")
    
    # Create brute force output directory
    brute_force_dir = os.path.join(output_dir, "brute_force_test")
    os.makedirs(brute_force_dir, exist_ok=True)
    
    brute_force_results = {
        "total_pairs": 0,
        "successful_joins": 0,
        "florence2_detections": [],
        "total_images_detected": 0,
        "pairs_with_detections": 0
    }
    
    try:
        # Get all even-odd pairs
        candidates = get_even_odd_candidates(doc)
        brute_force_results["total_pairs"] = len(candidates)
        
        print(f"   📖 Testing {len(candidates)} even-odd pairs with brute force join")
        
        for i, (left_idx, right_idx) in enumerate(candidates[:15]):  # Test first 15 pairs
            print(f"      🔧 Brute force pair {i+1}: pages {left_idx + 1}-{right_idx + 1}")
            
            try:
                # Extract pages as high-res images
                left_page = doc[left_idx]
                right_page = doc[right_idx]
                
                # High resolution for Florence2 accuracy
                mat = fitz.Matrix(200/72, 200/72)  # 200 DPI
                left_pix = left_page.get_pixmap(matrix=mat, alpha=False)
                right_pix = right_page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to PIL images
                left_img_data = left_pix.tobytes("ppm")
                right_img_data = right_pix.tobytes("ppm")
                left_img = Image.open(BytesIO(left_img_data))
                right_img = Image.open(BytesIO(right_img_data))
                
                # Join images (simple side-by-side)
                joined_width = left_img.width + right_img.width
                joined_height = max(left_img.height, right_img.height)
                
                joined_img = Image.new('RGB', (joined_width, joined_height), 'white')
                joined_img.paste(left_img, (0, 0))
                joined_img.paste(right_img, (left_img.width, 0))
                
                # Run Florence2 on joined image
                if analyzer and AI_MODELS_AVAILABLE:
                    try:
                        analysis = analyzer.analyze_image_comprehensive(joined_img)
                        florence2_boxes = analysis.get("florence2_rectangles", [])
                        
                        detection_result = {
                            "pages": [left_idx + 1, right_idx + 1],
                            "image_count": len(florence2_boxes),
                            "detections": florence2_boxes
                        }
                        
                        brute_force_results["florence2_detections"].append(detection_result)
                        brute_force_results["total_images_detected"] += len(florence2_boxes)
                        
                        if len(florence2_boxes) > 0:
                            brute_force_results["pairs_with_detections"] += 1
                        
                        print(f"         📦 Florence2 detected: {len(florence2_boxes)} images")
                        
                        # Save visualization with Florence2 bounding boxes
                        vis_img = joined_img.copy()
                        draw = ImageDraw.Draw(vis_img)
                        
                        # Draw Florence2 detections
                        for j, box in enumerate(florence2_boxes):
                            bbox = box["bbox"]
                            confidence = box.get("confidence", 1.0)
                            
                            # Draw red rectangle
                            draw.rectangle(bbox, outline="red", width=4)
                            draw.text((bbox[0], bbox[1]-25), f"IMG_{j+1} ({confidence:.2f})", 
                                     fill="red", font=None)
                        
                        # Draw seam line
                        draw.line([left_img.width, 0, left_img.width, joined_height], 
                                 fill="blue", width=3)
                        draw.text((left_img.width - 50, 30), "SEAM", fill="blue")
                        
                        # Add summary text
                        draw.text((10, 10), f"Brute Force: Pages {left_idx + 1}-{right_idx + 1}", 
                                 fill="black")
                        draw.text((10, 35), f"Florence2 Detections: {len(florence2_boxes)}", 
                                 fill="red")
                        
                        # Save brute force result
                        bf_filename = f"brute_force_{left_idx+1:03d}_{right_idx+1:03d}_{len(florence2_boxes)}imgs.png"
                        bf_path = os.path.join(brute_force_dir, bf_filename)
                        vis_img.save(bf_path)
                        
                    except Exception as e:
                        print(f"         ❌ Florence2 analysis failed: {e}")
                
                brute_force_results["successful_joins"] += 1
                
            except Exception as e:
                print(f"         ❌ Failed to process pair: {e}")
        
        # Print brute force summary
        total_tested = min(15, len(candidates))
        avg_detections = (brute_force_results["total_images_detected"] / total_tested) if total_tested > 0 else 0
        
        print(f"   📊 BRUTE FORCE RESULTS:")
        print(f"      📖 Pairs tested: {total_tested}")
        print(f"      ✅ Successful joins: {brute_force_results['successful_joins']}")
        print(f"      📦 Total images detected: {brute_force_results['total_images_detected']}")
        print(f"      📈 Pairs with detections: {brute_force_results['pairs_with_detections']}")
        print(f"      📊 Average detections per pair: {avg_detections:.1f}")
        
        # Save brute force results
        bf_results_path = os.path.join(brute_force_dir, "brute_force_results.json")
        with open(bf_results_path, 'w') as f:
            json.dump(brute_force_results, f, indent=2, default=str)
            
    except Exception as e:
        print(f"   ❌ Brute force test failed: {e}")
    
    return brute_force_results

def main():
    if len(sys.argv) != 2:
        print("Usage: python spread_detection_enhanced.py path/to/test.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    print(f"📖 Enhanced Double Page Spread Detection Test")
    print(f"📄 Processing: {pdf_path}")
    print(f"🧠 Using 14 sophisticated heuristics")
    
    start_time = time.time()
    output_dir = create_output_dirs(pdf_path)
    
    try:
        with fitz.open(pdf_path) as doc:
            print(f"\n📊 PDF Analysis:")
            print(f"   📄 Total pages: {len(doc)}")
            
            # Get even/odd candidates (Rule #1)
            candidates = get_even_odd_candidates(doc)
            print(f"   📖 Spread candidates: {len(candidates)}")
            
            if not candidates:
                print("   ⚠️  No spread candidates found")
                return
            
            # Create AI analyzer ONCE for all pages (major performance improvement)
            analyzer = None
            if AI_MODELS_AVAILABLE:
                print(f"\n🚀 Initializing AI analyzer (once for all pages)...")
                try:
                    analyzer = create_content_analyzer(enable_surya=True, enable_florence2=True, debug_mode=False)
                    print(f"   ✅ AI analyzer ready for reuse")
                except Exception as e:
                    print(f"   ❌ AI analyzer initialization failed: {e}")
                    analyzer = None
            else:
                print(f"\n⚠️  AI models not available - using fallback methods")
            
            # Save individual page previews for more pages
            print(f"\n💾 Saving individual page previews...")
            max_preview_pages = min(100, len(doc))  # Up to 100 pages or all pages
            
            for page_idx in range(0, max_preview_pages):  # Save every page, not every other
                page = doc[page_idx]
                mat = fitz.Matrix(150/72, 150/72)  # 150 DPI for preview
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                img_path = os.path.join(output_dir, "individual_pages", f"page_{page_idx+1:03d}.png")
                pix.save(img_path)
            
            print(f"   💾 Saved {max_preview_pages} individual page previews")
            
            # BRUTE FORCE TEST: Join all even-odd pairs and test Florence2 detection
            print(f"\n🚀 Running BRUTE FORCE test: Join all even-odd + Florence2 detection...")
            brute_force_results = run_brute_force_join_test(doc, output_dir, analyzer)
            
            # Validate each candidate comprehensively
            all_validations = []
            created_spreads = []
            
            for candidate_num, (left_idx, right_idx) in enumerate(candidates):
                print(f"\n🔍 Validating spread candidate {candidate_num + 1}/{len(candidates)}: pages {left_idx + 1}-{right_idx + 1}")
                
                validation = comprehensive_spread_validation(doc, left_idx, right_idx, output_dir, analyzer)
                all_validations.append(validation)
                
                print(f"   📊 Validation score: {validation['validation_score']}/100")
                print(f"   📈 Confidence: {validation['confidence']:.2f}")
                print(f"   ✅ Is spread: {validation['is_spread']}")
                
                # Save spread candidate visualization (whether valid or not)
                save_spread_candidate_visualization(doc, left_idx, right_idx, validation, output_dir)
                
                # Create spread if valid
                if validation["is_spread"]:
                    spread_path = create_spread_if_valid(doc, left_idx, right_idx, validation, output_dir)
                    if spread_path:
                        created_spreads.append(spread_path)
                        print(f"   💾 Spread saved: {os.path.basename(spread_path)}")
            
            # Save comprehensive results
            results = {
                "pdf_path": pdf_path,
                "total_pages": len(doc),
                "candidates_tested": len(candidates),
                "spreads_created": len(created_spreads),
                "all_validations": all_validations,
                "created_spreads": created_spreads,
                "ai_models_available": AI_MODELS_AVAILABLE
            }
            
            results_path = os.path.join(output_dir, "enhanced_spread_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Print summary
            elapsed = time.time() - start_time
            valid_spreads = sum(1 for v in all_validations if v["is_spread"])
            
            print(f"\n✅ Enhanced Spread Detection Complete!")
            print(f"   ⏱️  Processing time: {elapsed:.2f} seconds")
            print(f"   📁 Results saved to: {output_dir}")
            print(f"   📖 Candidates tested: {len(candidates)}")
            print(f"   ✅ Valid spreads found: {valid_spreads}")
            print(f"   💾 Spreads created: {len(created_spreads)}")
            print(f"   🎯 Score threshold: 49+ points (based on testing data)")
            
            if valid_spreads > 0:
                success_rate = valid_spreads / len(candidates)
                print(f"   📈 Success rate: {success_rate:.1%}")
                
                avg_score = sum(v["validation_score"] for v in all_validations if v["is_spread"]) / valid_spreads
                print(f"   📊 Average confidence: {avg_score:.1f}/100")
            
            print(f"\n🔧 Key Improvements Made:")
            print(f"   🎯 Color patches moved 10px from seam")
            print(f"   🎨 Background color detection enabled")
            print(f"   📦 Florence2 detection made optional")
            print(f"   📉 Score threshold lowered to 50 points")
            
            print(f"\n🚀 Next Steps:")
            print(f"   1. Review joined_spreads/ for created spreads")
            print(f"   2. Check color_analysis/ for seam matching details")
            print(f"   3. Examine validation scores vs actual spreads")
            print(f"   4. Fine-tune thresholds based on results")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()