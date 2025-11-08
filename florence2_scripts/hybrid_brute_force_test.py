"""
Hybrid Brute Force Test

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
Hybrid Brute Force Test Script - Ultimate Spread Detection + Florence2 + Surya
Combines: Brute force joining + Spread detection + Dual AI + Smart filtering

Usage: python hybrid_brute_force_test.py path/to/test.pdf
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
from PIL import Image, ImageDraw, ImageFont, ImageStat
from io import BytesIO

try:
    import pymupdf as fitz
except ImportError:
    import fitz

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
    base_dir = f"hybrid_test_{pdf_name}_{timestamp}"
    
    dirs = [
        base_dir,
        os.path.join(base_dir, "stage1_raw_detection"),
        os.path.join(base_dir, "stage2_spread_classification"),
        os.path.join(base_dir, "stage3_smart_filtering"),
        os.path.join(base_dir, "final_results"),
        os.path.join(base_dir, "analysis_data")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dir

def get_even_odd_candidates(doc) -> List[Tuple[int, int]]:
    """Get spread candidates using even/odd strategy"""
    candidates = []
    total_pages = len(doc)
    
    # Join pages 1-2, 3-4, 5-6, etc. (0-indexed: 0-1, 2-3, 4-5)
    for i in range(1, total_pages - 1, 2):
        left_page_idx = i
        right_page_idx = i + 1
        
        if right_page_idx < total_pages:
            candidates.append((left_page_idx, right_page_idx))
    
    return candidates

def quick_spread_detection(left_img: Image.Image, right_img: Image.Image) -> Dict[str, Any]:
    """Simplified spread detection using color patches (fast version)"""
    
    try:
        # Quick color patch analysis - fewer patches for speed
        patch_count = 10
        height1, height2 = left_img.height, right_img.height
        min_height = min(height1, height2)
        patch_height = min_height // (patch_count + 1)
        patch_size = 10
        
        matches = 0
        total_brightness = 0
        
        for i in range(1, patch_count + 1):
            y = i * patch_height
            
            # Get patches at seam
            left_patch = left_img.crop((left_img.width - patch_size, y, left_img.width, y + patch_size))
            right_patch = right_img.crop((0, y, patch_size, y + patch_size))
            
            left_avg = ImageStat.Stat(left_patch).mean
            right_avg = ImageStat.Stat(right_patch).mean
            
            # Color difference
            color_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(left_avg, right_avg)))
            total_brightness += (sum(left_avg) + sum(right_avg)) / 6
            
            if color_diff < 30:  # Match threshold
                matches += 1
        
        # Check for white background
        avg_brightness = total_brightness / patch_count
        is_white_background = avg_brightness > 240
        
        # Score
        if matches >= 6 and not is_white_background:
            spread_score = 30
        elif matches >= 4 and not is_white_background:
            spread_score = 15
        else:
            spread_score = 0
        
        return {
            "is_likely_spread": spread_score > 0,
            "spread_score": spread_score,
            "color_matches": matches,
            "total_patches": patch_count,
            "avg_brightness": avg_brightness,
            "is_white_background": is_white_background
        }
        
    except Exception as e:
        return {
            "is_likely_spread": False,
            "spread_score": 0,
            "error": str(e)
        }

def detect_seam_crossing_boxes(boxes: List[Dict], left_width: int, seam_tolerance: int = 50) -> List[Dict]:
    """Detect boxes that cross the seam between pages"""
    
    seam_x = left_width
    crossing_boxes = []
    
    for box in boxes:
        bbox = box["bbox"]
        left_edge = bbox[0]
        right_edge = bbox[2]
        
        # Check if box crosses the seam (within tolerance)
        crosses_seam = (left_edge < seam_x + seam_tolerance and 
                       right_edge > seam_x - seam_tolerance)
        
        if crosses_seam:
            box["crosses_seam"] = True
            crossing_boxes.append(box)
        else:
            box["crosses_seam"] = False
    
    return crossing_boxes

def filter_nested_boxes(boxes: List[Dict], iou_threshold: float = 0.8) -> List[Dict]:
    """Remove boxes that are nested inside other boxes"""
    
    def calculate_iou(box1, box2):
        """Calculate intersection over union of two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1["bbox"]
        x1_2, y1_2, x2_2, y2_2 = box2["bbox"]
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Sort by area (largest first)
    sorted_boxes = sorted(boxes, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)
    
    filtered_boxes = []
    
    for box in sorted_boxes:
        is_nested = False
        
        for existing_box in filtered_boxes:
            iou = calculate_iou(box, existing_box)
            if iou > iou_threshold:
                is_nested = True
                break
        
        if not is_nested:
            filtered_boxes.append(box)
    
    return filtered_boxes

def filter_by_size(boxes: List[Dict], image_width: int, image_height: int, 
                  min_area_pct: float = 5.0, max_area_pct: float = 80.0) -> List[Dict]:
    """Filter boxes by size - remove too small or too large boxes"""
    
    total_area = image_width * image_height
    min_area = total_area * (min_area_pct / 100)
    max_area = total_area * (max_area_pct / 100)
    
    filtered_boxes = []
    
    for box in boxes:
        bbox = box["bbox"]
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if min_area <= box_area <= max_area:
            box["area"] = box_area
            box["area_pct"] = (box_area / total_area) * 100
            filtered_boxes.append(box)
        else:
            box["filtered_reason"] = f"Size: {box_area} outside range {min_area}-{max_area}"
    
    return filtered_boxes

def validate_with_surya(boxes: List[Dict], surya_regions: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
    """Validate Florence2 boxes against Surya layout regions"""
    
    def boxes_overlap(box1_bbox, box2_bbox, threshold):
        """Check if two boxes overlap by at least threshold"""
        x1_1, y1_1, x2_1, y2_1 = box1_bbox
        x1_2, y1_2, x2_2, y2_2 = box2_bbox
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False, 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        overlap_ratio = intersection / area1 if area1 > 0 else 0
        return overlap_ratio >= threshold, overlap_ratio
    
    validated_boxes = []
    
    for box in boxes:
        box["surya_validation"] = {
            "overlaps_image_region": False,
            "overlaps_text_region": False,
            "best_overlap_ratio": 0.0,
            "best_overlap_label": "none"
        }
        
        best_overlap = 0.0
        best_label = "none"
        overlaps_image = False
        overlaps_text = False
        
        for region in surya_regions:
            is_overlap, overlap_ratio = boxes_overlap(box["bbox"], region["bbox"], overlap_threshold)
            
            if is_overlap:
                label = region.get("semantic_label", "").lower()
                
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_label = label
                
                if "image" in label or "figure" in label or "picture" in label:
                    overlaps_image = True
                elif "text" in label:
                    overlaps_text = True
        
        box["surya_validation"]["overlaps_image_region"] = overlaps_image
        box["surya_validation"]["overlaps_text_region"] = overlaps_text
        box["surya_validation"]["best_overlap_ratio"] = best_overlap
        box["surya_validation"]["best_overlap_label"] = best_label
        
        # Keep boxes that overlap with image regions or have no strong text overlap
        if overlaps_image or (not overlaps_text and best_overlap < 0.5):
            validated_boxes.append(box)
        else:
            box["filtered_reason"] = f"Surya validation: overlaps text region ({best_label})"
    
    return validated_boxes

def hybrid_process_pair(doc, left_idx: int, right_idx: int, analyzer, output_dir: str) -> Dict[str, Any]:
    """Process a single page pair with hybrid approach"""
    
    pair_id = f"{left_idx+1:03d}_{right_idx+1:03d}"
    print(f"    🔄 Processing pair {left_idx+1}-{right_idx+1}")
    
    # Extract pages as high-res images
    left_page = doc[left_idx]
    right_page = doc[right_idx]
    
    mat = fitz.Matrix(200/72, 200/72)  # 200 DPI
    left_pix = left_page.get_pixmap(matrix=mat, alpha=False)
    right_pix = right_page.get_pixmap(matrix=mat, alpha=False)
    
    left_img_data = left_pix.tobytes("ppm")
    right_img_data = right_pix.tobytes("ppm")
    left_img = Image.open(BytesIO(left_img_data))
    right_img = Image.open(BytesIO(right_img_data))
    
    # Join images
    joined_width = left_img.width + right_img.width
    joined_height = max(left_img.height, right_img.height)
    joined_img = Image.new('RGB', (joined_width, joined_height), 'white')
    joined_img.paste(left_img, (0, 0))
    joined_img.paste(right_img, (left_img.width, 0))
    
    results = {
        "pair_id": pair_id,
        "pages": [left_idx + 1, right_idx + 1],
        "stage1_raw": {},
        "stage2_spread": {},
        "stage3_filtered": {},
        "final_boxes": [],
        "processing_time": 0
    }
    
    start_time = time.time()
    
    try:
        # STAGE 1: Raw Detection
        print(f"      📊 Stage 1: Raw detection...")
        
        if analyzer and AI_MODELS_AVAILABLE:
            analysis = analyzer.analyze_image_comprehensive(joined_img)
            florence2_boxes = analysis.get("florence2_rectangles", [])
            surya_regions = analysis.get("surya_layout", [])
            
            results["stage1_raw"] = {
                "florence2_count": len(florence2_boxes),
                "surya_count": len(surya_regions),
                "florence2_boxes": florence2_boxes,
                "surya_regions": surya_regions
            }
            
            print(f"         📦 Florence2: {len(florence2_boxes)} boxes")
            print(f"         🎯 Surya: {len(surya_regions)} regions")
            
            # Save Stage 1 visualization
            save_stage_visualization(joined_img, florence2_boxes, surya_regions, 
                                   pair_id, "stage1_raw", output_dir, left_img.width)
        else:
            print(f"         ❌ AI models not available")
            return results
        
        # STAGE 2: Spread Classification
        print(f"      🎯 Stage 2: Spread classification...")
        
        spread_analysis = quick_spread_detection(left_img, right_img)
        is_spread = spread_analysis["is_likely_spread"]
        
        results["stage2_spread"] = {
            "is_spread": is_spread,
            "spread_score": spread_analysis["spread_score"],
            "analysis": spread_analysis
        }
        
        print(f"         📊 Spread: {'YES' if is_spread else 'NO'} (score: {spread_analysis['spread_score']})")
        
        # STAGE 3: Smart Filtering
        print(f"      🔧 Stage 3: Smart filtering...")
        
        filtered_boxes = florence2_boxes.copy()
        
        # Step 3a: Confidence filtering
        conf_threshold = 0.7 if is_spread else 0.4  # Lowered from 0.5 to 0.4
        conf_filtered = [box for box in filtered_boxes if box.get("confidence", 1.0) >= conf_threshold]
        print(f"         📊 After confidence filter (>{conf_threshold}): {len(conf_filtered)} boxes")
        
        # Debug: Show what was filtered by confidence
        conf_rejected = len(filtered_boxes) - len(conf_filtered)
        if conf_rejected > 0:
            print(f"             🚫 Confidence rejected: {conf_rejected} boxes")
            
        filtered_boxes = conf_filtered
        
        # Step 3b: Size filtering (made less aggressive)
        size_filtered = filter_by_size(filtered_boxes, joined_width, joined_height, 
                                     min_area_pct=2.0, max_area_pct=90.0)  # More permissive
        print(f"         📏 After size filter (2%-90%): {len(size_filtered)} boxes")
        
        # Debug: Show what was filtered by size
        size_rejected = len(filtered_boxes) - len(size_filtered)
        if size_rejected > 0:
            print(f"             🚫 Size rejected: {size_rejected} boxes")
            
        filtered_boxes = size_filtered
        
        # Step 3c: Seam analysis for spreads
        if is_spread:
            seam_crossing = detect_seam_crossing_boxes(filtered_boxes, left_img.width)
            # For spreads, prefer boxes that cross the seam
            if seam_crossing:
                filtered_boxes = seam_crossing
                print(f"         🔗 Spread mode - keeping seam-crossing boxes: {len(filtered_boxes)} boxes")
        
        # Step 3d: Surya validation (made less aggressive)
        surya_filtered = validate_with_surya(filtered_boxes, surya_regions, overlap_threshold=0.2)  # Lowered from 0.3
        print(f"         ✅ After Surya validation: {len(surya_filtered)} boxes")
        
        # Debug: Show what was filtered by Surya
        surya_rejected = len(filtered_boxes) - len(surya_filtered)
        if surya_rejected > 0:
            print(f"             🚫 Surya rejected: {surya_rejected} boxes")
            
        filtered_boxes = surya_filtered
        
        # Step 3e: Remove nested boxes (made less aggressive)
        nested_filtered = filter_nested_boxes(filtered_boxes, iou_threshold=0.9)  # Raised from 0.8
        print(f"         🎯 After nested removal: {len(nested_filtered)} boxes")
        
        # Debug: Show what was filtered by nested removal
        nested_rejected = len(filtered_boxes) - len(nested_filtered)
        if nested_rejected > 0:
            print(f"             🚫 Nested rejected: {nested_rejected} boxes")
            
        filtered_boxes = nested_filtered
        
        results["stage3_filtered"] = {
            "confidence_threshold": conf_threshold,
            "final_count": len(filtered_boxes),
            "filtering_steps": {
                "confidence": len([box for box in florence2_boxes if box.get("confidence", 1.0) >= conf_threshold]),
                "size": "applied",
                "seam_crossing": len(detect_seam_crossing_boxes(filtered_boxes, left_img.width)) if is_spread else "not_applicable",
                "surya_validation": "applied",
                "nested_removal": "applied"
            }
        }
        
        results["final_boxes"] = filtered_boxes
        
        # Save final visualization
        save_final_visualization(joined_img, filtered_boxes, spread_analysis, 
                               pair_id, output_dir, left_img.width, is_spread)
        
    except Exception as e:
        print(f"         ❌ Error processing pair: {e}")
        results["error"] = str(e)
    
    results["processing_time"] = time.time() - start_time
    return results

def save_stage_visualization(img: Image.Image, florence2_boxes: List[Dict], surya_regions: List[Dict],
                           pair_id: str, stage: str, output_dir: str, seam_x: int):
    """Save visualization for a processing stage"""
    
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    
    # Draw Florence2 boxes in red
    for i, box in enumerate(florence2_boxes):
        bbox = box["bbox"]
        confidence = box.get("confidence", 1.0)
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1]-20), f"F2_{i+1} ({confidence:.2f})", fill="red")
    
    # Draw Surya regions in blue
    for i, region in enumerate(surya_regions):
        bbox = region["bbox"]
        label = region.get("semantic_label", "unknown")[:8]
        draw.rectangle(bbox, outline="blue", width=2)
        draw.text((bbox[2]-50, bbox[1]), f"S_{label}", fill="blue")
    
    # Draw seam
    draw.line([seam_x, 0, seam_x, img.height], fill="green", width=2)
    draw.text((seam_x-30, 30), "SEAM", fill="green")
    
    # Add summary
    draw.text((10, 10), f"Stage 1: F2={len(florence2_boxes)}, Surya={len(surya_regions)}", 
             fill="black")
    
    vis_path = os.path.join(output_dir, stage, f"{pair_id}_{stage}.png")
    vis_img.save(vis_path)

def save_final_visualization(img: Image.Image, final_boxes: List[Dict], spread_analysis: Dict,
                           pair_id: str, output_dir: str, seam_x: int, is_spread: bool):
    """Save final filtered results visualization"""
    
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    
    # Draw final boxes in green (good) or orange (filtered)
    for i, box in enumerate(final_boxes):
        bbox = box["bbox"]
        confidence = box.get("confidence", 1.0)
        crosses_seam = box.get("crosses_seam", False)
        
        color = "lime" if crosses_seam else "green"
        draw.rectangle(bbox, outline=color, width=4)
        draw.text((bbox[0], bbox[1]-25), f"FINAL_{i+1} ({confidence:.2f})", fill=color)
    
    # Draw seam
    draw.line([seam_x, 0, seam_x, img.height], fill="blue", width=3)
    draw.text((seam_x-30, 30), "SEAM", fill="blue")
    
    # Add summary info
    spread_text = f"SPREAD: {'YES' if is_spread else 'NO'} (score: {spread_analysis.get('spread_score', 0)})"
    final_text = f"FINAL BOXES: {len(final_boxes)}"
    
    draw.text((10, 10), spread_text, fill="blue")
    draw.text((10, 35), final_text, fill="green")
    
    vis_path = os.path.join(output_dir, "final_results", f"{pair_id}_final.png")
    vis_img.save(vis_path)

def main():
    if len(sys.argv) != 2:
        print("Usage: python hybrid_brute_force_test.py path/to/test.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not AI_MODELS_AVAILABLE:
        print("❌ AI models not available. Please check imports.")
        sys.exit(1)
    
    print(f"🚀 Hybrid Brute Force Test - Ultimate Intelligence")
    print(f"📄 Processing: {pdf_path}")
    print(f"🧠 Using: Brute Force + Spread Detection + Florence2 + Surya + Smart Filtering")
    
    start_time = time.time()
    output_dir = create_output_dirs(pdf_path)
    
    try:
        with fitz.open(pdf_path) as doc:
            print(f"\n📊 PDF Analysis:")
            print(f"   📄 Total pages: {len(doc)}")
            
            # Get candidates
            candidates = get_even_odd_candidates(doc)
            print(f"   📖 Even-odd pairs: {len(candidates)}")
            
            # Create AI analyzer once
            print(f"\n🚀 Initializing AI analyzer...")
            analyzer = create_content_analyzer(enable_surya=True, enable_florence2=True, debug_mode=False)
            print(f"   ✅ AI analyzer ready")
            
            # Process first 10 pairs for testing
            test_pairs = min(10, len(candidates))
            print(f"\n🔧 Processing first {test_pairs} pairs...")
            
            all_results = []
            total_original_boxes = 0
            total_final_boxes = 0
            spread_count = 0
            
            for i, (left_idx, right_idx) in enumerate(candidates[:test_pairs]):
                print(f"\n  🔄 Pair {i+1}/{test_pairs}: pages {left_idx+1}-{right_idx+1}")
                
                result = hybrid_process_pair(doc, left_idx, right_idx, analyzer, output_dir)
                all_results.append(result)
                
                # Accumulate stats
                original_count = result.get("stage1_raw", {}).get("florence2_count", 0)
                final_count = result.get("stage3_filtered", {}).get("final_count", 0)
                is_spread = result.get("stage2_spread", {}).get("is_spread", False)
                
                total_original_boxes += original_count
                total_final_boxes += final_count
                if is_spread:
                    spread_count += 1
                
                print(f"    📊 Result: {original_count} → {final_count} boxes ({'SPREAD' if is_spread else 'SINGLE'})")
            
            # Save comprehensive results
            final_results = {
                "pdf_path": pdf_path,
                "total_pages": len(doc),
                "pairs_tested": test_pairs,
                "spreads_detected": spread_count,
                "total_original_boxes": total_original_boxes,
                "total_final_boxes": total_final_boxes,
                "filtering_efficiency": ((total_original_boxes - total_final_boxes) / total_original_boxes * 100) if total_original_boxes > 0 else 0,
                "all_results": all_results
            }
            
            results_path = os.path.join(output_dir, "analysis_data", "hybrid_results.json")
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            # Print final summary
            elapsed = time.time() - start_time
            avg_original = total_original_boxes / test_pairs if test_pairs > 0 else 0
            avg_final = total_final_boxes / test_pairs if test_pairs > 0 else 0
            filtering_rate = final_results["filtering_efficiency"]
            
            print(f"\n✅ Hybrid Test Complete!")
            print(f"   ⏱️  Processing time: {elapsed:.2f} seconds")
            print(f"   📁 Results saved to: {output_dir}")
            print(f"   📖 Pairs tested: {test_pairs}")
            print(f"   📊 Spreads detected: {spread_count} ({spread_count/test_pairs*100:.1f}%)")
            print(f"   📦 Original boxes: {total_original_boxes} (avg {avg_original:.1f} per pair)")
            print(f"   🎯 Final boxes: {total_final_boxes} (avg {avg_final:.1f} per pair)")
            print(f"   🔧 Filtering efficiency: {filtering_rate:.1f}% reduction")
            
            print(f"\n🎯 Key Improvements:")
            print(f"   🚀 3-stage intelligent filtering")
            print(f"   📊 Spread-aware confidence thresholds")
            print(f"   🔗 Seam-crossing detection for spreads")
            print(f"   ✅ Surya semantic validation")
            print(f"   📏 Smart size filtering")
            print(f"   🎯 Nested box elimination")
            
            print(f"\n📁 Output Folders:")
            print(f"   📊 stage1_raw_detection/ - Florence2 + Surya raw results")
            print(f"   🎯 stage2_spread_classification/ - Spread vs single classification")
            print(f"   🔧 stage3_smart_filtering/ - Filtered intermediate results")
            print(f"   ✅ final_results/ - Final optimized bounding boxes")
            print(f"   📊 analysis_data/ - Comprehensive statistics and data")
            
            print(f"\n🚀 Next Steps:")
            print(f"   1. Review final_results/ for optimized bounding boxes")
            print(f"   2. Check filtering efficiency vs accuracy trade-off")
            print(f"   3. Adjust confidence thresholds based on results")
            print(f"   4. Consider integrating into main PDF extractor if results are good")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()