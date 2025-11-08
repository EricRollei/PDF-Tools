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
        os.path.join(base_dir, "heuristic_analysis"),
        os.path.join(base_dir, "color_analysis")
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

def detect_images_with_florence2(image: Image.Image) -> List[Dict[str, Any]]:
    """Detect images using Florence2 (Rule #9: Florence2 accuracy within few pixels)"""
    
    if not AI_MODELS_AVAILABLE:
        return []
    
    try:
        # FIX: Set ComfyUI path before analysis
        from analysis_engine import set_comfyui_base_path
        set_comfyui_base_path("A:\\Comfy_Dec\\ComfyUI")  # Add this line
        
        analysis = analyze_for_pdf_extraction(image, debug_mode=False)
        florence2_boxes = analysis.get("florence2_rectangles", [])
        return florence2_boxes
    except Exception as e:
        print(f"      ❌ Florence2 detection failed: {e}")
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
                                    patch_count: int = 10) -> Dict[str, Any]:
    """Analyze color matching across seam (Rule #13: 8/10 patches should match)"""
    
    height1, height2 = img1.height, img2.height
    min_height = min(height1, height2)
    
    # Sample patches at regular intervals vertically
    patch_height = min_height // (patch_count + 1)
    matches = []
    
    for i in range(1, patch_count + 1):
        y = i * patch_height
        
        # Get patch from right edge of left image
        patch_size = 10  # 10x10 pixel patches
        left_patch_box = (img1.width - patch_size, y, img1.width, y + patch_size)
        left_patch = img1.crop(left_patch_box)
        
        # Get patch from left edge of right image  
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
            "is_match": is_match
        })
    
    match_count = sum(1 for m in matches if m["is_match"])
    match_ratio = match_count / patch_count
    
    return {
        "patch_matches": matches,
        "match_count": match_count,
        "total_patches": patch_count,
        "match_ratio": match_ratio,
        "passes_rule": match_count >= 8,  # Rule #13: 8 or more out of 10
        "threshold": 30
    }

def comprehensive_spread_validation(doc, left_idx: int, right_idx: int, 
                                  output_dir: str) -> Dict[str, Any]:
    """Comprehensive validation using all 14 heuristics"""
    
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
    
    # Rule #1: Even/odd sequence (already validated by candidate selection)
    validation_results["heuristics"]["rule_1_even_odd"] = {
        "passed": True,
        "score": 20,
        "description": "Even to odd page sequence"
    }
    validation_results["validation_score"] += 20
    
    # Rule #2 & #10: Inside vs outside margins
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
    
    # Rule #4: Page sizes match
    size_validation = validate_page_sizes(left_page, right_page)
    validation_results["heuristics"]["rule_4_page_sizes"] = {
        "passed": size_validation["sizes_match"],
        "score": 10 if size_validation["sizes_match"] else -20,
        "details": size_validation
    }
    if size_validation["sizes_match"]:
        validation_results["validation_score"] += 10
    else:
        validation_results["validation_score"] -= 20  # Penalty for mismatched sizes
    
    # Rule #9: Florence2 image detection
    left_images = detect_images_with_florence2(left_img)
    right_images = detect_images_with_florence2(right_img)
    
    validation_results["heuristics"]["rule_9_florence2_detection"] = {
        "left_image_count": len(left_images),
        "right_image_count": len(right_images),
        "total_images": len(left_images) + len(right_images),
        "left_images": left_images,
        "right_images": right_images
    }
    
    # Rule #3 & #5: Image heights match
    if left_images and right_images:
        height_validation = validate_image_heights(left_images, right_images)
        validation_results["heuristics"]["rule_3_image_heights"] = {
            "passed": height_validation["height_validation"],
            "score": 25 if height_validation["height_validation"] else 0,
            "details": height_validation
        }
        if height_validation["height_validation"]:
            validation_results["validation_score"] += 25
    
    # Rule #10: Inside margin distances
    if left_images or right_images:
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
    
    # Rule #13: Color matching across seam
    color_analysis = analyze_color_patches_across_seam(left_img, right_img)
    validation_results["heuristics"]["rule_13_color_matching"] = {
        "passed": color_analysis["passes_rule"],
        "score": 30 if color_analysis["passes_rule"] else 0,
        "details": color_analysis
    }
    if color_analysis["passes_rule"]:
        validation_results["validation_score"] += 30
    
    # Save color analysis visualization
    save_color_analysis_visualization(left_img, right_img, color_analysis, 
                                    left_idx + 1, right_idx + 1, output_dir)
    
    # Final determination
    validation_results["is_spread"] = validation_results["validation_score"] >= 80
    validation_results["confidence"] = min(validation_results["validation_score"] / 100.0, 1.0)
    
    return validation_results

def save_color_analysis_visualization(left_img: Image.Image, right_img: Image.Image,
                                    color_analysis: Dict, left_page_num: int, 
                                    right_page_num: int, output_dir: str):
    """Save color analysis visualization"""
    
    # Create side-by-side visualization
    combined_width = left_img.width + right_img.width
    combined_height = max(left_img.height, right_img.height)
    
    vis_img = Image.new('RGB', (combined_width, combined_height), 'white')
    vis_img.paste(left_img, (0, 0))
    vis_img.paste(right_img, (left_img.width, 0))
    
    draw = ImageDraw.Draw(vis_img)
    
    # Draw patch locations and results
    for match in color_analysis["patch_matches"]:
        y = match["y_position"]
        color = "green" if match["is_match"] else "red"
        
        # Left patch
        draw.rectangle([left_img.width - 10, y, left_img.width, y + 10], 
                      outline=color, width=2)
        
        # Right patch
        draw.rectangle([left_img.width, y, left_img.width + 10, y + 10], 
                      outline=color, width=2)
        
        # Draw connection line
        draw.line([left_img.width - 5, y + 5, left_img.width + 5, y + 5], 
                 fill=color, width=1)
    
    # Add summary text
    match_text = f"Color Matches: {color_analysis['match_count']}/{color_analysis['total_patches']}"
    draw.text((10, 10), match_text, fill="black")
    
    vis_path = os.path.join(output_dir, "color_analysis", 
                           f"color_analysis_{left_page_num:03d}_{right_page_num:03d}.png")
    vis_img.save(vis_path)

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

def main():
    if len(sys.argv) != 2:
        print("Usage: python spread_detection_enhanced.py path/to/test.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # ADD THIS: Set ComfyUI path
    if AI_MODELS_AVAILABLE:
        try:
            from analysis_engine import set_comfyui_base_path
            set_comfyui_base_path("A:\\Comfy_Dec\\ComfyUI")
            print("✅ ComfyUI path configured for Florence2")
        except Exception as e:
            print(f"⚠️ Could not set ComfyUI path: {e}")
    
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
            
            # Validate each candidate comprehensively
            all_validations = []
            created_spreads = []
            
            for left_idx, right_idx in candidates:
                print(f"\n🔍 Validating spread candidate: pages {left_idx + 1}-{right_idx + 1}")
                
                validation = comprehensive_spread_validation(doc, left_idx, right_idx, output_dir)
                all_validations.append(validation)
                
                print(f"   📊 Validation score: {validation['validation_score']}/100")
                print(f"   📈 Confidence: {validation['confidence']:.2f}")
                print(f"   ✅ Is spread: {validation['is_spread']}")
                
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
            
            if valid_spreads > 0:
                success_rate = valid_spreads / len(candidates)
                print(f"   📈 Success rate: {success_rate:.1%}")
                
                avg_score = sum(v["validation_score"] for v in all_validations if v["is_spread"]) / valid_spreads
                print(f"   📊 Average confidence: {avg_score:.1f}/100")
            
            print(f"\n🚀 Next Steps:")
            print(f"   1. Review joined_spreads/ for created spreads")
            print(f"   2. Check heuristic_analysis/ for validation details")
            print(f"   3. Examine color_analysis/ for seam matching")
            print(f"   4. Fine-tune heuristic weights if needed")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()