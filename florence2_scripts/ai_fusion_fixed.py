"""
Ai Fusion Fixed

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
AI Fusion Test Script - Florence2 + Surya Image Detection
Tests combining Florence2 precision with Surya Layout completeness

Usage: python ai_fusion_test.py path/to/test.pdf
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO

try:
    import pymupdf as fitz
except ImportError:
    import fitz

# Import working analysis engine - fix Python path issue
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
    print("✅ AI models imported from analysis_engine")
    AI_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import analysis_engine: {e}")
    try:
        # Try importing as a module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("analysis_engine", "analysis_engine.py")
        analysis_engine_module = importlib.util.module_from_spec(spec)
        sys.modules["analysis_engine"] = analysis_engine_module
        spec.loader.exec_module(analysis_engine_module)
        
        create_content_analyzer = analysis_engine_module.create_content_analyzer
        analyze_for_pdf_extraction = analysis_engine_module.analyze_for_pdf_extraction
        print("✅ AI models imported via direct file loading")
        AI_MODELS_AVAILABLE = True
    except Exception as e2:
        print(f"❌ Direct file loading also failed: {e2}")
        print(f"   📋 There might be an error in analysis_engine.py itself")
        AI_MODELS_AVAILABLE = False

def create_output_dirs(pdf_path: str) -> str:
    """Create test output directories"""
    pdf_name = Path(pdf_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"ai_fusion_test_{pdf_name}_{timestamp}"
    
    dirs = [
        base_dir,
        os.path.join(base_dir, "surya_results"),
        os.path.join(base_dir, "florence2_results"), 
        os.path.join(base_dir, "fusion_results"),
        os.path.join(base_dir, "comparison")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dir

def run_florence2_detection(image: Image.Image, page_num: int, output_dir: str) -> List[Dict]:
    """Run Florence2 detection using working analysis engine"""
    
    if not AI_MODELS_AVAILABLE:
        print(f"      ❌ Florence2 skipped - AI models not available")
        return []
    
    try:
        print(f"      🔍 Running Florence2 detection...")
        
        # Use analyze_for_pdf_extraction directly
        analysis = analyze_for_pdf_extraction(image, debug_mode=True)
        florence2_boxes = analysis.get("florence2_rectangles", [])
        
        print(f"      📦 Florence2 detected: {len(florence2_boxes)} regions")
        
        # Save detection visualization
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        for i, box in enumerate(florence2_boxes):
            bbox = box["bbox"]
            confidence = box.get("confidence", 1.0)
            
            # Draw red rectangle
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0], bbox[1]-20), f"F2_{i+1} ({confidence:.2f})", fill="red")
        
        vis_path = os.path.join(output_dir, "florence2_results", f"page_{page_num:03d}_florence2.png")
        vis_image.save(vis_path)
        
        return florence2_boxes
        
    except Exception as e:
        print(f"      ❌ Florence2 failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_surya_detection(image: Image.Image, page_num: int, output_dir: str) -> List[Dict]:
    """Run Surya layout detection using working analysis engine"""
    
    if not AI_MODELS_AVAILABLE:
        print(f"      ❌ Surya skipped - AI models not available")
        return []
    
    try:
        print(f"      🎯 Running Surya Layout detection...")
        
        # FIX: Use analyze_for_pdf_extraction instead of create_content_analyzer
        analysis = analyze_for_pdf_extraction(image, debug_mode=True)
        
        surya_regions = analysis.get("surya_layout", [])
        print(f"      📋 Surya detected: {len(surya_regions)} regions")
        
        # Filter for image-like regions from semantic_regions
        semantic_regions = analysis.get("semantic_regions", {})
        image_regions = semantic_regions.get("image_regions", [])
        
        print(f"      🖼️  Surya image regions: {len(image_regions)}")
        
        # Save detection visualization - use semantic regions for better filtering
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Draw all surya regions with color coding
        for i, region in enumerate(surya_regions):
            bbox = region["bbox"]
            label = region.get("label", "unknown")
            confidence = region.get("confidence", 0.0)
            
            # Color code by type
            if "image" in label.lower() or "figure" in label.lower():
                color = "blue"
            elif "text" in label.lower():
                color = "green"
            else:
                color = "orange"
            
            draw.rectangle(bbox, outline=color, width=2)
            draw.text((bbox[0], bbox[1]-15), f"{label[:8]} ({confidence:.2f})", fill=color)
        
        vis_path = os.path.join(output_dir, "surya_results", f"page_{page_num:03d}_surya.png")
        vis_image.save(vis_path)
        
        return image_regions  # Return semantic image regions, not all surya regions
        
    except Exception as e:
        print(f"      ❌ Surya failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_fusion_analysis(image: Image.Image, page_num: int, output_dir: str) -> Dict[str, Any]:
    """Run both detection methods and create fusion analysis"""
    
    if not AI_MODELS_AVAILABLE:
        print(f"      ❌ Fusion skipped - AI models not available")
        return {"error": "AI models not available"}
    
    try:
        print(f"      🚀 Running AI Fusion analysis...")
        
        # Use the complete analysis engine with both models
        analysis = analyze_for_pdf_extraction(image, debug_mode=True)
        
        florence2_boxes = analysis.get("florence2_rectangles", [])
        surya_regions = analysis.get("surya_layout", [])
        semantic_regions = analysis.get("semantic_regions", {})
        
        # Filter Surya for image regions
        surya_image_regions = semantic_regions.get("image_regions", [])
        
        fusion_results = {
            "page_num": page_num,
            "florence2_count": len(florence2_boxes),
            "surya_total_count": len(surya_regions),
            "surya_image_count": len(surya_image_regions),
            "florence2_boxes": florence2_boxes,
            "surya_image_regions": surya_image_regions,
            "analysis_summary": analysis.get("analysis_summary", {})
        }
        
        # Create fusion visualization
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Draw Florence2 in red
        for i, box in enumerate(florence2_boxes):
            bbox = box["bbox"]
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0], bbox[1]-20), f"F2_{i+1}", fill="red")
        
        # Draw Surya image regions in blue
        for i, region in enumerate(surya_image_regions):
            bbox = region["bbox"]
            draw.rectangle(bbox, outline="blue", width=2)
            draw.text((bbox[2]-30, bbox[1]), f"S_{i+1}", fill="blue")
        
        vis_path = os.path.join(output_dir, "fusion_results", f"page_{page_num:03d}_fusion.png")
        vis_image.save(vis_path)
        
        print(f"      ✅ Fusion complete: F2={len(florence2_boxes)}, Surya={len(surya_image_regions)}")
        return fusion_results
        
    except Exception as e:
        print(f"      ❌ Fusion analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    if len(sys.argv) != 2:
        print("Usage: python ai_fusion_test.py path/to/test.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not AI_MODELS_AVAILABLE:
        print("❌ AI models not available. Please check imports.")
        sys.exit(1)
    
    print(f"🚀 AI Fusion Test - Florence2 + Surya")
    print(f"📄 Processing: {pdf_path}")
    
    start_time = time.time()
    output_dir = create_output_dirs(pdf_path)
    
    try:
        with fitz.open(pdf_path) as doc:
            print(f"📊 PDF Analysis:")
            print(f"   📄 Total pages: {len(doc)}")
            
            # Test first 5 pages
            test_pages = min(5, len(doc))
            print(f"   🔍 Testing first {test_pages} pages")
            
            all_results = []
            
            for page_num in range(test_pages):
                print(f"\n📖 Processing page {page_num + 1}...")
                
                page = doc[page_num]
                mat = fitz.Matrix(200/72, 200/72)  # 200 DPI
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(BytesIO(img_data))
                
                # Run all detection methods
                florence2_results = run_florence2_detection(image, page_num + 1, output_dir)
                surya_results = run_surya_detection(image, page_num + 1, output_dir)
                fusion_results = run_fusion_analysis(image, page_num + 1, output_dir)
                
                all_results.append({
                    "page": page_num + 1,
                    "florence2": florence2_results,
                    "surya": surya_results,
                    "fusion": fusion_results
                })
        
        # Save summary results
        summary_path = os.path.join(output_dir, "fusion_test_results.json")
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\n✅ AI Fusion Test Complete!")
        print(f"   ⏱️  Processing time: {elapsed:.2f} seconds")
        print(f"   📁 Results saved to: {output_dir}")
        print(f"   📊 Summary saved to: fusion_test_results.json")
        
        # Quick stats
        total_florence2 = sum(len(r["florence2"]) for r in all_results)
        total_surya = sum(len(r["surya"]) for r in all_results)
        
        print(f"\n📈 Detection Summary:")
        print(f"   📦 Florence2 total detections: {total_florence2}")
        print(f"   🎯 Surya image regions: {total_surya}")
        print(f"   📊 Average per page: F2={total_florence2/test_pages:.1f}, Surya={total_surya/test_pages:.1f}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()