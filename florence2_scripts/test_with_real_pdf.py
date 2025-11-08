"""
Test With Real Pdf

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

# Create: test_with_real_pdf.py
import os
import sys
from PIL import Image
from io import BytesIO

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure ComfyUI path
COMFYUI_PATH = "A:\\Comfy_Dec\\ComfyUI"
PDF_PATH = "A:/test2.pdf"

print(f"🔍 Real PDF Test Script")
print(f"   ComfyUI Path: {COMFYUI_PATH}")
print(f"   PDF Path: {PDF_PATH}")

try:
    import pymupdf as fitz
    from analysis_engine import ContentAnalysisEngine, analyze_for_pdf_extraction, set_comfyui_base_path
    
    # Set ComfyUI path
    if set_comfyui_base_path(COMFYUI_PATH):
        print('✅ ComfyUI path configured successfully')
    
    # Create analyzer
    analyzer = ContentAnalysisEngine(debug_mode=True)
    
    # Test with real PDF page
    with fitz.open(PDF_PATH) as doc:
        page = doc[5]  # Page 6
        mat = fitz.Matrix(200/72, 200/72)  # 200 DPI
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes('ppm')
        image = Image.open(BytesIO(img_data))
        
        print(f"\n📄 Testing with PDF page 6: {image.size}")
        
        # Test analysis
        result = analyze_for_pdf_extraction(image, debug_mode=True)
        
        print(f"\n📊 Real PDF Analysis Results:")
        print(f"   Surya regions: {len(result.get('surya_layout', []))}")
        print(f"   Florence2 rectangles: {len(result.get('florence2_rectangles', []))}")
        print(f"   Semantic regions keys: {list(result.get('semantic_regions', {}).keys())}")
        
        # Show details of found rectangles
        florence2_rects = result.get('florence2_rectangles', [])
        if florence2_rects:
            print(f"   🎯 Florence2 found {len(florence2_rects)} rectangles:")
            for i, rect in enumerate(florence2_rects[:5]):  # Show first 5
                bbox = rect.get('bbox', [0,0,0,0])
                conf = rect.get('confidence', 0)
                print(f"      {i+1}: bbox={bbox}, confidence={conf:.3f}")

except Exception as e:
    print(f"❌ Real PDF test failed: {e}")
    import traceback
    traceback.print_exc()