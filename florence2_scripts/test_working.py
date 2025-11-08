"""
Test Working

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

# Update test_working.py with path configuration:
import os
import sys
from PIL import Image, ImageDraw

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"🔍 Working Test Script")
print(f"   Current dir: {current_dir}")

# Configure ComfyUI path for testing
COMFYUI_PATH = "A:\\Comfy_Dec\\ComfyUI"
print(f"🔧 Setting ComfyUI path for testing: {COMFYUI_PATH}")

# Test step by step
print(f"\n1️⃣ Testing Florence2 Direct Import:")
try:
    from florence2_detector import Florence2RectangleDetector, BoundingBox
    print('✅ Florence2 direct import successful')
    florence2_available = True
except Exception as e:
    print(f'❌ Florence2 direct import failed: {e}')
    florence2_available = False

print(f"\n2️⃣ Testing Surya Import:")
try:
    from surya.layout import LayoutPredictor
    print('✅ Surya import successful')
    surya_available = True
except Exception as e:
    print(f'❌ Surya import failed: {e}')
    surya_available = False

print(f"\n3️⃣ Testing Analysis Engine:")
try:
    from analysis_engine import ContentAnalysisEngine, analyze_for_pdf_extraction, set_comfyui_base_path
    print('✅ Analysis engine import successful')
    
    # Set the correct ComfyUI path for testing
    if set_comfyui_base_path(COMFYUI_PATH):
        print('✅ ComfyUI path configured successfully')
    else:
        print('⚠️ ComfyUI path configuration failed, models may not load')
    
    # Create engine with available components
    engine = ContentAnalysisEngine(
        enable_florence2=florence2_available,
        enable_surya=surya_available,
        enable_ocr=True,
        debug_mode=True
    )
    print('✅ Analysis engine created successfully')
    
    # Test with simple image
    test_image = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([100, 100, 300, 200], outline='red', width=3)
    draw.text((110, 110), "Test text", fill='black')
    
    print(f"\n4️⃣ Testing Analysis:")
    result = engine.analyze_image_comprehensive(test_image)
    
    print(f"📊 Analysis Results:")
    print(f"   Surya regions: {len(result.get('surya_layout', []))}")
    print(f"   Florence2 rectangles: {len(result.get('florence2_rectangles', []))}")
    print(f"   Analysis methods: {result.get('analysis_summary', {}).get('analysis_methods', [])}")
    
    # Test the convenience function too
    print(f"\n5️⃣ Testing Convenience Function:")
    result2 = analyze_for_pdf_extraction(test_image, debug_mode=True)
    print(f"   Result keys: {list(result2.keys())}")
    print(f"   Florence2 rectangles: {len(result2.get('florence2_rectangles', []))}")
    
    print(f"\n🎉 All tests passed!")
    
except Exception as e:
    print(f'❌ Analysis engine test failed: {e}')
    import traceback
    traceback.print_exc()