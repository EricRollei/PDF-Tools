"""
Diagnostic

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

import os
import sys

# Add the current directory to Python path for local imports  
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"🔍 Diagnostic Information:")
print(f"   Current dir: {current_dir}")
print(f"   Files in current dir: {[f for f in os.listdir(current_dir) if f.endswith('.py')]}")

# Test Florence2 import using local path (not PDF_tools)
print(f"\n🔍 Testing Florence2 Local Import:")
try:
    from florence2_detector import Florence2RectangleDetector, BoundingBox
    print('✅ Florence2 local import successful')
except Exception as e:
    print(f'❌ Florence2 local import failed: {e}')
    
    # Check if the file exists
    florence2_path = os.path.join(current_dir, "florence2_detector.py")
    print(f"   florence2_detector.py exists: {os.path.exists(florence2_path)}")
    
    if os.path.exists(florence2_path):
        print(f"   File size: {os.path.getsize(florence2_path)} bytes")
        # Check if it has the classes we need
        try:
            with open(florence2_path, 'r') as f:
                content = f.read()
                has_detector = 'class Florence2RectangleDetector' in content
                has_bbox = 'class BoundingBox' in content
                print(f"   Contains Florence2RectangleDetector: {has_detector}")
                print(f"   Contains BoundingBox: {has_bbox}")
        except Exception as e3:
            print(f"   Could not read file: {e3}")

# Test Surya import
print(f"\n🔍 Testing Surya Import:")
try:
    from surya.layout import LayoutPredictor
    print('✅ Surya import successful')
except Exception as e:
    print(f'❌ Surya import failed: {e}')

# Test analysis engine import
print(f"\n🔍 Testing Analysis Engine Import:")
try:
    from analysis_engine import ContentAnalysisEngine, analyze_for_pdf_extraction
    print('✅ Analysis engine import successful')
    
    # Test creating the engine
    engine = ContentAnalysisEngine(debug_mode=True)
    print('✅ Analysis engine creation successful')
    
except Exception as e:
    print(f'❌ Analysis engine import/creation failed: {e}')
    import traceback
    traceback.print_exc()