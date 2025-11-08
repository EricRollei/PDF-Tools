"""
Test Fixed

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

# Create: test_fixed.py in florence2_scripts directory
import os
import sys
from PIL import Image, ImageDraw
from io import BytesIO

# Ensure we can import from current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Pre-load local modules BEFORE importing analysis_engine
try:
    from module_loader import ensure_local_modules
    ensure_local_modules()  # This will load modeling_florence2 and configuration_florence2
except ImportError:
    print("⚠️ Module loader not available, continuing without pre-loading")

def test_basic_functionality():
    """Test basic analysis engine functionality"""
    print("🧪 Testing Basic Analysis Engine Functionality")
    
    try:
        # Configure ComfyUI path
        COMFYUI_PATH = "A:\\Comfy_Dec\\ComfyUI"
        
        # Import analysis engine (after module pre-loading)
        from analysis_engine import ContentAnalysisEngine, analyze_for_pdf_extraction, set_comfyui_base_path
        
        # Set ComfyUI path
        set_comfyui_base_path(COMFYUI_PATH)
        
        # Create a simple test image
        test_image = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(test_image)
        
        # Draw some test content
        draw.rectangle([100, 100, 300, 200], outline='red', width=3, fill='lightblue')
        draw.rectangle([400, 150, 600, 350], outline='blue', width=3, fill='lightgreen')
        draw.rectangle([200, 400, 500, 500], outline='green', width=3, fill='lightyellow')
        draw.text((110, 110), "Test Text Region 1", fill='black')
        draw.text((410, 160), "Test Text Region 2", fill='black')
        
        print(f"   📄 Created test image: {test_image.size}")
        
        # Test analysis with single-use engine
        print(f"   🔍 Running analysis...")
        result = analyze_for_pdf_extraction(test_image, debug_mode=True)  # Enable debug to see what's happening
        
        print(f"\n📊 Analysis Results:")
        print(f"   ✅ Analysis completed successfully")
        print(f"   📋 Surya regions: {len(result.get('surya_layout', []))}")
        print(f"   📦 Florence2 rectangles: {len(result.get('florence2_rectangles', []))}")
        
        # Show semantic regions
        semantic = result.get('semantic_regions', {})
        print(f"   🎯 Semantic regions:")
        for region_type, regions in semantic.items():
            print(f"      {region_type}: {len(regions)} regions")
        
        # Show analysis methods used
        methods = result.get('analysis_summary', {}).get('analysis_methods', [])
        print(f"   🔧 Methods used: {', '.join(methods)}")
        
        # Consider test successful if any analysis method worked
        return len(methods) > 1  # Should have more than just OCR
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_pdf():
    """Test with actual PDF page"""
    print("\n🧪 Testing with Real PDF")
    
    try:
        import pymupdf as fitz
        from analysis_engine import analyze_for_pdf_extraction, set_comfyui_base_path
        
        # Configure paths
        COMFYUI_PATH = "A:\\Comfy_Dec\\ComfyUI"
        PDF_PATH = "A:/test2.pdf"
        
        set_comfyui_base_path(COMFYUI_PATH)
        
        # Load PDF page
        with fitz.open(PDF_PATH) as doc:
            page = doc[5]  # Page 6
            mat = fitz.Matrix(200/72, 200/72)  # 200 DPI
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes('ppm')
            image = Image.open(BytesIO(img_data))
            
            print(f"   📄 PDF page loaded: {image.size}")
            
            # Analyze
            result = analyze_for_pdf_extraction(image, debug_mode=False)
            
            print(f"\n📊 PDF Analysis Results:")
            surya_count = len(result.get('surya_layout', []))
            florence2_count = len(result.get('florence2_rectangles', []))
            
            print(f"   📋 Surya regions: {surya_count}")
            print(f"   📦 Florence2 rectangles: {florence2_count}")
            
            if florence2_count > 0:
                print(f"   🎉 Florence2 is working! Found {florence2_count} rectangles")
                # Show first few rectangles
                rects = result.get('florence2_rectangles', [])[:3]
                for i, rect in enumerate(rects):
                    bbox = rect.get('bbox', [0,0,0,0])
                    conf = rect.get('confidence', 0)
                    print(f"      Rectangle {i+1}: {bbox} (conf: {conf:.3f})")
            else:
                print(f"   ⚠️ Florence2 found no rectangles (may need debugging)")
            
            # Show semantic analysis
            semantic = result.get('semantic_regions', {})
            image_regions = len(semantic.get('image_regions', []))
            text_regions = len(semantic.get('text_regions', []))
            
            print(f"   🖼️ Image regions identified: {image_regions}")
            print(f"   📝 Text regions identified: {text_regions}")
            
            return florence2_count > 0 or surya_count > 0
            
    except Exception as e:
        print(f"❌ PDF test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Florence2 Analysis Engine Test Suite")
    print("=" * 50)
    
    # Run basic test
    basic_success = test_basic_functionality()
    
    # Run PDF test
    pdf_success = test_with_real_pdf()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Basic functionality: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   PDF processing: {'✅ PASS' if pdf_success else '❌ FAIL'}")
    
    if basic_success and pdf_success:
        print("\n🎉 All tests passed! Analysis engine is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")