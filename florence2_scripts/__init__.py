"""
Florence2 Scripts Package
Computer vision and analysis tools for PDF processing
"""

# Import local modules
try:
    from .florence2_detector import Florence2RectangleDetector, BoundingBox
except ImportError:
    print("⚠️ Florence2RectangleDetector not available in florence2_scripts")

try:
    from .analysis_engine import analyze_for_pdf_extraction, ContentAnalysisEngine
except ImportError:
    print("⚠️ Analysis engine not available")

try:
    from .modern_image_enhancer import ModernImageEnhancer
except ImportError:
    print("⚠️ ModernImageEnhancer not available")

__all__ = [
    'Florence2RectangleDetector',
    'BoundingBox', 
    'analyze_for_pdf_extraction',
    'ContentAnalysisEngine',
    'ModernImageEnhancer'
]