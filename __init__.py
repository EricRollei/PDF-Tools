"""
  Init  

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

"""
PDF Tools - ComfyUI Custom Node Package

Description: Main initialization module for PDF Tools custom nodes package.
    Loads and registers all PDF processing, media downloading, and AI vision nodes for ComfyUI.

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
This package integrates multiple third-party libraries and tools:
- gallery-dl (GNU GPL v2) by Mike Fährmann: https://github.com/mikf/gallery-dl
- yt-dlp (Unlicense/Public Domain): https://github.com/yt-dlp/yt-dlp
- Florence-2 models (MIT License) by Microsoft
- SAM2 models (Apache 2.0) by Meta AI
- Surya OCR (GNU GPL v3) by Vik Paruchuri
- PyMuPDF/fitz (AGPL v3/Commercial) by Artifex Software
See CREDITS.md for complete list of dependencies and their licenses.
"""

# PDF-tools/__init__.py
import importlib.util
import os
import sys
import shutil

# Add this import for ComfyUI paths
try:
    import folder_paths
except ImportError:
    folder_paths = None

# Import the new Florence2RectangleDetector
try: 
    from .florence2_scripts.florence2_detector import Florence2RectangleDetector, BoundingBox
except ImportError:
    print("Florence2RectangleDetector not available")

try:
    from .sam2_scripts.sam2_florence_segmentation import SAM2FlorenceSegmenter
except ImportError:
    print("SAM2FlorenceSegmenter not available")

try:
    from .sam2_scripts.sam2_integration import SAM2FlorenceIntegration
except ImportError:
    print("SAM2FlorenceIntegration not available")

try:
    from .sam2_scripts.modeling_florence2 import Florence2LanguageForConditionalGeneration
except ImportError:
    print("Florence2LanguageForConditionalGeneration not available")

try:
    from .florence2_scripts.modern_image_enhancer import ModernImageEnhancer
except ImportError:
    print("ModernImageEnhancer not available")


# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Get the nodes directory
def get_nodes_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


# Import nodes from the nodes directory
nodes_dir = get_nodes_dir("nodes")
for file in os.listdir(nodes_dir):
    if not file.endswith(".py") or file.startswith("__"):
        continue
    
    # Skip downloader nodes - they are now in the download-tools package
    if file.endswith("_downloader.py"):
        print(f"Skipping {file} (moved to download-tools package)")
        continue
        
    name = os.path.splitext(file)[0]
    try:
        # Import the module
        imported_module = importlib.import_module(".nodes.{}".format(name), __name__)
        
        # Extract and update node mappings
        if hasattr(imported_module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
            print(f"Added {len(imported_module.NODE_CLASS_MAPPINGS)} node classes from {name}")
            
        if hasattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
            
        print(f"Loaded node module: {name}")
    except Exception as e:
        print(f"Error loading node module {name}: {str(e)}")
        import traceback
        traceback.print_exc()

# Print summary
print(f"Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
for node_name in NODE_CLASS_MAPPINGS.keys():
    print(f"  - {node_name}")

# Version info
__version__ = "0.1.0"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "Florence2RectangleDetector", "BoundingBox"]