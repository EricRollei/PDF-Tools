#!/usr/bin/env python3
"""
Script to add license headers to all Python files in the PDF_tools project.

This script will add standardized license headers to all .py files that don't already have them.
"""

import os
import re
from pathlib import Path

# License header template
LICENSE_HEADER = '''"""
{module_name}

Description: {description}
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
{third_party_notes}
"""

'''

# Module descriptions based on file patterns
MODULE_DESCRIPTIONS = {
    'pdf_extractor': 'PDF extraction and processing node for ComfyUI with advanced layout detection and quality assessment',
    'gallery_dl_downloader': 'Gallery-dl downloader node for ComfyUI - downloads media from 100+ websites including Instagram, Reddit, Twitter, etc.',
    'yt_dlp_downloader': 'Yt-dlp downloader node for ComfyUI - downloads videos and audio from 1000+ platforms including YouTube, TikTok, etc.',
    'enhanced_layout_parser': 'Enhanced layout parsing node for ComfyUI using Surya OCR, Florence2 vision models, and advanced analysis',
    'florence2': 'Florence2 vision model integration for object detection, image analysis, and caption generation',
    'sam2': 'SAM2 (Segment Anything Model 2) integration for advanced image segmentation',
    'surya': 'Surya OCR integration for multilingual text recognition and layout analysis',
    'eric_rectangle_detector': 'Rectangle and bounding box detection node using AI vision models',
    'paddleocr': 'PaddleOCR integration for advanced OCR with multiple language support',
    'layoutlmv3': 'LayoutLMv3 model integration for document understanding and layout analysis',
}

# Third-party notes for specific modules
THIRD_PARTY_NOTES = {
    'gallery_dl': '- Uses gallery-dl (GNU GPL v2) by Mike Fährmann: https://github.com/mikf/gallery-dl',
    'yt_dlp': '- Uses yt-dlp (Unlicense/Public Domain): https://github.com/yt-dlp/yt-dlp',
    'florence2': '- Uses Florence-2 models (MIT License) by Microsoft: https://huggingface.co/microsoft/Florence-2-large',
    'sam2': '- Uses SAM2 models (Apache 2.0) by Meta AI: https://github.com/facebookresearch/segment-anything-2',
    'surya': '- Uses Surya OCR (GNU GPL v3) by Vik Paruchuri: https://github.com/VikParuchuri/surya',
    'paddleocr': '- Uses PaddleOCR (Apache 2.0) by PaddlePaddle: https://github.com/PaddlePaddle/PaddleOCR',
}

def get_module_description(filepath):
    """Get appropriate description based on filename."""
    filename = Path(filepath).stem.lower()
    
    for key, desc in MODULE_DESCRIPTIONS.items():
        if key in filename:
            return desc
    
    # Default description
    return 'ComfyUI custom node for PDF tools and media processing'

def get_third_party_notes(filepath):
    """Get third-party notes based on filename."""
    filename = Path(filepath).stem.lower()
    notes = []
    
    for key, note in THIRD_PARTY_NOTES.items():
        if key in filename:
            notes.append(note)
    
    if not notes:
        return '- See CREDITS.md for complete list of dependencies'
    
    return '\n'.join(notes) + '\n- See CREDITS.md for complete list of all dependencies'

def has_license_header(content):
    """Check if file already has a license header."""
    license_indicators = [
        'Copyright (c) 2025 Eric Hiss',
        'Dual License',
        'Creative Commons Attribution-NonCommercial',
        'eric@historic.camera'
    ]
    
    # Check first 50 lines
    lines = content.split('\n')[:50]
    first_section = '\n'.join(lines)
    
    return any(indicator in first_section for indicator in license_indicators)

def add_license_header(filepath):
    """Add license header to a Python file if it doesn't have one."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already has license
        if has_license_header(content):
            print(f"SKIP (has license): {filepath}")
            return False
        
        # Get module name from filename
        module_name = Path(filepath).stem.replace('_', ' ').title()
        
        # Get appropriate description and notes
        description = get_module_description(filepath)
        third_party_notes = get_third_party_notes(filepath)
        
        # Create header
        header = LICENSE_HEADER.format(
            module_name=module_name,
            description=description,
            third_party_notes=third_party_notes
        )
        
        # Add header at the beginning
        new_content = header + content
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"ADDED license: {filepath}")
        return True
        
    except Exception as e:
        print(f"ERROR processing {filepath}: {e}")
        return False

def process_directory(root_dir, exclude_dirs=None):
    """Process all Python files in directory."""
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.venv', 'venv', 'env', '.git', 'oldfiles', 'Docs']
    
    root_path = Path(root_dir)
    modified_count = 0
    skipped_count = 0
    
    for py_file in root_path.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
        
        # Skip __init__.py in subdirectories (but process root __init__.py)
        if py_file.name == '__init__.py' and py_file.parent != root_path:
            continue
        
        if add_license_header(py_file):
            modified_count += 1
        else:
            skipped_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Modified: {modified_count} files")
    print(f"Skipped: {skipped_count} files")

if __name__ == "__main__":
    # Get the PDF_tools root directory
    script_dir = Path(__file__).parent
    
    print("Adding license headers to Python files...")
    print(f"Root directory: {script_dir}")
    print("-" * 60)
    
    process_directory(script_dir)
    
    print("\nDone! Review the changes and commit if everything looks good.")
