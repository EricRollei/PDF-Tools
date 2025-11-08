"""
Pdf Extractor V08

Description: PDF extraction and processing node for ComfyUI with advanced layout detection and quality assessment
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
Enhanced PDF Extractor v0.8.0 - Integrated with Shared Analysis Engine
Major upgrade incorporating Surya Layout + Florence2 + Modern Image Enhancement

Key Features v0.8.0:
- Shared analysis engine (Surya Layout + Florence2)
- Smart text layer removal before image detection
- Modern image enhancement for OCR quality
- Intelligent double-page spread detection using semantic analysis
- Full-page image detection and handling
- Smart cropping recommendations (avoid captions)
- Enhanced text extraction using layout analysis

Author: Eric Hiss (GitHub: EricRollei)
Enhanced by: Claude Sonnet 4 AI Assistant
Version: 0.8.0
Date: June 2025
"""

import os
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import torch
from io import BytesIO

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available - some image quality assessments will be limited")

# PDF processing imports
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("❌ PyMuPDF not available")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("❌ PyPDF2 not available")

# ComfyUI imports
try:
    import folder_paths
    COMFYUI_BASE_PATH = os.path.dirname(folder_paths.models_dir)
    
    def tensor_to_PIL(tensor):
        return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def PIL_to_tensor(pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
        
except ImportError:
    COMFYUI_BASE_PATH = "."
    print("⚠️ ComfyUI folder_paths not available")
    
    def tensor_to_PIL(tensor):
        if isinstance(tensor, torch.Tensor):
            return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        return tensor
    
    def PIL_to_tensor(pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

# Import shared analysis engine and modern enhancer
try:
    from PDF_tools.florence2_scripts.analysis_engine import (
        create_content_analyzer, 
        analyze_for_pdf_extraction,
        analyze_for_smart_cropping
    )
    ANALYSIS_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from .florence2_scripts.analysis_engine import (
            create_content_analyzer,
            analyze_for_pdf_extraction, 
            analyze_for_smart_cropping
        )
        ANALYSIS_ENGINE_AVAILABLE = True
    except ImportError:
        print("❌ Analysis Engine not available")
        ANALYSIS_ENGINE_AVAILABLE = False

try:
    from PDF_tools.florence2_scripts.modern_image_enhancer import ModernImageEnhancer
    MODERN_ENHANCER_AVAILABLE = True
except ImportError:
    try:
        from .florence2_scripts.modern_image_enhancer import ModernImageEnhancer
        MODERN_ENHANCER_AVAILABLE = True
    except ImportError:
        print("❌ Modern Image Enhancer not available")
        MODERN_ENHANCER_AVAILABLE = False


class SpreadDetectionStrategy(Enum):
    """Strategy for detecting double-page spreads"""
    SEMANTIC_LAYOUT = "semantic_layout"    # Use Surya Layout semantic analysis
    FLORENCE2_BOXES = "florence2_boxes"    # Use Florence2 bounding box analysis
    HYBRID = "hybrid"                      # Combine both methods
    DISABLED = "disabled"                  # No spread detection


@dataclass
class PDFPage:
    """Enhanced PDF page data structure"""
    page_number: int
    image: Image.Image
    original_size: Tuple[int, int]
    text_layers_removed: bool = False
    analysis_results: Optional[Dict] = None
    detected_images: List[Dict] = None
    detected_text_regions: List[Dict] = None
    is_spread_candidate: bool = False
    spread_partner_page: Optional[int] = None
    enhancement_applied: bool = False
    original_text: str = "" 
    native_images: List[Dict] = None

class EnhancedPDFExtractorNode:
    """
    Enhanced PDF Extractor with integrated analysis engine and modern enhancement
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_path": ("STRING", {"default": "", "multiline": False}),
                "output_directory": ("STRING", {"default": "output/pdf_extraction", "multiline": False}),
                "remove_text_layers": ("BOOLEAN", {"default": True}),
                "enable_image_enhancement": ("BOOLEAN", {"default": True}),
                "join_spreads": ("BOOLEAN", {"default": True}),  # Changed from detect_spreads
                "extract_text": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "start_page": ("INT", {"default": 1, "min": 1}),
                "end_page": ("INT", {"default": -1, "min": -1}),
                "min_image_size": ("INT", {"default": 200, "min": 50, "max": 1000, "step": 10}),
                "dpi": ("INT", {"default": 300, "min": 150, "max": 600, "step": 50}),
                "surya_confidence": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "florence2_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "enhancement_profile": ([
                    "Digital Magazine", "Scanned Photo", "Vintage/Compressed", "Minimal"
                ], {"default": "Digital Magazine"}),
                "enhancement_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "spread_detection_strategy": ([
                    "semantic_layout", "florence2_boxes", "hybrid", "disabled"
                ], {"default": "hybrid"}),
                "avoid_full_page_boxes": ("BOOLEAN", {"default": True}),
                "smart_crop_captions": ("BOOLEAN", {"default": True}),
                "save_images_to_disk": ("BOOLEAN", {"default": True}),
                "save_text_to_file": ("BOOLEAN", {"default": True}),
                "debug_mode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST", "STRING", "DICT", "STRING")  # Added output path
    RETURN_NAMES = ("extracted_images", "enhanced_images", "page_analysis", "extracted_text", "extraction_stats", "output_path")

    FUNCTION = "extract_enhanced"
    CATEGORY = "Enhanced PDF/Extraction"

    def __init__(self):
        self.analysis_engine = None
        self.image_enhancer = None
        self.debug_mode = False
        
        print("🚀 Enhanced PDF Extractor v08 - Analysis Engine Integration")

    def _init_analysis_engine(self, debug_mode: bool = False):
        """Initialize the shared analysis engine"""
        if not ANALYSIS_ENGINE_AVAILABLE:
            print("❌ Analysis Engine not available")
            return False
        
        try:
            self.analysis_engine = create_content_analyzer(
                enable_surya=True,
                enable_florence2=True,
                enable_ocr=True,
                debug_mode=debug_mode
            )
            if debug_mode:
                print("✅ Analysis Engine initialized")
            return True
        except Exception as e:
            print(f"❌ Analysis Engine initialization failed: {e}")
            return False

    def _init_image_enhancer(self, debug_mode: bool = False):
        """Initialize the modern image enhancer with GPU optimization"""
        if not MODERN_ENHANCER_AVAILABLE:
            print("❌ Modern Image Enhancer not available")
            return False
        
        try:
            # Initialize with GPU acceleration enabled
            self.image_enhancer = ModernImageEnhancer(
                debug_mode=debug_mode, 
                use_gpu=True  # Enable GPU acceleration
            )
            
            if debug_mode:
                print("✅ Modern Image Enhancer initialized")
                
                # Check GPU status and performance
                if hasattr(self.image_enhancer, 'validate_configuration'):
                    config = self.image_enhancer.validate_configuration()
                    gpu_info = config.get('gpu_acceleration', {})
                    
                    print(f"   🚀 GPU Available: {gpu_info.get('available', False)}")
                    if gpu_info.get('available'):
                        gpu_mem = gpu_info.get('gpu_memory', {})
                        if isinstance(gpu_mem, dict):
                            print(f"   💾 GPU Memory: {gpu_mem.get('total_gb', 0):.1f}GB total")
                            print(f"   💾 GPU Memory Free: {gpu_mem.get('free_gb', 0):.1f}GB")
                    
                    algorithms = config.get('algorithms', {})
                    print(f"   🌊 Wavelets: {algorithms.get('wavelets', False)}")
                    print(f"   🔬 Scikit-image: {algorithms.get('skimage', False)}")
                
            return True
            
        except Exception as e:
            print(f"❌ Image Enhancer initialization failed: {e}")
            return False

    def extract_enhanced(self, pdf_path: str, output_directory: str = "output/pdf_extraction",
                        remove_text_layers: bool = True, enable_image_enhancement: bool = True, 
                        join_spreads: bool = True, extract_text: bool = True,  
                        start_page: int = 1, end_page: int = -1, min_image_size: int = 200,
                        dpi: int = 300, surya_confidence: float = 0.3, florence2_confidence: float = 0.5, 
                        enhancement_profile: str = "Digital Magazine", enhancement_strength: float = 1.0, 
                        spread_detection_strategy: str = "hybrid",
                        avoid_full_page_boxes: bool = True, smart_crop_captions: bool = True,
                        save_images_to_disk: bool = True, save_text_to_file: bool = True,
                        debug_mode: bool = True) -> Tuple[List, List, List, str, Dict, str]:
        """Enhanced PDF extraction with GPU optimization and monitoring"""
        
        self.debug_mode = debug_mode
        
        # GPU Performance Benchmarking (optional)
        if debug_mode and enable_image_enhancement:
            print(f"🚀 GPU Performance Check:")
            if self.image_enhancer and hasattr(self.image_enhancer, 'validate_configuration'):
                config = self.image_enhancer.validate_configuration()
                gpu_available = config.get('gpu_acceleration', {}).get('available', False)
                
                if gpu_available:
                    print(f"   ✅ GPU acceleration ready")
                    
                    # Optional: Quick benchmark with a test image
                    try:
                        test_image = Image.new('RGB', (512, 512), 'white')
                        if hasattr(self.image_enhancer, 'benchmark_performance'):
                            benchmark = self.image_enhancer.benchmark_performance(test_image, iterations=2)
                            speedup = benchmark.get('speedup', 'N/A')
                            print(f"   🏃 GPU Speedup: {speedup}x faster than CPU")
                    except Exception as e:
                        if debug_mode:
                            print(f"   ⚠️ Benchmark failed: {e}")
                else:
                    print(f"   💻 Using CPU processing (GPU not available)")
        
        # Hardcoded proven Florence2 prompt
        florence2_image_prompt = "rectangular images in page"
        
        # Initialize engines
        if not self._init_analysis_engine(debug_mode):
            raise RuntimeError("Analysis Engine initialization failed")
        
        if enable_image_enhancement and not self._init_image_enhancer(debug_mode):
            print("⚠️ Image enhancement disabled due to initialization failure")
            enable_image_enhancement = False

        # Create output directory
        output_path = self._create_output_directory(pdf_path, output_directory, debug_mode)

        if debug_mode:
            print(f"🔍 Enhanced PDF Extraction Started:")
            print(f"   📄 PDF: {pdf_path}")
            print(f"   📁 Output: {output_path}")
            print(f"   🎯 DPI: {dpi}")
            print(f"   📏 Min image size: {min_image_size}px")
            print(f"   📝 Remove text layers: {remove_text_layers}")
            print(f"   🎨 Enhancement: {enable_image_enhancement}")
            print(f"   📖 Join spreads: {join_spreads}")

        # Load and process PDF
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # GPU Memory monitoring during processing
        if debug_mode and enable_image_enhancement:
            initial_memory = self._monitor_gpu_memory(debug_mode)
        
        # Extract pages as images
        pages = self._extract_pdf_pages(
            pdf_path, dpi, start_page, end_page, 
            remove_text_layers, debug_mode
        )

        if not pages:
            raise RuntimeError("No pages extracted from PDF")

        if debug_mode:
            print(f"📄 Extracted {len(pages)} pages from PDF")

        # Calculate min_image_area from min_image_size
        min_image_area = min_image_size * min_image_size  # Convert to area

        # Analyze each page with the analysis engine
        analyzed_pages = []  
        all_extracted_text = []

        for i, page in enumerate(pages):
            if debug_mode:
                print(f"\n🔍 Analyzing page {page.page_number}...")

            # Run comprehensive analysis
            analysis = analyze_for_pdf_extraction(page.image, debug_mode=debug_mode)
            page.analysis_results = analysis

            # Extract semantic regions
            semantic_regions = analysis["semantic_regions"]
            page.detected_images = semantic_regions["image_regions"]
            page.detected_text_regions = (
                semantic_regions["text_regions"] + 
                semantic_regions["header_regions"] +
                semantic_regions["caption_regions"]
            )

            # Add the analyzed page to the list!
            analyzed_pages.append(page)

            # Extract text - PREFER ORIGINAL DIGITAL TEXT!
            if extract_text:
                if page.original_text and page.text_layers_removed:
                    # Use the original digital text (best quality)
                    page_text = f"=== Page {page.page_number} ===\n[ORIGINAL_TEXT] {page.original_text}"
                    all_extracted_text.append(page_text)
                    
                    if debug_mode:
                        print(f"   📝 Using original digital text ({len(page.original_text)} chars)")
                        
                elif page.detected_text_regions:
                    # Fallback to OCR on detected regions
                    if self.analysis_engine:  # FIX: Add safety check
                        extracted_texts = self.analysis_engine.extract_text_from_regions(
                            page.image, page.detected_text_regions, ocr_engine="tesseract"
                        )
                        
                        # Format extracted text
                        page_text = self._format_extracted_text(extracted_texts, page.page_number)
                        all_extracted_text.append(page_text)
                        
                        if debug_mode:
                            print(f"   📝 Using OCR text extraction")
                    else:
                        page_text = f"=== Page {page.page_number} ===\n[ERROR] Analysis engine not available for text extraction"
                        all_extracted_text.append(page_text)
                        
                        if debug_mode:
                            print(f"   ⚠️ Analysis engine not available for text extraction on page {page.page_number}")
                else:
                    # FIX: Add empty page text so list indices stay aligned
                    page_text = f"=== Page {page.page_number} ===\n[NO_TEXT] No text found on this page"
                    all_extracted_text.append(page_text)
                    
                    if debug_mode:
                        print(f"   📝 No text found on page {page.page_number}")

        # Detect and optionally join double-page spreads
        if join_spreads:
            analyzed_pages = self._detect_and_join_spreads(
                analyzed_pages, SpreadDetectionStrategy(spread_detection_strategy), debug_mode
            )
        
        if debug_mode:
            print(f"\n📊 Analysis complete: {len(analyzed_pages)} pages ready for image processing")
            
        # Extract and enhance images
        extracted_images = []
        enhanced_images = []
        page_analysis = []
        saved_image_paths = []

        # FIX: Initialize crop_recommendations (was missing)
        crop_recommendations = []  # You may want to implement smart cropping logic here

        # Process images - USE THE EXISTING _process_page_images METHOD (not the non-existent batch method)
        for page in analyzed_pages:
            if debug_mode:
                print(f"\n🖼️  Processing images for page {page.page_number}...")
                
            # Monitor memory before processing each page
            if debug_mode and enable_image_enhancement:
                self._monitor_gpu_memory(debug_mode)
            
            # FIX: Use the existing method that actually exists in your code
            page_images, page_enhanced, page_saved_paths = self._process_page_images(
                page, min_image_area, florence2_confidence, 
                avoid_full_page_boxes, crop_recommendations,
                enable_image_enhancement, enhancement_profile, 
                enhancement_strength, output_path, save_images_to_disk, debug_mode
            )
            
            extracted_images.extend(page_images)
            enhanced_images.extend(page_enhanced)
            saved_image_paths.extend(page_saved_paths)
            
            # Create page analysis summary
            page_summary = self._create_page_summary(page, len(page_images))
            page_analysis.append(page_summary)
        
        # Final GPU memory check
        if debug_mode and enable_image_enhancement:
            final_memory = self._monitor_gpu_memory(debug_mode)
            print(f"\n🔧 GPU Processing Summary:")
            print(f"   🖼️  Total images processed: {len(extracted_images)}")
            print(f"   ✨ Enhanced images: {len(enhanced_images)}")
            
            # Calculate enhancement efficiency
            if len(extracted_images) > 0:
                enhancement_rate = len(enhanced_images) / len(extracted_images) * 100
                print(f"   📊 Enhancement success rate: {enhancement_rate:.1f}%")

        # Combine extracted text
        combined_text = "\n\n".join(all_extracted_text) if all_extracted_text else ""

        # Save text files in multiple formats
        text_files = {}
        if save_text_to_file and all_extracted_text:
            text_files = self._save_text_to_files(all_extracted_text, analyzed_pages, output_path, pdf_path, debug_mode)

        # Create extraction statistics
        extraction_stats = self._create_extraction_stats(
            analyzed_pages, len(extracted_images), len(enhanced_images), 
            len(combined_text), output_path, saved_image_paths, text_files, debug_mode
        )

        # Save comprehensive analysis files
        analysis_files = self._save_analysis_files(analyzed_pages, extraction_stats, output_path, pdf_path, debug_mode)

        if debug_mode:
            print(f"\n🎯 Extraction Complete:")
            print(f"   🖼️  Total images: {len(extracted_images)}")
            print(f"   ✨ Enhanced images: {len(enhanced_images)}")
            print(f"   📝 Text characters: {len(combined_text)}")
            print(f"   📊 Pages analyzed: {len(page_analysis)}")
            print(f"   📁 Output saved to: {output_path}")
        
        if debug_mode:
            print(f"\n💾 File Saving Summary:")
            print(f"   📁 Output directory: {output_path}")
            print(f"   🖼️  Images to save: {len(extracted_images)} original, {len(enhanced_images)} enhanced")
            print(f"   💾 Save to disk enabled: {save_images_to_disk}")
            print(f"   📝 Text to save: {len(combined_text)} characters")
            print(f"   💾 Save text enabled: {save_text_to_file}")
            if saved_image_paths:
                print(f"   ✅ Files saved: {len(saved_image_paths)} total")
                print(f"       📂 First few: {saved_image_paths[:3]}")
            else:
                print(f"   ⚠️  No files were saved!")

        return (extracted_images, enhanced_images, page_analysis, combined_text, extraction_stats, output_path)
        

    def _create_output_directory(self, pdf_path: str, output_directory: str, debug_mode: bool) -> str:
        """Create output directory structure"""
        
        # Get PDF filename without extension
        pdf_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique output directory
        unique_output_dir = os.path.join(output_directory, f"{pdf_name}_{timestamp}")
        
        # Create directories
        os.makedirs(unique_output_dir, exist_ok=True)
        os.makedirs(os.path.join(unique_output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(unique_output_dir, "enhanced"), exist_ok=True)
        os.makedirs(os.path.join(unique_output_dir, "analysis"), exist_ok=True)
        
        if debug_mode:
            print(f"📁 Created output directory: {unique_output_dir}")
        
        return unique_output_dir


    def _extract_pdf_pages(self, pdf_path: str, dpi: int, start_page: int, 
                        end_page: int, remove_text_layers: bool, debug_mode: bool) -> List[PDFPage]:
        """Extract pages from PDF with optional text layer removal and native image extraction"""
        
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not available for PDF processing")

        pages = []
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
                if end_page == -1:
                    end_page = total_pages
                else:
                    end_page = min(end_page, total_pages)
                
                start_page = max(1, start_page)
                
                if debug_mode:
                    print(f"📄 PDF has {total_pages} pages, extracting {start_page}-{end_page}")

                for page_num in range(start_page - 1, end_page):
                    page = doc[page_num]
                    
                    # FIRST: Extract native embedded images using optimized PyMuPDF method
                    native_images = self._extract_native_images(page, page_num + 1, debug_mode)
                    
                    # SECOND: Extract the original digital text before removing anything!
                    original_text = ""
                    if remove_text_layers:  # Only extract if we're going to remove it
                        original_text = self._extract_original_text(page, debug_mode)
                    
                    # THIRD: Remove text layers if requested
                    if remove_text_layers:
                        page = self._remove_text_layers(page, debug_mode)
                    
                    # FOURTH: Render page to image for analysis
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("ppm")
                    
                    # Convert to PIL Image
                    image = Image.open(BytesIO(img_data))
                    
                    # Create PDFPage object
                    pdf_page = PDFPage(
                        page_number=page_num + 1,
                        image=image,
                        original_size=image.size,
                        text_layers_removed=remove_text_layers,
                        original_text=original_text,
                        native_images=native_images  # Store extracted native images
                    )
                    
                    pages.append(pdf_page)
                    
                    if debug_mode:
                        print(f"   📄 Page {page_num + 1}: {image.size[0]}x{image.size[1]} pixels")
                        if original_text:
                            print(f"   📝 Saved {len(original_text)} characters of original text")
                        if native_images:
                            print(f"   🖼️  Found {len(native_images)} native embedded images")
                            total_native_area = sum(img["area"] for img in native_images)
                            print(f"       📊 Total native image area: {total_native_area:,} pixels")
                            
                            # Quality summary
                            quality_summary = {}
                            for img in native_images:
                                rec = img["enhancement_recommendation"]
                                quality_summary[rec] = quality_summary.get(rec, 0) + 1
                            
                            print(f"       ✨ Enhancement needs: {dict(quality_summary)}")

        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF pages: {e}")

        return pages

    def _extract_native_images(self, page: fitz.Page, page_number: int, debug_mode: bool) -> List[Dict]:
        """Extract native embedded images from PDF page using PyMuPDF's optimized method"""
        native_images = []
        
        try:
            # Get all images on the page using PyMuPDF's optimized method
            image_list = page.get_images()
            
            if debug_mode and image_list:
                print(f"   🖼️  Found {len(image_list)} embedded images on page {page_number}")
            
            for image_index, img in enumerate(image_list):
                try:
                    # Extract image using PyMuPDF's method
                    xref = img[0]  # get the XREF of the image
                    pix = fitz.Pixmap(page.parent, xref)  # create a Pixmap
                    
                    # Handle CMYK images - convert to RGB first
                    if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    # Convert Pixmap to PIL Image
                    if pix.alpha:
                        # Handle images with alpha channel
                        img_data = pix.tobytes("ppm")
                        image_pil = Image.open(BytesIO(img_data))
                    else:
                        # Direct conversion for images without alpha
                        img_data = pix.tobytes("ppm")
                        image_pil = Image.open(BytesIO(img_data))
                    
                    # Get image placement on page - try to find actual placement
                    image_bbox = self._find_image_bbox_on_page(page, xref, image_pil.size, debug_mode)
                    
                    # Assess image quality for enhancement decisions
                    quality_metrics = self._assess_native_image_quality(image_pil, debug_mode)
                    
                    # Store native image information
                    native_image_info = {
                        "image": image_pil,
                        "bbox": image_bbox,
                        "xref": xref,
                        "format": "png",  # We convert everything to PNG for consistency
                        "size": image_pil.size,
                        "area": image_pil.width * image_pil.height,
                        "extraction_method": "native_embedded_pymupdf",
                        "confidence": 1.0,  # Native images have perfect confidence
                        "page_number": page_number,
                        "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                        "has_alpha": bool(pix.alpha),
                        "original_format": img[8] if len(img) > 8 else "unknown",  # Original format from metadata
                        "quality_metrics": quality_metrics,  # NEW: Quality assessment
                        "enhancement_recommendation": quality_metrics["enhancement_recommendation"]  # NEW: Enhancement advice
                    }
                    native_images.append(native_image_info)
                    
                    if debug_mode:
                        print(f"       📷 Native image {image_index + 1}: {image_pil.size}, colorspace: {native_image_info['colorspace']}")
                        print(f"           📐 Bbox: {image_bbox}")
                        print(f"           📋 Format: {native_image_info['original_format']}, Alpha: {native_image_info['has_alpha']}")
                        print(f"           ✨ Enhancement: {quality_metrics['enhancement_recommendation']} (quality: {quality_metrics['quality_score']:.2f})")
                    
                    # Clean up pixmap
                    pix = None
                    
                except Exception as e:
                    if debug_mode:
                        print(f"       ⚠️ Failed to extract native image {image_index + 1}: {e}")
                    continue
                    
        except Exception as e:
            if debug_mode:
                print(f"   ⚠️ Native image extraction failed for page {page_number}: {e}")
        
        return native_images

    def _assess_native_image_quality(self, image_pil: Image.Image, debug_mode: bool) -> Dict[str, Any]:
        """Assess the quality of a native embedded image to determine enhancement needs"""
        try:
            img_array = np.array(image_pil)
            
            # Calculate quality metrics
            width, height = image_pil.size
            total_pixels = width * height
            
            # 1. Resolution assessment
            is_high_res = total_pixels > 1000000  # > 1MP
            is_very_high_res = total_pixels > 4000000  # > 4MP
            
            # 2. Sharpness assessment (Laplacian variance)
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2).astype(np.uint8)
            else:
                gray = img_array
            
            # Use cv2 if available, otherwise use numpy approximation
            if CV2_AVAILABLE:
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            else:
                # Numpy approximation of Laplacian
                laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                from scipy import ndimage
                try:
                    laplacian = ndimage.convolve(gray.astype(float), laplacian_kernel)
                    laplacian_var = np.var(laplacian)
                except ImportError:
                    # Ultimate fallback - use gradient magnitude
                    grad_x = np.gradient(gray.astype(float), axis=1)
                    grad_y = np.gradient(gray.astype(float), axis=0)
                    laplacian_var = np.var(np.sqrt(grad_x**2 + grad_y**2))
            
            is_sharp = laplacian_var > 500  # Threshold for sharpness
            
            # 3. Noise assessment (standard deviation of pixel values)
            noise_level = np.std(img_array)
            is_noisy = noise_level < 5  # Very low std might indicate compression artifacts
            
            # 4. Dynamic range assessment
            if len(img_array.shape) == 3:
                dynamic_range = np.max(img_array) - np.min(img_array)
            else:
                dynamic_range = np.max(img_array) - np.min(img_array)
            
            has_good_range = dynamic_range > 200  # Good contrast
            
            # 5. Color depth assessment
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)) if len(img_array.shape) == 3 else len(np.unique(img_array))
            has_rich_colors = unique_colors > 1000
            
            # Calculate overall quality score (0-1)
            quality_factors = [
                is_high_res * 0.3,
                is_very_high_res * 0.2,
                is_sharp * 0.2,
                (not is_noisy) * 0.1,
                has_good_range * 0.1,
                has_rich_colors * 0.1
            ]
            quality_score = sum(quality_factors)
            
            # Determine enhancement recommendation
            if quality_score >= 0.8:
                enhancement_recommendation = "none"  # Excellent quality
                enhancement_strength_override = 0.0
            elif quality_score >= 0.6:
                enhancement_recommendation = "minimal"  # Good quality, light touch
                enhancement_strength_override = 0.3
            elif quality_score >= 0.4:
                enhancement_recommendation = "light"  # Moderate quality
                enhancement_strength_override = 0.6
            else:
                enhancement_recommendation = "standard"  # Poor quality, needs help
                enhancement_strength_override = 1.0
            
            return {
                "quality_score": quality_score,
                "resolution": {"width": width, "height": height, "megapixels": total_pixels / 1000000},
                "is_high_resolution": is_high_res,
                "is_very_high_resolution": is_very_high_res,
                "sharpness_score": laplacian_var,
                "is_sharp": is_sharp,
                "noise_level": noise_level,
                "is_noisy": is_noisy,
                "dynamic_range": dynamic_range,
                "has_good_contrast": has_good_range,
                "unique_colors": unique_colors,
                "has_rich_colors": has_rich_colors,
                "enhancement_recommendation": enhancement_recommendation,
                "enhancement_strength_override": enhancement_strength_override,
                "assessment_method": "cv2" if CV2_AVAILABLE else "numpy_fallback"
            }
            
        except Exception as e:
            if debug_mode:
                print(f"           ⚠️ Quality assessment failed: {e}")
            
            # Fallback - assume native images are good quality
            return {
                "quality_score": 0.8,
                "enhancement_recommendation": "minimal",
                "enhancement_strength_override": 0.3,
                "assessment_failed": True
            }

    def _process_page_images_batch_optimized(self, page: PDFPage, min_image_area: int, 
                                            florence2_confidence: float, avoid_full_page_boxes: bool,
                                            crop_recommendations: List[Dict], enable_enhancement: bool,
                                            enhancement_profile: str, enhancement_strength: float,
                                            output_path: str, save_to_disk: bool, debug_mode: bool) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        """GPU-optimized batch processing version of _process_page_images"""
        
        # First, get all the detections (same logic as before)
        final_detections = self._get_all_detections(page, min_image_area, florence2_confidence, 
                                                    avoid_full_page_boxes, crop_recommendations, debug_mode)
        
        if not final_detections:
            return [], [], []
        
        # Extract all images first (batch-friendly)
        extracted_images = []
        native_images = []
        enhancement_settings = []
        
        for i, detection in enumerate(final_detections):
            try:
                source = detection.get("detection_source", "unknown")
                
                # Handle native images differently
                if source == "native_embedded":
                    cropped = detection["native_image"]
                    native_images.append(cropped)
                    
                    # Determine enhancement settings based on quality
                    quality_metrics = detection.get("quality_metrics", {})
                    enhancement_rec = quality_metrics.get("enhancement_recommendation", "minimal")
                    strength_override = quality_metrics.get("enhancement_strength_override", enhancement_strength)
                    
                    if enhancement_rec == "none":
                        enhancement_settings.append(None)  # Skip enhancement
                    else:
                        profile_override = "Minimal" if enhancement_rec == "minimal" else enhancement_profile
                        enhancement_settings.append({
                            "profile": profile_override,
                            "strength": strength_override,
                            "source": "native"
                        })
                else:
                    # Regular cropping from page image
                    bbox = detection.get("bbox", [0, 0, 0, 0])
                    if len(bbox) < 4:
                        continue
                    
                    # Bounds checking
                    x1, y1, x2, y2 = bbox
                    img_width, img_height = page.image.size
                    
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(x1, min(x2, img_width))
                    y2 = max(y1, min(y2, img_height))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    cropped = page.image.crop([x1, y1, x2, y2])
                    extracted_images.append(cropped)
                    
                    # Standard enhancement for rendered images
                    enhancement_settings.append({
                        "profile": enhancement_profile,
                        "strength": enhancement_strength,
                        "source": "rendered"
                    })
                    
            except Exception as e:
                if debug_mode:
                    print(f"       ❌ Failed to extract detection {i}: {e}")
                continue
        
        # Combine all images for processing
        all_images = native_images + extracted_images
        
        if not all_images:
            return [], [], []
        
        if debug_mode:
            print(f"       🔄 Processing {len(all_images)} images ({len(native_images)} native, {len(extracted_images)} rendered)")
        
        # BATCH ENHANCEMENT for GPU efficiency
        page_enhanced = []
        if enable_enhancement and self.image_enhancer and hasattr(self.image_enhancer, 'enhance_images_batch'):
            # Use batch processing if available
            try:
                enhanced_results = self.image_enhancer.enhance_images_batch(
                    all_images, 
                    [setting for setting in enhancement_settings if setting is not None],
                    debug_mode=debug_mode
                )
                page_enhanced = [PIL_to_tensor(img) for img in enhanced_results]
                
                if debug_mode:
                    print(f"       ✨ Batch enhancement completed: {len(enhanced_results)} images")
                    
            except Exception as e:
                if debug_mode:
                    print(f"       ⚠️ Batch enhancement failed, falling back to individual: {e}")
                # Fallback to individual processing
                page_enhanced = self._enhance_images_individually(all_images, enhancement_settings, debug_mode)
        
        else:
            # Individual enhancement (fallback)
            page_enhanced = self._enhance_images_individually(all_images, enhancement_settings, debug_mode)
        
        # Convert originals to tensors
        page_images = [PIL_to_tensor(img) for img in all_images]
        
        # Save to disk if requested
        saved_paths = []
        if save_to_disk:
            saved_paths = self._save_processed_images(all_images, page_enhanced, page, 
                                                    final_detections, output_path, debug_mode)
        
        return page_images, page_enhanced, saved_paths

    def _enhance_images_individually(self, images: List[Image.Image], 
                                    enhancement_settings: List[Optional[Dict]], 
                                    debug_mode: bool) -> List[torch.Tensor]:
        """Fallback individual enhancement processing"""
        enhanced_images = []
        
        for i, (image, settings) in enumerate(zip(images, enhancement_settings)):
            if settings is None:
                # Skip enhancement
                enhanced_images.append(PIL_to_tensor(image))
                if debug_mode:
                    print(f"           ⏭️ Skipped enhancement for image {i+1} (excellent quality)")
            else:
                try:
                    enhanced = self.image_enhancer.enhance_image(
                        image,
                        profile=settings["profile"],
                        strength=settings["strength"]
                    )
                    enhanced_images.append(PIL_to_tensor(enhanced))
                    
                    if debug_mode:
                        source = settings["source"]
                        print(f"           ✨ Enhanced {source} image {i+1}: {settings['profile']} ({settings['strength']:.1f})")
                        
                except Exception as e:
                    if debug_mode:
                        print(f"           ⚠️ Enhancement failed for image {i+1}: {e}")
                    enhanced_images.append(PIL_to_tensor(image))
        
        return enhanced_images

    def _extract_original_text(self, page: fitz.Page, debug_mode: bool) -> str:
        """Extract original digital text from PDF page before removal"""
        try:
            # Get text with position information
            text_dict = page.get_text("dict")
            
            extracted_lines = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    block_lines = []
                    
                    for line in block["lines"]:
                        line_text = ""
                        for span in line.get("spans", []):
                            span_text = span.get("text", "").strip()
                            if span_text:
                                line_text += span_text + " "
                        
                        if line_text.strip():
                            block_lines.append(line_text.strip())
                    
                    if block_lines:
                        # Join lines in this block
                        block_text = "\n".join(block_lines)
                        extracted_lines.append(block_text)
            
            # Join all blocks
            full_text = "\n\n".join(extracted_lines)
            
            if debug_mode and full_text:
                print(f"   💾 Extracted {len(full_text)} characters of original text")
            
            return full_text
            
        except Exception as e:
            if debug_mode:
                print(f"   ⚠️ Failed to extract original text: {e}")
            return ""
            

    def _remove_text_layers(self, page: fitz.Page, debug_mode: bool) -> fitz.Page:
        """Remove text and annotation layers from PDF page"""
        try:
            # Get all text blocks and annotations
            text_blocks = page.get_text("dict")
            annotations = page.annots()
            
            # Remove text blocks
            for block in text_blocks.get("blocks", []):
                if "lines" in block:  # Text block
                    # Create redaction annotation to remove text
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            bbox = span.get("bbox")
                            if bbox:
                                # Add redaction
                                page.add_redact_annot(fitz.Rect(bbox), fill=(1, 1, 1))
            
            # Remove existing annotations
            for annot in annotations:
                page.delete_annot(annot)
            
            # Apply redactions
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            
            if debug_mode:
                print(f"   🗑️  Removed text layers from page (text was saved first)")
            
            return page
            
        except Exception as e:
            if debug_mode:
                print(f"   ⚠️ Failed to remove text layers: {e}")
            return page

    def _detect_and_join_spreads(self, pages: List[PDFPage], strategy: SpreadDetectionStrategy, 
                            debug_mode: bool) -> List[PDFPage]:
        """Detect and join double-page spreads into single images"""
        
        if strategy == SpreadDetectionStrategy.DISABLED:
            return pages

        if debug_mode:
            print(f"\n📖 Detecting and joining spreads using strategy: {strategy.value}")

        # First detect spreads (existing logic)
        spread_pairs = []
        
        for i in range(len(pages) - 1):
            left_page = pages[i]
            right_page = pages[i + 1]
            
            # Skip if either page is already part of a spread
            if left_page.is_spread_candidate or right_page.is_spread_candidate:
                continue
            
            is_spread = False

            if strategy in [SpreadDetectionStrategy.SEMANTIC_LAYOUT, SpreadDetectionStrategy.HYBRID]:
                is_spread |= self._detect_spread_semantic(left_page, right_page, debug_mode)

            if strategy in [SpreadDetectionStrategy.FLORENCE2_BOXES, SpreadDetectionStrategy.HYBRID]:
                is_spread |= self._detect_spread_florence2(left_page, right_page, debug_mode)

            if is_spread:
                left_page.is_spread_candidate = True
                left_page.spread_partner_page = right_page.page_number
                right_page.is_spread_candidate = True
                right_page.spread_partner_page = left_page.page_number
                
                spread_pairs.append((i, i + 1))  # Store indices
                
                if debug_mode:
                    print(f"   📖 Detected spread: pages {left_page.page_number}-{right_page.page_number}")

        if debug_mode:
            print(f"📖 Total spreads detected: {len(spread_pairs)}")

        # Now join the spreads
        if spread_pairs:
            joined_pages = self._join_spread_pages(pages, spread_pairs, debug_mode)
            return joined_pages
        
        return pages

    def _join_spread_pages(self, pages: List[PDFPage], spread_pairs: List[Tuple[int, int]], 
                        debug_mode: bool) -> List[PDFPage]:
        """Join detected spread pages into single wide images"""
        
        if debug_mode:
            print(f"\n🔗 Joining {len(spread_pairs)} detected spreads...")
        
        joined_pages = []
        processed_indices = set()
        
        for i, page in enumerate(pages):
            if i in processed_indices:
                continue  # Skip pages that were already joined
            
            # Check if this page is the left page of a spread
            is_left_spread = any(pair[0] == i for pair in spread_pairs)
            
            if is_left_spread:
                # Find the corresponding right page
                right_index = next(pair[1] for pair in spread_pairs if pair[0] == i)
                left_page = pages[i]
                right_page = pages[right_index]
                
                # Join the two pages
                joined_page = self._create_joined_spread_page(left_page, right_page, debug_mode)
                joined_pages.append(joined_page)
                
                # Mark both pages as processed
                processed_indices.add(i)
                processed_indices.add(right_index)
                
                if debug_mode:
                    print(f"   🔗 Joined pages {left_page.page_number}-{right_page.page_number}")
            
            else:
                # Regular single page (not part of a spread)
                joined_pages.append(page)
        
        if debug_mode:
            print(f"🔗 Spread joining complete: {len(pages)} pages -> {len(joined_pages)} pages")
        
        return joined_pages

    def _create_joined_spread_page(self, left_page: PDFPage, right_page: PDFPage, 
                                debug_mode: bool) -> PDFPage:
        """Create a single PDFPage by joining two spread pages horizontally"""
        
        # Get image dimensions
        left_img = left_page.image
        right_img = right_page.image
        
        # Calculate joined image size
        max_height = max(left_img.height, right_img.height)
        total_width = left_img.width + right_img.width
        
        # Create new joined image
        joined_image = Image.new('RGB', (total_width, max_height), 'white')
        
        # Paste left and right images
        joined_image.paste(left_img, (0, 0))
        joined_image.paste(right_img, (left_img.width, 0))
        
        # Combine analysis results
        combined_analysis = self._combine_spread_analysis(left_page, right_page, left_img.width, debug_mode)
        
        # Create joined page object
        joined_page = PDFPage(
            page_number=f"{left_page.page_number}-{right_page.page_number}",  # Combined page number
            image=joined_image,
            original_size=joined_image.size,
            text_layers_removed=left_page.text_layers_removed and right_page.text_layers_removed,
            analysis_results=combined_analysis,
            detected_images=combined_analysis["semantic_regions"]["image_regions"],
            detected_text_regions=(
                combined_analysis["semantic_regions"]["text_regions"] + 
                combined_analysis["semantic_regions"]["header_regions"] +
                combined_analysis["semantic_regions"]["caption_regions"]
            ),
            is_spread_candidate=False,  # No longer a candidate, it IS a joined spread
            spread_partner_page=None,
            enhancement_applied=False
        )
        
        if debug_mode:
            print(f"   🎨 Created joined spread: {joined_image.size[0]}x{joined_image.size[1]} pixels")
        
        return joined_page

    def _combine_spread_analysis(self, left_page: PDFPage, right_page: PDFPage, 
                            left_width: int, debug_mode: bool) -> Dict[str, Any]:
        """Combine analysis results from two spread pages, adjusting coordinates"""
        
        left_analysis = left_page.analysis_results or {}
        right_analysis = right_page.analysis_results or {}
        
        # Initialize combined structure
        combined_analysis = {
            "surya_layout": [],
            "florence2_rectangles": [],
            "semantic_regions": {
                "text_regions": [],
                "image_regions": [],
                "caption_regions": [],
                "header_regions": [],
                "other_regions": []
            },
            "analysis_summary": {}
        }
        
        # Combine Surya Layout results
        if left_analysis.get("surya_layout"):
            combined_analysis["surya_layout"].extend(left_analysis["surya_layout"])
        
        if right_analysis.get("surya_layout"):
            # Adjust coordinates for right page (shift by left page width)
            for region in right_analysis["surya_layout"]:
                adjusted_region = region.copy()
                bbox = adjusted_region["bbox"]
                adjusted_region["bbox"] = [bbox[0] + left_width, bbox[1], bbox[2] + left_width, bbox[3]]
                combined_analysis["surya_layout"].append(adjusted_region)
        
        # Combine Florence2 results  
        if left_analysis.get("florence2_rectangles"):
            combined_analysis["florence2_rectangles"].extend(left_analysis["florence2_rectangles"])
        
        if right_analysis.get("florence2_rectangles"):
            # Adjust coordinates for right page
            for region in right_analysis["florence2_rectangles"]:
                adjusted_region = region.copy()
                bbox = adjusted_region["bbox"]
                adjusted_region["bbox"] = [bbox[0] + left_width, bbox[1], bbox[2] + left_width, bbox[3]]
                combined_analysis["florence2_rectangles"].append(adjusted_region)
        
        # Combine semantic regions
        for region_type in combined_analysis["semantic_regions"]:
            # Add left page regions
            left_regions = left_analysis.get("semantic_regions", {}).get(region_type, [])
            combined_analysis["semantic_regions"][region_type].extend(left_regions)
            
            # Add right page regions (with adjusted coordinates)
            right_regions = right_analysis.get("semantic_regions", {}).get(region_type, [])
            for region in right_regions:
                adjusted_region = region.copy()
                bbox = adjusted_region["bbox"]
                adjusted_region["bbox"] = [bbox[0] + left_width, bbox[1], bbox[2] + left_width, bbox[3]]
                combined_analysis["semantic_regions"][region_type].append(adjusted_region)
        
        # Create combined summary
        combined_analysis["analysis_summary"] = {
            "total_surya_regions": len(combined_analysis["surya_layout"]),
            "total_florence2_regions": len(combined_analysis["florence2_rectangles"]),
            "text_regions_count": len(combined_analysis["semantic_regions"]["text_regions"]),
            "image_regions_count": len(combined_analysis["semantic_regions"]["image_regions"]),
            "caption_regions_count": len(combined_analysis["semantic_regions"]["caption_regions"]),
            "header_regions_count": len(combined_analysis["semantic_regions"]["header_regions"]),
            "has_semantic_data": len(combined_analysis["surya_layout"]) > 0,
            "has_rectangle_data": len(combined_analysis["florence2_rectangles"]) > 0,
            "analysis_methods": ["surya_layout", "florence2", "ocr"],
            "spread_joined": True,
            "original_pages": [left_page.page_number, right_page.page_number]
        }
        
        if debug_mode:
            print(f"   📊 Combined analysis: {len(combined_analysis['surya_layout'])} total regions")
        
        return combined_analysis


    def _detect_spread_semantic(self, left_page: PDFPage, right_page: PDFPage, debug_mode: bool) -> bool:
        """Detect spread using Surya Layout semantic analysis"""
        
        if not (left_page.analysis_results and right_page.analysis_results):
            return False

        left_semantic = left_page.analysis_results["semantic_regions"]
        right_semantic = right_page.analysis_results["semantic_regions"]

        # Look for large images that might span across pages
        left_images = left_semantic["image_regions"]
        right_images = right_semantic["image_regions"]

        # Check if both pages have large images near the inner edges
        left_has_edge_image = any(
            self._is_near_right_edge(img["bbox"], left_page.image.size) 
            for img in left_images
        )
        
        right_has_edge_image = any(
            self._is_near_left_edge(img["bbox"], right_page.image.size) 
            for img in right_images
        )

        # Also check for minimal text content (characteristic of spreads)
        left_text_density = len(left_semantic["text_regions"]) / (left_page.image.size[0] * left_page.image.size[1])
        right_text_density = len(right_semantic["text_regions"]) / (right_page.image.size[0] * right_page.image.size[1])

        # Spread indicators
        has_edge_images = left_has_edge_image and right_has_edge_image
        low_text_density = (left_text_density + right_text_density) < 0.000001  # Very low text density

        spread_detected = has_edge_images or low_text_density

        if debug_mode and spread_detected:
            print(f"   🎯 Semantic spread indicators: edge_images={has_edge_images}, low_text={low_text_density}")

        return spread_detected

    def _detect_spread_florence2(self, left_page: PDFPage, right_page: PDFPage, debug_mode: bool) -> bool:
        """Detect spread using Florence2 bounding box analysis"""
        
        if not (left_page.analysis_results and right_page.analysis_results):
            return False

        left_florence2 = left_page.analysis_results.get("florence2_rectangles", [])
        right_florence2 = right_page.analysis_results.get("florence2_rectangles", [])

        # Look for large boxes that might indicate spread images
        left_large_boxes = [
            box for box in left_florence2 
            if box["area"] > (left_page.image.size[0] * left_page.image.size[1] * 0.3)
        ]
        
        right_large_boxes = [
            box for box in right_florence2 
            if box["area"] > (right_page.image.size[0] * right_page.image.size[1] * 0.3)
        ]

        # Check if large boxes are positioned at page edges
        left_edge_boxes = [
            box for box in left_large_boxes
            if self._is_near_right_edge(box["bbox"], left_page.image.size)
        ]
        
        right_edge_boxes = [
            box for box in right_large_boxes  
            if self._is_near_left_edge(box["bbox"], right_page.image.size)
        ]

        spread_detected = len(left_edge_boxes) > 0 and len(right_edge_boxes) > 0

        if debug_mode and spread_detected:
            print(f"   📦 Florence2 spread indicators: left_edge_boxes={len(left_edge_boxes)}, right_edge_boxes={len(right_edge_boxes)}")

        return spread_detected

    def _is_near_right_edge(self, bbox: List[int], page_size: Tuple[int, int]) -> bool:
        """Check if bounding box is near the right edge of the page"""
        x1, y1, x2, y2 = bbox
        page_width, page_height = page_size
        
        # Box extends to within 10% of right edge
        return x2 > (page_width * 0.9)

    def _is_near_left_edge(self, bbox: List[int], page_size: Tuple[int, int]) -> bool:
        """Check if bounding box is near the left edge of the page"""
        x1, y1, x2, y2 = bbox
        page_width, page_height = page_size
        
        # Box starts within 10% of left edge
        return x1 < (page_width * 0.1)

    def _process_page_images(self, page: PDFPage, min_image_area: int, 
                            florence2_confidence: float, avoid_full_page_boxes: bool,
                            crop_recommendations: List[Dict], enable_enhancement: bool,
                            enhancement_profile: str, enhancement_strength: float,
                            output_path: str, save_to_disk: bool, debug_mode: bool) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        """Process images with priority: Native > Florence2 > Surya > Full-page fallback"""
        
        page_images = []
        page_enhanced = []
        saved_paths = []

        if debug_mode:
            print(f"   🔍 Processing page {page.page_number} images...")

        # Initialize variables
        final_detections = []
        full_page_candidates = []

        # STEP 0: Check for native embedded images first (highest quality)
        native_images = page.native_images or []
        
        if debug_mode:
            print(f"       📷 Native embedded images: {len(native_images)}")
        
        # Add valid native images first (highest priority)
        for native_img in native_images:
            area = native_img.get("area", 0)
            
            if area >= min_image_area:
                # Convert native image info to detection format
                detection = {
                    "bbox": native_img["bbox"],
                    "confidence": 1.0,  # Native images have perfect confidence
                    "area": area,
                    "detection_source": "native_embedded",
                    "priority": 0,  # Highest priority
                    "native_image": native_img["image"]  # Store the actual PIL image
                }
                final_detections.append(detection)
                
                if debug_mode:
                    print(f"       ✅ Added native image: {native_img['size']}, format: {native_img['format']}")

        # STEP 1: If no native images, proceed with Florence2 detections
        if not final_detections:
            florence2_detections = page.analysis_results.get("florence2_rectangles", [])
            
            if debug_mode:
                print(f"       📦 No native images found, checking Florence2: {len(florence2_detections)}")
            
            # Add all valid Florence2 detections first
            for f2_detection in florence2_detections:
                confidence = f2_detection.get("confidence", 1.0)
                area = f2_detection.get("area", 0)
                bbox = f2_detection.get("bbox", [0, 0, 0, 0])
                
                if debug_mode:
                    print(f"       📦 F2 detection: bbox={bbox}, conf={confidence:.2f}, area={area}")
                
                # Apply Florence2 filtering
                if (confidence >= florence2_confidence and 
                    area >= min_image_area and 
                    len(bbox) >= 4):
                    
                    # Check if it's a full-page box
                    is_full_page = self._is_full_page_box(bbox, page.image.size)
                    
                    if is_full_page:
                        # Store full-page candidates separately
                        full_page_candidates.append({
                            **f2_detection,
                            "detection_source": "florence2",
                            "priority": 3,  # Lower priority - use as fallback
                            "is_full_page": True
                        })
                        
                        if debug_mode:
                            print(f"       📄 Found F2 full-page candidate: {bbox}")
                        
                        # If avoid_full_page_boxes is False, add it to regular detections too
                        if not avoid_full_page_boxes:
                            final_detections.append({
                                **f2_detection,
                                "detection_source": "florence2",
                                "priority": 1
                            })
                            if debug_mode:
                                print(f"       ✅ Added F2 full-page image (avoid_full_page_boxes=False): {bbox}")
                    else:
                        # Regular image detection
                        final_detections.append({
                            **f2_detection,
                            "detection_source": "florence2",
                            "priority": 1  # Highest priority
                        })
                        
                        if debug_mode:
                            print(f"       ✅ Added Florence2 image: {bbox}, conf: {confidence:.2f}")

            # STEP 2: Add Surya detections that don't overlap significantly with Florence2
            surya_detections = page.detected_images or []
            
            if debug_mode:
                print(f"       📊 Surya detections available: {len(surya_detections)}")
            
            for surya_detection in surya_detections:
                area = surya_detection.get("area", 0)
                surya_bbox = surya_detection.get("bbox", [0, 0, 0, 0])
                
                if area < min_image_area or len(surya_bbox) < 4:
                    if debug_mode:
                        print(f"       📊 Skipping small Surya detection: area={area}, bbox={surya_bbox}")
                    continue
                
                # Check if this is a full-page detection
                is_full_page = self._is_full_page_box(surya_bbox, page.image.size)
                
                if is_full_page:
                    # Add to full-page candidates
                    full_page_candidates.append({
                        **surya_detection,
                        "detection_source": "surya_layout",
                        "confidence": surya_detection.get("confidence", 0.8),
                        "priority": 4,  # Lowest priority
                        "is_full_page": True
                    })
                    
                    if debug_mode:
                        print(f"       📄 Found Surya full-page candidate: {surya_bbox}")
                    
                    # If avoid_full_page_boxes is False, continue with overlap checking
                    if not avoid_full_page_boxes:
                        # Continue to overlap checking below
                        pass
                    else:
                        # Skip this detection for regular processing
                        continue
                
                # Check if this Surya detection overlaps significantly with any Florence2 detection
                overlaps_with_florence2 = False
                max_overlap = 0.0
                
                for f2_detection in final_detections:
                    if f2_detection["detection_source"] == "florence2":
                        overlap_ratio = self._calculate_overlap_ratio(surya_bbox, f2_detection["bbox"])
                        max_overlap = max(max_overlap, overlap_ratio)
                        
                        # If overlap is significant, Florence2 wins (it has better bounding boxes)
                        if overlap_ratio > 0.3:  # 30% overlap threshold
                            overlaps_with_florence2 = True
                            if debug_mode:
                                print(f"       🔄 Surya detection overlaps with Florence2 ({overlap_ratio:.2f}): {surya_bbox}")
                            break
                
                # If no significant overlap with Florence2, add the Surya detection
                if not overlaps_with_florence2:
                    final_detections.append({
                        **surya_detection,
                        "detection_source": "surya_layout",
                        "confidence": surya_detection.get("confidence", 0.8),
                        "priority": 2  # Lower priority than Florence2
                    })
                    
                    if debug_mode:
                        print(f"       ✅ Added Surya image (F2 missed): {surya_bbox}, area: {area}")
                elif debug_mode:
                    print(f"       🚫 Skipped Surya detection (overlap={max_overlap:.2f} with Florence2): {surya_bbox}")

        # STEP 3: Fallback logic - If no regular images found, use full-page candidates
        if not final_detections and full_page_candidates:
            if debug_mode:
                print(f"       📄 No regular images found, using full-page fallback")
                print(f"       📄 Available full-page candidates: {len(full_page_candidates)}")
            
            # Sort full-page candidates by priority and confidence
            full_page_candidates.sort(key=lambda x: (x["priority"], -x.get("confidence", 0)))
            
            # Use the best full-page candidate
            best_full_page = full_page_candidates[0]
            
            # Check if this is likely a content page (not just text on white background)
            if self._is_content_page(page.image, best_full_page["bbox"], debug_mode):
                final_detections.append(best_full_page)
                if debug_mode:
                    source = best_full_page["detection_source"]
                    print(f"       ✅ Using full-page fallback from {source}: {best_full_page['bbox']}")
            else:
                if debug_mode:
                    print(f"       ⚠️ Full-page candidate appears to be mostly text/blank - skipping")

        if debug_mode:
            print(f"       🎯 Final detections after smart merging: {len(final_detections)}")
            florence2_count = sum(1 for d in final_detections if d["detection_source"] == "florence2")
            surya_count = sum(1 for d in final_detections if d["detection_source"] == "surya_layout")
            native_count = sum(1 for d in final_detections if d["detection_source"] == "native_embedded")
            full_page_count = sum(1 for d in final_detections if d.get("is_full_page", False))
            print(f"           📷 Native: {native_count}, 📦 Florence2: {florence2_count}, 📊 Surya: {surya_count}, 📄 Full-page: {full_page_count}")

        # STEP 4: Apply smart cropping recommendations
        if crop_recommendations:
            final_detections = self._apply_smart_cropping(final_detections, crop_recommendations, debug_mode)

        # STEP 5: Final size filtering
        min_size_px = int(min_image_area ** 0.5)
        filtered_detections = []
        
        for detection in final_detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                if width >= min_size_px and height >= min_size_px:
                    filtered_detections.append(detection)
                elif debug_mode:
                    source = detection.get("detection_source", "unknown")
                    print(f"       📏 Filtered out small {source} image: {width}x{height}px (min: {min_size_px}px)")
            elif debug_mode:
                print(f"       ⚠️ Invalid bbox format: {bbox}")

        if debug_mode:
            print(f"       ✅ Final detections to process: {len(filtered_detections)}")

        # STEP 6: Extract and process images
        for i, detection in enumerate(filtered_detections):
            try:
                source = detection.get("detection_source", "unknown")
                confidence = detection.get("confidence", 1.0)
                
                # Handle native images differently - they're already extracted
                if source == "native_embedded":
                    cropped = detection["native_image"]  # Use the native PIL image directly
                    
                    if debug_mode:
                        print(f"       📷 Processing native image {i+1}: {cropped.size} (direct extraction)")
                
                else:
                    # Regular cropping from page image
                    bbox = detection.get("bbox", [0, 0, 0, 0])
                    if len(bbox) < 4:
                        if debug_mode:
                            print(f"       ❌ Invalid bbox for detection {i+1}: {bbox}")
                        continue
                    
                    # Ensure bbox values are within image bounds
                    x1, y1, x2, y2 = bbox
                    img_width, img_height = page.image.size
                    
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(x1, min(x2, img_width))
                    y2 = max(y1, min(y2, img_height))
                    
                    if x2 <= x1 or y2 <= y1:
                        if debug_mode:
                            print(f"       ❌ Invalid bbox dimensions: {[x1, y1, x2, y2]}")
                        continue
                    
                    cropped = page.image.crop([x1, y1, x2, y2])
                    
                    if debug_mode:
                        print(f"       🖼️  Processing image {i+1}: {cropped.size} from {source} (conf: {confidence:.2f})")
                
                # Convert to tensor
                image_tensor = PIL_to_tensor(cropped)
                page_images.append(image_tensor)

                # Apply SMART enhancement based on image source and quality
                enhanced_image = cropped
                if enable_enhancement and self.image_enhancer:
                    try:
                        # Determine enhancement settings based on source
                        if source == "native_embedded":
                            # Native images: Use quality-based enhancement
                            quality_metrics = detection.get("quality_metrics", {})
                            enhancement_rec = quality_metrics.get("enhancement_recommendation", "minimal")
                            strength_override = quality_metrics.get("enhancement_strength_override", enhancement_strength)
                            
                            if enhancement_rec == "none":
                                # Skip enhancement for excellent quality native images
                                enhanced_image = cropped
                                if debug_mode:
                                    print(f"           ✨ Native image excellent quality - no enhancement needed")
                            else:
                                # Light enhancement for native images
                                profile_override = "Minimal" if enhancement_rec == "minimal" else enhancement_profile
                                enhanced_image = self.image_enhancer.enhance_image(
                                    cropped, 
                                    profile=profile_override,
                                    strength=strength_override
                                )
                                if debug_mode:
                                    print(f"           ✨ Native image enhanced: {enhancement_rec} ({strength_override:.1f} strength)")
                        
                        else:
                            # Flattened/rendered images: Use full enhancement
                            enhanced_image = self.image_enhancer.enhance_image(
                                cropped, 
                                profile=enhancement_profile,
                                strength=enhancement_strength
                            )
                            if debug_mode:
                                print(f"           ✨ Rendered image enhanced: {cropped.size} -> {enhanced_image.size}")
                        
                        enhanced_tensor = PIL_to_tensor(enhanced_image)
                        page_enhanced.append(enhanced_tensor)
                        
                    except Exception as e:
                        if debug_mode:
                            print(f"           ⚠️ Enhancement failed: {e}")
                        enhanced_image = cropped
                        page_enhanced.append(image_tensor)
                else:
                    page_enhanced.append(image_tensor)

                # Save to disk if requested
                if save_to_disk:
                    try:
                        # Ensure directories exist
                        images_dir = os.path.join(output_path, "images")
                        enhanced_dir = os.path.join(output_path, "enhanced")
                        os.makedirs(images_dir, exist_ok=True)
                        os.makedirs(enhanced_dir, exist_ok=True)
                        
                        # Handle page numbers that might be strings (like "12-13" for spreads)
                        page_str = str(page.page_number).replace("-", "_")
                        
                        # Include detection source in filename for debugging
                        source_prefix = {
                            "native_embedded": "nat",
                            "florence2": "f2", 
                            "surya_layout": "sy"
                        }.get(source, "unk")
                        
                        # Include quality info for native images
                        quality_suffix = ""
                        if source == "native_embedded":
                            quality_metrics = detection.get("quality_metrics", {})
                            enhancement_rec = quality_metrics.get("enhancement_recommendation", "unk")
                            quality_suffix = f"_{enhancement_rec}"
                        
                        # Save original
                        original_filename = f"page_{page_str}_img_{i+1:02d}_{source_prefix}{quality_suffix}_original.png"
                        original_path = os.path.join(images_dir, original_filename)
                        cropped.save(original_path, "PNG")
                        
                        # Save enhanced
                        enhanced_filename = f"page_{page_str}_img_{i+1:02d}_{source_prefix}{quality_suffix}_enhanced.png"
                        enhanced_path = os.path.join(enhanced_dir, enhanced_filename)
                        enhanced_image.save(enhanced_path, "PNG")
                        
                        saved_paths.extend([original_path, enhanced_path])
                        
                        if debug_mode:
                            print(f"           💾 Saved: {original_filename} and {enhanced_filename}")
                            
                    except Exception as e:
                        if debug_mode:
                            print(f"           ❌ Failed to save image: {e}")
                            print(f"               Output path: {output_path}")
                            print(f"               Images dir: {os.path.join(output_path, 'images')}")

            except Exception as e:
                if debug_mode:
                    print(f"       ❌ Failed to process detection {detection}: {e}")

        if debug_mode:
            print(f"       🎯 Page {page.page_number} complete: {len(page_images)} images extracted, {len(saved_paths)} files saved")

        return page_images, page_enhanced, saved_paths

    def _is_content_page(self, image: Image.Image, bbox: List[int], debug_mode: bool) -> bool:
        """Check if a full-page image contains actual visual content (not just text on white)"""
        try:
            # FIX: Add bounds checking
            if len(bbox) < 4:
                if debug_mode:
                    print(f"       ⚠️ Invalid bbox for content analysis: {bbox}")
                return True  # Default to including if bbox is invalid
                
            # Crop the full-page area
            x1, y1, x2, y2 = bbox
            img_width, img_height = image.size
            
            # FIX: Ensure bbox is within image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(x1, min(x2, img_width))
            y2 = max(y1, min(y2, img_height))
            
            if x2 <= x1 or y2 <= y1:
                if debug_mode:
                    print(f"       ⚠️ Invalid bbox dimensions after bounds checking: {[x1, y1, x2, y2]}")
                return True
                
            cropped = image.crop([x1, y1, x2, y2])
            
            # Convert to numpy for analysis
            img_array = np.array(cropped)
            
            # Calculate color variance - pure text pages have low color variance
            if len(img_array.shape) == 3:  # RGB
                gray = np.mean(img_array, axis=2)
            else:  # Already grayscale
                gray = img_array
            
            # Calculate statistics
            color_variance = np.var(gray)
            mean_brightness = np.mean(gray)
            
            # Check for image content indicators
            has_visual_content = (
                color_variance > 500 or  # Sufficient color variation
                mean_brightness < 200 or  # Not predominantly white
                mean_brightness > 50      # Not predominantly black
            )
            
            if debug_mode:
                print(f"       🔍 Content analysis: variance={color_variance:.0f}, brightness={mean_brightness:.0f}, has_content={has_visual_content}")
            
            return has_visual_content
            
        except Exception as e:
            if debug_mode:
                print(f"       ⚠️ Content analysis failed: {e}")
            return True  # Default to including the image if analysis fails

    def _is_full_page_box(self, bbox: List[int], page_size: Tuple[int, int]) -> bool:
        """Check if bounding box covers most of the page"""
        x1, y1, x2, y2 = bbox
        page_width, page_height = page_size
        
        box_area = (x2 - x1) * (y2 - y1)
        page_area = page_width * page_height
        
        # Consider it full-page if it covers >85% of the page
        coverage = box_area / page_area
        return coverage > 0.85

    def _remove_duplicate_detections(self, detections: List[Dict], debug_mode: bool) -> List[Dict]:
        """Remove overlapping detections, keeping the best one"""
        
        if len(detections) <= 1:
            return detections

        unique_detections = []
        
        for detection in detections:
            is_duplicate = False
            detection_confidence = detection.get("confidence", 1.0)  # Safe confidence access
            
            for existing in unique_detections:
                existing_confidence = existing.get("confidence", 1.0)  # Safe confidence access
                
                bbox1 = detection.get("bbox", [0, 0, 0, 0])
                bbox2 = existing.get("bbox", [0, 0, 0, 0])
                
                if len(bbox1) >= 4 and len(bbox2) >= 4:
                    overlap_ratio = self._calculate_overlap_ratio(bbox1, bbox2)
                    
                    if overlap_ratio > 0.5:  # 50% overlap threshold
                        is_duplicate = True
                        
                        # Keep the detection with higher confidence
                        if detection_confidence > existing_confidence:
                            unique_detections.remove(existing)
                            unique_detections.append(detection)
                            if debug_mode:
                                print(f"   🔄 Replaced duplicate detection (conf: {existing_confidence:.2f} -> {detection_confidence:.2f})")
                        
                        break
            
            if not is_duplicate:
                unique_detections.append(detection)

        if debug_mode:
            print(f"   🎯 Removed {len(detections) - len(unique_detections)} duplicate detections")

        return unique_detections

    def _calculate_overlap_ratio(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _apply_smart_cropping(self, detections: List[Dict], crop_recommendations: List[Dict], 
                            debug_mode: bool) -> List[Dict]:
        """Apply smart cropping recommendations to avoid captions"""
        
        smart_detections = []
        
        for detection in detections:
            # Check if this detection has cropping recommendations
            matching_recommendation = None
            
            for recommendation in crop_recommendations:
                overlap = self._calculate_overlap_ratio(detection["bbox"], recommendation["bbox"])
                if overlap > 0.7:  # High overlap
                    matching_recommendation = recommendation
                    break
            
            if matching_recommendation and matching_recommendation.get("avoid_regions"):
                # Adjust bounding box to avoid captions
                adjusted_bbox = self._adjust_bbox_avoid_captions(
                    detection["bbox"], 
                    matching_recommendation["avoid_regions"]
                )
                
                if adjusted_bbox != detection["bbox"]:
                    detection = {**detection, "bbox": adjusted_bbox}
                    if debug_mode:
                        print(f"   📏 Adjusted bbox to avoid captions: {detection['bbox']}")
            
            smart_detections.append(detection)

        return smart_detections

    def _adjust_bbox_avoid_captions(self, bbox: List[int], avoid_regions: List[Dict]) -> List[int]:
        """Adjust bounding box to avoid caption regions"""
        
        x1, y1, x2, y2 = bbox
        
        for avoid_region in avoid_regions:
            avoid_bbox = avoid_region["bbox"]
            ax1, ay1, ax2, ay2 = avoid_bbox
            
            # Check if caption overlaps with image
            overlap = self._calculate_overlap_ratio(bbox, avoid_bbox)
            
            if overlap > 0.1:  # 10% overlap
                # Try to adjust image bbox to exclude caption
                
                # If caption is below image, adjust bottom edge
                if ay1 > y1 and ay1 < y2:
                    y2 = min(y2, ay1 - 5)  # 5 pixel buffer
                
                # If caption is above image, adjust top edge  
                elif ay2 < y2 and ay2 > y1:
                    y1 = max(y1, ay2 + 5)  # 5 pixel buffer
                
                # If caption is to the right, adjust right edge
                elif ax1 > x1 and ax1 < x2:
                    x2 = min(x2, ax1 - 5)
                
                # If caption is to the left, adjust left edge
                elif ax2 < x2 and ax2 > x1:
                    x1 = max(x1, ax2 + 5)

        return [x1, y1, x2, y2]

    def _find_image_bbox_on_page(self, page: fitz.Page, xref: int, image_size: Tuple[int, int], 
                                debug_mode: bool) -> List[int]:
        """Find the actual placement bbox of an image on the page"""
        try:
            # Method 1: Search through page resources for image placement
            page_dict = page.get_text("dict")
            
            # Look for image blocks that might correspond to our image
            for block in page_dict.get("blocks", []):
                if block.get("type") == 1:  # Image block type
                    # Found an image block, check if it matches our dimensions
                    block_bbox = block.get("bbox", [0, 0, 0, 0])
                    if len(block_bbox) == 4:
                        # Estimate if this could be our image based on aspect ratio
                        block_width = block_bbox[2] - block_bbox[0]
                        block_height = block_bbox[3] - block_bbox[1]
                        
                        if block_width > 0 and block_height > 0:
                            block_aspect = block_width / block_height
                            image_aspect = image_size[0] / image_size[1] if image_size[1] > 0 else 1.0
                            
                            # If aspect ratios are similar (within 20%), this might be our image
                            if abs(block_aspect - image_aspect) / image_aspect < 0.2:
                                if debug_mode:
                                    print(f"           🎯 Found matching image block by aspect ratio")
                                return list(block_bbox)
            
            # Method 2: Try to get image rect from page annotations or objects
            page_rect = page.rect
            
            # Method 3: Check if there are any drawing operations that might place our image
            drawings = page.get_drawings()
            for drawing in drawings:
                if 'rect' in drawing and drawing.get('type') == 'image':
                    drawing_rect = drawing['rect']
                    if drawing_rect:
                        return [drawing_rect.x0, drawing_rect.y0, drawing_rect.x1, drawing_rect.y1]
            
            # Method 4: Fallback - estimate placement based on page size and image size
            # Assume image is placed at top-left corner with original dimensions (scaled to page)
            page_width = page_rect.width
            page_height = page_rect.height
            
            # Scale image to fit within page while maintaining aspect ratio
            scale_x = page_width / image_size[0] if image_size[0] > 0 else 1.0
            scale_y = page_height / image_size[1] if image_size[1] > 0 else 1.0
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            estimated_width = image_size[0] * scale
            estimated_height = image_size[1] * scale
            
            # Center the image on the page
            x_offset = (page_width - estimated_width) / 2
            y_offset = (page_height - estimated_height) / 2
            
            estimated_bbox = [
                int(x_offset),
                int(y_offset),
                int(x_offset + estimated_width),
                int(y_offset + estimated_height)
            ]
            
            if debug_mode:
                print(f"           📐 Using estimated bbox (no direct placement found)")
            
            return estimated_bbox
            
        except Exception as e:
            if debug_mode:
                print(f"           ⚠️ Error finding image bbox: {e}")
            
            # Ultimate fallback - use full page
            page_rect = page.rect
            return [0, 0, int(page_rect.width), int(page_rect.height)]
            

    def _format_extracted_text(self, extracted_texts: List[Dict], page_number: int) -> str:
        """Format extracted text with semantic labels"""
        
        if not extracted_texts:
            return f"Page {page_number}: No text extracted"

        lines = [f"=== Page {page_number} ==="]
        
        for text_region in extracted_texts:
            semantic_label = text_region.get("semantic_label", "text")
            text_content = text_region.get("text", "").strip()
            
            if text_content:
                # Add semantic label prefix
                label_mapping = {
                    'SectionHeader': 'HEADER',
                    'PageHeader': 'PAGE_HEADER', 
                    'PageFooter': 'PAGE_FOOTER',
                    'Caption': 'CAPTION',
                    'Title': 'TITLE',
                    'Text': 'TEXT'
                }
                
                display_label = label_mapping.get(semantic_label, semantic_label.upper())
                lines.append(f"[{display_label}] {text_content}")

        return "\n".join(lines)

    def _save_text_to_files(self, all_extracted_text: List[str], pages: List[PDFPage], 
                        output_path: str, pdf_path: str, debug_mode: bool) -> Dict[str, str]:
        """Save extracted text in multiple formats"""
        saved_files = {}
        
        try:
            pdf_name = Path(pdf_path).stem
            text_dir = os.path.join(output_path, "text")
            os.makedirs(text_dir, exist_ok=True)
            
            # 1. Combined text file (existing behavior)
            combined_text = "\n\n".join(all_extracted_text) if all_extracted_text else ""
            if combined_text:
                combined_filename = f"{pdf_name}_all_text.txt"
                combined_path = os.path.join(text_dir, combined_filename)
                
                with open(combined_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"EXTRACTED TEXT FROM: {pdf_path}\n")
                    f.write(f"EXTRACTION DATE: {datetime.now().isoformat()}\n")
                    f.write(f"TOTAL PAGES: {len(pages)}\n")
                    f.write("="*80 + "\n\n")
                    f.write(combined_text)
                
                saved_files["combined_text"] = combined_path
                
                if debug_mode:
                    print(f"   📄 Saved combined text: {combined_filename}")
            
            # 2. Individual page files
            page_files = []
            # FIX: Ensure list lengths match
            for i, page in enumerate(pages):
                if i < len(all_extracted_text):
                    page_text = all_extracted_text[i]
                    if page_text and page_text.strip():
                        page_filename = f"{pdf_name}_page_{page.page_number}_text.txt"
                        page_path = os.path.join(text_dir, page_filename)
                        
                        with open(page_path, 'w', encoding='utf-8') as f:
                            f.write(f"Page {page.page_number} Text\n")
                            f.write("="*40 + "\n\n")
                            f.write(page_text)
                        
                        page_files.append(page_path)
                        
                        if debug_mode:
                            print(f"   📝 Saved page text: {page_filename}")
            
            saved_files["individual_pages"] = page_files
            
            # 3. Text summary/index
            summary_filename = f"{pdf_name}_text_summary.json"
            summary_path = os.path.join(text_dir, summary_filename)
            
            text_summary = {
                "pdf_source": pdf_path,
                "extraction_date": datetime.now().isoformat(),
                "total_pages": len(pages),
                "pages_with_text": len([t for t in all_extracted_text if t and t.strip()]),
                "total_characters": len(combined_text),
                "page_breakdown": []
            }
            
            # FIX: Ensure safe iteration
            for i, page in enumerate(pages):
                page_text = all_extracted_text[i] if i < len(all_extracted_text) else ""
                page_info = {
                    "page_number": page.page_number,
                    "has_text": bool(page_text and page_text.strip()),
                    "character_count": len(page_text) if page_text else 0,
                    "text_source": "original_digital" if page.original_text else "ocr_extracted",
                    "spread_page": page.is_spread_candidate
                }
                text_summary["page_breakdown"].append(page_info)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(text_summary, f, indent=2, ensure_ascii=False)
            
            saved_files["text_summary"] = summary_path
            
            if debug_mode:
                print(f"   📊 Saved text summary: {summary_filename}")
            
            return saved_files
            
        except Exception as e:
            if debug_mode:
                print(f"   ❌ Failed to save text files: {e}")
            return {}

    def _save_analysis_files(self, pages: List[PDFPage], extraction_stats: Dict, 
                            output_path: str, pdf_path: str, debug_mode: bool) -> List[str]:
        """Save comprehensive analysis data"""
        saved_files = []
        
        try:
            analysis_dir = os.path.join(output_path, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            pdf_name = Path(pdf_path).stem
            
            # 1. Extraction statistics (existing)
            stats_filename = f"{pdf_name}_extraction_stats.json"
            stats_path = os.path.join(analysis_dir, stats_filename)
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_stats, f, indent=2, ensure_ascii=False)
            
            saved_files.append(stats_path)
            
            # 2. Page-by-page analysis breakdown
            page_analysis_filename = f"{pdf_name}_page_analysis.json"
            page_analysis_path = os.path.join(analysis_dir, page_analysis_filename)
            
            page_breakdown = []
            for page in pages:
                analysis = page.analysis_results or {}
                semantic_regions = analysis.get("semantic_regions", {})
                
                page_data = {
                    "page_number": page.page_number,
                    "original_size": page.original_size,
                    "text_layers_removed": page.text_layers_removed,
                    "is_spread": page.is_spread_candidate,
                    "spread_partner": page.spread_partner_page,
                    "native_images_found": len(page.native_images or []),
                    "surya_detections": {
                        "total_regions": len(analysis.get("surya_layout", [])),
                        "text_regions": len(semantic_regions.get("text_regions", [])),
                        "image_regions": len(semantic_regions.get("image_regions", [])),
                        "caption_regions": len(semantic_regions.get("caption_regions", [])),
                        "header_regions": len(semantic_regions.get("header_regions", [])),
                        "other_regions": len(semantic_regions.get("other_regions", []))
                    },
                    "florence2_detections": {
                        "total_rectangles": len(analysis.get("florence2_rectangles", [])),
                        "high_confidence": len([r for r in analysis.get("florence2_rectangles", []) if r.get("confidence", 0) > 0.8])
                    },
                    "text_extraction": {
                        "has_original_text": bool(page.original_text),
                        "original_text_length": len(page.original_text or ""),
                        "ocr_regions_available": len(page.detected_text_regions or [])
                    }
                }
                page_breakdown.append(page_data)
            
            with open(page_analysis_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "pdf_source": pdf_path,
                    "analysis_date": datetime.now().isoformat(),
                    "total_pages": len(pages),
                    "page_analysis": page_breakdown
                }, f, indent=2, ensure_ascii=False)
            
            saved_files.append(page_analysis_path)
            
            # 3. Detection data (bounding boxes, etc.)
            detections_filename = f"{pdf_name}_detection_data.json"
            detections_path = os.path.join(analysis_dir, detections_filename)
            
            detection_data = {
                "pdf_source": pdf_path,
                "detection_date": datetime.now().isoformat(),
                "pages": []
            }
            
            for page in pages:
                analysis = page.analysis_results or {}
                
                page_detections = {
                    "page_number": page.page_number,
                    "page_size": page.original_size,
                    "surya_layout": analysis.get("surya_layout", []),
                    "florence2_rectangles": analysis.get("florence2_rectangles", []),
                    "native_images": [
                        {
                            "bbox": img["bbox"],
                            "size": img["size"],
                            "format": img["format"],
                            "area": img["area"]
                        } for img in (page.native_images or [])
                    ]
                }
                detection_data["pages"].append(page_detections)
            
            with open(detections_path, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, indent=2, ensure_ascii=False)
            
            saved_files.append(detections_path)
            
            if debug_mode:
                print(f"   📊 Saved analysis files:")
                for file_path in saved_files:
                    print(f"       📁 {os.path.basename(file_path)}")
            
            return saved_files
            
        except Exception as e:
            if debug_mode:
                print(f"   ❌ Failed to save analysis files: {e}")
            return []
            

    def _monitor_gpu_memory(self, debug_mode: bool) -> Dict[str, Any]:
        """Monitor GPU memory usage during processing"""
        if not self.image_enhancer or not hasattr(self.image_enhancer, 'get_memory_usage'):
            return {}
        
        try:
            memory_info = self.image_enhancer.get_memory_usage()
            
            if debug_mode and memory_info.get('gpu_available'):
                gpu_mem = memory_info.get('gpu_memory', {})
                if isinstance(gpu_mem, dict):
                    used_percent = gpu_mem.get('percent', 0)
                    print(f"   💾 GPU Memory Usage: {used_percent:.1f}%")
                    
                    # Warning if memory usage is high
                    if used_percent > 80:
                        print(f"   ⚠️ High GPU memory usage detected!")
            
            return memory_info
            
        except Exception as e:
            if debug_mode:
                print(f"   ⚠️ GPU memory monitoring failed: {e}")
            return {}

    def _create_page_summary(self, page: PDFPage, num_images_extracted: int) -> Dict[str, Any]:
        """Create summary of page analysis"""
        
        analysis = page.analysis_results
        semantic_regions = analysis["semantic_regions"] if analysis else {}
        
        return {
            "page_number": page.page_number,
            "original_size": page.original_size,
            "text_layers_removed": page.text_layers_removed,
            "is_spread_candidate": page.is_spread_candidate,
            "spread_partner_page": page.spread_partner_page,
            "enhancement_applied": page.enhancement_applied,
            "images_extracted": num_images_extracted,
            "semantic_analysis": {
                "text_regions": len(semantic_regions.get("text_regions", [])),
                "image_regions": len(semantic_regions.get("image_regions", [])),
                "caption_regions": len(semantic_regions.get("caption_regions", [])),
                "header_regions": len(semantic_regions.get("header_regions", [])),
                "other_regions": len(semantic_regions.get("other_regions", []))
            },
            "florence2_detections": len(analysis.get("florence2_rectangles", [])) if analysis else 0,
            "analysis_summary": analysis.get("analysis_summary", {}) if analysis else {}
        }

    def _create_extraction_stats(self, pages: List[PDFPage], num_images: int, 
                                num_enhanced: int, text_length: int, output_path: str,
                                saved_image_paths: List[str], text_files: Dict[str, Any], debug_mode: bool) -> Dict[str, Any]:
        """Create comprehensive extraction statistics"""
        
        total_spreads = sum(1 for page in pages if page.is_spread_candidate) // 2
        pages_with_images = sum(1 for page in pages if page.detected_images)
        pages_with_text = sum(1 for page in pages if page.detected_text_regions)
        
        # Aggregate semantic analysis
        total_semantic_regions = {
            "text_regions": 0,
            "image_regions": 0, 
            "caption_regions": 0,
            "header_regions": 0,
            "other_regions": 0
        }
        
        total_florence2_detections = 0
        
        for page in pages:
            if page.analysis_results:
                semantic = page.analysis_results["semantic_regions"]
                for region_type in total_semantic_regions:
                    total_semantic_regions[region_type] += len(semantic.get(region_type, []))
                
                total_florence2_detections += len(page.analysis_results.get("florence2_rectangles", []))

        stats = {
            "extraction_timestamp": datetime.now().isoformat(),
            "output_directory": output_path,
            "total_pages_processed": len(pages),
            "total_images_extracted": num_images,
            "total_enhanced_images": num_enhanced,
            "total_text_characters": text_length,
            "spreads_detected": total_spreads,
            "pages_with_images": pages_with_images,
            "pages_with_text": pages_with_text,
            "semantic_analysis_totals": total_semantic_regions,
            "florence2_detections_total": total_florence2_detections,
            "enhancement_success_rate": (num_enhanced / num_images * 100) if num_images > 0 else 0,
            "analysis_engine_available": ANALYSIS_ENGINE_AVAILABLE,
            "modern_enhancer_available": MODERN_ENHANCER_AVAILABLE,
            "file_outputs": {
                "saved_images_count": len(saved_image_paths),
                "saved_image_paths": saved_image_paths,
                "text_files": text_files,  # Updated to use the new text_files dict
                "output_structure": {
                    "images_folder": os.path.join(output_path, "images"),
                    "enhanced_folder": os.path.join(output_path, "enhanced"),
                    "text_folder": os.path.join(output_path, "text"),  # Added text folder
                    "analysis_folder": os.path.join(output_path, "analysis")
                }
            },
            "extraction_quality_indicators": {
                "high_resolution_pages": sum(1 for p in pages if p.original_size[0] > 2000),
                "text_layer_removal_applied": sum(1 for p in pages if p.text_layers_removed),
                "semantic_understanding_coverage": (
                    sum(len(p.analysis_results.get("surya_layout", [])) for p in pages if p.analysis_results) / len(pages)
                ) if pages else 0
            }
        }

        # Save stats to JSON file
        try:
            stats_filename = "extraction_stats.json"
            stats_file_path = os.path.join(output_path, "analysis", stats_filename)
            os.makedirs(os.path.join(output_path, "analysis"), exist_ok=True)  # Ensure directory exists
            
            with open(stats_file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            if debug_mode:
                print(f"   💾 Saved extraction stats to: {stats_filename}")
                
        except Exception as e:
            if debug_mode:
                print(f"   ❌ Failed to save stats file: {e}")

        return stats


# Node registration
NODE_CLASS_MAPPINGS = {
    "EnhancedPDFExtractor_v08": EnhancedPDFExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedPDFExtractor_v08": "Enhanced PDF Extractor v08 (Analysis Engine)",
}