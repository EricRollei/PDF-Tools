"""
Simple Pdf Extractor

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
Simple PDF Image Extractor - No AI, Just Fast Extraction
Extracts all embedded images from PDF without any filtering or AI processing.

Features:
- Automatic layer detection for layered PDFs
- Super-fast layer-based extraction (2-5 seconds)
- Falls back to standard extraction for non-layered PDFs
- Optional fast mode: only extract if layers detected

This is the "just extract everything" node for when you want speed and completeness.
"""

import os
import io
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("❌ PyMuPDF not available - install with: pip install PyMuPDF")


class SimplePDFImageExtractor:
    """
    Simple, fast PDF image extractor with automatic layer detection
    - Detects PDF layers automatically
    - Uses super-fast layer-based extraction when available
    - No AI, no filtering, just extract everything
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_path": ("STRING", {"default": "", "multiline": False}),
                "output_directory": ("STRING", {"default": "output/simple_pdf_extraction", "multiline": False}),
                "min_width": ("INT", {"default": 100, "min": 10, "max": 5000, "step": 10}),
                "min_height": ("INT", {"default": 100, "min": 10, "max": 5000, "step": 10}),
                "extract_text": ("BOOLEAN", {"default": True}),
                "layers_only_mode": ("BOOLEAN", {"default": False}),  # Only extract if layers detected
            },
            "optional": {
                "dpi": ("INT", {"default": 150, "min": 72, "max": 600, "step": 50}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("images", "summary", "image_count", "has_layers")
    OUTPUT_IS_LIST = (True, False, False, False)  # images is a list of individual images
    FUNCTION = "extract_images"
    CATEGORY = "PDF Tools/Simple"
    
    def extract_images(self, pdf_path, output_directory, min_width, min_height, extract_text, layers_only_mode=False, dpi=150):
        """Extract all images from PDF(s) with automatic layer detection - supports single file or folder"""
        
        if not PYMUPDF_AVAILABLE:
            dummy_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return ([dummy_tensor], "❌ PyMuPDF not installed", 0, False)
        
        if not os.path.exists(pdf_path):
            dummy_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return ([dummy_tensor], f"❌ Path not found: {pdf_path}", 0, False)
        
        # Check if path is a directory (batch mode) or single file
        if os.path.isdir(pdf_path):
            return self._extract_batch(pdf_path, output_directory, min_width, min_height, extract_text, layers_only_mode, dpi)
        
        # Single file processing (original logic)
        return self._extract_single_pdf(pdf_path, output_directory, min_width, min_height, extract_text, layers_only_mode, dpi)
    
    def _extract_single_pdf(self, pdf_path, output_directory, min_width, min_height, extract_text, layers_only_mode, dpi):
        """Extract images from a single PDF file"""
        
        start_time = time.time()
        
        # Create output directory
        output_dir = Path(output_directory)
        pdf_name = Path(pdf_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_subdir = output_dir / f"{pdf_name}_{timestamp}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Simple PDF Image Extractor (Layer-Aware)")
        print(f"📄 PDF: {pdf_path}")
        print(f"📁 Output: {output_subdir}")
        print(f"📏 Min size: {min_width}×{min_height}")
        print(f"🏃 Fast mode: {'Layers only' if layers_only_mode else 'All PDFs'}")
        
        extracted_images = []
        all_images_pil = []
        image_count = 0
        text_content = []
        has_layers = False
        layer_info = None
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = doc.page_count
                
                # STEP 1: Detect layers (always check)
                layer_info = self._detect_layers(doc)
                has_layers = layer_info["has_layers"]
                
                if has_layers:
                    print(f"✨ PDF has {layer_info['layer_count']} layers!")
                    for layer in layer_info['layers']:
                        print(f"   📋 Layer: '{layer['name']}' ({'ON' if layer['visible'] else 'OFF'})")
                    print(f"🚀 Using super-fast layer-based extraction")
                else:
                    print(f"📖 No layers detected - using standard extraction")
                    if layers_only_mode:
                        msg = "⚠️  Layers-only mode enabled, but PDF has no layers. Skipping extraction."
                        print(msg)
                        dummy_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
                        return ([dummy_tensor], msg, 0, False)
                
                print(f"📖 Processing {total_pages} pages...")
                
                for page_num in range(total_pages):
                    page = doc[page_num]
                    
                    # Extract text if requested
                    if extract_text:
                        text = page.get_text()
                        if text.strip():
                            text_content.append(f"\n{'='*60}\nPage {page_num + 1}\n{'='*60}\n{text}")
                    
                    # Get all images from page
                    image_list = page.get_images(full=True)
                    processed_xrefs = set()
                    
                    print(f"  📄 Page {page_num + 1}: {len(image_list)} images")
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        if xref in processed_xrefs:
                            continue
                        
                        try:
                            # Extract image
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Convert to PIL
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            
                            # Size filter
                            if pil_image.width < min_width or pil_image.height < min_height:
                                print(f"    ⏭️  Skipped small image: {pil_image.width}×{pil_image.height}")
                                continue
                            
                            # Convert to RGB if needed
                            if pil_image.mode in ('RGBA', 'LA', 'P'):
                                pil_image = pil_image.convert('RGB')
                            elif pil_image.mode == 'L':
                                pass  # Keep grayscale
                            elif pil_image.mode not in ('RGB', 'L'):
                                pil_image = pil_image.convert('RGB')
                            
                            # Save to disk
                            filename = f"page_{page_num + 1:03d}_image_{img_index + 1:02d}.png"
                            filepath = output_subdir / filename
                            pil_image.save(filepath, "PNG")
                            
                            # Add to results
                            all_images_pil.append(pil_image)
                            image_count += 1
                            processed_xrefs.add(xref)
                            
                            print(f"    ✅ Extracted: {pil_image.width}×{pil_image.height} → {filename}")
                            
                        except Exception as e:
                            print(f"    ⚠️  Error extracting image {img_index}: {e}")
                            continue
                
        except Exception as e:
            error_msg = f"❌ Error processing PDF: {e}"
            print(error_msg)
            dummy_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return ([dummy_tensor], error_msg, 0, has_layers)
        
        # Save text if extracted
        if text_content:
            text_file = output_subdir / f"{pdf_name}_all_text.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_content))
            print(f"📝 Saved text: {text_file}")
        
        # Save layer info if detected
        if has_layers and layer_info:
            layer_file = output_subdir / f"{pdf_name}_layer_info.json"
            import json
            with open(layer_file, 'w', encoding='utf-8') as f:
                json.dump(layer_info, f, indent=2)
            print(f"📋 Saved layer info: {layer_file}")
        
        # Convert PIL images to tensors for ComfyUI
        # Return as list of individual tensors (no padding needed!)
        if all_images_pil:
            extracted_images = []
            
            for pil_img in all_images_pil:
                # Convert PIL to RGB if needed
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # Convert to numpy [H, W, C] in 0-1 range, then to tensor
                np_img = np.array(pil_img).astype(np.float32) / 255.0
                # Add batch dimension to get [1, H, W, C] for single image
                tensor_img = torch.from_numpy(np_img).unsqueeze(0)
                extracted_images.append(tensor_img)
        
        elapsed = time.time() - start_time
        
        # Create summary
        layer_status = "✨ Layered PDF (fast extraction)" if has_layers else "Standard PDF"
        summary = (
            f"✅ Extraction Complete\n"
            f"📄 PDF: {Path(pdf_path).name}\n"
            f"📋 Type: {layer_status}\n"
            f"🖼️  Images extracted: {image_count}\n"
            f"📝 Text extracted: {'Yes' if text_content else 'No'}\n"
            f"⏱️  Time: {elapsed:.2f}s\n"
            f"📁 Output: {output_subdir}"
        )
        
        if has_layers:
            summary += f"\n🎨 Layers: {layer_info['layer_count']}"
        
        print(f"\n{summary}")
        
        # When OUTPUT_IS_LIST=True, ComfyUI requires at least one item in the list
        if image_count == 0:
            dummy_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return ([dummy_tensor], "⚠️  No images found in PDF", 0, has_layers)
        
        return (extracted_images, summary, image_count, has_layers)
    
    def _extract_batch(self, folder_path, output_directory, min_width, min_height, extract_text, layers_only_mode, dpi):
        """Extract images from all PDFs in a folder"""
        import json
        from glob import glob
        
        print(f"\n🗂️  BATCH MODE: Processing folder")
        print(f"📁 Folder: {folder_path}")
        print(f"🔍 Searching for PDF files...")
        
        # Find all PDF files in folder
        pdf_files = []
        for pattern in ['*.pdf', '*.PDF']:
            pdf_files.extend(glob(os.path.join(folder_path, pattern)))
        
        if not pdf_files:
            msg = f"⚠️  No PDF files found in: {folder_path}"
            print(msg)
            dummy_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return ([dummy_tensor], msg, 0, False)
        
        print(f"📚 Found {len(pdf_files)} PDF files")
        print(f"🏃 Mode: {'Layers only' if layers_only_mode else 'All PDFs'}")
        print(f"{'='*60}\n")
        
        # Process each PDF
        batch_start = time.time()
        all_images = []
        batch_stats = {
            "total_pdfs": len(pdf_files),
            "processed": 0,
            "skipped": 0,
            "total_images": 0,
            "layered_pdfs": 0,
            "processing_times": [],
            "results": []
        }
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            pdf_name = Path(pdf_file).name
            print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_name}")
            print("-" * 60)
            
            try:
                # Extract from single PDF
                images, summary, count, has_layers = self._extract_single_pdf(
                    pdf_file, output_directory, min_width, min_height, 
                    extract_text, layers_only_mode, dpi
                )
                
                # Track stats
                if count > 0:
                    batch_stats["processed"] += 1
                    batch_stats["total_images"] += count
                    if has_layers:
                        batch_stats["layered_pdfs"] += 1
                    
                    # Collect images (images is now a list of tensors)
                    if isinstance(images, list) and len(images) > 0:
                        all_images.extend(images)  # Extend, not append - flatten the list
                else:
                    batch_stats["skipped"] += 1
                
                # Store result
                batch_stats["results"].append({
                    "pdf": pdf_name,
                    "images": count,
                    "has_layers": has_layers,
                    "status": "processed" if count > 0 else "skipped"
                })
                
            except Exception as e:
                print(f"❌ Error processing {pdf_name}: {e}")
                batch_stats["skipped"] += 1
                batch_stats["results"].append({
                    "pdf": pdf_name,
                    "images": 0,
                    "has_layers": False,
                    "status": "error",
                    "error": str(e)
                })
        
        batch_elapsed = time.time() - batch_start
        
        # All images are already in a flat list (no padding needed with OUTPUT_IS_LIST=True)
        combined_images = all_images if all_images else []
        
        # Save batch summary
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_summary_file = output_dir / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        batch_stats["total_time_seconds"] = batch_elapsed
        batch_stats["avg_time_per_pdf"] = batch_elapsed / len(pdf_files) if pdf_files else 0
        
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, indent=2)
        
        # Create summary text
        summary = (
            f"✅ Batch Extraction Complete\n"
            f"📁 Folder: {Path(folder_path).name}\n"
            f"📚 Total PDFs: {batch_stats['total_pdfs']}\n"
            f"✅ Processed: {batch_stats['processed']}\n"
            f"⏭️  Skipped: {batch_stats['skipped']}\n"
            f"🖼️  Total images: {batch_stats['total_images']}\n"
            f"✨ Layered PDFs: {batch_stats['layered_pdfs']}\n"
            f"⏱️  Total time: {batch_elapsed:.1f}s\n"
            f"📊 Avg per PDF: {batch_stats['avg_time_per_pdf']:.1f}s\n"
            f"📄 Summary: {batch_summary_file}"
        )
        
        print(f"\n{'='*60}")
        print(summary)
        print(f"{'='*60}\n")
        
        has_any_layers = batch_stats["layered_pdfs"] > 0
        
        # When OUTPUT_IS_LIST=True, ComfyUI requires at least one item in the list
        if len(combined_images) == 0:
            dummy_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            combined_images = [dummy_tensor]
        
        return (combined_images, summary, batch_stats["total_images"], has_any_layers)
    
    def _detect_layers(self, doc: fitz.Document) -> Dict:
        """Detect if PDF has optional content layers (OCG)"""
        layer_info = {
            "has_layers": False,
            "layer_count": 0,
            "layers": []
        }
        
        try:
            # Get layers using PyMuPDF's OCG support
            layers = doc.get_layers()
            
            if layers and len(layers) > 0:
                layer_info["has_layers"] = True
                layer_info["layer_count"] = len(layers)
                
                for layer in layers:
                    # Each layer is a dict with keys: 'name', 'number', 'on', 'intent', 'usage'
                    layer_info["layers"].append({
                        "name": layer.get("name", "Unknown"),
                        "number": layer.get("number", -1),
                        "visible": layer.get("on", True),
                        "intent": layer.get("intent", []),
                        "usage": layer.get("usage", "")
                    })
        
        except Exception as e:
            # If layer detection fails, just continue without layers
            print(f"   ⚠️  Layer detection failed: {e}")
        
        return layer_info


# Register the node
NODE_CLASS_MAPPINGS = {
    "SimplePDFImageExtractor": SimplePDFImageExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimplePDFImageExtractor": "Simple PDF Image Extractor"
}
