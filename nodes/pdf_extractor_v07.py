"""
Pdf Extractor V07

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
Enhanced PDF Extractor v0.7.0 - Simplified Florence2-focused Implementation
Dramatically simplified from v0.6.0, focusing on Florence2 detection and clean architecture.

Key Changes from v0.6.0:
- Removed GroundingDINO/SAM complexity entirely
- Florence2-only detection pipeline
- Simplified spread detection using Florence2 bounding boxes
- Clean color profile management
- Join spreads BEFORE cropping and enhancement
- Linear processing flow
- Minimal configuration options

Author: Eric Hiss (GitHub: EricRollei)
Enhanced by: Claude Sonnet 4 AI Assistant
Version: 0.7.0
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
from typing import Dict, List, Any, Optional, Tuple
import folder_paths
import numpy as np
from PIL import Image, ImageEnhance


# PDF processing imports
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not available")

# ComfyUI folder paths
try:
    import folder_paths
    # Get the ComfyUI base path
    COMFYUI_BASE_PATH = os.path.dirname(folder_paths.models_dir)
except ImportError:
    COMFYUI_BASE_PATH = "."
    print("ComfyUI folder_paths not available, using current directory")

# Import our working Florence2 detector and modern enhancer
try:
    from PDF_tools import ModernImageEnhancer
    ENHANCER_AVAILABLE = True
    print("✅ Modern image enhancer available")
except ImportError:
    ENHANCER_AVAILABLE = False
    print("❌ Modern image enhancer not available")
    class ModernImageEnhancer: pass

try:
    from PDF_tools import Florence2RectangleDetector, BoundingBox
    FLORENCE2_AVAILABLE = True
    print("✅ Florence2 detector available")
except ImportError:
    FLORENCE2_AVAILABLE = False
    print("❌ Florence2 detector not available")
    class Florence2RectangleDetector: pass
    class BoundingBox: pass

# Color profile management
try:
    from PIL import ImageCms
    COLORMGMT_AVAILABLE = True
except ImportError:
    COLORMGMT_AVAILABLE = False
    print("Color management not available")


# --- Data Classes (Simplified) ---

class PageType(Enum):
    SINGLE_PAGE = "single_page"
    DOUBLE_PAGE_LEFT = "double_page_left" 
    DOUBLE_PAGE_RIGHT = "double_page_right"
    JOINED_SPREAD = "joined_spread"

class ImageQuality(Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair" 
    POOR = "Poor"

@dataclass
class ExtractedImage:
    filename: str
    page_num: int
    image_index: int
    width: int
    height: int
    file_size_bytes: int
    quality_score: ImageQuality
    page_type: PageType
    detection_confidence: float
    bbox: Tuple[float, float, float, float]
    original_colorspace: str
    extraction_method: str

@dataclass
class ExtractedText:
    page_num: int
    text_content: str
    bbox: Optional[Tuple[float, float, float, float]] = None

@dataclass 
class JoinedSpread:
    filename: str
    left_page_num: int
    right_page_num: int
    left_image_filename: str
    right_image_filename: str
    combined_width: int
    combined_height: int
    confidence_score: float
    join_method: str = "simple_florence2"

@dataclass
class ProcessingReport:
    pdf_filename: str
    total_pages: int
    processing_time: float
    images_extracted: int
    images_joined: int
    text_extracted_pages: int
    output_directory: str
    extracted_images: List[ExtractedImage]
    extracted_text: List[ExtractedText]
    joined_spreads: List[JoinedSpread]

@dataclass
class ProcessingConfig:
    """Simplified configuration with only essential parameters"""
    min_image_size: int = 200
    min_image_area: int = 40000
    crop_margin: int = 5
    join_spreads: bool = True
    save_debug_images: bool = False
    florence2_prompt: str = "rectangular images in page OR photograph OR illustration OR diagram"
    min_detection_confidence: float = 0.3
    debug_mode: bool = False
    # Enhancement settings
    enable_enhancement: bool = True
    enhancement_profile: str = "Digital Magazine"
    enhancement_strength: float = 1.0


# --- Color Profile Management ---

class ColorProfileManager:
    """Clean color profile management"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self._srgb_profile = None
        
    def ensure_color_profile(self, image: Image.Image) -> Image.Image:
        """Ensure image has a color profile, add sRGB if missing"""
        if not COLORMGMT_AVAILABLE:
            return image
            
        # Check if already has profile
        if hasattr(image, 'info') and 'icc_profile' in image.info and image.info['icc_profile']:
            if self.debug_mode:
                print(f"    Image has existing ICC profile ({len(image.info['icc_profile'])} bytes)")
            return image
        
        # Add sRGB profile for RGB images
        if image.mode == 'RGB':
            try:
                if self._srgb_profile is None:
                    srgb_profile_obj = ImageCms.createProfile('sRGB')
                    # Use tobytes() method for newer PIL versions
                    self._srgb_profile = srgb_profile_obj.tobytes()
                    
                if not hasattr(image, 'info'):
                    image.info = {}
                image.info['icc_profile'] = self._srgb_profile
                
                if self.debug_mode:
                    print(f"    Added sRGB profile ({len(self._srgb_profile)} bytes)")
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"    Could not add sRGB profile: {e}")
        
        return image
    
    def preserve_profile(self, source: Image.Image, target: Image.Image) -> Image.Image:
        """Copy color profile from source to target"""
        try:
            if (hasattr(source, 'info') and 'icc_profile' in source.info and 
                source.info['icc_profile'] and target.mode in ('RGB', 'L')):
                
                if not hasattr(target, 'info'):
                    target.info = {}
                target.info['icc_profile'] = source.info['icc_profile']
                
                if self.debug_mode:
                    print(f"    Preserved color profile ({len(source.info['icc_profile'])} bytes)")
        except Exception as e:
            if self.debug_mode:
                print(f"    Could not preserve profile: {e}")
        
        return target


# --- Spread Detection ---

class SpreadDetector:
    """Simplified spread detection using Florence2 bounding boxes"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        

    def filter_florence2_boxes(self, image: Image.Image, bounding_boxes: List[BoundingBox], 
                            min_box_area: int) -> List[BoundingBox]:
        """Smart filtering using PIL dimensions and container box logic"""
        
        if not bounding_boxes:
            return []
        
        width, height = image.size
        image_area = width * height
        
        if self.debug_mode:
            print(f"    Filtering {len(bounding_boxes)} Florence2 boxes for PIL image {width}×{height}")
        
        # Normalize all boxes first
        normalized_boxes = self.normalize_bounding_boxes(bounding_boxes, width, height)
        
        # Categorize boxes
        full_page_boxes = []
        content_boxes = []
        small_boxes = []
        
        for norm_box in normalized_boxes:
            box = norm_box["original_box"]
            
            if norm_box["is_full_page"]:
                full_page_boxes.append(norm_box)
                if self.debug_mode:
                    print(f"      Full-page box: {box.to_tuple()}, coverage={norm_box['coverage_ratio']:.1%}")
            elif box.area >= min_box_area:
                content_boxes.append(norm_box)
                if self.debug_mode:
                    print(f"      Content box: {box.to_tuple()}, coverage={norm_box['coverage_ratio']:.1%}")
            else:
                small_boxes.append(norm_box)
                if self.debug_mode:
                    print(f"      Too small box: {box.to_tuple()}, area={box.area} < {min_box_area}")
        
        # Apply container box logic - remove boxes that are contained within others
        content_boxes = self._filter_container_boxes(content_boxes)
        
        # Apply filtering rules
        if len(bounding_boxes) == 1:
            # Single box - check if it's full page or content
            if full_page_boxes:
                if self.debug_mode:
                    print(f"    Rule: Single full-page box -> skip")
                return []  # Skip full-page-only images
            else:
                if self.debug_mode:
                    print(f"    Rule: Single content box -> keep")
                return [norm_box["original_box"] for norm_box in content_boxes]
        
        elif len(full_page_boxes) >= 1 and content_boxes:
            # Check if this looks like a full-page image with detected elements inside
            largest_full_page = max(full_page_boxes, key=lambda x: x["original_box"].area)
            largest_content = max(content_boxes, key=lambda x: x["original_box"].area)
            
            # If the largest content box is much smaller than the full-page box, 
            # this is likely a full-page image with detected elements inside
            area_ratio = largest_content["original_box"].area / largest_full_page["original_box"].area
            
            if area_ratio < 0.3:  # Content is less than 30% of full page
                if self.debug_mode:
                    print(f"    Rule: Full-page image with internal elements (ratio: {area_ratio:.2f}) -> keep full-page")
                return [largest_full_page["original_box"]]
            else:
                # Content boxes are significant, keep them
                if self.debug_mode:
                    print(f"    Rule: Full-page + significant content boxes -> keep {len(content_boxes)} content boxes")
                return [norm_box["original_box"] for norm_box in content_boxes]
        
        elif len(full_page_boxes) > 1 and not content_boxes:
            # Multiple full-page boxes: treat as single full-page image, use smallest
            smallest_full_page = min(full_page_boxes, key=lambda x: x["original_box"].area)
            if self.debug_mode:
                print(f"    Rule: Multiple full-page boxes -> keep smallest")
            return [smallest_full_page["original_box"]]
        
        else:
            # Other cases: return content boxes only
            if self.debug_mode:
                print(f"    Rule: Default -> keep {len(content_boxes)} content boxes")
            return [norm_box["original_box"] for norm_box in content_boxes]

    def _filter_container_boxes(self, normalized_boxes: List[Dict]) -> List[Dict]:
        """Remove boxes that are contained within other boxes"""
        
        filtered = []
        
        for i, box_a in enumerate(normalized_boxes):
            is_container = True
            
            for j, box_b in enumerate(normalized_boxes):
                if i != j and self._box_contains_relative(box_b, box_a):
                    is_container = False  # box_a is contained in box_b
                    if self.debug_mode:
                        print(f"      Removing contained box: {box_a['original_box'].to_tuple()}")
                    break
            
            if is_container:
                filtered.append(box_a)
        
        return filtered

    def _box_contains_relative(self, outer: Dict, inner: Dict) -> bool:
        """Check if outer box completely contains inner box using relative coordinates"""
        return (outer["rel_x1"] <= inner["rel_x1"] and 
                outer["rel_y1"] <= inner["rel_y1"] and 
                outer["rel_x2"] >= inner["rel_x2"] and 
                outer["rel_y2"] >= inner["rel_y2"])
            

    def analyze_spread_characteristics(self, normalized_boxes: List[Dict], page_num: int = None) -> Dict:
        """Analyze spread characteristics using dual-pattern detection"""
        
        if not normalized_boxes:
            return {"page_type": PageType.SINGLE_PAGE, "confidence": 0.0, "pattern": "no_content"}
        
        # Find the main content box (largest non-full-page box)
        content_boxes = [box for box in normalized_boxes if not box["is_full_page"]]
        if not content_boxes:
            return {"page_type": PageType.SINGLE_PAGE, "confidence": 0.5, "pattern": "only_full_page"}
        
        # Use the largest content box for analysis
        main_box = max(content_boxes, key=lambda x: x["rel_area"])
        
        if self.debug_mode:
            print(f"    Analyzing main content box: {main_box['original_box'].to_tuple()}")
            print(f"    Content size: {main_box['rel_width']:.3f} × {main_box['rel_height']:.3f}")
        
        # Try Pattern 1: Full-Page Spreads
        full_page_result = self.detect_full_page_spread(main_box, page_num)
        if full_page_result:
            return full_page_result
        
        # Try Pattern 2: Margin-Based Spreads  
        margin_result = self.detect_margin_based_spread(main_box)
        if margin_result:
            return margin_result
        
        # Default to single page
        return {
            "page_type": PageType.SINGLE_PAGE,
            "confidence": 0.5,
            "distribution": "single_default",
            "pattern": "single",
            "main_content_box": main_box
        }

    def detect_full_page_spread(self, normalized_box: Dict, page_num: int = None) -> Dict:
        """Detect full-page spreads where both pages use full dimensions"""
        
        left_margin = normalized_box["rel_x1"]
        right_margin = 1.0 - normalized_box["rel_x2"]
        top_margin = normalized_box["rel_y1"] 
        bottom_margin = 1.0 - normalized_box["rel_y2"]
        
        # Very small margins all around = full page usage
        all_margins_small = (
            left_margin < 0.05 and      # Left margin < 5%
            right_margin < 0.05 and     # Right margin < 5%
            top_margin < 0.05 and       # Top margin < 5%
            bottom_margin < 0.05        # Bottom margin < 5%
        )
        
        uses_full_dimensions = (
            normalized_box["rel_width"] > 0.9 and
            normalized_box["rel_height"] > 0.9
        )
        
        if self.debug_mode:
            print(f"    Full-page check: margins L={left_margin:.3f} R={right_margin:.3f} T={top_margin:.3f} B={bottom_margin:.3f}")
            print(f"    All small: {all_margins_small}, Full dims: {uses_full_dimensions}")
        
        if all_margins_small and uses_full_dimensions:
            # Use pagination to determine left vs right
            if page_num is not None:
                if page_num % 2 == 1:  # Even page number = left page
                    page_type = PageType.DOUBLE_PAGE_LEFT
                    distribution = "full_page_left"
                elif page_num % 2 == 0:  # Odd page number = right page  
                    page_type = PageType.DOUBLE_PAGE_RIGHT
                    distribution = "full_page_right"
                else:
                    page_type = PageType.SINGLE_PAGE
                    distribution = "full_page_unknown"
            else:
                # No page number, default to left
                page_type = PageType.DOUBLE_PAGE_LEFT
                distribution = "full_page_no_pagenum"
            
            if self.debug_mode:
                print(f"    ✅ FULL-PAGE SPREAD detected: {page_type}, page {page_num + 1 if page_num is not None else 'unknown'}")
            
            return {
                "page_type": page_type,
                "confidence": 0.8,
                "distribution": distribution,
                "pattern": "full_page",
                "left_margin": left_margin,
                "right_margin": right_margin,
                "main_content_box": normalized_box
            }
        
        return None

    def detect_margin_based_spread(self, normalized_box: Dict) -> Dict:
        """Detect spreads where one page has large outside margin"""
        
        left_margin = normalized_box["rel_x1"]
        right_margin = 1.0 - normalized_box["rel_x2"]
        top_margin = normalized_box["rel_y1"]
        bottom_margin = 1.0 - normalized_box["rel_y2"]
        
        # Relaxed criteria for margin-based detection
        margin_difference = abs(left_margin - right_margin)
        smaller_margin = min(left_margin, right_margin)
        uses_height = normalized_box["rel_height"] > 0.7  # Relaxed from 0.8
        uses_width = normalized_box["rel_width"] > 0.6   # Relaxed from 0.7
        
        # Vertical usage should be good
        good_vertical_usage = (
            normalized_box["rel_height"] > 0.7 and
            top_margin < 0.15 and 
            bottom_margin < 0.15
        )
        
        if self.debug_mode:
            print(f"    Margin-based check: L={left_margin:.3f} R={right_margin:.3f}")
            print(f"    Margin diff: {margin_difference:.3f}, smaller: {smaller_margin:.3f}")
            print(f"    Uses height: {uses_height}, width: {uses_width}, vertical: {good_vertical_usage}")
        
        # One margin significantly larger than the other + good usage
        if (margin_difference > 0.1 and smaller_margin < 0.1 and  # Relaxed from 0.15 and 0.08
            uses_width and good_vertical_usage):
            
            if left_margin > right_margin:
                page_type = PageType.DOUBLE_PAGE_LEFT
                distribution = "margin_left"
            else:
                page_type = PageType.DOUBLE_PAGE_RIGHT  
                distribution = "margin_right"
            
            confidence = 0.6 + min(0.3, margin_difference * 2)
            
            if self.debug_mode:
                print(f"    ✅ MARGIN-BASED SPREAD detected: {page_type}, confidence={confidence:.2f}")
            
            return {
                "page_type": page_type,
                "confidence": confidence,
                "distribution": distribution,
                "pattern": "margin_based",
                "left_margin": left_margin,
                "right_margin": right_margin,
                "margin_difference": margin_difference,
                "main_content_box": normalized_box
            }
        
        if self.debug_mode:
            print(f"    ❌ No spread pattern detected")
        
        return None

    def detect_pagination_spreads(self, page_analyses: List[Dict]) -> List[Tuple[int, int, float, str]]:
        """Detect spreads using pagination logic + full-page analysis"""
        
        if self.debug_mode:
            print(f"\n🔍 Pagination spread analysis for {len(page_analyses)} pages:")
            for i, analysis in enumerate(page_analyses):
                page_type = analysis.get("page_type", "unknown")
                is_full_page = analysis.get("is_full_page_candidate", False)
                aspect_ratio = analysis.get("aspect_ratio", 0)
                print(f"    Page {i+1}: {page_type}, full-page={is_full_page}, AR={aspect_ratio:.2f}")
        
        spread_pairs = []
        
        for i in range(len(page_analyses) - 1):
            left_page_num = i
            right_page_num = i + 1
            
            # Skip page 0 (cover) - start from page 1
            if left_page_num == 0:
                continue
                
            # Check if this follows even-odd pagination pattern
            is_even_odd_pattern = (left_page_num % 2 == 1) and (right_page_num % 2 == 0)
            
            left_analysis = page_analyses[i]
            right_analysis = page_analyses[i + 1]
            
            # Case 1: Both pages are full-page images
            # Use enhanced validation
            is_valid, validation_conf, reason = self._validate_spread_pair_enhanced(
                left_analysis, right_analysis, left_page_num, right_page_num
            )

            if is_valid and (self._is_likely_full_page_spread(left_analysis) and 
                            self._is_likely_full_page_spread(right_analysis)):
                
                base_confidence = 0.8 if is_even_odd_pattern else 0.6
                
                # Boost confidence with partition analysis if images are available
                partition_boost = 0.0
                if "image" in left_analysis and "image" in right_analysis:
                    left_partition = self._analyze_image_partitions(left_analysis["image"])
                    right_partition = self._analyze_image_partitions(right_analysis["image"])
                    
                    if left_partition["detected"] or right_partition["detected"]:
                        avg_partition_conf = (left_partition["confidence"] + right_partition["confidence"]) / 2
                        partition_boost = avg_partition_conf * 0.2  # Up to 0.2 boost
                        
                        if self.debug_mode:
                            print(f"    📊 Partition analysis boost: {partition_boost:.3f}")
                
                final_confidence = min(0.95, validation_conf + partition_boost)
                spread_pairs.append((i, i + 1, final_confidence, "full_page_pagination"))
                
                if self.debug_mode:
                    print(f"    📖 Full-page spread detected: pages {left_page_num+1}-{right_page_num+1}, confidence={final_confidence:.2f}")
                    
                
            # Case 2: Traditional margin-based detection  
            elif (left_analysis["page_type"] == PageType.DOUBLE_PAGE_LEFT and
                right_analysis["page_type"] == PageType.DOUBLE_PAGE_RIGHT):
                
                base_confidence = min(left_analysis["confidence"], right_analysis["confidence"])
                pagination_boost = 0.2 if is_even_odd_pattern else -0.1
                
                # Add partition analysis boost
                partition_boost = 0.0
                if "image" in left_analysis and "image" in right_analysis:
                    left_partition = self._analyze_image_partitions(left_analysis["image"])
                    right_partition = self._analyze_image_partitions(right_analysis["image"])
                    
                    if left_partition["detected"] or right_partition["detected"]:
                        avg_partition_conf = (left_partition["confidence"] + right_partition["confidence"]) / 2
                        partition_boost = avg_partition_conf * 0.15  # Up to 0.15 boost
                
                final_confidence = base_confidence + pagination_boost + partition_boost
                
                if final_confidence > 0.5:
                    method = "margin_based_partition" if partition_boost > 0 else "margin_based"
                    spread_pairs.append((i, i + 1, final_confidence, method))
                    
                    if self.debug_mode:
                        print(f"    📖 Margin-based spread detected: pages {left_page_num+1}-{right_page_num+1}, confidence={final_confidence:.2f}")

            # Case 3: Partition analysis suggests spread even if other methods didn't detect it
            elif ("image" in left_analysis and "image" in right_analysis):
                left_partition = self._analyze_image_partitions(left_analysis["image"])
                right_partition = self._analyze_image_partitions(right_analysis["image"])
                
                # Strong partition evidence can stand alone
                if (left_partition["confidence"] > 0.6 or right_partition["confidence"] > 0.6):
                    avg_partition_conf = (left_partition["confidence"] + right_partition["confidence"]) / 2
                    pagination_boost = 0.1 if is_even_odd_pattern else 0.0
                    final_confidence = avg_partition_conf + pagination_boost
                    
                    if final_confidence > 0.5:
                        spread_pairs.append((i, i + 1, final_confidence, "partition_only"))
                        
                        if self.debug_mode:
                            print(f"    📊 Partition-only spread detected: pages {left_page_num+1}-{right_page_num+1}, confidence={final_confidence:.2f}")
            

        return spread_pairs

    def _is_likely_full_page_spread(self, analysis: Dict) -> bool:
        """Check if this looks like a full-page part of a spread"""
        
        # Check the new flag first
        if analysis.get("is_full_page_candidate", False):
            aspect_ratio = analysis.get("aspect_ratio", 1.0)
            
            # Full-page images with spread-like characteristics
            if aspect_ratio > 0.6:  # Not too tall/narrow
                if self.debug_mode:
                    print(f"        Full-page spread candidate: AR={aspect_ratio:.2f}")
                return True
        
        # Fallback to original logic
        normalized_boxes = analysis.get("normalized_boxes", [])
        
        for box_data in normalized_boxes:
            if not box_data["is_full_page"]:
                continue
                
            box = box_data["original_box"]
            aspect_ratio = box.width / box.height
            
            if aspect_ratio > 1.2:
                if self.debug_mode:
                    print(f"        Box-based spread candidate: AR={aspect_ratio:.2f}")
                return True
        
        return False

    def _detect_center_gutter(self, image: Image.Image) -> Dict:
        """Detect vertical center line that suggests a spread"""
        if not CV2_AVAILABLE:
            return {"detected": False}
        
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        height, width = gray.shape
        
        # Look for vertical lines in center third of image
        center_start = width // 3
        center_end = 2 * width // 3
        center_region = gray[:, center_start:center_end]
        
        # Edge detection
        edges = cv2.Canny(center_region, 30, 100)
        
        # Look for strong vertical lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(height*0.3))
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 10 and abs(y2 - y1) > height * 0.4:  # Vertical line
                    if self.debug_mode:
                        print(f"    ✅ Center gutter detected at x={center_start + x1}")
                    return {
                        "detected": True, 
                        "confidence": 0.7,
                        "gutter_x": center_start + x1
                    }
        
        return {"detected": False}

    def _validate_spread_pair_enhanced(self, left_analysis: Dict, right_analysis: Dict, 
                                    left_page_num: int, right_page_num: int) -> Tuple[bool, float, str]:
        """Enhanced validation using height matching and inside/outside margin analysis"""
        
        # Must be consecutive pages
        if right_page_num != left_page_num + 1:
            return False, 0.0, "non_consecutive"
        
        # Get image dimensions
        left_size = left_analysis.get("image_size", (0, 0))
        right_size = right_analysis.get("image_size", (0, 0))
        
        if left_size[1] == 0 or right_size[1] == 0:
            return False, 0.0, "no_dimensions"
        
        # CRITICAL: Height matching within 1%
        height_diff_pct = abs(left_size[1] - right_size[1]) / max(left_size[1], right_size[1])
        if height_diff_pct > 0.01:  # More than 1% height difference
            if self.debug_mode:
                print(f"    ❌ Height mismatch: {left_size[1]} vs {right_size[1]} ({height_diff_pct:.1%})")
            return False, 0.0, f"height_mismatch_{height_diff_pct:.1%}"
        
        confidence = 0.5  # Base confidence for height match
        validation_reasons = ["height_match"]
        
        # Analyze inside vs outside margins from Florence2 data
        margin_boost = self._analyze_inside_outside_margins(left_analysis, right_analysis)
        confidence += margin_boost
        if margin_boost > 0:
            validation_reasons.append(f"margin_analysis_{margin_boost:.2f}")
        
        # Pagination boost
        is_even_odd = (left_page_num % 2 == 1) and (right_page_num % 2 == 0)
        if is_even_odd:
            confidence += 0.15
            validation_reasons.append("even_odd_pagination")
        
        # Aspect ratio similarity (less important but still useful)
        left_aspect = left_size[0] / left_size[1]
        right_aspect = right_size[0] / right_size[1]
        aspect_diff = abs(left_aspect - right_aspect)
        if aspect_diff < 0.2:  # Similar aspect ratios
            confidence += 0.1
            validation_reasons.append("similar_aspect")
        
        final_confidence = min(0.95, confidence)
        reason = "+".join(validation_reasons)
        
        if self.debug_mode:
            print(f"    📏 Enhanced validation: heights {left_size[1]}={right_size[1]} ({height_diff_pct:.2%}), "
                f"confidence={final_confidence:.2f}, reasons={reason}")
        
        return final_confidence > 0.6, final_confidence, reason


    def _analyze_inside_outside_margins(self, left_analysis: Dict, right_analysis: Dict) -> float:
        """Analyze inside vs outside margins to validate spread characteristics"""
        
        left_boxes = left_analysis.get("normalized_boxes", [])
        right_boxes = right_analysis.get("normalized_boxes", [])
        
        if not left_boxes or not right_boxes:
            return 0.0
        
        # Find the main content box for each page (largest non-full-page box)
        left_content = None
        right_content = None
        
        for box_data in left_boxes:
            if not box_data.get("is_full_page", False):
                if left_content is None or box_data["rel_area"] > left_content["rel_area"]:
                    left_content = box_data
        
        for box_data in right_boxes:
            if not box_data.get("is_full_page", False):
                if right_content is None or box_data["rel_area"] > right_content["rel_area"]:
                    right_content = box_data
        
        if not left_content or not right_content:
            return 0.0
        
        # Calculate margins as percentages
        left_left_margin = left_content["rel_x1"]          # Outside margin
        left_right_margin = 1.0 - left_content["rel_x2"]   # Inside margin  
        right_left_margin = right_content["rel_x1"]        # Inside margin
        right_right_margin = 1.0 - right_content["rel_x2"] # Outside margin
        
        # For spreads: inside margins should be smaller than outside margins
        left_inside_smaller = left_right_margin < left_left_margin
        right_inside_smaller = right_left_margin < right_right_margin
        
        # Calculate margin ratios
        left_margin_ratio = left_right_margin / (left_left_margin + 0.001)  # inside/outside
        right_margin_ratio = right_left_margin / (right_right_margin + 0.001)  # inside/outside
        
        confidence_boost = 0.0
        
        # Both pages show inside < outside pattern
        if left_inside_smaller and right_inside_smaller:
            # Stronger pattern = higher boost
            avg_ratio = (left_margin_ratio + right_margin_ratio) / 2
            confidence_boost = max(0.0, (1.0 - avg_ratio) * 0.3)  # Up to 0.3 boost
            
            if self.debug_mode:
                print(f"    📐 Inside/outside margins: L({left_right_margin:.1%}/{left_left_margin:.1%}={left_margin_ratio:.2f}) "
                    f"R({right_left_margin:.1%}/{right_right_margin:.1%}={right_margin_ratio:.2f}) -> boost={confidence_boost:.2f}")
        
        # Very small inside margins suggest gutter
        small_inside_margins = (left_right_margin < 0.05 and right_left_margin < 0.05)
        if small_inside_margins:
            confidence_boost += 0.2
            if self.debug_mode:
                print(f"    📐 Small inside margins detected: L={left_right_margin:.1%}, R={right_left_margin:.1%}")
        
        return min(0.4, confidence_boost)  # Cap at 0.4
        

    def _analyze_image_partitions(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze left and right halves for content differences - adapted from v0.6"""
        if not CV2_AVAILABLE:
            return {"detected": False, "confidence": 0.0, "method": "cv2_unavailable"}
        
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            height, width = gray.shape
            
            # Split image into left and right halves
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            
            # Calculate histograms
            left_hist = cv2.calcHist([left_half], [0], None, [256], [0, 256])
            right_hist = cv2.calcHist([right_half], [0], None, [256], [0, 256])
            
            # Compare histograms using multiple methods
            correlation = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
            chi_square = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CHISQR)
            
            # Calculate standard deviation differences
            left_std = np.std(left_half)
            right_std = np.std(right_half)
            std_ratio = abs(left_std - right_std) / max(left_std, right_std, 1.0)
            
            # Calculate mean differences
            left_mean = np.mean(left_half)
            right_mean = np.mean(right_half)
            mean_diff = abs(left_mean - right_mean) / 255.0
            
            # Edge density comparison
            left_edges = cv2.Canny(left_half, 50, 150)
            right_edges = cv2.Canny(right_half, 50, 150)
            left_edge_density = np.sum(left_edges > 0) / left_edges.size
            right_edge_density = np.sum(right_edges > 0) / right_edges.size
            edge_density_diff = abs(left_edge_density - right_edge_density)
            
            # Scoring: Lower correlation and higher differences suggest spread
            correlation_score = 1 - correlation  # Invert correlation (lower = more different)
            std_score = min(1.0, std_ratio * 2)
            mean_score = min(1.0, mean_diff * 3)
            edge_score = min(1.0, edge_density_diff * 5)
            
            # Combine scores with weights
            total_score = (correlation_score * 0.4 + std_score * 0.25 + 
                        mean_score * 0.2 + edge_score * 0.15)
            
            # Determine if this indicates a spread
            detected = total_score > 0.3  # Threshold for detection
            confidence = min(1.0, total_score)
            
            if self.debug_mode and detected:
                print(f"    📊 Partition analysis: correlation={correlation:.3f}, std_ratio={std_ratio:.3f}")
                print(f"    📊 mean_diff={mean_diff:.3f}, edge_diff={edge_density_diff:.3f}, score={total_score:.3f}")
            
            return {
                "detected": detected,
                "confidence": confidence,
                "correlation": correlation,
                "std_ratio": std_ratio,
                "mean_diff": mean_diff,
                "edge_density_diff": edge_density_diff,
                "total_score": total_score,
                "method": "partition_analysis"
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ❌ Partition analysis failed: {e}")
            return {"detected": False, "confidence": 0.0, "method": "partition_error"}
            
    def normalize_bounding_boxes(self, bounding_boxes: List[BoundingBox], image_width: int, image_height: int) -> List[Dict]:
        """Normalize Florence2 bounding boxes to relative coordinates using PIL dimensions"""
        
        normalized_boxes = []
        image_area = image_width * image_height
        
        for box in bounding_boxes:
            # Calculate relative coordinates (0.0 to 1.0)
            rel_x1 = box.x1 / image_width
            rel_y1 = box.y1 / image_height
            rel_x2 = box.x2 / image_width
            rel_y2 = box.y2 / image_height
            rel_center_x = (box.x1 + box.x2) / 2 / image_width
            rel_center_y = (box.y1 + box.y2) / 2 / image_height
            
            # Calculate coverage and determine if it's a full-page box
            coverage_ratio = box.area / image_area
            width_ratio = box.width / image_width
            height_ratio = box.height / image_height
            
            # Use margin-based detection with fallback to coverage
            left_margin_pct = box.x1 / image_width
            top_margin_pct = box.y1 / image_height  
            right_margin_pct = (image_width - box.x2) / image_width
            bottom_margin_pct = (image_height - box.y2) / image_height

            # Primary criteria: ALL margins must be <= 8%
            strict_margin_threshold = 0.08
            is_full_page_strict = (
                left_margin_pct <= strict_margin_threshold and
                top_margin_pct <= strict_margin_threshold and
                right_margin_pct <= strict_margin_threshold and
                bottom_margin_pct <= strict_margin_threshold
            )

            # Fallback criteria: coverage > 90% AND no margin > 15%
            max_margin = max(left_margin_pct, top_margin_pct, right_margin_pct, bottom_margin_pct)
            is_full_page_fallback = (coverage_ratio > 0.90 and max_margin <= 0.15)

            # Use either criteria
            is_full_page = is_full_page_strict or is_full_page_fallback

            if self.debug_mode and is_full_page:
                method = "strict_margins" if is_full_page_strict else "coverage_fallback"
                print(f"      Full-page by {method}: L={left_margin_pct:.1%} T={top_margin_pct:.1%} R={right_margin_pct:.1%} B={bottom_margin_pct:.1%}")
                
            
            normalized = {
                "rel_x1": rel_x1,
                "rel_y1": rel_y1,
                "rel_x2": rel_x2,
                "rel_y2": rel_y2,
                "rel_width": rel_x2 - rel_x1,
                "rel_height": rel_y2 - rel_y1,
                "rel_center_x": rel_center_x,
                "rel_center_y": rel_center_y,
                "rel_area": box.area / image_area,
                "coverage_ratio": coverage_ratio,
                "width_ratio": width_ratio,
                "height_ratio": height_ratio,
                "is_full_page": is_full_page,
                "original_box": box
            }
            
            if self.debug_mode:
                if is_full_page:
                    print(f"    Box {box.to_tuple()}: rel_center=({rel_center_x:.2f},{rel_center_y:.2f}), "
                        f"FULL-PAGE (by margins: L={left_margin_pct:.1%} T={top_margin_pct:.1%} R={right_margin_pct:.1%} B={bottom_margin_pct:.1%})")
                else:
                    print(f"    Box {box.to_tuple()}: rel_center=({rel_center_x:.2f},{rel_center_y:.2f}), "
                        f"coverage={coverage_ratio:.1%}, not full-page")
            
            normalized_boxes.append(normalized)
        
        return normalized_boxes

    def analyze_page_content(self, image: Image.Image, bounding_boxes: List[BoundingBox], page_num: int = None) -> Dict[str, Any]:
        """Analyze content distribution using PIL dimensions and relative coordinates"""
        
        # Use PIL for accurate dimensions
        width, height = image.size
        aspect_ratio = width / height
        
        if self.debug_mode:
            print(f"    PIL Image dimensions: {width}×{height} (AR: {aspect_ratio:.2f})")
        
        # Normalize all bounding boxes to relative coordinates
        normalized_boxes = self.normalize_bounding_boxes(bounding_boxes, width, height)
        
        # Filter out full-page boxes for content analysis
        content_boxes = [box for box in normalized_boxes if not box["is_full_page"]]
        
        if self.debug_mode:
            print(f"    Total boxes: {len(normalized_boxes)}, Content boxes: {len(content_boxes)}")
        
        # If no content boxes (Florence2 not available or only full-page detections)
        if not content_boxes:
            if self.debug_mode:
                print(f"    No content boxes, using aspect ratio analysis (AR: {aspect_ratio:.2f})")
            
            # Fallback to aspect ratio analysis
            if aspect_ratio > 1.6:
                page_type = PageType.DOUBLE_PAGE_LEFT  # Will be corrected by alternating logic
                confidence = 0.7
                distribution = "wide_format"
            else:
                page_type = PageType.SINGLE_PAGE
                confidence = 0.6
                distribution = "normal_aspect"
            
            return {
                "page_type": page_type,
                "confidence": confidence,
                "content_distribution": distribution,
                "left_ratio": 0.0,
                "right_ratio": 0.0,
                "center_ratio": 1.0,
                "aspect_ratio": aspect_ratio,
                "normalized_boxes": normalized_boxes
            }
        
        # Analyze using margin and positioning instead of area distribution
        spread_analysis = self.analyze_spread_characteristics(normalized_boxes, page_num)

        # Store image for gutter detection
        spread_analysis["image"] = image

        # Extract results
        page_type = spread_analysis["page_type"]
        confidence = spread_analysis["confidence"] 
        distribution = spread_analysis["distribution"]

        # For backward compatibility, calculate ratios based on margin analysis
        if spread_analysis.get("left_margin", 0) > spread_analysis.get("right_margin", 0):
            left_ratio = 0.8  # Indicates left-biased (left page of spread)
            right_ratio = 0.2
            center_ratio = 0.0
        elif spread_analysis.get("right_margin", 0) > spread_analysis.get("left_margin", 0):
            left_ratio = 0.2  
            right_ratio = 0.8  # Indicates right-biased (right page of spread)
            center_ratio = 0.0
        else:
            left_ratio = 0.0
            right_ratio = 0.0
            center_ratio = 1.0  # Balanced/centered content
        
                # Apply pagination logic weighting
        pagination_confidence_boost = 0.0
        expected_type = None

        if page_num is not None:
            if page_num == 0:  # Page 1 (0-indexed)
                expected_type = PageType.SINGLE_PAGE
                pagination_confidence_boost = 0.2  # Boost single page confidence
            elif page_num % 2 == 1:  # Even page numbers (2, 4, 6...) - Left pages
                expected_type = PageType.DOUBLE_PAGE_LEFT
                if page_type == PageType.DOUBLE_PAGE_LEFT:
                    pagination_confidence_boost = 0.15
            elif page_num % 2 == 0:  # Odd page numbers (3, 5, 7...) - Right pages  
                expected_type = PageType.DOUBLE_PAGE_RIGHT
                if page_type == PageType.DOUBLE_PAGE_RIGHT:
                    pagination_confidence_boost = 0.15

        # Apply confidence boost
        if expected_type == page_type:
            confidence = min(0.95, confidence + pagination_confidence_boost)
            if self.debug_mode:
                print(f"    Pagination boost: Page {page_num + 1} expected {expected_type}, got {page_type}, confidence +{pagination_confidence_boost:.2f}")


        if self.debug_mode:
            print(f"    Relative areas - Left: {left_ratio:.2f}, Right: {right_ratio:.2f}, Center: {center_ratio:.2f}")
            print(f"    Result: {page_type}, confidence={confidence:.2f}, distribution={distribution}")
        
        return {
            "page_type": page_type,
            "confidence": confidence,
            "content_distribution": distribution,
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "center_ratio": center_ratio,
            "aspect_ratio": aspect_ratio,
            "normalized_boxes": normalized_boxes,
            "is_full_page_candidate": len([box for box in normalized_boxes if box["is_full_page"]]) > 0
        }
    
    
    def find_spread_pairs(self, page_analyses: List[Dict]) -> List[Tuple[int, int]]:
        """Find consecutive pages that should be joined as spreads using enhanced detection"""
        
        # First, fix alternating pattern for simple aspect-ratio detection
        self._fix_alternating_pattern(page_analyses)
        
        # Use new pagination-based detection
        enhanced_pairs = self.detect_pagination_spreads(page_analyses)
        
        # Convert to old format for compatibility, but keep high confidence pairs
        final_pairs = []
        for left_idx, right_idx, confidence, method in enhanced_pairs:
            if confidence >= 0.5:  # Only keep reasonably confident pairs
                final_pairs.append((left_idx, right_idx))
                
                if self.debug_mode:
                    print(f"    Found spread pair: pages {left_idx+1}-{right_idx+1} ({method}, conf: {confidence:.2f})")
        
        # Also check for center gutters in potential spreads
        for left_idx, right_idx in final_pairs[:]:  # Copy list to avoid modification during iteration
            left_analysis = page_analyses[left_idx]
            right_analysis = page_analyses[right_idx]
            
            # If we have the actual images, check for center gutters
            if "image" in left_analysis:
                gutter_result = self._detect_center_gutter(left_analysis["image"])
                if gutter_result["detected"]:
                    if self.debug_mode:
                        print(f"    Center gutter confirms spread: pages {left_idx+1}-{right_idx+1}")
        
        return final_pairs

    
    def _fix_alternating_pattern(self, page_analyses: List[Dict]):
        """Fix alternating left/right pattern for simple aspect-ratio-based detection"""
        
        # Find consecutive wide pages that were all marked as LEFT
        for i in range(len(page_analyses) - 1):
            current = page_analyses[i]
            next_page = page_analyses[i + 1]
            
            # If both are wide and marked as LEFT, alternate them
            if (current["page_type"] == PageType.DOUBLE_PAGE_LEFT and
                next_page["page_type"] == PageType.DOUBLE_PAGE_LEFT and
                current.get("aspect_ratio", 0) > 1.6 and
                next_page.get("aspect_ratio", 0) > 1.6 and
                'wide' in current.get("content_distribution", "") and
                'wide' in next_page.get("content_distribution", "")):
                
                # Make the second one RIGHT to create a potential pair
                page_analyses[i + 1]["page_type"] = PageType.DOUBLE_PAGE_RIGHT
                
                if self.debug_mode:
                    print(f"    Fixed alternating pattern: page {i+2} changed from LEFT to RIGHT")


# --- Simple Image Joiner ---

class SimpleJoiner:
    """Simple side-by-side image joining without complex stitching"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
    def join_images(self, left_image: Image.Image, right_image: Image.Image) -> Optional[Image.Image]:
        """Simple side-by-side joining with height matching"""
        try:
            # Match heights by scaling to the smaller height
            target_height = min(left_image.height, right_image.height)
            
            # Scale left image if needed
            if left_image.height != target_height:
                scale_factor = target_height / left_image.height
                new_width = int(left_image.width * scale_factor)
                left_image = left_image.resize((new_width, target_height), Image.Resampling.LANCZOS)
            
            # Scale right image if needed
            if right_image.height != target_height:
                scale_factor = target_height / right_image.height
                new_width = int(right_image.width * scale_factor)
                right_image = right_image.resize((new_width, target_height), Image.Resampling.LANCZOS)
            
            # Create combined image
            total_width = left_image.width + right_image.width
            combined = Image.new('RGB', (total_width, target_height))
            
            # Paste images side by side
            combined.paste(left_image, (0, 0))
            combined.paste(right_image, (left_image.width, 0))
            
            if self.debug_mode:
                print(f"    Joined {left_image.size} + {right_image.size} -> {combined.size}")
            
            return combined
            
        except Exception as e:
            if self.debug_mode:
                print(f"    Image joining failed: {e}")
            return None


# --- Main PDF Processor ---

class PDFProcessor:
    """Simplified PDF processor focused on Florence2 detection"""
    
    def __init__(self, output_dir: str, config: ProcessingConfig):
        self.output_dir = Path(output_dir)
        self.config = config
        self.debug_mode = getattr(config, 'debug_mode', False) 
        
        # Initialize components
        self.florence2_detector = None 
        if FLORENCE2_AVAILABLE:
            try:
                llm_models_dir = os.path.join(COMFYUI_BASE_PATH, "models", "LLM")
                self.florence2_detector = Florence2RectangleDetector(
                    model_name="CogFlorence-2.2-Large",
                    comfyui_base_path=COMFYUI_BASE_PATH,
                    min_box_area=config.min_image_area // 10  # Lower threshold for detection
                )
                if config.debug_mode:
                    print("✅ Florence2 detector initialized")
            except Exception as e:
                if config.debug_mode:
                    print(f"❌ Florence2 detector failed to initialize: {e}")
                self.florence2_detector = None
        
        self.spread_detector = SpreadDetector(config.debug_mode)
        self.color_manager = ColorProfileManager(config.debug_mode)
        self.joiner = SimpleJoiner(config.debug_mode)
        
        # Initialize modern image enhancer
        if ENHANCER_AVAILABLE and config.enable_enhancement:
            self.enhancer = ModernImageEnhancer(config.debug_mode)
            if config.debug_mode:
                print("✅ Modern image enhancer initialized")
        else:
            self.enhancer = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_image_with_proper_colorspace(self, page, xref: int) -> Optional[Image.Image]:
        """Extract image using proper colorspace handling for CMYK images"""
        
        try:
            # First, try extract_image to get raw data with profile
            img_data = page.parent.extract_image(xref)
            
            if img_data and 'image' in img_data:
                from PIL import Image, ImageCms
                import io
                
                # Open the raw image data
                raw_image = Image.open(io.BytesIO(img_data['image']))
                
                if self.debug_mode:
                    print(f"    📸 Raw image: mode={raw_image.mode}, size={raw_image.size}")
                    if hasattr(raw_image, 'info') and 'icc_profile' in raw_image.info:
                        print(f"    📸 Has embedded ICC profile: {len(raw_image.info['icc_profile'])} bytes")
                
                # Handle CMYK images properly
                if raw_image.mode == 'CMYK':
                    if self.debug_mode:
                        print(f"    🎨 Converting CMYK to RGB with profile preservation")
                    
                    try:
                        # Try ICC profile-aware conversion
                        if hasattr(raw_image, 'info') and 'icc_profile' in raw_image.info and COLORMGMT_AVAILABLE:
                            # Use PIL's color management for proper CMYK->RGB conversion
                            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(raw_image.info['icc_profile']))
                            output_profile = ImageCms.createProfile('sRGB')
                            
                            rgb_image = ImageCms.profileToProfile(
                                raw_image, 
                                input_profile, 
                                output_profile, 
                                renderingIntent=ImageCms.INTENT_PERCEPTUAL,
                                outputMode='RGB'
                            )
                            
                            if self.debug_mode:
                                print(f"    ✅ ICC profile conversion successful")
                            
                            return rgb_image
                        
                        else:
                            # Fallback: Manual CMYK conversion
                            if self.debug_mode:
                                print(f"    ⚠️ No ICC profile, using manual CMYK conversion")
                            
                            rgb_image = raw_image.convert('RGB')
                            return rgb_image
                            
                    except Exception as e:
                        if self.debug_mode:
                            print(f"    ❌ ICC conversion failed: {e}, trying manual conversion")
                        
                        # Ultimate fallback
                        rgb_image = raw_image.convert('RGB')
                        return rgb_image
                
                # For non-CMYK images, convert to RGB if needed
                elif raw_image.mode not in ['RGB', 'L']:
                    return raw_image.convert('RGB')
                else:
                    return raw_image
            
            else:
                if self.debug_mode:
                    print(f"    ⚠️ extract_image failed, no image data")
                return None
                
        except Exception as e:
            if self.debug_mode:
                print(f"    ❌ extract_image method failed: {e}")
            return None

    def detect_pdf_structure(self, page_images: List[Dict], image_width: int, image_height: int) -> str:
        """Detect if PDF has individual objects vs full-page renders using PIL dimensions"""
        
        if not page_images:
            return "unknown"
        
        if self.config.debug_mode:
            print(f"    Analyzing PDF structure: {len(page_images)} images")
        
        # If multiple images, likely individual objects
        if len(page_images) > 1:
            if self.config.debug_mode:
                print(f"      → Detected INDIVIDUAL_OBJECTS (multiple images)")
            return "individual_objects"
        
        # Single image - check aspect ratio and content
        img_data = page_images[0]
        img_w, img_h = img_data["image"].size
        aspect_ratio = img_w / img_h
        
        # Very wide images might be spreads or layouts
        if aspect_ratio > 1.8:
            if self.config.debug_mode:
                print(f"      → Detected FULL_PAGE_RENDER (wide aspect ratio: {aspect_ratio:.2f})")
            return "full_page_render"
        
        # Default to individual object for normal aspect ratios
        if self.config.debug_mode:
            print(f"      → Detected INDIVIDUAL_OBJECTS (normal aspect ratio: {aspect_ratio:.2f})")
        return "individual_objects"

    def analyze_florence2_coverage(self, bounding_boxes: List[BoundingBox], image_size: Tuple[int, int]) -> str:
        """Determine PDF type based on Florence2 detection patterns"""
        
        if not bounding_boxes:
            return "unknown"
        
        img_area = image_size[0] * image_size[1]
        
        if self.config.debug_mode:
            print(f"    Florence2 coverage analysis:")
        
        # Check for single box covering most of the image
        for i, box in enumerate(bounding_boxes):
            coverage = box.area / img_area
            if self.config.debug_mode:
                print(f"      Box {i+1}: {coverage:.1%} coverage")
            
            if coverage > 0.95:
                if self.config.debug_mode:
                    print(f"      → Florence2 says INDIVIDUAL_OBJECT (box covers {coverage:.1%})")
                return "individual_object"
        
        # Multiple boxes or partial coverage suggests page layout
        if self.config.debug_mode:
            print(f"      → Florence2 says FULL_PAGE_LAYOUT (partial coverage)")
        return "full_page_layout"

    def should_crop_image(self, img_data: Dict, bounding_boxes: List[BoundingBox]) -> bool:
        """Determine if image should be cropped based on normalized coordinate analysis"""
        
        image = img_data["image"]
        width, height = image.size
        
        # Analyze PDF structure using PIL dimensions
        pdf_structure = self.detect_pdf_structure([img_data], width, height)
        florence2_analysis = self.analyze_florence2_coverage(bounding_boxes, (width, height))
        
        # Decision logic
        if pdf_structure == "individual_objects" or florence2_analysis == "individual_object":
            if self.config.debug_mode:
                print(f"    🚫 SKIP CROPPING: {pdf_structure} + {florence2_analysis}")
            return False
        else:
            if self.config.debug_mode:
                print(f"    ✂️ APPLY CROPPING: {pdf_structure} + {florence2_analysis}")
            return True



    def process_pdf(self, pdf_path: str, extract_images: bool = True, 
                   extract_text: bool = True) -> ProcessingReport:
        """Main processing method with simplified flow"""
        
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        # Initialize report
        report = ProcessingReport(
            pdf_filename=pdf_path.name,
            total_pages=0,
            processing_time=0.0,
            images_extracted=0,
            images_joined=0,
            text_extracted_pages=0,
            output_directory=str(self.output_dir),
            extracted_images=[],
            extracted_text=[],
            joined_spreads=[]
        )
        
        if not PYMUPDF_AVAILABLE:
            print("❌ PyMuPDF not available, cannot process PDF")
            return report
        
        try:
            with fitz.open(str(pdf_path)) as doc:
                report.total_pages = doc.page_count
                
                if self.config.debug_mode:
                    print(f"📖 Processing {report.total_pages} pages from {pdf_path.name}")
                
                # Phase 1: Extract raw images and analyze for spreads
                raw_images = []
                page_analyses = []
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    # Extract text
                    if extract_text:
                        self._extract_text(page, page_num, report)
                    
                    # Extract and analyze images
                    if extract_images:
                        page_images = self._extract_page_images(page, page_num)
                        
                        for img_data in page_images:
                            # Detect content using Florence2
                            if self.florence2_detector:
                                raw_bounding_boxes, _ = self.florence2_detector.detect_rectangles(
                                    image=img_data["image"],
                                    text_input=self.config.florence2_prompt,
                                    return_mask=False,
                                    keep_model_loaded=True
                                )
                                
                                if self.config.debug_mode:
                                    print(f"  Page {page_num + 1} - Image size: {img_data['image'].size}")
                                    print(f"  Raw Florence2 boxes: {[(box.x1, box.y1, box.x2, box.y2) for box in raw_bounding_boxes]}")
                                    for i, box in enumerate(raw_bounding_boxes):
                                        coverage = box.area / (img_data['image'].width * img_data['image'].height)
                                        print(f"    Box {i+1}: {box.to_tuple()}, area={box.area}, coverage={coverage:.1%}")
                                
                                # Apply smart filtering using PIL dimensions
                                bounding_boxes = self.spread_detector.filter_florence2_boxes(
                                    img_data["image"], 
                                    raw_bounding_boxes, 
                                    self.config.min_image_area
                                )
                                
                                if self.config.debug_mode:
                                    print(f"  After filtering: {len(bounding_boxes)} boxes kept from {len(raw_bounding_boxes)} original")
                                    
                            else:
                                bounding_boxes = []

                            # Analyze for spread detection
                            analysis = self.spread_detector.analyze_page_content(
                                img_data["image"], bounding_boxes, page_num  # Add page_num parameter
                            )
                            
                            img_data["analysis"] = analysis
                            img_data["bounding_boxes"] = bounding_boxes
                            raw_images.append(img_data)
                            page_analyses.append(analysis)
                
                # Phase 2: Find and join spreads
                if self.config.join_spreads and len(raw_images) > 1:
                    raw_images = self._join_spreads(raw_images, page_analyses, report)
                
                # Phase 3: Crop and save all images
                for img_data in raw_images:
                    self._crop_and_save_image(img_data, report)
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"❌ Error processing PDF: {e}")
                import traceback
                traceback.print_exc()
        
        report.processing_time = time.time() - start_time
        
        if self.config.debug_mode:
            print(f"✅ Processing complete: {report.images_extracted} images, "
                  f"{report.images_joined} joined, {report.text_extracted_pages} text pages")
        
        return report
    
    def _extract_page_images(self, page, page_num: int) -> List[Dict]:
        """Extract images from a single page with proper CMYK handling"""
        page_images = []

        if self.debug_mode:
            print(f"  Processing page {page_num + 1} images...")

        image_list = page.get_images(full=True)
        processed_xrefs = set()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            if xref in processed_xrefs:
                continue
            
            try:
                # Try proper colorspace extraction first (handles CMYK properly)
                pil_image = self._extract_image_with_proper_colorspace(page, xref)
                original_colorspace = "Unknown"
                
                if pil_image is None:
                    # Fallback to pixmap method
                    if self.debug_mode:
                        print(f"    🔄 Falling back to pixmap extraction")
                    
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Size filtering
                    if (pix.width < self.config.min_image_size or 
                        pix.height < self.config.min_image_size or
                        pix.width * pix.height < self.config.min_image_area):
                        pix = None
                        continue
                    
                    # Handle colorspace conversion robustly
                    original_colorspace = str(pix.colorspace) if pix.colorspace else "Unknown"
                    
                    if self.debug_mode:
                        print(f"    Original pixmap: {pix.width}x{pix.height}, n={pix.n}, alpha={pix.alpha}, colorspace={original_colorspace}")
                    
                    # Simplified conversion for fallback
                    if pix.n == 4 and not pix.alpha:
                        if self.debug_mode:
                            print(f"    Converting n=4 format, forcing RGB conversion")
                        old_pix = pix
                        pix = fitz.Pixmap(fitz.csRGB, old_pix)
                        old_pix = None
                    elif pix.n == 4 and pix.alpha:
                        if self.debug_mode:
                            print(f"    Detected RGBA format, converting to RGB")
                        old_pix = pix
                        pix = fitz.Pixmap(fitz.csRGB, old_pix)
                        old_pix = None
                    elif pix.n == 2:
                        if self.debug_mode:
                            print(f"    Detected Grayscale+Alpha, converting to RGB")
                        old_pix = pix
                        pix = fitz.Pixmap(fitz.csRGB, old_pix)
                        old_pix = None
                    
                    # Remove any remaining alpha
                    if pix.alpha:
                        if self.debug_mode:
                            print(f"    Removing remaining alpha channel")
                        old_pix = pix
                        pix = fitz.Pixmap(pix, 0)
                        old_pix = None
                    
                    if self.debug_mode:
                        print(f"    Final pixmap: {pix.width}x{pix.height}, n={pix.n}, alpha={pix.alpha}")
                    
                    # Convert to PIL
                    if pix.n == 1:  # Grayscale
                        pil_image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                    elif pix.n == 3:  # RGB
                        pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    else:
                        if self.debug_mode:
                            print(f"    Unsupported pixel format: n={pix.n}")
                        pix = None
                        continue
                    
                    pix = None
                
                # Size filtering (for both extraction methods)
                if (pil_image.width < self.config.min_image_size or 
                    pil_image.height < self.config.min_image_size or
                    pil_image.width * pil_image.height < self.config.min_image_area):
                    continue
                
                # Ensure color profile
                pil_image = self.color_manager.ensure_color_profile(pil_image)
                
                page_images.append({
                    "image": pil_image,
                    "page_num": page_num,
                    "image_index": img_index,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "colorspace": original_colorspace,
                    "xref": xref,
                    "is_joined": False
                })
                
                processed_xrefs.add(xref)
                
                if self.debug_mode:
                    print(f"    ✅ Extracted image: {pil_image.size}, mode={pil_image.mode}")
                
                # Limit images per page
                if len(page_images) >= self.config.max_images_per_page:
                    break
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"    Error extracting image {img_index} from page {page_num + 1}: {e}")
                continue
        
        return page_images

    def _extract_text(self, page, page_num: int, report: ProcessingReport):
        """Extract text from page"""
        try:
            text_content = page.get_text()
            
            if text_content.strip():
                extracted_text = ExtractedText(
                    page_num=page_num,
                    text_content=text_content
                )
                
                report.extracted_text.append(extracted_text)
                report.text_extracted_pages += 1
                
                # Save text file
                text_file = self.output_dir / f"page_{page_num + 1:03d}_text.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                    
        except Exception as e:
            if self.config.debug_mode:
                print(f"    Error extracting text from page {page_num + 1}: {e}")
    
    def _join_spreads(self, raw_images: List[Dict], page_analyses: List[Dict], 
                    report: ProcessingReport) -> List[Dict]:
        """Join consecutive pages that form spreads with enhanced detection"""
        
        # Find spread pairs using enhanced detection
        enhanced_pairs = self.spread_detector.detect_pagination_spreads(page_analyses)
        
        if not enhanced_pairs:
            if self.debug_mode:
                print("No spread pairs detected")
            return raw_images
        
        # Process joins with confidence levels
        return self._process_spreads_with_confidence_levels(raw_images, page_analyses, report)

    def _process_spreads_with_confidence_levels(self, raw_images: List[Dict], page_analyses: List[Dict], 
                                            report: ProcessingReport) -> List[Dict]:
        """Process spreads with different confidence levels and save appropriately"""
        
        # Get enhanced spread pairs with confidence scores
        enhanced_pairs = self.spread_detector.detect_pagination_spreads(page_analyses)
        
        high_confidence_pairs = []
        medium_confidence_pairs = []
        
        for left_idx, right_idx, confidence, method in enhanced_pairs:
            if confidence >= 0.8:
                high_confidence_pairs.append((left_idx, right_idx, method))
            elif confidence >= 0.5:
                medium_confidence_pairs.append((left_idx, right_idx, method, confidence))
        
        joined_images = []
        processed_indices = set()
        
        # Always join high-confidence pairs
        for left_idx, right_idx, method in high_confidence_pairs:
            if left_idx in processed_indices or right_idx in processed_indices:
                continue
                
            left_img_data = raw_images[left_idx]
            right_img_data = raw_images[right_idx]
            
            joined_image = self.joiner.join_images(left_img_data["image"], right_img_data["image"])
            
            if joined_image:
                joined_data = self._create_joined_image_data(
                    joined_image, left_img_data, right_img_data, f"high_conf_{method}"
                )
                joined_images.append(joined_data)
                processed_indices.add(left_idx)
                processed_indices.add(right_idx)
                
                if self.debug_mode:
                    print(f"    ✅ High confidence join: pages {left_img_data['page_num']+1}-{right_img_data['page_num']+1}")
        
        # For medium-confidence pairs, note them but let user decide
        for left_idx, right_idx, method, confidence in medium_confidence_pairs:
            if left_idx in processed_indices or right_idx in processed_indices:
                continue
                
            left_img_data = raw_images[left_idx]
            right_img_data = raw_images[right_idx]
            
            joined_image = self.joiner.join_images(left_img_data["image"], right_img_data["image"])
            
            if joined_image:
                joined_data = self._create_joined_image_data(
                    joined_image, left_img_data, right_img_data, f"medium_conf_{method}_{confidence:.1f}"
                )
                joined_images.append(joined_data)
                processed_indices.add(left_idx)
                processed_indices.add(right_idx)
                
                print(f"📋 Medium confidence spread: pages {left_img_data['page_num']+1}-{right_img_data['page_num']+1} "
                    f"(confidence: {confidence:.2f}). Review and delete if incorrect.")
        
        # Add non-joined images
        for i, img_data in enumerate(raw_images):
            if i not in processed_indices:
                joined_images.append(img_data)
        
        return joined_images
        

    def _create_joined_image_data(self, joined_image: Image.Image, left_img_data: Dict, 
                                right_img_data: Dict, join_method: str) -> Dict:
        """Create joined image data structure"""
        return {
            "image": joined_image,
            "page_num": left_img_data["page_num"],  # Use left page number
            "image_index": 0,  # Joined images get index 0
            "width": joined_image.width,
            "height": joined_image.height,
            "colorspace": left_img_data["colorspace"],  # Use left colorspace
            "xref": f"{left_img_data['xref']}-{right_img_data['xref']}",
            "is_joined": True,  # Mark as joined
            "source_images": (left_img_data, right_img_data),  # Store source info
            "left_page": left_img_data["page_num"],  # Add missing left_page key
            "right_page": right_img_data["page_num"],  # Add missing right_page key
            "page_type": PageType.SINGLE_PAGE,  # Joined spreads are treated as single images
            "confidence": min(left_img_data.get("confidence", 0.7), right_img_data.get("confidence", 0.7)),
            "join_method": join_method,
            "analysis": {  # ADD THIS ANALYSIS BLOCK
                "page_type": PageType.SINGLE_PAGE,
                "confidence": min(left_img_data.get("confidence", 0.7), right_img_data.get("confidence", 0.7)),
                "segmentation_data": None,
                "image_index": 0,
                "image_page_num": left_img_data["page_num"],
                "image_size": (joined_image.width, joined_image.height)
            }
        }
        
    def _crop_and_save_image(self, img_data: Dict, report: ProcessingReport):
        """Crop using Florence2 detection and save image"""
        
        image = img_data["image"]
        page_num = img_data["page_num"]
        img_index = img_data["image_index"]
        is_joined = img_data.get("is_joined", False)
        
        # Re-detect content if this is a joined image
        if is_joined and self.florence2_detector:
            raw_bounding_boxes, _ = self.florence2_detector.detect_rectangles(
                image=image,
                text_input=self.config.florence2_prompt,
                return_mask=False,
                keep_model_loaded=True
            )
            # Apply smart filtering for joined images too
            bounding_boxes = self.spread_detector.filter_florence2_boxes(
                image, 
                raw_bounding_boxes, 
                self.config.min_image_area
            )
            img_data["bounding_boxes"] = bounding_boxes
        else:
            bounding_boxes = img_data.get("bounding_boxes", [])
        
        # Determine if we should crop based on PIL image analysis
        if bounding_boxes and self.should_crop_image(img_data, bounding_boxes):
            cropped_image = self._apply_florence2_crop(image, bounding_boxes)
            if self.config.debug_mode:
                print(f"    Applied cropping from {image.size} to {cropped_image.size}")
        else:
            cropped_image = image
            if self.config.debug_mode:
                if bounding_boxes:
                    print(f"    Skipped cropping - keeping original size {image.size}")
                else:
                    print(f"    No bounding boxes - keeping original size {image.size}")
        
        # Apply modern enhancement if available  
        if self.enhancer and self.config.enable_enhancement:
            if self.config.debug_mode:
                print(f"🎨 Enhancing Page {page_num + 1}, Image {img_index + 1}")
            
            enhanced_image = self.enhancer.enhance_image(
                cropped_image,
                profile=self.config.enhancement_profile,
                strength=self.config.enhancement_strength,
                subject_boxes=bounding_boxes
            )
        else:
            enhanced_image = cropped_image
        
        # Generate filename
        if is_joined:
            filename = f"page_{page_num + 1:03d}-{img_data['right_page'] + 1:03d}_joined.png"
        else:
            filename = f"page_{page_num + 1:03d}_img_{img_index + 1:02d}.png"
        
        # Save image
        filepath = self.output_dir / filename
        enhanced_image.save(filepath, "PNG")
        
        # Create record
        quality = self._assess_quality(enhanced_image)
        file_size = filepath.stat().st_size if filepath.exists() else 0
        
        extracted_img = ExtractedImage(
            filename=filename,
            page_num=page_num,
            image_index=img_index,
            width=enhanced_image.width,
            height=enhanced_image.height,
            file_size_bytes=file_size,
            quality_score=quality,
            page_type=img_data["analysis"]["page_type"],
            detection_confidence=img_data["analysis"]["confidence"],
            bbox=(0, 0, enhanced_image.width, enhanced_image.height),
            original_colorspace=img_data["colorspace"],
            extraction_method="Florence2_Enhanced" if self.enhancer else "Florence2_Simple"
        )
        
        report.extracted_images.append(extracted_img)
        report.images_extracted += 1
        
        # Record join if applicable
        if is_joined:
            joined_spread = JoinedSpread(
                filename=filename,
                left_page_num=img_data["left_page"],
                right_page_num=img_data["right_page"],
                left_image_filename="",
                right_image_filename="",
                combined_width=enhanced_image.width,
                combined_height=enhanced_image.height,
                confidence_score=img_data["analysis"]["confidence"]
            )
            
            report.joined_spreads.append(joined_spread)
            report.images_joined += 1
    
    def _apply_florence2_crop(self, image: Image.Image, bounding_boxes: List[BoundingBox]) -> Image.Image:
        """Apply cropping based on Florence2 detections"""
        
        if not bounding_boxes:
            return image
        
        # Find encompassing bounding box
        min_x = min(box.x1 for box in bounding_boxes)
        min_y = min(box.y1 for box in bounding_boxes)
        max_x = max(box.x2 for box in bounding_boxes)
        max_y = max(box.y2 for box in bounding_boxes)
        
        # Add margin
        margin = self.config.crop_margin
        width, height = image.size
        
        crop_x1 = max(0, min_x - margin)
        crop_y1 = max(0, min_y - margin)
        crop_x2 = min(width, max_x + margin)
        crop_y2 = min(height, max_y + margin)
        
        # Validate crop
        if crop_x2 > crop_x1 + 50 and crop_y2 > crop_y1 + 50:
            cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            if self.config.debug_mode:
                print(f"    Florence2 crop: ({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}) from {image.size} to {cropped.size}")
            
            return cropped
        
        return image
    
    def _assess_quality(self, image: Image.Image) -> ImageQuality:
        """Simple quality assessment"""
        try:
            # Basic quality heuristics
            area = image.width * image.height
            
            if area > 1000000:  # > 1MP
                return ImageQuality.EXCELLENT
            elif area > 500000:  # > 0.5MP
                return ImageQuality.GOOD
            elif area > 100000:  # > 0.1MP
                return ImageQuality.FAIR
            else:
                return ImageQuality.POOR
                
        except Exception:
            return ImageQuality.FAIR


# --- ComfyUI Node ---

class EnhancedPDFExtractorNode:
    """Simplified ComfyUI Node for PDF extraction using Florence2"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_paths": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter PDF file paths, one per line"
                }),
                "output_directory": ("STRING", {
                    "default": "pdf_output_v7",
                    "tooltip": "Directory to save extracted images and text"
                }),
                "extract_images": ("BOOLEAN", {"default": True}),
                "extract_text": ("BOOLEAN", {"default": True}),
                "join_spreads": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically detect and join double-page spreads"
                }),
                "enable_enhancement": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply modern image enhancement"
                }),
            },
            "optional": {
                "min_image_size": ("INT", {
                    "default": 200, "min": 50, "max": 1000, "step": 10,
                    "tooltip": "Minimum width/height for extracted images"
                }),
                "crop_margin": ("INT", {
                    "default": 5, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Margin around detected content when cropping"
                }),
                "enhancement_profile": (["Digital Magazine", "Scanned Photo", "Vintage/Compressed", "Minimal"], {
                    "default": "Digital Magazine",
                    "tooltip": "Enhancement profile for image quality improvement"
                }),
                "enhancement_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Overall enhancement strength multiplier"
                }),
                "florence2_prompt": ("STRING", {
                    "default": "rectangular images in page OR photograph OR illustration OR diagram",
                    "tooltip": "Prompt for Florence2 content detection"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print detailed processing information"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("summary", "output_path", "images_extracted", "text_pages", "processing_time")
    FUNCTION = "extract_pdf_content"
    CATEGORY = "Eric/PDF Tools"
    
    def __init__(self):
        self.output_base_dir = folder_paths.get_output_directory()
    
    def extract_pdf_content(self, pdf_paths, output_directory, extract_images, extract_text, 
                           join_spreads, enable_enhancement, min_image_size=200, crop_margin=5, 
                           enhancement_profile="Digital Magazine", enhancement_strength=1.0,
                           florence2_prompt="rectangular images in page OR photograph OR illustration OR diagram",
                           debug_mode=False):
        
        # Parse PDF paths
        pdf_list = [path.strip() for path in pdf_paths.strip().split('\n') if path.strip()]
        
        if not pdf_list:
            return "No PDF files provided", "", 0, 0, "0.0s"
        
        # Validate PDF files
        valid_pdfs = []
        for pdf_path in pdf_list:
            if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
                valid_pdfs.append(pdf_path)
            else:
                print(f"⚠️ Invalid PDF: {pdf_path}")
        
        if not valid_pdfs:
            return "No valid PDF files found", "", 0, 0, "0.0s"
        
        # Setup output directory
        if os.path.isabs(output_directory):
            output_path = output_directory
        else:
            output_path = os.path.join(self.output_base_dir, output_directory)
        
        # Create configuration
        config = ProcessingConfig(
            min_image_size=min_image_size,
            crop_margin=crop_margin,
            join_spreads=join_spreads,
            florence2_prompt=florence2_prompt,
            debug_mode=debug_mode,
            enable_enhancement=enable_enhancement,
            enhancement_profile=enhancement_profile,
            enhancement_strength=enhancement_strength
        )
        
        # Process PDFs
        all_reports = []
        total_images = 0
        total_text_pages = 0
        total_time = 0.0
        
        for pdf_path in valid_pdfs:
            if debug_mode:
                print(f"\n📖 Processing: {os.path.basename(pdf_path)}")
            
            try:
                # Create processor for this PDF
                pdf_output_dir = os.path.join(output_path, Path(pdf_path).stem)
                processor = PDFProcessor(pdf_output_dir, config)
                
                # Process PDF
                report = processor.process_pdf(pdf_path, extract_images, extract_text)
                
                all_reports.append(report)
                total_images += report.images_extracted
                total_text_pages += report.text_extracted_pages
                total_time += report.processing_time
                
                # Save report
                report_path = os.path.join(pdf_output_dir, "processing_report.json")
                with open(report_path, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
                
                if debug_mode:
                    print(f"✅ Completed: {report.images_extracted} images, "
                          f"{report.images_joined} joined, {report.text_extracted_pages} text pages")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_path}: {e}")
                if debug_mode:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Create summary
        summary_lines = [
            "=== PDF EXTRACTION SUMMARY v0.7.0 ===",
            f"PDFs Processed: {len(all_reports)}/{len(valid_pdfs)}",
            f"Total Images Extracted: {total_images}",
            f"Total Spreads Joined: {sum(r.images_joined for r in all_reports)}",
            f"Total Text Pages: {total_text_pages}",
            f"Total Processing Time: {total_time:.1f}s",
            "",
            "FLORENCE2 DETECTION:",
            f"  Available: {'✅' if FLORENCE2_AVAILABLE else '❌'}",
            f"  Prompt: '{florence2_prompt}'",
            "",
            "IMAGE ENHANCEMENT:",
            f"  Available: {'✅' if ENHANCER_AVAILABLE else '❌'}",
            f"  Profile: {enhancement_profile}",
            f"  Strength: {enhancement_strength}",
            "",
            "INDIVIDUAL RESULTS:"
        ]
        
        for report in all_reports:
            summary_lines.append(
                f"  • {report.pdf_filename}: {report.images_extracted} images "
                f"({report.images_joined} joined), {report.text_extracted_pages} text pages"
            )
        
        summary = "\n".join(summary_lines)
        
        # Save combined report
        if all_reports:
            combined_report_path = os.path.join(output_path, "combined_report.json")
            combined_data = {
                "version": "0.7.0",
                "total_pdfs": len(valid_pdfs),
                "successful_pdfs": len(all_reports),
                "total_images": total_images,
                "total_text_pages": total_text_pages,
                "total_time": total_time,
                "florence2_available": FLORENCE2_AVAILABLE,
                "reports": [asdict(report) for report in all_reports]
            }
            
            with open(combined_report_path, 'w') as f:
                json.dump(combined_data, f, indent=2, default=str)
        
        print(f"\n🎉 Processing complete! Results saved to {output_path}")
        
        return (summary, output_path, total_images, total_text_pages, f"{total_time:.1f}s")


# Node registration
NODE_CLASS_MAPPINGS = {
    "Eric_PDF_Extractor_Enhanced_V07": EnhancedPDFExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Eric_PDF_Extractor_Enhanced_V07": "Eric PDF Extractor Enhanced v0.7.0"
}