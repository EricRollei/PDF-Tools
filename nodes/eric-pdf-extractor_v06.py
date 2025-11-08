"""
ComfyUI Node: Enhanced PDF Image and Text Extractor v0.6.0
Description: 
Advanced PDF processing with Segment Anything and Grounding Dino models, intelligent border analysis,
constrained OpenCV stitching, and GPU-accelerated Kornia enhancement pipeline.

Key Improvements:
- Segment Anything integration for intelligent cropping
- Enhanced border pattern analysis for spread identification
- Constrained OpenCV stitching for magazine/book layouts
- GPU-accelerated Kornia enhancement pipeline optimized for SD training
- Smart cropping with multiple detection methods

Credits to Storyicon on GitHub for the original Comfy-segment-anything nodes https://github.com/storyicon/comfyui_segment_anything/
for which I have addapted the code to work with the pdf extractor node.

Author: Eric Hiss (GitHub: EricRollei)
Enhanced by: Claude Sonnet 4 AI Assistant, Gemini 2.5 Pro AI Assistant
Version: 0.4.0
Date: [June 2025]
License: Dual License (Non-Commercial and Commercial Use)
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
import folder_paths
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from local_groundingdino.datasets import transforms as T
import gc
import weakref
from contextlib import contextmanager
import functools
import traceback
from typing import Callable, Any
import glob
import hashlib

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available - memory monitoring disabled")


# PDF processing imports
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available - using PyPDF2 fallback mode")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - image processing features limited")

try:
    import kornia
    import kornia.enhance
    import kornia.filters
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Kornia not available - GPU acceleration disabled")

# Get the directory of the current script (e.g., .../PDF-tools/nodes/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (e.g., .../PDF-tools/)
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path: # Optional: prevent duplicates
    sys.path.append(parent_dir)


# --- Imports for GroundingDINO --- 
try:
    from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig  # FIXED: Use exact same name
    from local_groundingdino.models import build_model as local_groundingdino_build_model    # FIXED: Use exact same name
    from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict  # FIXED: Use exact same name
    GROUNDINGDINO_LIB_AVAILABLE = True
    print("INFO: PDF-tools node using bundled local_groundingdino library.")
except ImportError:
    print("WARNING: Bundled local_groundingdino library components not found in PDF-tools. Ensure it's copied correctly.")
    print("         GroundingDINOModel will not function.")
    GROUNDINGDINO_LIB_AVAILABLE = False


try:
    # Import the working Florence2 detector
    from PDF_tools import Florence2RectangleDetector, BoundingBox
    FLORENCE2_AVAILABLE = True
    print("INFO: Florence2 detector available for PDF processing")
except ImportError:
    FLORENCE2_AVAILABLE = False
    print("WARNING: Florence2 detector not available - falling back to other methods")
    class Florence2RectangleDetector: pass
    class BoundingBox: pass



# Only define dummy functions if the import actually failed
if not GROUNDINGDINO_LIB_AVAILABLE:
    class local_groundingdino_SLConfig:  # FIXED: Use exact same name
        @staticmethod
        def fromfile(config_file): 
            return None

    def local_groundingdino_build_model(args):  # FIXED: Use exact same name
        return None

    def local_groundingdino_clean_state_dict(state_dict):  # FIXED: Use exact same name
        return state_dict

def get_bert_base_uncased_model_path_helper():
    """Simple BERT path helper based on working segment_anything node"""
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        if hasattr(print, '__call__'):  # Simple check to avoid issues
            print('GroundingDINOModel: Using local bert-base-uncased from ComfyUI models directory.')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def ensure_bert_model_available():
    """Ensure BERT model is available for GroundingDINO"""
    bert_path = get_bert_base_uncased_model_path_helper()
    bert_dir = Path(bert_path) if bert_path != 'bert-base-uncased' else None
    
    if bert_dir and bert_dir.exists():
        print(f"INFO: BERT model found at {bert_dir}")
        return True
    
    print("INFO: BERT model not found locally. GroundingDINO will download it automatically on first use.")
    print("INFO: This may take a few minutes and requires internet connection.")
    return False

# Call this during initialization
if GROUNDINGDINO_LIB_AVAILABLE:
    ensure_bert_model_available()


try:
    from sam_hq.predictor import SamPredictorHQ
    from sam_hq.build_sam_hq import sam_model_registry  # FIXED: Use sam_model_registry, not sam_hq_model_registry
    SEGMENT_ANYTHING_AVAILABLE = True
    print("INFO: PDF-tools node using bundled sam_hq library for segmentation.")
except ImportError:
    SEGMENT_ANYTHING_AVAILABLE = False
    print("WARNING: Bundled sam_hq library not found in PDF-tools. Ensure it's copied correctly. Segmentation-based cropping will be disabled.")
    # Define dummy classes if library is not available to prevent NameErrors later
    class SamPredictorHQ: pass
    sam_model_registry = {}  # FIXED: Use sam_model_registry




# --- GroundingDINO Configuration Notes --- 
# The DINO configuration and model weights are selected via node inputs 
# ("dino_config_name", "sam_model_name") and resolved by SegmentationAnalyzer.
# Default DINO model directory is expected at: ComfyUI/models/grounding-dino/
# Default SAM model directory is expected at: ComfyUI/models/sams/


class ImageQuality(Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good" 
    FAIR = "Fair"
    POOR = "Poor"
    UNUSABLE = "Unusable"

class PageType(Enum):
    SINGLE_PAGE = "single_page"
    DOUBLE_PAGE_LEFT = "double_page_left"
    DOUBLE_PAGE_RIGHT = "double_page_right"
    UNCERTAIN = "uncertain"

@dataclass
class ExtractedImage:
    """Information about an extracted image""" 
    filename: str
    page_num: int
    image_index: int
    width: int
    height: int
    file_size_bytes: int
    quality_score: ImageQuality
    page_type: PageType
    border_confidence: float
    is_multi_page_candidate: bool
    bbox: Tuple[float, float, float, float]
    original_colorspace: str
    extraction_method: str

@dataclass
class ExtractedText:
    """Information about extracted text"""
    page_num: int
    text_content: str
    bbox: Tuple[float, float, float, float] = None
    font_info: str = ""
    is_ocr_layer: bool = False

@dataclass
class JoinedImage:
    """Information about a joined double-page image"""
    filename: str
    left_page_num: int
    right_page_num: int
    left_image_filename: str
    right_image_filename: str
    combined_width: int
    combined_height: int
    confidence_score: float
    join_method: str

@dataclass
class ProcessingReport:
    """Complete processing report"""
    pdf_filename: str
    total_pages: int
    processing_time: float
    total_images_found: int
    images_extracted: int
    images_filtered_out: int
    images_enhanced: int
    images_joined: int
    text_layers_found: int
    text_extracted_pages: int
    quality_breakdown: Dict[str, int]
    page_type_breakdown: Dict[str, int]
    multi_page_candidates: List[str]
    output_directory: str
    extracted_images: List[ExtractedImage]
    extracted_text: List[ExtractedText]
    joined_images: List[JoinedImage]


def get_individual_bounding_boxes_from_mask(sam_mask_tensor: torch.Tensor, min_area_threshold: int = 100) -> list:
    """
    Extracts individual bounding boxes for multiple disconnected regions in a SAM mask.
    """
    if not CV2_AVAILABLE:
        print("OpenCV is not available. Cannot extract multiple bounding boxes from mask.")
        return []
    if sam_mask_tensor is None:
        print("Input mask tensor is None.")
        return []

    # --- Convert mask to a 2D uint8 NumPy array (0 for background, 255 for foreground) ---
    if sam_mask_tensor.ndim == 3:
        if sam_mask_tensor.shape[0] == 1: # Batch dimension CHW
            mask_tensor_squeezed = sam_mask_tensor.squeeze(0)
        elif sam_mask_tensor.shape[-1] == 1: # Channel dimension HWC
            mask_tensor_squeezed = sam_mask_tensor.squeeze(-1)
        else: # Assuming HWC if not CHW, take first channel if multiple (e.g. RGB mask)
            print(f"Warning: Mask tensor has 3 channels ({sam_mask_tensor.shape}). Taking first channel.")
            mask_tensor_squeezed = sam_mask_tensor[..., 0]
    elif sam_mask_tensor.ndim == 2: # HW
        mask_tensor_squeezed = sam_mask_tensor
    else:
        print(f"Error: Unsupported mask tensor dimensions: {sam_mask_tensor.shape}")
        return []

    mask_np = mask_tensor_squeezed.cpu().numpy()

    # Ensure correct data type and scale (0-255 for cv2.findContours)
    if mask_np.dtype != np.uint8:
        if mask_np.max() <= 1.0 and mask_np.min() >= 0.0 and mask_np.dtype != bool:
            mask_np = (mask_np * 255).astype(np.uint8)
        else: # If it's some other type, try to convert directly
            mask_np = mask_np.astype(np.uint8)

    # Ensure it's binary 0 or 255 after conversion
    if np.any((mask_np != 0) & (mask_np != 255)):
        mask_np = np.where(mask_np > 0, np.uint8(255), np.uint8(0))

    # --- Find contours ---
    contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        # --- Get bounding rectangle for each contour ---
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= min_area_threshold:
            bounding_boxes.append({"x": x, "y": y, "width": w, "height": h, "area": area})

    # COMPREHENSIVE FIX: Multiple sorting approaches with error handling
    if bounding_boxes:
        try:
            # Method 1: Try normal sorting with lambda
            bounding_boxes.sort(key=lambda b: b['area'], reverse=True)
            print(f"Successfully sorted {len(bounding_boxes)} bounding boxes by area")
        except TypeError as e:
            print(f"Lambda sorting failed: {e}")
            try:
                # Method 2: Manual key extraction with error checking
                def safe_get_area(box):
                    if isinstance(box, dict) and 'area' in box:
                        return box['area']
                    print(f"Warning: Invalid box format: {box}")
                    return 0
                
                bounding_boxes.sort(key=safe_get_area, reverse=True)
                print(f"Successfully sorted {len(bounding_boxes)} bounding boxes using safe method")
            except Exception as e2:
                print(f"Safe sorting also failed: {e2}")
                try:
                    # Method 3: Manual bubble sort as ultimate fallback
                    n = len(bounding_boxes)
                    for i in range(n):
                        for j in range(0, n - i - 1):
                            try:
                                area1 = bounding_boxes[j].get('area', 0) if isinstance(bounding_boxes[j], dict) else 0
                                area2 = bounding_boxes[j + 1].get('area', 0) if isinstance(bounding_boxes[j + 1], dict) else 0
                                if area1 < area2:  # Sort descending
                                    bounding_boxes[j], bounding_boxes[j + 1] = bounding_boxes[j + 1], bounding_boxes[j]
                            except Exception:
                                continue
                    print(f"Successfully sorted {len(bounding_boxes)} bounding boxes using manual sort")
                except Exception as e3:
                    print(f"All sorting methods failed: {e3}. Returning unsorted boxes.")
                    # Just return what we have - unsorted is better than crashing
        except Exception as e:
            print(f"Unexpected error during sorting: {e}. Returning unsorted boxes.")

    return bounding_boxes


@dataclass
class ProcessingConfig:
    """Centralized configuration for processing parameters"""
    # Image filtering
    min_image_size: int = 200
    max_image_width: int = 6000
    max_image_height: int = 6000
    min_image_area: int = 40000
    max_images_per_page: int = 10
    
    # Analysis thresholds
    border_confidence_threshold: float = 0.7
    geometric_confidence_threshold: float = 0.5
    uncertainty_threshold: float = 0.15
    
    # Processing settings
    enhancement_strength: float = 1.0
    crop_margin: int = 5 
    
    # Memory management - FIXED: Better defaults
    memory_optimization: bool = True
    batch_size: int = 15  # Increased from 10
    max_gpu_memory_mb: int = 8192  # Increased from 4096
    
    # Model settings
    use_border_analysis: bool = True
    dino_config_name: Optional[str] = None 
    sam_model_name: Optional[str] = None   
    
    # Segmentation settings
    enable_image_segmentation: bool = True # New
    image_segmentation_prompt: str = "photograph OR main image OR illustration OR diagram OR chart OR map" # Renamed and expanded
    enable_text_segmentation: bool = False # New, default to False for now
    text_segmentation_prompt: str = "text block OR paragraph OR caption OR headline OR title" # New
    segmentation_box_threshold: float = 0.3 # New
    segmentation_text_threshold: float = 0.25 # New
    
    # Output settings
    save_debug_images: bool = True
    compression_quality: int = 95
    



    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration parameters"""
        errors = []
        
        # Size validation
        if not (50 <= self.min_image_size <= 1000):
            errors.append("min_image_size must be between 50 and 1000")
        
        if self.max_image_width <= self.min_image_size:
            errors.append("max_image_width must be greater than min_image_size")
            
        if self.max_image_height <= self.min_image_size:
            errors.append("max_image_height must be greater than min_image_size")
        
        # Threshold validation
        for threshold_name, threshold_value in [
            ("border_confidence_threshold", self.border_confidence_threshold),
            ("geometric_confidence_threshold", self.geometric_confidence_threshold),
            ("uncertainty_threshold", self.uncertainty_threshold)
        ]:
            if not (0.0 <= threshold_value <= 1.0):
                errors.append(f"{threshold_name} must be between 0.0 and 1.0")
        
        # Enhancement validation
        if not (0.0 <= self.enhancement_strength <= 3.0):
            errors.append("enhancement_strength must be between 0.0 and 3.0")
        
        # Memory validation
        if not (1 <= self.batch_size <= 50):
            errors.append("batch_size must be between 1 and 50")
            
        if not (512 <= self.max_gpu_memory_mb <= 32768):
            errors.append("max_gpu_memory_mb must be between 512 and 32768")
        
        # Quality validation
        if not (10 <= self.compression_quality <= 100):
            errors.append("compression_quality must be between 10 and 100")
        
        return len(errors) == 0, errors

    def adjust_for_system(self):
        """Auto-adjust configuration based on system capabilities"""
        # Adjust for available GPU memory - FIXED: Better thresholds
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            if gpu_memory_mb >= 12000:  # 12GB or more VRAM
                self.batch_size = min(25, self.batch_size * 2)
                self.max_images_per_page = min(15, self.max_images_per_page + 5)
                if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                    print(f"High VRAM system ({gpu_memory_mb}MB), increased batch size to {self.batch_size}")
            elif gpu_memory_mb >= 8000:  # 8-12GB VRAM
                self.batch_size = min(20, int(self.batch_size * 1.5))
                if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                    print(f"Mid VRAM system ({gpu_memory_mb}MB), batch size: {self.batch_size}")
            elif gpu_memory_mb < 6000:  # Less than 6GB VRAM
                self.batch_size = max(5, self.batch_size // 2)
                self.max_images_per_page = min(self.max_images_per_page, 8)
                if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                    print(f"Low VRAM system ({gpu_memory_mb}MB), reduced batch size to {self.batch_size}")
        else:
            # CPU only - moderate batch size
            self.batch_size = min(10, self.batch_size)
            self.memory_optimization = True
        
        # Adjust for system RAM - FIXED: Better logic
        if PSUTIL_AVAILABLE:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            if total_ram_gb >= 32:  # 32GB+ RAM
                self.batch_size = min(30, self.batch_size * 2)
            elif total_ram_gb < 8:   # Less than 8GB RAM
                self.batch_size = max(3, self.batch_size // 2)
                self.memory_optimization = True

    def get_analysis_weights(self) -> Dict[str, float]:
        """Get analysis method weights based on availability"""
        weights = {}
        total_weight = 0.0
        
        
        if self.use_border_analysis:
            weights["border"] = 0.5
            total_weight += 0.5
        
        weights["geometric"] = 0.1
        total_weight += 0.1
        
        # Normalize weights
        if total_weight > 0:
            for key in weights:
                weights[key] = weights[key] / total_weight
        
        return weights
    


    @classmethod
    def from_node_inputs(cls, **kwargs):
        """Create config from ComfyUI node inputs"""
        config = cls()
        
        # Map node parameters to config
        config.min_image_size = kwargs.get("min_image_size", config.min_image_size) 
        config.border_confidence_threshold = kwargs.get("border_confidence_threshold", config.border_confidence_threshold)
        config.enhancement_strength = kwargs.get("enhancement_strength", config.enhancement_strength)
        config.crop_margin = kwargs.get("crop_margin", config.crop_margin) 

        # Model selections
        config.dino_config_name = kwargs.get("dino_config_name", None)
        config.sam_model_name = kwargs.get("sam_model_name", None)
        
        # Segmentation settings from node inputs
        config.enable_image_segmentation = kwargs.get("enable_image_segmentation", config.enable_image_segmentation)
        config.image_segmentation_prompt = kwargs.get("image_segmentation_prompt", config.image_segmentation_prompt)
        config.enable_text_segmentation = kwargs.get("enable_text_segmentation", config.enable_text_segmentation)
        config.text_segmentation_prompt = kwargs.get("text_segmentation_prompt", config.text_segmentation_prompt)
        config.segmentation_box_threshold = kwargs.get("segmentation_box_threshold", config.segmentation_box_threshold)
        config.segmentation_text_threshold = kwargs.get("segmentation_text_threshold", config.segmentation_text_threshold)
        
        # Adjust for system
        config.adjust_for_system()
        
        # Validate the configuration
        is_valid, errors = config.validate()
        if not is_valid:
            # In a ComfyUI node, raising an exception here might be too disruptive.
            # Logging errors is better. The node's execution will proceed with potentially invalid config.
            print("Warning: ProcessingConfig validation failed:")
            for error in errors:
                print(f"  - {error}")

        return config


class ProcessingError(Exception):
    """Custom exception for processing errors with context"""
    def __init__(self, message: str, original_error: Exception = None, context: Dict = None):
        super().__init__(message)
        self.original_error = original_error
        self.context = context or {}

class ErrorRecoveryManager:
    """Manages error recovery and retry logic"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.error_counts = {}
        self.max_errors_per_type = 5
    
    def with_retry(self, max_retries: int = 2, backoff_factor: float = 0.1, 
                   exceptions: Tuple = (Exception,)):
        """Decorator for retry logic with exponential backoff"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_retries:
                            # Final attempt failed
                            error_type = type(e).__name__
                            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                            
                            if self.debug_mode:
                                print(f"❌ Function {func.__name__} failed after {max_retries + 1} attempts")
                                print(f"   Final error: {e}")
                                traceback.print_exc()
                            
                            raise ProcessingError(
                                f"Failed after {max_retries + 1} attempts: {e}",
                                original_error=e,
                                context={"function": func.__name__, "attempts": max_retries + 1}
                            )
                        else:
                            # Wait before retry
                            wait_time = backoff_factor * (2 ** attempt)
                            if self.debug_mode:
                                print(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {e}")
                                print(f"   Retrying in {wait_time:.2f}s...")
                            
                            time.sleep(wait_time)
                
                return None
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, *args, fallback_result=None, 
                    context: str = "", **kwargs):
        """Safely execute a function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type = type(e).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            if self.debug_mode:
                print(f"⚠️ Safe execution failed in {context}: {e}")
                if self.error_counts[error_type] <= 3:  # Only show traceback for first few
                    traceback.print_exc()
            
            return fallback_result
    
    def should_continue_processing(self) -> bool:
        """Check if processing should continue based on error history"""
        total_errors = sum(self.error_counts.values())
        
        # Stop if too many total errors
        if total_errors > 20:
            print(f"❌ Stopping processing due to too many errors ({total_errors})")
            return False
        
        # Stop if too many of the same error type
        for error_type, count in self.error_counts.items():
            if count > self.max_errors_per_type:
                print(f"❌ Stopping processing due to repeated {error_type} errors ({count})")
                return False
        
        return True
    
    def get_error_summary(self) -> str:
        """Get a summary of all errors encountered"""
        if not self.error_counts:
            return "No errors encountered"
        
        total = sum(self.error_counts.values())
        summary = [f"Total errors: {total}"]
        
        for error_type, count in sorted(self.error_counts.items()):
            summary.append(f"  {error_type}: {count}")
        
        return "\n".join(summary)

class EnhancedBorderAnalyzer:
    """Advanced border pattern analysis for double-page spread detection"""
    
    def __init__(self):
        self.debug_mode = False
    
    def analyze_borders(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze border patterns to determine page type"""
        if not CV2_AVAILABLE:
            return {"page_type": PageType.UNCERTAIN, "confidence": 0.0, "border_analysis": {}}
        
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Analyze each border
        border_analysis = {
            'top': self._analyze_border_edge(gray, 'top'),
            'bottom': self._analyze_border_edge(gray, 'bottom'),
            'left': self._analyze_border_edge(gray, 'left'),
            'right': self._analyze_border_edge(gray, 'right')
        }
        
        # Determine page type based on border patterns
        page_type, confidence = self._classify_from_borders(border_analysis)
        
        if self.debug_mode:
            print(f"Border analysis: {border_analysis}")
            print(f"Classified as: {page_type} (confidence: {confidence:.3f})")
        
        return {
            "page_type": page_type,
            "confidence": confidence,
            "border_analysis": border_analysis
        }
    
    def _analyze_border_edge(self, gray: np.ndarray, edge: str) -> Dict[str, float]:
        """Analyze a specific edge for border characteristics"""
        h, w = gray.shape
        border_width = min(50, min(h, w) // 20)  # Check more of the edge
        
        if edge == 'top':
            region = gray[:border_width, :]
        elif edge == 'bottom':
            region = gray[-border_width:, :]
        elif edge == 'left':
            region = gray[:, :border_width]
        elif edge == 'right':
            region = gray[:, -border_width:]
        else:
            return {"uniformity": 0.0, "brightness": 0.0, "has_border": False}
        
        # Calculate border characteristics
        uniformity = 1.0 - (np.std(region) / 255.0)
        brightness = np.mean(region) / 255.0
        
        # More lenient edge detection
        # Check if the inner edge of the border region has a strong transition
        if edge in ['left', 'right']:
            inner_strip = region[:, -10:] if edge == 'left' else region[:, :10]
            outer_strip = region[:, :10] if edge == 'left' else region[:, -10:]
        else:
            inner_strip = region[-10:, :] if edge == 'top' else region[:10, :]
            outer_strip = region[:10, :] if edge == 'top' else region[-10:, :]
        
        edge_gradient = abs(np.mean(inner_strip) - np.mean(outer_strip))
        
        # More lenient border detection
        # A border exists if: somewhat uniform AND (bright OR dark) AND some edge transition
        has_border = uniformity > 0.6 and (brightness > 0.85 or brightness < 0.15) and edge_gradient > 5
        
        if self.debug_mode:
            print(f"      {edge}: uniformity={uniformity:.2f}, brightness={brightness:.2f}, gradient={edge_gradient:.1f}, has_border={has_border}")
        
        return {
            "uniformity": uniformity,
            "brightness": brightness,
            "has_border": has_border,
            "gradient": edge_gradient
        }
        

    def _classify_from_borders(self, border_analysis: Dict) -> Tuple[PageType, float]:
        """Classify page type based on border analysis with improved logic"""
        borders = {side: analysis["has_border"] for side, analysis in border_analysis.items()}
        border_count = sum(borders.values())
        
        if self.debug_mode:
            print(f"    Border detection: {borders}, total: {border_count}")
        
        uniformity_scores = {side: analysis["uniformity"] for side, analysis in border_analysis.items()}
        avg_uniformity = np.mean(list(uniformity_scores.values()))
        
        # Priority 1: Strongest evidence for double pages (specific 3 borders present, one side missing)
        if borders["top"] and borders["bottom"] and borders["left"] and not borders["right"]:
            return PageType.DOUBLE_PAGE_LEFT, 0.85
        if borders["top"] and borders["bottom"] and borders["right"] and not borders["left"]:
            return PageType.DOUBLE_PAGE_RIGHT, 0.85
        
        # Priority 2: Asymmetric uniformity (content vs. border edge)
        left_right_asymmetry = abs(uniformity_scores["left"] - uniformity_scores["right"])
        if left_right_asymmetry > 0.22:  # Adjusted threshold
            if uniformity_scores["left"] > uniformity_scores["right"]:
                return PageType.DOUBLE_PAGE_LEFT, 0.70 # Adjusted confidence
            else:
                return PageType.DOUBLE_PAGE_RIGHT, 0.70 # Adjusted confidence
        
        # Priority 3: Three borders detected - specifically missing a side border (and at least one top/bottom is present)
        if border_count == 3:
            if not borders["right"] and borders["left"] and (borders["top"] or borders["bottom"]):
                return PageType.DOUBLE_PAGE_LEFT, 0.65 # Adjusted confidence
            elif not borders["left"] and borders["right"] and (borders["top"] or borders["bottom"]):
                return PageType.DOUBLE_PAGE_RIGHT, 0.65 # Adjusted confidence
            # If 3 borders but not fitting above (e.g., top is missing), it might be a single page.
            # This will be handled by single page rules below.

        # Priority 4: Strongest evidence for single page
        if border_count == 4:
            return PageType.SINGLE_PAGE, 0.80
        # Very uniform page (e.g., text page with good margins, even if some border detections failed)
        if avg_uniformity > 0.8: # Increased threshold for high confidence single page
             return PageType.SINGLE_PAGE, 0.75

        # Priority 5: Moderate evidence for single page (e.g. 3 borders not fitting double page pattern, or fairly uniform)
        if border_count == 3 and avg_uniformity > 0.65: # e.g. missing top/bottom border but otherwise uniform
            return PageType.SINGLE_PAGE, 0.60
        
        # Priority 6: Weaker evidence for single page
        if border_count >= 2 and avg_uniformity > 0.55: # At least two borders and moderate uniformity
            return PageType.SINGLE_PAGE, 0.45
        if border_count >= 1: # Fallback if at least one border is detected
            return PageType.SINGLE_PAGE, 0.30
        
        # Priority 7: Fallback based on asymmetry if no borders detected but asymmetry exists
        if border_count == 0 and left_right_asymmetry > 0.15:
            if uniformity_scores["left"] > uniformity_scores["right"]:
                return PageType.DOUBLE_PAGE_LEFT, 0.25
            else:
                return PageType.DOUBLE_PAGE_RIGHT, 0.25
                
        return PageType.UNCERTAIN, 0.2
        



class ConstrainedStitcher:
    """Constrained OpenCV stitching optimized for magazine/book layouts"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None): # Modified
        self.debug_mode = False
        self.config = config if config else ProcessingConfig() # Store config, or a default
        if CV2_AVAILABLE:
            self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
            self._configure_for_magazines()
    
    def _configure_for_magazines(self):
        """Configure stitcher parameters for magazine/book layouts"""
        if CV2_AVAILABLE:
            # Disable wave correction for flat book pages
            self.stitcher.setWaveCorrection(False)
            # Lower resolution for faster processing
            self.stitcher.setSeamEstimationResol(0.1)
            # Configure for horizontal stitching
            self.stitcher.setPanoConfidenceThresh(0.3)  # Lower threshold for magazine pages
    
    def stitch_magazine_spread(self, left_image: Image.Image, right_image: Image.Image) -> Optional[Image.Image]:
        """Stitch two images using constrained OpenCV stitching"""
        if not CV2_AVAILABLE:
            return self._fallback_simple_join(left_image, right_image)
        
        try:
            # Convert PIL to OpenCV format
            left_cv = cv2.cvtColor(np.array(left_image), cv2.COLOR_RGB2BGR)
            right_cv = cv2.cvtColor(np.array(right_image), cv2.COLOR_RGB2BGR)
            
            # Prepare images for magazine stitching
            images = self._prepare_for_magazine_layout([left_cv, right_cv])
            
            if self.debug_mode:
                print(f"Attempting OpenCV stitching on {len(images)} images")
            
            # Attempt OpenCV stitching
            status, stitched = self.stitcher.stitch(images)
            
            if status == cv2.Stitcher_OK:
                # Convert back to PIL
                stitched_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
                result = Image.fromarray(stitched_rgb)
                
                if self.debug_mode:
                    print(f"OpenCV stitching successful: {result.size}")
                
                return self._post_process_magazine_result(result)
            else:
                if self.debug_mode:
                    status_msg = {
                        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
                    }.get(status, f"Unknown error: {status}")
                    print(f"OpenCV stitching failed: {status_msg}")
                
                # Fallback to simple joining
                return self._fallback_simple_join(left_image, right_image)
                
        except Exception as e:
            if self.debug_mode:
                print(f"Stitching error: {e}")
            return self._fallback_simple_join(left_image, right_image)
    
    def _prepare_for_magazine_layout(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Prepare images specifically for magazine layout stitching"""
        prepared = []
        target_height = min(img.shape[0] for img in images)
        
        for img in images:
            # Resize to consistent height while maintaining aspect ratio
            if img.shape[0] != target_height:
                scale = target_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_width, target_height))
            
            prepared.append(img)
        
        return prepared
    
    def _post_process_magazine_result(self, stitched: Image.Image) -> Image.Image:
        """Post-process stitched magazine spread"""
        if not CV2_AVAILABLE:
            return stitched # Cannot perform CV2 operations

        # Crop out any black borders that might have been introduced
        img_array = np.array(stitched)
        
        # Find non-black regions
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Create a binary mask of non-black pixels
        # Ensure the mask is in uint8 format for OpenCV morphological operations
        non_black_mask = (gray > 10).astype(np.uint8) * 255 
        
        # Optional: Apply erosion to remove small feathery artifacts
        # This can help get a tighter crop if edges are noisy
        kernel_size = 3 # Small kernel for subtle erosion
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask = cv2.erode(non_black_mask, kernel, iterations=1)
        
        # Find bounding box of non-black content from the (potentially eroded) mask
        # This method (np.argwhere + min/max) gets a SINGLE bounding box for ALL non-zero pixels.
        # For multiple distinct objects from a SAM mask, use get_individual_bounding_boxes_from_mask.
        coords = np.argwhere(eroded_mask > 0) # Use eroded_mask here
        
        if len(coords) > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            
            # Add small margin from config
            margin = self.config.crop_margin # Use config margin
            y0 = max(0, y0 - margin)
            x0 = max(0, x0 - margin)
            y1 = min(gray.shape[0] -1, y1 + margin) # Ensure y1 is within bounds
            x1 = min(gray.shape[1] -1, x1 + margin) # Ensure x1 is within bounds
            
            # Ensure crop coordinates are valid
            if x1 > x0 and y1 > y0:
                return stitched.crop((x0, y0, x1 + 1, y1 + 1)) # PIL crop is exclusive for x2, y2
            else:
                if self.debug_mode:
                    print(f"    Post-process crop: Invalid coordinates after erosion/margin ({x0},{y0},{x1},{y1}). Returning original stitched.")
                return stitched
        
        if self.debug_mode:
            print("    Post-process crop: No non-black content found after processing. Returning original stitched.")
        return stitched
    
    def _fallback_simple_join(self, left_image: Image.Image, right_image: Image.Image) -> Image.Image:
        """Simple side-by-side joining with overlap detection"""
        # Ensure same height
        min_height = min(left_image.height, right_image.height)
        
        if left_image.height != min_height:
            scale = min_height / left_image.height
            new_width = int(left_image.width * scale)
            left_image = left_image.resize((new_width, min_height), Image.Resampling.LANCZOS)
        
        if right_image.height != min_height:
            scale = min_height / right_image.height  
            new_width = int(right_image.width * scale)
            right_image = right_image.resize((new_width, min_height), Image.Resampling.LANCZOS)
        
        # Try to detect overlap/gap by comparing edge pixels
        overlap = self._detect_overlap(left_image, right_image)
        
        # Create combined image with detected overlap
        if overlap > 0:
            # Images overlap
            total_width = left_image.width + right_image.width - overlap
            combined = Image.new('RGB', (total_width, min_height))
            combined.paste(left_image, (0, 0))
            combined.paste(right_image, (left_image.width - overlap, 0))
        else:
            # Images have gap or perfect edge
            total_width = left_image.width + right_image.width + overlap  # overlap is negative for gaps
            combined = Image.new('RGB', (total_width, min_height))
            combined.paste(left_image, (0, 0))
            combined.paste(right_image, (left_image.width - overlap, 0))
        
        if self.debug_mode:
            print(f"Fallback joining with overlap: {overlap}px")
        
        return combined

    def _detect_overlap(self, left_img: Image.Image, right_img: Image.Image) -> int:
        """Detect overlap between images by comparing edge similarity"""
        if not CV2_AVAILABLE:
            return 0
        
        # Convert to arrays
        left_array = np.array(left_img)
        right_array = np.array(right_img)
        
        h = left_array.shape[0] # Assuming heights are already matched
        
        # Get right edge of left image and left edge of right image
        # Search up to 20% of the narrower image's width, max 100px
        max_search_overlap = min(100, left_img.width // 5, right_img.width // 5) 
        
        best_offset = 0 # Positive for overlap, negative for gap
        min_diff = float('inf')
        
        # Try different overlaps (positive values) and gaps (negative values)
        # Range from a small gap to a significant overlap
        for offset in range(-10, max_search_overlap): 
            current_diff = float('inf')
            
            if offset >= 0: # Testing for overlap of 'offset' pixels
                if left_array.shape[1] > offset and right_array.shape[1] > offset:
                    # Region from left image that would be overlapped
                    left_overlap_region = left_array[:, left_array.shape[1]-offset:]
                    # Region from right image that would do the overlapping
                    right_overlap_region = right_array[:, :offset]
                    
                    if left_overlap_region.shape[1] > 0 and right_overlap_region.shape[1] > 0: # Ensure regions are not empty
                        # Ensure comparison regions are of the same width (which they are by definition of offset)
                        current_diff = np.mean(np.abs(left_overlap_region.astype(float) - right_overlap_region.astype(float)))
            else: # Testing for a gap of 'abs(offset)' pixels
                # Penalize gaps; simpler to just let overlap detection find the best match
                # or handle gaps by not finding a low current_diff.
                # For simplicity, we can just assign a high diff for gaps or rely on no good match.
                current_diff = 1000 # Arbitrary high difference for gaps

            if current_diff < min_diff:
                min_diff = current_diff
                best_offset = offset
        
        # If no good match found (high difference), or if a gap was preferred (best_offset < 0),
        # default to a small positive overlap to avoid visible gaps in simple join.
        # Threshold for "good match" can be tuned.
        if min_diff > 50 or best_offset < 0: # Increased threshold for difference, also ensure positive overlap
            if self.debug_mode:
                print(f"    Overlap detection: No strong match (min_diff: {min_diff:.2f}, best_offset: {best_offset}). Defaulting to 2px overlap.")
            return 2 
        
        if self.debug_mode:
            print(f"    Overlap detection: Best offset {best_offset}px with difference {min_diff:.2f}")
        return best_offset


class KorniaGPUEnhancer:
    """GPU-accelerated image enhancement using Kornia"""
    
    def __init__(self, device=None, debug_mode: bool = False): # Add debug_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.available = KORNIA_AVAILABLE and torch.cuda.is_available()
        self.debug_mode = debug_mode # Initialize debug_mode

        if not self.available:
            print("Kornia GPU enhancement not available - falling back to CPU processing")
    
    def enhance_for_sd_training(self, image: Image.Image, enhancement_strength: float = 1.0) -> Image.Image:
        """Enhanced image processing with color profile preservation"""
        if not self.available:
            return self._cpu_fallback_enhancement(image, enhancement_strength)
        
        if enhancement_strength == 0:
            return image

        # Store original color profile
        original_profile = None
        if hasattr(image, 'info') and 'icc_profile' in image.info:
            original_profile = image.info['icc_profile']
            if self.debug_mode:
                print(f"    Kornia: Preserving ICC profile: {len(original_profile)} bytes")

        try:
            # FIXED: Convert PIL to tensor properly
            import torchvision.transforms as transforms
            
            # Use torchvision transforms instead of the problematic T.Compose
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Apply GPU-accelerated enhancements
            enhanced = self._apply_kornia_pipeline_auto(img_tensor, enhancement_strength)
            
            # Convert back to PIL with profile preservation
            enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
            
            enhanced_image = Image.fromarray(enhanced_np, mode=image.mode)
            
            # Restore color profile
            if original_profile:
                enhanced_image.info['icc_profile'] = original_profile
            
            return enhanced_image
            
        except Exception as e:
            if self.debug_mode:
                print(f"    Kornia enhancement failed: {e}, using CPU fallback")
            return self._cpu_fallback_enhancement(image, enhancement_strength)
    

    def _cpu_fallback_enhancement(self, image: Image.Image, strength: float) -> Image.Image: # Simplified signature
        """CPU fallback enhancement using PIL with auto-detection (conceptual)"""
        if strength == 0:
            return image
            
        # Store original profile
        original_profile = None
        if hasattr(image, 'info') and 'icc_profile' in image.info:
            original_profile = image.info['icc_profile']
            
        enhanced = image.copy()
        
        # Apply enhancements (your existing logic)
        if strength > 0.5:
            if self.debug_mode: 
                print(f"CPU: Applying Denoising (strength: {strength:.2f})")

        # Conceptual: Add auto-detection logic for CPU as well if desired
        # For now, it applies filters based on strength, similar to previous version but without explicit flags

        # --- 1. Denoising (Example: apply if strength is high) ---
        if strength > 0.5: # Simplified condition
            if self.debug_mode: print(f"CPU: Applying Denoising (strength: {strength:.2f})")
            denoise_amount = strength - 0.5 
            if denoise_amount > 0.3: 
                enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
                enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5 + (0.5 * denoise_amount)))
            elif denoise_amount > 0.1:
                enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            else: 
                enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.3 + (0.7 * denoise_amount)))

        # --- 2. Dynamic Range / Contrast (Example: always apply some contrast based on strength) ---
        if strength > 0.1:
            if self.debug_mode: print(f"CPU: Adjusting Contrast/Brightness (strength: {strength:.2f})")
            
            contrast_factor = 1.0 + (0.25 * strength)
            enhancer_c = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer_c.enhance(contrast_factor)

            # Basic auto-contrast for very high strength
            if strength > 0.8:
                 from PIL import ImageOps
                 enhanced = ImageOps.autocontrast(enhanced, cutoff=1) 

        # --- 3. Sharpening (Example: apply if strength is moderate and not too much denoising happened) ---
        if strength > 0.15 and strength < 0.8: # Avoid over-sharpening heavily processed images
            if self.debug_mode: print(f"CPU: Applying Sharpening (strength: {strength:.2f})")
            sharpness_factor = 1.0 + (0.35 * strength)
            enhancer_s = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer_s.enhance(sharpness_factor)
        
        # Restore profile
        if original_profile:
            enhanced.info['icc_profile'] = original_profile

        return enhanced
    # --- Auto-detection helper methods ---
    def _detect_dynamic_range_issues(self, img_tensor: torch.Tensor, shadow_thresh=0.05, highlight_thresh=0.95, saturation_percent_thresh=5.0) -> Tuple[bool, float]:
        """Detects blocked shadows or highlights. Returns (has_issue, severity_score)."""
        if img_tensor.shape[1] == 3: # Color image
            gray_tensor = kornia.color.rgb_to_grayscale(img_tensor)
        else: # Grayscale
            gray_tensor = img_tensor
        
        hist = torch.histc(gray_tensor, bins=256, min=0.0, max=1.0)
        total_pixels = gray_tensor.numel()
        
        shadow_pixels = torch.sum(hist[:int(256 * shadow_thresh)])
        highlight_pixels = torch.sum(hist[int(256 * highlight_thresh):])
        
        shadow_percent = (shadow_pixels / total_pixels) * 100
        highlight_percent = (highlight_pixels / total_pixels) * 100
        
        has_issue = shadow_percent > saturation_percent_thresh or highlight_percent > saturation_percent_thresh
        severity = max(shadow_percent, highlight_percent) / 50.0 # Normalize severity (0-2, higher is worse)
        
        if self.debug_mode and has_issue:
            print(f"Kornia DR Detect: Shadow Sat: {shadow_percent:.2f}%, Highlight Sat: {highlight_percent:.2f}%, Severity: {severity:.2f}")
        return has_issue, min(severity, 1.0) # Cap severity at 1.0 for multiplier

    def _detect_noise(self, img_tensor: torch.Tensor, laplacian_thresh_factor=0.03, high_laplacian_percent_thresh=10.0) -> Tuple[bool, float]:
        """Detects noise using Laplacian variance. Returns (is_noisy, noise_level_score)."""
        if img_tensor.shape[1] == 3:
            gray_tensor = kornia.color.rgb_to_grayscale(img_tensor)
        else:
            gray_tensor = img_tensor

        laplacian_map = kornia.filters.laplacian(gray_tensor, kernel_size=3)
        laplacian_abs_mean = torch.mean(torch.abs(laplacian_map))
        
        # Consider it noisy if the average absolute Laplacian response is high
        # This threshold is empirical and might need tuning
        noise_threshold = laplacian_thresh_factor 
        is_noisy = laplacian_abs_mean > noise_threshold
        
        # Severity score based on how much it exceeds threshold
        noise_level = 0.0
        if is_noisy:
            noise_level = min((laplacian_abs_mean / (noise_threshold * 5)), 1.0) # Normalize, cap at 1.0

        if self.debug_mode and is_noisy:
            print(f"Kornia Noise Detect: Laplacian Abs Mean: {laplacian_abs_mean:.4f}, Noise Level: {noise_level:.2f}")
        return is_noisy, noise_level

    def _detect_artifacts(self, img_tensor: torch.Tensor, edge_strength_thresh=0.2, artifact_score_thresh=0.1) -> Tuple[bool, float]:
        """Detects potential compression artifacts (e.g. ringing, blockiness) by looking for unnatural edge patterns.
           Returns (has_artifacts, artifact_severity_score).
           This is a simplified heuristic.
        """
        if img_tensor.shape[1] == 3:
            gray_tensor = kornia.color.rgb_to_grayscale(img_tensor)
        else:
            gray_tensor = img_tensor

        # Use Sobel to find edges
        sobel_edges = kornia.filters.sobel(gray_tensor) # Magnitude of Sobel
        
        # Artifacts often create unusually strong responses in otherwise flat areas,
        # or very regular grid patterns.
        # A simple heuristic: high variance in edge map might indicate ringing or busy artifacts.
        edge_variance = torch.var(sobel_edges)
        
        # Another heuristic: difference between median and mean of edge map.
        # Ringing might create sparse strong edges, pulling mean away from median.
        edge_mean = torch.mean(sobel_edges)
        edge_median = torch.median(sobel_edges)
        mean_median_diff = torch.abs(edge_mean - edge_median)

        # Thresholds are empirical
        has_artifacts = edge_variance > 0.005 or mean_median_diff > 0.01 
        artifact_severity = 0.0
        if has_artifacts:
            artifact_severity = min((edge_variance / 0.01) + (mean_median_diff / 0.02), 1.0)


        if self.debug_mode and has_artifacts:
            print(f"Kornia Artifact Detect: Edge Variance: {edge_variance:.4f}, Mean-Median Diff: {mean_median_diff:.4f}, Severity: {artifact_severity:.2f}")
        return has_artifacts, artifact_severity

    def _apply_kornia_pipeline_auto(self, img_tensor: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply Kornia enhancement pipeline with auto-detection of issues."""
        enhanced = img_tensor.clone()

        # --- 1. Auto-detect and fix dynamic range ---
        dr_issue_detected, dr_severity = self._detect_dynamic_range_issues(enhanced)
        if dr_issue_detected and strength > 0.1:
            # dr_severity can be a tensor, ensure clahe_strength_mod becomes a float for calculations
            dr_severity_val = dr_severity.item() if isinstance(dr_severity, torch.Tensor) else dr_severity
            clahe_strength_mod = strength * (0.5 + dr_severity_val * 0.5)
            
            clahe_clip_limit = 1.0 + (1.5 * clahe_strength_mod) # This is now float

            clahe_grid_size = (8,8) if clahe_strength_mod < 0.7 else (12,12)
            if self.debug_mode: print(f"Kornia: Applying CLAHE (strength_mod: {clahe_strength_mod:.2f}, clip: {clahe_clip_limit:.2f})")
            enhanced = kornia.enhance.equalize_clahe(enhanced, clip_limit=clahe_clip_limit, grid_size=clahe_grid_size)
            enhanced = torch.clamp(enhanced, 0, 1)

        # --- 2. Auto-detect and fix noise ---
        is_noisy, noise_level = self._detect_noise(enhanced) # Analyze potentially DR-corrected image
        applied_denoising = False
        if is_noisy and strength > 0.1:
            # noise_level can be a tensor
            noise_level_val = noise_level.item() if isinstance(noise_level, torch.Tensor) else noise_level
            denoise_strength_mod = strength * (0.3 + noise_level_val * 0.7) # float
            
            if denoise_strength_mod > 0.05:
                applied_denoising = True
                if self.debug_mode: 
                    print(f"Kornia: Applying Denoising (strength_mod: {denoise_strength_mod:.2f}, noise_level: {noise_level_val:.2f})") # Added print
                
                gauss_sigma_val = 0.4 + (denoise_strength_mod * 0.8) # float
                gauss_k_size = 3 if gauss_sigma_val < 0.8 else 5
                # Ensure sigma is a tuple of floats
                enhanced = kornia.filters.gaussian_blur2d(enhanced, kernel_size=(gauss_k_size,gauss_k_size), sigma=(gauss_sigma_val, gauss_sigma_val))
                enhanced = torch.clamp(enhanced, 0, 1)

        # --- 3. Auto-detect and fix artifacts ---
        has_artifacts, artifact_severity = self._detect_artifacts(enhanced) # Analyze potentially denoised image
        applied_deartifacting = False
        if has_artifacts and strength > 0.2: # Higher base strength threshold for this
            # artifact_severity can be a tensor
            artifact_severity_val = artifact_severity.item() if isinstance(artifact_severity, torch.Tensor) else artifact_severity
            deartifact_strength_mod = strength * (0.5 + artifact_severity_val * 0.5) # float
            
            if deartifact_strength_mod > 0.1:
                applied_deartifacting = True
                if self.debug_mode: 
                    print(f"Kornia: Applying Bilateral Blur for artifacts (strength_mod: {deartifact_strength_mod:.2f}, artifact_severity: {artifact_severity_val:.2f})") # Added print
                
                bilateral_sigma_color_val = 0.03 + (0.07 * deartifact_strength_mod) # float
                bilateral_sigma_space_val = 0.5 + deartifact_strength_mod # float
                kernel_s = 5 if deartifact_strength_mod < 0.7 else 7
                
                enhanced = kornia.filters.bilateral_blur(
                    enhanced, kernel_size=(kernel_s, kernel_s),
                    sigma_color=bilateral_sigma_color_val, # float
                    sigma_space=(bilateral_sigma_space_val, bilateral_sigma_space_val) # tuple of floats
                )
                enhanced = torch.clamp(enhanced, 0, 1)
        
        # --- 4. Sharpening ---
        # Apply sharpening if strength is sufficient and image wasn't heavily processed for noise/artifacts
        sharpen_condition = strength > 0.15
        if applied_denoising and noise_level > 0.6: sharpen_condition = False # Don't sharpen if heavily denoised
        if applied_deartifacting and artifact_severity > 0.6: sharpen_condition = False # Don't sharpen if heavily de-artifacted
        
        if sharpen_condition:
            if self.debug_mode: print(f"Kornia: Applying Sharpening (strength: {strength:.2f})")
            unsharp_amount = 0.3 + (0.7 * strength)
            unsharp_sigma = 0.5 + (0.5 * strength)
            # Ensure kernel_size for gaussian_blur2d is odd
            blur_k_size = 3 if unsharp_sigma < 0.8 else 5 
            blurred_for_sharpen = kornia.filters.gaussian_blur2d(enhanced, (blur_k_size,blur_k_size), (unsharp_sigma, unsharp_sigma)) # Changed to gaussian_blur2d
            enhanced = torch.clamp(enhanced + (enhanced - blurred_for_sharpen) * unsharp_amount, 0, 1)

        return torch.clamp(enhanced, 0, 1)

class SmartDoublePageDetector:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
        # Initialize Florence2 detector as the primary model
        self.florence2_detector = None
        if FLORENCE2_AVAILABLE:
            try:
                self.florence2_detector = Florence2RectangleDetector(
                    model_name="CogFlorence-2.2-Large",
                    comfyui_base_path=".",  # Will use relative path
                    min_box_area=1000
                )
                if debug_mode:
                    print("✅ Florence2 detector initialized successfully")
            except Exception as e:
                if debug_mode:
                    print(f"⚠️ Florence2 detector failed to initialize: {e}")
                self.florence2_detector = None
        
        # Initialize fallback analyzers
        self.border_analyzer = EnhancedBorderAnalyzer()
        self.border_analyzer.debug_mode = debug_mode
        self.double_page_detector = AdvancedDoublePageDetector(debug_mode=debug_mode)
        
        # Report status
        self._report_model_status()


    def _report_model_status(self):
        """Report which models are available"""
        status_lines = []
        
        if self.florence2_detector:
            status_lines.append(f"🎯 Florence2 (Primary): ✅ Available")
        else:
            status_lines.append(f"🎯 Florence2 (Primary): ❌ Not Available")
        
        status_lines.append(f"📐 Border Analysis: ✅ Available")
        status_lines.append(f"🔍 Advanced Double-Page Detection: ✅ Available")
        status_lines.append(f"📏 Geometric Analysis: ✅ Available")
        
        print("\n".join(status_lines))
    
    def analyze_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Analyze a list of images to detect double-page spreads"""
        results = []
        
        for i, image in enumerate(images):
            if self.debug_mode:
                print(f"🔍 Analyzing image {i+1}/{len(images)} (size: {image.size})")
            
            # Method 1: Florence2 detection (HIGHEST PRIORITY)
            florence2_result = self._florence2_analysis(image)
            
            # Method 2: Advanced double-page detection
            double_page_result = self.double_page_detector.analyze_double_page_characteristics(image)
            
            # Method 3: Border pattern analysis
            border_result = self.border_analyzer.analyze_borders(image)
            
            # Method 4: Basic geometric analysis
            geometric_result = self._geometric_analysis(image)
            
            if self.debug_mode:
                print(f"    Florence2: confidence {florence2_result.get('confidence', 0):.3f}")
                print(f"    Double-page: confidence {double_page_result.get('confidence', 0):.3f}")
                print(f"    Border: confidence {border_result.get('confidence', 0):.3f}")
                print(f"    Geometric: confidence {geometric_result.get('confidence', 0):.3f}")
            
            # Combine results with Florence2 having highest weight
            final_result = self._combine_analysis_results_with_florence2(
                florence2_result,
                border_result,
                geometric_result,
                double_page_result,
                image,
                i
            )
            
            results.append(final_result)
            
            if self.debug_mode:
                print(f"    Final: {final_result['page_type']} (confidence: {final_result['confidence']:.3f})")
        
        return results

    def _florence2_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image using Florence2 for rectangular content detection"""
        if not self.florence2_detector:
            return {
                "page_type": PageType.UNCERTAIN,
                "confidence": 0.0,
                "method": "florence2_unavailable",
                "detected_rectangles": [],
                "has_content": False
            }
        
        try:
            # Detect rectangular content in the image
            bounding_boxes, mask_image = self.florence2_detector.detect_rectangles(
                image=image,
                text_input="rectangular images in page OR photograph OR illustration OR diagram OR chart OR map OR text block",
                return_mask=True,
                keep_model_loaded=True
            )
            
            if self.debug_mode:
                print(f"    Florence2 detected {len(bounding_boxes)} rectangles")
            
            # Analyze the detected rectangles to determine page type
            page_type, confidence = self._analyze_florence2_rectangles(bounding_boxes, image.size)
            
            return {
                "page_type": page_type,
                "confidence": confidence,
                "method": "florence2",
                "detected_rectangles": [
                    {
                        "bbox": box.to_tuple(),
                        "label": box.label,
                        "area": box.area,
                        "confidence": box.confidence
                    } for box in bounding_boxes
                ],
                "has_content": len(bounding_boxes) > 0,
                "mask_image": mask_image
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"    Florence2 analysis failed: {e}")
            return {
                "page_type": PageType.UNCERTAIN,
                "confidence": 0.0,
                "method": "florence2_error",
                "detected_rectangles": [],
                "has_content": False,
                "error": str(e)
            }

    def _analyze_florence2_rectangles(self, bounding_boxes: List[BoundingBox], 
                                    image_size: Tuple[int, int]) -> Tuple[PageType, float]:
        """Analyze Florence2 detected rectangles to determine page type"""
        if not bounding_boxes:
            return PageType.UNCERTAIN, 0.0
        
        width, height = image_size
        aspect_ratio = width / height
        
        # Analyze spatial distribution of rectangles
        left_side_boxes = []
        right_side_boxes = []
        center_boxes = []
        
        center_x = width // 2
        margin = width // 6  # 1/6 of width as center margin
        
        for box in bounding_boxes:
            box_center_x = box.x1 + (box.width // 2)
            
            if box_center_x < center_x - margin:
                left_side_boxes.append(box)
            elif box_center_x > center_x + margin:
                right_side_boxes.append(box)
            else:
                center_boxes.append(box)
        
        total_boxes = len(bounding_boxes)
        left_count = len(left_side_boxes)
        right_count = len(right_side_boxes)
        center_count = len(center_boxes)
        
        if self.debug_mode:
            print(f"      Rectangle distribution: Left={left_count}, Center={center_count}, Right={right_count}")
        
        # Calculate areas for weight analysis
        left_area = sum(box.area for box in left_side_boxes)
        right_area = sum(box.area for box in right_side_boxes)
        total_area = sum(box.area for box in bounding_boxes)
        
        # Decision logic with confidence scoring
        confidence = 0.0
        page_type = PageType.UNCERTAIN
        
        # Strong evidence for double page spread
        if aspect_ratio > 1.6 and total_boxes >= 2:
            # Check for asymmetric distribution
            if left_count > 0 and right_count == 0 and center_count <= 1:
                # Content only on left side - likely left page of spread
                page_type = PageType.DOUBLE_PAGE_LEFT
                confidence = min(0.9, 0.6 + (left_count / total_boxes) * 0.3)
            elif right_count > 0 and left_count == 0 and center_count <= 1:
                # Content only on right side - likely right page of spread
                page_type = PageType.DOUBLE_PAGE_RIGHT
                confidence = min(0.9, 0.6 + (right_count / total_boxes) * 0.3)
            elif left_count > 0 and right_count > 0:
                # Content on both sides - analyze balance
                area_ratio = abs(left_area - right_area) / max(left_area, right_area, 1)
                if area_ratio > 0.3:  # Significant imbalance
                    if left_area > right_area:
                        page_type = PageType.DOUBLE_PAGE_LEFT
                    else:
                        page_type = PageType.DOUBLE_PAGE_RIGHT
                    confidence = 0.5 + min(0.3, area_ratio)
                else:
                    # Balanced spread - default to left
                    page_type = PageType.DOUBLE_PAGE_LEFT
                    confidence = 0.4
        
        # Evidence for single page
        elif aspect_ratio <= 1.6 or total_boxes == 1 or center_count >= total_boxes * 0.6:
            page_type = PageType.SINGLE_PAGE
            if total_boxes == 1:
                confidence = 0.8
            elif center_count >= total_boxes * 0.6:
                confidence = 0.7
            else:
                confidence = 0.5
        
        # Boost confidence based on number of detected rectangles
        if total_boxes >= 3:
            confidence = min(1.0, confidence + 0.1)
        
        if self.debug_mode:
            print(f"      Florence2 analysis: {page_type} (confidence: {confidence:.3f})")
        
        return page_type, confidence

    def _combine_analysis_results_with_florence2(self, florence2_result: Dict, border_result: Dict, 
                                               geometric_result: Dict, double_page_result: Dict,
                                               image: Image.Image, index: int) -> Dict[str, Any]:
        """Combine analysis results with Florence2 having highest priority"""
        
        # Set weights with Florence2 as primary
        weights = {
            "florence2": 0.7,      # Highest weight for Florence2
            "double_page": 0.21,   # Reduced weight for advanced detection
            "border": 0.15,        # Border analysis
            "geometric": 0.04       # Lowest weight for geometric
        }
        
        votes: Dict[PageType, float] = {}
        
        # Florence2 vote (highest priority)
        florence2_page_type = florence2_result["page_type"]
        florence2_confidence = florence2_result["confidence"]
        if florence2_confidence > 0:
            votes[florence2_page_type] = votes.get(florence2_page_type, 0) + (florence2_confidence * weights["florence2"])
            if self.debug_mode:
                print(f"    Florence2 vote: {florence2_page_type} score {florence2_confidence * weights['florence2']:.3f}")
        
        # Advanced double-page detection vote
        if double_page_result.get("is_double_page", False):
            adv_page_type = double_page_result["page_type"]
            adv_raw_confidence = double_page_result["confidence"]
            votes[adv_page_type] = votes.get(adv_page_type, 0) + (adv_raw_confidence * weights["double_page"])
            if self.debug_mode:
                print(f"    Double-page vote: {adv_page_type} score {adv_raw_confidence * weights['double_page']:.3f}")
        
        # Border analysis vote
        border_page_type = border_result["page_type"]
        border_raw_confidence = border_result["confidence"]
        votes[border_page_type] = votes.get(border_page_type, 0) + (border_raw_confidence * weights["border"])
        if self.debug_mode:
            print(f"    Border vote: {border_page_type} score {border_raw_confidence * weights['border']:.3f}")
        
        # Geometric analysis vote
        geo_page_type = geometric_result["page_type"]
        geo_raw_confidence = geometric_result["confidence"]
        votes[geo_page_type] = votes.get(geo_page_type, 0) + (geo_raw_confidence * weights["geometric"])
        if self.debug_mode:
            print(f"    Geometric vote: {geo_page_type} score {geo_raw_confidence * weights['geometric']:.3f}")
        
        if not votes:
            final_page_type = PageType.UNCERTAIN
            final_confidence = 0.0
        else:
            # Determine winner based on highest accumulated score
            final_page_type, winner_score = max(votes.items(), key=lambda x: x[1])
            final_confidence = min(winner_score, 1.0)
            
            # Special case: If Florence2 has high confidence, boost final confidence
            if florence2_result["confidence"] > 0.7 and florence2_result["page_type"] == final_page_type:
                final_confidence = min(1.0, final_confidence + 0.1)
        
        return {
            "page_type": final_page_type,
            "confidence": final_confidence,
            "florence2_result": florence2_result,
            "border_result": border_result,
            "geometric_result": geometric_result,
            "double_page_result": double_page_result,
            "image_index": index,
            "image_size": image.size
        }


    
    def _geometric_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Basic geometric analysis for page type detection"""
        width, height = image.size
        aspect_ratio = width / height
        
        # Simple heuristics based on aspect ratio
        if aspect_ratio > 2.0:  # Very wide - likely double page
            page_type = PageType.DOUBLE_PAGE_LEFT  # Default, will be refined
            confidence = 0.6
        elif aspect_ratio > 1.5:  # Moderately wide - possibly double page
            page_type = PageType.DOUBLE_PAGE_LEFT
            confidence = 0.4
        elif 0.7 <= aspect_ratio <= 1.3:  # Square-ish - likely single page
            page_type = PageType.SINGLE_PAGE
            confidence = 0.7
        else:  # Other ratios
            page_type = PageType.UNCERTAIN
            confidence = 0.3
        
        return {
            "page_type": page_type,
            "confidence": confidence,
            "aspect_ratio": aspect_ratio
        }
    
    def _combine_analysis_results_fixed(self, border_result: Dict, geometric_result: Dict, 
                                       double_page_result: Dict, image: Image.Image, index: int) -> Dict[str, Any]:
        """Fixed analysis combination with proper weighting"""
        
        weights = {
            "double_page": 0.6,
            "border": 0.3,      
            "geometric": 0.1    
        }
        
        votes: Dict[PageType, float] = {}
        
        # Advanced double-page detection vote
        # Note: double_page_result["page_type"] from AdvancedDoublePageDetector is currently always DOUBLE_PAGE_LEFT if is_double_page=True
        if double_page_result.get("is_double_page", False):
            adv_page_type = double_page_result["page_type"] 
            adv_raw_confidence = double_page_result["confidence"]
            votes[adv_page_type] = votes.get(adv_page_type, 0) + (adv_raw_confidence * weights["double_page"])
            if self.debug_mode:
                print(f"    Advanced vote: {adv_page_type} score {adv_raw_confidence * weights['double_page']:.3f} (raw_conf: {adv_raw_confidence:.3f})")
        
        # Border analysis vote (key for L/R orientation)
        border_page_type = border_result["page_type"]
        border_raw_confidence = border_result["confidence"]
        votes[border_page_type] = votes.get(border_page_type, 0) + (border_raw_confidence * weights["border"])
        if self.debug_mode:
            print(f"    Border vote: {border_page_type} score {border_raw_confidence * weights['border']:.3f} (raw_conf: {border_raw_confidence:.3f})")
        
        # Geometric analysis vote
        geo_page_type = geometric_result["page_type"]
        geo_raw_confidence = geometric_result["confidence"]
        votes[geo_page_type] = votes.get(geo_page_type, 0) + (geo_raw_confidence * weights["geometric"])
        if self.debug_mode:
            print(f"    Geometric vote: {geo_page_type} score {geo_raw_confidence * weights['geometric']:.3f} (raw_conf: {geo_raw_confidence:.3f})")
        
        if not votes:
            final_page_type = PageType.UNCERTAIN
            final_confidence = 0.0
        else:
            # Determine winner based on highest accumulated score
            final_page_type, winner_score = max(votes.items(), key=lambda x: x[1])
            
            # Normalize confidence: The winner_score is a sum of (raw_confidence * weight).
            # Max possible score is sum of all weights (1.0). So winner_score is effectively the confidence.
            final_confidence = min(winner_score, 1.0) 

            # Conditional L/R adjustment using index as a weaker heuristic
            # This applies if the current winner is DOUBLE_PAGE_LEFT,
            # it's an odd page, and there isn't strong counter-evidence from border analysis.
            if (final_page_type == PageType.DOUBLE_PAGE_LEFT and
                index % 2 == 1 and # Odd index might suggest a right page
                double_page_result.get("is_double_page", False) and # Advanced detector identified a spread
                double_page_result["page_type"] == PageType.DOUBLE_PAGE_LEFT): # And it used the default LEFT orientation

                # Check if border analysis strongly suggested RIGHT, which might have been outvoted
                border_suggested_right = (border_result["page_type"] == PageType.DOUBLE_PAGE_RIGHT and 
                                          border_result["confidence"] > 0.6) # Threshold for strong border opinion

                if not border_suggested_right and final_confidence < 0.75: # If border doesn't strongly say RIGHT and overall confidence isn't maxed out
                    if self.debug_mode:
                        print(f"    Applying index-based heuristic: Tentatively flipping {final_page_type} to DOUBLE_PAGE_RIGHT for index {index} (current conf: {final_confidence:.3f})")
                    final_page_type = PageType.DOUBLE_PAGE_RIGHT
                    # Optionally, slightly adjust confidence if a heuristic is applied
                    # final_confidence = max(0.0, final_confidence - 0.05) 
        
        return {
            "page_type": final_page_type,
            "confidence": final_confidence,
            "vision_result": {},     # Retained for compatibility
            "border_result": border_result,
            "geometric_result": geometric_result,
            "double_page_result": double_page_result,
            "image_index": index,
            "image_size": image.size
        }




class AdvancedDoublePageDetector:
    """Advanced double-page detection using edge detection and line analysis"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
    
    def analyze_double_page_characteristics(self, image: Image.Image) -> Dict[str, Any]:
        """Comprehensive double-page analysis using multiple techniques"""
        if not CV2_AVAILABLE:
            return {"is_double_page": False, "confidence": 0.0, "method": "unavailable"}
        
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        height, width = gray.shape
        aspect_ratio = width / height
        
        results = {}
        
        # Method 1: Center gutter detection using Hough lines
        gutter_result = self._detect_center_gutter(gray)
        results["gutter"] = gutter_result
        
        # Method 2: Image partitioning and comparison
        partition_result = self._analyze_image_partitions(gray)
        results["partition"] = partition_result
        
        # Method 3: Edge density analysis
        edge_result = self._analyze_edge_distribution(gray)
        results["edge"] = edge_result
        
        # Method 4: Aspect ratio analysis
        aspect_result = self._analyze_aspect_ratio(aspect_ratio)
        results["aspect"] = aspect_result
        
        # Combine results
        final_result = self._combine_detection_results(results, image.size)
        
        if self.debug_mode:
            print(f"    Double-page analysis: {final_result['method']} -> {final_result['is_double_page']} (confidence: {final_result['confidence']:.3f})")
            for method, result in results.items():
                print(f"      {method}: {result.get('detected', False)} (conf: {result.get('confidence', 0):.3f})")
        
        return final_result
    
    def _detect_center_gutter(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect vertical gutter line in the center using Hough transform"""
        height, width = gray.shape
        
        # Edge detection with optimized parameters for gutter detection
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        
        # Focus on the center region where gutter would be
        center_margin = width // 6  # Look within central 1/3 of image
        center_region = edges[:, width//2 - center_margin:width//2 + center_margin]
        
        # Hough line transform for vertical lines
        lines = cv2.HoughLinesP(
            center_region, 
            rho=1, 
            theta=np.pi/180, 
            threshold=int(height * 0.3),  # Line must span at least 30% of height
            minLineLength=int(height * 0.4),  # Minimum 40% of image height
            maxLineGap=int(height * 0.1)     # Allow gaps up to 10% of height
        )
        
        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is mostly vertical (angle within 10 degrees of vertical)
                if abs(x1 - x2) < 20:  # Very small horizontal deviation
                    line_length = abs(y2 - y1)
                    # Adjust x coordinates back to full image coordinates
                    center_x = x1 + (width//2 - center_margin)
                    vertical_lines.append({
                        'x': center_x,
                        'length': line_length,
                        'center_distance': abs(center_x - width//2)
                    })
        
        # Find best center line
        best_line = None
        if vertical_lines:
            # Sort by combination of length and proximity to center
            scored_lines = []
            for line in vertical_lines:
                length_score = line['length'] / height
                center_score = 1 - (line['center_distance'] / (width//4))
                total_score = (length_score * 0.7) + (center_score * 0.3)
                scored_lines.append((total_score, line))
            
            scored_lines.sort(reverse=True)
            best_line = scored_lines[0][1]
        
        if best_line:
            confidence = min(1.0, (best_line['length'] / height) * 1.5)
            return {
                "detected": True,
                "confidence": confidence,
                "gutter_x": best_line['x'],
                "gutter_length": best_line['length'],
                "method": "hough_lines"
            }
        
        return {"detected": False, "confidence": 0.0, "method": "hough_lines"}
    
    def _analyze_image_partitions(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze left and right halves for content differences"""
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
        std_ratio = abs(left_std - right_std) / max(left_std, right_std)
        
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
        
        # Scoring: Lower correlation and higher differences suggest double page
        correlation_score = 1 - correlation  # Invert correlation (lower = more different)
        std_score = min(1.0, std_ratio * 2)
        mean_score = min(1.0, mean_diff * 3)
        edge_score = min(1.0, edge_density_diff * 5)
        
        # Combine scores
        total_score = (correlation_score * 0.3 + std_score * 0.3 + 
                      mean_score * 0.2 + edge_score * 0.2)
        
        detected = total_score > 0.3  # Threshold for detection
        
        return {
            "detected": detected,
            "confidence": min(1.0, total_score),
            "correlation": correlation,
            "std_ratio": std_ratio,
            "mean_diff": mean_diff,
            "edge_density_diff": edge_density_diff,
            "method": "partition_analysis"
        }
    
    def _analyze_edge_distribution(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze edge distribution patterns"""
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide image into vertical strips
        num_strips = 10
        strip_width = width // num_strips
        edge_densities = []
        
        for i in range(num_strips):
            start_x = i * strip_width
            end_x = min((i + 1) * strip_width, width)
            strip = edges[:, start_x:end_x]
            density = np.sum(strip > 0) / strip.size
            edge_densities.append(density)
        
        # Check for patterns indicating double page
        # Double pages often have:
        # 1. Lower edge density in the center (gutter)
        # 2. Higher edge density on the sides (content)
        
        center_strips = edge_densities[4:6]  # Middle strips
        side_strips = edge_densities[:2] + edge_densities[-2:]  # Side strips
        
        center_density = np.mean(center_strips)
        side_density = np.mean(side_strips)
        
        # Look for center "valley" pattern
        density_ratio = side_density / (center_density + 0.001)  # Avoid division by zero
        
        # Check for symmetry (double pages often have similar left/right patterns)
        left_densities = edge_densities[:5]
        right_densities = edge_densities[5:]
        right_densities.reverse()  # Reverse for comparison
        
        symmetry_score = 1 - np.mean([abs(l - r) for l, r in zip(left_densities, right_densities)])
        
        # Scoring
        valley_score = min(1.0, (density_ratio - 1) / 2)  # Higher ratio = more likely double page
        valley_score = max(0, valley_score)
        
        final_score = (valley_score * 0.6) + (symmetry_score * 0.4)
        detected = final_score > 0.3
        
        return {
            "detected": detected,
            "confidence": final_score,
            "center_density": center_density,
            "side_density": side_density,
            "density_ratio": density_ratio,
            "symmetry_score": symmetry_score,
            "method": "edge_distribution"
        }
    
    def _analyze_aspect_ratio(self, aspect_ratio: float) -> Dict[str, Any]:
        """Analyze aspect ratio for double-page likelihood"""
        # Typical aspect ratios:
        # Single page: 0.7-1.4 (portrait to landscape)
        # Double page: 1.4-2.5 (landscape to wide landscape)
        
        if aspect_ratio > 2.2:
            confidence = 0.9  # Very wide, almost certainly double page
        elif aspect_ratio > 1.8:
            confidence = 0.7  # Wide, likely double page
        elif aspect_ratio > 1.4:
            confidence = 0.5  # Moderately wide, possibly double page
        else:
            confidence = 0.1  # Not wide enough for typical double page
        
        detected = aspect_ratio > 1.4
        
        return {
            "detected": detected,
            "confidence": confidence,
            "aspect_ratio": aspect_ratio,
            "method": "aspect_ratio"
        }
    
    def _combine_detection_results(self, results: Dict, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Combine all detection methods into final result with better thresholds"""
        # Weight the different methods
        weights = {
            "gutter": 0.4,      # Gutter detection is most reliable
            "partition": 0.3,   # Content analysis is important
            "edge": 0.2,        # Edge patterns provide good hints
            "aspect": 0.1       # Aspect ratio is a basic check
        }
        
        total_score = 0.0
        total_weight = 0.0
        methods_used = []
        
        for method, result in results.items():
            if result.get("detected", False):
                weight = weights.get(method, 0.1)
                confidence = result.get("confidence", 0.0)
                total_score += confidence * weight
                total_weight += weight
                methods_used.append(method)
        
        # Normalize score
        if total_weight > 0:
            final_confidence = total_score / total_weight
        else:
            final_confidence = 0.0
        
        # LOWER the threshold for detection
        is_double_page = final_confidence > 0.25  # Lowered from 0.4
        
        # Determine page side if it's a double page
        page_type = PageType.SINGLE_PAGE
        if is_double_page:
            page_type = PageType.DOUBLE_PAGE_LEFT  # Will be adjusted later
        
        return {
            "is_double_page": is_double_page,
            "page_type": page_type,
            "confidence": final_confidence,
            "methods_used": methods_used,
            "method": f"combined({','.join(methods_used)})",
            "image_size": image_size,
            "detailed_results": results
        }


class EnhancedPDFExtractorNode:
    """Enhanced ComfyUI Node for PDF Image and Text Extraction"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get lists of available DINO config and SAM model files
        dino_config_folder = Path(folder_paths.models_dir) / "grounding-dino"
        sam_model_folder = Path(folder_paths.models_dir) / "sams"

        dino_configs = ["autodetect"] 
        if dino_config_folder.exists():
            dino_configs.extend([f.name for f in dino_config_folder.iterdir() if f.is_file() and f.name.endswith(".cfg.py")])
        
        sam_models = ["autodetect"]
        if sam_model_folder.exists():
            sam_models.extend([f.name for f in sam_model_folder.iterdir() if f.is_file() and (f.name.endswith(".pth") or f.name.endswith(".pt"))])

        return {
            "required": {
                "pdf_paths": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter PDF file paths, one per line"
                }),
                "output_directory": ("STRING", {
                    "default": "pdf_output_v3",
                    "tooltip": "Directory to save extracted images and text"
                }),
                "extract_images": ("BOOLEAN", {"default": True}),
                "extract_text": ("BOOLEAN", {"default": True}),
                
                "save_options": ([
                    "enhanced_only",     # Save only enhanced images (cropped + enhanced)
                    "original_only",     # Save only raw extracted images
                    "both"              # Save both versions
                ], {
                    "default": "enhanced_only",
                    "tooltip": "Enhanced uses smart cropping + GPU enhancement for SD training"
                }),
                
                "enable_smart_crop": ("BOOLEAN", { 
                    "default": True,
                    "tooltip": "Enable smart segmentation-based cropping to isolate main content. If False, no cropping is applied."
                }),
                
                "enhancement_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Enhancement intensity for SD training optimization"
                }),
                # Segmentation Prompts and Controls moved to optional for clarity
            },
            "optional": {
                "min_image_size": ("INT", {
                    "default": 200, 
                    "min": 50,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Minimum width/height for extracted images (pixels)"
                }),
                "crop_margin": ("INT", { 
                    "default": 5, 
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Margin (pixels) to add around cropped regions"
                }),
                "join_double_pages": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use advanced analysis to detect and join double-page spreads" 
                }),
                "use_gpu_acceleration": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use Kornia GPU acceleration for image enhancement"
                }),
                "border_confidence_threshold": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Minimum confidence for border pattern detection"
                }),
                "dino_config_name": (dino_configs, {
                    "default": "autodetect",
                    "tooltip": "Select GroundingDINO configuration file. 'autodetect' tries SwinT then SwinB."
                }),
                "sam_model_name": (sam_models, {
                    "default": "autodetect",
                    "tooltip": "Select SAM model file. 'autodetect' picks a preferred HQ model."
                }),
                "enable_image_segmentation": ("BOOLEAN", { # New
                    "default": True,
                    "tooltip": "Enable segmentation pass for images."
                }),
                "image_segmentation_prompt": ("STRING", { # Renamed
                    "multiline": True, # Allow more complex prompts
                    "default": "photograph OR main image OR illustration OR diagram OR chart OR map",
                    "tooltip": "Prompt for image segmentation (e.g., 'main subject', 'all people')"
                }),
                "enable_text_segmentation": ("BOOLEAN", { # New
                    "default": False, # Default to False to not slow down existing workflows initially
                    "tooltip": "Enable segmentation pass for text blocks."
                }),
                "text_segmentation_prompt": ("STRING", { # New
                    "multiline": True,
                    "default": "text block OR paragraph OR caption OR headline OR title",
                    "tooltip": "Prompt for text segmentation (e.g., 'all text', 'captions')"
                }),
                "segmentation_box_threshold": ("FLOAT", { # New
                    "default": 0.3, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "GroundingDINO box confidence threshold for segmentation."
                }),
                "segmentation_text_threshold": ("FLOAT", { # New
                    "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "GroundingDINO text confidence threshold for segmentation."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print detailed analysis and processing information"
                }),
                "save_debug_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save debug segmentation masks for analysis"
            }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("report_summary", "output_path", "total_images", "total_text_pages", "processing_stats")
    FUNCTION = "extract_pdf_content_enhanced"
    CATEGORY = "Eric/PDF Enhanced"
    
    def __init__(self):
        self.output_base_dir = folder_paths.get_output_directory()
        self.use_pymupdf = PYMUPDF_AVAILABLE
        
        if not (PYMUPDF_AVAILABLE or PYPDF2_AVAILABLE):
            raise ImportError("Either PyMuPDF or PyPDF2 is required for PDF processing")

    def extract_pdf_content_enhanced(self, pdf_paths, output_directory, extract_images, extract_text, 
                                save_options, enable_smart_crop, enhancement_strength, 
                                min_image_size=None, 
                                crop_margin=None,    
                                join_double_pages=True, use_gpu_acceleration=True, 
                                border_confidence_threshold=None,   
                                dino_config_name="autodetect", 
                                sam_model_name="autodetect",
                                enable_image_segmentation=True, 
                                image_segmentation_prompt="pictures inside picture OR text", 
                                enable_text_segmentation=False, 
                                text_segmentation_prompt="text block OR paragraph OR caption OR headline OR title", 
                                segmentation_box_threshold=0.3, 
                                segmentation_text_threshold=0.25,
                                save_debug_images=True, 
                                debug_mode=False):
        
        # Build config_kwargs first
        config_kwargs = {
            "join_double_pages": join_double_pages, 
            "use_gpu_acceleration": use_gpu_acceleration, 
            "debug_mode": debug_mode,
            "enhancement_strength": enhancement_strength, # Moved from below
            "dino_config_name": dino_config_name,         # Moved from below
            "sam_model_name": sam_model_name,           # Moved from below
            "enable_image_segmentation": enable_image_segmentation, # Moved from below
            "image_segmentation_prompt": image_segmentation_prompt, # Moved from below
            "enable_text_segmentation": enable_text_segmentation, # Moved from below
            "text_segmentation_prompt": text_segmentation_prompt, # Moved from below
            "segmentation_box_threshold": segmentation_box_threshold, # Moved from below
            "segmentation_text_threshold": segmentation_text_threshold,
            "save_debug_images": save_debug_images,
        }
        if min_image_size is not None: config_kwargs["min_image_size"] = min_image_size
        if crop_margin is not None: config_kwargs["crop_margin"] = crop_margin
        if border_confidence_threshold is not None: config_kwargs["border_confidence_threshold"] = border_confidence_threshold
        
        # Create config object once
        config = ProcessingConfig.from_node_inputs(**config_kwargs)

        current_detector = SmartDoublePageDetector(debug_mode=debug_mode)
        current_stitcher = ConstrainedStitcher(config=config) # Pass the created config
        current_stitcher.debug_mode = debug_mode 
        current_enhancer = KorniaGPUEnhancer(debug_mode=debug_mode)

        # Removed redundant config_kwargs build-up and second config creation
        
        if debug_mode:
            is_valid, errors = config.validate() # Validate again after all inputs are set
            print(f"📋 Configuration validation: Valid={is_valid}, Errors={errors if not is_valid else 'None'}")
            print(f"📋 Effective Config: {config}")

        
        # Configure debug mode for all components (already done for current_detector, current_enhancer)
        # current_detector.debug_mode = debug_mode # Already set at instantiation
        # current_stitcher.debug_mode = debug_mode # Already set
        # current_enhancer.debug_mode = debug_mode # Already set at instantiation

        # Parse PDF paths
        pdf_list = [path.strip() for path in pdf_paths.strip().split('\n') if path.strip()]
        
        if not pdf_list:
            return "No PDF files provided", "", 0, 0, "Error: No input files"
        
        # Validate PDF files
        valid_pdfs = []
        for pdf_path in pdf_list:
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file not found: {pdf_path}")
                continue
            if not pdf_path.lower().endswith('.pdf'):
                print(f"Warning: Not a PDF file: {pdf_path}")
                continue
            valid_pdfs.append(pdf_path)
        
        if not valid_pdfs:
            return "No valid PDF files found", "", 0, 0, "Error: No valid input files"
        
        # Set up output directory
        if os.path.isabs(output_directory):
            output_path = output_directory
        else:
            output_path = os.path.join(self.output_base_dir, output_directory)
        
        os.makedirs(output_path, exist_ok=True)
        
        # Print processing configuration
        print(f"\n🚀 Enhanced PDF Extractor v0.6.0")
        print(f"📁 Processing {len(valid_pdfs)} PDF files...")
        print(f"💾 Save Mode: {save_options}")
        print(f"✂️ Smart Cropping Enabled: {enable_smart_crop}")
        print(f"⚡ Enhancement Strength: {enhancement_strength}")
        print(f"🔗 Join Double Pages: {join_double_pages}")
        print(f"🖥️  GPU Acceleration: {use_gpu_acceleration and current_enhancer.available}") 
        
        
        # Process each PDF
        all_reports = []
        total_images = 0
        total_text_pages = 0
        total_joined = 0
        total_enhanced = 0
        
        for pdf_path in valid_pdfs:
            print(f"\n📖 Processing: {os.path.basename(pdf_path)}")
            
            try:
                # Create subdirectory for this PDF
                pdf_name = Path(pdf_path).stem
                pdf_output_dir = os.path.join(output_path, pdf_name)
                os.makedirs(pdf_output_dir, exist_ok=True)
                
                # Process PDF with enhanced pipeline
                processor = EnhancedPDFProcessor(
                    pdf_output_dir, 
                    detector=current_detector, # Use instance for this call
                    stitcher=current_stitcher, # Use instance for this call
                    enhancer=current_enhancer, # Use instance for this call
                    config=config,
                    debug_mode=debug_mode
                )
                
                processor.use_pymupdf = self.use_pymupdf

                report = processor.process_pdf_enhanced(
                    pdf_path,
                    extract_images=extract_images,
                    extract_text=extract_text,
                    save_options=save_options,
                    enable_smart_crop=enable_smart_crop,
                    enhancement_strength=enhancement_strength,
                    join_double_pages=join_double_pages,
                    use_gpu_acceleration=use_gpu_acceleration
                )
                
                all_reports.append(report)
                total_images += report.images_extracted
                total_text_pages += report.text_extracted_pages
                total_joined += report.images_joined
                total_enhanced += report.images_enhanced
                
                # Save individual report
                report_path = os.path.join(pdf_output_dir, "enhanced_processing_report.json")
                with open(report_path, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
                
                # Print results
                print(f"✅ Completed: {report.images_extracted} images, {report.text_extracted_pages} text pages")
                if report.images_joined > 0:
                    print(f"🔗 Joined {report.images_joined} double-page spreads")
                if report.images_enhanced > 0:
                    print(f"⚡ Enhanced {report.images_enhanced} images")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_path}: {e}")
                if debug_mode:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Create summary report
        summary = self._create_enhanced_summary_report(
            all_reports, len(valid_pdfs), save_options, total_joined, total_enhanced
        )
        
        # Create processing stats
        processing_stats = self._create_processing_stats(all_reports)
        
        # Save combined report
        if all_reports:
            combined_report_path = os.path.join(output_path, "enhanced_combined_report.json")
            combined_data = {
                "version": "0.4.0",
                "summary": {
                    "total_pdfs": len(valid_pdfs),
                    "successful_pdfs": len(all_reports),
                    "total_images": total_images,
                    "total_text_pages": total_text_pages,
                    "total_joined": total_joined,
                    "total_enhanced": total_enhanced,
                    "processing_time": sum(r.processing_time for r in all_reports),
                    "save_options": save_options,
                    "enable_smart_crop": enable_smart_crop,
                    "enhancement_strength": enhancement_strength
                },
                "individual_reports": [asdict(report) for report in all_reports]
            }
            
            with open(combined_report_path, 'w') as f:
                json.dump(combined_data, f, indent=2, default=str)
        
        print(f"\n🎉 Processing complete! Check {output_path} for results.")
        
        return summary, output_path, total_images, total_text_pages, processing_stats
    
    def _create_enhanced_summary_report(self, reports, total_pdfs, save_options, total_joined, total_enhanced):
        """Create enhanced summary report"""
        if not reports:
            return "No PDFs were successfully processed."
        
        successful_pdfs = len(reports)
        total_images = sum(r.images_extracted for r in reports)
        total_text_pages = sum(r.text_extracted_pages for r in reports)
        total_time = sum(r.processing_time for r in reports)
        
        # Page type breakdown
        page_type_totals = {}
        for report in reports:
            for page_type, count in report.page_type_breakdown.items():
                page_type_totals[page_type] = page_type_totals.get(page_type, 0) + count
        
        # Quality breakdown
        quality_totals = {} 
        for report in reports:
            for quality, count in report.quality_breakdown.items():
                quality_totals[quality] = quality_totals.get(quality, 0) + count
        
        summary_lines = [
            "=== ENHANCED PDF EXTRACTION SUMMARY v0.6.0 ===",
            f"PDFs Processed: {successful_pdfs}/{total_pdfs}",
            f"Total Processing Time: {total_time:.2f}s",
            "",
            "SMART ANALYSIS RESULTS:",
            f"  Images Extracted: {total_images}",
            f"  Double-Page Spreads Joined: {total_joined}",
            f"  Images Enhanced: {total_enhanced}",
            f"  Save Mode: {save_options}",
            "",
            "PAGE TYPE DETECTION:"
        ]
        
        for page_type, count in page_type_totals.items():
            if count > 0:
                summary_lines.append(f"  {page_type}: {count}")
        
        summary_lines.extend([
            "",
            "IMAGE QUALITY BREAKDOWN:"
        ])
        
        for quality, count in quality_totals.items():
            if count > 0:
                summary_lines.append(f"  {quality}: {count}")
        
        summary_lines.extend([
            "",
            "TEXT EXTRACTION:",
            f"  Pages with Text: {total_text_pages}",
            "",
            "INDIVIDUAL PDF RESULTS:"
        ])
        
        for report in reports:
            enhanced_info = f" ({report.images_enhanced} enhanced)" if report.images_enhanced > 0 else ""
            joined_info = f" ({report.images_joined} joined)" if report.images_joined > 0 else ""
            summary_lines.append(f"  • {report.pdf_filename}: {report.images_extracted} images{enhanced_info}{joined_info}, {report.text_extracted_pages} text pages")
        
        return "\n".join(summary_lines)
    
    def _create_processing_stats(self, reports):
        """Create detailed processing statistics"""
        if not reports:
            return "No processing statistics available"
        
        total_time = sum(r.processing_time for r in reports)
        avg_time = total_time / len(reports)
        
        stats = {
            "total_processing_time": f"{total_time:.2f}s",
            "average_time_per_pdf": f"{avg_time:.2f}s",
            "total_pdfs": len(reports),
            "kornia_gpu_available": KORNIA_AVAILABLE and torch.cuda.is_available(),
            "opencv_available": CV2_AVAILABLE
        }
        
        return json.dumps(stats, indent=2)




class GroundingDINOModel:
    def __init__(self, config_path: str, model_path: str, device: str, debug_mode: bool = False):
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        self.debug_mode = debug_mode  # ADD this line
        self.model = None
        self.transform = None
        
        try:
            # Use the exact same approach as the working segment_anything node
            dino_model_args = local_groundingdino_SLConfig.fromfile(config_path)  # FIXED: Use correct name
            
            if dino_model_args is None:
                raise ValueError(f"Failed to load config from {config_path}")

            # Handle BERT path exactly like the working node
            if hasattr(dino_model_args, 'text_encoder_type') and dino_model_args.text_encoder_type == 'bert-base-uncased':
                dino_model_args.text_encoder_type = get_bert_base_uncased_model_path_helper()
                if self.debug_mode:
                    print(f"GroundingDINOModel: Set BERT path to {dino_model_args.text_encoder_type}")
            
            # Build model exactly like the working node
            dino = local_groundingdino_build_model(dino_model_args)  # FIXED: Use correct name
            if dino is None:
                raise ValueError("Failed to build model")
            
            # Load checkpoint exactly like the working node
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Extract model state dict - handle different formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Load state dict with strict=False like the working node
            load_result = dino.load_state_dict(local_groundingdino_clean_state_dict(state_dict), strict=False)  # FIXED: Use correct name    
            
            if self.debug_mode:
                print(f"GroundingDINOModel: Load result - Missing: {len(load_result.missing_keys)}, Unexpected: {len(load_result.unexpected_keys)}")
            
            # Move to device
            self.model = dino.to(device=self.device)
            self.model.eval()
            
            if self.debug_mode:
                print("GroundingDINOModel: Model loaded successfully")

        except Exception as e:
            if self.debug_mode:
                print(f"GroundingDINOModel: Error loading model: {e}")
                import traceback
                traceback.print_exc()
            raise

    def predict(self, image_pil: Image.Image, text_prompt: str, 
                box_threshold: float, text_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use the exact same prediction approach as the working segment_anything node"""
        if self.model is None:
            return torch.empty((0, 4), device="cpu"), torch.empty((0,), device="cpu")

        # Now this will work because T.RandomResize exists in local_groundingdino.datasets.transforms
        def load_dino_image(image_pil):
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),  # This will work now!
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image, _ = transform(image_pil, None)  # This signature is correct for the local transforms
            return image

        def get_grounding_output(model, image, caption, box_threshold):
            caption = caption.lower().strip()
            if not caption.endswith("."):
                caption = caption + "."
            
            image = image.to(self.device)
            with torch.no_grad():
                outputs = model(image[None], captions=[caption])
            
            logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
            boxes = outputs["pred_boxes"][0]  # (nq, 4)
            
            # Filter output
            filt_mask = logits.max(dim=1)[0] > box_threshold
            boxes_filt = boxes[filt_mask]  # num_filt, 4
            
            return boxes_filt.cpu()

        try:
            dino_image = load_dino_image(image_pil.convert("RGB"))
            boxes_filt = get_grounding_output(self.model, dino_image, text_prompt, box_threshold)
            
            # Convert boxes from cxcywh to xyxy format
            H, W = image_pil.size[1], image_pil.size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
            
            # Create dummy scores
            scores = torch.ones(boxes_filt.size(0)) * box_threshold
            
            if self.debug_mode:
                print(f"GroundingDINOModel: Found {boxes_filt.size(0)} boxes")
            
            return boxes_filt, scores

        except Exception as e:
            if self.debug_mode:
                print(f"GroundingDINOModel: Prediction error: {e}")
            return torch.empty((0, 4), device="cpu"), torch.empty((0,), device="cpu")

class ContentAwareBorderDetector:
    """Advanced border detection using multiple CV2 techniques"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
    
    def detect_content_borders(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Detect actual content borders using multiple methods"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        methods = [
            self._detect_by_edge_density,
            self._detect_by_color_variance,
            self._detect_by_histogram_analysis,
            self._detect_by_morphological_operations
        ]
        
        candidates = []
        for method in methods:
            try:
                result = method(gray, image.size)
                if result:
                    candidates.append(result)
            except Exception as e:
                if self.debug_mode:
                    print(f"Border detection method failed: {e}")
        
        if candidates:
            # Vote on the best bounding box
            return self._consensus_bounding_box(candidates, image.size)
        
        return None
    
    def _detect_by_edge_density(self, gray: np.ndarray, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Detect content borders by analyzing edge density"""
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Divide image into grid and calculate edge density
        h, w = gray.shape
        grid_size = 20
        rows, cols = h // grid_size, w // grid_size
        
        edge_density_map = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                y1, y2 = i * grid_size, min((i + 1) * grid_size, h)
                x1, x2 = j * grid_size, min((j + 1) * grid_size, w)
                
                region = edges[y1:y2, x1:x2]
                edge_density_map[i, j] = np.sum(region > 0) / region.size
        
        # Find content region (areas with significant edge density)
        threshold = np.mean(edge_density_map) + np.std(edge_density_map) * 0.5
        content_mask = edge_density_map > threshold
        
        if np.any(content_mask):
            # Find bounding box of content regions
            content_rows, content_cols = np.where(content_mask)
            
            top = max(0, content_rows.min() * grid_size - grid_size)
            bottom = min(h, (content_rows.max() + 1) * grid_size + grid_size)
            left = max(0, content_cols.min() * grid_size - grid_size)
            right = min(w, (content_cols.max() + 1) * grid_size + grid_size)
            
            return (left, top, right, bottom)
        
        return None
    
    def _detect_by_color_variance(self, gray: np.ndarray, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Detect borders by analyzing color variance"""
        h, w = gray.shape
        
        # Calculate variance for each row and column
        row_variance = np.var(gray, axis=1)
        col_variance = np.var(gray, axis=0)
        
        # Find content boundaries based on variance thresholds
        var_threshold = np.mean(row_variance) * 0.3
        
        # Find top and bottom boundaries
        content_rows = np.where(row_variance > var_threshold)[0]
        if len(content_rows) == 0:
            return None
        
        top = max(0, content_rows[0] - 20)
        bottom = min(h, content_rows[-1] + 20)
        
        # Find left and right boundaries
        content_cols = np.where(col_variance > var_threshold)[0]
        if len(content_cols) == 0:
            return None
        
        left = max(0, content_cols[0] - 20)
        right = min(w, content_cols[-1] + 20)
        
        return (left, top, right, bottom)
    
    def _detect_by_histogram_analysis(self, gray: np.ndarray, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Detect borders using histogram analysis for uniform background detection"""
        h, w = gray.shape
        border_width = min(50, min(h, w) // 10)
        
        # Analyze border regions
        top_border = gray[:border_width, :]
        bottom_border = gray[-border_width:, :]
        left_border = gray[:, :border_width]
        right_border = gray[:, -border_width:]
        
        # Calculate histogram peaks for each border
        borders = {
            'top': (top_border, 0, border_width),
            'bottom': (bottom_border, h - border_width, h),
            'left': (left_border, 0, border_width),
            'right': (right_border, w - border_width, w)
        }
        
        crop_bounds = [0, 0, w, h]  # left, top, right, bottom
        
        for border_name, (border_region, start_pos, end_pos) in borders.items():
            hist = cv2.calcHist([border_region], [0], None, [256], [0, 256])
            
            # Check if border is uniform (has a dominant color)
            peak_value = np.max(hist)
            total_pixels = border_region.size
            uniformity = peak_value / total_pixels
            
            if uniformity > 0.7:  # Highly uniform border
                if border_name == 'top':
                    crop_bounds[1] = end_pos
                elif border_name == 'bottom':
                    crop_bounds[3] = start_pos
                elif border_name == 'left':
                    crop_bounds[0] = end_pos
                elif border_name == 'right':
                    crop_bounds[2] = start_pos
        
        # Validate bounds
        if crop_bounds[2] > crop_bounds[0] and crop_bounds[3] > crop_bounds[1]:
            return tuple(crop_bounds)
        
        return None
    
    def _detect_by_morphological_operations(self, gray: np.ndarray, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Use morphological operations to find content regions"""
        # Apply threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the main content)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add margin
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(gray.shape[1] - x, w + 2 * margin)
            h = min(gray.shape[0] - y, h + 2 * margin)
            
            return (x, y, x + w, y + h)
        
        return None
    
    def _consensus_bounding_box(self, candidates: List[Tuple[int, int, int, int]], 
                               image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Find consensus bounding box from multiple detection methods"""
        if len(candidates) == 1:
            return candidates[0]
        
        # Calculate median bounds for robustness
        lefts = [c[0] for c in candidates]
        tops = [c[1] for c in candidates]
        rights = [c[2] for c in candidates]
        bottoms = [c[3] for c in candidates]
        
        # Use median to avoid outliers
        consensus_left = int(np.median(lefts))
        consensus_top = int(np.median(tops))
        consensus_right = int(np.median(rights))
        consensus_bottom = int(np.median(bottoms))
        
        # Ensure valid bounds
        w, h = image_size
        consensus_left = max(0, min(consensus_left, w - 1))
        consensus_top = max(0, min(consensus_top, h - 1))
        consensus_right = max(consensus_left + 1, min(consensus_right, w))
        consensus_bottom = max(consensus_top + 1, min(consensus_bottom, h))
        
        if self.debug_mode:
            print(f"Consensus from {len(candidates)} methods: ({consensus_left}, {consensus_top}, {consensus_right}, {consensus_bottom})")
        
        return (consensus_left, consensus_top, consensus_right, consensus_bottom)

class StreamingPDFProcessor:
    """Streaming processor for handling large PDFs efficiently"""
    
    def __init__(self, config: ProcessingConfig, debug_mode: bool = False):
        self.config = config
        self.debug_mode = debug_mode
        self.memory_monitor = MemoryMonitor(debug_mode)

    def _should_use_streaming(self, pdf_path: Path) -> bool:
        """Determine if streaming processing should be used - FIXED: Better thresholds"""
        try:
            # Check file size - FIXED: Higher threshold
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 200:  # Files larger than 200MB (was 50MB)
                if self.debug_mode:
                    print(f"    Using streaming: Large file ({file_size_mb:.1f}MB)")
                return True
            
            # Quick page count check - FIXED: Higher threshold
            doc = fitz.open(str(pdf_path))
            page_count = doc.page_count
            doc.close()
            
            if page_count > 200:  # PDFs with more than 200 pages (was 100)
                if self.debug_mode:
                    print(f"    Using streaming: Many pages ({page_count})")
                return True
            
            # Check available system memory - FIXED: More realistic threshold
            if PSUTIL_AVAILABLE:
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                available_ram_gb = psutil.virtual_memory().available / (1024**3)
                
                # Only use streaming if we have less than 4GB available AND less than 8GB total
                if available_ram_gb < 4 and total_ram_gb < 8:
                    if self.debug_mode:
                        print(f"    Using streaming: Low available RAM ({available_ram_gb:.1f}GB available, {total_ram_gb:.1f}GB total)")
                    return True
            
            if self.debug_mode:
                print(f"    Using regular processing: File={file_size_mb:.1f}MB, Pages={page_count}")
            return False
            
        except Exception:
            # If we can't determine, use regular processing for smaller files
            if self.debug_mode:
                print("    Using streaming: Could not determine file characteristics")
            return True

    def process_pdf_streaming(self, pdf_path: Path, processor: 'EnhancedPDFProcessor', 
                            extract_images: bool, extract_text: bool, 
                            save_options: str, enable_smart_crop: bool, enhancement_strength: float,
                            join_double_pages: bool, use_gpu_acceleration: bool) -> ProcessingReport:
        """Process PDF in memory-efficient streaming chunks"""
        
        # Use the same report creation method
        report = processor._create_initial_report(pdf_path)
        
        doc = fitz.open(str(pdf_path))
        report.total_pages = doc.page_count
        
        try:
            # Determine optimal chunk size based on system capabilities
            chunk_size = self._calculate_optimal_chunk_size(doc.page_count)
            current_page = 0  # Track current page position
            
            if self.debug_mode:
                print(f"📊 Processing {doc.page_count} pages in chunks of {chunk_size}")
                print(f"💾 Memory optimization: {self.config.memory_optimization}")
            
            # Process in chunks
            while current_page < doc.page_count:
                chunk_end = min(current_page + chunk_size, doc.page_count)
                
                if self.debug_mode:
                    memory_before = self.memory_monitor.get_memory_usage()
                    print(f"\n📄 Processing chunk: pages {current_page + 1}-{chunk_end}")
                    print(f"💾 Memory before chunk: {memory_before['ram_mb']:.1f}MB RAM, {memory_before['gpu_mb']:.1f}MB GPU")
                
                # Process chunk with memory monitoring
                with self._chunk_processing_context():
                    chunk_result = self._process_chunk(
                        doc, current_page, chunk_end, processor, report,
                        extract_images, extract_text, save_options, enable_smart_crop,
                        enhancement_strength, use_gpu_acceleration
                    )
                
                # Merge chunk results
                self._merge_chunk_results(report, chunk_result)
                
                if self.debug_mode:
                    memory_after = self.memory_monitor.get_memory_usage()
                    print(f"💾 Memory after chunk: {memory_after['ram_mb']:.1f}MB RAM, {memory_after['gpu_mb']:.1f}MB GPU")
                    
                    # FIXED: Much higher memory thresholds
                    # Check actual memory increase, not total system memory
                    memory_increase = memory_after['ram_mb'] - memory_before['ram_mb']
                    
                    # Only reduce chunk size if we're seeing significant memory growth per chunk
                    if memory_increase > 2000:  # More than 2GB increase per chunk
                        old_chunk_size = chunk_size
                        chunk_size = max(1, chunk_size // 2)
                        if self.debug_mode:
                            print(f"⚠️ Large memory increase detected ({memory_increase:.1f}MB), reducing chunk size from {old_chunk_size} to {chunk_size}")
                    elif memory_after['ram_mb'] > 60000:  # Only worry if using more than 60GB total
                        old_chunk_size = chunk_size
                        chunk_size = max(1, chunk_size // 2)
                        if self.debug_mode:
                            print(f"⚠️ Very high total memory usage detected ({memory_after['ram_mb']:.1f}MB), reducing chunk size from {old_chunk_size} to {chunk_size}")
                
                # Update current_page to continue from where we left off
                current_page = chunk_end
                
                # Force cleanup between chunks
                self._cleanup_between_chunks()
            
            # Post-process for double-page joining if requested
            if extract_images and join_double_pages and len(report.extracted_images) > 1:
                self._streaming_double_page_joining(report, processor)
            
        finally:
            doc.close()
        
        return report

    def _calculate_optimal_chunk_size(self, total_pages: int) -> int:
        """Calculate optimal chunk size based on system resources"""
        base_chunk_size = self.config.batch_size
        
        # Adjust based on total pages
        if total_pages > 1000:
            base_chunk_size = max(5, base_chunk_size // 2)  # Larger minimum for huge PDFs
        elif total_pages > 500:
            base_chunk_size = max(8, int(base_chunk_size // 1.5))
        
        # Adjust based on available memory - FIXED: More realistic thresholds
        memory_info = self.memory_monitor.get_memory_usage()
        
        # Get total system memory to make better decisions
        total_ram_gb = 32  # Default assumption
        if PSUTIL_AVAILABLE:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if total_ram_gb >= 32:  # High-end system (32GB+)
            base_chunk_size = min(25, base_chunk_size * 2)
            if self.debug_mode:
                print(f"    High-RAM system detected ({total_ram_gb:.1f}GB), using larger chunks: {base_chunk_size}")
        elif total_ram_gb >= 16:  # Mid-range system (16-32GB)
            base_chunk_size = min(20, int(base_chunk_size * 1.5))
            if self.debug_mode:
                print(f"    Mid-RAM system detected ({total_ram_gb:.1f}GB), using standard chunks: {base_chunk_size}")
        elif total_ram_gb < 8:  # Low RAM system (less than 8GB)
            base_chunk_size = max(3, base_chunk_size // 2)
            if self.debug_mode:
                print(f"    Low-RAM system detected ({total_ram_gb:.1f}GB), using smaller chunks: {base_chunk_size}")
        
        # Adjust based on GPU memory - FIXED: More realistic thresholds
        if memory_info['gpu_mb'] < 4000:  # Less than 4GB VRAM
            base_chunk_size = max(5, base_chunk_size // 2)
            if self.debug_mode:
                print(f"    Low VRAM detected ({memory_info['gpu_mb']:.1f}MB), reducing chunks: {base_chunk_size}")
        
        return int(base_chunk_size)

    @contextmanager
    def _chunk_processing_context(self):
        """Context manager for chunk processing with cleanup"""
        try:
            yield
        finally:
            # Force cleanup after each chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _process_chunk(self, doc, start_page: int, end_page: int, 
                    processor: 'EnhancedPDFProcessor', global_report: ProcessingReport,
                    extract_images: bool, extract_text: bool, save_options: str, 
                    enable_smart_crop: bool, enhancement_strength: float, 
                    use_gpu_acceleration: bool) -> Dict:
        """Process a single chunk of pages"""
        
        if self.debug_mode:
            print(f"    📋 Processing chunk: pages {start_page + 1} to {end_page}")
        
        chunk_result = {
            "images": [],
            "text": [],
            "images_processed": 0,
            "images_filtered": 0,
            "text_pages": 0
        }
        
        # Collect all images from chunk for batch analysis
        chunk_images = []
        
        for page_num in range(start_page, end_page):
            if self.debug_mode:
                print(f"      📄 Processing page {page_num + 1}")
            
            page = doc[page_num]
            
            # Extract text
            if extract_text:
                text_result = processor.safe_execute(
                    processor._extract_text_pymupdf,
                    page, page_num, global_report,
                    fallback_result=None,
                    context=f"text extraction page {page_num + 1}"
                )
                if text_result:
                    chunk_result["text_pages"] += 1
            
            # Extract images for analysis
            if extract_images:
                page_images = processor.safe_execute(
                    self._extract_page_images_chunk,
                    page, page_num, global_report,
                    fallback_result=[],
                    context=f"image extraction page {page_num + 1}"
                )
                if self.debug_mode and page_images:
                    print(f"        🖼️  Found {len(page_images)} images on page {page_num + 1}")
                chunk_images.extend(page_images)
        
        # Batch analyze all images from this chunk
        if chunk_images:
            if self.debug_mode:
                print(f"    🔍 Batch analyzing {len(chunk_images)} images from chunk")
            
            chunk_result = self._process_chunk_images(
                chunk_images, processor, chunk_result, save_options,
                enable_smart_crop, enhancement_strength, use_gpu_acceleration
            )
        
        if self.debug_mode:
            print(f"    ✅ Chunk complete: {chunk_result['images_processed']} images processed, {chunk_result['images_filtered']} filtered")
        
        return chunk_result

    def _extract_page_images_chunk(self, page, page_num: int, 
                                  report: ProcessingReport) -> List[Dict]:
        """Extract images from page with streaming optimizations"""
        page_images = []
        image_list = page.get_images(full=True)
        
        # Limit processing to avoid memory issues
        max_images_per_page = min(self.config.max_images_per_page, 5)
        processed_count = 0
        seen_xrefs = set()
        
        for img_index, img in enumerate(image_list):
            if processed_count >= max_images_per_page:
                break
            
            try:
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                
                # Quick size check before extracting
                try:
                    pix = fitz.Pixmap(page.parent, xref)
                    if not self._quick_size_check(pix.width, pix.height):
                        pix = None
                        report.images_filtered_out += 1
                        continue
                    
                    # Convert and store with minimal processing
                    image_data = self._convert_pixmap_minimal(pix, page_num, img_index, xref)
                    if image_data:
                        page_images.append(image_data)
                        processed_count += 1
                        seen_xrefs.add(xref)
                    
                    pix = None
                    
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                    report.images_filtered_out += 1
                    continue
            
            except Exception as e:
                if self.debug_mode:
                    print(f"Error processing image reference {img_index}: {e}")
                continue
        
        return page_images

    def _quick_size_check(self, width: int, height: int) -> bool:
        """Quick size validation without full processing"""
        return (width >= self.config.min_image_size and 
                height >= self.config.min_image_size and
                width <= self.config.max_image_width and
                height <= self.config.max_image_height and
                width * height >= self.config.min_image_area)

    def _convert_pixmap_minimal(self, pix, page_num: int, img_index: int, xref: int) -> Optional[Dict]:
        """Convert pixmap with minimal memory footprint"""
        try:
            # Handle colorspace
            if pix.colorspace and pix.colorspace.n == 4:
                old_pix = pix
                pix = fitz.Pixmap(fitz.csRGB, pix)
                old_pix = None
            
            # Convert to PIL efficiently
            import io
            img_data = pix.tobytes("png")  # Use PNG for better compression
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Store minimal data
            result = {
                "image": pil_image,
                "page_num": page_num,
                "image_index": img_index,
                "width": pix.width,
                "height": pix.height,
                "xref": xref,
                "colorspace": str(pix.colorspace) if pix.colorspace else "RGB"
            }
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error converting pixmap: {e}")
            return None

    def _process_chunk_images(self, chunk_images: List[Dict], 
                            processor: 'EnhancedPDFProcessor', chunk_result: Dict,
                            save_options: str, enable_smart_crop: bool, enhancement_strength: float,
                            use_gpu_acceleration: bool) -> Dict:
        """Process all images from a chunk efficiently"""
        
        if not chunk_images:
            return chunk_result
        
        # Batch analyze for efficiency
        try:
            images_for_analysis = [img_data["image"] for img_data in chunk_images]
            
            if self.debug_mode:
                print(f"      🧠 Running analysis on {len(images_for_analysis)} images")
            
            # Use detector with smaller batch if memory constrained
            if len(images_for_analysis) > 10:
                # Process in sub-batches
                analyses = []
                for i in range(0, len(images_for_analysis), 5):
                    sub_batch = images_for_analysis[i:i+5]
                    if self.debug_mode:
                        print(f"        🔍 Sub-batch {i//5 + 1}: analyzing {len(sub_batch)} images")
                    try:
                        sub_analyses = processor.detector.analyze_images(sub_batch)
                        analyses.extend(sub_analyses)
                        if self.debug_mode:
                            print(f"        ✅ Sub-batch {i//5 + 1} completed successfully")
                    except Exception as e:
                        if self.debug_mode:
                            print(f"        ❌ Sub-batch {i//5 + 1} failed: {e}")
                        # Create dummy analyses for failed sub-batch
                        dummy_analyses = [{
                            "page_type": PageType.UNCERTAIN,
                            "confidence": 0.0,
                            "vision_result": {},
                            "border_result": None,
                            "geometric_result": None,
                            "double_page_result": None,
                            "segmentation_data": None,
                            "image_index": i + j,
                            "image_size": (100, 100)
                        } for j in range(len(sub_batch))]
                        analyses.extend(dummy_analyses)
            else:
                try:
                    analyses = processor.detector.analyze_images(images_for_analysis)
                    if self.debug_mode:
                        print(f"      ✅ Analysis complete, processing {len(analyses)} results")
                except Exception as e:
                    if self.debug_mode:
                        print(f"      ❌ Main analysis failed: {e}")
                    # Create dummy analyses for all images
                    analyses = [{
                        "page_type": PageType.UNCERTAIN,
                        "confidence": 0.0,
                        "vision_result": {},
                        "border_result": None,
                        "geometric_result": None,
                        "double_page_result": None,
                        "segmentation_data": None,
                        "image_index": i,
                        "image_size": img_data["image"].size if img_data.get("image") else (100, 100)
                    } for i, img_data in enumerate(chunk_images)]
            
            # Process each image
            for img_data, analysis in zip(chunk_images, analyses):
                try:
                    result = processor.safe_execute(
                        self._process_single_chunk_image,
                        img_data, analysis, processor, save_options, enable_smart_crop,
                        enhancement_strength, use_gpu_acceleration,
                        fallback_result=None,
                        context=f"processing image {img_data['page_num']+1}_{img_data['image_index']+1}"
                    )
                    
                    if result:
                        chunk_result["images"].append(result)
                        chunk_result["images_processed"] += 1
                        if self.debug_mode:
                            print(f"        ✅ Processed image from page {img_data['page_num']+1}")
                    else:
                        chunk_result["images_filtered"] += 1
                        if self.debug_mode:
                            print(f"        🗑️  Filtered image from page {img_data['page_num']+1}")
                            
                except Exception as e:
                    if self.debug_mode:
                        print(f"        ❌ Error processing chunk image from page {img_data['page_num']+1}: {e}")
                    chunk_result["images_filtered"] += 1
                finally:
                    # Clean up image immediately after processing
                    if img_data.get("image"):
                        try:
                            img_data["image"].close()
                        except:
                            pass
                        del img_data["image"]
            
        except Exception as e:
            if self.debug_mode:
                print(f"      ❌ Batch analysis failed: {e}")
                import traceback
                traceback.print_exc()
            chunk_result["images_filtered"] += len(chunk_images)
        
        return chunk_result

    def _process_single_chunk_image(self, img_data: Dict, analysis: Dict,
                                processor: 'EnhancedPDFProcessor', save_options: str,
                                enable_smart_crop: bool, enhancement_strength: float,
                                use_gpu_acceleration: bool) -> Optional[ExtractedImage]:
        """Process a single image from chunk"""
        
        image = img_data["image"]
        page_num = img_data["page_num"]
        img_index = img_data["image_index"]
        
        # Apply confidence filtering
        if analysis["page_type"] == PageType.UNCERTAIN and analysis["confidence"] < 0.15:
            return None
        
        # ADD: Save segmentation masks if available and debug mode enabled
        if processor.config.save_debug_images and analysis.get("segmentation_data"):
            processor._save_segmentation_masks(analysis["segmentation_data"], page_num, img_index)
        
        # Apply cropping
        if enable_smart_crop:
            processed_image = processor._apply_smart_cropping(image, analysis)
        else:
            processed_image = image

        # Save and enhance
        saved_images = processor._save_enhanced_images(
            image, processed_image, page_num, img_index, save_options,
            enhancement_strength, use_gpu_acceleration
        )
        
        if saved_images:
            # Return the first saved image info (enhanced version if available)
            filename, saved_img, is_enhanced = saved_images[0]
            
            quality = processor._assess_image_quality(saved_img)
            file_size = processor._get_file_size(filename)
            
            return ExtractedImage(
                filename=filename,
                page_num=page_num,
                image_index=img_index,
                width=saved_img.width,
                height=saved_img.height,
                file_size_bytes=file_size,
                quality_score=quality,
                page_type=analysis["page_type"],
                border_confidence=analysis["confidence"],
                is_multi_page_candidate=processor._is_enhanced_multi_page_candidate(
                    saved_img, analysis["page_type"]
                ),
                bbox=(0, 0, saved_img.width, saved_img.height),
                original_colorspace=img_data["colorspace"],
                extraction_method=f"Streaming_{'enhanced' if is_enhanced else 'raw'}"
            )
        
        return None

    def _merge_chunk_results(self, global_report: ProcessingReport, chunk_result: Dict):
        """Merge chunk results into global report"""
        
        # Add images
        global_report.extracted_images.extend(chunk_result["images"])
        global_report.images_extracted += chunk_result["images_processed"]
        global_report.images_filtered_out += chunk_result["images_filtered"]
        global_report.text_extracted_pages += chunk_result["text_pages"]
        
        # Update quality and page type breakdowns
        for extracted_img in chunk_result["images"]:
            global_report.quality_breakdown[extracted_img.quality_score.value] += 1
            global_report.page_type_breakdown[extracted_img.page_type.value] += 1
            
            if extracted_img.is_multi_page_candidate:
                global_report.multi_page_candidates.append(extracted_img.filename)

    def _cleanup_between_chunks(self):
        """Aggressive cleanup between chunks"""
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Additional cleanup for transformers models
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

    def _streaming_double_page_joining(self, report: ProcessingReport, 
                                     processor: 'EnhancedPDFProcessor'):
        """Memory-efficient double-page joining"""
        
        # Group potential pairs
        left_pages = [img for img in report.extracted_images 
                     if img.page_type == PageType.DOUBLE_PAGE_LEFT]
        right_pages = [img for img in report.extracted_images 
                      if img.page_type == PageType.DOUBLE_PAGE_RIGHT]
        
        if self.debug_mode:
            print(f"🔗 Streaming join: {len(left_pages)} left + {len(right_pages)} right pages")
        
        # Process pairs in small batches to manage memory
        for left_img in left_pages:
            right_candidates = [r for r in right_pages if r.page_num == left_img.page_num + 1]
            
            if not right_candidates:
                continue
            
            right_img = max(right_candidates, key=lambda x: x.border_confidence)
            
            if processor._validate_spread_pair_with_florence2(left_img, right_img):
                # Process join with memory monitoring
                join_result = processor.safe_execute(
                    processor._perform_enhanced_join,
                    left_img, right_img, report.output_directory,
                    fallback_result=None,
                    context=f"joining pages {left_img.page_num+1}-{right_img.page_num+1}"
                )
                
                if join_result:
                    report.joined_images.append(join_result)
                    report.images_joined += 1
                    
                    if self.debug_mode:
                        print(f"  ✅ Joined {left_img.filename} + {right_img.filename}")
                
                # Cleanup after each join
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

class MemoryMonitor:
    """Monitor system memory usage during processing"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.baseline_ram = None
        self.baseline_gpu = None
        
        # Establish baseline
        if PSUTIL_AVAILABLE:
            self.baseline_ram = psutil.virtual_memory().used / (1024 * 1024)
        if torch.cuda.is_available():
            self.baseline_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage with better context"""
        if not PSUTIL_AVAILABLE:
            return {"ram_mb": 0.0, "ram_percent": 0.0, "gpu_mb": 0.0, "ram_total_gb": 0.0}
        
        # RAM usage
        ram_info = psutil.virtual_memory()
        ram_used_mb = ram_info.used / (1024 * 1024)
        ram_total_gb = ram_info.total / (1024**3)
        
        # GPU memory usage
        gpu_used_mb = 0.0
        if torch.cuda.is_available():
            gpu_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Calculate increases from baseline
        ram_increase = 0.0
        gpu_increase = 0.0
        if self.baseline_ram:
            ram_increase = ram_used_mb - self.baseline_ram
        if self.baseline_gpu:
            gpu_increase = gpu_used_mb - self.baseline_gpu
        
        return {
            "ram_mb": ram_used_mb,
            "ram_percent": ram_info.percent,
            "gpu_mb": gpu_used_mb,
            "ram_total_gb": ram_total_gb,
            "ram_available_gb": ram_info.available / (1024**3),
            "ram_increase_mb": ram_increase,
            "gpu_increase_mb": gpu_increase
        }
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is at critical levels"""
        memory = self.get_memory_usage()
        
        # Critical if less than 2GB available RAM
        if memory['ram_available_gb'] < 2.0:
            return True
        
        # Critical if using more than 90% of total RAM
        if memory['ram_percent'] > 90:
            return True
        
        return False

class SegmentationAnalyzer:
    """Analyzes images using GroundingDINO and SAM for object segmentation."""
    def __init__(self, dino_config_name: Optional[str] = None, 
                sam_model_name: Optional[str] = None,
                device: Optional[str] = None, 
                debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.available = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.dino_model = None
        self.sam_model = None 
        self.sam_predictor = None
        self.config = None  # Will be set later
        
        try:
            # FIXED: Use SEGMENT_ANYTHING_AVAILABLE instead of SAM_LIB_AVAILABLE
            if not GROUNDINGDINO_LIB_AVAILABLE or not SEGMENT_ANYTHING_AVAILABLE:
                if self.debug_mode:
                    print("SegmentationAnalyzer: Required libraries not available")
                    print(f"  GROUNDINGDINO_LIB_AVAILABLE: {GROUNDINGDINO_LIB_AVAILABLE}")
                    print(f"  SEGMENT_ANYTHING_AVAILABLE: {SEGMENT_ANYTHING_AVAILABLE}")
                return
            
            # Load DINO model
            dino_config_path, dino_model_path = self._resolve_dino_paths(dino_config_name)
            if dino_config_path and dino_model_path:
                self.dino_model = GroundingDINOModel(
                    dino_config_path, dino_model_path, self.device, debug_mode
                )
                if self.debug_mode:
                    print(f"SegmentationAnalyzer: GroundingDINO model loaded successfully")
            
            # Load SAM model  
            sam_model_path = self._resolve_sam_path(sam_model_name)
            if sam_model_path:
                self.sam_model = self._load_sam_model(sam_model_path)
                if self.sam_model:
                    # IMPORTANT: Ensure SAM model is on the correct device
                    self.sam_model = self.sam_model.to(self.device)
                    
                    # Determine if it's an HQ model
                    sam_is_hq = 'hq' in Path(sam_model_path).name.lower()
                    self.sam_predictor = SamPredictorHQ(self.sam_model, sam_is_hq)
                    
                    if self.debug_mode:
                        print(f"SegmentationAnalyzer: SAM model loaded successfully (HQ: {sam_is_hq}, Device: {self.device})")
            
            # Check if both models loaded successfully
            if self.dino_model and self.sam_model and self.sam_predictor:
                self.available = True
                if self.debug_mode:
                    print("SegmentationAnalyzer: Initialization successful - all models available")
            else:
                if self.debug_mode:
                    print("SegmentationAnalyzer: Initialization failed - some models unavailable")
                    print(f"  dino_model: {self.dino_model is not None}")
                    print(f"  sam_model: {self.sam_model is not None}")
                    print(f"  sam_predictor: {self.sam_predictor is not None}")
        
        except Exception as e:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Initialization error: {e}")
                traceback.print_exc()
            self.available = False


    def _resolve_dino_config_path(self, dino_config_name: Optional[str]) -> Optional[str]:
        if not dino_config_name or dino_config_name.lower() == "autodetect" or dino_config_name.lower() == "none":
            preferred_configs = [
                "GroundingDINO_SwinT_OGC.cfg.py", 
                "GroundingDINO_SwinB.cfg.py"
            ]
            user_dino_dir = Path(folder_paths.models_dir) / "grounding-dino" 
            if user_dino_dir.exists(): # Check if base directory exists
                for cfg_name in preferred_configs:
                    path = user_dino_dir / cfg_name
                    if path.exists():
                        if self.debug_mode: print(f"SegmentationAnalyzer: Autodetected DINO config: {path}")
                        return str(path)
            if self.debug_mode: print("SegmentationAnalyzer: Autodetect DINO config failed (user_dino_dir or files not found).")
            return None

        # User provided a specific name
        user_dino_dir = Path(folder_paths.models_dir) / "grounding-dino" # <--- CORRECTED THIS LINE
        if user_dino_dir.exists(): # Check if base directory exists
            config_path = user_dino_dir / dino_config_name
            if config_path.exists() and config_path.is_file():
                if self.debug_mode: print(f"SegmentationAnalyzer: Resolved DINO config: {config_path}")
                return str(config_path)
        
        if self.debug_mode: print(f"SegmentationAnalyzer: DINO config '{dino_config_name}' not found at {user_dino_dir}")
        return None



    def _find_local_dino_model(self, model_name_override: Optional[str], config_path: Optional[str]) -> Optional[str]:
        if model_name_override: # Check if model_name_override is not None or empty
            override_path = Path(model_name_override)
            if override_path.exists():
                if self.debug_mode: print(f"SegmentationAnalyzer: Using DINO model override: {model_name_override}")
                return model_name_override
        
        if not config_path: # Check if config_path is None or empty
            if self.debug_mode: print("SegmentationAnalyzer: DINO config path not available, cannot determine DINO model.")
            return None

        user_dino_dir = Path(folder_paths.models_dir) / "grounding-dino"
        
        # Determine model based on config
        config_filename = Path(config_path).name # This is safe now due to the check above
        expected_model_filename = None
        if "GroundingDINO_SwinB" in config_filename:
            expected_model_filename = "groundingdino_swinb_cogcoor.pth"
        elif "GroundingDINO_SwinT_OGC" in config_filename:
            expected_model_filename = "groundingdino_swint_ogc.pth"
        
        if expected_model_filename:
            model_path = user_dino_dir / expected_model_filename
            if model_path.exists():
                if self.debug_mode: print(f"SegmentationAnalyzer: Found DINO model at {model_path}")
                return str(model_path)
            else:
                if self.debug_mode: print(f"SegmentationAnalyzer: DINO model '{expected_model_filename}' not found at {user_dino_dir}")
        else:
            if self.debug_mode: print(f"SegmentationAnalyzer: Could not determine DINO model filename from config '{config_filename}'")

        # Fallback to general search if specific model not found (less likely needed with specific paths)
        generic_dino_files = ["groundingdino_swinb_cogcoor.pth", "groundingdino_swint_ogc.pth"]
        for fname in generic_dino_files:
            model_path = user_dino_dir / fname
            if model_path.exists():
                if self.debug_mode: print(f"SegmentationAnalyzer: Found DINO model (fallback search) at {model_path}")
                return str(model_path)
        
        if self.debug_mode: print(f"SegmentationAnalyzer: DINO model not found in {user_dino_dir}")
        return None

    def _find_local_sam_model(self, sam_model_name_from_config: Optional[str]) -> Optional[str]:
        user_sams_dir = Path(folder_paths.models_dir) / "sams"

        if not user_sams_dir.exists():
            if self.debug_mode: print(f"SegmentationAnalyzer: SAM models directory not found at {user_sams_dir}")
            return None

        # Handle specific model name provided by user
        if sam_model_name_from_config and sam_model_name_from_config.lower() != "autodetect" and sam_model_name_from_config.lower() != "none":
            specific_model_path = user_sams_dir / sam_model_name_from_config
            if specific_model_path.exists() and specific_model_path.is_file():
                if self.debug_mode: print(f"SegmentationAnalyzer: Using user-specified SAM model: {specific_model_path}")
                return str(specific_model_path)
            else:
                if self.debug_mode: print(f"SegmentationAnalyzer: User-specified SAM model '{sam_model_name_from_config}' not found at {specific_model_path}. Falling back to autodetect.")
                # Fall through to autodetect if specific model not found

        # Autodetect logic (if "autodetect", "none", or specific model not found)
        sam_filenames_priority = [
            "sam_hq_vit_h.pth", 
            "sam_vit_h_4b8939.pth", # Standard VIT-H
            "sam_hq_vit_l.pth", 
            "sam_vit_l_0b3195.pth", # Standard VIT-L
            "sam_hq_vit_b.pth", 
            "sam_vit_b_01ec64.pth", # Standard VIT-B
            "mobile_sam.pt",
        ]
        
        for fname in sam_filenames_priority:
            model_path = user_sams_dir / fname
            if model_path.exists():
                if self.debug_mode: print(f"SegmentationAnalyzer: Autodetected SAM model at {model_path}")
                return str(model_path)
        
        if self.debug_mode: print(f"SegmentationAnalyzer: No SAM model found in {user_sams_dir} (autodetect failed).")
        return None

    def _run_single_segmentation_pass(self, image: Image.Image, prompt: str, 
                                    segmentation_type: str = "image") -> Dict[str, Any]:
        """Run a single segmentation pass with GroundingDINO + SAM"""
        
        if not self.available:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Segmentation not available for prompt: '{prompt}'")
            return {"boxes_cxcywh_norm": None, "scores": None, "mask_combined": None}
        
        try:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Running {segmentation_type} segmentation pass with prompt: '{prompt}'")
            
            # Step 1: Get bounding boxes from GroundingDINO
            boxes_xyxy, scores = self.dino_model.predict(
                image, prompt, 
                self.config.segmentation_box_threshold,
                self.config.segmentation_text_threshold
            )
            
            if self.debug_mode:
                print(f"SegmentationAnalyzer: DINO found {boxes_xyxy.size(0)} boxes for '{prompt}'.")
            
            if boxes_xyxy.size(0) == 0:
                if self.debug_mode:
                    print(f"SegmentationAnalyzer: No boxes found for prompt: '{prompt}'")
                return {"boxes_cxcywh_norm": None, "scores": None, "mask_combined": None}
            
            # Step 2: Transform boxes for SAM (following the working node.py pattern)
            image_np = np.array(image)
            image_np_rgb = image_np[..., :3]
            
            # Set the image for SAM predictor
            self.sam_predictor.set_image(image_np_rgb)
            
            # Transform boxes to the format expected by SAM
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy, image_np.shape[:2]
            )
            
            # CRITICAL FIX: Ensure boxes are on the same device as SAM model
            sam_device = self.sam_predictor.model.device if hasattr(self.sam_predictor.model, 'device') else 'cuda'
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Moving boxes to device: {sam_device}")
            
            # Step 3: Run SAM segmentation with properly placed tensors
            sam_masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(sam_device),  # <- This is the key fix
                multimask_output=False
            )
            
            if self.debug_mode:
                print(f"SegmentationAnalyzer: SAM generated {sam_masks.size(0)} masks.")
            
            # Step 4: Process results (convert back to original coordinate space)
            sam_masks = sam_masks.permute(1, 0, 2, 3).cpu().numpy()  # Move back to CPU for processing
            
            # Convert boxes back to normalized cxcywh format
            boxes_cxcywh_norm = self._convert_boxes_to_cxcywh_norm(boxes_xyxy, image.size)
            
            # Combine all masks into a single mask
            combined_mask = self._combine_masks(sam_masks)
            
            return {
                "boxes_cxcywh_norm": boxes_cxcywh_norm,
                "scores": scores.cpu() if scores.is_cuda else scores,
                "mask_combined": combined_mask
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"SegmentationAnalyzer._run_single_segmentation_pass: Error during inference for prompt '{prompt}': {e}")
                traceback.print_exc()
            return {"boxes_cxcywh_norm": None, "scores": None, "mask_combined": None}
            

    def get_segmentation_results(self, image_pil: Image.Image, 
                                image_prompt: Optional[str] = None, 
                                text_prompt: Optional[str] = None,
                                box_threshold: float = 0.3, 
                                text_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Performs segmentation for images and/or text based on provided prompts.
        Returns a dictionary with keys like 'image_boxes', 'image_mask', 'text_boxes', 'text_mask'.
        """
        results = {
            "image_boxes_cxcywh_norm": None, "image_scores": None, "image_mask_combined": None,
            "text_boxes_cxcywh_norm": None, "text_scores": None, "text_mask_combined": None,
        }

        if not self.available:
            if self.debug_mode: print("SegmentationAnalyzer.get_segmentation_results: Analyzer not available.")
            return results

        if image_prompt:
            if self.debug_mode: print(f"SegmentationAnalyzer: Running image segmentation pass with prompt: '{image_prompt}'")
            # FIXED: Use the correct method signature and extract results properly
            img_result = self._run_single_segmentation_pass(image_pil, image_prompt, "image")
            results["image_boxes_cxcywh_norm"] = img_result.get("boxes_cxcywh_norm")
            results["image_scores"] = img_result.get("scores")
            results["image_mask_combined"] = img_result.get("mask_combined")
        
        if text_prompt:
            if self.debug_mode: print(f"SegmentationAnalyzer: Running text segmentation pass with prompt: '{text_prompt}'")
            # FIXED: Use the correct method signature and extract results properly
            txt_result = self._run_single_segmentation_pass(image_pil, text_prompt, "text")
            results["text_boxes_cxcywh_norm"] = txt_result.get("boxes_cxcywh_norm")
            results["text_scores"] = txt_result.get("scores")
            results["text_mask_combined"] = txt_result.get("mask_combined")
            
        return results


    def _get_sam_model_type_from_path(self, sam_model_path: Optional[str]) -> Optional[str]:
        if not sam_model_path: return None
        model_file_name = Path(sam_model_path).name
        # sam_hq_vit_h.pth -> sam_hq_vit_h
        # sam_vit_h_4b8939.pth -> sam_vit_h
        # mobile_sam.pt -> mobile_sam
        model_type = model_file_name.split('.')[0] 

        if 'hq' not in model_type and 'mobile' not in model_type:
            # This part ensures "sam_vit_h_4b8939" becomes "sam_vit_h"
            model_type = '_'.join(model_type.split('_')[:-1])
        
        if self.debug_mode:
            print(f"SegmentationAnalyzer: SAM model_file_name='{model_file_name}', derived model_type='{model_type}' for sam_hq_model_registry.")
        return model_type

    def _load_models(self):
        self.dino_model = GroundingDINOModel(
            config_path=self.dino_config_path,
            model_path=self.dino_model_path,
            device=self.device,
            debug_mode=self.debug_mode
        )
        
        try:
            # Use the sam_model_registry from the bundled sam_hq library
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_model_path)
            
            # The example node sets sam.model_name, which SamPredictorHQ uses to check if 'hq' is in the name.
            if not hasattr(sam, 'model_name'):
                sam.model_name = Path(self.sam_model_path).name

            sam.to(device=self.device)
            sam.eval()

            sam_is_hq = False
            if hasattr(sam, 'model_name') and 'hq' in sam.model_name:
                sam_is_hq = True
            
            self.sam_predictor = SamPredictorHQ(sam, sam_is_hq=sam_is_hq)
            if self.debug_mode: 
                print(f"SegmentationAnalyzer: SAM predictor (SamPredictorHQ, is_hq={sam_is_hq}) initialized with type '{self.sam_model_type}' from {self.sam_model_path}.")
        except KeyError as e:
            if self.debug_mode: 
                print(f"SegmentationAnalyzer: SAM model type '{self.sam_model_type}' not in sam_model_registry. Available keys: {list(sam_model_registry.keys())}. Error: {e}")
            raise
        except Exception as e:
            if self.debug_mode: 
                print(f"SegmentationAnalyzer: Failed to load SAM model: {e}")
            raise


    def get_mask(self, image_pil: Image.Image, prompt: str, 
                 box_threshold: float = 0.3, text_threshold: float = 0.25) -> Optional[torch.Tensor]:
        """Legacy method for single prompt, returns only the mask."""
        if self.debug_mode:
            print("SegmentationAnalyzer.get_mask (legacy) called. Consider using get_segmentation_results for richer output.")
        
        _, _, combined_mask = self._run_single_segmentation_pass(
            image_pil, prompt, box_threshold, text_threshold
        )
        return combined_mask

    def _resolve_dino_paths(self, dino_config_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Resolve GroundingDINO config and model paths"""
        config_path = self._resolve_dino_config_path(dino_config_name)
        model_path = self._find_local_dino_model(None, config_path)
        return config_path, model_path

    def _resolve_sam_path(self, sam_model_name: Optional[str]) -> Optional[str]:
        """Resolve SAM model path"""
        return self._find_local_sam_model(sam_model_name)

    def _load_sam_model(self, sam_model_path: str):
        """Load SAM model from path"""
        try:
            model_type = self._get_sam_model_type_from_path(sam_model_path)
            if model_type in sam_model_registry:
                sam_model = sam_model_registry[model_type](checkpoint=sam_model_path)
                sam_model.model_name = Path(sam_model_path).name
                return sam_model
            else:
                if self.debug_mode:
                    print(f"SegmentationAnalyzer: Unknown SAM model type: {model_type}")
                return None
        except Exception as e:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Error loading SAM model: {e}")
            return None

    def _convert_boxes_to_cxcywh_norm(self, boxes_xyxy: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Convert xyxy boxes to normalized cxcywh format"""
        if boxes_xyxy.size(0) == 0:
            return torch.empty((0, 4))
        
        width, height = image_size
        
        # Convert from xyxy to cxcywh
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # Normalize by image dimensions
        cx_norm = cx / width
        cy_norm = cy / height
        w_norm = w / width
        h_norm = h / height
        
        return torch.stack([cx_norm, cy_norm, w_norm, h_norm], dim=-1)

    def _combine_masks(self, sam_masks: np.ndarray) -> Optional[torch.Tensor]:
        """Combine multiple SAM masks into a single mask"""
        if sam_masks is None or sam_masks.size == 0:
            return None
        
        # sam_masks shape: (num_masks, height, width)
        # Combine by taking the logical OR of all masks
        combined_mask = np.any(sam_masks, axis=0)
        
        # Convert to tensor
        return torch.from_numpy(combined_mask.astype(np.float32))

    def set_config(self, config: ProcessingConfig):
        """Set the processing configuration"""
        self.config = config

def tensor_mask_to_pil(mask_tensor: torch.Tensor, mode='L') -> Optional[Image.Image]:
    """Converts a boolean or 0/1 tensor mask to a PIL Image."""
    if mask_tensor is None or not isinstance(mask_tensor, torch.Tensor) or mask_tensor.ndim < 2:
        return None
    
    # Ensure mask is 2D (H, W)
    if mask_tensor.ndim == 3:
        if mask_tensor.shape[0] == 1: # Squeeze channel dim if [1, H, W]
            mask_tensor = mask_tensor.squeeze(0)
        else: # If it's [C, H, W] with C > 1, this function might not be appropriate
              # or you might want to take a specific channel or combine them.
              # For typical SAM masks, it's [1, H, W] or [H, W].
            print(f"Warning: tensor_mask_to_pil received 3D tensor with shape {mask_tensor.shape}, taking first channel.")
            mask_tensor = mask_tensor[0] 
    
    if mask_tensor.ndim != 2:
        print(f"Warning: tensor_mask_to_pil received tensor with unexpected ndim {mask_tensor.ndim} after squeeze.")
        return None

    mask_np = mask_tensor.cpu().numpy()
    
    # Convert boolean or 0/1 float to 0/255 uint8
    if mask_np.dtype == bool:
        mask_np = mask_np.astype(np.uint8) * 255
    elif mask_np.dtype in [np.float32, np.float64, np.float16]:
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255 # Threshold if it's float
    elif mask_np.dtype != np.uint8: # If some other int type
        mask_np = (mask_np > 0).astype(np.uint8) * 255

    return Image.fromarray(mask_np, mode=mode)


# Helper function to convert box formats (cxcywh to xyxy)
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class EnhancedPDFProcessor:
    def __init__(self, output_dir: str, detector: SmartDoublePageDetector, 
                stitcher: ConstrainedStitcher, enhancer: KorniaGPUEnhancer, 
                config: ProcessingConfig = None, debug_mode: bool = False):
        self.output_dir = Path(output_dir)
        self.detector = detector
        self.stitcher = stitcher
        self.enhancer = enhancer if enhancer else KorniaGPUEnhancer(debug_mode=debug_mode)
        self.debug_mode = debug_mode
        self.use_pymupdf = PYMUPDF_AVAILABLE
        self.error_manager = ErrorRecoveryManager(debug_mode)
        self.config = config or ProcessingConfig()

        # Initialize Florence2 for cropping if available
        self.florence2_cropper = None
        if FLORENCE2_AVAILABLE and self.detector and self.detector.florence2_detector:
            self.florence2_cropper = self.detector.florence2_detector
            if debug_mode:
                print("✅ Florence2 cropper available for PDF processor")

        self.florence2_analysis_cache = {}  # Key: (page_num, image_index), Value: florence2_data
        
        if debug_mode:
            print("✅ Florence2 analysis cache initialized")

    def _try_florence2_cropping(self, image: Image.Image, analysis: Dict) -> Dict[str, Any]:
        """Try Florence2-based content cropping with intelligent full-page detection"""
        if not self.florence2_cropper:
            return {"success": False, "method": "florence2_unavailable"}
        
        try:
            # Use more specific prompts for better detection
            content_prompts = [
                "photograph OR main image OR illustration",
                "text block OR paragraph OR content area", 
                "diagram OR chart OR map OR figure",
                "rectangular content area"
            ]
            
            best_result = None
            best_box_count = 0
            
            for prompt in content_prompts:
                try:
                    bounding_boxes, mask_image = self.florence2_cropper.detect_rectangles(
                        image=image,
                        text_input=prompt,
                        return_mask=True,
                        keep_model_loaded=True
                    )
                    
                    if len(bounding_boxes) > best_box_count:
                        best_result = (bounding_boxes, mask_image)
                        best_box_count = len(bounding_boxes)
                        
                except Exception as e:
                    if self.debug_mode:
                        print(f"      Florence2 prompt '{prompt}' failed: {e}")
                    continue
            
            if best_result and best_box_count > 0:
                bounding_boxes, mask_image = best_result
                
                # NEW: Intelligent full-page vs content detection
                filtered_boxes, crop_strategy = self._filter_florence2_boxes(bounding_boxes, image.size)
                
                if filtered_boxes:
                    # Calculate crop box from filtered boxes
                    crop_box = self._calculate_florence2_crop_box(filtered_boxes, image.size)
                    
                    if crop_box:
                        if self.debug_mode:
                            print(f"      Florence2 strategy '{crop_strategy}': {len(filtered_boxes)}/{len(bounding_boxes)} boxes, crop: {crop_box}")
                        
                        return {
                            "success": True,
                            "method": f"florence2_{crop_strategy}",
                            "crop_box": crop_box,
                            "detected_rectangles": len(bounding_boxes),
                            "filtered_rectangles": len(filtered_boxes),
                            "confidence": min(1.0, 0.7 + (len(filtered_boxes) * 0.1)),  # Higher base confidence
                            "mask_image": mask_image,
                            "strategy": crop_strategy
                        }
                else:
                    if self.debug_mode:
                        print(f"      Florence2 found {len(bounding_boxes)} boxes but all were filtered out")
            
            return {"success": False, "method": "florence2_no_content"}
            
        except Exception as e:
            if self.debug_mode:
                print(f"      Florence2 cropping failed: {e}")
            return {"success": False, "method": "florence2_error", "error": str(e)}

    def _filter_florence2_boxes(self, bounding_boxes: List[BoundingBox], 
                               image_size: Tuple[int, int]) -> Tuple[List[BoundingBox], str]:
        """Filter Florence2 bounding boxes to handle full-page vs content detection"""
        if not bounding_boxes:
            return [], "no_boxes"
        
        width, height = image_size
        image_area = width * height
        
        # Categorize boxes by size
        full_page_boxes = []
        content_boxes = []
        
        # Define what constitutes a "full page" box (covers most of the image)
        full_page_threshold = 0.85  # 85% of image area
        
        for box in bounding_boxes:
            box_area_ratio = box.area / image_area
            
            # Check if box covers most of the image area
            if box_area_ratio >= full_page_threshold:
                full_page_boxes.append(box)
                if self.debug_mode:
                    print(f"        Full-page box: {box.to_tuple()}, area ratio: {box_area_ratio:.3f}")
            else:
                content_boxes.append(box)
                if self.debug_mode:
                    print(f"        Content box: {box.to_tuple()}, area ratio: {box_area_ratio:.3f}")
        
        # Decision logic based on box types found
        
        # Case 1: Only one box detected and it's full-page -> whole page is an image
        if len(bounding_boxes) == 1 and len(full_page_boxes) == 1:
            if self.debug_mode:
                print(f"        Strategy: single_full_page - only one box covers whole page")
            return [], "single_full_page"  # Don't crop, use whole image
        
        # Case 2: Two boxes where one is full-page and the other is small content with <95% area
        if len(bounding_boxes) == 2 and len(full_page_boxes) == 1 and len(content_boxes) == 1:
            content_box = content_boxes[0]
            full_page_box = full_page_boxes[0]
            
            # Check if content box is less than 95% of full-page box area
            content_vs_full_ratio = content_box.area / full_page_box.area
            
            if content_vs_full_ratio < 0.95:
                if self.debug_mode:
                    print(f"        Strategy: single_content_with_border - content box is {content_vs_full_ratio:.3f} of full page")
                return content_boxes, "single_content_with_border"  # Crop to content box only
            else:
                if self.debug_mode:
                    print(f"        Strategy: potential_collage - content box is {content_vs_full_ratio:.3f} of full page (too large)")
                # Could be a collage situation, use other methods
                return [], "potential_collage"
        
        # Case 3: Multiple boxes with at least one full-page -> discard full-page, keep content boxes
        if len(bounding_boxes) >= 2 and len(full_page_boxes) >= 1 and len(content_boxes) >= 1:
            if self.debug_mode:
                print(f"        Strategy: multi_content_ignore_fullpage - {len(content_boxes)} content boxes, ignoring {len(full_page_boxes)} full-page boxes")
            return content_boxes, "multi_content_ignore_fullpage"
        
        # Case 4: Only content boxes (no full-page detection) -> use all content boxes
        if len(content_boxes) > 0 and len(full_page_boxes) == 0:
            if self.debug_mode:
                print(f"        Strategy: pure_content - {len(content_boxes)} content boxes, no full-page detected")
            return content_boxes, "pure_content"
        
        # Case 5: Multiple full-page boxes (shouldn't happen but handle it)
        if len(full_page_boxes) > 1:
            if self.debug_mode:
                print(f"        Strategy: multiple_fullpage - {len(full_page_boxes)} full-page boxes detected, using largest")
            # Use the largest full-page box only
            largest_full_page = max(full_page_boxes, key=lambda x: x.area)
            return [largest_full_page], "multiple_fullpage"
        
        # Default case: return all boxes
        if self.debug_mode:
            print(f"        Strategy: default_all - using all {len(bounding_boxes)} boxes")
        return bounding_boxes, "default_all"


    def _calculate_florence2_crop_box(self, bounding_boxes: List[BoundingBox], 
                                    image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Calculate optimal crop box from Florence2 detected rectangles"""
        if not bounding_boxes:
            return None
        
        # Filter out very small boxes
        min_area = (image_size[0] * image_size[1]) * 0.01  # At least 1% of image
        valid_boxes = [box for box in bounding_boxes if box.area >= min_area]
        
        if not valid_boxes:
            return None
        
        # Find encompassing bounding box
        min_x = min(box.x1 for box in valid_boxes)
        min_y = min(box.y1 for box in valid_boxes)
        max_x = max(box.x2 for box in valid_boxes)
        max_y = max(box.y2 for box in valid_boxes)
        
        # Add margin
        margin = self.config.crop_margin
        width, height = image_size
        
        crop_x1 = max(0, min_x - margin)
        crop_y1 = max(0, min_y - margin)
        crop_x2 = min(width, max_x + margin)
        crop_y2 = min(height, max_y + margin)
        
        # Validate crop box
        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1
            
            # Ensure minimum size
            if crop_width >= self.config.min_image_size and crop_height >= self.config.min_image_size:
                return (crop_x1, crop_y1, crop_x2, crop_y2)
        
        return None

    def _analyze_page_content_enhanced(self, image: Image.Image, page_num: int, image_index: int) -> Dict[str, Any]:
        """Comprehensive analysis of a single image (extracted from a page)."""
        if self.debug_mode:
            print(f"  🔬 Analyzing content for page {page_num + 1}, image_idx {image_index} (size: {image.size})")
            print(f"    📋 Config check:")
            print(f"       - enable_image_segmentation: {self.config.enable_image_segmentation}")
            print(f"       - enable_text_segmentation: {self.config.enable_text_segmentation}")
            if self.segmentation_analyzer:
                print(f"       - segmentation_analyzer.available: {self.segmentation_analyzer.available}")
            else:
                print(f"       - segmentation_analyzer is None")

        analysis_results = {
            "page_type": PageType.UNCERTAIN, # Default
            "confidence": 0.0,
            "vision_result": None,
            "border_result": None,
            "geometric_result": None,
            "double_page_result": None,
            "segmentation_data": None, 
            "image_index": image_index, # Store image_index
            "image_page_num": page_num, # Store page_num
            "image_size": image.size
        }
        
        # --- ENHANCED DEBUG: Check segmentation analyzer state ---
        if self.debug_mode:
            print(f"    🔍 Segmentation Analyzer Debug:")
            print(f"       - self.segmentation_analyzer exists: {self.segmentation_analyzer is not None}")
            if self.segmentation_analyzer:
                print(f"       - analyzer.available: {self.segmentation_analyzer.available}")
                print(f"       - config.enable_image_segmentation: {self.config.enable_image_segmentation}")
                print(f"       - config.enable_text_segmentation: {self.config.enable_text_segmentation}")
                print(f"       - config.image_segmentation_prompt: '{self.config.image_segmentation_prompt}'")
                print(f"       - config.text_segmentation_prompt: '{self.config.text_segmentation_prompt}'")
        
        # --- Run Segmentation Analyzer if enabled ---
        if self.segmentation_analyzer and self.segmentation_analyzer.available and \
        (self.config.enable_image_segmentation or self.config.enable_text_segmentation):
            
            image_prompt_to_use = self.config.image_segmentation_prompt if self.config.enable_image_segmentation else None
            text_prompt_to_use = self.config.text_segmentation_prompt if self.config.enable_text_segmentation else None

            if self.debug_mode:
                print(f"    🔍 Final prompts to use:")
                print(f"       - image_prompt_to_use: '{image_prompt_to_use}'")
                print(f"       - text_prompt_to_use: '{text_prompt_to_use}'")

            if image_prompt_to_use or text_prompt_to_use:
                if self.debug_mode:
                    print(f"    🧠 Running DINO+SAM segmentation with prompts - Image: '{image_prompt_to_use}', Text: '{text_prompt_to_use}'")
                
                try:
                    segmentation_output = self.segmentation_analyzer.get_segmentation_results(
                        image_pil=image,
                        image_prompt=image_prompt_to_use,
                        text_prompt=text_prompt_to_use,
                        box_threshold=self.config.segmentation_box_threshold,
                        text_threshold=self.config.segmentation_text_threshold
                    )
                    analysis_results["segmentation_data"] = segmentation_output
                    if self.debug_mode:
                        print(f"    🎯 Segmentation completed. Results keys: {list(segmentation_output.keys())}")
                        # Print more details about what was found
                        for key, value in segmentation_output.items():
                            if value is not None:
                                if hasattr(value, 'shape'):
                                    print(f"       - {key}: tensor shape {value.shape}")
                                else:
                                    print(f"       - {key}: {type(value)} = {value}")
                            else:
                                print(f"       - {key}: None")
                except Exception as e:
                    if self.debug_mode:
                        print(f"    ❌ Segmentation failed with error: {e}")
                        import traceback
                        traceback.print_exc()
            elif self.debug_mode:
                print("    🧠 DINO+SAM segmentation skipped: No active prompts.")
        elif self.debug_mode:
            print("    🧠 DINO+SAM segmentation skipped: Analyzer not available or segmentation disabled.")

        # --- Call SmartDoublePageDetector (which now internally uses AdvancedDoublePageDetector) ---
        # The SmartDoublePageDetector's analyze_images expects a list, so we pass a single image in a list.
        # Its _combine_analysis_results_fixed will be the primary source for page_type and confidence.
        # Note: SmartDoublePageDetector itself doesn't use segmentation_data yet. This is for future.
        try:
            combined_page_analysis = self.detector.analyze_images([image])[0] 
            analysis_results.update(combined_page_analysis)
            
            # NEW: Store Florence2 data in cache if available
            if "florence2_result" in analysis_results:
                florence2_data = analysis_results["florence2_result"]
                cache_key = (page_num, image_index)
                self.florence2_analysis_cache[cache_key] = florence2_data
                
                if self.debug_mode:
                    print(f"    💾 Stored Florence2 data for page {page_num+1}, img {image_index+1}")
            
            if self.debug_mode:
                print(f"    📊 Combined Analysis: Type={analysis_results['page_type']}, Conf={analysis_results['confidence']:.3f}")
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ⚠️ Error during combined page analysis: {e}")
                import traceback
                traceback.print_exc()
            # Keep defaults if analysis fails
        
        return analysis_results

    def _safe_cleanup_pixmap(self, pix):
        """Safely cleanup PyMuPDF pixmap"""
        if pix is not None:
            try:
                # PyMuPDF pixmaps need explicit cleanup
                pix = None
                del pix
            except:
                pass

    def _safe_cleanup_image(self, image):
        """Safely cleanup PIL Image"""
        if image is not None:
            try:
                if hasattr(image, 'close'):
                    image.close()
                del image
            except:
                pass

    @contextmanager
    def _batch_image_processing(self, max_images=50):
        """Context manager for batched image processing"""
        processed_count = 0
        image_refs = []
        
        try:
            yield processed_count, image_refs
        finally:
            # Cleanup all image references
            for img_ref in image_refs:
                if isinstance(img_ref, weakref.ref):
                    img = img_ref()
                    if img is not None:
                        self._safe_cleanup_image(img)
                else:
                    self._safe_cleanup_image(img_ref)
            
            # Force garbage collection after batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def safe_execute(self, func, *args, fallback_result=None, context="", **kwargs):
        """Convenient access to safe execution"""
        return self.error_manager.safe_execute(func, *args, fallback_result=fallback_result, context=context, **kwargs)
    
    def _extract_page_images_for_analysis_safe(self, page, page_num: int, 
                                              report: ProcessingReport, 
                                              image_refs: list) -> List[Dict]:
        """Memory-safe image extraction with error recovery"""
        @self.error_manager.with_retry(max_retries=2, exceptions=(fitz.FileDataError, MemoryError))
        def extract_single_image(img_index, img):
            """Extract a single image with retry logic"""
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            
            # Validation and processing
            if pix.n - pix.alpha < 1 or pix.n - pix.alpha > 4:
                raise ProcessingError("Invalid colorspace")
            
            # Convert colorspace if needed
            if pix.colorspace and pix.colorspace.n == 4:
                old_pix = pix
                pix = fitz.Pixmap(fitz.csRGB, pix)
                self._safe_cleanup_pixmap(old_pix)
            
            return pix
        
        page_images = []
        image_list = page.get_images(full=True)
        
        processed_xrefs = set()
        seen_hashes = set()
        
        for img_index, img in enumerate(image_list):
            # Check if we should continue processing
            if not self.error_manager.should_continue_processing():
                break
            
            xref = img[0]
            if xref in processed_xrefs:
                continue
            
            # Safe image extraction with retry
            pix = self.safe_execute(
                extract_single_image, 
                img_index, img,
                fallback_result=None,
                context=f"page {page_num + 1}, image {img_index + 1}"
            )
            
            if pix is None:
                report.images_filtered_out += 1
                continue
            
            try:
                # Size filtering
                if not self._passes_size_filter(pix.width, pix.height):
                    report.images_filtered_out += 1
                    self._safe_cleanup_pixmap(pix)
                    continue
                
                # Deduplication
                import hashlib
                img_hash = hashlib.md5(pix.samples).hexdigest()
                if img_hash in seen_hashes:
                    report.images_filtered_out += 1
                    self._safe_cleanup_pixmap(pix)
                    continue
                
                seen_hashes.add(img_hash)
                processed_xrefs.add(xref)
                
                # Convert to PIL with error handling
                pil_image = self.safe_execute(
                    self._pixmap_to_pil,
                    pix,
                    fallback_result=None,
                    context=f"pixmap conversion page {page_num + 1}"
                )
                
                if pil_image is None:
                    self._safe_cleanup_pixmap(pix)
                    continue
                
                # Add to tracking
                image_refs.append(weakref.ref(pil_image))
                
                page_images.append({
                    "image": pil_image,
                    "page_num": page_num,
                    "image_index": img_index,
                    "width": pix.width,
                    "height": pix.height,
                    "colorspace": str(pix.colorspace) if pix.colorspace else "Unknown",
                    "xref": xref
                })
                
                # Clean up pixmap
                self._safe_cleanup_pixmap(pix)
                
                # Limit images per page
                if len(page_images) >= self.config.max_images_per_page:
                    break
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Error processing image {img_index} from page {page_num + 1}: {e}")
                report.images_filtered_out += 1
                self._safe_cleanup_pixmap(pix)
                continue
        
        return page_images
    
    def _pixmap_to_pil(self, pix) -> Image.Image:
        """Convert PyMuPDF pixmap to PIL Image with error handling"""
        def _convert():
            import io
            img_data = pix.tobytes("ppm")
            return Image.open(io.BytesIO(img_data))
        
        return self.safe_execute(_convert, fallback_result=None, context="pixmap conversion")
    
    
    def process_pdf_enhanced(self, pdf_path: str, extract_images: bool = True, 
                        extract_text: bool = True, save_options: str = "enhanced_only",
                        enable_smart_crop: bool = True, enhancement_strength: float = 1.0, 
                        join_double_pages: bool = True, use_gpu_acceleration: bool = True) -> ProcessingReport:
        """Enhanced processing with early spread detection and raw joining"""
        
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        try:
            # Check if we should use streaming processing
            streaming_processor = StreamingPDFProcessor(self.config, self.debug_mode)
            should_stream = streaming_processor._should_use_streaming(pdf_path)
            
            if should_stream and self.config.memory_optimization:
                if self.debug_mode:
                    print("📊 Large PDF detected - using streaming processing")
                
                report = streaming_processor.process_pdf_streaming(
                    pdf_path, self, extract_images, extract_text, save_options,
                    enable_smart_crop, enhancement_strength, join_double_pages, use_gpu_acceleration
                )
            else:
                # Use NEW restructured processing for smaller PDFs
                report = self._create_initial_report(pdf_path)
                
                if self.use_pymupdf:
                    # NEW: Restructured processing pipeline
                    self._process_with_pymupdf_restructured(
                        pdf_path, report, extract_images, extract_text, 
                        save_options, enable_smart_crop, enhancement_strength,
                        join_double_pages, use_gpu_acceleration
                    )
                else:
                    self._process_with_pypdf2_enhanced(
                        pdf_path, report, extract_images, extract_text, 
                        save_options, enable_smart_crop, enhancement_strength
                    )
            
            # Add error summary to report if there were errors
            if self.error_manager.error_counts:
                print(f"📊 Processing completed with errors:\n{self.error_manager.get_error_summary()}")
                
        except Exception as e:
            error_summary = self.error_manager.get_error_summary()
            print(f"❌ Enhanced processing failed: {e}")
            print(f"📊 Error summary:\n{error_summary}")
            
            if self.debug_mode:
                traceback.print_exc()
            
            # Try to save partial results
            if 'report' in locals() and (report.extracted_images or report.extracted_text):
                print("💾 Saving partial results...")
            
            # Create minimal report for error case
            if 'report' not in locals():
                report = self._create_initial_report(pdf_path)
        
        finally:
            # Clean up caches - ALWAYS runs regardless of streaming or regular processing
            self._cleanup_florence2_cache()
        
        report.processing_time = time.time() - start_time
        
        # Final error summary
        if self.error_manager.error_counts:
            print(f"📊 Processing completed with errors:\n{self.error_manager.get_error_summary()}")
        
        return report

    def _create_initial_report(self, pdf_path: Path) -> ProcessingReport:
        """Create initial processing report - centralized method"""
        return ProcessingReport(
            pdf_filename=pdf_path.name,
            total_pages=0,
            processing_time=0.0,
            total_images_found=0,
            images_extracted=0,
            images_filtered_out=0,
            images_enhanced=0,
            images_joined=0,
            text_layers_found=0,
            text_extracted_pages=0,
            quality_breakdown={q.value: 0 for q in ImageQuality},
            page_type_breakdown={p.value: 0 for p in PageType},
            multi_page_candidates=[],
            output_directory=str(self.output_dir),
            extracted_images=[],
            extracted_text=[],
            joined_images=[]
        )


    def process_pdf_enhanced(self, pdf_path: str, extract_images: bool = True, 
                        extract_text: bool = True, save_options: str = "enhanced_only",
                        enable_smart_crop: bool = True, enhancement_strength: float = 1.0, 
                        join_double_pages: bool = True, use_gpu_acceleration: bool = True) -> ProcessingReport:
        """Enhanced processing with early spread detection and raw joining"""
        
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        # Check if we should use streaming processing
        streaming_processor = StreamingPDFProcessor(self.config, self.debug_mode)
        should_stream = streaming_processor._should_use_streaming(pdf_path)
        
        if should_stream and self.config.memory_optimization:
            if self.debug_mode:
                print("📊 Large PDF detected - using streaming processing")
            
            report = streaming_processor.process_pdf_streaming(
                pdf_path, self, extract_images, extract_text, save_options,
                enable_smart_crop, enhancement_strength, join_double_pages, use_gpu_acceleration
            )
        else:
            # Use NEW restructured processing for smaller PDFs
            report = self._create_initial_report(pdf_path)
            
            try:
                if self.use_pymupdf:
                    # NEW: Restructured processing pipeline
                    self._process_with_pymupdf_restructured(
                        pdf_path, report, extract_images, extract_text, 
                        save_options, enable_smart_crop, enhancement_strength,
                        join_double_pages, use_gpu_acceleration
                    )
                else:
                    self._process_with_pypdf2_enhanced(
                        pdf_path, report, extract_images, extract_text, 
                        save_options, enable_smart_crop, enhancement_strength
                    )
                    
            except Exception as e:
                error_summary = self.error_manager.get_error_summary()
                print(f"❌ Enhanced processing failed: {e}")
                print(f"📊 Error summary:\n{error_summary}")
                
                if self.debug_mode:
                    traceback.print_exc()
                
                # Try to save partial results
                if report.extracted_images or report.extracted_text:
                    print("💾 Saving partial results...")
        
        report.processing_time = time.time() - start_time
        
        # Add error summary to report if there were errors
        if self.error_manager.error_counts:
            print(f"📊 Processing completed with errors:\n{self.error_manager.get_error_summary()}")
        
        return report

    def _process_with_pymupdf_restructured(self, pdf_path: Path, report: ProcessingReport,
                                        extract_images: bool, extract_text: bool, 
                                        save_options: str, enable_smart_crop: bool, 
                                        enhancement_strength: float, join_double_pages: bool, 
                                        use_gpu_acceleration: bool):
        """NEW: Restructured PyMuPDF processing with early spread detection"""
        doc = fitz.open(str(pdf_path))
        report.total_pages = doc.page_count
        
        try:
            # PHASE 1: Extract raw images and minimal analysis
            if self.debug_mode:
                print("🔍 PHASE 1: Extracting raw images and analyzing for spreads...")
            
            raw_images = []  # Store raw images with basic info
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text
                if extract_text:
                    self._extract_text_pymupdf(page, page_num, report)
                
                # Extract raw images with minimal processing
                if extract_images:
                    page_raw_images = self._extract_raw_images_for_analysis(page, page_num, report)
                    raw_images.extend(page_raw_images)
            
            # PHASE 2: Early spread detection on raw images
            spread_pairs = []
            if join_double_pages and len(raw_images) > 1:
                if self.debug_mode:
                    print("🔍 PHASE 2: Analyzing raw images for double-page spreads...")
                
                spread_pairs = self._detect_spreads_early(raw_images)
                
                if self.debug_mode:
                    print(f"Found {len(spread_pairs)} potential spread pairs")
            
            # PHASE 3: Join raw spreads BEFORE any enhancement
            joined_raw_images = []
            processed_page_nums = set()
            
            if spread_pairs:
                if self.debug_mode:
                    print("🔗 PHASE 3: Joining raw spread pairs...")
                
                for left_raw, right_raw in spread_pairs:
                    joined_raw = self._join_raw_images(left_raw, right_raw)
                    if joined_raw:
                        joined_raw_images.append(joined_raw)
                        processed_page_nums.add(left_raw["page_num"])
                        processed_page_nums.add(right_raw["page_num"])
            
            # PHASE 4: Process all images (single + joined) uniformly
            if self.debug_mode:
                print("⚡ PHASE 4: Processing and enhancing all images uniformly...")
            
            all_images_to_process = []
            
            # Add joined spreads
            all_images_to_process.extend(joined_raw_images)
            
            # Add single images (not part of spreads)
            for raw_img in raw_images:
                if raw_img["page_num"] not in processed_page_nums:
                    all_images_to_process.append(raw_img)
            
            # Process all images with the same pipeline
            for img_data in all_images_to_process:
                self._process_single_image_unified(
                    img_data, report, save_options, enable_smart_crop, 
                    enhancement_strength, use_gpu_acceleration
                )
                
        finally:
            doc.close()


    def _extract_raw_images_for_analysis(self, page, page_num: int, report: ProcessingReport) -> List[Dict]:
        """Extract raw images with minimal processing for spread analysis"""
        page_images = []
        image_list = page.get_images(full=True)
        
        processed_xrefs = set()
        seen_hashes = set()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            if xref in processed_xrefs:
                continue
            
            try:
                pix = fitz.Pixmap(page.parent, xref)
                
                # Basic validation
                if not self._passes_size_filter(pix.width, pix.height):
                    report.images_filtered_out += 1
                    self._safe_cleanup_pixmap(pix)
                    continue
                
                # Handle colorspace
                if pix.colorspace and pix.colorspace.n == 4:
                    old_pix = pix
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    self._safe_cleanup_pixmap(old_pix)
                
                # Convert to PIL
                pil_image = self._pixmap_to_pil(pix)
                if pil_image is None:
                    self._safe_cleanup_pixmap(pix)
                    continue
                
                # Deduplication
                img_hash = hashlib.md5(np.array(pil_image).tobytes()).hexdigest()
                if img_hash in seen_hashes:
                    report.images_filtered_out += 1
                    self._safe_cleanup_image(pil_image)
                    self._safe_cleanup_pixmap(pix)
                    continue
                
                seen_hashes.add(img_hash)
                processed_xrefs.add(xref)
                
                # Store raw image data
                page_images.append({
                    "image": pil_image,
                    "page_num": page_num,
                    "image_index": img_index,
                    "width": pix.width,
                    "height": pix.height,
                    "colorspace": str(pix.colorspace) if pix.colorspace else "RGB",
                    "xref": xref,
                    "is_joined": False,  # Mark as single image
                    "source_images": None  # No source images for single
                })
                
                self._safe_cleanup_pixmap(pix)
                
                # Limit images per page
                if len(page_images) >= self.config.max_images_per_page:
                    break
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"Error extracting raw image {img_index} from page {page_num + 1}: {e}")
                report.images_filtered_out += 1
                continue
        
        return page_images

    def _detect_spreads_early(self, raw_images: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Detect potential spread pairs from raw images before processing"""
        
        # Quick analysis for spread detection
        analyzed_images = []
        
        for raw_img in raw_images:
            # Do lightweight analysis for page type detection
            quick_analysis = self._quick_page_type_analysis(raw_img["image"])
            raw_img["page_type"] = quick_analysis["page_type"]
            raw_img["confidence"] = quick_analysis["confidence"]
            analyzed_images.append(raw_img)
        
        # Find spread pairs
        spread_pairs = []
        left_candidates = [img for img in analyzed_images if img["page_type"] == PageType.DOUBLE_PAGE_LEFT]
        right_candidates = [img for img in analyzed_images if img["page_type"] == PageType.DOUBLE_PAGE_RIGHT]
        
        for left_img in left_candidates:
            # Look for corresponding right page
            right_matches = [r for r in right_candidates if r["page_num"] == left_img["page_num"] + 1]
            
            if right_matches:
                right_img = max(right_matches, key=lambda x: x["confidence"])
                
                # Validate spread pair with raw data
                if self._validate_raw_spread_pair(left_img, right_img):
                    spread_pairs.append((left_img, right_img))
                    if self.debug_mode:
                        print(f"  📖 Detected spread pair: pages {left_img['page_num']+1}-{right_img['page_num']+1}")
        
        return spread_pairs

    def _quick_page_type_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Lightweight page type analysis for early spread detection"""
        
        # Use only the fastest methods for early detection
        width, height = image.size
        aspect_ratio = width / height
        
        # Quick geometric analysis
        if aspect_ratio > 2.0:
            return {"page_type": PageType.DOUBLE_PAGE_LEFT, "confidence": 0.7}
        elif aspect_ratio > 1.6:
            return {"page_type": PageType.DOUBLE_PAGE_LEFT, "confidence": 0.5}
        elif 0.7 <= aspect_ratio <= 1.3:
            return {"page_type": PageType.SINGLE_PAGE, "confidence": 0.6}
        
        # If available, use quick border analysis
        if CV2_AVAILABLE:
            try:
                border_result = self._quick_border_analysis(image)
                if border_result["confidence"] > 0.5:
                    return border_result
            except:
                pass
        
        return {"page_type": PageType.UNCERTAIN, "confidence": 0.3}

    def _quick_border_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Quick border analysis for early detection"""
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        h, w = gray.shape
        border_width = min(30, min(h, w) // 20)
        
        # Check border uniformity
        borders = {
            'top': gray[:border_width, :],
            'bottom': gray[-border_width:, :],
            'left': gray[:, :border_width],
            'right': gray[:, -border_width:]
        }
        
        border_scores = {}
        for side, region in borders.items():
            uniformity = 1.0 - (np.std(region) / 255.0)
            brightness = np.mean(region) / 255.0
            has_border = uniformity > 0.7 and (brightness > 0.85 or brightness < 0.15)
            border_scores[side] = has_border
        
        border_count = sum(border_scores.values())
        
        # Quick classification
        if border_count == 3:
            if not border_scores["right"]:
                return {"page_type": PageType.DOUBLE_PAGE_LEFT, "confidence": 0.8}
            elif not border_scores["left"]:
                return {"page_type": PageType.DOUBLE_PAGE_RIGHT, "confidence": 0.8}
        elif border_count == 4:
            return {"page_type": PageType.SINGLE_PAGE, "confidence": 0.7}
        
        return {"page_type": PageType.UNCERTAIN, "confidence": 0.3}

    def _validate_raw_spread_pair(self, left_img: Dict, right_img: Dict) -> bool:
        """Validate spread pair using raw image data"""
        
        # Check consecutive pages
        if right_img["page_num"] != left_img["page_num"] + 1:
            return False
        
        # Check confidence
        min_confidence = 0.4
        if left_img["confidence"] < min_confidence or right_img["confidence"] < min_confidence:
            return False
        
        # Check height similarity
        height_diff_ratio = abs(left_img["height"] - right_img["height"]) / max(left_img["height"], right_img["height"])
        if height_diff_ratio > 0.15:  # Allow more tolerance for raw images
            return False
        
        # Check combined aspect ratio
        combined_width = left_img["width"] + right_img["width"]
        combined_ratio = combined_width / left_img["height"]
        if not (1.4 < combined_ratio < 4.5):  # Broader range for raw images
            return False
        
        return True

    def _join_raw_images(self, left_raw: Dict, right_raw: Dict) -> Optional[Dict]:
        """Join two raw images before any processing"""
        try:
            left_image = left_raw["image"]
            right_image = right_raw["image"]
            
            if self.debug_mode:
                print(f"    🔗 Joining raw images: {left_image.size} + {right_image.size}")
            
            # Use simple joining for raw images (before enhancement)
            joined_image = self._simple_raw_join(left_image, right_image)
            
            if joined_image:
                return {
                    "image": joined_image,
                    "page_num": left_raw["page_num"],  # Use left page number
                    "image_index": 0,  # Joined images get index 0
                    "width": joined_image.width,
                    "height": joined_image.height,
                    "colorspace": left_raw["colorspace"],  # Use left colorspace
                    "xref": f"{left_raw['xref']}-{right_raw['xref']}",
                    "is_joined": True,  # Mark as joined
                    "source_images": (left_raw, right_raw),  # Store source info
                    "page_type": PageType.SINGLE_PAGE,  # Joined spreads are treated as single images
                    "confidence": min(left_raw["confidence"], right_raw["confidence"])
                }
        
        except Exception as e:
            if self.debug_mode:
                print(f"    ❌ Raw joining failed: {e}")
        
        return None

    def _simple_raw_join(self, left_image: Image.Image, right_image: Image.Image) -> Optional[Image.Image]:
        """Simple side-by-side joining for raw images"""
        try:
            # Ensure same height
            min_height = min(left_image.height, right_image.height)
            
            if left_image.height != min_height:
                scale = min_height / left_image.height
                new_width = int(left_image.width * scale)
                left_image = left_image.resize((new_width, min_height), Image.Resampling.LANCZOS)
            
            if right_image.height != min_height:
                scale = min_height / right_image.height
                new_width = int(right_image.width * scale)
                right_image = right_image.resize((new_width, min_height), Image.Resampling.LANCZOS)
            
            # Create combined image
            total_width = left_image.width + right_image.width
            combined = Image.new('RGB', (total_width, min_height))
            combined.paste(left_image, (0, 0))
            combined.paste(right_image, (left_image.width, 0))
            
            if self.debug_mode:
                print(f"    ✅ Raw join successful: {combined.size}")
            
            return combined
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ❌ Simple raw join failed: {e}")
            return None

    def _process_single_image_unified(self, img_data: Dict, report: ProcessingReport,
                                    save_options: str, enable_smart_crop: bool, 
                                    enhancement_strength: float, use_gpu_acceleration: bool):
        """Unified processing for both single images and joined spreads"""
        
        image = img_data["image"]
        page_num = img_data["page_num"]
        img_index = img_data["image_index"]
        is_joined = img_data.get("is_joined", False)
        
        # Color profile management
        image = self._ensure_color_profile(image)
        img_data["image"] = image
        
        if self.debug_mode:
            image_type = "joined spread" if is_joined else "single image"
            print(f"Processing {image_type} from page {page_num + 1}")
        
        # For joined images, use stored analysis; for single images, do full analysis
        if is_joined:
            # Use the page type and confidence from early detection
            page_type = img_data.get("page_type", PageType.SINGLE_PAGE)
            confidence = img_data.get("confidence", 0.7)
            
            # Create minimal analysis dict for joined images
            analysis = {
                "page_type": page_type,
                "confidence": confidence,
                "segmentation_data": None,  # No segmentation needed for joined
                "image_index": img_index,
                "image_page_num": page_num,
                "image_size": image.size
            }
        else:
            # Full analysis for single images
            analysis = self._analyze_page_content_enhanced(image, page_num, img_index)
            page_type = analysis["page_type"]
            confidence = analysis["confidence"]
            
            # NEW: Ensure Florence2 data is stored for later spread validation
            if "florence2_result" in analysis:
                cache_key = (page_num, img_index)
                self.florence2_analysis_cache[cache_key] = analysis["florence2_result"]
                
                if self.debug_mode:
                    print(f"    💾 Stored Florence2 data for page {page_num+1}, img {img_index+1}")
        
        # Update page type breakdown
        report.page_type_breakdown[page_type.value] += 1
        
        # Confidence filtering
        should_save = True
        if page_type == PageType.UNCERTAIN and confidence < 0.15:
            should_save = False
            report.images_filtered_out += 1
            if self.debug_mode:
                print(f"Filtering out uncertain image (confidence: {confidence})")
        
        if not should_save:
            return
        
        # Apply smart cropping (even to joined images for consistency)
        if enable_smart_crop:
            processed_image = self._apply_smart_cropping(image, analysis)
            processed_image = self._preserve_color_profile(image, processed_image)
        else:
            processed_image = image
        
        # Save images
        saved_images = self._save_enhanced_images(
            image, processed_image, page_num, img_index, save_options,
            enhancement_strength, use_gpu_acceleration
        )
        
        # Create records
        for filename, saved_img, is_enhanced in saved_images:
            quality = self._assess_image_quality(saved_img)
            file_size = self._get_file_size(filename)
            
            # Special handling for joined images
            if is_joined:
                source_info = img_data.get("source_images", (None, None))
                if source_info[0] and source_info[1]:
                    extraction_method = f"Enhanced_PyMuPDF_Joined_P{source_info[0]['page_num']+1}-P{source_info[1]['page_num']+1}_{'enhanced' if is_enhanced else 'raw'}"
                else:
                    extraction_method = f"Enhanced_PyMuPDF_Joined_{'enhanced' if is_enhanced else 'raw'}"
            else:
                extraction_method = f"Enhanced_PyMuPDF_{'enhanced' if is_enhanced else 'raw'}"
            
            extracted_img = ExtractedImage(
                filename=filename,
                page_num=page_num,
                image_index=img_index,
                width=saved_img.width,
                height=saved_img.height,
                file_size_bytes=file_size,
                quality_score=quality,
                page_type=page_type,
                border_confidence=confidence,
                is_multi_page_candidate=False,  # Joined spreads are no longer candidates
                bbox=(0, 0, saved_img.width, saved_img.height),
                original_colorspace=img_data["colorspace"],
                extraction_method=extraction_method
            )
            
            report.extracted_images.append(extracted_img)
            report.images_extracted += 1
            report.quality_breakdown[quality.value] += 1
            
            if is_enhanced:
                report.images_enhanced += 1
            
            if is_joined:
                # Create a joined image record
                source_info = img_data.get("source_images", (None, None))
                if source_info[0] and source_info[1]:
                    join_record = JoinedImage(
                        filename=filename,
                        left_page_num=source_info[0]["page_num"],
                        right_page_num=source_info[1]["page_num"],
                        left_image_filename="raw_joined",
                        right_image_filename="raw_joined",
                        combined_width=saved_img.width,
                        combined_height=saved_img.height,
                        confidence_score=confidence,
                        join_method="early_raw_joining"
                    )
                    report.joined_images.append(join_record)
                    report.images_joined += 1
            
            if self.debug_mode:
                print(f"Created record: {filename}, enhanced: {is_enhanced}, joined: {is_joined}")

    def _process_image_batch_safe(self, batch_images: List[Dict], report: ProcessingReport,
                                save_options: str, enable_smart_crop: bool, enhancement_strength: float, 
                                use_gpu_acceleration: bool):
        """Process a batch of images with memory safety"""
        if not batch_images:
            return
        
        try:
            # Use enhanced analysis instead of regular detector analysis
            analyses = []
            for img_data in batch_images:
                analysis = self._analyze_page_content_enhanced(
                    img_data["image"], 
                    img_data["page_num"], 
                    img_data["image_index"]
                )
                analyses.append(analysis)
            
            # Process each image
            for img_data, analysis in zip(batch_images, analyses):
                try:
                    self._process_analyzed_image(
                        img_data, analysis, report, save_options, 
                        enable_smart_crop, enhancement_strength, use_gpu_acceleration
                    )
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error processing image: {e}")
                    continue
                finally:
                    # Clean up the image after processing
                    self._safe_cleanup_image(img_data.get("image"))
                    
        except Exception as e:
            if self.debug_mode:
                print(f"Batch processing error: {e}")


        
    
    def _process_analyzed_image(self, img_data: Dict, analysis: Dict, report: ProcessingReport,
                            save_options: str, enable_smart_crop: bool, enhancement_strength: float, 
                            use_gpu_acceleration: bool):
        """Process a single image with color management"""
        image = img_data["image"]
        page_num = img_data["page_num"]
        img_index = img_data["image_index"]
        
        # MOVE THIS TO THE VERY TOP BEFORE ANY OTHER PROCESSING:
        image = self._ensure_color_profile(image)
        img_data["image"] = image  # Update the dict with the profile-enhanced image
        
        # Debug original color info
        self._debug_color_info(image, "Original (with profile)")

        if self.debug_mode:
            print(f"DEBUG: Processing image from page {page_num + 1}, index {img_index}")
        
        # Get page type from analysis
        page_type = analysis["page_type"]
        confidence = analysis["confidence"]
        
        # Update page type breakdown
        report.page_type_breakdown[page_type.value] += 1

        # ADD BACK: Save segmentation masks if available and debug mode enabled
        if self.config.save_debug_images and analysis.get("segmentation_data"):
            self._save_segmentation_masks(analysis["segmentation_data"], page_num, img_index)
        
        # Determine if this image should be saved based on confidence thresholds
        should_save = True
        if page_type == PageType.UNCERTAIN and confidence < 0.15:  # Lowered from 0.3
            should_save = False
            report.images_filtered_out += 1
            if self.debug_mode:
                print(f"DEBUG: Filtering out uncertain image (confidence: {confidence})")
        
        if not should_save:
            return
        
        # Apply smart cropping
        if enable_smart_crop:
            processed_image = self._apply_smart_cropping(image, analysis)
            # Preserve profile from original to processed
            processed_image = self._preserve_color_profile(image, processed_image)
        else:
            processed_image = image
        
        # Debug processed color info
        self._debug_color_info(processed_image, "After Cropping")

        # Save images based on save options
        saved_images = self._save_enhanced_images(
            image, processed_image, page_num, img_index, save_options, 
            enhancement_strength, use_gpu_acceleration
        )
        
        if self.debug_mode:
            print(f"DEBUG: Creating ExtractedImage records for {len(saved_images)} saved images")
        
        # Create records for saved images
        for filename, saved_img, is_enhanced in saved_images:
            if self.debug_mode:
                print(f"DEBUG: Creating record for {filename}")
            
            quality = self._assess_image_quality(saved_img)
            file_size = self._get_file_size(filename)
            
            extracted_img = ExtractedImage(
                filename=filename,
                page_num=page_num,
                image_index=img_index,
                width=saved_img.width,
                height=saved_img.height,
                file_size_bytes=file_size,
                quality_score=quality,
                page_type=page_type,
                border_confidence=confidence,
                is_multi_page_candidate=self._is_enhanced_multi_page_candidate(saved_img, page_type),
                bbox=(0, 0, saved_img.width, saved_img.height),
                original_colorspace=img_data["colorspace"],
                extraction_method=f"Enhanced_PyMuPDF_{'enhanced' if is_enhanced else 'raw'}"
            )
            
            report.extracted_images.append(extracted_img)
            report.images_extracted += 1
            report.quality_breakdown[quality.value] += 1
            
            if is_enhanced:
                report.images_enhanced += 1
            
            if extracted_img.is_multi_page_candidate:
                report.multi_page_candidates.append(filename)
            
            if self.debug_mode:
                print(f"DEBUG: Created ExtractedImage record: {filename}, size: {file_size} bytes")
                

    def _apply_smart_cropping(self, image: Image.Image, analysis: Dict) -> Image.Image:
        """Apply intelligent cropping with multiple fallback methods"""
        return self._apply_smart_cropping_enhanced(image, analysis)

    def _apply_smart_cropping_enhanced(self, image: Image.Image, analysis: Dict) -> Image.Image:
        """Enhanced smart cropping with Florence2 as primary method"""
        if self.debug_mode:
            print(f"    Applying smart cropping to {image.size} image")
        
        crop_attempts = []
        
        # Method 1: Florence2 cropping (HIGHEST PRIORITY with increased weight)
        florence2_result = self._try_florence2_cropping(image, analysis)
        if florence2_result["success"]:
            # Give Florence2 even higher priority in selection
            florence2_priority_boost = 200  # Increased from 100
            crop_attempts.append(("florence2", florence2_result, florence2_priority_boost))
        
        # Method 2: Segmentation-based cropping (if Florence2 available for segmentation)
        if self.florence2_cropper:
            seg_result = self._try_florence2_segmentation_cropping(image, analysis)
            if seg_result["success"]:
                crop_attempts.append(("florence2_segmentation", seg_result, 150))  # Also boost Florence2 segmentation
        
        # Method 3: Traditional GroundingDINO + SAM cropping
        groundingdino_result = self._try_groundingdino_sam_cropping(image, analysis)
        if groundingdino_result["success"]:
            crop_attempts.append(("groundingdino_sam", groundingdino_result, 0))
        
        # Method 4: CV2 content detection
        cv2_result = self._try_cv2_content_detection(image)
        if cv2_result["success"]:
            crop_attempts.append(("cv2_content", cv2_result, 0))
        
        # Method 5: Adaptive background removal
        adaptive_result = self._try_adaptive_background_removal(image)
        if adaptive_result["success"]:
            crop_attempts.append(("adaptive_background", adaptive_result, 0))
        
        # Select best crop attempt with enhanced Florence2 prioritization
        if crop_attempts:
            # Enhanced prioritization: Florence2 methods get massive priority boost + confidence
            best_method, best_result, priority_boost = max(crop_attempts, key=lambda x: (
                x[2],  # Priority boost (Florence2 methods get 200/150)
                x[1].get("confidence", 0)  # Then by confidence
            ))
            
            # Special handling for Florence2 strategies that return no crop
            if best_method.startswith("florence2") and best_result.get("strategy") in ["single_full_page", "potential_collage"]:
                if self.debug_mode:
                    strategy = best_result.get("strategy", "unknown")
                    print(f"    ✅ Florence2 strategy '{strategy}': No cropping needed, using full image")
                return image  # No cropping for full-page images
            
            crop_box = best_result.get("crop_box")
            if crop_box and self._validate_crop_box(crop_box, image.size):
                cropped_image = image.crop(crop_box)
                
                if self.debug_mode:
                    original_area = image.size[0] * image.size[1]
                    cropped_area = cropped_image.size[0] * cropped_image.size[1]
                    reduction = (1 - cropped_area / original_area) * 100
                    strategy_info = f" ({best_result.get('strategy', 'unknown')})" if best_method.startswith("florence2") else ""
                    print(f"    ✅ Cropped using {best_method}{strategy_info}: {image.size} → {cropped_image.size} ({reduction:.1f}% reduction)")
                
                # Save debug mask if available
                if self.config.save_debug_images and "mask_image" in best_result:
                    self._save_crop_debug_info(best_result, best_method)
                
                return cropped_image
        
        # Fallback: minimal border cleanup
        if self.debug_mode:
            print(f"    ⚠️ No effective cropping found, applying minimal cleanup")
        
        return self._minimal_border_cleanup(image)

    def _try_florence2_segmentation_cropping(self, image: Image.Image, analysis: Dict) -> Dict[str, Any]:
        """Try Florence2-based segmentation for more precise cropping"""
        if not self.florence2_cropper:
            return {"success": False, "method": "florence2_seg_unavailable"}
        
        try:
            # Use segmentation prompts
            segmentation_prompts = [
                "main content area",
                "text and images", 
                "content without margins",
                "foreground content"
            ]
            
            for prompt in segmentation_prompts:
                try:
                    bounding_boxes, mask_image = self.florence2_cropper.detect_rectangles(
                        image=image,
                        text_input=prompt,
                        return_mask=True,
                        keep_model_loaded=True
                    )
                    
                    if bounding_boxes and mask_image:
                        # Use the mask to find tighter bounds
                        crop_box = self._mask_to_crop_box(mask_image, image.size)
                        
                        if crop_box:
                            return {
                                "success": True,
                                "method": "florence2_segmentation",
                                "crop_box": crop_box,
                                "confidence": 0.8,
                                "mask_image": mask_image
                            }
                            
                except Exception:
                    continue
            
            return {"success": False, "method": "florence2_seg_no_content"}
            
        except Exception as e:
            if self.debug_mode:
                print(f"      Florence2 segmentation failed: {e}")
            return {"success": False, "method": "florence2_seg_error", "error": str(e)}

    def _mask_to_crop_box(self, mask_image: Image.Image, 
                         original_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Convert mask image to crop box coordinates"""
        if not mask_image:
            return None
        
        try:
            import numpy as np
            
            # Convert mask to numpy array
            mask_array = np.array(mask_image.convert('L'))
            
            # Find non-zero pixels
            coords = np.argwhere(mask_array > 128)  # Threshold for white pixels
            
            if len(coords) == 0:
                return None
            
            # Get bounding box
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            # Add margin
            margin = self.config.crop_margin
            width, height = original_size
            
            crop_x1 = max(0, min_x - margin)
            crop_y1 = max(0, min_y - margin)
            crop_x2 = min(width, max_x + margin)
            crop_y2 = min(height, max_y + margin)
            
            return (crop_x1, crop_y1, crop_x2, crop_y2)
            
        except Exception as e:
            if self.debug_mode:
                print(f"      Mask to crop box conversion failed: {e}")
            return None

    def _save_crop_debug_info(self, crop_result: Dict, method: str):
        """Save debug information for crop analysis"""
        try:
            debug_dir = self.output_dir / "debug_crops"
            debug_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%H%M%S")
            
            # Save mask if available
            if "mask_image" in crop_result and crop_result["mask_image"]:
                mask_path = debug_dir / f"crop_mask_{method}_{timestamp}.png"
                crop_result["mask_image"].save(mask_path)
                
            # Save crop info with enhanced Florence2 details
            info_path = debug_dir / f"crop_info_{method}_{timestamp}.txt"
            with open(info_path, 'w') as f:
                f.write(f"Method: {method}\n")
                f.write(f"Success: {crop_result['success']}\n")
                f.write(f"Confidence: {crop_result.get('confidence', 'N/A')}\n")
                f.write(f"Crop Box: {crop_result.get('crop_box', 'N/A')}\n")
                if 'detected_rectangles' in crop_result:
                    f.write(f"Detected Rectangles: {crop_result['detected_rectangles']}\n")
                if 'filtered_rectangles' in crop_result:
                    f.write(f"Filtered Rectangles: {crop_result['filtered_rectangles']}\n")
                if 'strategy' in crop_result:
                    f.write(f"Florence2 Strategy: {crop_result['strategy']}\n")
                    
        except Exception as e:
            if self.debug_mode:
                print(f"      Failed to save crop debug info: {e}")

    def _validate_spread_pair_with_florence2(self, left_img: ExtractedImage, right_img: ExtractedImage) -> bool:
        """Enhanced spread validation using Florence2 bounding box data"""
        
        # Basic validation first
        if not self._validate_spread_pair(left_img, right_img):
            return False
        
        # Try to get Florence2 analysis data for both images
        left_florence_data = self._get_florence2_analysis_for_image(left_img)
        right_florence_data = self._get_florence2_analysis_for_image(right_img)
        
        if not (left_florence_data and right_florence_data):
            if self.debug_mode:
                print(f"    No Florence2 data available for spread analysis")
            return True  # Fall back to basic validation
        
        # Enhanced Florence2-based validation
        florence2_confidence_boost = self._analyze_florence2_spread_indicators(
            left_florence_data, right_florence_data, left_img, right_img
        )
        
        if self.debug_mode:
            print(f"    Florence2 spread analysis boost: {florence2_confidence_boost:.3f}")
        
        # Higher threshold with Florence2 data available
        enhanced_threshold = 0.7
        combined_confidence = min(left_img.border_confidence, right_img.border_confidence) + florence2_confidence_boost
        
        return combined_confidence >= enhanced_threshold

    def _analyze_florence2_spread_indicators(self, left_data: Dict, right_data: Dict, 
                                        left_img: ExtractedImage, right_img: ExtractedImage) -> float:
        """Analyze Florence2 bounding boxes for spread indicators"""
        confidence_boost = 0.0
        
        left_boxes = left_data.get("detected_rectangles", [])
        right_boxes = right_data.get("detected_rectangles", [])
        
        if not (left_boxes and right_boxes):
            return 0.0
        
        # Find main content boxes (largest by area)
        left_main = max(left_boxes, key=lambda x: x.get("area", 0)) if left_boxes else None
        right_main = max(right_boxes, key=lambda x: x.get("area", 0)) if right_boxes else None
        
        if not (left_main and right_main):
            return 0.0
        
        left_bbox = left_main["bbox"]  # (x1, y1, x2, y2)
        right_bbox = right_main["bbox"]
        
        # Check 1: Vertical alignment of content
        left_center_y = (left_bbox[1] + left_bbox[3]) / 2
        right_center_y = (right_bbox[1] + right_bbox[3]) / 2
        
        # Normalize by image height
        left_center_y_norm = left_center_y / left_img.height
        right_center_y_norm = right_center_y / right_img.height
        
        vertical_alignment_diff = abs(left_center_y_norm - right_center_y_norm)
        if vertical_alignment_diff < 0.05:  # Within 5% of image height
            confidence_boost += 0.2
            if self.debug_mode:
                print(f"      ✅ Excellent vertical alignment: {vertical_alignment_diff:.3f}")
        elif vertical_alignment_diff < 0.1:  # Within 10%
            confidence_boost += 0.1
            if self.debug_mode:
                print(f"      ✅ Good vertical alignment: {vertical_alignment_diff:.3f}")
        
        # Check 2: Border analysis (inside vs outside borders)
        # Left page: right border should be smaller than left border
        # Right page: left border should be smaller than right border
        
        left_left_border = left_bbox[0]  # Distance from left edge
        left_right_border = left_img.width - left_bbox[2]  # Distance from right edge
        right_left_border = right_bbox[0]  # Distance from left edge  
        right_right_border = right_img.width - right_bbox[2]  # Distance from right edge
        
        # Normalize borders by image width
        left_left_norm = left_left_border / left_img.width
        left_right_norm = left_right_border / left_img.width
        right_left_norm = right_left_border / right_img.width
        right_right_norm = right_right_border / right_img.width
        
        # Check spread pattern: inside borders < outside borders
        left_spread_pattern = left_right_norm < left_left_norm  # Right border < left border
        right_spread_pattern = right_left_norm < right_right_norm  # Left border < right border
        
        if left_spread_pattern and right_spread_pattern:
            # Calculate how pronounced the pattern is
            left_border_ratio = left_right_norm / (left_left_norm + 0.001)  # Avoid division by zero
            right_border_ratio = right_left_norm / (right_right_norm + 0.001)
            
            # Stronger pattern = higher confidence
            pattern_strength = (1 - left_border_ratio) + (1 - right_border_ratio)
            pattern_boost = min(0.3, pattern_strength * 0.15)  # Max 0.3 boost
            confidence_boost += pattern_boost
            
            if self.debug_mode:
                print(f"      ✅ Spread border pattern detected: L_ratio={left_border_ratio:.3f}, R_ratio={right_border_ratio:.3f}, boost={pattern_boost:.3f}")
        
        # Check 3: Content height consistency
        left_content_height = left_bbox[3] - left_bbox[1]
        right_content_height = right_bbox[3] - right_bbox[1]
        
        height_diff_ratio = abs(left_content_height - right_content_height) / max(left_content_height, right_content_height)
        if height_diff_ratio < 0.05:  # Content heights within 5%
            confidence_boost += 0.15
            if self.debug_mode:
                print(f"      ✅ Excellent content height consistency: {height_diff_ratio:.3f}")
        elif height_diff_ratio < 0.1:  # Within 10%
            confidence_boost += 0.1
            if self.debug_mode:
                print(f"      ✅ Good content height consistency: {height_diff_ratio:.3f}")
        
        # Check 4: Florence2 detection confidence
        left_florence_conf = left_data.get("confidence", 0)
        right_florence_conf = right_data.get("confidence", 0)
        avg_florence_conf = (left_florence_conf + right_florence_conf) / 2
        
        if avg_florence_conf > 0.8:
            confidence_boost += 0.1
            if self.debug_mode:
                print(f"      ✅ High Florence2 detection confidence: {avg_florence_conf:.3f}")
        
        return min(confidence_boost, 0.5)  # Cap total boost at 0.5

    def _get_florence2_analysis_for_image(self, img: ExtractedImage) -> Optional[Dict]:
        """Retrieve Florence2 analysis data for a specific image"""
        cache_key = (img.page_num, img.image_index)
        florence2_data = self.florence2_analysis_cache.get(cache_key)
        
        if florence2_data and self.debug_mode:
            print(f"    📖 Retrieved Florence2 data for {img.filename}")
            print(f"        - Method: {florence2_data.get('method', 'unknown')}")
            print(f"        - Confidence: {florence2_data.get('confidence', 0):.3f}")
            print(f"        - Rectangles: {len(florence2_data.get('detected_rectangles', []))}")
        
        return florence2_data


    def _try_segmentation_cropping(self, image: Image.Image, analysis: Dict) -> Dict[str, Any]:
        """Try segmentation cropping with SAM2+Florence as primary, GroundingDINO+SAM as fallback"""
        
        # Method 1: Try SAM2+Florence first (new primary method)
        if self.sam2_florence and self.sam2_florence.available:
            if self.debug_mode:
                print("    🚀 Using SAM2+Florence segmentation (primary)")
            
            sam2_result = self.sam2_florence.segment_pil_image(
                image, 
                prompt="main image OR photograph OR illustration OR diagram OR chart OR map",
                confidence=0.3
            )
            
            if sam2_result["success"] and sam2_result.get("combined_mask") is not None:
                # Convert SAM2 mask to bounding boxes
                try:
                    boxes = get_individual_bounding_boxes_from_mask(
                        sam2_result["combined_mask"],
                        min_area_threshold=int(image.width * image.height * 0.02)
                    )
                    
                    if self.debug_mode:
                        print(f"    🎯 SAM2+Florence found {len(boxes)} bounding boxes")
                    
                    if boxes and len(boxes) > 0:
                        if len(boxes) == 1:
                            crop_box = self._create_crop_box_with_margin(
                                boxes[0], image.size, margin=self.config.crop_margin
                            )
                            method = "sam2_florence_single"
                        else:
                            crop_box = self._create_encompassing_crop_box(
                                boxes, image.size, margin=self.config.crop_margin * 2
                            )
                            method = "sam2_florence_multi"
                        
                        if self._validate_crop_box(crop_box, image.size):
                            cropped_image = image.crop(crop_box)
                            if self.debug_mode:
                                print(f"    ✅ SAM2+Florence cropping successful: {method}")
                            return {
                                "success": True,
                                "image": cropped_image,
                                "method": method,
                                "boxes_found": len(boxes)
                            }
                        else:
                            if self.debug_mode:
                                print(f"    ⚠️ SAM2+Florence found boxes but crop validation failed")
                    else:
                        if self.debug_mode:
                            print(f"    ⚠️ SAM2+Florence found no valid boxes")
                            
                except Exception as e:
                    if self.debug_mode:
                        print(f"    ❌ SAM2+Florence processing failed: {e}")
            else:
                if self.debug_mode:
                    print(f"    ⚠️ SAM2+Florence segmentation unsuccessful or no mask returned")
        else:
            if self.debug_mode:
                print("    ⚠️ SAM2+Florence not available, skipping primary method")
        
        # Method 2: Fallback to GroundingDINO+SAM (legacy method)
        if self.debug_mode:
            print("    🔄 Falling back to GroundingDINO+SAM segmentation")
        
        groundingdino_result = self._try_groundingdino_sam_cropping(image, analysis)
        if groundingdino_result["success"]:
            if self.debug_mode:
                print(f"    ✅ GroundingDINO+SAM fallback successful")
            return groundingdino_result
        
        # Final fallback
        if self.debug_mode:
            print("    ❌ All segmentation methods failed")
        return {"success": False, "reason": "All segmentation methods failed"}

    def _try_groundingdino_sam_cropping(self, image: Image.Image, analysis: Dict) -> Dict[str, Any]:
        """GroundingDINO+SAM cropping (fallback method, renamed from original _try_segmentation_cropping)"""
        if not (self.segmentation_analyzer and self.segmentation_analyzer.available):
            return {"success": False, "reason": "No segmentation analyzer"}
        
        segmentation_data = analysis.get("segmentation_data")
        if not segmentation_data:
            return {"success": False, "reason": "No segmentation data"}
        
        main_image_mask_tensor = segmentation_data.get("image_mask_combined")
        if main_image_mask_tensor is None:
            return {"success": False, "reason": "No image mask"}
        
        # Handle multi-object scenarios with COMPREHENSIVE error handling
        try:
            boxes = get_individual_bounding_boxes_from_mask(
                main_image_mask_tensor, 
                min_area_threshold=int(image.width * image.height * 0.02)  # 2% of image area
            )
            
            if self.debug_mode:
                print(f"    🔍 GroundingDINO+SAM cropping: Found {len(boxes)} bounding boxes")
                    
        except Exception as e:
            if self.debug_mode:
                print(f"    ❌ Error getting bounding boxes from mask: {e}")
                import traceback
                traceback.print_exc()
            return {"success": False, "reason": f"Bounding box extraction failed: {e}"}
        
        if not boxes:
            return {"success": False, "reason": "No valid bounding boxes"}
        
        # Strategy for multiple objects
        try:
            if len(boxes) == 1:
                # Single object - crop tightly
                crop_box = self._create_crop_box_with_margin(boxes[0], image.size, margin=self.config.crop_margin)
                method = "groundingdino_sam_single"
                
            elif len(boxes) <= 3:
                # Few objects - create encompassing crop
                crop_box = self._create_encompassing_crop_box(boxes, image.size, margin=self.config.crop_margin * 2)
                method = "groundingdino_sam_multi"
                
            else:
                # Many objects - might be full page, check coverage
                total_area = sum(box['area'] for box in boxes if isinstance(box, dict) and 'area' in box)
                page_area = image.width * image.height
                coverage = total_area / page_area if page_area > 0 else 0
                
                if coverage > 0.6:  # More than 60% coverage suggests full page
                    return {"success": False, "reason": "Full page coverage detected"}
                else:
                    # Take largest objects only
                    valid_boxes = [box for box in boxes if isinstance(box, dict) and 'area' in box]
                    if not valid_boxes:
                        return {"success": False, "reason": "No valid boxes with area"}
                    
                    largest_boxes = sorted(valid_boxes, key=lambda x: x['area'], reverse=True)[:3]
                    crop_box = self._create_encompassing_crop_box(largest_boxes, image.size, margin=self.config.crop_margin)
                    method = "groundingdino_sam_largest"
            
            # Validate crop box
            if self._validate_crop_box(crop_box, image.size):
                cropped_image = image.crop(crop_box)
                return {
                    "success": True, 
                    "image": cropped_image, 
                    "method": method,
                    "boxes_found": len(boxes)
                }
            else:
                return {"success": False, "reason": "Invalid crop box"}
                
        except Exception as e:
            if self.debug_mode:
                print(f"    ❌ Error in GroundingDINO+SAM cropping strategy: {e}")
            return {"success": False, "reason": f"Cropping strategy failed: {e}"}


    def _try_cv2_content_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Try CV2-based content detection"""
        if not CV2_AVAILABLE:
            return {"success": False, "reason": "CV2 not available"}
        
        detector = ContentAwareBorderDetector(self.debug_mode)
        borders = detector.detect_content_borders(image)
        
        if borders and self._validate_crop_box(borders, image.size):
            cropped_image = image.crop(borders)
            return {"success": True, "image": cropped_image}
        
        return {"success": False, "reason": "No content borders detected"}

    def _try_adaptive_background_removal(self, image: Image.Image) -> Dict[str, Any]:
        """Try adaptive background removal using color clustering"""
        try:
            img_array = np.array(image)
            
            # Convert to RGB if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Reshape for clustering
                pixels = img_array.reshape(-1, 3)
                
                # Use KMeans to find dominant colors (background candidates)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(5, len(np.unique(pixels.view(np.void), axis=0))), 
                            random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Identify background color (most common in border regions)
                background_color = self._identify_background_color(img_array, kmeans)
                
                if background_color is not None:
                    # Create mask for non-background pixels
                    color_diff = np.linalg.norm(img_array - background_color, axis=2)
                    content_mask = color_diff > 30  # Threshold for background similarity
                    
                    # Find bounding box of content
                    if np.any(content_mask):
                        coords = np.argwhere(content_mask)
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        
                        # Add margin
                        margin = 20
                        crop_box = (
                            max(0, x_min - margin),
                            max(0, y_min - margin),
                            min(image.width, x_max + margin + 1),
                            min(image.height, y_max + margin + 1)
                        )
                        
                        if self._validate_crop_box(crop_box, image.size):
                            cropped_image = image.crop(crop_box)
                            return {"success": True, "image": cropped_image}
            
        except ImportError:
            if self.debug_mode:
                print("    scikit-learn not available for adaptive background removal")
        except Exception as e:
            if self.debug_mode:
                print(f"    Adaptive background removal failed: {e}")
        
        return {"success": False, "reason": "Background removal failed"}

    def _minimal_border_cleanup(self, image: Image.Image) -> Image.Image:
        """Minimal border cleanup when other methods fail"""
        if not CV2_AVAILABLE:
            return image
        
        try:
            # Very conservative border detection
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            
            h, w = gray.shape
            border_size = min(30, min(h, w) // 20)  # Small border check
            
            # Check for obvious uniform borders
            crop_bounds = [0, 0, w, h]  # left, top, right, bottom
            
            # Top border
            if np.std(gray[:border_size, :]) < 10:  # Very uniform
                crop_bounds[1] = border_size
            
            # Bottom border  
            if np.std(gray[-border_size:, :]) < 10:
                crop_bounds[3] = h - border_size
            
            # Left border
            if np.std(gray[:, :border_size]) < 10:
                crop_bounds[0] = border_size
            
            # Right border
            if np.std(gray[:, -border_size:]) < 10:
                crop_bounds[2] = w - border_size
            
            # Only crop if we found definitive borders
            if (crop_bounds[0] > 0 or crop_bounds[1] > 0 or 
                crop_bounds[2] < w or crop_bounds[3] < h):
                return image.crop(tuple(crop_bounds))
            
        except Exception as e:
            if self.debug_mode:
                print(f"    Minimal border cleanup failed: {e}")
        
        return image

    def _create_crop_box_with_margin(self, box: Dict, image_size: Tuple[int, int], margin: int) -> Tuple[int, int, int, int]:
        """Create crop box from bounding box with margin"""
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        
        # Add margin
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(image_size[0], x + w + margin)
        bottom = min(image_size[1], y + h + margin)
        
        return (left, top, right, bottom)

    def _create_encompassing_crop_box(self, boxes: List[Dict], image_size: Tuple[int, int], margin: int) -> Tuple[int, int, int, int]:
        """Create encompassing crop box for multiple objects"""
        if not boxes:
            return (0, 0, image_size[0], image_size[1])
        
        # Find bounding box that encompasses all objects
        min_x = min(box['x'] for box in boxes)
        min_y = min(box['y'] for box in boxes)
        max_x = max(box['x'] + box['width'] for box in boxes)
        max_y = max(box['y'] + box['height'] for box in boxes)
        
        # Add margin
        left = max(0, min_x - margin)
        top = max(0, min_y - margin)
        right = min(image_size[0], max_x + margin)
        bottom = min(image_size[1], max_y + margin)
        
        return (left, top, right, bottom)

    def _identify_background_color(self, img_array: np.ndarray, kmeans) -> Optional[np.ndarray]:
        """Identify background color from clustering results"""
        try:
            # Check border regions to identify most likely background color
            h, w = img_array.shape[:2]
            border_width = min(20, min(h, w) // 10)
            
            # Sample border pixels
            border_pixels = []
            border_pixels.extend(img_array[:border_width, :].reshape(-1, 3))  # Top
            border_pixels.extend(img_array[-border_width:, :].reshape(-1, 3))  # Bottom
            border_pixels.extend(img_array[:, :border_width].reshape(-1, 3))  # Left
            border_pixels.extend(img_array[:, -border_width:].reshape(-1, 3))  # Right
            
            border_pixels = np.array(border_pixels)
            
            # Find which cluster center is most common in border regions
            border_labels = kmeans.predict(border_pixels)
            most_common_label = np.bincount(border_labels).argmax()
            
            return kmeans.cluster_centers_[most_common_label]
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error identifying background color: {e}")
            return None
            
    def _pixmap_to_pil_with_profile(self, pix) -> Image.Image:
        """Convert PyMuPDF pixmap to PIL Image with color management"""
        try:
            # Handle colorspace conversion carefully
            original_colorspace = pix.colorspace
            
            if pix.colorspace and pix.colorspace.n == 4:
                # CMYK to RGB conversion
                if self.debug_mode:
                    print(f"    Converting from {pix.colorspace.name} to RGB")
                old_pix = pix
                pix = fitz.Pixmap(fitz.csRGB, pix)
                old_pix = None
            
            # Convert to PIL
            import io
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Store original colorspace info
            if hasattr(pix, 'colorspace') and pix.colorspace:
                colorspace_name = pix.colorspace.name
                pil_image.info['pdf_colorspace'] = colorspace_name
                if self.debug_mode:
                    print(f"    Original PDF colorspace: {colorspace_name}")
            
            # Ensure the image has a color profile
            pil_image = self._ensure_color_profile(pil_image, 
                                                getattr(pix.colorspace, 'name', 'DeviceRGB') if pix.colorspace else 'DeviceRGB')
            
            return pil_image
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in profile-aware pixmap conversion: {e}")
            return self._pixmap_to_pil_original(pix)

    def _pixmap_to_pil_original(self, pix) -> Image.Image:
        """Original pixmap conversion as fallback"""
        import io
        img_data = pix.tobytes("ppm")
        return Image.open(io.BytesIO(img_data))


    def _validate_crop_box(self, crop_box: Tuple[int, int, int, int], 
                        image_size: Tuple[int, int]) -> bool:
        """Validate that crop box is reasonable"""
        left, top, right, bottom = crop_box
        width, height = image_size
        
        # Check bounds
        if left < 0 or top < 0 or right > width or bottom > height:
            return False
        
        # Check minimum size
        crop_width = right - left
        crop_height = bottom - top
        
        if crop_width < 50 or crop_height < 50:
            return False
        
        # Check that we're not cropping too much (keep at least 20% of original)
        crop_area = crop_width * crop_height
        original_area = width * height
        
        if crop_area < original_area * 0.2:
            return False
        
        return True


    def _save_enhanced_images(self, original_image: Image.Image, processed_image: Image.Image,
                            page_num: int, img_index: int, save_options: str, 
                            enhancement_strength: float, use_gpu_acceleration: bool) -> List[Tuple[str, Image.Image, bool]]:
        """Save images with color profile preservation"""
        saved_images = []
        
        # Get color profile info for debugging
        if self.debug_mode:
            profile_info = self._get_color_profile_info(original_image)
            print(f"DEBUG: Original image profile: {profile_info}")
        
        if save_options == "original_only":
            filename = f"page_{page_num + 1:03d}_img_{img_index + 1:02d}_raw_with_text.png"
            filepath = self.output_dir / filename
            
            # Save with profile preservation
            self._save_with_profile(original_image, filepath)
            saved_images.append((filename, original_image, False))
            
        elif save_options == "enhanced_only":
            enhanced_image = self.enhancer.enhance_for_sd_training(processed_image, enhancement_strength)
            
            # Ensure profile is preserved from processed_image
            enhanced_image = self._preserve_color_profile(processed_image, enhanced_image)
            
            filename = f"page_{page_num + 1:03d}_img_{img_index + 1:02d}_enhanced_clean.png"
            filepath = self.output_dir / filename
            
            self._save_with_profile(enhanced_image, filepath)
            saved_images.append((filename, enhanced_image, True))
            
        elif save_options == "both":
            # Original
            orig_filename = f"page_{page_num + 1:03d}_img_{img_index + 1:02d}_raw_with_text.png"
            orig_filepath = self.output_dir / orig_filename
            self._save_with_profile(original_image, orig_filepath)
            saved_images.append((orig_filename, original_image, False))
            
            # Enhanced
            enhanced_image = self.enhancer.enhance_for_sd_training(processed_image, enhancement_strength)
            enhanced_image = self._preserve_color_profile(processed_image, enhanced_image)
            
            enh_filename = f"page_{page_num + 1:03d}_img_{img_index + 1:02d}_enhanced_clean.png"
            enh_filepath = self.output_dir / enh_filename
            self._save_with_profile(enhanced_image, enh_filepath)
            saved_images.append((enh_filename, enhanced_image, True))
        
        return saved_images

    def _save_with_profile(self, image: Image.Image, filepath: Path):
        """Save image preserving color profile and using optimal settings"""
        try:
            save_kwargs = {
                "format": "PNG",
                "optimize": False,  # Don't optimize to preserve quality
                "compress_level": 1  # Minimal compression for speed
            }
            
            # Preserve ICC profile if present
            if hasattr(image, 'info') and 'icc_profile' in image.info:
                save_kwargs['icc_profile'] = image.info['icc_profile']
                if self.debug_mode:
                    print(f"    Saving with ICC profile: {len(image.info['icc_profile'])} bytes")
            
            image.save(filepath, **save_kwargs)
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error saving with profile: {e}, falling back to standard save")
            image.save(filepath, "PNG")

    def _preserve_color_profile(self, source_image: Image.Image, target_image: Image.Image) -> Image.Image:
        """Preserve color profile from source to target image"""
        try:
            # Get ICC profile from source
            if hasattr(source_image, 'info') and 'icc_profile' in source_image.info:
                icc_profile = source_image.info['icc_profile']
                
                # Apply to target image
                if target_image.mode in ('RGB', 'CMYK', 'L'):
                    target_image.info['icc_profile'] = icc_profile
                    if self.debug_mode:
                        print(f"    Color profile preserved: {len(icc_profile)} bytes")
                
            return target_image
        except Exception as e:
            if self.debug_mode:
                print(f"    Warning: Could not preserve color profile: {e}")
            return target_image

    def _get_color_profile_info(self, image: Image.Image) -> Dict[str, Any]:
        """Get color profile information from image"""
        profile_info = {
            "has_profile": False,
            "profile_size": 0,
            "color_mode": image.mode,
            "profile_description": None
        }
        
        try:
            if hasattr(image, 'info') and 'icc_profile' in image.info:
                profile_info["has_profile"] = True
                profile_info["profile_size"] = len(image.info['icc_profile'])
                
                # Try to get profile description
                try:
                    from PIL import ImageCms
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info['icc_profile']))
                    profile_info["profile_description"] = profile.profile.profile_description
                except:
                    pass
        except Exception as e:
            if self.debug_mode:
                print(f"Error reading color profile: {e}")
        
        return profile_info

    def _debug_color_info(self, image: Image.Image, context: str = ""):
        """Debug color information"""
        if not self.debug_mode:
            return
        
        profile_info = self._get_color_profile_info(image)
        colorspace_info = image.info.get('pdf_colorspace', 'Unknown')
        
        print(f"🎨 Color Info {context}:")
        print(f"    Mode: {profile_info['color_mode']}")
        print(f"    PDF Colorspace: {colorspace_info}")
        print(f"    Has ICC Profile: {profile_info['has_profile']}")
        if profile_info['has_profile']:
            print(f"    Profile Size: {profile_info['profile_size']} bytes")
            if profile_info['profile_description']:
                print(f"    Description: {profile_info['profile_description']}")
        else:
            print(f"    ⚠️  No ICC profile - colors may be unpredictable")

    def _ensure_color_profile(self, image: Image.Image, default_colorspace: str = "DeviceRGB") -> Image.Image:
        """Ensure image has a color profile, assign default sRGB if none exists"""
        
        # Check if image already has a profile
        if hasattr(image, 'info') and 'icc_profile' in image.info and image.info['icc_profile']:
            if self.debug_mode:
                print(f"    Image already has ICC profile ({len(image.info['icc_profile'])} bytes)")
            return image
        
        # FORCE profile creation
        try:
            from PIL import ImageCms
            
            if image.mode == 'RGB':
                # Try to get a proper sRGB profile
                try:
                    # Method 1: Create sRGB profile using PIL
                    srgb_profile = ImageCms.createProfile('sRGB')
                    icc_profile = srgb_profile.tobytes()
                    
                    if self.debug_mode:
                        print(f"    Created sRGB profile using ImageCms.createProfile")
                        
                except Exception as e:
                    if self.debug_mode:
                        print(f"    ImageCms.createProfile failed: {e}, trying embedded sRGB")
                    
                    # Method 2: Use a minimal embedded sRGB profile (base64 encoded)
                    icc_profile = self._get_embedded_srgb_profile()
                
                if icc_profile:
                    if not hasattr(image, 'info'):
                        image.info = {}
                    image.info['icc_profile'] = icc_profile
                    
                    if self.debug_mode:
                        print(f"    🎨 FORCED sRGB profile assignment ({len(icc_profile)} bytes)")
            
            elif image.mode == 'L':
                try:
                    # Create Gray profile
                    gray_profile = ImageCms.createProfile('GRAY_D50')
                    icc_profile = gray_profile.tobytes()
                except:
                    # Fallback to a basic grayscale profile
                    icc_profile = self._get_embedded_gray_profile()
                
                if icc_profile:
                    if not hasattr(image, 'info'):
                        image.info = {}
                    image.info['icc_profile'] = icc_profile
                    
                    if self.debug_mode:
                        print(f"    🎨 FORCED Gray profile assignment ({len(icc_profile)} bytes)")
            
        except ImportError:
            if self.debug_mode:
                print(f"    ❌ PIL ImageCms not available - cannot assign color profile")
        except Exception as e:
            if self.debug_mode:
                print(f"    ❌ Could not assign color profile: {e}")
        
        return image

    def _get_embedded_srgb_profile(self) -> bytes:
        """Get a minimal embedded sRGB ICC profile"""
        # This is a minimal sRGB ICC profile (base64 encoded)
        # You can generate this from any image with an sRGB profile
        srgb_profile_b64 = """
        AAAMSEFEQkUCEAAAbW50clJHQiBYWVogB84AAgAJAAYAMQAAYWNzcEFQUEwAAAAAbm9uZQAAAAAA
        AAAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1BREJFAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEWRlc2MAAAFQAAAAeWNwcnQAAAHMAAAAPIGd
        bWRkAAAClAAAAIhnWVlaAAACqAAAABRnVFJDAAACwAAAAA5nQk1HAAABWAAAABdnQjJHAAACyAAA
        AA5gQjJIAAABWAAAABdnUnNDAAAC2AAAAA5nUmVsAAAB2AAAABdnUnNMAAADYAAAABRgVGVzdAAA
        A3QAAAAUYWx1ZgAAA4gAAAAMZG1uZAAAA5QAAABwZG1kZAAAAlQAAACI
        """
        
        try:
            import base64
            return base64.b64decode(srgb_profile_b64.replace('\n', '').replace(' ', ''))
        except:
            return b''  # Return empty if decoding fails

    def _get_embedded_gray_profile(self) -> bytes:
        """Get a minimal embedded grayscale ICC profile"""
        # Minimal Gray D50 profile (you'd need to generate this)
        # For now, return empty and let the system handle it
        return b''

    def _save_segmentation_masks(self, segmentation_data: Dict, page_num: int, img_index: int):
        """Save segmentation masks for debugging purposes"""
        if not segmentation_data:
            return
        
        try:
            # Create debug subdirectory
            debug_dir = self.output_dir / "debug_masks"
            debug_dir.mkdir(exist_ok=True)
            
            base_filename = f"page_{page_num + 1:03d}_img_{img_index + 1:02d}"
            
            # Save image segmentation mask if available
            if segmentation_data.get("image_mask_combined") is not None:
                image_mask = segmentation_data["image_mask_combined"]
                if self.debug_mode:
                    print(f"    💾 Saving image segmentation mask: {image_mask.shape}")
                
                # Convert tensor mask to PIL image
                mask_pil = tensor_mask_to_pil(image_mask)
                if mask_pil:
                    mask_filename = debug_dir / f"{base_filename}_image_mask.png"
                    mask_pil.save(mask_filename)
                    if self.debug_mode:
                        print(f"    ✅ Saved image mask: {mask_filename}")
            
            # Save text segmentation mask if available
            if segmentation_data.get("text_mask_combined") is not None:
                text_mask = segmentation_data["text_mask_combined"]
                if self.debug_mode:
                    print(f"    💾 Saving text segmentation mask: {text_mask.shape}")
                
                # Convert tensor mask to PIL image
                mask_pil = tensor_mask_to_pil(text_mask)
                if mask_pil:
                    mask_filename = debug_dir / f"{base_filename}_text_mask.png"
                    mask_pil.save(mask_filename)
                    if self.debug_mode:
                        print(f"    ✅ Saved text mask: {mask_filename}")
            
            # Save bounding box visualization if we have boxes
            if (segmentation_data.get("image_boxes_cxcywh_norm") is not None or 
                segmentation_data.get("text_boxes_cxcywh_norm") is not None):
                
                # This would require the original image to draw boxes on
                # We'll save box coordinates as JSON instead
                boxes_data = {}
                
                if segmentation_data.get("image_boxes_cxcywh_norm") is not None:
                    boxes_data["image_boxes"] = segmentation_data["image_boxes_cxcywh_norm"].tolist()
                    boxes_data["image_scores"] = segmentation_data.get("image_scores", []).tolist() if segmentation_data.get("image_scores") is not None else []
                
                if segmentation_data.get("text_boxes_cxcywh_norm") is not None:
                    boxes_data["text_boxes"] = segmentation_data["text_boxes_cxcywh_norm"].tolist()
                    boxes_data["text_scores"] = segmentation_data.get("text_scores", []).tolist() if segmentation_data.get("text_scores") is not None else []
                
                if boxes_data:
                    import json
                    boxes_filename = debug_dir / f"{base_filename}_boxes.json"
                    with open(boxes_filename, 'w') as f:
                        json.dump(boxes_data, f, indent=2)
                    if self.debug_mode:
                        print(f"    ✅ Saved bounding boxes: {boxes_filename}")
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ❌ Error saving segmentation masks: {e}")


    def _process_with_pypdf2_enhanced(self, pdf_path: Path, report: ProcessingReport,
                                    extract_images: bool, extract_text: bool, 
                                    save_options: str, enable_smart_crop: bool, enhancement_strength: float):
        """Enhanced PyPDF2 processing (fallback method)"""
        print("ℹ️ Using PyPDF2 enhanced mode - limited capabilities")
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            report.total_pages = len(reader.pages)
            
            for page_num, page in enumerate(reader.pages):
                if extract_text:
                    self._extract_text_pypdf2(page, page_num, report)
                
                if extract_images:
                    self._extract_images_pypdf2_enhanced(
                        page, page_num, report, save_options, 
                        enable_smart_crop, enhancement_strength
                    )
    
    def _extract_images_pypdf2_enhanced(self, page, page_num: int, report: ProcessingReport, 
                                      save_options: str, enable_smart_crop: bool, enhancement_strength: float):
        """Enhanced PyPDF2 image extraction"""
        # Simplified enhanced extraction for PyPDF2
        # Implementation would be similar to original but with enhancement pipeline
        pass
    
    def _extract_text_pymupdf(self, page, page_num: int, report: ProcessingReport):
        """Extract text using PyMuPDF"""
        try:
            text_content = page.get_text()
            
            if text_content.strip():
                extracted_text = ExtractedText(
                    page_num=page_num,
                    text_content=text_content,
                    is_ocr_layer=False
                )
                
                report.extracted_text.append(extracted_text)
                report.text_extracted_pages += 1
                
                # Save text to file
                text_file = self.output_dir / f"page_{page_num + 1:03d}_text.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                    
        except Exception as e:
            print(f"Error extracting text from page {page_num + 1}: {e}")
    
    def _enhanced_double_page_joining(self, report: ProcessingReport):
        """Enhanced double-page joining using intelligent analysis"""
        if self.debug_mode:
            print(f"\n🔗 Analyzing {len(report.extracted_images)} images for double-page spreads...")
            self._print_florence2_cache_stats()

        # Group images by page type
        left_pages = [img for img in report.extracted_images if img.page_type == PageType.DOUBLE_PAGE_LEFT]
        right_pages = [img for img in report.extracted_images if img.page_type == PageType.DOUBLE_PAGE_RIGHT]
        
        if self.debug_mode:
            print(f"Found {len(left_pages)} left pages and {len(right_pages)} right pages")
        
        # Try to match consecutive left and right pages
        for left_img in left_pages:
            # Look for corresponding right page (next page number)
            right_candidates = [
                right_img for right_img in right_pages 
                if right_img.page_num == left_img.page_num + 1
            ]
            
            if not right_candidates:
                continue
            
            # Take the best matching right page (highest confidence)
            right_img = max(right_candidates, key=lambda x: x.border_confidence)
            
            # Additional validation
            if self._validate_spread_pair_with_florence2(left_img, right_img):
                join_result = self._perform_enhanced_join(left_img, right_img, report.output_directory)
                
                if join_result:
                    report.joined_images.append(join_result)
                    report.images_joined += 1
                    
                    if self.debug_mode:
                        print(f"  ✅ Joined {left_img.filename} + {right_img.filename} → {join_result.filename}")
        
        if report.images_joined > 0:
            print(f"🎉 Successfully created {report.images_joined} enhanced joined images!")
        elif self.debug_mode:
            print("📄 No suitable double-page spreads detected with current thresholds")
    
    def _validate_spread_pair(self, left_img: ExtractedImage, right_img: ExtractedImage) -> bool:
        """Validate if a pair of images is a good candidate for a spread."""
        
        # Check 1: Must be consecutive pages
        if right_img.page_num != left_img.page_num + 1:
            if self.debug_mode:
                print(f"DEBUG: Spread validation fail for {left_img.filename} & {right_img.filename}: Not consecutive pages ({left_img.page_num}, {right_img.page_num})")
            return False
        
        # Check 2: Both must have reasonable confidence
        min_req_confidence = min(self.config.border_confidence_threshold * 0.8, 0.5)
        if left_img.border_confidence < min_req_confidence or right_img.border_confidence < min_req_confidence:
            if self.debug_mode:
                print(f"DEBUG: Spread validation fail for {left_img.filename} & {right_img.filename}: Low confidence (L: {left_img.border_confidence:.2f}, R: {right_img.border_confidence:.2f} vs Thresh: {min_req_confidence:.2f})")
            return False
        
        # Check 3: Height similarity check - FIXED
        if left_img.height == 0 or right_img.height == 0:
            if self.debug_mode:
                print(f"DEBUG: Spread validation fail for {left_img.filename} & {right_img.filename}: Zero height detected.")
            return False
        
        max_height = max(left_img.height, right_img.height)
        height_diff_ratio = abs(left_img.height - right_img.height) / max_height
        if height_diff_ratio > 0.1:  # More than 10% height difference
            if self.debug_mode:
                print(f"DEBUG: Spread validation fail for {left_img.filename} & {right_img.filename}: Height difference too large ({height_diff_ratio:.2f})")
            return False
        
        # Check 4: Combined aspect ratio should be reasonable for a spread - FIXED
        if left_img.height == 0:
            if self.debug_mode:
                print(f"DEBUG: Spread validation fail for {left_img.filename} & {right_img.filename}: Left image height is zero.")
            return False
        
        combined_width = left_img.width + right_img.width
        combined_ratio = combined_width / left_img.height
        if not (1.5 < combined_ratio < 4.0):  # Reasonable spread ratios
            if self.debug_mode:
                print(f"DEBUG: Spread validation fail for {left_img.filename} & {right_img.filename}: Combined ratio out of range ({combined_ratio:.2f})")
            return False
        
        if self.debug_mode:
            print(f"DEBUG: Spread validation PASS for {left_img.filename} & {right_img.filename}")
        return True


    def _perform_enhanced_join(self, left_img: ExtractedImage, right_img: ExtractedImage, 
                              output_dir: str) -> Optional[JoinedImage]:
        """Perform enhanced image joining using constrained stitching"""
        try:
            left_path = Path(output_dir) / left_img.filename
            right_path = Path(output_dir) / right_img.filename
            
            if not (left_path.exists() and right_path.exists()):
                return None
            
            left_pil = Image.open(left_path)
            right_pil = Image.open(right_path)
            
            if self.debug_mode:
                print(f"    Joining {left_pil.size} + {right_pil.size}")
            
            # Use constrained stitching
            joined_pil = self.stitcher.stitch_magazine_spread(left_pil, right_pil)
            
            if joined_pil is None:
                return None
            
            # Calculate confidence based on successful join and image analysis
            confidence = min(left_img.border_confidence, right_img.border_confidence)
            
            joined_filename = f"page_{left_img.page_num + 1:03d}-{right_img.page_num + 1:03d}_joined_enhanced.png"
            joined_path = Path(output_dir) / joined_filename
            
            joined_pil.save(joined_path, "PNG")
            
            join_record = JoinedImage(
                filename=joined_filename,
                left_page_num=left_img.page_num,
                right_page_num=right_img.page_num,
                left_image_filename=left_img.filename,
                right_image_filename=right_img.filename,
                combined_width=joined_pil.width,
                combined_height=joined_pil.height,
                confidence_score=confidence,
                join_method="enhanced_constrained_stitching"
            )
            
            return join_record
            
        except Exception as e:
            if self.debug_mode:
                print(f"  ⚠️ Enhanced join failed for {left_img.filename} + {right_img.filename}: {e}")
            return None
    
    def _passes_size_filter(self, width: int, height: int) -> bool:
        """Check if image passes size filters"""
        if width < self.config.min_image_size or height < self.config.min_image_size:
            return False
        if width > self.config.max_image_width or height > self.config.max_image_height:
            return False
        if width * height < self.config.min_image_area:
            return False
        return True
    
    def _assess_image_quality(self, image: Image.Image) -> ImageQuality:
        """Assess image quality"""
        try:
            gray = image.convert('L')
            img_array = np.array(gray)
            
            variance = np.var(img_array)
            size_score = min(image.width * image.height / 100000, 1.0)
            contrast_score = min(variance / 1000, 1.0)
            overall_score = (size_score + contrast_score) / 2
            
            if overall_score > 0.8:
                return ImageQuality.EXCELLENT
            elif overall_score > 0.6:
                return ImageQuality.GOOD
            elif overall_score > 0.4:
                return ImageQuality.FAIR
            elif overall_score > 0.2:
                return ImageQuality.POOR
            else:
                return ImageQuality.UNUSABLE
        except Exception:
            return ImageQuality.FAIR
    
    def _is_enhanced_multi_page_candidate(self, image: Image.Image, page_type: PageType) -> bool:
        """Enhanced multi-page candidate detection"""
        # Use page type analysis instead of just aspect ratio
        return page_type in [PageType.DOUBLE_PAGE_LEFT, PageType.DOUBLE_PAGE_RIGHT]
    
    def _get_file_size(self, filename: str) -> int:
        """Get file size for a given filename"""
        try:
            file_path = self.output_dir / filename
            if self.debug_mode:
                print(f"DEBUG: Checking file size for {file_path}, exists: {file_path.exists()}")
            if file_path.exists():
                size = file_path.stat().st_size
                if self.debug_mode:
                    print(f"DEBUG: File size: {size} bytes")
                return size
            else:
                if self.debug_mode:
                    print(f"DEBUG: File does not exist: {file_path}")
                return 0
        except Exception as e:
            if self.debug_mode:
                print(f"DEBUG: Error getting file size for {filename}: {e}")
            return 0

    def _cleanup_florence2_cache(self):
        """Clean up Florence2 analysis cache to free memory"""
        if self.florence2_analysis_cache:
            cache_size = len(self.florence2_analysis_cache)
            self.florence2_analysis_cache.clear()
            if self.debug_mode:
                print(f"🧹 Cleared Florence2 analysis cache ({cache_size} entries)")

    def _print_florence2_cache_stats(self):
        """Print Florence2 cache statistics for debugging"""
        if not self.debug_mode:
            return
        
        cache_size = len(self.florence2_analysis_cache)
        print(f"\n📊 Florence2 Cache Statistics:")
        print(f"    Total entries: {cache_size}")
        
        if cache_size > 0:
            successful_detections = sum(1 for data in self.florence2_analysis_cache.values() 
                                    if data.get("confidence", 0) > 0.5)
            print(f"    Successful detections: {successful_detections}")
            print(f"    Success rate: {successful_detections/cache_size*100:.1f}%")
            
            # Show sample of stored data
            sample_keys = list(self.florence2_analysis_cache.keys())[:3]
            for key in sample_keys:
                data = self.florence2_analysis_cache[key]
                print(f"    Sample [{key}]: {data.get('method', 'unknown')} conf={data.get('confidence', 0):.3f}")



# Node registration
NODE_CLASS_MAPPINGS = {
    "Eric_PDF_Extractor_Enhanced_V06": EnhancedPDFExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Eric_PDF_Extractor_Enhanced_V06": "Eric PDF Extractor Enhanced v0.6.0"
}