"""
Visual Box Comparison 2

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

#!/usr/bin/env python3
"""
Comprehensive Multi-Method Image Detection Analysis - FIXED VERSION
Shows Florence2, Surya Layout, SAM+GroundingDINO results for comparison
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import importlib.util
import tempfile
import cv2

try:
    import pymupdf as fitz
except ImportError:
    import fitz

# Import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available")

# Import analysis engine with proper path management
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from analysis_engine import analyze_for_pdf_extraction, set_comfyui_base_path
    ANALYSIS_ENGINE_AVAILABLE = True
    print("✅ Analysis engine imported")
except ImportError as e:
    print(f"❌ Analysis engine import failed: {e}")
    ANALYSIS_ENGINE_AVAILABLE = False


# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_BASE = os.path.dirname(folder_paths.models_dir)
    print(f"✅ ComfyUI models directory: {folder_paths.models_dir}")
except ImportError:
    COMFYUI_BASE = "A:\\Comfy_Dec\\ComfyUI"  # Your known path
    print(f"⚠️ ComfyUI folder_paths not available, using fallback: {COMFYUI_BASE}")
    
    # Create mock folder_paths for compatibility
    class MockFolderPaths:
        models_dir = os.path.join(COMFYUI_BASE, "models")
    
    folder_paths = MockFolderPaths()

# Initialize results dictionary
results = {
    "total_pairs": 0,
    "successful_analyses": 0,
    "total_florence2_boxes": 0,
    "total_surya_regions": 0,
    "total_sam_dino_detections": 0,
    "all_crop_info": []
}


def import_sam_modules():
    """Import SAM modules from sam2_scripts"""
    global SAM_AVAILABLE
    
    if not TORCH_AVAILABLE:
        print("❌ SAM skipped - PyTorch not available")
        return False
    
    try:
        # Import from your actual sam2_scripts
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from sam2_scripts.sam2_integration import SAM2FlorenceIntegration
        print("✅ SAM imported from sam2_scripts.sam2_integration")
        return True
        
    except ImportError as e:
        print(f"❌ SAM import failed: {e}")
        return False

def import_groundingdino_modules():
    """Import GroundingDINO modules with fallback strategies"""
    global GROUNDINGDINO_AVAILABLE, local_groundingdino_SLConfig, local_groundingdino_build_model, local_groundingdino_clean_state_dict
    
    try:
        # Strategy 1: Try local GroundingDINO
        try:
            from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
            from local_groundingdino.models import build_model as local_groundingdino_build_model
            from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
            print("✅ GroundingDINO imported successfully (local)")
            return True
        except ImportError:
            pass
        
        # Strategy 2: Try standard GroundingDINO
        try:
            from groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
            from groundingdino.models import build_model as local_groundingdino_build_model
            from groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
            print("✅ GroundingDINO imported successfully (standard)")
            return True
        except ImportError:
            pass
        
        print("❌ GroundingDINO not available")
        return False
        
    except Exception as e:
        print(f"❌ GroundingDINO import failed: {e}")
        return False

# Initialize imports
SAM_AVAILABLE = import_sam_modules()
GROUNDINGDINO_AVAILABLE = import_groundingdino_modules()

# Try to import ComfyUI folder paths for model locations with fallback
try:
    import folder_paths
    COMFYUI_BASE = os.path.dirname(folder_paths.models_dir)
    print(f"✅ ComfyUI models directory: {folder_paths.models_dir}")
except ImportError:
    COMFYUI_BASE = "A:\\Comfy_Dec\\ComfyUI"  # Your known path
    print(f"⚠️ ComfyUI folder_paths not available, using fallback: {COMFYUI_BASE}")
    
    # Create mock folder_paths for compatibility
    class MockFolderPaths:
        models_dir = os.path.join(COMFYUI_BASE, "models")
    
    folder_paths = MockFolderPaths()

def get_large_font():
    """Get a larger font for better visibility"""
    try:
        # Try to get a larger system font
        return ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            return ImageFont.load_default()

def get_even_odd_pairs(doc) -> List[Tuple[int, int]]:
    """Get all even-odd page pairs (skip cover page)"""
    pairs = []
    # Start from page 2 (index 1) since page 1 (index 0) is cover
    for i in range(1, len(doc) - 1, 2):
        left_idx = i      # Even page (2,4,6...) = left page
        right_idx = i + 1 # Odd page (3,5,7...) = right page
        if right_idx < len(doc):
            pairs.append((left_idx, right_idx))
    return pairs

# Replace the SimpleSegmentationAnalyzer class with this working implementation from your PDF extractor:

class SegmentationAnalyzer:
    """Analyzes images using GroundingDINO and SAM for object segmentation - copied from PDF extractor"""
    
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
        
        try:
            if not GROUNDINGDINO_AVAILABLE:
                if self.debug_mode:
                    print("SegmentationAnalyzer: GroundingDINO not available")
                return
            
            # Load DINO model
            dino_config_path, dino_model_path = self._resolve_dino_paths(dino_config_name)
            if dino_config_path and dino_model_path:
                self.dino_model = GroundingDINOModel(
                    dino_config_path, dino_model_path, self.device, debug_mode
                )
                if self.debug_mode:
                    print(f"SegmentationAnalyzer: GroundingDINO model loaded successfully")
            
            # Load SAM2 model for segmentation
            sam2_loaded = self._load_sam2_model(sam_model_name)
            if sam2_loaded and self.debug_mode:
                print("SegmentationAnalyzer: SAM2 model loaded successfully")
            
            # Set availability based on GroundingDINO (SAM2 is optional enhancement)
            if self.dino_model:
                self.available = True
                if self.debug_mode:
                    sam_status = "with SAM2" if sam2_loaded else "without SAM2"
                    print(f"SegmentationAnalyzer: Initialization successful - GroundingDINO {sam_status}")
        
        except Exception as e:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Initialization error: {e}")
            self.available = False

    def _load_sam2_model(self, sam_model_name: Optional[str] = None) -> bool:
        """Load SAM2 model from your local sam2 directory"""
        try:
            # Find SAM2 model in your models/sam2/ directory
            sam2_dir = Path(folder_paths.models_dir) / "sam2"
            if not sam2_dir.exists():
                if self.debug_mode:
                    print("SegmentationAnalyzer: SAM2 directory not found")
                return False
            
            # Preferred model order (largest to smallest)
            preferred_models = [
                "sam2_hiera_large.safetensors",
                "sam2.1_hiera_base_plus-fp16.safetensors", 
                "sam2_hiera_base.safetensors",
                "sam2_hiera_small.safetensors",
                "sam2_hiera_tiny.safetensors"
            ]
            
            model_path = None
            if sam_model_name:
                # User specified a model
                specified_path = sam2_dir / sam_model_name
                if specified_path.exists():
                    model_path = str(specified_path)
            else:
                # Auto-detect best available model
                for model_name in preferred_models:
                    candidate_path = sam2_dir / model_name
                    if candidate_path.exists():
                        model_path = str(candidate_path)
                        break
            
            if not model_path:
                if self.debug_mode:
                    print("SegmentationAnalyzer: No SAM2 model found in models/sam2/")
                return False
            
            # Try to load SAM2 using kijai's implementation (like your PDF extractor)
            try:
                # Import kijai's SAM2 nodes
                import sys
                sam2_path = os.path.join(COMFYUI_BASE, "custom_nodes", "ComfyUI-segment-anything-2")
                if sam2_path not in sys.path:
                    sys.path.append(sam2_path)
                
                from nodes.sam2_nodes import SAM2Model, load_sam2_model
                
                # Load the model
                self.sam_model = load_sam2_model(model_path, self.device)
                
                # Create predictor interface
                if hasattr(self.sam_model, 'set_image'):
                    self.sam_predictor = self.sam_model
                else:
                    # Wrap in a predictor interface
                    self.sam_predictor = SAM2Predictor(self.sam_model)
                
                if self.debug_mode:
                    print(f"SegmentationAnalyzer: SAM2 loaded from {model_path}")
                return True
                
            except ImportError as e:
                if self.debug_mode:
                    print(f"SegmentationAnalyzer: SAM2 nodes not available: {e}")
                # Try direct SAM2 import as fallback
                return self._load_sam2_direct(model_path)
                
        except Exception as e:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: SAM2 loading error: {e}")
            return False

    def _load_sam2_direct(self, model_path: str) -> bool:
        """Fallback SAM2 loading (if available)"""
        try:
            # This would require direct SAM2 installation
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Determine config based on model name
            model_name = Path(model_path).name.lower()
            if "large" in model_name:
                config = "sam2_hiera_l.yaml"
            elif "base" in model_name:
                config = "sam2_hiera_b+.yaml"
            elif "small" in model_name:
                config = "sam2_hiera_s.yaml"
            elif "tiny" in model_name:
                config = "sam2_hiera_t.yaml"
            else:
                config = "sam2_hiera_l.yaml"  # Default
            
            # Build SAM2 model
            self.sam_model = build_sam2(config, model_path, device=self.device)
            self.sam_predictor = SAM2ImagePredictor(self.sam_model)
            
            if self.debug_mode:
                print(f"SegmentationAnalyzer: SAM2 loaded directly from {model_path}")
            return True
            
        except ImportError:
            if self.debug_mode:
                print("SegmentationAnalyzer: Direct SAM2 not available")
            return False

    def get_segmentation_results(self, image_pil: Image.Image, 
                                image_prompt: Optional[str] = None, 
                                text_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced segmentation with SAM2 refinement of GroundingDINO detections"""
        results = {
            "image_boxes_cxcywh_norm": None, "image_scores": None, "image_mask_combined": None,
            "text_boxes_cxcywh_norm": None, "text_scores": None, "text_mask_combined": None,
        }

        if not self.available:
            if self.debug_mode: 
                print("SegmentationAnalyzer.get_segmentation_results: Analyzer not available.")
            return results

        if image_prompt:
            if self.debug_mode: 
                print(f"SegmentationAnalyzer: Running image segmentation with prompt: '{image_prompt}'")
            img_result = self._run_single_segmentation_pass(image_pil, image_prompt, "image")
            results["image_boxes_cxcywh_norm"] = img_result.get("boxes_cxcywh_norm")
            results["image_scores"] = img_result.get("scores")
            results["image_mask_combined"] = img_result.get("mask_combined")
        
        if text_prompt:
            if self.debug_mode: 
                print(f"SegmentationAnalyzer: Running text segmentation with prompt: '{text_prompt}'")
            txt_result = self._run_single_segmentation_pass(image_pil, text_prompt, "text")
            results["text_boxes_cxcywh_norm"] = txt_result.get("boxes_cxcywh_norm")
            results["text_scores"] = txt_result.get("scores")
            results["text_mask_combined"] = txt_result.get("mask_combined")
            
        return results

    def _run_single_segmentation_pass(self, image: Image.Image, prompt: str, 
                                    segmentation_type: str = "image") -> Dict[str, Any]:
        """Run GroundingDINO detection with optional SAM2 refinement"""
        
        if not self.available:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Not available for prompt: '{prompt}'")
            return {"boxes_cxcywh_norm": None, "scores": None, "mask_combined": None}
        
        try:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Running {segmentation_type} detection with prompt: '{prompt}'")
            
            # Step 1: GroundingDINO detection
            boxes_xyxy, scores = self.dino_model.predict(
                image, prompt, 
                box_threshold=0.3,  # Same thresholds as PDF extractor
                text_threshold=0.25
            )
            
            if self.debug_mode:
                print(f"SegmentationAnalyzer: DINO found {boxes_xyxy.size(0)} boxes for '{prompt}'.")
            
            if boxes_xyxy.size(0) == 0:
                return {"boxes_cxcywh_norm": None, "scores": None, "mask_combined": None}
            
            # Step 2: SAM2 refinement (if available)
            detections = []
            if self.sam_predictor:
                # Convert PIL to numpy for SAM2
                image_np = np.array(image.convert("RGB"))
                self.sam_predictor.set_image(image_np)
                
                for i, (box, score) in enumerate(zip(boxes_xyxy, scores)):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    confidence = score.cpu().numpy()
                    
                    # Validate coordinates
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    try:
                        # Use SAM2 to refine the detection
                        masks, sam_scores, _ = self.sam_predictor.predict(
                            box=np.array([x1, y1, x2, y2]),
                            multimask_output=False
                        )
                        
                        # Use the best mask to refine bounding box
                        if len(masks) > 0 and masks[0] is not None:
                            mask = masks[0]
                            # Find tight bounding box around mask
                            coords = np.where(mask)
                            if len(coords[0]) > 0 and len(coords[1]) > 0:
                                y_min, y_max = coords[0].min(), coords[0].max()
                                x_min, x_max = coords[1].min(), coords[1].max()
                                
                                # Use refined coordinates
                                x1, y1, x2, y2 = float(x_min), float(y_min), float(x_max), float(y_max)
                                
                                detections.append({
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "label": f"sam2_dino_{segmentation_type}_{prompt.split()[0]}",
                                    "confidence": float(confidence),
                                    "has_mask": True,
                                    "mask": mask,
                                    "sam_score": float(sam_scores[0]) if len(sam_scores) > 0 else 0.0
                                })
                            else:
                                # No valid mask, use original box
                                detections.append({
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "label": f"dino_{segmentation_type}_{prompt.split()[0]}",
                                    "confidence": float(confidence),
                                    "has_mask": False
                                })
                        else:
                            # No mask, use original box
                            detections.append({
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "label": f"dino_{segmentation_type}_{prompt.split()[0]}",
                                "confidence": float(confidence),
                                "has_mask": False
                            })
                            
                    except Exception as sam_error:
                        if self.debug_mode:
                            print(f"SAM2 processing failed for box {i}: {sam_error}")
                        # Fall back to DINO-only detection
                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "label": f"dino_{segmentation_type}_{prompt.split()[0]}",
                            "confidence": float(confidence),
                            "has_mask": False
                        })
                
                if self.debug_mode:
                    masked_count = sum(1 for d in detections if d.get('has_mask', False))
                    print(f"SegmentationAnalyzer: SAM2 refined {masked_count}/{len(detections)} detections")
            else:
                # DINO-only detections (no SAM2)
                for i, (box, score) in enumerate(zip(boxes_xyxy, scores)):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    confidence = score.cpu().numpy()
                    
                    # Validate coordinates
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "label": f"dino_{segmentation_type}_{prompt.split()[0]}",
                        "confidence": float(confidence),
                        "has_mask": False
                    })
                
                if self.debug_mode:
                    print(f"SegmentationAnalyzer: DINO-only detections (no SAM2)")
            
            return {
                "boxes_cxcywh_norm": boxes_xyxy,
                "scores": scores,
                "detections": detections
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"SegmentationAnalyzer: Error during detection for '{prompt}': {e}")
            return {"boxes_cxcywh_norm": None, "scores": None, "mask_combined": None}

# Add the GroundingDINOModel class from your PDF extractor:
class GroundingDINOModel:
    def __init__(self, config_path: str, model_path: str, device: str, debug_mode: bool = False):
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        self.debug_mode = debug_mode
        self.model = None
        
        try:
            if GROUNDINGDINO_AVAILABLE:
                dino_model_args = local_groundingdino_SLConfig.fromfile(config_path)
                
                if dino_model_args is None:
                    raise ValueError(f"Failed to load config from {config_path}")

                # Build model
                dino = local_groundingdino_build_model(dino_model_args)
                if dino is None:
                    raise ValueError("Failed to build model")
                
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location="cpu")
                
                # Extract model state dict
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # Load state dict
                load_result = dino.load_state_dict(local_groundingdino_clean_state_dict(state_dict), strict=False)
                
                # Move to device
                self.model = dino.to(device=self.device)
                self.model.eval()
                
                if self.debug_mode:
                    print("GroundingDINOModel: Model loaded successfully")

        except Exception as e:
            if self.debug_mode:
                print(f"GroundingDINOModel: Error loading model: {e}")
            raise

    def predict(self, image_pil: Image.Image, text_prompt: str, 
                box_threshold: float, text_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced prediction with proper preprocessing"""
        if self.model is None:
            return torch.empty((0, 4), device="cpu"), torch.empty((0,), device="cpu")

        # Import the transforms
        try:
            from local_groundingdino.datasets import transforms as T
        except ImportError:
            if self.debug_mode:
                print("Could not import transforms")
            return torch.empty((0, 4), device="cpu"), torch.empty((0,), device="cpu")

        def load_dino_image(image_pil):
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image, _ = transform(image_pil, None)
            return image

        def get_grounding_output(model, image, caption, box_threshold):
            caption = caption.lower().strip()
            if not caption.endswith("."):
                caption = caption + "."
            
            image = image.to(self.device)
            with torch.no_grad():
                outputs = model(image[None], captions=[caption])
            
            logits = outputs["pred_logits"].sigmoid()[0]
            boxes = outputs["pred_boxes"][0]
            
            # Filter output
            filt_mask = logits.max(dim=1)[0] > box_threshold
            boxes_filt = boxes[filt_mask]
            
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
            
            # Create scores
            scores = torch.ones(boxes_filt.size(0)) * box_threshold
            
            if self.debug_mode:
                print(f"GroundingDINOModel: Found {boxes_filt.size(0)} boxes")
            
            return boxes_filt, scores

        except Exception as e:
            if self.debug_mode:
                print(f"GroundingDINOModel: Prediction error: {e}")
            return torch.empty((0, 4), device="cpu"), torch.empty((0,), device="cpu")

def process_pdf(pdf_path: str):
    """Process PDF with enhanced GroundingDINO prompts"""
    
    if not ANALYSIS_ENGINE_AVAILABLE:
        print("❌ Analysis engine not available - cannot run comparison")
        return
    
    print(f"📖 Processing PDF: {pdf_path}")
    
    # Set ComfyUI path for Florence2
    try:
        set_comfyui_base_path(COMFYUI_BASE)
        print(f"✅ ComfyUI path set: {COMFYUI_BASE}")
    except Exception as e:
        print(f"⚠️ Could not set ComfyUI path: {e}")
    
    # Create output directory
    pdf_name = Path(pdf_path).stem
    output_dir = f"comprehensive_analysis_{pdf_name}_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize enhanced analyzer
    sam_analyzer = SegmentationAnalyzer(debug_mode=True)
    print(f"📊 Enhanced DINO analyzer available: {sam_analyzer.available}")
    
    with fitz.open(pdf_path) as doc:
        pairs = get_even_odd_pairs(doc)
        results["total_pairs"] = len(pairs)
        
        print(f"📊 Processing {len(pairs)} even-odd pairs...")
        
        for i, (left_idx, right_idx) in enumerate(pairs):
            pair_id = f"{left_idx+1:03d}_{right_idx+1:03d}"
            print(f"  🔄 Pair {i+1}/{len(pairs)}: pages {left_idx+1}-{right_idx+1}")
            
            try:
                # Extract and join pages
                print(f"    📄 Extracting pages {left_idx+1} and {right_idx+1}...")
                left_page = doc[left_idx]
                right_page = doc[right_idx]
                
                mat = fitz.Matrix(200/72, 200/72)
                left_pix = left_page.get_pixmap(matrix=mat, alpha=False)
                right_pix = right_page.get_pixmap(matrix=mat, alpha=False)
                
                left_img = Image.open(BytesIO(left_pix.tobytes("ppm")))
                right_img = Image.open(BytesIO(right_pix.tobytes("ppm")))
                
                # Join images
                joined_width = left_img.width + right_img.width
                joined_height = max(left_img.height, right_img.height)
                joined_img = Image.new('RGB', (joined_width, joined_height), 'white')
                joined_img.paste(left_img, (0, 0))
                joined_img.paste(right_img, (left_img.width, 0))
                print(f"    ✅ Images joined: {joined_img.size}")
                
                # Florence2 + Surya analysis
                print(f"    📊 Running Florence2 + Surya analysis...")
                try:
                    analysis = analyze_for_pdf_extraction(joined_img, debug_mode=False)
                    print(f"    ✅ Florence2/Surya complete: F2={len(analysis.get('florence2_rectangles', []))}, Surya={len(analysis.get('surya_layout', []))}")
                except Exception as analysis_error:
                    print(f"    ❌ Florence2/Surya analysis failed: {analysis_error}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Enhanced GroundingDINO analysis
                print(f"    📊 Running enhanced GroundingDINO analysis...")
                sam_dino_results = []
                if sam_analyzer.available:
                    try:
                        # Use enhanced prompts
                        image_segmentation_prompt = "photograph OR main image OR illustration OR diagram OR chart OR map"
                        
                        segmentation_results = sam_analyzer.get_segmentation_results(
                            image_pil=joined_img,
                            image_prompt=image_segmentation_prompt,
                            text_prompt=None
                        )
                        
                        # Extract detections
                        sam_dino_results = segmentation_results.get("detections", [])
                        
                        if sam_dino_results and sam_analyzer.debug_mode:
                            print(f"      ✅ Enhanced DINO found {len(sam_dino_results)} detections")
                    except Exception as e:
                        if sam_analyzer.debug_mode:
                            print(f"      ❌ Enhanced DINO detection error: {e}")
                        sam_dino_results = []
                else:
                    print(f"      ⚠️ Enhanced DINO analyzer not available")
                
                # Create detection results
                detection_results = {
                    "florence2_rectangles": analysis.get("florence2_rectangles", []),
                    "surya_layout": analysis.get("surya_layout", []),
                    "sam_dino_detections": sam_dino_results
                }
                
                # Create visualization and save crops
                try:
                    print(f"    🖼️ Creating visualization...")
                    create_comprehensive_visualization(
                        joined_img, detection_results, pair_id, output_dir
                    )
                    print(f"    ✅ Visualization saved")
                except Exception as viz_error:
                    print(f"    ❌ Visualization failed: {viz_error}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                try:
                    print(f"    💾 Saving crops...")
                    crop_info = save_all_crops(
                        joined_img, detection_results, pair_id, output_dir, analyze_for_pdf_extraction
                    )
                    results["all_crop_info"].extend(crop_info)
                    print(f"    ✅ Saved {len(crop_info)} crops")
                except Exception as crop_error:
                    print(f"    ❌ Crop saving failed: {crop_error}")
            
                # Update stats
                results["successful_analyses"] += 1
                results["total_florence2_boxes"] += len(detection_results["florence2_rectangles"])
                results["total_surya_regions"] += len(detection_results["surya_layout"])
                results["total_sam_dino_detections"] += len(detection_results["sam_dino_detections"])
                
                print(f"    ✅ Pair {i+1} complete: F2:{len(detection_results['florence2_rectangles'])} | Surya:{len(detection_results['surya_layout'])} | DINO:{len(detection_results['sam_dino_detections'])}")
                
            except Exception as e:
                print(f"    ❌ Error processing pair {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Print summary
    print(f"\n✅ Comprehensive Analysis Complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Processed: {results['successful_analyses']}/{results['total_pairs']} pairs")
    print(f"📦 Detection Summary:")
    print(f"   🔴 Florence2: {results['total_florence2_boxes']} boxes total")
    print(f"   🔵 Surya Layout: {results['total_surya_regions']} regions total")
    print(f"   🟢 Enhanced DINO: {results['total_sam_dino_detections']} detections total")
    print(f"   🖼️  Total crops: {len(results['all_crop_info'])}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python visual_box_comparison_2.py path/to/test.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    process_pdf(pdf_path)

if __name__ == "__main__":
    main()


class SAM2Predictor:
    """Wrapper to provide predictor interface for SAM2 models"""
    
    def __init__(self, sam2_model):
        self.model = sam2_model
        self.image = None
    
    def set_image(self, image: np.ndarray):
        """Set image for prediction"""
        self.image = image
        if hasattr(self.model, 'set_image'):
            self.model.set_image(image)
    
    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        """Run prediction"""
        if hasattr(self.model, 'predict'):
            return self.model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output
            )
        else:
            # Implement basic prediction logic if needed
            raise NotImplementedError("SAM2 model doesn't provide predict interface")

def create_comprehensive_visualization(joined_img: Image.Image, detection_results: Dict,
                                     pair_id: str, output_dir: str):
    """Create comprehensive visualization with all detection methods"""
    
    vis_img = joined_img.copy()
    draw = ImageDraw.Draw(vis_img)
    font = get_large_font()
    
    def validate_bbox(bbox):
        """Ensure bbox coordinates are valid"""
        x1, y1, x2, y2 = bbox
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        if x2 - x1 < 1:
            x2 = x1 + 1
        if y2 - y1 < 1:
            y2 = y1 + 1
            
        x1 = max(0, min(x1, vis_img.width - 1))
        y1 = max(0, min(y1, vis_img.height - 1))
        x2 = max(x1 + 1, min(x2, vis_img.width))
        y2 = max(y1 + 1, min(y2, vis_img.height))
        
        return [x1, y1, x2, y2]
    
    # Get results
    florence2_boxes = detection_results.get("florence2_rectangles", [])
    surya_regions = detection_results.get("surya_layout", [])
    sam_dino_boxes = detection_results.get("sam_dino_detections", [])
    
    # Draw Florence2 boxes in RED
    for i, box in enumerate(florence2_boxes):
        try:
            bbox = validate_bbox(box["bbox"])
            confidence = box.get("confidence", 1.0)
            draw.rectangle(bbox, outline="red", width=4)
            text = f"F2_{i+1}({confidence:.2f})"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x, text_y = bbox[0], bbox[1] - text_height - 5
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill="red")
            draw.text((text_x, text_y), text, fill="white", font=font)
        except Exception as e:
            print(f"⚠️ Error drawing Florence2 box {i}: {e}")
    
    # Draw Surya regions in BLUE 
    color_map = {
        "text": "blue", "title": "navy", "heading": "darkblue",
        "caption": "lightblue", "image": "cyan", "figure": "turquoise",
        "table": "purple", "header": "gray", "footer": "darkgray"
    }
    
    for i, region in enumerate(surya_regions):
        try:
            bbox = validate_bbox(region["bbox"])
            label = region.get("semantic_label", "unknown").lower()
            
            color = "blue"
            for key, c in color_map.items():
                if key in label:
                    color = c
                    break
            
            draw.rectangle(bbox, outline=color, width=3)
            text = f"S_{label[:6]}_{i+1}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = bbox[2] - text_width
            text_y = bbox[1]
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=color)
            draw.text((text_x, text_y), text, fill="white", font=font)
        except Exception as e:
            print(f"⚠️ Error drawing Surya region {i}: {e}")
    
    # Draw Enhanced DINO boxes in GREEN
    if sam_dino_boxes:
        for i, box in enumerate(sam_dino_boxes):
            try:
                bbox = validate_bbox(box["bbox"])
                confidence = box.get("confidence", 1.0)
                
                draw.rectangle(bbox, outline="lime", width=3)
                text = f"DINO_{i+1}({confidence:.2f})"
                draw.text((bbox[0], bbox[3] + 5), text, fill="lime", font=font)
            except Exception as e:
                print(f"⚠️ Error drawing Enhanced DINO box {i}: {e}")
    
    # Draw seam line
    seam_x = joined_img.width // 2
    draw.line([seam_x, 0, seam_x, joined_img.height], fill="yellow", width=5)
    
    # Add summary
    summary_lines = [
        f"Florence2: {len(florence2_boxes)} boxes",
        f"Surya: {len(surya_regions)} regions",
        f"Enhanced DINO: {len(sam_dino_boxes)} detections"
    ]
    
    y_pos = 10
    for line in summary_lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([10, y_pos, 10 + text_width + 10, y_pos + text_height + 5], fill="black")
        draw.text((15, y_pos + 2), line, fill="white", font=font)
        y_pos += text_height + 10
    
    # Save visualization
    vis_dir = os.path.join(output_dir, "comprehensive_analysis")
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, f"{pair_id}_comprehensive.png")
    vis_img.save(output_path)
    
    return output_path

def save_all_crops(joined_img: Image.Image, detection_results: Dict, 
                   pair_id: str, output_dir: str, analysis_func) -> List[Dict]:
    """Save crops from all detection methods"""
    
    crop_info = []
    
    # Save Florence2 crops
    florence2_boxes = detection_results.get("florence2_rectangles", [])
    for i, box in enumerate(florence2_boxes):
        try:
            bbox = box["bbox"]
            x1, y1, x2, y2 = bbox
            
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                continue
                
            crop = joined_img.crop((x1, y1, x2, y2))
            if crop.size[0] > 0 and crop.size[1] > 0:
                crop_filename = f"{pair_id}_F2_{i+1:03d}.png"
                crop_path = os.path.join(output_dir, "florence2_crops", crop_filename)
                os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                crop.save(crop_path)
                
                crop_info.append({
                    "source": "florence2",
                    "crop_path": crop_path,
                    "bbox": bbox,
                    "confidence": box.get("confidence", 1.0),
                    "pair_id": pair_id,
                    "crop_id": i+1
                })
        except Exception as e:
            print(f"⚠️ Error saving Florence2 crop {i}: {e}")
    
    return crop_info

