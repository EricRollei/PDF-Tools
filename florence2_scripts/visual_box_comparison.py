"""
Visual Box Comparison

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
Comprehensive Multi-Method Image Detection Analysis
Shows Florence2, Surya Layout, SAM+GroundingDINO results for comparison

Usage: python visual_box_comparison.py path/to/test.pdf

AI ASSISTANT INSTRUCTIONS:
- No simplified/alternate solutions that reduce functionality  
- No "disable X until Y is working" suggestions
- Implement complete, working solutions only
- Copy patterns from existing working code in this project
- Focus on robust, production-ready implementations
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

# ADD MISSING TORCH IMPORT
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

# Try to import SAM + GroundingDINO with enhanced strategy (like spread_detection)
SAM_AVAILABLE = False
GROUNDINGDINO_AVAILABLE = False

# Initialize results dictionary - ADD THIS
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
    global GROUNDINGDINO_AVAILABLE
    
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
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.models import build_model
            from groundingdino.util.utils import clean_state_dict
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

# Replace the SimpleSegmentationAnalyzer class with this direct implementation:

class SimpleSegmentationAnalyzer:
    """Direct SAM2 + GroundingDINO analyzer - no subprocess calls"""
    
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.available = False
        self.device = self._setup_device()
        
        # Model components
        self.sam2_model = None
        self.sam2_predictor = None
        self.dino_model = None
        
        if not TORCH_AVAILABLE:
            if debug_mode:
                print("❌ Torch not available - skipping SAM segmentation")
            return
        
        # Load models directly (no Florence2 to avoid conflicts)
        try:
            # Skip SAM2 loading since you deleted it
            sam2_loaded = False
            if debug_mode:
                print("⚠️ SAM2 loading skipped - models not installed")
            
            dino_loaded = self._load_groundingdino_model()
            
            # Make available with just GroundingDINO
            self.available = dino_loaded  # GroundingDINO only
            
            if debug_mode:
                print(f"📊 Direct model loading results:")
                print(f"   SAM2: {sam2_loaded} (skipped)")
                print(f"   GroundingDINO: {dino_loaded}")
                print(f"   Available: {self.available} (GroundingDINO only)")
                
        except Exception as e:
            if debug_mode:
                print(f"❌ Direct model loading failed: {e}")
    
    def _setup_device(self) -> str:
        """Setup compute device like your SAM2 script"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_sam2_model(self) -> bool:
        """Load SAM2 model - exact copy from working sam2_florence_segmentation.py"""
        try:
            model_path = self._find_sam2_model()
            if not model_path:
                if self.debug_mode:
                    print("❌ No SAM2 model files found")
                return False
            
            # Exact import strategy from your working script
            try:
                import sys
                import os
                
                # Add the exact path from your working script
                sam2_node_path = os.path.join(COMFYUI_BASE, "custom_nodes", "ComfyUI-segment-anything-2", "nodes")
                if sam2_node_path not in sys.path:
                    sys.path.insert(0, sam2_node_path)
                
                # Import the actual working module
                import sam2_nodes
                
                # Use the exact same function call
                self.sam2_model = sam2_nodes.load_sam2_model(model_path, self.device)
                
                # Create predictor exactly like your working script
                if hasattr(self.sam2_model, 'predict'):
                    self.sam2_predictor = self.sam2_model
                else:
                    self.sam2_predictor = SAM2Predictor(self.sam2_model)
                
                if self.debug_mode:
                    print(f"✅ SAM2 loaded: {model_path}")
                return True
                
            except Exception as e:
                if self.debug_mode:
                    print(f"❌ SAM2 loading failed: {e}")
                return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ SAM2 error: {e}")
            return False

    def _find_sam2_model(self) -> Optional[str]:
        """Find SAM2 model - copied from your working script"""
        try:
            sam_dir = Path(folder_paths.models_dir) / "sams"
            if not sam_dir.exists():
                return None
            
            # Look for actual SAM2 models (not SAM1)
            for pt_file in sam_dir.glob("*.pt"):
                if "sam2" in pt_file.name.lower():
                    return str(pt_file)
            
            # Fallback to any .pth file
            for pth_file in sam_dir.glob("*.pth"):
                return str(pth_file)
            
            return None
        except Exception:
            return None
    
    def _load_groundingdino_model(self) -> bool:
        """Load GroundingDINO model directly"""
        try:
            if not GROUNDINGDINO_AVAILABLE:
                if self.debug_mode:
                    print("⚠️ GroundingDINO not available")
                return False
            
            # Find GroundingDINO model files (fix the .cfg.py issue we found)
            config_path, model_path = self._find_groundingdino_model()
            if not (config_path and model_path):
                if self.debug_mode:
                    print("⚠️ GroundingDINO model files not found")
                return False
            
            # Load model using existing imports
            try:
                from local_groundingdino.util.slconfig import SLConfig
                from local_groundingdino.models import build_model
                from local_groundingdino.util.utils import clean_state_dict
            except ImportError:
                from groundingdino.util.slconfig import SLConfig
                from groundingdino.models import build_model
                from groundingdino.util.utils import clean_state_dict
            
            # Load config and build model
            args = SLConfig.fromfile(config_path)
            self.dino_model = build_model(args)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location="cpu")
            self.dino_model.load_state_dict(
                clean_state_dict(checkpoint["model"]), strict=False
            )
            self.dino_model.to(self.device)
            self.dino_model.eval()
            
            if self.debug_mode:
                print(f"✅ GroundingDINO loaded: {model_path}")
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ GroundingDINO loading failed: {e}")
            return False
    
    def _find_groundingdino_model(self) -> tuple:
        """Find GroundingDINO config and model files (fixed for .cfg.py)"""
        try:
            dino_dir = Path(folder_paths.models_dir) / "grounding-dino"
            if not dino_dir.exists():
                return None, None
            
            # Fix the config file extensions we found earlier
            config_variants = [
                ("GroundingDINO_SwinT_OGC.cfg.py", "groundingdino_swint_ogc.pth"),
                ("GroundingDINO_SwinB.cfg.py", "groundingdino_swinb_cogcoor.pth"),
                ("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth"),
                ("GroundingDINO_SwinB.py", "groundingdino_swinb_cogcoor.pth")
            ]
            
            for config_name, model_name in config_variants:
                config_file = dino_dir / config_name
                model_file = dino_dir / model_name
                
                if config_file.exists() and model_file.exists():
                    if self.debug_mode:
                        print(f"    ✅ Found GroundingDINO: {config_file}, {model_file}")
                    return str(config_file), str(model_file)
            
            return None, None
        except Exception:
            return None, None
    
    def detect_images(self, image: Image.Image, prompt: str = "photograph . image . illustration") -> List[Dict]:
        """GroundingDINO detection only (no SAM2 segmentation)"""
        if not self.available:
            if self.debug_mode:
                print("      ⚠️ GroundingDINO analyzer not available")
            return []
        
        try:
            # Use GroundingDINO for object detection
            detections = self._run_groundingdino_detection(image, prompt)
            
            if not detections:
                if self.debug_mode:
                    print("      ⚠️ GroundingDINO found no objects")
                return []
            
            # Mark all detections as having no masks (no SAM2)
            for detection in detections:
                detection['has_mask'] = False
            
            if self.debug_mode:
                print(f"      ✅ GroundingDINO only: {len(detections)} detections")
            
            return detections
            
        except Exception as e:
            if self.debug_mode:
                print(f"      ❌ Detection pipeline failed: {e}")
            return []
    
    def _run_groundingdino_detection(self, image: Image.Image, prompt: str) -> List[Dict]:
        """Run GroundingDINO detection with proper inference"""
        try:
            if not self.dino_model:
                return []
            
            # Import GroundingDINO inference utilities
            try:
                from local_groundingdino.util.inference import load_image, predict
            except ImportError:
                from groundingdino.util.inference import load_image, predict
            
            # Save image temporarily for GroundingDINO's load_image function
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file.name, 'JPEG')
                tmp_path = tmp_file.name
            
            try:
                # Load image using GroundingDINO's format
                image_source, transformed_image = load_image(tmp_path)
                
                # Run GroundingDINO prediction
                boxes, logits, phrases = predict(
                    model=self.dino_model,
                    image=transformed_image,
                    caption=prompt,
                    box_threshold=0.3,
                    text_threshold=0.25,
                    device=self.device
                )
                
                # Convert to pixel coordinates
                H, W, _ = image_source.shape
                boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
                
                detections = []
                for i, (box, logit, phrase) in enumerate(zip(boxes_xyxy, logits, phrases)):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    confidence = logit.cpu().numpy()
                    
                    # Validate bounding box coordinates
                    if x1 >= x2 or y1 >= y2:
                        if self.debug_mode:
                            print(f"      ⚠️ Invalid bbox from GroundingDINO: [{x1}, {y1}, {x2}, {y2}] - skipping")
                        continue
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, W))
                    y1 = max(0, min(y1, H))
                    x2 = max(x1 + 1, min(x2, W))
                    y2 = max(y1 + 1, min(y2, H))
                    
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "label": f"grounding_dino_{phrase}",
                        "confidence": float(confidence)
                    })
                
                return detections
                
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
                
        except Exception as e:
            if self.debug_mode:
                print(f"      ❌ GroundingDINO detection failed: {e}")
            return []
    
    def _run_sam2_on_detections(self, image: Image.Image, detections: List[Dict]) -> List[Dict]:
        """Run SAM2 segmentation on GroundingDINO detections"""
        try:
            if not self.sam2_predictor or not detections:
                return detections
            
            # Convert PIL to numpy array
            image_np = np.array(image)
            
            # Set image for SAM2
            self.sam2_predictor.set_image(image_np)
            
            enhanced_detections = []
            for detection in detections:
                try:
                    bbox = detection["bbox"]
                    x1, y1, x2, y2 = bbox
                    
                    # Create input box for SAM2
                    input_box = np.array([x1, y1, x2, y2])
                    
                    # Run SAM2 prediction
                    masks, scores, logits = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box,
                        multimask_output=False
                    )
                    
                    if len(masks) > 0 and len(scores) > 0:
                        # Add mask information to detection
                        enhanced_detection = detection.copy()
                        enhanced_detection['mask'] = masks[0]
                        enhanced_detection['sam_score'] = float(scores[0])
                        enhanced_detection['has_mask'] = True
                        enhanced_detections.append(enhanced_detection)
                    else:
                        # Keep original detection
                        detection['has_mask'] = False
                        enhanced_detections.append(detection)
                        
                except Exception as e:
                    if self.debug_mode:
                        print(f"      ⚠️ SAM2 failed for box {bbox}: {e}")
                    # Keep original detection
                    detection['has_mask'] = False
                    enhanced_detections.append(detection)
            
            return enhanced_detections
            
        except Exception as e:
            if self.debug_mode:
                print(f"      ❌ SAM2 processing failed: {e}")
            return detections

# Add SAM2Predictor wrapper class if needed
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
        """Ensure bbox coordinates are valid [x1, y1, x2, y2] where x1 < x2 and y1 < y2"""
        x1, y1, x2, y2 = bbox
        
        # Swap coordinates if they're in wrong order
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Ensure minimum size (at least 1 pixel)
        if x2 - x1 < 1:
            x2 = x1 + 1
        if y2 - y1 < 1:
            y2 = y1 + 1
            
        # Clamp to image bounds
        x1 = max(0, min(x1, vis_img.width - 1))
        y1 = max(0, min(y1, vis_img.height - 1))
        x2 = max(x1 + 1, min(x2, vis_img.width))
        y2 = max(y1 + 1, min(y2, vis_img.height))
        
        return [x1, y1, x2, y2]
    
    # Get results
    florence2_boxes = detection_results.get("florence2_rectangles", [])
    surya_regions = detection_results.get("surya_layout", [])
    sam_dino_boxes = detection_results.get("sam_dino_detections", [])
    
    # Draw Florence2 boxes in RED with thick lines
    for i, box in enumerate(florence2_boxes):
        try:
            bbox = validate_bbox(box["bbox"])
            confidence = box.get("confidence", 1.0)
            draw.rectangle(bbox, outline="red", width=4)
            # Larger text with background for better visibility
            text = f"F2_{i+1}({confidence:.2f})"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw text background
            text_x, text_y = bbox[0], bbox[1] - text_height - 5
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill="red")
            draw.text((text_x, text_y), text, fill="white", font=font)
        except Exception as e:
            print(f"⚠️ Error drawing Florence2 box {i}: {e}")
    
    # Draw ALL Surya regions in BLUE with different styles for different types
    color_map = {
        "text": "blue",
        "title": "navy", 
        "heading": "darkblue",
        "caption": "lightblue",
        "image": "cyan",
        "figure": "turquoise",
        "table": "purple",
        "header": "gray",
        "footer": "darkgray"
    }
    
    for i, region in enumerate(surya_regions):
        try:
            bbox = validate_bbox(region["bbox"])
            label = region.get("semantic_label", "unknown").lower()
            
            # Choose color based on semantic label
            color = "blue"  # default
            for key, c in color_map.items():
                if key in label:
                    color = c
                    break
            
            draw.rectangle(bbox, outline=color, width=3)
            
            # Draw label with background
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
    
    # Draw SAM+DINO boxes in GREEN with validation
    if sam_dino_boxes:
        for i, box in enumerate(sam_dino_boxes):
            try:
                bbox = validate_bbox(box["bbox"])
                confidence = box.get("confidence", 1.0)
                label = box.get("label", "unknown")
                
                draw.rectangle(bbox, outline="lime", width=3)
                text = f"DINO_{i+1}({confidence:.2f})"
                
                # Add debug info for the problematic box
                if confidence == 0 or any(coord < 0 for coord in box["bbox"]):
                    print(f"⚠️ Suspicious DINO box {i}: original={box['bbox']}, fixed={bbox}, conf={confidence}, label={label}")
                
                draw.text((bbox[0], bbox[3] + 5), text, fill="lime", font=font)
            except Exception as e:
                print(f"⚠️ Error drawing SAM+DINO box {i}: {e}")
                print(f"   Original bbox: {box.get('bbox', 'None')}")
                print(f"   Box data: {box}")
    
    # Draw seam line - make it more prominent
    seam_x = joined_img.width // 2
    draw.line([seam_x, 0, seam_x, joined_img.height], fill="yellow", width=5)
    # Seam label with background
    seam_text = "SEAM"
    text_bbox = draw.textbbox((0, 0), seam_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.rectangle([seam_x-text_width//2, 20, seam_x+text_width//2, 20+text_height], fill="yellow")
    draw.text((seam_x-text_width//2, 20), seam_text, fill="black", font=font)
    
    # Add comprehensive summary with larger text and background
    summary_lines = [
        f"Florence2: {len(florence2_boxes)} boxes",
        f"Surya: {len(surya_regions)} regions",
        f"SAM+DINO: {len(sam_dino_boxes)} detections"
    ]
    
    y_pos = 10
    for line in summary_lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Background for better readability
        draw.rectangle([10, y_pos, 10 + text_width + 10, y_pos + text_height + 5], fill="black")
        draw.text((15, y_pos + 2), line, fill="white", font=font)
        y_pos += text_height + 10
    
    # Draw rejected Florence2 boxes in ORANGE (dashed style would be ideal)
    rejected_boxes = detection_results.get("florence2_rejected", [])
    if rejected_boxes:
        for i, box in enumerate(rejected_boxes):
            try:
                bbox = validate_bbox(box["bbox"])
                confidence = box.get("confidence", 1.0)
                # Draw with dashed effect (multiple thin lines)
                for offset in range(0, 8, 2):
                    draw.rectangle([bbox[0]+offset, bbox[1]+offset, bbox[2]+offset, bbox[3]+offset], 
                                 outline="orange", width=1)
                text = f"F2_REJ_{i+1}({confidence:.2f})"
                draw.text((bbox[0], bbox[1] - 20), text, fill="orange", font=font)
            except Exception as e:
                print(f"⚠️ Error drawing rejected Florence2 box {i}: {e}")
    
    # Save visualization
    vis_dir = os.path.join(output_dir, "comprehensive_analysis")
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, f"{pair_id}_comprehensive.png")
    vis_img.save(output_path)
    
    return output_path

def save_all_crops(joined_img: Image.Image, detection_results: Dict,
                   pair_id: str, output_dir: str, analyzer):
    """Save crops from all detection methods"""
    
    florence2_boxes = detection_results.get("florence2_rectangles", [])
    
    if not florence2_boxes:
        return []
    
    # Create output directories
    original_crops_dir = os.path.join(output_dir, "florence2_crops")
    refined_crops_dir = os.path.join(output_dir, "refined_crops") 
    os.makedirs(original_crops_dir, exist_ok=True)
    os.makedirs(refined_crops_dir, exist_ok=True)
    
    saved_crops = []
    
    for i, box in enumerate(florence2_boxes):
        bbox = box["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(joined_img.width, x2)
        y2 = min(joined_img.height, y2)
        
        if x2 > x1 and y2 > y1:
            # Original crop
            cropped = joined_img.crop((x1, y1, x2, y2))
            confidence = box.get("confidence", 1.0)
            
            # Save original crop
            orig_filename = f"{pair_id}_F2_{i+1:02d}_conf{confidence:.2f}.png"
            orig_path = os.path.join(original_crops_dir, orig_filename)
            cropped.save(orig_path)
            
            # Refine crop with Florence2 recursion
            try:
                refined_crop = refine_crop_with_florence2(cropped, analyzer)
                refined_filename = f"{pair_id}_refined_{i+1:02d}_conf{confidence:.2f}.png"
                refined_path = os.path.join(refined_crops_dir, refined_filename)
                refined_crop.save(refined_path)
                
                saved_crops.append({
                    "original": orig_path,
                    "refined": refined_path,
                    "confidence": confidence,
                    "area_pct": ((x2-x1) * (y2-y1)) / (joined_img.width * joined_img.height) * 100
                })
            except Exception as e:
                print(f"      ⚠️ Refinement failed for crop {i+1}: {e}")
                saved_crops.append({
                    "original": orig_path,
                    "refined": None,
                    "confidence": confidence,
                    "area_pct": ((x2-x1) * (y2-y1)) / (joined_img.width * joined_img.height) * 100
                })
    
    return saved_crops

def refine_crop_with_florence2(cropped_img: Image.Image, analyzer, confidence_threshold: float = 0.3) -> Image.Image:
    """Run Florence2 again on cropped image to get tighter bounding box"""
    
    try:
        # Run Florence2 on the already cropped image
        analysis = analyze_for_pdf_extraction(cropped_img, debug_mode=False)
        florence2_boxes = analysis.get("florence2_rectangles", [])
        
        if not florence2_boxes:
            return cropped_img
        
        # Find the best box (largest area with good confidence)
        best_box = None
        best_score = 0
        
        for box in florence2_boxes:
            confidence = box.get("confidence", 1.0)
            bbox = box["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Score based on confidence and relative size
            score = confidence * (area / (cropped_img.width * cropped_img.height))
            
            if score > best_score and confidence >= confidence_threshold:
                best_box = box
                best_score = score
        
        if best_box:
            bbox = best_box["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within crop bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(cropped_img.width, x2)
            y2 = min(cropped_img.height, y2)
            
            if x2 > x1 and y2 > y1:
                refined_crop = cropped_img.crop((x1, y1, x2, y2))
                return refined_crop
        
        return cropped_img
        
    except Exception as e:
        print(f"      ⚠️ Refinement failed: {e}")
        return cropped_img

# Replace the generic prompt system with PDF-optimized prompts:

def get_pdf_optimized_prompts():
    """Get prompts optimized for PDF illustration/photo detection and filtering"""
    return {
        "main_images": "photograph . photo . illustration . picture . image",
        "logos_icons": "logo . icon . symbol . emblem . badge",
        "decorative": "decoration . ornament . pattern . border . divider",
        "all_visual": "photograph . photo . illustration . picture . image . logo . icon . symbol"
    }

# Update the SimpleSegmentationAnalyzer to support multiple prompt strategies:

def detect_images_with_filtering(self, image: Image.Image, florence2_boxes: List[Dict] = None) -> Dict:
    """Enhanced GroundingDINO detection with Florence2 filtering and complementary detection"""
    if not self.available:
        if self.debug_mode:
            print("      ⚠️ GroundingDINO analyzer not available")
        return {
            "filtered_florence2": florence2_boxes or [],
            "additional_detections": [],
            "rejected_florence2": []
        }
    
    prompts = get_pdf_optimized_prompts()
    results = {
        "filtered_florence2": [],
        "additional_detections": [],
        "rejected_florence2": []
    }
    
    try:
        # Step 1: Run comprehensive GroundingDINO detection
        all_detections = self._run_groundingdino_detection(image, prompts["all_visual"])
        
        if self.debug_mode:
            print(f"      ✅ GroundingDINO found {len(all_detections)} total detections")
        
        # Step 2: Filter Florence2 boxes against GroundingDINO detections
        if florence2_boxes:
            results["filtered_florence2"], results["rejected_florence2"] = self._filter_florence2_with_dino(
                florence2_boxes, all_detections
            )
            
            if self.debug_mode:
                print(f"      📋 Florence2 filtering: {len(results['filtered_florence2'])} kept, {len(results['rejected_florence2'])} rejected")
        
        # Step 3: Find additional detections (things Florence2 missed)
        if florence2_boxes:
            results["additional_detections"] = self._find_additional_detections(
                all_detections, florence2_boxes
            )
            
            if self.debug_mode and results["additional_detections"]:
                print(f"      🔍 Found {len(results['additional_detections'])} additional items Florence2 missed")
        else:
            results["additional_detections"] = all_detections
        
        return results
        
    except Exception as e:
        if self.debug_mode:
            print(f"      ❌ Enhanced detection failed: {e}")
        return {
            "filtered_florence2": florence2_boxes or [],
            "additional_detections": [],
            "rejected_florence2": []
        }

def _filter_florence2_with_dino(self, florence2_boxes: List[Dict], dino_detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Filter Florence2 boxes by checking if GroundingDINO confirms them as actual images"""
    
    def boxes_overlap(box1, box2, threshold=0.3):
        """Check if two boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = intersection / (area1 + area2 - intersection)
        return iou >= threshold
    
    confirmed_boxes = []
    rejected_boxes = []
    
    for f2_box in florence2_boxes:
        f2_bbox = f2_box["bbox"]
        
        # Check if any GroundingDINO detection overlaps and confirms this is actually an image
        confirmed = False
        for dino_det in dino_detections:
            if boxes_overlap(f2_bbox, dino_det["bbox"]):
                # Check if GroundingDINO label suggests this is actually an image/photo/illustration
                dino_label = dino_det["label"].lower()
                if any(term in dino_label for term in ["photograph", "photo", "illustration", "picture", "image"]):
                    confirmed = True
                    # Add GroundingDINO confidence to Florence2 box
                    f2_box["dino_confirmation"] = {
                        "confidence": dino_det["confidence"],
                        "label": dino_det["label"]
                    }
                    break
        
        if confirmed:
            confirmed_boxes.append(f2_box)
        else:
            # Add reason for rejection
            f2_box["rejection_reason"] = "No GroundingDINO confirmation"
            rejected_boxes.append(f2_box)
    
    return confirmed_boxes, rejected_boxes

def _find_additional_detections(self, dino_detections: List[Dict], florence2_boxes: List[Dict]) -> List[Dict]:
    """Find GroundingDINO detections that don't overlap with Florence2 (things Florence2 missed)"""
    
    def boxes_overlap(box1, box2, threshold=0.2):
        """Check if boxes overlap (lower threshold for finding missed items)"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        overlap_ratio = intersection / area1
        return overlap_ratio >= threshold
    
    additional = []
    
    for dino_det in dino_detections:
        # Check if this GroundingDINO detection overlaps with any Florence2 box
        overlaps_with_f2 = False
        for f2_box in florence2_boxes:
            if boxes_overlap(dino_det["bbox"], f2_box["bbox"]):
                overlaps_with_f2 = True
                break
        
        # If it doesn't overlap, it's something Florence2 missed
        if not overlaps_with_f2:
            dino_det["source"] = "grounding_dino_additional"
            dino_det["has_mask"] = False
            additional.append(dino_det)
    
    return additional
def process_pdf(pdf_path: str):
    """Process PDF with comprehensive multi-method analysis"""
    
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
    
    # Initialize SAM analyzer (but it's disabled for now)
    sam_analyzer = SimpleSegmentationAnalyzer(debug_mode=True)
    print(f"📊 SAM+DINO analyzer available: {sam_analyzer.available}")
    
    with fitz.open(pdf_path) as doc:
        pairs = get_even_odd_pairs(doc)
        results["total_pairs"] = len(pairs)
        
        print(f"📊 Processing {len(pairs)} even-odd pairs...")
        
        for i, (left_idx, right_idx) in enumerate(pairs):
            pair_id = f"{left_idx+1:03d}_{right_idx+1:03d}"
            print(f"  🔄 Pair {i+1}/{len(pairs)}: pages {left_idx+1}-{right_idx+1}")
            
            try:
                # Extract and join pages
                left_page = doc[left_idx]
                right_page = doc[right_idx]
                
                mat = fitz.Matrix(200/72, 200/72)  # 200 DPI
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
                
                # KEEP THE WORKING FLORENCE2 + SURYA ANALYSIS
                print(f"    📊 Running Florence2 + Surya analysis...")
                analysis = analyze_for_pdf_extraction(joined_img, debug_mode=False)
                
                # SAM+DINO is disabled for now - returns empty list
                print(f"    📊 Running SAM2 + GroundingDINO analysis...")
                sam_dino_results = []
                if sam_analyzer.available:
                    try:
                        sam_dino_results = sam_analyzer.detect_images(joined_img, "photograph image illustration")
                        if sam_dino_results and sam_analyzer.debug_mode:
                            print(f"      ✅ SAM2 found {len(sam_dino_results)} detections")
                    except Exception as e:
                        if sam_analyzer.debug_mode:
                            print(f"      ❌ SAM2 detection error: {e}")
                        sam_dino_results = []
                else:
                    print(f"      ⚠️ SAM2 analyzer not available")
                
                # Combine results (Florence2 + Surya working, SAM empty)
                detection_results = {
                    "florence2_rectangles": analysis.get("florence2_rectangles", []),
                    "surya_layout": analysis.get("surya_layout", []),
                    "sam_dino_detections": sam_dino_results
                }
                
                # Create comprehensive visualization
                create_comprehensive_visualization(
                    joined_img, detection_results, pair_id, output_dir
                )
                
                # Save all crops using working Florence2
                crop_info = save_all_crops(
                    joined_img, detection_results, pair_id, output_dir, analyze_for_pdf_extraction
                )
                
                # Update stats
                results["successful_analyses"] += 1
                results["total_florence2_boxes"] += len(detection_results["florence2_rectangles"])
                results["total_surya_regions"] += len(detection_results["surya_layout"])
                results["total_sam_dino_detections"] += len(detection_results["sam_dino_detections"])
                results["all_crop_info"].extend(crop_info)
                
                print(f"    ✅ F2:{len(detection_results['florence2_rectangles'])} | "
                      f"Surya:{len(detection_results['surya_layout'])} | "
                      f"SAM+DINO:{len(detection_results['sam_dino_detections'])} | "
                      f"Crops:{len(crop_info)}")
                
            except Exception as e:
                print(f"    ❌ Error processing pair: {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print(f"\n✅ Comprehensive Analysis Complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Processed: {results['successful_analyses']}/{results['total_pairs']} pairs")
    print(f"📦 Detection Summary:")
    print(f"   🔴 Florence2: {results['total_florence2_boxes']} boxes total")
    print(f"   🔵 Surya Layout: {results['total_surya_regions']} regions total")
    print(f"   🟢 SAM+DINO: {results['total_sam_dino_detections']} detections total")
    print(f"   🖼️  Total crops: {len(results['all_crop_info'])}")
    
    # Analysis suggestions
    print(f"\n💡 Analysis Tips:")
    print(f"   • Red boxes = Florence2 image detection")
    print(f"   • Blue boxes = Surya layout (all types, color-coded)")
    print(f"   • Green boxes = SAM+GroundingDINO (if available)")
    print(f"   • Compare original vs refined crops for border trimming effectiveness")
    print(f"   • Look for patterns in missed/spurious detections")

def main():
    if len(sys.argv) != 2:
        print("Usage: python visual_box_comparison.py path/to/test.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    process_pdf(pdf_path)

if __name__ == "__main__":
    main()