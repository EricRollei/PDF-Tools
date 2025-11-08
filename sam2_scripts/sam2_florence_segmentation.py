"""
Sam2 Florence Segmentation

Description: SAM2 (Segment Anything Model 2) integration for advanced image segmentation
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
- Uses SAM2 models (Apache 2.0) by Meta AI: https://github.com/facebookresearch/segment-anything-2
- See CREDITS.md for complete list of all dependencies
"""

#!/usr/bin/env python3
"""
Standalone SAM2 + CogFlorence2.2 Image Segmentation Script
For use with Enhanced PDF Extractor

This script provides high-quality image segmentation using SAM2 and CogFlorence2.2
which can significantly outperform GroundingDINO + SAM for complex multi-object scenarios.

Usage:
    python sam2_florence_segmentation.py input_image.jpg --prompt "main subject" --output masks/

Dependencies:
    - ComfyUI-segment-anything-2 (https://github.com/kijai/ComfyUI-segment-anything-2)
    - Florence2 models
    - SAM2 models

Author: Eric Hiss
License: Apache 2.0 (compatible with kijai's SAM2 implementation)
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

# Add ComfyUI path if available
COMFYUI_PATH = os.environ.get('COMFYUI_PATH', '../../../')
if Path(COMFYUI_PATH).exists():
    sys.path.append(str(Path(COMFYUI_PATH).resolve()))

try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("Warning: ComfyUI not found, using fallback paths")

class SAM2FlorenceSegmenter:
    """Standalone segmenter using SAM2 and CogFlorence2.2"""
    
    def __init__(self, 
                 sam2_model_path: Optional[str] = None,
                 florence_model_path: Optional[str] = None,
                 device: str = "auto",
                 debug: bool = False):
        
        self.debug = debug
        self.device = self._setup_device(device)
        
        # Initialize models
        self.sam2_model = None
        self.florence_model = None
        self.florence_processor = None
        
        # Try to load models
        self._load_sam2_model(sam2_model_path)
        self._load_florence_model(florence_model_path)
        
        self.available = (self.sam2_model is not None and 
                         self.florence_model is not None)
        
        if self.debug:
            print(f"SAM2FlorenceSegmenter initialized:")
            print(f"  Device: {self.device}")
            print(f"  SAM2 loaded: {self.sam2_model is not None}")
            print(f"  Florence loaded: {self.florence_model is not None}")
            print(f"  Available: {self.available}")

    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _get_model_paths(self) -> Dict[str, str]:
        """Get default model paths"""
        if COMFYUI_AVAILABLE:
            base_path = Path(folder_paths.models_dir)
        else:
            base_path = Path("models")
        
        sam2_path = base_path / "sam2"
        florence_path = base_path / "LLM"  # Florence models typically in LLM folder
        
        return {
            "sam2_base": str(sam2_path),
            "florence_base": str(florence_path),
            "sam2_configs": str(sam2_path / "configs"),
            "sam2_checkpoints": str(sam2_path / "checkpoints"),
        }

    def _load_sam2_model(self, model_path: Optional[str] = None):
        """Load SAM2 model"""
        try:
            # Try to import SAM2 from kijai's implementation
            try:
                from nodes.sam2_nodes import SAM2Model, load_sam2_model
                sam2_available = True
            except ImportError:
                # Try alternative import paths
                try:
                    sys.path.append("custom_nodes/ComfyUI-segment-anything-2")
                    from nodes.sam2_nodes import SAM2Model, load_sam2_model
                    sam2_available = True
                except ImportError:
                    sam2_available = False
            
            if not sam2_available:
                if self.debug:
                    print("SAM2 nodes not found - trying direct SAM2 import")
                return self._load_sam2_direct(model_path)
            
            # Use kijai's SAM2 implementation
            paths = self._get_model_paths()
            
            if model_path is None:
                # Auto-detect SAM2 model
                model_path = self._find_sam2_model(paths["sam2_checkpoints"])
            
            if model_path and Path(model_path).exists():
                self.sam2_model = load_sam2_model(model_path, self.device)
                if self.debug:
                    print(f"SAM2 model loaded from: {model_path}")
            else:
                if self.debug:
                    print(f"SAM2 model not found at: {model_path}")
                    
        except Exception as e:
            if self.debug:
                print(f"Error loading SAM2 model: {e}")
            self.sam2_model = None

    def _load_sam2_direct(self, model_path: Optional[str] = None):
        """Load SAM2 directly (fallback method)"""
        try:
            # This would require SAM2 to be installed directly
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Implementation would go here
            if self.debug:
                print("Direct SAM2 loading not implemented yet")
                
        except ImportError:
            if self.debug:
                print("Direct SAM2 import failed")

    def _find_sam2_model(self, checkpoint_dir: str) -> Optional[str]:
        """Find best available SAM2 model"""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
        
        # Preferred model order
        preferred_models = [
            "sam2_hiera_large.pt",
            "sam2_hiera_base_plus.pt", 
            "sam2_hiera_small.pt",
            "sam2_hiera_tiny.pt"
        ]
        
        for model_name in preferred_models:
            model_path = checkpoint_path / model_name
            if model_path.exists():
                return str(model_path)
        
        # Fall back to any .pt file
        pt_files = list(checkpoint_path.glob("*.pt"))
        if pt_files:
            return str(pt_files[0])
        
        return None

    def _load_cogflorence_model(self, model_path: str):
        """Special handling for CogFlorence models"""
        try:
            # CogFlorence models often need different loading approach
            from transformers import AutoProcessor
            
            # Try to load processor first
            self.florence_processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # For CogFlorence, try different model loading approaches
            loading_strategies = [
                # Strategy 1: Standard AutoModel with specific config
                lambda: self._load_with_custom_config(model_path),
                # Strategy 2: Direct model class loading
                lambda: self._load_with_direct_class(model_path),
                # Strategy 3: Fallback to base Florence2
                lambda: self._load_fallback_florence(model_path),
            ]
            
            for strategy in loading_strategies:
                try:
                    self.florence_model = strategy()
                    if self.florence_model is not None:
                        if self.debug:
                            print(f"CogFlorence loaded with strategy")
                        return
                except Exception as e:
                    if self.debug:
                        print(f"Loading strategy failed: {e}")
                    continue
                    
            raise Exception("All CogFlorence loading strategies failed")
            
        except Exception as e:
            raise Exception(f"CogFlorence loading failed: {e}")

    def _load_with_custom_config(self, model_path: str):
        """Load with custom configuration"""
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Load config and modify if needed
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Apply any necessary config modifications for CogFlorence
        if hasattr(config, 'vision_config'):
            # Handle vision-language model specifics
            pass
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True  # Important for custom models
        ).to(self.device)
        
        return model

    def _load_with_direct_class(self, model_path: str):
        """Try to load using direct model class"""
        try:
            # Try to import specific model class if available
            from transformers import Florence2ForConditionalGeneration
            
            model = Florence2ForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            ).to(self.device)
            
            return model
            
        except ImportError:
            # If specific class doesn't exist, try generic approach
            from transformers import AutoModelForCausalLM
            
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                device_map="auto" if self.device == "cuda" else None,
            ).to(self.device)

    def _load_fallback_florence(self, model_path: str):
        """Fallback to standard Florence2 if CogFlorence fails"""
        from transformers import AutoModelForCausalLM
        
        # Try to load as standard Florence2
        fallback_models = [
            "microsoft/Florence-2-large",
            "microsoft/Florence-2-base"
        ]
        
        for fallback in fallback_models:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    fallback,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
                
                if self.debug:
                    print(f"Using fallback Florence2 model: {fallback}")
                return model
                
            except Exception:
                continue
        
        return None

    def _load_florence_model(self, model_path: Optional[str] = None):
        """Load Florence2 model"""
        try:
            # Try to import Florence from kijai's implementation
            try:
                from nodes.florence2_nodes import Florence2Model, load_florence2_model
                florence_available = True
            except ImportError:
                try:
                    sys.path.append("custom_nodes/ComfyUI-segment-anything-2")
                    from nodes.florence2_nodes import Florence2Model, load_florence2_model
                    florence_available = True
                except ImportError:
                    florence_available = False
            
            if not florence_available:
                if self.debug:
                    print("Florence2 nodes not found - trying direct import")
                return self._load_florence_direct(model_path)
            
            # Use kijai's Florence implementation
            paths = self._get_model_paths()
            
            if model_path is None:
                model_path = self._find_florence_model(paths["florence_base"])
            
            if model_path and Path(model_path).exists():
                self.florence_model = load_florence2_model(model_path, self.device)
                if self.debug:
                    print(f"Florence2 model loaded from: {model_path}")
            else:
                if self.debug:
                    print(f"Florence2 model not found at: {model_path}")
                    
        except Exception as e:
            if self.debug:
                print(f"Error loading Florence2 model: {e}")
            self.florence_model = None

    def _load_florence_direct(self, model_path: Optional[str] = None):
        """Load Florence2 directly (fallback method)"""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
            
            if model_path is None:
                # Try different model variants in order of preference
                model_variants = [
                    "microsoft/Florence-2-large",
                    "microsoft/Florence-2-base", 
                    "microsoft/Florence-2-large-ft",
                ]
            else:
                model_variants = [model_path]
            
            for variant in model_variants:
                try:
                    if self.debug:
                        print(f"Trying to load Florence2 variant: {variant}")
                    
                    # First try to load config to check compatibility
                    try:
                        config = AutoConfig.from_pretrained(
                            variant, 
                            trust_remote_code=True
                        )
                        
                        # Check if this is CogFlorence which needs special handling
                        if "cogflorence" in variant.lower() or hasattr(config, 'vision_config'):
                            return self._load_cogflorence_model(variant)
                            
                    except Exception as config_e:
                        if self.debug:
                            print(f"Config loading failed for {variant}: {config_e}")
                        continue
                    
                    # Standard Florence2 loading
                    self.florence_processor = AutoProcessor.from_pretrained(
                        variant, 
                        trust_remote_code=True
                    )
                    
                    self.florence_model = AutoModelForCausalLM.from_pretrained(
                        variant, 
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        low_cpu_mem_usage=True
                    ).to(self.device)
                    
                    if self.debug:
                        print(f"Florence2 loaded successfully from: {variant}")
                    return
                    
                except Exception as e:
                    if self.debug:
                        print(f"Failed to load {variant}: {e}")
                    continue
            
            # If all variants failed, raise the last error
            raise Exception("No compatible Florence2 model found")
            
        except Exception as e:
            if self.debug:
                print(f"Direct Florence2 loading failed: {e}")


    def _find_florence_model(self, model_dir: str) -> Optional[str]:
        """Find best available Florence2 model"""
        model_path = Path(model_dir)
        if not model_path.exists():
            return None
        
        # Look for Florence models in order of preference
        preferred_models = [
            "CogFlorence-2.2-Large",
            "cogflorence-2.2-large", 
            "Florence-2-large",
            "Florence-2-base",
            "microsoft--Florence-2-large",
            "microsoft--Florence-2-base",
            "microsoft/Florence-2-large",
            "microsoft/Florence-2-base"
        ]
        
        for model_name in preferred_models:
            full_path = model_path / model_name
            if full_path.exists() and (
                (full_path / "config.json").exists() or 
                (full_path / "pytorch_model.bin").exists() or
                (full_path / "model.safetensors").exists()
            ):
                return str(full_path)
        
        return None

    def segment_image(self, 
                     image: Image.Image, 
                     prompt: str = "objects",
                     task: str = "OD",  # Object Detection
                     confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Segment image using SAM2 + Florence2
        
        Args:
            image: PIL Image
            prompt: Text prompt for Florence2
            task: Florence2 task ('OD' for object detection, 'CAPTION_TO_PHRASE_GROUNDING' for specific objects)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with masks, boxes, and metadata
        """
        
        if not self.available:
            return {"success": False, "error": "Models not available"}
        
        try:
            # Step 1: Use Florence2 to detect objects/regions
            florence_results = self._run_florence_detection(image, prompt, task)
            
            if not florence_results["success"]:
                return florence_results
            
            # Step 2: Use SAM2 to generate high-quality masks
            sam2_results = self._run_sam2_segmentation(image, florence_results)
            
            # Step 3: Combine and process results
            final_results = self._process_combined_results(
                image, florence_results, sam2_results, confidence_threshold
            )
            
            return final_results
            
        except Exception as e:
            if self.debug:
                print(f"Segmentation error: {e}")
                import traceback
                traceback.print_exc()
            
            return {"success": False, "error": str(e)}

    def _run_florence_detection(self, image: Image.Image, prompt: str, task: str) -> Dict[str, Any]:
        """Run Florence2 object detection"""
        try:
            if self.florence_processor is not None:
                # Direct transformers approach
                return self._run_florence_direct(image, prompt, task)
            else:
                # Kijai's node approach
                return self._run_florence_nodes(image, prompt, task)
                
        except Exception as e:
            return {"success": False, "error": f"Florence detection failed: {e}"}

    def _run_florence_direct(self, image: Image.Image, prompt: str, task: str) -> Dict[str, Any]:
        """Run Florence2 using direct transformers"""
        try:
            # Prepare inputs
            if task == "OD":
                task_prompt = "<OD>"
            elif task == "CAPTION_TO_PHRASE_GROUNDING":
                task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{prompt}"
            else:
                task_prompt = f"<OD>{prompt}"
            
            inputs = self.florence_processor(
                text=task_prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3
                )
            
            # Decode results
            generated_text = self.florence_processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            
            # Parse Florence2 output
            parsed_results = self.florence_processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            # Convert to standard format
            boxes = []
            labels = []
            scores = []
            
            if task_prompt in parsed_results:
                result_data = parsed_results[task_prompt]
                if 'bboxes' in result_data and 'labels' in result_data:
                    boxes = result_data['bboxes']
                    labels = result_data['labels']
                    scores = [0.8] * len(boxes)  # Florence doesn't provide scores, use default
            
            return {
                "success": True,
                "boxes": boxes,
                "labels": labels, 
                "scores": scores,
                "method": "florence_direct"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Florence direct failed: {e}"}

    def _run_florence_nodes(self, image: Image.Image, prompt: str, task: str) -> Dict[str, Any]:
        """Run Florence2 using kijai's nodes"""
        try:
            # This would use kijai's Florence2 node implementation
            # Implementation depends on the specific node interface
            
            # Placeholder - would need actual node calling code
            return {"success": False, "error": "Florence nodes not implemented yet"}
            
        except Exception as e:
            return {"success": False, "error": f"Florence nodes failed: {e}"}

    def _run_sam2_segmentation(self, image: Image.Image, florence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run SAM2 segmentation on Florence2 detections"""
        try:
            if not florence_results["success"] or not florence_results["boxes"]:
                return {"success": False, "error": "No boxes from Florence2"}
            
            # Convert image to numpy array
            image_np = np.array(image)
            
            # Convert boxes to SAM2 format
            boxes = np.array(florence_results["boxes"])
            
            # Initialize SAM2 predictor
            if hasattr(self.sam2_model, 'set_image'):
                self.sam2_model.set_image(image_np)
            
                # Generate masks for each box
                masks = []
                for box in boxes:
                    try:
                        mask, _, _ = self.sam2_model.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box,
                            multimask_output=False
                        )
                        masks.append(mask[0])  # Take first mask
                    except Exception as e:
                        if self.debug:
                            print(f"SAM2 prediction failed for box {box}: {e}")
                        continue
                
                return {
                    "success": True,
                    "masks": masks,
                    "method": "sam2_predict"
                }
            else:
                return {"success": False, "error": "SAM2 model interface not recognized"}
                
        except Exception as e:
            return {"success": False, "error": f"SAM2 segmentation failed: {e}"}

    def _process_combined_results(self, 
                                image: Image.Image,
                                florence_results: Dict[str, Any], 
                                sam2_results: Dict[str, Any],
                                confidence_threshold: float) -> Dict[str, Any]:
        """Process and combine Florence2 + SAM2 results"""
        
        if not (florence_results["success"] and sam2_results["success"]):
            return {"success": False, "error": "Component failure"}
        
        try:
            boxes = florence_results["boxes"]
            labels = florence_results["labels"]
            scores = florence_results["scores"]
            masks = sam2_results["masks"]
            
            # Filter by confidence
            filtered_indices = [
                i for i, score in enumerate(scores) 
                if score >= confidence_threshold
            ]
            
            filtered_boxes = [boxes[i] for i in filtered_indices]
            filtered_labels = [labels[i] for i in filtered_indices]
            filtered_scores = [scores[i] for i in filtered_indices]
            filtered_masks = [masks[i] for i in filtered_indices]
            
            # Combine masks
            combined_mask = None
            if filtered_masks:
                combined_mask = np.zeros_like(filtered_masks[0], dtype=bool)
                for mask in filtered_masks:
                    combined_mask = np.logical_or(combined_mask, mask)
            
            return {
                "success": True,
                "boxes": filtered_boxes,
                "labels": filtered_labels,
                "scores": filtered_scores,
                "masks": filtered_masks,
                "combined_mask": combined_mask,
                "image_size": (image.width, image.height),
                "num_objects": len(filtered_masks),
                "method": "sam2_florence_combined"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Result processing failed: {e}"}

    def save_results(self, results: Dict[str, Any], output_dir: str, base_name: str):
        """Save segmentation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not results["success"]:
            if self.debug:
                print(f"Cannot save failed results: {results.get('error', 'Unknown error')}")
            return
        
        try:
            # Save combined mask
            if results.get("combined_mask") is not None:
                combined_mask = results["combined_mask"]
                mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8), mode='L')
                mask_path = output_path / f"{base_name}_combined_mask.png"
                mask_img.save(mask_path)
                
                if self.debug:
                    print(f"Saved combined mask: {mask_path}")
            
            # Save individual masks
            for i, mask in enumerate(results.get("masks", [])):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                mask_path = output_path / f"{base_name}_mask_{i:02d}.png"
                mask_img.save(mask_path)
            
            # Save metadata
            metadata = {
                "num_objects": results.get("num_objects", 0),
                "image_size": results.get("image_size", [0, 0]),
                "method": results.get("method", "unknown"),
                "boxes": results.get("boxes", []),
                "labels": results.get("labels", []),
                "scores": results.get("scores", [])
            }
            
            metadata_path = output_path / f"{base_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if self.debug:
                print(f"Saved metadata: {metadata_path}")
                
        except Exception as e:
            if self.debug:
                print(f"Error saving results: {e}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="SAM2 + Florence2 Image Segmentation")
    
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--prompt", default="objects", help="Segmentation prompt")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--task", default="OD", choices=["OD", "CAPTION_TO_PHRASE_GROUNDING"])
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--sam2-model", help="Path to SAM2 model")
    parser.add_argument("--florence-model", help="Path to Florence2 model")
    parser.add_argument("--device", default="auto", help="Compute device")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return 1
    
    # Load image
    try:
        image = Image.open(input_path).convert("RGB")
        print(f"Loaded image: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1
    
    # Initialize segmenter
    print("Initializing SAM2 + Florence2 segmenter...")
    segmenter = SAM2FlorenceSegmenter(
        sam2_model_path=args.sam2_model,
        florence_model_path=args.florence_model,
        device=args.device,
        debug=args.debug
    )
    
    if not segmenter.available:
        print("Error: Segmenter initialization failed")
        return 1
    
    print("Models loaded successfully!")
    
    # Run segmentation
    print(f"Running segmentation with prompt: '{args.prompt}'")
    results = segmenter.segment_image(
        image=image,
        prompt=args.prompt,
        task=args.task,
        confidence_threshold=args.confidence
    )
    
    # Check results
    if not results["success"]:
        print(f"Segmentation failed: {results.get('error', 'Unknown error')}")
        return 1
    
    print(f"Segmentation successful! Found {results['num_objects']} objects")
    
    # Save results
    base_name = input_path.stem
    segmenter.save_results(results, args.output, base_name)
    
    print(f"Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())