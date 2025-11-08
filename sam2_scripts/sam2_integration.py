"""
Sam2 Integration

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

# sam2_integration.py
"""Integration helper for calling SAM2+Florence from PDF extractor"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile

class SAM2FlorenceIntegration:
    """Integration class for calling standalone SAM2+Florence script"""
    
    def __init__(self, script_path: str = "sam2_florence_segmentation.py", debug: bool = False):
        self.script_path = Path(script_path)
        self.debug = debug
        self.available = self.script_path.exists()
        
        if not self.available and self.debug:
            print(f"SAM2+Florence script not found at: {self.script_path}")
    
    def segment_image_file(self, 
                          image_path: str, 
                          prompt: str = "main content",
                          confidence: float = 0.3) -> Dict[str, Any]:
        """Segment an image file using the standalone script"""
        
        if not self.available:
            return {"success": False, "error": "Script not available"}
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run the standalone script
                cmd = [
                    "python", str(self.script_path),
                    str(image_path),
                    "--prompt", prompt,
                    "--output", temp_dir,
                    "--confidence", str(confidence)
                ]
                
                if self.debug:
                    cmd.append("--debug")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False, 
                        "error": f"Script failed: {result.stderr}"
                    }
                
                # Load results
                base_name = Path(image_path).stem
                metadata_path = Path(temp_dir) / f"{base_name}_metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    # Load combined mask if it exists
                    mask_path = Path(temp_dir) / f"{base_name}_combined_mask.png"
                    if mask_path.exists():
                        from PIL import Image
                        import torch
                        import numpy as np
                        
                        mask_img = Image.open(mask_path).convert('L')
                        mask_array = np.array(mask_img) > 128  # Convert to boolean
                        mask_tensor = torch.from_numpy(mask_array.astype(np.float32))
                        
                        return {
                            "success": True,
                            "combined_mask": mask_tensor,
                            "metadata": metadata,
                            "method": "sam2_florence_external"
                        }
                
                return {"success": False, "error": "No results found"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def segment_pil_image(self, 
                         pil_image, 
                         prompt: str = "main content",
                         confidence: float = 0.3) -> Dict[str, Any]:
        """Segment a PIL image by saving temporarily and calling script"""
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                pil_image.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                result = self.segment_image_file(temp_path, prompt, confidence)
                return result
            finally:
                Path(temp_path).unlink()  # Clean up temp file
                
        except Exception as e:
            return {"success": False, "error": str(e)}