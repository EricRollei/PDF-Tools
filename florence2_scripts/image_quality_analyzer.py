"""
Image Quality Analyzer

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
Image Quality Analyzer for Auto Enhancement Mode

Analyzes images to determine optimal enhancement parameters:
- Noise level → denoise_strength
- Edge sharpness → sharpen_strength  
- Dynamic range → tone_map_strength
- Color saturation → color_enhance_strength
- JPEG artifacts → artifact_removal
"""

import numpy as np
from PIL import Image
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class ImageQualityAnalyzer:
    """Analyzes image characteristics to determine optimal enhancement settings"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
    
    def analyze(self, image: Image.Image) -> Dict[str, float]:
        """
        Analyze image and return optimal enhancement parameters
        
        Returns:
            dict with keys: denoise, sharpen, tone_map, color_enhance, artifact_removal
            All values 0.0-1.0 except artifact_removal (bool)
        """
        
        # Convert to numpy for analysis
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Run all analyses
        noise_level = self._analyze_noise(img_array)
        sharpness = self._analyze_sharpness(img_array)
        dynamic_range = self._analyze_dynamic_range(img_array)
        color_saturation = self._analyze_color_saturation(img_array)
        has_artifacts = self._detect_jpeg_artifacts(img_array, image)
        
        # Convert to enhancement strengths
        denoise = self._noise_to_denoise_strength(noise_level)
        sharpen = self._sharpness_to_sharpen_strength(sharpness)
        tone_map = self._dynamic_range_to_tonemap_strength(dynamic_range)
        color_enhance = self._saturation_to_color_strength(color_saturation)
        
        result = {
            "denoise": denoise,
            "sharpen": sharpen,
            "tone_map": tone_map,
            "color_enhance": color_enhance,
            "artifact_removal": has_artifacts,
            # Raw metrics for debugging
            "noise_level": noise_level,
            "sharpness_score": sharpness,
            "dynamic_range": dynamic_range,
            "saturation_score": color_saturation
        }
        
        if self.debug_mode:
            print(f"    📊 Auto-Analysis Results:")
            print(f"       Noise: {noise_level:.3f} → Denoise: {denoise:.2f}")
            print(f"       Sharpness: {sharpness:.3f} → Sharpen: {sharpen:.2f}")
            print(f"       Dynamic Range: {dynamic_range:.3f} → Tone Map: {tone_map:.2f}")
            print(f"       Saturation: {color_saturation:.3f} → Color: {color_enhance:.2f}")
            print(f"       JPEG Artifacts: {'Yes' if has_artifacts else 'No'}")
        
        return result
    
    def _analyze_noise(self, img_array: np.ndarray) -> float:
        """
        Estimate noise level using Laplacian variance method
        Returns: 0.0 (no noise) to 1.0 (very noisy)
        """
        try:
            # Convert to grayscale for noise analysis
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Compute Laplacian
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            
            # Convolve (simplified - using basic numpy)
            h, w = gray.shape
            result = np.zeros_like(gray)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    result[i, j] = np.sum(gray[i-1:i+2, j-1:j+2] * laplacian)
            
            # Variance of Laplacian as noise estimate
            variance = np.var(result)
            
            # Normalize: typical range is 0.0001 to 0.01
            # High variance = high noise
            noise_level = np.clip(variance * 100, 0, 1)
            
            return float(noise_level)
            
        except Exception as e:
            if self.debug_mode:
                print(f"       ⚠️ Noise analysis failed: {e}")
            return 0.1  # Default low noise
    
    def _analyze_sharpness(self, img_array: np.ndarray) -> float:
        """
        Estimate image sharpness using gradient magnitude
        Returns: 0.0 (blurry) to 1.0 (sharp)
        """
        try:
            # Convert to grayscale
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Compute gradients (simple finite differences)
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            
            # Gradient magnitude (mean of absolute gradients)
            mag_x = np.mean(np.abs(grad_x))
            mag_y = np.mean(np.abs(grad_y))
            sharpness = (mag_x + mag_y) / 2
            
            # Normalize: typical range is 0.01 to 0.15
            # Low gradient = blurry, high gradient = sharp
            normalized = np.clip((sharpness - 0.02) / 0.10, 0, 1)
            
            return float(normalized)
            
        except Exception as e:
            if self.debug_mode:
                print(f"       ⚠️ Sharpness analysis failed: {e}")
            return 0.5  # Default medium sharpness
    
    def _analyze_dynamic_range(self, img_array: np.ndarray) -> float:
        """
        Analyze dynamic range (histogram spread)
        Returns: 0.0 (full range used) to 1.0 (narrow range, needs expansion)
        """
        try:
            # Convert to grayscale
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Get percentiles (avoid extreme outliers)
            p1 = np.percentile(gray, 1)
            p99 = np.percentile(gray, 99)
            
            # Dynamic range (0 to 1)
            used_range = p99 - p1
            
            # If range is narrow, needs tone mapping
            # Full range (0.98) = 0.0 (no tone mapping needed)
            # Narrow range (0.5) = 0.5 (moderate tone mapping)
            # Very narrow (0.2) = 1.0 (strong tone mapping)
            
            if used_range > 0.85:
                need_tonemap = 0.0  # Good range
            elif used_range > 0.6:
                need_tonemap = (0.85 - used_range) / 0.25  # 0.0 to 1.0
            else:
                need_tonemap = 0.5 + (0.6 - used_range) / 0.8  # 0.5 to 1.0
            
            return float(np.clip(need_tonemap, 0, 1))
            
        except Exception as e:
            if self.debug_mode:
                print(f"       ⚠️ Dynamic range analysis failed: {e}")
            return 0.0  # Default no tone mapping
    
    def _analyze_color_saturation(self, img_array: np.ndarray) -> float:
        """
        Analyze color saturation
        Returns: 0.0 (oversaturated) to 1.0 (desaturated, needs enhancement)
        """
        try:
            # Convert to HSV to get saturation channel
            # Simple RGB to HSV conversion
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            v = np.maximum(np.maximum(r, g), b)
            s = v - np.minimum(np.minimum(r, g), b)
            
            # Avoid division by zero
            s = np.where(v > 0, s / (v + 1e-8), 0)
            
            # Mean saturation
            mean_sat = np.mean(s)
            
            # If saturation is low, needs enhancement
            # High sat (0.8+) = 0.0 (no enhancement needed)
            # Medium sat (0.4-0.7) = 0.3 (light enhancement)  
            # Low sat (<0.3) = 0.8 (strong enhancement)
            
            if mean_sat > 0.6:
                need_color = 0.0
            elif mean_sat > 0.3:
                need_color = (0.6 - mean_sat) / 0.3 * 0.5  # 0.0 to 0.5
            else:
                need_color = 0.5 + (0.3 - mean_sat) / 0.3 * 0.5  # 0.5 to 1.0
            
            return float(np.clip(need_color, 0, 1))
            
        except Exception as e:
            if self.debug_mode:
                print(f"       ⚠️ Color saturation analysis failed: {e}")
            return 0.3  # Default light enhancement
    
    def _detect_jpeg_artifacts(self, img_array: np.ndarray, image: Image.Image) -> bool:
        """
        Detect JPEG compression artifacts
        Returns: True if artifacts likely present
        """
        try:
            # Check image format/quality from metadata
            format_info = image.format if hasattr(image, 'format') else None
            
            if format_info == 'JPEG':
                # Check quality if available
                if hasattr(image, 'info') and 'quality' in image.info:
                    quality = image.info['quality']
                    if quality < 85:
                        return True
                
                # Check for block artifacts (8x8 DCT blocks)
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
                
                # Sample a region and check for 8-pixel periodicities
                h, w = gray.shape
                if h > 100 and w > 100:
                    sample = gray[50:150, 50:150]
                    
                    # Check horizontal differences at 8-pixel intervals
                    diffs_8 = []
                    diffs_7 = []
                    
                    for i in range(0, sample.shape[0]-8, 8):
                        diff_8 = np.abs(sample[i+7] - sample[i+8] if i+8 < sample.shape[0] else 0)
                        diff_7 = np.abs(sample[i+6] - sample[i+7] if i+7 < sample.shape[0] else 0)
                        diffs_8.append(np.mean(diff_8))
                        diffs_7.append(np.mean(diff_7))
                    
                    if len(diffs_8) > 0 and len(diffs_7) > 0:
                        # If 8-pixel boundaries have stronger discontinuities
                        if np.mean(diffs_8) > np.mean(diffs_7) * 1.3:
                            return True
            
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"       ⚠️ Artifact detection failed: {e}")
            return False
    
    # Conversion functions: metric → enhancement strength
    
    def _noise_to_denoise_strength(self, noise_level: float) -> float:
        """Convert noise level to denoise strength"""
        # Low noise (0.0-0.2) = 0.0 (no denoising)
        # Medium noise (0.2-0.5) = 0.3-0.5 (light/medium)
        # High noise (0.5-1.0) = 0.5-0.8 (strong)
        
        if noise_level < 0.15:
            return 0.0
        elif noise_level < 0.4:
            return noise_level * 0.8  # 0.12 to 0.32
        else:
            return 0.3 + (noise_level - 0.4) * 0.5  # 0.3 to 0.6
    
    def _sharpness_to_sharpen_strength(self, sharpness: float) -> float:
        """Convert sharpness score to sharpen strength (inverted)"""
        # Sharp image (0.8-1.0) = 0.0-0.2 (minimal sharpening)
        # Medium (0.4-0.7) = 0.3-0.5 (moderate)
        # Blurry (0.0-0.3) = 0.6-0.8 (strong)
        
        if sharpness > 0.7:
            return 0.1
        elif sharpness > 0.4:
            return 0.3 + (0.7 - sharpness) * 0.7  # 0.3 to 0.5
        else:
            return 0.5 + (0.4 - sharpness) * 0.75  # 0.5 to 0.8
    
    def _dynamic_range_to_tonemap_strength(self, need_tonemap: float) -> float:
        """Direct mapping - already computed correctly"""
        return need_tonemap
    
    def _saturation_to_color_strength(self, need_color: float) -> float:
        """Direct mapping - already computed correctly"""
        return need_color
