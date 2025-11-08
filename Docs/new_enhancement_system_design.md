# PDF Extractor v09 - NEW Clear Enhancement System

**Date:** October 7, 2025  
**Major Redesign:** Replaced confusing profiles with clear, intuitive controls

---

## 🎯 Design Philosophy

### OLD SYSTEM (Confusing)
- ❌ "Auto" was only for sharpening, not full auto
- ❌ Profiles had hidden settings users couldn't see
- ❌ Auto-detection would override user choices
- ❌ Strength multiplier was non-intuitive
- ❌ Mixed terminology (profile vs mode vs method)

### NEW SYSTEM (Clear)
- ✅ **Enhancement Mode** = Top-level choice (auto/none/manual)
- ✅ **Auto** = Truly automatic - analyzes EVERY parameter
- ✅ **Manual** = Direct control with clear 0.0-1.0 sliders
- ✅ **None** = Fast extraction, no processing
- ✅ No hidden overrides - you get what you select

---

## 🎨 New UI Structure

### Main Control: `enhancement_mode`

**"auto"** (Default) - Full Auto Intelligence
- Analyzes each image individually
- Determines optimal settings for ALL parameters:
  - Noise level → denoise_strength (0.0-0.8)
  - Edge sharpness → sharpen_strength (0.0-0.8)
  - Dynamic range → tone_map_strength (0.0-1.0)
  - Color saturation → color_enhance_strength (0.0-1.0)
  - JPEG quality → artifact_removal (true/false)
- **Best for:** PDFs with varied image sources and quality
- **Speed:** ~30-40s (analyzes then enhances)

**"none"** - No Enhancement
- Extracts images without any processing
- Identical to simple extractor
- **Best for:** High-quality modern PDFs, speed priority
- **Speed:** ~5s

**"manual"** - You Control Everything
- Uses the exact values you specify in sliders below
- No auto-detection, no overrides
- **Best for:** Specific requirements, fine-tuning
- **Speed:** Depends on your settings

---

## 🎛️ Manual Controls

### When `enhancement_mode = "manual"`, these sliders are used:

**`denoise_strength`** (0.0-1.0, default: 0.0)
- 0.0 = No denoising
- 0.3 = Light noise reduction
- 0.5 = Medium noise reduction
- 0.8 = Heavy noise reduction (for scanned/noisy images)
- **Tip:** Start with 0.0, only increase if you see noise

**`sharpen_strength`** (0.0-1.0, default: 0.5)
- 0.0 = No sharpening
- 0.3 = Subtle sharpening
- 0.5 = Moderate sharpening (good default)
- 0.8 = Strong sharpening (for very blurry images)
- **Tip:** 0.5 is safe for most images

**`tone_map_strength`** (0.0-1.0, default: 0.0)
- 0.0 = No tone mapping
- 0.3 = Light contrast enhancement
- 0.5 = Moderate dynamic range expansion
- 0.8 = Strong tone mapping (for flat/dull images)
- **Tip:** Only use if images look flat or lack contrast

**`color_enhance_strength`** (0.0-1.0, default: 0.3)
- 0.0 = No color enhancement
- 0.3 = Light color boost (good default)
- 0.5 = Moderate color vibrancy
- 0.8 = Strong color saturation
- **Tip:** 0.3-0.5 works well for most content

**`artifact_removal`** (true/false, default: false)
- false = Don't remove artifacts
- true = Remove JPEG compression artifacts (8x8 blocks)
- **Tip:** Only enable if you see blocky JPEG artifacts

---

## 🔧 Sharpening Methods

Clear names that describe what they do:

**`gpu_basic`** (Default - Recommended)
- Fast GPU-accelerated unsharp mask
- Excellent results, <1 second processing
- **Best for:** Most use cases with GPU available

**`smart_adaptive`** (CPU - Slow but Intelligent)
- Analyzes image content and adapts sharpening
- Detects and prevents overshoot
- Takes 30-60s per image
- **Best for:** Critical quality work, varied content

**`hiraloam_gentle`** (CPU - Natural Look)
- High Radius, Low Amount technique
- Very natural, subtle enhancement
- Takes 20-40s per image
- **Best for:** Photos, portraits, natural scenes

**`edge_directional`** (CPU - Edge-Aware)
- Sharpens along edges, preserves smooth areas
- Direction-sensitive processing
- Takes 40-60s per image
- **Best for:** Technical drawings, diagrams

**`multiscale_quality`** (CPU - Highest Quality)
- Multi-scale Laplacian pyramid
- Best quality but slowest
- Takes 60-90s per image
- **Best for:** Publication-quality output

**`guided_filter`** (CPU - Edge-Preserving)
- Guided filter technique
- Preserves edges while smoothing
- Takes 30-50s per image
- **Best for:** Medical images, detailed content

---

## 📊 Auto Mode - How It Works

### Image Analysis

When `enhancement_mode = "auto"`, each image is analyzed:

```
1. NOISE ANALYSIS
   - Uses Laplacian variance method
   - Low variance → no denoising needed
   - High variance → apply denoising
   
2. SHARPNESS ANALYSIS
   - Measures gradient magnitude
   - Sharp edges → minimal sharpening
   - Blurry → strong sharpening
   
3. DYNAMIC RANGE ANALYSIS
   - Checks histogram spread (1st-99th percentile)
   - Full range → no tone mapping
   - Narrow range → expand dynamic range
   
4. COLOR SATURATION ANALYSIS
   - Converts to HSV, analyzes S channel
   - High saturation → no boost needed
   - Low saturation → enhance colors
   
5. JPEG ARTIFACT DETECTION
   - Checks image format and quality metadata
   - Looks for 8x8 block discontinuities
   - Detected → enable artifact removal
```

### Example Output

```
📊 Auto-Analysis Results:
   Noise: 0.156 → Denoise: 0.12
   Sharpness: 0.623 → Sharpen: 0.42
   Dynamic Range: 0.234 → Tone Map: 0.23
   Saturation: 0.512 → Color: 0.26
   JPEG Artifacts: Yes
   
🎨 Applying Auto-Detected Settings:
   Denoise: 0.12, Sharpen: 0.42, Tone: 0.23, Color: 0.26
   Artifacts: Remove, Method: gpu_basic
```

---

## ⚡ Performance Comparison

| Mode | Settings | Time | Use Case |
|------|----------|------|----------|
| **none** | N/A | ~5s | High-quality PDFs, speed priority |
| **manual** | All 0.0 | ~5s | Same as none |
| **manual** | Minimal (sharpen 0.3) | ~15s | Light touch-up |
| **manual** | Moderate (all 0.5) | ~30s | Standard enhancement |
| **manual** | Heavy (all 0.8) | ~50s | Problem sources |
| **auto** | Adaptive | ~30-40s | Varied quality (best choice) |
| **auto** + CPU sharpen | Adaptive + slow method | ~60-90s | Maximum quality |

---

## 🎯 Recommended Settings

### For Speed (5-10s)
```
enhancement_mode: none
```

### For Most PDFs (30s, balanced)
```
enhancement_mode: auto
sharpening_method: gpu_basic
```

### For Fine Control (your choice)
```
enhancement_mode: manual
denoise_strength: 0.0
sharpen_strength: 0.5
tone_map_strength: 0.0
color_enhance_strength: 0.3
artifact_removal: false
sharpening_method: gpu_basic
```

### For Problem PDFs (40-50s, aggressive)
```
enhancement_mode: manual
denoise_strength: 0.6
sharpen_strength: 0.7
tone_map_strength: 0.5
color_enhance_strength: 0.6
artifact_removal: true
sharpening_method: gpu_basic
```

### For Maximum Quality (60-90s)
```
enhancement_mode: auto
sharpening_method: multiscale_quality
```

---

## 🔄 Migration from Old System

### Old Profile → New Settings

**"None"** → `enhancement_mode: none`

**"Minimal"** → `enhancement_mode: manual`
- denoise: 0.0, sharpen: 0.2, tone_map: 0.0, color: 0.3
- artifact_removal: false

**"Digital Magazine"** → `enhancement_mode: manual`
- denoise: 0.1, sharpen: 0.5, tone_map: 0.2, color: 0.6
- artifact_removal: true

**"Scanned Photo"** → `enhancement_mode: manual`
- denoise: 0.5, sharpen: 0.7, tone_map: 0.6, color: 0.8
- artifact_removal: true

**"Vintage/Compressed"** → `enhancement_mode: manual`
- denoise: 0.8, sharpen: 0.8, tone_map: 0.8, color: 1.0
- artifact_removal: true

**But honestly, just use `auto` - it's smarter than the old profiles!**

---

## 💡 Tips & Best Practices

### 1. Start with Auto
```
enhancement_mode: auto
```
Let the system analyze and decide. Check the console output to see what it detected.

### 2. If Auto is Wrong
Switch to manual and adjust the specific parameter that's off:
```
enhancement_mode: manual
# Copy the auto-detected values from console
# Tweak only the parameter that needs adjustment
```

### 3. Batch Processing Mixed PDFs
```
enhancement_mode: auto
```
Each image gets individually analyzed and optimally processed.

### 4. Consistent Source Quality
If all images in a PDF are similar quality:
```
enhancement_mode: manual
# Set once, applies to all images
```

### 5. Speed vs Quality Trade-off
- **Speed priority:** `enhancement_mode: none` (5s)
- **Balanced:** `enhancement_mode: auto` + `gpu_basic` (30s)
- **Quality priority:** `enhancement_mode: auto` + `multiscale_quality` (90s)

---

## 🧪 Testing Checklist

- [ ] Test `enhancement_mode: none` - should match simple extractor speed
- [ ] Test `enhancement_mode: auto` - should analyze and print metrics
- [ ] Test `enhancement_mode: manual` with all sliders at 0.0 - should be fast
- [ ] Test `enhancement_mode: manual` with custom values - should use exact values
- [ ] Test each sharpening method individually
- [ ] Test with mixed-quality PDF (scanned + digital)
- [ ] Verify no auto-override when manual mode selected
- [ ] Check console output shows analysis results in auto mode

---

## 📚 Technical Details

### Image Quality Analyzer
**File:** `florence2_scripts/image_quality_analyzer.py`

**Methods:**
- `analyze(image)` → Returns dict with optimal parameters
- `_analyze_noise()` → Laplacian variance
- `_analyze_sharpness()` → Gradient magnitude
- `_analyze_dynamic_range()` → Histogram percentiles
- `_analyze_color_saturation()` → HSV analysis
- `_detect_jpeg_artifacts()` → Block discontinuity detection

### Modern Image Enhancer
**File:** `florence2_scripts/modern_image_enhancer.py`

**New API:**
```python
enhance_image_with_params(
    image,
    denoise=0.0,
    sharpen=0.5,
    tone_map=0.0,
    color_pop=0.3,
    artifact_removal=False,
    subject_boxes=None,
    sharpening_method="gpu_basic"
)
```

**Legacy API** (still supported):
```python
enhance_image(
    image,
    profile="Digital Magazine",
    strength=1.0,
    subject_boxes=None,
    sharpening_method="auto"
)
```

---

## ✨ Benefits of New System

1. **Clarity** - You know exactly what each parameter does
2. **Control** - Manual mode gives you direct control, no surprises
3. **Intelligence** - Auto mode is truly automatic for all parameters
4. **Honesty** - No hidden overrides or mysterious profile settings
5. **Flexibility** - Can mix auto mode with specific sharpening methods
6. **Transparency** - Console shows exactly what auto-detection found
7. **Speed options** - Clear choice from 5s to 90s processing
8. **Intuitiveness** - 0.0-1.0 scales are standard and understood

---

**Status:** Implementation in progress 🚧
**Next:** Wire up the new parameters in PDF extractor node
