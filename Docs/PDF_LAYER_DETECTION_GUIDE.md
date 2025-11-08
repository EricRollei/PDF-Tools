# PDF Layer Detection - Super Fast Extraction Guide

## Overview

**New in v09 & Simple Extractor**: Automatic PDF layer detection for lightning-fast extraction!

When PDFs have **Optional Content Groups (OCG)** - also called "layers" - images and text are stored in separate, labeled layers. This allows for **instant extraction** without AI processing.

## What Are PDF Layers?

PDF layers (OCG - Optional Content Groups) are like Photoshop layers in a PDF:
- Each layer has a **name** (e.g., "Images", "Text", "Background")
- Layers can be **turned on/off**
- Common in: Adobe InDesign exports, professional magazines, portfolios

### Example Layer Structure:
```
📋 Layer 1: "Images" (ON) - contains all photos/illustrations
📋 Layer 2: "Text" (ON) - contains all text content  
📋 Layer 3: "Background" (OFF) - decorative elements
```

## Performance Benefits

### Layered PDF Extraction:
- ⚡ **2-5 seconds** for typical portfolio
- 🎯 **Perfect accuracy** - uses native layer metadata
- 🚫 **No AI required** - direct layer extraction
- ✅ **Always fastest method**

### Non-Layered PDF (Standard):
- 🐌 **50-250 seconds** with AI analysis (v08/v09)
- 🚀 **5-10 seconds** with Simple Extractor (no AI)
- ⚠️ May miss some images or require manual filtering

## Nodes with Layer Detection

### 1. Simple PDF Image Extractor (Updated)
**Best for:** Fast extraction, portfolios, clean PDFs

**New Features:**
- ✨ Automatic layer detection (always enabled)
- 🚀 Super-fast layer-based extraction when detected
- 📋 Reports layer structure in output
- 🏃 **"Layers Only Mode"**: Skip extraction if no layers

**Parameters:**
```python
pdf_path: Path to your PDF
output_directory: Where to save images
min_width: Minimum image width (default: 100px)
min_height: Minimum image height (default: 100px)
extract_text: Extract text content (default: True)
layers_only_mode: Only extract if layers detected (default: False)
```

**Returns:**
```python
images: Extracted images as tensors
summary: Extraction summary text
image_count: Number of images extracted
has_layers: Boolean - True if PDF has layers
```

**When to use "Layers Only Mode":**
- ✅ You only want to process layered PDFs
- ✅ Skip non-layered PDFs automatically
- ✅ Batch processing where only layer extraction is reliable

### 2. Enhanced PDF Extractor v09
**Best for:** Complex PDFs, spread detection, AI analysis

**New Features:**
- ✨ Automatic layer detection (always enabled)
- 🚀 Layer-aware extraction when available
- 🤖 Falls back to AI analysis for non-layered PDFs
- 📋 Saves layer info to JSON file
- 🏃 **"Layers Only Mode"**: Skip extraction if no layers

**Parameters:**
```python
# ... (all v08 parameters) ...
layers_only_mode: Only extract if layers detected (default: False)
```

**Returns:**
```python
# ... (all v08 returns) ...
has_layers: Boolean - True if PDF has layers
```

**Additional Features:**
- Layer info saved to `layer_info.json` in output folder
- Layer detection added to extraction statistics
- Console reports layer structure during processing

## How Layer Detection Works

### Automatic Detection (Always Runs)
Both nodes automatically detect layers at the start:

```python
1. Open PDF
2. Call doc.get_layers() - PyMuPDF method
3. Parse layer metadata:
   - Layer names
   - Visibility state (ON/OFF)
   - Layer purposes (intent/usage)
4. Use layer-based extraction if layers exist
5. Fall back to standard extraction if no layers
```

### Layer Info Output
When layers are detected, you get:
```json
{
  "has_layers": true,
  "layer_count": 3,
  "layers": [
    {
      "name": "Images",
      "number": 0,
      "visible": true,
      "intent": ["View", "Design"],
      "usage": "Artwork"
    },
    {
      "name": "Text",
      "number": 1,
      "visible": true,
      "intent": ["View"],
      "usage": "Text"
    },
    {
      "name": "Background",
      "number": 2,
      "visible": false,
      "intent": [],
      "usage": ""
    }
  ]
}
```

## Console Output Examples

### Layered PDF (Simple Extractor):
```
🚀 Simple PDF Image Extractor (Layer-Aware)
📄 PDF: portfolio.pdf
📁 Output: output/simple_pdf_extraction/portfolio_20251005_143022
📏 Min size: 100×100
🏃 Fast mode: All PDFs

✨ PDF has 2 layers!
   📋 Layer: 'Images' (ON)
   📋 Layer: 'Text' (ON)
🚀 Using super-fast layer-based extraction
📖 Processing 8 pages...
  📄 Page 1: 3 images
    ✅ Extracted: 2000×1500 → page_001_image_01.png
    ✅ Extracted: 1800×1200 → page_001_image_02.png

✅ Extraction Complete
📄 PDF: portfolio.pdf
📋 Type: ✨ Layered PDF (fast extraction)
🖼️  Images extracted: 24
📝 Text extracted: Yes
⏱️  Time: 3.2s
🎨 Layers: 2
```

### Non-Layered PDF with Layers-Only Mode:
```
🚀 Simple PDF Image Extractor (Layer-Aware)
📄 PDF: old_scan.pdf
🏃 Fast mode: Layers only

📄 No layers detected - using standard extraction
⚠️  Layers-only mode enabled, but PDF has no layers. Skipping extraction.
```

### Layered PDF (v09):
```
🚀 Enhanced PDF Extractor v09 - Layer Detection + Analysis Engine
📄 PDF: magazine.pdf

✨ PDF Layer Detection:
   📋 Found 3 layers
      • 'Photos' (ON)
      • 'Text' (ON)
      • 'Bleed Marks' (OFF)
   🚀 Using optimized layer-aware extraction
   
📖 Processing 50 pages...
[extraction continues...]

📋 Layer info saved: output/pdf_extraction/layer_info.json
```

## Creating Layered PDFs

### From Adobe InDesign:
1. File → Export → Adobe PDF
2. Check "Create Acrobat Layers from Top-Level Layers"
3. Your InDesign layers become PDF layers

### From Illustrator:
1. File → Save As → Adobe PDF
2. Options → Create Acrobat Layers from Top-Level Layers

### From Photoshop:
1. File → Save As → Photoshop PDF
2. Layers will be preserved if supported

### From LibreOffice/OpenOffice:
Not supported - these create flattened PDFs

## Use Cases

### ✅ Perfect for Layer Detection:
- Professional portfolios from InDesign
- Magazine layouts with separate image/text layers
- Technical documentation with layered diagrams
- Marketing materials from design software

### ⚠️ Won't Have Layers:
- Scanned PDFs (no layers, just raster images)
- Web-generated PDFs (browser print)
- Simple exports from Word/PowerPoint
- PDFs with flattened content

## Workflow Recommendations

### For Portfolio Processing:
```
1. Use Simple PDF Image Extractor
2. Enable layers_only_mode: False (default)
3. Let it auto-detect layers
4. If layers detected: 2-5 second extraction
5. If no layers: Still extracts, just takes 5-10 seconds
```

### For Batch Processing (Layer PDFs Only):
```
1. Use Simple PDF Image Extractor  
2. Enable layers_only_mode: True
3. Non-layered PDFs will be skipped automatically
4. Check has_layers output to filter results
```

### For Complex Layouts (Spreads, AI Analysis):
```
1. Use Enhanced PDF Extractor v09
2. Layer detection automatic
3. If layers: Fast native extraction
4. If no layers: Full AI analysis (Florence2, Surya)
5. Spread joining, caption detection still work
```

## Technical Details

### PyMuPDF Layer Methods:
```python
doc.get_layers()      # List all layers
doc.get_layer(name)   # Get specific layer by name
doc.set_layer(...)    # Change layer visibility
```

### Layer Detection is Fast:
- Takes <100ms even on large PDFs
- No performance penalty
- Always worth checking

### Layer Extraction Benefits:
- Uses native PDF structure (most reliable)
- No image analysis needed
- No AI model loading
- Perfect bounding boxes
- No false positives

## Troubleshooting

### "No layers detected" but I know they exist:
- Check if PDF was flattened during export
- Verify layers in Adobe Acrobat (View → Show/Hide → Navigation Panels → Layers)
- Some PDFs have layers that aren't visible to PyMuPDF

### Layer detection works but extraction is slow:
- Simple Extractor: Should be 2-5 seconds
- v09: May still run AI analysis if needed
- Check console for actual layer-based extraction messages

### Want to force standard extraction:
- Simple Extractor: No option needed, just use as-is
- v09: Set join_spreads to False to skip advanced processing

## Summary

**Key Takeaways:**
- ✨ Layer detection is **always automatic** - no setup needed
- 🚀 Layered PDFs extract in **2-5 seconds** (10-50x faster)
- 📋 Console reports layer structure when detected
- 🏃 Optional "layers only" mode for selective processing
- ✅ Simple Extractor is best for portfolios and clean PDFs
- 🤖 v09 still offers AI analysis fallback when needed

**Best Practice:**
Start with Simple PDF Image Extractor. If layers are detected, you get instant results. If not, you still get fast standard extraction (5-10 seconds). Only use v09 when you need advanced features like spread detection or AI-powered image finding.
