# Batch/Folder Processing Guide

## Overview

**NEW Feature**: Both Simple PDF Image Extractor and Enhanced PDF Extractor v09 now support **automatic batch processing** of entire folders!

Simply provide a **folder path** instead of a file path, and the node will automatically detect and process all PDFs in the folder.

## How It Works

### Automatic Detection

The nodes automatically detect whether you've provided a file or folder path:

```python
# Single PDF (original behavior)
pdf_path = "C:/Documents/portfolio.pdf"
→ Extracts images from one PDF

# Folder of PDFs (NEW - auto-detected)
pdf_path = "C:/Documents/portfolios/"
→ Processes all PDFs in folder
```

**No configuration needed** - just change the path!

## Usage in ComfyUI

### Step 1: Prepare Your PDFs
```
my_portfolios/
  ├─ designer1.pdf
  ├─ designer2.pdf
  ├─ designer3.pdf
  └─ artist_work.pdf
```

### Step 2: Enter Folder Path
In the node's `pdf_path` parameter, enter the folder path:
```
C:/Documents/my_portfolios/
```
or
```
C:/Documents/my_portfolios
```
*(Both work - trailing slash optional)*

### Step 3: Run the Node
The node will:
1. ✅ Detect it's a folder
2. 🔍 Find all `.pdf` files
3. 📊 Report how many PDFs found
4. 🔄 Process each PDF sequentially
5. 💾 Save results in organized subfolders
6. 📄 Create a batch summary JSON

## Output Structure

### Folder Organization
```
output/simple_pdf_extraction/  (or output/pdf_extraction for v09)
  ├─ designer1_20251005_120000/
  │  ├─ page_001_image_01.png
  │  ├─ page_001_image_02.png
  │  └─ designer1_all_text.txt
  │
  ├─ designer2_20251005_120015/
  │  ├─ page_001_image_01.png
  │  └─ designer2_all_text.txt
  │
  ├─ designer3_20251005_120030/
  │  └─ page_001_image_01.png
  │
  └─ batch_summary_20251005_120000.json  ← Batch statistics
```

Each PDF gets its own timestamped subfolder, plus a batch summary file.

### Batch Summary JSON

**Simple Extractor** creates `batch_summary_YYYYMMDD_HHMMSS.json`:
```json
{
  "total_pdfs": 4,
  "processed": 3,
  "skipped": 1,
  "total_images": 47,
  "layered_pdfs": 2,
  "total_time_seconds": 15.3,
  "avg_time_per_pdf": 3.8,
  "processing_times": [2.1, 3.4, 5.2, 4.6],
  "results": [
    {
      "pdf": "designer1.pdf",
      "images": 12,
      "has_layers": true,
      "status": "processed"
    },
    {
      "pdf": "designer2.pdf",
      "images": 0,
      "has_layers": false,
      "status": "skipped"
    }
  ]
}
```

**v09** creates `batch_summary_v09_YYYYMMDD_HHMMSS.json`:
```json
{
  "total_pdfs": 4,
  "processed": 3,
  "skipped": 1,
  "total_images": 47,
  "total_enhanced": 45,
  "layered_pdfs": 2,
  "total_time_seconds": 480.5,
  "avg_time_per_pdf": 120.1,
  "processing_times": [95.2, 180.4, 120.3, 84.6],
  "results": [
    {
      "pdf": "designer1.pdf",
      "images": 12,
      "enhanced": 12,
      "has_layers": true,
      "time": 95.2,
      "status": "processed"
    }
  ]
}
```

## Console Output

### Simple Extractor (Batch Mode)
```
🗂️  BATCH MODE: Processing folder
📁 Folder: C:/Documents/my_portfolios
🔍 Searching for PDF files...
📚 Found 4 PDF files
🏃 Mode: All PDFs
============================================================

[1/4] Processing: designer1.pdf
------------------------------------------------------------
🚀 Simple PDF Image Extractor (Layer-Aware)
📄 PDF: C:/Documents/my_portfolios/designer1.pdf
✨ PDF has 2 layers!
   📋 Layer: 'Images' (ON)
   📋 Layer: 'Text' (ON)
🚀 Using super-fast layer-based extraction
📖 Processing 8 pages...
  📄 Page 1: 2 images
    ✅ Extracted: 2000×1500 → page_001_image_01.png
[... extraction continues ...]

[2/4] Processing: designer2.pdf
------------------------------------------------------------
[... continues for all PDFs ...]

============================================================
✅ Batch Extraction Complete
📁 Folder: my_portfolios
📚 Total PDFs: 4
✅ Processed: 3
⏭️  Skipped: 1
🖼️  Total images: 47
✨ Layered PDFs: 2
⏱️  Total time: 15.3s
📊 Avg per PDF: 3.8s
📄 Summary: output/simple_pdf_extraction/batch_summary_20251005_120000.json
============================================================
```

### v09 (Batch Mode)
```
🗂️  BATCH MODE: Enhanced PDF Extractor v09
📁 Folder: C:/Documents/my_portfolios
🔍 Searching for PDF files...
📚 Found 4 PDF files
🏃 Mode: All PDFs
============================================================

============================================================
[1/4] Processing: designer1.pdf
============================================================
🚀 Enhanced PDF Extractor v09 - Layer Detection + Analysis Engine
🔍 Enhanced PDF Extraction Started:
   📄 PDF: C:/Documents/my_portfolios/designer1.pdf
[... full v09 extraction process ...]

[2/4] Processing: designer2.pdf
[... continues ...]

============================================================
✅ Batch Extraction Complete (v09)
📁 Folder: my_portfolios
📚 Total PDFs: 4
✅ Processed: 3
⏭️  Skipped: 1
🖼️  Total images: 47
✨ Enhanced: 45
📋 Layered PDFs: 2
⏱️  Total time: 480.5s
📊 Avg per PDF: 120.1s
📄 Summary: output/pdf_extraction/batch_summary_v09_20251005_120000.json
============================================================
```

## Return Values (Batch Mode)

### Simple Extractor
```python
images, summary, image_count, has_layers = node.extract_images(folder_path, ...)

# images: Combined tensor of ALL images from ALL PDFs
# summary: Batch summary text (see example above)
# image_count: Total images from all PDFs
# has_layers: True if ANY PDF had layers
```

### v09
```python
extracted, enhanced, analysis, text, stats, output_path, has_layers = node.extract_enhanced(folder_path, ...)

# extracted: List of all extracted images from all PDFs
# enhanced: List of all enhanced images
# analysis: List of all page analyses
# text: Combined text from all PDFs
# stats: Batch statistics dict
# output_path: Base output directory
# has_layers: True if ANY PDF had layers
```

## Features in Batch Mode

### Works With All Parameters
All node parameters work identically in batch mode:

**Simple Extractor:**
- `min_width`, `min_height` - Applied to all PDFs
- `extract_text` - Text extracted from all PDFs
- `layers_only_mode` - Skips non-layered PDFs across entire batch
- `dpi` - Used for all PDFs

**v09:**
- All v09 parameters apply to entire batch
- `join_spreads`, `enable_image_enhancement`, etc.
- Each PDF gets full v09 treatment

### Layers-Only Mode in Batch

Perfect for filtering layered PDFs only:

```python
# Only process PDFs with layers, skip the rest
layers_only_mode = True

# Result: Non-layered PDFs show "skipped" status in batch summary
```

**Console output:**
```
[1/4] Processing: layered_portfolio.pdf
✨ PDF has 2 layers!
🚀 Using super-fast layer-based extraction
[... extraction ...]

[2/4] Processing: scanned_doc.pdf
📄 No layers detected - using standard extraction
⚠️  Layers-only mode enabled, but PDF has no layers. Skipping extraction.

[... continues ...]

Processed: 2 (only layered PDFs)
Skipped: 2 (non-layered PDFs)
```

### Error Handling

If a PDF fails to process:
- ❌ Error logged to console
- ⏭️  Batch continues with next PDF
- 📊 Error recorded in batch summary
- ✅ Other PDFs process normally

**Batch summary for failed PDF:**
```json
{
  "pdf": "corrupted.pdf",
  "images": 0,
  "enhanced": 0,
  "has_layers": false,
  "status": "error",
  "error": "PDF file is corrupted or encrypted"
}
```

## Performance

### Simple Extractor Batch Performance

**Layered PDFs (best case):**
- Per PDF: 2-5 seconds
- 10 PDFs: ~30 seconds
- 50 PDFs: ~150 seconds (2.5 minutes)

**Non-layered PDFs:**
- Per PDF: 5-10 seconds
- 10 PDFs: ~75 seconds
- 50 PDFs: ~500 seconds (8 minutes)

### v09 Batch Performance

**Layered PDFs (with layer detection):**
- Per PDF: 2-5 seconds (layer extraction)
- Plus AI analysis if needed: +30-120 seconds

**Non-layered PDFs (full AI analysis):**
- Per PDF: 50-250 seconds depending on pages
- 10 PDFs: 500-2500 seconds (8-42 minutes)
- Consider using Simple Extractor for large batches

### Performance Tips

1. **Use Simple Extractor for large batches** - Much faster if you don't need v09 features
2. **Enable layers_only_mode** if you only care about layered PDFs
3. **Check batch summary** to identify slow PDFs
4. **Process small batches first** to estimate total time

## Use Cases

### Portfolio Screening
```
Input: Folder with 20 portfolio PDFs
Mode: Simple Extractor with layers_only_mode=True
Result: Only extracts from professional layered portfolios
Time: 1-2 minutes
```

### Magazine Archive Digitization
```
Input: Folder with 100 magazine PDFs
Mode: Simple Extractor (standard)
Result: All images extracted from all magazines
Time: 10-15 minutes for layered, 1-2 hours for non-layered
```

### Client Project Assets
```
Input: Folder with mixed project PDFs
Mode: v09 with full analysis
Result: Enhanced images with layout analysis
Time: Varies by PDF complexity
```

### Quality Control
```
Input: Folder with new submissions
Mode: Simple Extractor
Result: Quick preview of all images
Review: Check batch_summary.json for statistics
```

## Comparison: Single vs Batch

| Feature | Single PDF Mode | Batch/Folder Mode |
|---------|----------------|-------------------|
| Input | File path | Folder path |
| Detection | N/A | Automatic |
| Output | One subfolder | Multiple subfolders |
| Summary | Text summary | JSON + Text summary |
| Statistics | Per-PDF stats | Aggregate + per-PDF stats |
| Return Values | Single PDF results | Combined results |
| Progress | Single progress | Per-PDF progress |
| Error Handling | Fails immediately | Continues on error |

## Tips & Best Practices

### 1. Organize Your PDFs First
Put all PDFs you want to process in one folder. Node only processes PDFs in the root folder (doesn't search subfolders).

### 2. Check Batch Summary
The JSON file contains detailed statistics:
- Which PDFs were processed successfully
- Which were skipped (and why)
- Processing time per PDF
- Total images found
- Layer detection results

### 3. Use Appropriate Node
- **Simple Extractor**: Fast, simple extraction, great for batches
- **v09**: Slow but thorough, use for smaller batches or when you need advanced features

### 4. Monitor Console Output
Watch for:
- PDFs being skipped (layers_only_mode)
- Errors processing specific files
- Layer detection results
- Processing time per PDF

### 5. Test First
Before processing 100 PDFs, test with 2-3 PDFs first to:
- Verify settings
- Check output quality
- Estimate total processing time

## Troubleshooting

**Q: "No images extracted from batch"**
- Check if PDFs actually have images
- Look at batch summary JSON to see per-PDF results
- Try single PDF mode on one file to debug

**Q: "Batch taking forever"**
- v09 is slow for large batches - use Simple Extractor
- Check if PDFs have layers (faster processing)
- Consider enabling layers_only_mode to skip slow PDFs

**Q: "Some PDFs skipped"**
- Check batch summary JSON for skip reasons
- If layers_only_mode=True, non-layered PDFs are skipped
- Errors logged in console and summary

**Q: "Can I process subfolders?"**
- Not currently - only PDFs in the root folder are processed
- Move all PDFs to a single folder first

**Q: "How to cancel batch?"**
- Stop ComfyUI execution (standard cancel)
- Already processed PDFs will remain in output folder
- Partial results are saved

## Examples

### Example 1: Quick Portfolio Review
```python
# In ComfyUI node:
pdf_path = "C:/Submissions/portfolios/"
output_directory = "output/portfolio_review"
min_width = 200
min_height = 200
extract_text = False
layers_only_mode = True  # Only layered portfolios

# Result: Fast extraction of professional work only
```

### Example 2: Archive All Magazine Images
```python
# In ComfyUI node:
pdf_path = "C:/Archives/magazines_2024/"
output_directory = "output/magazine_archive"
min_width = 100
min_height = 100
extract_text = True
layers_only_mode = False  # Get everything

# Result: All images + text from all magazines
```

### Example 3: Quality Analysis with v09
```python
# In ComfyUI node (v09):
pdf_path = "C:/Client/deliverables/"
output_directory = "output/client_analysis"
enable_image_enhancement = True
join_spreads = True
debug_mode = True

# Result: Full analysis with enhanced images
# Warning: Will be slow for many PDFs
```

## Summary

✅ **Auto-detection**: Just provide folder path, node handles the rest
✅ **Organized output**: Each PDF gets its own subfolder + batch summary
✅ **Statistics**: Detailed JSON with per-PDF and aggregate stats
✅ **Error resilient**: Continues processing even if one PDF fails
✅ **All features work**: Every parameter works in batch mode
✅ **Performance aware**: Simple Extractor recommended for large batches

**Getting Started:**
1. Put PDFs in a folder
2. Enter folder path in node
3. Run - node auto-detects and processes all PDFs
4. Check batch_summary.json for results
