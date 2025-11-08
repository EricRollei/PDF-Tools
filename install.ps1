# Quick Installation Script for PDF Tools ComfyUI Node
# Run this from: A:\Comfy25\ComfyUI_windows_portable

Write-Host "================================" -ForegroundColor Cyan
Write-Host "PDF Tools Installation Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path ".\python_embeded\python.exe")) {
    Write-Host "ERROR: python_embeded\python.exe not found!" -ForegroundColor Red
    Write-Host "Please run this script from: A:\Comfy25\ComfyUI_windows_portable" -ForegroundColor Red
    exit 1
}

Write-Host "Found Python executable" -ForegroundColor Green
Write-Host ""

# Step 1: Check current installations
Write-Host "Step 1: Checking current installations..." -ForegroundColor Cyan
Write-Host ""

# Check PyMuPDF
Write-Host "Checking PyMuPDF..." -NoNewline
$result = & .\python_embeded\python.exe -c "import fitz; print('OK')" 2>&1
if ($result -match "OK") { Write-Host " ✓ Installed" -ForegroundColor Green } else { Write-Host " ✗ NOT installed" -ForegroundColor Yellow }

# Check Pillow
Write-Host "Checking Pillow..." -NoNewline
$result = & .\python_embeded\python.exe -c "import PIL; print('OK')" 2>&1
if ($result -match "OK") { Write-Host " ✓ Installed" -ForegroundColor Green } else { Write-Host " ✗ NOT installed" -ForegroundColor Yellow }

# Check numpy
Write-Host "Checking numpy..." -NoNewline
$result = & .\python_embeded\python.exe -c "import numpy; print('OK')" 2>&1
if ($result -match "OK") { Write-Host " ✓ Installed" -ForegroundColor Green } else { Write-Host " ✗ NOT installed" -ForegroundColor Yellow }

# Check opencv
Write-Host "Checking opencv-python..." -NoNewline
$result = & .\python_embeded\python.exe -c "import cv2; print('OK')" 2>&1
if ($result -match "OK") { Write-Host " ✓ Installed" -ForegroundColor Green } else { Write-Host " ✗ NOT installed" -ForegroundColor Yellow }

# Check transformers
Write-Host "Checking transformers..." -NoNewline
$result = & .\python_embeded\python.exe -c "import transformers; print('OK')" 2>&1
if ($result -match "OK") { Write-Host " ✓ Installed" -ForegroundColor Green } else { Write-Host " ✗ NOT installed" -ForegroundColor Yellow }

# Check gallery-dl
Write-Host "Checking gallery-dl..." -NoNewline
$result = & .\python_embeded\python.exe -c "import gallery_dl; print('OK')" 2>&1
if ($result -match "OK") { Write-Host " ✓ Installed" -ForegroundColor Green } else { Write-Host " ✗ NOT installed" -ForegroundColor Yellow }

# Check yt-dlp
Write-Host "Checking yt-dlp..." -NoNewline
$result = & .\python_embeded\python.exe -c "import yt_dlp; print('OK')" 2>&1
if ($result -match "OK") { Write-Host " ✓ Installed" -ForegroundColor Green } else { Write-Host " ✗ NOT installed" -ForegroundColor Yellow }

# Check browser-cookie3
Write-Host "Checking browser-cookie3..." -NoNewline
$result = & .\python_embeded\python.exe -c "import browser_cookie3; print('OK')" 2>&1
if ($result -match "OK") { Write-Host " ✓ Installed" -ForegroundColor Green } else { Write-Host " ✗ NOT installed" -ForegroundColor Yellow }

Write-Host ""

# Step 2: Offer to install
Write-Host "Step 2: Installation..." -ForegroundColor Cyan
$response = Read-Host "Do you want to install all requirements now? (y/n)"

if ($response -eq 'y' -or $response -eq 'Y' -or $response -eq 'yes') {
    Write-Host ""
    Write-Host "Installing from requirements.txt..." -ForegroundColor Cyan
    & .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\PDF_tools\requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Installation completed!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Installation encountered errors. Check output above." -ForegroundColor Red
    }
} else {
    Write-Host "Skipping installation. You can install manually with:" -ForegroundColor Yellow
    Write-Host ".\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\PDF_tools\requirements.txt" -ForegroundColor Yellow
}

Write-Host ""

# Step 3: Test external tools
Write-Host "Step 3: Checking external tools..." -ForegroundColor Cyan
Write-Host ""

# Test gallery-dl
$galleryDlTest = & .\python_embeded\python.exe -m gallery_dl --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ gallery-dl: $galleryDlTest" -ForegroundColor Green
} else {
    Write-Host "✗ gallery-dl not working" -ForegroundColor Yellow
}

# Test yt-dlp
$ytDlpTest = & .\python_embeded\python.exe -m yt_dlp --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ yt-dlp: $ytDlpTest" -ForegroundColor Green
} else {
    Write-Host "✗ yt-dlp not working" -ForegroundColor Yellow
}

# Test ffmpeg
$ffmpegTest = & ffmpeg -version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ ffmpeg is available" -ForegroundColor Green
} else {
    Write-Host "✗ ffmpeg not found (optional, needed for yt-dlp audio extraction)" -ForegroundColor Yellow
    Write-Host "  Download from: https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Yellow
}

Write-Host ""

# Step 4: Check CUDA
Write-Host "Step 4: Checking GPU support..." -ForegroundColor Cyan
Write-Host ""

$cudaTest = & .\python_embeded\python.exe -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 2>&1
Write-Host $cudaTest

Write-Host ""

# Step 5: Summary
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Read INSTALLATION_GUIDE.md for detailed setup" -ForegroundColor White
Write-Host "2. Read CODE_OVERVIEW.md to understand the codebase" -ForegroundColor White
Write-Host "3. Check Docs/ folder for specific guides" -ForegroundColor White
Write-Host "4. Test with example URLs:" -ForegroundColor White
Write-Host "   - Gallery-dl: https://www.instagram.com/janaioannaa/" -ForegroundColor Gray
Write-Host "   - Yt-dlp: https://www.youtube.com/watch?v=dQw4w9WgXcQ" -ForegroundColor Gray
Write-Host ""

Write-Host "For authentication (Instagram, Reddit, etc.):" -ForegroundColor Cyan
Write-Host "1. Export cookies from your browser" -ForegroundColor White
Write-Host "2. Save to: custom_nodes\PDF_tools\configs\instagram_cookies.json" -ForegroundColor White
Write-Host "3. Or enable use_browser_cookies in the node" -ForegroundColor White
Write-Host ""

Write-Host "Installation script complete!" -ForegroundColor Green
Write-Host ""
