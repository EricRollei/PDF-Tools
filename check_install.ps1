# Quick Installation Check for PDF Tools ComfyUI Node
# Run this from: A:\Comfy25\ComfyUI_windows_portable

Write-Host "================================" -ForegroundColor Cyan
Write-Host "PDF Tools Installation Check" -ForegroundColor Cyan
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

# Step 1: Check installations
Write-Host "Checking installed packages..." -ForegroundColor Cyan
Write-Host ""

$packages = @{
    "PyMuPDF" = "fitz"
    "Pillow" = "PIL"
    "numpy" = "numpy"
    "opencv-python" = "cv2"
    "transformers" = "transformers"
    "gallery-dl" = "gallery_dl"
    "yt-dlp" = "yt_dlp"
    "browser-cookie3" = "browser_cookie3"
}

foreach ($pkg in $packages.GetEnumerator()) {
    $name = $pkg.Key
    $import = $pkg.Value
    Write-Host "Checking $name..." -NoNewline
    
    $check = "import $import; print('OK')"
    $result = & .\python_embeded\python.exe -c $check 2>&1
    
    if ($result -match "OK") {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " MISSING" -ForegroundColor Yellow
    }
}

Write-Host ""

# Step 2: Offer installation
Write-Host "To install all requirements, run:" -ForegroundColor Cyan
Write-Host ".\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\PDF_tools\requirements.txt" -ForegroundColor White
Write-Host ""

$response = Read-Host "Install now? (y/n)"

if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "Installing packages..." -ForegroundColor Cyan
    & .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\PDF_tools\requirements.txt
    Write-Host ""
    Write-Host "Installation complete!" -ForegroundColor Green
} else {
    Write-Host "Skipped installation" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Check INSTALLATION_GUIDE.md for detailed setup instructions" -ForegroundColor Cyan
Write-Host ""
