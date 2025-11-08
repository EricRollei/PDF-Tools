@echo off
REM ComfyUI Startup Script with Admin Elevation for Gallery-dl Cookie Access

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as Administrator - Gallery-dl will have full cookie access
    goto :main
) else (
    echo Requesting Administrator privileges for cookie access...
    echo This allows Gallery-dl to read browser cookies from Chrome/Edge
    
    REM Re-run this script as administrator
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~f0\"' -Verb RunAs"
    exit /b
)

:main
echo.
echo ================================
echo  ComfyUI with Admin Privileges
echo  Gallery-dl Cookie Access: ENABLED
echo ================================
echo.

REM Change to ComfyUI directory (adjust this path to match your setup)
cd /d "A:\Comfy_Dec\ComfyUI"

REM Activate your Python environment if needed (uncomment and adjust)
REM call "A:\Comfy_Dec\ComfyUI\venv\Scripts\activate.bat"

REM Start ComfyUI
echo Starting ComfyUI...
python main.py --auto-launch

REM Keep window open if there's an error
if %errorLevel% neq 0 (
    echo.
    echo ComfyUI exited with error code: %errorLevel%
    pause
)
