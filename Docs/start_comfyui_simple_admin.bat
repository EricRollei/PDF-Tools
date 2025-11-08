@echo off
REM Simple ComfyUI Admin Launcher

echo Starting ComfyUI as Administrator for Gallery-dl Cookie Access...

REM Request admin privileges and run ComfyUI
powershell -Command "Start-Process cmd -ArgumentList '/c cd /d \"A:\Comfy_Dec\ComfyUI\" && python main.py --auto-launch && pause' -Verb RunAs -WorkingDirectory \"A:\Comfy_Dec\ComfyUI\""
