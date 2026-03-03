@echo off
REM Build panel memmap data (one-time preprocessing)
REM This will take some time but only needs to run once

echo ============================================================
echo Building Panel Memmap Data
echo ============================================================

REM Activate conda environment
call conda activate AI_stock_environment
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'AI_stock_environment'
    echo Please check if the environment exists: conda env list
    pause
    exit /b 1
)

REM Run preprocessing script
cd /d "%~dp0"
python 1_build_panel_memmap.py

if errorlevel 1 (
    echo ERROR: Preprocessing failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Preprocessing complete!
echo Output directory: panel\memmap_data
echo ============================================================
pause
