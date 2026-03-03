@echo off
chcp 65001 >nul
REM 0_add_row_present.bat
REM Analyze data coverage and understand row_present concept

echo ================================================================================
echo Row Present Data Coverage Analysis
echo ================================================================================
echo.
echo This script will analyze:
echo   1. Daily stock count with data
echo   2. Coverage statistics
echo   3. Row present concept explanation
echo.
echo Note: Analysis only, no data modification
echo.
pause

call conda activate AI_stock_environment

python 0_add_row_present.py

echo.
echo ================================================================================
echo Analysis Complete
echo ================================================================================
echo.
echo Next step: Run 1_build_panel_memmap.bat to build panel
echo   -^> row_present will be set automatically based on data existence
echo.
pause
