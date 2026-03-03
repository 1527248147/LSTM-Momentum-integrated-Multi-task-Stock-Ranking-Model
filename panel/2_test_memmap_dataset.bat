@echo off
REM Test memmap dataset to verify it works

echo ============================================================
echo Testing Memmap Dataset
echo ============================================================

call conda activate AI_stock_environment
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)

cd /d "%~dp0"
python 2_dataset_memmap.py ^
  --memmap_dir "memmap_data" ^
  --lookback 60 ^
  --k 512 ^
  --batch_size 4 ^
  --num_workers 0 ^
  --num_batches 5

pause
