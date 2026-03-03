@echo off
REM Generate comprehensive report from existing checkpoint
REM This script evaluates the model without retraining

cd /d "%~dp0"
call C:\ProgramData\anaconda3\Scripts\activate.bat AI_stock_environment

echo ================================================================================
echo Generate Report from Best Checkpoint
echo ================================================================================
echo This will:
echo   1. Load best.pt checkpoint
echo   2. Evaluate on selected splits (default: train,val,test)
echo   3. Generate comprehensive_report.json
echo   4. No training - evaluation only
echo.
echo Tip: Change --eval_splits to "val,test" or "test" for faster evaluation
echo ================================================================================
echo.

python generate_report_from_checkpoint.py ^
  --checkpoint "runs/stage2_lambdarank_alpha158/best.pt" ^
  --dataset_py ..\panel\2_dataset_memmap.py ^
  --memmap_dir ..\panel\memmap_data ^
  --lookback 60 ^
  --k 512 ^
  --batch_size 4 ^
  --exclude_features "^fund" ^
  --seed 42 ^
  --use_lambdarank ^
  --use_gating ^
  --use_cqb ^
  --beta0 0.5 ^
  --b_win 6 ^
  --ret_w 1.0 ^
  --cls_w 1.0 ^
  --lambda_ce 0.5 ^
  --lambdarank_k 50 ^
  --lambdarank_sigma 0.5 ^
  --lambdarank_bins 5 ^
  --embed_dim 128 ^
  --hidden_size 256 ^
  --num_layers 2 ^
  --dropout 0.3 ^
  --eval_splits "test"

REM Options for --eval_splits:
REM   "train,val,test" - All splits (full report, ~5-7 mins)
REM   "val,test"       - Val and Test only (~3-4 mins)
REM   "test"           - Test only (fastest, ~1-2 mins)

pause
