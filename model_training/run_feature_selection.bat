@echo off
REM Feature Selection: Logit-L1 Gating
REM Freezes backbone, only optimizes gate logits with L1 on ^|w^|
REM Outputs: feature ranking + keep/drop recommendation

cd /d "%~dp0"
call C:\ProgramData\anaconda3\Scripts\activate.bat AI_stock_environment

echo ================================================================================
echo Feature Selection via Logit-L1 Gating
echo ================================================================================
echo Strategy: Freeze backbone, optimize gate logits with L1 on raw logits
echo   - L1 on ^|w^| (not ^|sigmoid(w)^|) for stronger sparsity
echo   - Only gate_logits are trainable (~466 params)
echo   - Everything else frozen
echo ================================================================================
echo.

python 5_feature_selection.py ^
  --checkpoint "runs/stage2_all_features/best.pt" ^
  --dataset_py ..\panel\2_dataset_memmap.py ^
  --memmap_dir ..\panel\memmap_data ^
  --lookback 60 ^
  --k 512 ^
  --batch_size 4 ^
  --seed 42 ^
  --epochs 10 ^
  --gate_lr 0.05 ^
  --gate_lambda 0.1 ^
  --task_loss_weight 1.0 ^
  --gate_threshold 0.1 ^
  --embed_dim 128 ^
  --hidden_size 256 ^
  --num_layers 2 ^
  --dropout 0.3 ^
  --output_dir "runs/feature_selection"

REM ============================================================
REM Tuning guide:
REM   --gate_lambda 0.1    Too few dropped? Increase to 0.3~0.5
REM                        Too many dropped? Decrease to 0.05
REM   --gate_lr 0.05       Gate not moving? Increase to 0.1
REM   --epochs 10          Not converged? Increase to 20
REM   --gate_threshold 0.1 Adjust cutoff for keep/drop decision
REM ============================================================

pause
