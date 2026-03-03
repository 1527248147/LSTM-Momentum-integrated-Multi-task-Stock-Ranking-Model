@echo off
REM Stage2 Training: LambdaRank + Exclude Fund Features (Stable Config)
REM Combines LambdaRank loss with alpha158-only features
REM Stability settings: lr=1e-4, sigma=0.5, grad_clip=3.0, patience=15

cd /d "%~dp0"
call C:\ProgramData\anaconda3\Scripts\activate.bat AI_stock_environment

python 4_train_stage2.py ^
  --dataset_py ..\panel\2_dataset_memmap.py ^
  --memmap_dir ..\panel\memmap_data ^
  --lookback 60 ^
  --k 512 ^
  --batch_size 4 ^
  --epochs 100 ^
  --lr 1e-4 ^
  --weight_decay 1e-3 ^
  --patience 10 ^
  --exclude_features "^fund" ^
  --use_gating ^
  --gate_warmup_epochs 5 ^
  --gate_ramp_epochs 20 ^
  --gate_l1_max 5e-3 ^
  --grad_clip 3.0 ^
  --use_cqb ^
  --beta0 0.5 ^
  --b_win 6 ^
  --ret_w 1.0 ^
  --cls_w 1.0 ^
  --lambda_ce 0.5 ^
  --tau_ratio 0.2 ^
  --use_lambdarank ^
  --lambdarank_k 50 ^
  --lambdarank_sigma 0.5 ^
  --lambdarank_bins 5 ^
  --save_dir "runs/stage2_lambdarank_alpha158"

pause
