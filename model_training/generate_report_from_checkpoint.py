"""
Generate comprehensive report from existing checkpoint

Usage:
    python generate_report_from_checkpoint.py --checkpoint runs/stage2_lambdarank_alpha158/best.pt
    
This script:
1. Loads the specified checkpoint
2. Evaluates on train/val/test splits
3. Generates comprehensive_report.json
4. No training required - quick evaluation only
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import from 4_train_stage2.py
sys.path.insert(0, os.path.dirname(__file__))
from model_lstm_mtl import LSTMMultiTask, LSTMMTLConfig
from comprehensive_report import generate_comprehensive_report

# Import dataset utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'panel'))
from importlib.util import spec_from_file_location, module_from_spec


def load_dataset_module(dataset_py_path):
    """Dynamically load dataset module"""
    spec = spec_from_file_location("dataset_module", dataset_py_path)
    dataset_module = module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    return dataset_module


def compute_ic_rankic_batch(pred, label, mask):
    """Compute IC and RankIC with std. Inputs: numpy arrays (B, K)"""
    B, K = pred.shape
    ic_list, ric_list = [], []
    
    for b in range(B):
        m = mask[b] > 0.5
        if m.sum() < 3:
            continue
        p = pred[b][m]
        y = label[b][m]
        if np.std(p) < 1e-12 or np.std(y) < 1e-12:
            continue
        # IC: Pearson correlation (vectorized)
        p_c = p - p.mean()
        y_c = y - y.mean()
        ic = np.dot(p_c, y_c) / (np.linalg.norm(p_c) * np.linalg.norm(y_c) + 1e-12)
        # RankIC: Spearman = Pearson on ranks (argsort-based, no scipy)
        n = len(p)
        pr = np.empty(n, dtype=np.float64)
        yr = np.empty(n, dtype=np.float64)
        pr[p.argsort()] = np.arange(n, dtype=np.float64)
        yr[y.argsort()] = np.arange(n, dtype=np.float64)
        pr_c = pr - pr.mean()
        yr_c = yr - yr.mean()
        ric = np.dot(pr_c, yr_c) / (np.linalg.norm(pr_c) * np.linalg.norm(yr_c) + 1e-12)
        if np.isfinite(ic):
            ic_list.append(ic)
        if np.isfinite(ric):
            ric_list.append(ric)
    
    if len(ic_list) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    ic_mean = float(np.mean(ic_list))
    ic_std = float(np.std(ic_list))
    ric_mean = float(np.mean(ric_list))
    ric_std = float(np.std(ric_list))
    
    icir = ic_mean / (ic_std + 1e-8)
    rankicir = ric_mean / (ric_std + 1e-8)
    
    return ic_mean, ric_mean, ic_std, ric_std, icir, rankicir


def run_evaluation(model, dataloader, device, use_cqb=False, beta_r=0.5, beta_c=0.5, use_gating=False):
    """Run evaluation on a dataloader"""
    model.eval()
    
    total_loss = 0.0
    total_reg = 0.0
    total_ce = 0.0
    total_ndcg = 0.0
    total_gate = 0.0
    
    # Collect on GPU, batch transfer at end
    all_pred_ret = []
    all_y_ret = []
    all_ret_mask = []
    all_mom_logits = []
    all_y_mom = []
    
    n_batches = 0
    
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["X"].to(device, non_blocking=True)
            y_ret = batch["y_ret"].to(device, non_blocking=True)
            y_mom = batch["y_mom"].to(device, non_blocking=True)
            ret_mask = batch["ret_mask"].to(device, non_blocking=True)
            
            # Reshape: [B, K, L, D] -> [B*K, L, D]
            B, K, L, D_in = X.shape
            x_flat = X.reshape(B*K, L, D_in)
            
            # Forward pass
            pred_ret_flat, mom_logits_flat = model(x_flat)
            
            # Reshape back: [B*K] -> [B, K], [B*K, 5] -> [B, K, 5]
            pred_ret = pred_ret_flat.reshape(B, K)
            mom_logits = mom_logits_flat.reshape(B, K, -1)
            
            # Compute MSE loss
            valid_ret = ret_mask > 0.5
            if valid_ret.sum() > 0:
                mse_loss = F.mse_loss(pred_ret[valid_ret], y_ret[valid_ret])
            else:
                mse_loss = torch.tensor(0.0, device=device)
            
            # Compute CE loss
            valid_mom = (y_mom >= 0)
            if valid_mom.sum() > 0:
                ce_loss = F.cross_entropy(
                    mom_logits[valid_mom].view(-1, 5),
                    y_mom[valid_mom].view(-1).long(),
                    reduction="mean"
                )
            else:
                ce_loss = torch.tensor(0.0, device=device)
            
            # Compute gate loss
            if use_gating:
                gate_loss = model.gate_l1(reduction="mean")
            else:
                gate_loss = torch.tensor(0.0, device=device)
            
            total_loss += (mse_loss + ce_loss).item()
            total_reg += mse_loss.item()
            total_ce += ce_loss.item()
            total_gate += gate_loss.item()
            
            all_pred_ret.append(pred_ret.detach())
            all_y_ret.append(y_ret.detach())
            all_ret_mask.append(ret_mask.detach())
            all_mom_logits.append(mom_logits.detach())
            all_y_mom.append(y_mom.detach())
            
            n_batches += 1
    
    # Batch transfer GPU -> CPU once (avoid per-batch sync)
    all_pred_ret = torch.cat(all_pred_ret, dim=0).cpu().numpy()
    all_y_ret = torch.cat(all_y_ret, dim=0).cpu().numpy()
    all_ret_mask = torch.cat(all_ret_mask, dim=0).cpu().numpy()
    all_mom_logits = torch.cat(all_mom_logits, dim=0).cpu().numpy()
    all_y_mom = torch.cat(all_y_mom, dim=0).cpu().numpy()
    
    # Compute IC/RankIC (already numpy)
    ic_mean, ric_mean, ic_std, ric_std, icir, rankicir = compute_ic_rankic_batch(
        all_pred_ret, all_y_ret, all_ret_mask
    )
    
    # Compute classification accuracy
    mom_pred = all_mom_logits.argmax(axis=-1)
    valid_mask = (all_y_mom >= 0)
    
    if valid_mask.sum() > 0:
        cls_acc = (mom_pred[valid_mask] == all_y_mom[valid_mask]).mean()
    else:
        cls_acc = 0.0
    
    # Per-class accuracy
    cls_acc_per_class = {}
    for c in range(5):
        class_mask = valid_mask & (all_y_mom == c)
        if class_mask.sum() > 0:
            cls_acc_per_class[f'acc_class{c}'] = (mom_pred[class_mask] == all_y_mom[class_mask]).mean()
        else:
            cls_acc_per_class[f'acc_class{c}'] = 0.0
    
    results = {
        'loss': total_loss / max(n_batches, 1),
        'reg': total_reg / max(n_batches, 1),
        'ce': total_ce / max(n_batches, 1),
        'ndcgL': total_ndcg / max(n_batches, 1),
        'gate': total_gate / max(n_batches, 1),
        'ic': ic_mean,
        'ric': ric_mean,
        'ic_std': ic_std,
        'ric_std': ric_std,
        'icir': icir,
        'rankicir': rankicir,
        'cls_acc': cls_acc,
        'ndcg': 0.0,  # Not computed in this simplified version
        'k': 0,
        'tau': 0,
    }
    results.update(cls_acc_per_class)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate report from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (e.g., runs/stage2_lambdarank_alpha158/best.pt)")
    parser.add_argument("--dataset_py", type=str, default="../panel/2_dataset_memmap.py", help="Path to dataset module")
    parser.add_argument("--memmap_dir", type=str, default="../panel/memmap_data", help="Path to memmap data")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window")
    parser.add_argument("--k", type=int, default=512, help="Number of stocks per day")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--exclude_features", type=str, default="^fund", help="Regex to exclude features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_lambdarank", action="store_true", help="Whether LambdaRank was used")
    parser.add_argument("--use_gating", action="store_true", default=True, help="Whether gating was used")
    parser.add_argument("--use_cqb", action="store_true", help="Whether CQB was used")
    
    # CQB parameters
    parser.add_argument("--beta0", type=float, default=0.5, help="CQB initial beta")
    parser.add_argument("--b_win", type=int, default=6, help="CQB window size")
    
    # Additional parameters for comprehensive report
    parser.add_argument("--ret_w", type=float, default=1.0, help="Return loss weight")
    parser.add_argument("--cls_w", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--lambda_ce", type=float, default=0.5, help="Lambda for CE loss")
    parser.add_argument("--lambdarank_k", type=int, default=50, help="LambdaRank top-k")
    parser.add_argument("--lambdarank_sigma", type=float, default=0.5, help="LambdaRank sigma")
    parser.add_argument("--lambdarank_bins", type=int, default=5, help="LambdaRank bins")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--gate_reduction", type=str, default="mean", help="Gate reduction method")
    
    # Control which splits to evaluate
    parser.add_argument("--eval_splits", type=str, default="train,val,test", 
                        help="Comma-separated splits to evaluate (e.g., 'val,test' or 'test')")
    
    args = parser.parse_args()
    
    # Infer save_dir from checkpoint path
    args.save_dir = os.path.dirname(args.checkpoint)
    
    # Check if checkpoint exists
    if not os.path.isfile(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    print(f"{'='*80}")
    print("Generate Report from Checkpoint")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset_py}")
    print(f"Memmap Dir: {args.memmap_dir}")
    print()
    
    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    epoch = ckpt.get("epoch", 0)
    best_rankic = ckpt.get("best_rankic", ckpt.get("best_val", 0.0))  # Backward compatible
    print(f"✓ Checkpoint: Epoch {epoch}, Best Val RankIC {best_rankic:.6f}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset module
    print(f"\nLoading dataset from {args.dataset_py}...")
    dataset_module = load_dataset_module(args.dataset_py)
    MemmapDayWindowDataset = dataset_module.MemmapDayWindowDataset
    build_dataloader = dataset_module.build_dataloader
    
    # Load calendar for splits
    import pandas as pd
    meta_path = Path(args.memmap_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        cal_df = pd.DataFrame({"date_str": meta["dates"]})
        cal_df["year"] = pd.to_datetime(cal_df["date_str"]).dt.year
    else:
        print(f"❌ meta.json not found in {args.memmap_dir}")
        return
    
    # Data splits
    train_years = list(range(2007, 2015))
    val_years = [2015, 2016]
    test_years = [2017, 2018]
    
    train_indices = cal_df[cal_df["year"].isin(train_years)].index.tolist()
    val_indices = cal_df[cal_df["year"].isin(val_years)].index.tolist()
    test_indices = cal_df[cal_df["year"].isin(test_years)].index.tolist()
    
    tr_start, tr_end = max(train_indices[0], args.lookback-1), train_indices[-1]
    va_start, va_end = max(val_indices[0], args.lookback-1), val_indices[-1]
    te_start, te_end = max(test_indices[0], args.lookback-1), test_indices[-1]
    
    print(f"\nData splits:")
    print(f"  Train: {tr_start}-{tr_end} ({len(range(tr_start, tr_end+1))} days)")
    print(f"  Val:   {va_start}-{va_end} ({len(range(va_start, va_end+1))} days)")
    print(f"  Test:  {te_start}-{te_end} ({len(range(te_start, te_end+1))} days)")
    
    # Feature filtering
    import re
    feature_indices = None
    feature_names_selected = None
    if args.exclude_features:
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        all_features = meta_data.get("feat_cols", [])
        F_original = len(all_features)
        D_original = meta_data.get("D", 0)
        
        pattern = re.compile(args.exclude_features)
        selected_features = [f for f in all_features if not pattern.match(f)]
        feature_indices_feat = [i for i, f in enumerate(all_features) if f in selected_features]
        feature_names_selected = selected_features
        F_selected = len(selected_features)
        
        # Map feature indices to X dimension indices
        # X structure: [features(F), isna(F), row_present(1)] = 2F+1
        # We need to select both feature and its isna column
        feature_indices_list = []
        for i in feature_indices_feat:
            feature_indices_list.append(i)                # feature column
            feature_indices_list.append(F_original + i)   # corresponding isna column
        feature_indices_list.append(2 * F_original)       # always keep row_present
        
        feature_indices = feature_indices_list
        D_selected = len(feature_indices)
        
        print(f"\n✓ Feature filtering: {F_original} -> {F_selected} features")
        print(f"  Input dimension: D = {D_selected} (was {D_original})")
        print(f"  Structure: {F_selected} features + {F_selected} isna + 1 row_present")
    
    # Create datasets
    print("\nCreating datasets...")
    ds_train = MemmapDayWindowDataset(
        args.memmap_dir, args.lookback, tr_start, tr_end, args.k, 42,
        sample_both_valid_only=False, sample_ret_valid_only=False,
        feature_indices=feature_indices, sample_present_only=True,
    )
    ds_val = MemmapDayWindowDataset(
        args.memmap_dir, args.lookback, va_start, va_end, args.k, 42,
        sample_both_valid_only=False, sample_ret_valid_only=False,
        feature_indices=feature_indices, sample_present_only=True,
    )
    ds_test = MemmapDayWindowDataset(
        args.memmap_dir, args.lookback, te_start, te_end, args.k, 42,
        sample_both_valid_only=False, sample_ret_valid_only=False,
        feature_indices=feature_indices, sample_present_only=True,
    )
    
    dl_train = build_dataloader(ds_train, args.batch_size, num_workers=0, shuffle=False, pin_memory=False)
    dl_val = build_dataloader(ds_val, args.batch_size, num_workers=0, shuffle=False, pin_memory=False)
    dl_test = build_dataloader(ds_test, args.batch_size, num_workers=0, shuffle=False, pin_memory=False)
    
    # Create model
    print("\nCreating model...")
    # Get total dimension (D) from a sample
    sample_batch = next(iter(dl_train))
    D = sample_batch["X"].shape[-1]
    raw_F = (D - 1) // 2
    print(f"Data dims: D={D}, F={raw_F} (expect D=2F+1={2*raw_F+1})")
    if 2*raw_F+1 != D:
        raise ValueError(f"Dimension mismatch! D={D}, but 2*F+1={2*raw_F+1}")
    
    # Create model config
    cfg = LSTMMTLConfig(
        input_dim=D,
        raw_feature_dim=raw_F,
        embed_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        num_classes=5,
        use_layernorm=True,
        use_gating=args.use_gating,
        gating_init_logit=2.0,
    )
    
    model = LSTMMultiTask(cfg, fixed_zero_idx=None)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Parse eval_splits
    eval_splits = [s.strip().lower() for s in args.eval_splits.split(',')]
    print(f"\n{'='*80}")
    print(f"Evaluating on selected splits: {', '.join(eval_splits)}")
    print(f"{'='*80}\n")
    
    # Default empty results
    def get_empty_results():
        return {
            'loss': 0.0, 'reg': 0.0, 'ce': 0.0, 'ndcgL': 0.0, 'gate': 0.0,
            'ic': 0.0, 'ric': 0.0, 'ic_std': 0.0, 'ric_std': 0.0,
            'icir': 0.0, 'rankicir': 0.0, 'cls_acc': 0.0,
            'ndcg': 0.0, 'k': 0, 'tau': 0,
            'acc_class0': 0.0, 'acc_class1': 0.0, 'acc_class2': 0.0,
            'acc_class3': 0.0, 'acc_class4': 0.0,
        }
    
    # Evaluate selected splits
    if 'train' in eval_splits:
        print("Evaluating on Train set...")
        tr = run_evaluation(model, dl_train, device, use_gating=args.use_gating)
        print(f"✓ Train - IC: {tr['ic']:.4f} (±{tr['ic_std']:.4f}), ICIR: {tr['icir']:.4f}, Acc: {tr['cls_acc']:.4f}")
    else:
        print("⊘ Skipping Train set")
        tr = get_empty_results()
    
    if 'val' in eval_splits:
        print("\nEvaluating on Val set...")
        va = run_evaluation(model, dl_val, device, use_gating=args.use_gating)
        print(f"✓ Val   - IC: {va['ic']:.4f} (±{va['ic_std']:.4f}), ICIR: {va['icir']:.4f}, Acc: {va['cls_acc']:.4f}")
    else:
        print("⊘ Skipping Val set")
        va = get_empty_results()
    
    if 'test' in eval_splits:
        print("\nEvaluating on Test set...")
        te = run_evaluation(model, dl_test, device, use_gating=args.use_gating)
        print(f"✓ Test  - IC: {te['ic']:.4f} (±{te['ic_std']:.4f}), ICIR: {te['icir']:.4f}, Acc: {te['cls_acc']:.4f}")
    else:
        print("⊘ Skipping Test set")
        te = get_empty_results()
    
    # Generate comprehensive report
    print(f"\n{'='*80}")
    print("Generating comprehensive report...")
    print(f"{'='*80}\n")
    
    report = generate_comprehensive_report(
        args=args,
        model=model,
        tr=tr,
        va=va,
        te=te,
        epoch=epoch,
        best_rankic=best_rankic,
        ckpt_path=args.checkpoint,
        device=device,
        dl_train=dl_train,
        dl_val=dl_val,
        dl_test=dl_test,
        cqb=None,
    )
    
    # Save report
    checkpoint_dir = os.path.dirname(args.checkpoint)
    report_path = os.path.join(checkpoint_dir, "comprehensive_report.json")
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Report saved: {report_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Checkpoint: Epoch {epoch}")
    print(f"\nIC / ICIR:")
    print(f"  Train: {tr['ic']:7.4f} / {tr['icir']:6.3f}")
    print(f"  Val:   {va['ic']:7.4f} / {va['icir']:6.3f}")
    print(f"  Test:  {te['ic']:7.4f} / {te['icir']:6.3f}")
    print(f"\nRankIC / RankICIR:")
    print(f"  Train: {tr['ric']:7.4f} / {tr['rankicir']:6.3f}")
    print(f"  Val:   {va['ric']:7.4f} / {va['rankicir']:6.3f}")
    print(f"  Test:  {te['ric']:7.4f} / {te['rankicir']:6.3f}")
    print(f"\nClassification Accuracy:")
    print(f"  Train: {tr['cls_acc']:6.2%}")
    print(f"  Val:   {va['cls_acc']:6.2%}")
    print(f"  Test:  {te['cls_acc']:6.2%}")
    print(f"\n✓ Complete! See {report_path} for full details.")


if __name__ == "__main__":
    main()
