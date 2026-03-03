# -*- coding: utf-8 -*-
"""
Feature Selection Mode via Logit-L1 Gate Sparsification

Purpose:
  - Load a trained Stage2 checkpoint (or train from scratch)
  - Freeze backbone (LSTM + heads), only optimize gate_logits
  - Use L1 on raw logits |w| (not |sigmoid(w)|) for aggressive sparsity
  - Run for a few epochs to identify important features
  - Output: sorted feature ranking + recommended keep/drop mask

Why logit-L1 works better:
  Standard gate L1: penalizes |sigmoid(w)| → gradient = λ * sigmoid'(w) * sign(g)
    → When w=2 (g=0.88), sigmoid'(w) ≈ 0.10 → very weak gradient
  Logit L1: penalizes |w| → gradient = λ * sign(w)
    → Constant gradient magnitude regardless of w value
    → Easily pushes w to large negative values (g → 0)

Output Files (saved to --output_dir):
  - feature_ranking.csv: all features sorted by gate value
  - selected_features.json: machine-readable mask for stage2 training
  - gate_history.csv: gate values per epoch
"""

import os
import sys
import json
import math
import argparse
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from model_lstm_mtl import LSTMMTLConfig, LSTMMultiTask


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ic_rankic_batch(pred, y, mask):
    """Quick IC/RankIC for monitoring."""
    B = pred.shape[0]
    ics, rics = [], []
    for b in range(B):
        m = mask[b] > 0.5
        if m.sum() < 3:
            continue
        pb = pred[b][m]
        yb = y[b][m]
        if np.std(pb) < 1e-12 or np.std(yb) < 1e-12:
            continue
        pb_c = pb - pb.mean()
        yb_c = yb - yb.mean()
        ic = np.dot(pb_c, yb_c) / (np.linalg.norm(pb_c) * np.linalg.norm(yb_c) + 1e-12)
        n = len(pb)
        pr = np.empty(n, dtype=np.float64)
        yr = np.empty(n, dtype=np.float64)
        pr[pb.argsort()] = np.arange(n, dtype=np.float64)
        yr[yb.argsort()] = np.arange(n, dtype=np.float64)
        pr_c = pr - pr.mean()
        yr_c = yr - yr.mean()
        ric = np.dot(pr_c, yr_c) / (np.linalg.norm(pr_c) * np.linalg.norm(yr_c) + 1e-12)
        if np.isfinite(ic):
            ics.append(ic)
        if np.isfinite(ric):
            rics.append(ric)
    ic_mean = float(np.mean(ics)) if ics else 0.0
    ric_mean = float(np.mean(rics)) if rics else 0.0
    return ic_mean, ric_mean


def run_eval(model, dl, device):
    """Quick evaluation: IC, RankIC, CE, Acc"""
    model.eval()
    all_pred, all_y, all_mask = [], [], []
    all_logits, all_ymom = [], []
    total_ce = 0.0
    n = 0

    with torch.inference_mode():
        for batch in dl:
            X = batch["X"].to(device, non_blocking=True)
            y_ret = batch["y_ret"].to(device, non_blocking=True)
            y_mom = batch["y_mom"].to(device, non_blocking=True)
            ret_mask = batch["ret_mask"].to(device, non_blocking=True)

            B, K, L, D_in = X.shape
            x_flat = X.reshape(B * K, L, D_in)
            pred_ret_flat, mom_logits_flat = model(x_flat)
            pred_ret = pred_ret_flat.reshape(B, K)
            mom_logits = mom_logits_flat.reshape(B, K, -1)

            # CE
            valid_mom = (y_mom >= 0)
            if valid_mom.sum() > 0:
                ce = F.cross_entropy(
                    mom_logits[valid_mom].reshape(-1, mom_logits.shape[-1]),
                    y_mom[valid_mom].reshape(-1).long(),
                    reduction="mean"
                )
                total_ce += ce.item()
                n += 1

            all_pred.append(pred_ret.detach())
            all_y.append(y_ret.detach())
            all_mask.append(ret_mask.detach())
            all_logits.append(mom_logits.detach())
            all_ymom.append(y_mom.detach())

    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
    all_y = torch.cat(all_y, dim=0).cpu().numpy()
    all_mask = torch.cat(all_mask, dim=0).cpu().numpy()
    ic, ric = compute_ic_rankic_batch(all_pred, all_y, all_mask)

    all_logits = torch.cat(all_logits, dim=0).cpu().numpy()
    all_ymom = torch.cat(all_ymom, dim=0).cpu().numpy()
    pred_cls = all_logits.argmax(axis=-1)
    valid = (all_ymom >= 0)
    acc = (pred_cls[valid] == all_ymom[valid]).mean() if valid.sum() > 0 else 0.0

    return {
        "ic": ic, "ric": ric,
        "ce": total_ce / max(n, 1),
        "acc": float(acc),
    }


def main():
    parser = argparse.ArgumentParser(description="Feature Selection via Logit-L1 Gating")

    # Data
    parser.add_argument("--dataset_py", type=str, default="../panel/2_dataset_memmap.py")
    parser.add_argument("--memmap_dir", type=str, default="../panel/memmap_data")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoint to start from
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained Stage2 checkpoint (best.pt)")

    # Feature selection hyperparams
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to run gate optimization")
    parser.add_argument("--gate_lr", type=float, default=0.05,
                        help="Learning rate for gate logits (higher than backbone)")
    parser.add_argument("--gate_lambda", type=float, default=0.1,
                        help="L1 penalty coefficient on logits")
    parser.add_argument("--task_loss_weight", type=float, default=1.0,
                        help="Weight for task loss (keep predictions reasonable)")
    parser.add_argument("--gate_threshold", type=float, default=0.1,
                        help="Gate values below this are considered 'dropped'")
    parser.add_argument("--init_logit", type=float, default=None,
                        help="Reset gate logits to this value before selection (None=keep checkpoint)")

    # Model architecture (must match checkpoint)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Output
    parser.add_argument("--output_dir", type=str, default="runs/feature_selection",
                        help="Directory to save selection results")

    args = parser.parse_args()

    set_seed(args.seed)

    # =========================================================================
    # Setup
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print("Feature Selection Mode (Logit-L1 Gating)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Gate λ: {args.gate_lambda}  |  Gate LR: {args.gate_lr}  |  Epochs: {args.epochs}")
    print(f"Threshold: {args.gate_threshold}")
    print()

    # =========================================================================
    # Load data
    # =========================================================================
    spec = importlib.util.spec_from_file_location("dataset_memmap", args.dataset_py)
    ds_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds_mod)

    meta_path = os.path.join(args.memmap_dir, "meta.json")
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    feature_names = meta.get("feat_cols", [])
    dates = meta.get("dates", [])
    F_total = len(feature_names)
    D = meta.get("D", 2 * F_total + 1)

    print(f"Features: {F_total}")
    print(f"Input dim: D = {D}")

    # Date splits - use train+val for gate optimization (no test leakage)
    cal_df = pd.DataFrame({"date_str": dates, "date_idx": range(len(dates))})
    cal_df["year"] = pd.to_datetime(cal_df["date_str"]).dt.year

    train_years = list(range(2007, 2015))
    val_years = [2015, 2016]

    train_idx = cal_df[cal_df["year"].isin(train_years)].index.tolist()
    val_idx = cal_df[cal_df["year"].isin(val_years)].index.tolist()

    tr_s, tr_e = max(train_idx[0], args.lookback - 1), train_idx[-1]
    va_s, va_e = max(val_idx[0], args.lookback - 1), val_idx[-1]

    print(f"\nTrain: {tr_s}-{tr_e} ({tr_e - tr_s + 1} days)")
    print(f"Val:   {va_s}-{va_e} ({va_e - va_s + 1} days)")

    def make_loader(day_s, day_e, shuffle):
        dataset = ds_mod.MemmapDayWindowDataset(
            memmap_dir=args.memmap_dir,
            lookback=args.lookback,
            day_start=day_s,
            day_end=day_e,
            k=args.k,
            seed=args.seed,
            sample_both_valid_only=False,
            sample_ret_valid_only=False,
            sample_present_only=True,      # ✓ 仅用当日row_present采样，不依赖t+1
        )
        return ds_mod.build_dataloader(
            dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
        )

    dl_train = make_loader(tr_s, tr_e, shuffle=True)
    dl_val = make_loader(va_s, va_e, shuffle=False)

    # =========================================================================
    # Load model from checkpoint
    # =========================================================================
    print(f"\nLoading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    epoch_loaded = ckpt.get("epoch", 0)
    print(f"  Checkpoint epoch: {epoch_loaded}")

    raw_F = (D - 1) // 2
    cfg = LSTMMTLConfig(
        input_dim=D,
        raw_feature_dim=raw_F,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gating=True,
        gating_init_logit=2.0,
    )

    model = LSTMMultiTask(cfg, fixed_zero_idx=None)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Verify gating exists
    if model.gating is None:
        print("❌ Model has no gating layer! Cannot do feature selection.")
        sys.exit(1)

    gate_param = model.gating.gate_logits
    print(f"  Gate logits: {gate_param.shape[0]} features")
    print(f"  Initial gate stats: mean={torch.sigmoid(gate_param).mean():.4f}, "
          f"min={torch.sigmoid(gate_param).min():.4f}, max={torch.sigmoid(gate_param).max():.4f}")

    # Optional: reset gate logits
    if args.init_logit is not None:
        with torch.no_grad():
            gate_param.fill_(args.init_logit)
        print(f"  Reset gate logits to {args.init_logit} (gate={torch.sigmoid(gate_param[0]):.4f})")

    # =========================================================================
    # Freeze everything except gate logits
    # =========================================================================
    for name, p in model.named_parameters():
        if "gate_logits" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Frozen: {total - trainable:,} params")
    print(f"  Trainable (gate only): {trainable:,} params")

    # Optimizer: only gate logits
    optimizer = AdamW([gate_param], lr=args.gate_lr, weight_decay=0)

    # =========================================================================
    # Baseline evaluation (before selection)
    # =========================================================================
    print(f"\n{'='*80}")
    print("Baseline (before feature selection)")
    print(f"{'='*80}")
    baseline = run_eval(model, dl_val, device)
    print(f"  Val IC={baseline['ic']:.4f}  RankIC={baseline['ric']:.4f}  "
          f"CE={baseline['ce']:.4f}  Acc={baseline['acc']:.3f}")

    # =========================================================================
    # Feature selection training loop
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"Feature Selection Training ({args.epochs} epochs)")
    print(f"{'='*80}")
    print(f"  Strategy: minimize task_loss + λ*mean(|logits|)")
    print(f"  λ = {args.gate_lambda}, task_weight = {args.task_loss_weight}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    gate_history = []  # track gate values each epoch

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                  enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Only gate is trainable, but model needs train mode for dropout etc.
        # Actually, we want deterministic forward → use eval for backbone
        # But gate_logits still gets grad. This is fine since we froze everything else.

        total_task_loss = 0.0
        total_gate_loss = 0.0
        n_batches = 0

        for batch in dl_train:
            X = batch["X"].to(device, non_blocking=True)
            y_ret = batch["y_ret"].to(device, non_blocking=True)
            y_mom = batch["y_mom"].to(device, non_blocking=True)
            ret_mask = batch["ret_mask"].to(device, non_blocking=True)

            B, K, L, D_in = X.shape
            x_flat = X.reshape(B * K, L, D_in)

            with amp_ctx:
                pred_ret_flat, mom_logits_flat = model(x_flat)
                pred_ret = pred_ret_flat.reshape(B, K)
                mom_logits = mom_logits_flat.reshape(B, K, -1)

                # Task loss (MSE + CE, simple and stable)
                valid_ret = ret_mask > 0.5
                if valid_ret.sum() > 0:
                    L_reg = F.mse_loss(pred_ret[valid_ret], y_ret[valid_ret])
                else:
                    L_reg = torch.zeros((), device=device)

                valid_mom = (y_mom >= 0)
                if valid_mom.sum() > 0:
                    L_ce = F.cross_entropy(
                        mom_logits[valid_mom].reshape(-1, mom_logits.shape[-1]),
                        y_mom[valid_mom].reshape(-1).long(),
                        reduction="mean"
                    )
                else:
                    L_ce = torch.zeros((), device=device)

                task_loss = L_reg + L_ce

                # Logit L1: penalize |w| directly (not |sigmoid(w)|)
                gate_l1 = model.gate_l1_logit(reduction="mean")

                loss = args.task_loss_weight * task_loss + args.gate_lambda * gate_l1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_task_loss += task_loss.item()
            total_gate_loss += gate_l1.item()
            n_batches += 1

        # Gate statistics
        with torch.no_grad():
            g = torch.sigmoid(gate_param).cpu()
            logits = gate_param.cpu()

        gate_mean = g.mean().item()
        gate_min = g.min().item()
        gate_max = g.max().item()
        n_below_thresh = (g < args.gate_threshold).sum().item()
        n_above_half = (g > 0.5).sum().item()

        # Val evaluation
        val_metrics = run_eval(model, dl_val, device)

        avg_task = total_task_loss / max(n_batches, 1)
        avg_gate = total_gate_loss / max(n_batches, 1)

        print(f"[Epoch {epoch:02d}] "
              f"task={avg_task:.4f} gate_l1={avg_gate:.4f} | "
              f"Val IC={val_metrics['ic']:.4f} RIC={val_metrics['ric']:.4f} Acc={val_metrics['acc']:.3f} | "
              f"Gate: mean={gate_mean:.3f} min={gate_min:.3f} max={gate_max:.3f} "
              f"<{args.gate_threshold}:{n_below_thresh}/{F_total} >0.5:{n_above_half}/{F_total}")

        # Record gate history
        gate_record = {"epoch": epoch, "gate_mean": gate_mean, "gate_min": gate_min,
                       "gate_max": gate_max, "n_dropped": n_below_thresh,
                       "n_kept": n_above_half, "val_ic": val_metrics["ic"],
                       "val_ric": val_metrics["ric"], "val_acc": val_metrics["acc"]}
        gate_history.append(gate_record)

    # =========================================================================
    # Analysis & Output
    # =========================================================================
    print(f"\n{'='*80}")
    print("Feature Selection Results")
    print(f"{'='*80}")

    with torch.no_grad():
        final_gates = torch.sigmoid(gate_param).cpu().numpy()
        final_logits = gate_param.cpu().numpy()

    # Sort by gate value (ascending = least important first)
    ranking = sorted(zip(feature_names, final_gates, final_logits),
                     key=lambda x: x[1])

    # Save ranking CSV
    ranking_df = pd.DataFrame([
        {"feature": name, "gate": float(g), "logit": float(w),
         "status": "DROP" if g < args.gate_threshold else "KEEP"}
        for name, g, w in ranking
    ])
    ranking_path = os.path.join(args.output_dir, "feature_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)

    # Save gate history
    history_df = pd.DataFrame(gate_history)
    history_path = os.path.join(args.output_dir, "gate_history.csv")
    history_df.to_csv(history_path, index=False)

    # Compute keep/drop lists
    kept_features = [name for name, g, _ in ranking if g >= args.gate_threshold]
    dropped_features = [name for name, g, _ in ranking if g < args.gate_threshold]

    # Save selection JSON
    selection = {
        "threshold": args.gate_threshold,
        "gate_lambda": args.gate_lambda,
        "gate_lr": args.gate_lr,
        "epochs": args.epochs,
        "source_checkpoint": args.checkpoint,
        "total_features": F_total,
        "kept_count": len(kept_features),
        "dropped_count": len(dropped_features),
        "kept_features": kept_features,
        "dropped_features": dropped_features,
        "baseline_val_ic": baseline["ic"],
        "baseline_val_ric": baseline["ric"],
        "final_val_ic": val_metrics["ic"],
        "final_val_ric": val_metrics["ric"],
        # For use in stage2 --include_features regex
        "include_regex": "|".join(f"^{f}$" for f in kept_features) if len(kept_features) < F_total else None,
        # For use in stage2 --exclude_features regex
        "exclude_regex": "|".join(f"^{f}$" for f in dropped_features) if dropped_features else None,
    }
    selection_path = os.path.join(args.output_dir, "selected_features.json")
    with open(selection_path, 'w', encoding='utf-8') as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nTotal features: {F_total}")
    print(f"  KEEP ({len(kept_features)}): gate >= {args.gate_threshold}")
    print(f"  DROP ({len(dropped_features)}): gate < {args.gate_threshold}")

    print(f"\n--- Top 10 MOST important (highest gate) ---")
    for name, g, w in reversed(ranking[-10:]):
        print(f"  {g:.4f} (logit={w:+.2f})  {name}")

    print(f"\n--- Top 10 LEAST important (lowest gate) ---")
    for name, g, w in ranking[:10]:
        print(f"  {g:.4f} (logit={w:+.2f})  {name}")

    if len(dropped_features) > 0:
        print(f"\n--- All DROPPED features ({len(dropped_features)}) ---")
        for name, g, w in ranking:
            if g < args.gate_threshold:
                print(f"  {g:.4f} (logit={w:+.2f})  {name}")

    print(f"\nPerformance comparison:")
    print(f"  Baseline:  IC={baseline['ic']:.4f}  RankIC={baseline['ric']:.4f}  Acc={baseline['acc']:.3f}")
    print(f"  After sel: IC={val_metrics['ic']:.4f}  RankIC={val_metrics['ric']:.4f}  Acc={val_metrics['acc']:.3f}")

    ic_delta = val_metrics['ic'] - baseline['ic']
    ric_delta = val_metrics['ric'] - baseline['ric']
    print(f"  Delta:     IC={ic_delta:+.4f}  RankIC={ric_delta:+.4f}")

    if abs(ric_delta) < 0.01 and len(dropped_features) > 0:
        print(f"\n✓ Good: Dropping {len(dropped_features)} features barely affects RankIC")
        print(f"  → These features are likely noise and can be safely excluded")
    elif ric_delta < -0.02:
        print(f"\n⚠ Warning: RankIC dropped significantly. Consider:")
        print(f"  - Reducing gate_lambda (current: {args.gate_lambda})")
        print(f"  - Raising threshold (current: {args.gate_threshold})")
        print(f"  - Running more epochs")

    print(f"\n{'='*80}")
    print(f"Output files saved to: {args.output_dir}")
    print(f"  - {ranking_path}")
    print(f"  - {history_path}")
    print(f"  - {selection_path}")
    print(f"{'='*80}")

    # Print usage hint
    if selection.get("exclude_regex"):
        print(f"\nTo use in Stage2 training:")
        if len(dropped_features) <= 20:
            print(f'  --exclude_features "{selection["exclude_regex"]}"')
        else:
            print(f"  (Too many dropped features for regex. Use selected_features.json directly)")
    print()


if __name__ == "__main__":
    main()
