# -*- coding: utf-8 -*-
"""
3_train_stage1.py

Stage 1：先把训练闭环跑通（不做 Adaptive-k ApproxNDCG / CQB）
- loss = MSE(label_ret_1d) + CE(label_mom_cls)
- 输入 gating + L1（warmup 若干 epoch 为 0，然后线性 ramp 到最大值）

默认采用论文切分：
- Train: 2007–2014
- Val  : 2015–2016
- Test : 2017–2020

并使用论文量级超参默认值：
- lr=2e-4, weight_decay=1e-3, epochs<=100
"""


# python 3_train_stage1.py ^
#   --dataset_py "C:\AI_STOCK\model\Momentum-integrated Multi-task Stock Recommendation\panel\2_dataset_memmap.py" ^
#   --memmap_dir "C:\AI_STOCK\model\Momentum-integrated Multi-task Stock Recommendation\panel\memmap_data" ^
#   --calendar_csv "C:\AI_STOCK\dataset\ohlcv_ricequant_2004-2025_parquet_suspension_false\trading_calendar_from_merged.csv" ^
#   --lookback 60 ^
#   --k 512 ^
#   --batch_size 4 ^
#   --use_gating ^
#   --exclude_regex "(?i)dividend" ^
#   --amp ^
#   --shuffle ^
#   --save_dir "C:\AI_STOCK\model\Momentum-integrated Multi-task Stock Recommendation\model_training\runs\stage1_gating_memmap" ^
#   --num_workers 2 ^
#   --max_train_batches 200 ^
#   --max_eval_batches 200




# -*- coding: utf-8 -*-
"""
3_train_stage1.py (使用 memmap 高速数据加载)

Stage 1 Baseline（暂不做 ApproxNDCG / CQB）：
- loss = MSE(label_ret_1d) + CE(label_mom_cls)
- 动量无效标签用 -1 表示：必须在 CE 里 ignore_index=-1，否则 GPU 会 device-side assert
- 输入 gating：x' = g ⊙ x，对 g 加 L1（warmup=0 -> 线性 ramp 到 max）
- 数据源：使用 panel/memmap_data 预处理好的 memmap 文件（无需再读 parquet）

默认采用论文切分：
- Train: 2007–2014
- Val  : 2015–2016
- Test : 2017–2020
"""



import argparse
import importlib.util
import json
import os
import random
import re
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from model_lstm_mtl import LSTMMTLConfig, LSTMMultiTask


# -----------------------------
# 动态 import（2_dataset_memmap.py 文件名以数字开头，不能直接 import）
# -----------------------------
def load_module_from_path(py_path: str, module_name: str = "dataset_memmap_mod"):
    py_path = os.path.abspath(py_path)
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法从路径加载模块：{py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# GPU / 环境信息
# -----------------------------
def print_torch_env(device: torch.device):
    print(f"[ENV] torch.__version__ = {torch.__version__}")
    print(f"[ENV] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[ENV] torch.version.cuda = {torch.version.cuda}")
    print(f"[ENV] cudnn.enabled = {torch.backends.cudnn.enabled}")
    print(f"[ENV] cudnn.version = {torch.backends.cudnn.version()}")
    print(f"[ENV] cudnn.benchmark = {torch.backends.cudnn.benchmark}")
    print(f"[ENV] cudnn.deterministic = {torch.backends.cudnn.deterministic}")
    print(f"[ENV] device = {device}")

    if device.type == "cuda":
        cnt = torch.cuda.device_count()
        cur = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(cur)
        total_gb = prop.total_memory / (1024 ** 3)
        print(f"[GPU] cuda device count = {cnt}")
        print(f"[GPU] current cuda device = {cur}")
        print(f"[GPU] name = {prop.name}")
        print(f"[GPU] total_memory = {total_gb:.2f} GB")


def gpu_mem_str(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return f"gpu_mem(MB) alloc={alloc:.0f} reserved={reserved:.0f}"


# -----------------------------
# 日历与时间切分
# -----------------------------
def load_calendar_dates(calendar_csv: str) -> pd.Series:
    df = pd.read_csv(calendar_csv)
    # 尝试自动识别日期列名
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "datetime" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"calendar_csv 的日期列 {date_col} 存在无法解析的值")
    return dates


def year_range_to_idx(dates: pd.Series, start_year: int, end_year: int) -> Tuple[int, int]:
    years = dates.dt.year.values
    m = (years >= start_year) & (years <= end_year)
    idxs = np.where(m)[0]
    if len(idxs) == 0:
        raise ValueError(f"日历中找不到年份区间 {start_year}-{end_year}")
    return int(idxs.min()), int(idxs.max())


# -----------------------------
# IC / RankIC（按日）
# -----------------------------
def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    vx = torch.sqrt((x * x).mean() + eps)
    vy = torch.sqrt((y * y).mean() + eps)
    return float(((x * y).mean() / (vx * vy + eps)).item())


def rankdata_torch(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.argsort(x)
    ranks = torch.empty_like(tmp, dtype=torch.float32)
    ranks[tmp] = torch.arange(len(x), device=x.device, dtype=torch.float32)
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = rankdata_torch(x)
    ry = rankdata_torch(y)
    return pearson_corr(rx, ry)


@torch.no_grad()
def batch_ic_rankic(pred_ret: torch.Tensor, y_ret: torch.Tensor, ret_mask: torch.Tensor) -> Tuple[float, float, int]:
    """
    pred_ret, y_ret, ret_mask: [B,K]
    """
    B = pred_ret.shape[0]
    ics, rics = [], []
    for b in range(B):
        m = ret_mask[b].bool()
        if int(m.sum().item()) < 2:
            continue
        p = pred_ret[b][m].detach().cpu()
        t = y_ret[b][m].detach().cpu()
        ics.append(pearson_corr(p, t))
        rics.append(spearman_corr(p, t))
    if len(ics) == 0:
        return 0.0, 0.0, 0
    return float(np.mean(ics)), float(np.mean(rics)), len(ics)


# -----------------------------
# 动量标签清洗：关键（-1 必须 ignore）
# -----------------------------
def sanitize_mom_labels(y_mom: torch.Tensor, num_classes: int):
    """
    把 y_mom 清洗成：
      - 有效：0..C-1
      - 无效：-1
    返回：(y_norm, valid_bool_mask)
    """
    C = int(num_classes)
    y = y_mom.clone()
    valid = (y >= 0) & (y < C)
    y[~valid] = -1
    return y, valid


# -----------------------------
# gating L1 系数 schedule：warmup=0，然后线性 ramp 到 max
# -----------------------------
def gate_lambda(epoch: int, warmup_epochs: int, ramp_epochs: int, lam_max: float) -> float:
    if lam_max <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return lam_max
    t = epoch - warmup_epochs
    if t >= ramp_epochs:
        return lam_max
    return lam_max * (t / ramp_epochs)


# -----------------------------
# 计算 loss（MSE + CE）
# -----------------------------
def compute_losses(
    pred_ret: torch.Tensor,          # [B,K]
    mom_logits: torch.Tensor,        # [B,K,C]
    y_ret: torch.Tensor,             # [B,K]
    y_mom: torch.Tensor,             # [B,K]
    ret_mask: torch.Tensor,          # [B,K] float/0-1
    mom_mask: torch.Tensor,          # [B,K] float/0-1
):
    # 回归：masked MSE
    diff = (pred_ret - y_ret)
    mse = (diff * diff) * ret_mask
    ret_loss = mse.sum() / (ret_mask.sum() + 1e-12)

    # 分类：使用 ignore_index=-1 让 CE 自动忽略无效标签
    B, K, C = mom_logits.shape

    # 清洗标签：确保只有 [0, C-1] 和 -1
    y_norm, y_valid = sanitize_mom_labels(y_mom, C)

    logits_flat = mom_logits.reshape(B * K, C)
    y_flat = y_norm.reshape(B * K).long()

    # CE with ignore_index=-1：标签为 -1 的样本会被完全忽略（loss=0，不产生梯度）
    # reduction="mean" 会自动按有效样本数求均值（不包括 -1）
    ce_loss = F.cross_entropy(
        logits_flat,
        y_flat,
        reduction="mean",
        ignore_index=-1,
    )

    # 返回 y_valid 用于计算 Acc（只在有效标签上计算）
    return ret_loss, ce_loss, y_valid


# -----------------------------
# 评估
# -----------------------------
@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dl,
    device: torch.device,
    use_amp: bool,
    max_batches: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = True,
):
    model.eval()

    total_ret_loss = 0.0
    total_ce_loss = 0.0
    total_batches = 0

    total_ic = 0.0
    total_rankic = 0.0
    total_days = 0

    total_acc = 0.0
    total_acc_n = 0
    
    # 调试：统计 mask 和 label 分布
    debug_mom_mask_sum = 0.0
    debug_mom_mask_count = 0
    debug_y_mom_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    debug_valid_for_acc_count = 0
    debug_correct_count = 0

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda"))
    
    t_eval_start = time.time()

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break
        
        # 显示评估进度
        if show_progress and max_batches and (bi + 1) % max(1, max_batches // 10) == 0:
            progress_pct = 100.0 * (bi + 1) / max_batches
            print(f"      [EVAL Progress] {progress_pct:5.1f}% ({bi+1}/{max_batches} batches)", end="\r")

        X = batch["X"]
        y_ret = batch["y_ret"]
        y_mom = batch["y_mom"]
        ret_mask = batch["ret_mask"]
        mom_mask = batch["mom_mask"]

        X = X.to(device, non_blocking=True)
        y_ret = y_ret.to(device, non_blocking=True)
        y_mom = y_mom.to(device, non_blocking=True)
        ret_mask = ret_mask.to(device, non_blocking=True)
        mom_mask = mom_mask.to(device, non_blocking=True)

        B, K, L, D = X.shape
        x_flat = X.reshape(B * K, L, D)

        with amp_ctx:
            pred_ret_flat, mom_logits_flat = model(x_flat)
            pred_ret = pred_ret_flat.reshape(B, K)
            mom_logits = mom_logits_flat.reshape(B, K, -1)

            ret_loss, ce_loss, y_valid = compute_losses(pred_ret, mom_logits, y_ret, y_mom, ret_mask, mom_mask)

        total_ret_loss += float(ret_loss.item())
        total_ce_loss += float(ce_loss.item())
        total_batches += 1

        ic, ric, nd = batch_ic_rankic(pred_ret, y_ret, ret_mask)
        total_ic += ic
        total_rankic += ric
        total_days += nd

        # 分类准确率（诊断用）：只在合法标签上算
        pred_cls = mom_logits.argmax(dim=-1)  # [B,K]
        y_norm, _ = sanitize_mom_labels(y_mom, mom_logits.shape[-1])
        m = y_valid  # 直接使用 compute_losses 返回的 y_valid
        
        # 调试统计
        debug_mom_mask_sum += mom_mask.sum().item()
        debug_mom_mask_count += mom_mask.numel()
        y_mom_cpu = y_mom.cpu().numpy().flatten()
        for val in y_mom_cpu:
            if val in debug_y_mom_counts:
                debug_y_mom_counts[val] += 1
        debug_valid_for_acc_count += int(m.sum().item())
        
        if int(m.sum().item()) > 0:
            correct = (pred_cls[m] == y_norm[m]).sum().item()
            debug_correct_count += correct
            acc = correct / int(m.sum().item())
            total_acc += float(acc)
            total_acc_n += 1

        if verbose and bi == 0:
            # 只打印一次
            ym = y_mom.detach().cpu()
            print(f"    [EVAL DEBUG] first batch y_mom min={int(ym.min())} max={int(ym.max())} "
                  f"count(-1)={(ym==-1).sum().item()} / total={ym.numel()}")

    # 清除进度行
    if show_progress:
        print(" " * 80, end="\r")
    
    dt_eval = time.time() - t_eval_start
    
    # 调试输出
    if verbose or True:  # 总是打印调试信息
        mom_mask_mean = debug_mom_mask_sum / debug_mom_mask_count if debug_mom_mask_count > 0 else 0
        print(f"\n[EVAL DEBUG] mom_mask.mean() = {mom_mask_mean:.4f}")
        print(f"[EVAL DEBUG] y_mom distribution:")
        total_y = sum(debug_y_mom_counts.values())
        for val in sorted(debug_y_mom_counts.keys()):
            count = debug_y_mom_counts[val]
            pct = count / total_y * 100 if total_y > 0 else 0
            print(f"  y_mom={val:2d}: {count:8,} ({pct:5.2f}%)")
        print(f"[EVAL DEBUG] valid samples for Acc: {debug_valid_for_acc_count:,}")
        print(f"[EVAL DEBUG] correct predictions: {debug_correct_count:,}")
        if debug_valid_for_acc_count > 0:
            global_acc = debug_correct_count / debug_valid_for_acc_count
            print(f"[EVAL DEBUG] global Acc = {global_acc:.4f}")
        print(f"[EVAL DEBUG] Acc denominator (batches): {total_acc_n}")
    
    out = {
        "ret_loss": total_ret_loss / max(total_batches, 1),
        "ce_loss": total_ce_loss / max(total_batches, 1),
        "ic": total_ic / max(total_days, 1),
        "rankic": total_rankic / max(total_days, 1),
        "cls_acc": total_acc / max(total_acc_n, 1),
        "eval_time_s": dt_eval,
    }
    out["loss"] = out["ret_loss"] + out["ce_loss"]
    return out


# -----------------------------
# 训练
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    dl,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool,
    ret_w: float,
    ce_w: float,
    gate_lam: float,
    gate_reduction: str,
    grad_clip: float,
    grad_accum_steps: int,
    log_interval: int,
    show_gpu_mem: bool,
    max_batches: Optional[int] = None,
    debug_first_batch: bool = True,
    total_batches_estimate: Optional[int] = None,
):
    model.train()

    total_loss = 0.0
    total_ret = 0.0
    total_ce = 0.0
    total_gate = 0.0
    total_batches = 0

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda"))

    optimizer.zero_grad(set_to_none=True)

    t_epoch = time.time()
    t_last_log = t_epoch
    
    # 添加：详细计时统计
    total_data_time = 0.0
    total_gpu_time = 0.0
    t_batch_start = time.time()

    for bi, batch in enumerate(dl):
        # 数据加载时间
        t_data_end = time.time()
        data_time = t_data_end - t_batch_start
        total_data_time += data_time
        
        if max_batches is not None and bi >= max_batches:
            break

        # GPU计算开始
        t_gpu_start = time.time()
        t_step = time.time()

        X = batch["X"]
        y_ret = batch["y_ret"]
        y_mom = batch["y_mom"]
        ret_mask = batch["ret_mask"]
        mom_mask = batch["mom_mask"]

        # debug：前3批都打印数据统计（用于诊断NaN）
        if bi < 3:
            ym = y_mom.detach().cpu()
            yr = y_ret.detach().cpu()
            xc = X.detach().cpu()
            print(f"\n  [DEBUG Batch {bi}] X={tuple(X.shape)}")
            print(f"    y_mom: min={int(ym.min())} max={int(ym.max())} count(-1)={(ym==-1).sum().item()}")
            print(f"    y_ret: min={yr.min():.6f} max={yr.max():.6f} nan={torch.isnan(yr).sum().item()}")
            print(f"    X: min={xc.min():.6f} max={xc.max():.6f} nan={torch.isnan(xc).sum().item()}")
            print(f"    ret_mask sum={ret_mask.sum().item():.0f}, mom_mask sum={mom_mask.sum().item():.0f}")
            
            # 检查 row_present
            row_present = xc[:, :, :, -1]  # [B, K, L]，最后一个维度是 row_present
            print(f"    row_present: mean={row_present.mean():.3f}, zeros={((row_present==0).sum().item())}, ones={((row_present==1).sum().item())}")


        X = X.to(device, non_blocking=True)
        y_ret = y_ret.to(device, non_blocking=True)
        y_mom = y_mom.to(device, non_blocking=True)
        ret_mask = ret_mask.to(device, non_blocking=True)
        mom_mask = mom_mask.to(device, non_blocking=True)

        B, K, L, D = X.shape
        x_flat = X.reshape(B * K, L, D)

        with amp_ctx:
            pred_ret_flat, mom_logits_flat = model(x_flat)
            pred_ret = pred_ret_flat.reshape(B, K)
            mom_logits = mom_logits_flat.reshape(B, K, -1)

            ret_loss, ce_loss, _ = compute_losses(pred_ret, mom_logits, y_ret, y_mom, ret_mask, mom_mask)

            gate_loss = model.gate_l1(reduction=gate_reduction) if hasattr(model, "gate_l1") else torch.tensor(0.0, device=device)

            loss_full = ret_w * ret_loss + ce_w * ce_loss + gate_lam * gate_loss
            loss = loss_full / max(grad_accum_steps, 1)
        
        # debug：前3批打印 loss 详情
        if bi < 3:
            print(f"    ret_loss={ret_loss.item():.6f}, ce_loss={ce_loss.item():.6f}, gate_loss={gate_loss.item():.6f}")
            print(f"    loss_full={loss_full.item():.6f}, loss={loss.item():.6f}")
            pred_cpu = pred_ret.detach().cpu()
            print(f"    pred_ret: min={pred_cpu.min():.6f} max={pred_cpu.max():.6f} nan={torch.isnan(pred_cpu).sum().item()}\n")

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (bi + 1) % max(grad_accum_steps, 1) == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss_full.item())
        total_ret += float(ret_loss.item())
        total_ce += float(ce_loss.item())
        total_gate += float(gate_loss.item())
        total_batches += 1
        
        # GPU计算时间
        gpu_time = time.time() - t_gpu_start
        total_gpu_time += gpu_time
        
        # 准备下一个batch的计时
        t_batch_start = time.time()
        
        # GPU计算时间
        gpu_time = time.time() - t_gpu_start
        total_gpu_time += gpu_time

        # 进度日志（增强版：百分比、ETA、速度、数据/GPU时间）
        if log_interval > 0 and (bi + 1) % log_interval == 0:
            t_now = time.time()
            dt_since_last = t_now - t_last_log
            dt_total = t_now - t_epoch
            
            # 计算速度和ETA
            steps_per_sec = log_interval / dt_since_last if dt_since_last > 0 else 0
            
            # 计算数据/GPU时间比例
            avg_data_time = total_data_time / total_batches if total_batches > 0 else 0
            avg_gpu_time = total_gpu_time / total_batches if total_batches > 0 else 0
            total_time = avg_data_time + avg_gpu_time
            data_pct = (avg_data_time / total_time * 100) if total_time > 0 else 0
            gpu_pct = (avg_gpu_time / total_time * 100) if total_time > 0 else 0
            
            # 进度百分比和ETA
            progress_info = ""
            if total_batches_estimate and total_batches_estimate > 0:
                progress_pct = 100.0 * (bi + 1) / total_batches_estimate
                if total_batches > 0:
                    avg_time_per_batch = dt_total / total_batches
                    remaining_batches = total_batches_estimate - (bi + 1)
                    eta_sec = remaining_batches * avg_time_per_batch
                    eta_min = eta_sec / 60
                    progress_info = f"[{progress_pct:5.1f}%] ETA:{eta_min:5.1f}min "
            elif max_batches and max_batches > 0:
                progress_pct = 100.0 * (bi + 1) / max_batches
                progress_info = f"[{progress_pct:5.1f}%] "
            
            mem = gpu_mem_str(device) if show_gpu_mem else ""
            
            # 添加：显示数据加载vs GPU计算时间
            time_breakdown = f"⏱️data:{avg_data_time:.1f}s({data_pct:.0f}%) gpu:{avg_gpu_time:.1f}s({gpu_pct:.0f}%)"
            
            print(f"    [TRAIN] {progress_info}step={bi+1:6d} "
                  f"loss={total_loss/total_batches:.6f} "
                  f"(ret={total_ret/total_batches:.6f}, ce={total_ce/total_batches:.6f}, "
                  f"gate={total_gate/total_batches:.6f}) "
                  f"speed={steps_per_sec:.2f}step/s {time_breakdown} {mem}")
            
            t_last_log = t_now
        
        # 准备下一个batch的计时
        t_batch_start = time.time()

    dt_epoch = time.time() - t_epoch

    out = {
        "loss": total_loss / max(total_batches, 1),
        "ret_loss": total_ret / max(total_batches, 1),
        "ce_loss": total_ce / max(total_batches, 1),
        "gate_l1": total_gate / max(total_batches, 1),
        "epoch_time_s": dt_epoch,
    }
    return out


# -----------------------------
# 特征名 / exclude_regex（用于 dividend 固定 gate=0 + 导出 gate 排名）
# -----------------------------
def build_fixed_zero_idx(feature_names: List[str], exclude_regex: Optional[str]) -> List[int]:
    if not exclude_regex:
        return []
    pat = re.compile(exclude_regex)
    idxs = []
    for i, name in enumerate(feature_names):
        if pat.search(name) is not None:
            idxs.append(i)
    return idxs


def save_gate_report(save_path: str, feature_names: List[str], gate_values: np.ndarray):
    df = pd.DataFrame({"feature": feature_names, "gate": gate_values.astype(float)})
    df = df.sort_values("gate", ascending=False).reset_index(drop=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_py", type=str, default="../panel/2_dataset_memmap.py")
    ap.add_argument("--memmap_dir", type=str, required=True, help="memmap数据目录，例如: panel/memmap_data")
    ap.add_argument("--calendar_csv", type=str, required=True)

    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4, help="每批包含的天数，增大可提高GPU利用率")

    # 论文默认切分
    ap.add_argument("--train_years", type=str, default="2007-2014")
    ap.add_argument("--val_years", type=str, default="2015-2016")
    ap.add_argument("--test_years", type=str, default="2017-2020")

    # 训练超参（论文量级默认）
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)

    # 模型结构
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    # loss 权重
    ap.add_argument("--ret_w", type=float, default=1.0)
    ap.add_argument("--ce_w", type=float, default=1.0)

    # gating
    ap.add_argument("--use_gating", action="store_true")
    ap.add_argument("--exclude_regex", type=str, default=r"(?i)dividend")
    ap.add_argument("--gate_l1_max", type=float, default=5e-4)
    ap.add_argument("--gate_warmup_epochs", type=int, default=5)
    ap.add_argument("--gate_ramp_epochs", type=int, default=20)
    ap.add_argument("--gate_reduction", type=str, default="mean", choices=["mean", "sum"])

    # 工程/性能
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true", help="开启混合精度 AMP（GPU 推荐）")
    ap.add_argument("--num_workers", type=int, default=2, help="数据加载工作进程数，建议2-4（memmap已经很快）")
    ap.add_argument("--pin_memory", action="store_true", default=True, help="启用pin_memory加速GPU传输")
    ap.add_argument("--shuffle", action="store_true", help="是否打乱训练日期顺序")
    ap.add_argument("--persistent_workers", action="store_true", default=True, help="保持worker进程存活")
    ap.add_argument("--grad_accum_steps", type=int, default=1)

    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_eval_batches", type=int, default=0)

    ap.add_argument("--log_interval", type=int, default=200, help="每多少个 batch 打印一次训练进度")
    ap.add_argument("--show_gpu_mem", action="store_true", help="训练日志中显示 GPU 显存占用")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="每个worker预取的批次数")
    ap.add_argument("--eval_debug", action="store_true", help="eval 时打印第一批 y_mom 统计")

    ap.add_argument("--save_dir", type=str, default="runs/stage1_memmap")

    args = ap.parse_args()

    # Windows平台检测：动态导入模块无法pickle，必须禁用多进程
    import platform
    is_windows = platform.system() == "Windows"
    if is_windows and args.num_workers > 0:
        print(f"\n[WARNING] Windows平台检测到：动态导入模块无法在多进程中pickle")
        print(f"[WARNING] 将 num_workers 从 {args.num_workers} 调整为 0（单进程模式）")
        print(f"[INFO] memmap 数据加载已经很快，单进程足够高效\n")
        args.num_workers = 0
        args.persistent_workers = False  # num_workers=0时无效
        args.prefetch_factor = None

    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 推荐：固定 shape 时 cudnn.benchmark=True 会更快
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print_torch_env(device)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 加载 dataset 模块
    print(f"[STEP] loading dataset module from: {args.dataset_py}")
    ds_mod = load_module_from_path(args.dataset_py)
    if not hasattr(ds_mod, "build_dataloader"):
        raise RuntimeError("2_dataset_day_grouped.py 中找不到 build_dataloader 函数")

    # 切分 date_idx
    print(f"[STEP] loading calendar from: {args.calendar_csv}")
    dates = load_calendar_dates(args.calendar_csv)

    def parse_years(s: str) -> Tuple[int, int]:
        a, b = s.split("-")
        return int(a), int(b)

    tr_y0, tr_y1 = parse_years(args.train_years)
    va_y0, va_y1 = parse_years(args.val_years)
    te_y0, te_y1 = parse_years(args.test_years)

    tr_s, tr_e = year_range_to_idx(dates, tr_y0, tr_y1)
    va_s, va_e = year_range_to_idx(dates, va_y0, va_y1)
    te_s, te_e = year_range_to_idx(dates, te_y0, te_y1)

    min_start = args.lookback - 1
    tr_s = max(tr_s, min_start)
    va_s = max(va_s, min_start)
    te_s = max(te_s, min_start)

    print(f"[SPLIT] train date_idx: {tr_s}..{tr_e}  years={args.train_years}")
    print(f"[SPLIT] val   date_idx: {va_s}..{va_e}  years={args.val_years}")
    print(f"[SPLIT] test  date_idx: {te_s}..{te_e}  years={args.test_years}")

    # 从 memmap 元数据读取特征名
    print(f"[STEP] loading feature columns from memmap metadata: {args.memmap_dir}")
    meta_path = os.path.join(args.memmap_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        memmap_meta = json.load(f)
    
    feature_names = memmap_meta.get("feat_cols", [])
    T_total = memmap_meta["T"]
    N_total = memmap_meta["N"]
    D_total = memmap_meta["D"]
    
    print(f"[INFO] Memmap shape: T={T_total}, N={N_total}, D={D_total}")
    print(f"[INFO] Feature count: {len(feature_names)}")

    fixed_zero_idx = []
    if args.use_gating and feature_names:
        fixed_zero_idx = build_fixed_zero_idx(feature_names, args.exclude_regex)
        print(f"[GATING] exclude_regex={args.exclude_regex} 命中特征数={len(fixed_zero_idx)}（这些特征 gate 固定为 0）")

    # dataloader 构造（使用 memmap dataset）
    def make_loader(day_start: int, day_end: int, shuffle: bool):
        t0 = time.time()
        # 根据是否使用GPU动态设置pin_memory
        use_pin_memory = args.pin_memory and device.type == "cuda"
        
        # 创建 dataset
        if not hasattr(ds_mod, "MemmapDayWindowDataset"):
            raise RuntimeError("dataset module 中找不到 MemmapDayWindowDataset 类")
        
        dataset = ds_mod.MemmapDayWindowDataset(
            memmap_dir=args.memmap_dir,
            lookback=args.lookback,
            day_start=day_start,
            day_end=day_end,
            k=args.k,
            seed=args.seed,
            sample_both_valid_only=False,  # 不使用 both_valid
            sample_ret_valid_only=False,
            sample_present_only=True,      # ✓ 仅用当日row_present采样，不依赖t+1
        )
        
        # 创建 dataloader
        if not hasattr(ds_mod, "build_dataloader"):
            raise RuntimeError("dataset module 中找不到 build_dataloader 函数")
        
        dl = ds_mod.build_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=shuffle,
            pin_memory=use_pin_memory,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        dt = time.time() - t0
        print(f"[STEP] build_dataloader(day={day_start}..{day_end}, shuffle={shuffle}, workers={args.num_workers}, pin_mem={use_pin_memory}) done in {dt:.2f}s")
        return dl

    # 先拿一个 batch 推断 D/F
    print("[STEP] fetching first batch to infer dims...")
    tmp_dl = make_loader(tr_s, tr_e, shuffle=False)
    first_batch = next(iter(tmp_dl))
    X0 = first_batch["X"]
    _, _, _, D = X0.shape
    raw_F = (D - 1) // 2
    print(f"[INFO] inferred input_dim D={D}, raw_feature_dim F={raw_F}  (expect D=2F+1 -> {2*raw_F+1})")
    if 2 * raw_F + 1 != D:
        raise RuntimeError(f"维度不满足 2F+1：D={D}, F={raw_F}")

    # feature_names 对齐
    if not feature_names:
        feature_names = [f"f{i}" for i in range(raw_F)]
    else:
        if len(feature_names) != raw_F:
            print(f"[WARN] feature_names 长度={len(feature_names)} 与 raw_F={raw_F} 不一致，将截断/补齐对齐。")
            if len(feature_names) > raw_F:
                feature_names = feature_names[:raw_F]
            else:
                feature_names = feature_names + [f"f{i}" for i in range(len(feature_names), raw_F)]

    # 模型
    cfg = LSTMMTLConfig(
        input_dim=D,
        raw_feature_dim=raw_F,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=5,
        use_layernorm=True,
        use_gating=args.use_gating,
        gating_init_logit=2.0,
    )

    print("[STEP] building model...")
    model = LSTMMultiTask(cfg, fixed_zero_idx=fixed_zero_idx if args.use_gating else None).to(device)
    print(f"[GPU CHECK] model parameters on cuda = {next(model.parameters()).is_cuda if any(True for _ in model.parameters()) else False}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    print(f"[AMP] enabled = {use_amp}")

    # 保存配置
    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"args": vars(args), "model_cfg": asdict(cfg), "device": str(device)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    log_path = os.path.join(args.save_dir, "log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_ret,train_ce,gate_lam,train_gate,val_loss,val_ret,val_ce,val_ic,val_rankic,val_acc,epoch_time_s\n")

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    # ----------------- training loop -----------------
    # 估算每个epoch的总批次数（用于进度显示）
    total_train_days = tr_e - tr_s + 1
    estimated_batches = (total_train_days + args.batch_size - 1) // args.batch_size
    if args.max_train_batches > 0:
        estimated_batches = min(estimated_batches, args.max_train_batches)
    
    print(f"\n[INFO] 训练数据: {total_train_days}天, 预计~{estimated_batches}批/epoch (batch_size={args.batch_size})")
    print(f"[INFO] 验证数据: {va_e-va_s+1}天")
    print(f"[INFO] 每批: K={args.k}股票 × L={args.lookback}天 × B={args.batch_size}天")
    print(f"[INFO] 数据加载: num_workers={args.num_workers}, pin_memory={args.pin_memory and device.type=='cuda'}, persistent_workers={args.persistent_workers}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"  EPOCH {epoch}/{args.epochs}  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"{'='*70}")
        print(f"[STEP] building train/val dataloaders ...")
        train_dl = make_loader(tr_s, tr_e, shuffle=args.shuffle)
        val_dl = make_loader(va_s, va_e, shuffle=False)

        lam = gate_lambda(epoch, args.gate_warmup_epochs, args.gate_ramp_epochs, args.gate_l1_max) if args.use_gating else 0.0
        print(f"[GATING] gate_lam(epoch={epoch}) = {lam:.2e}  (warmup={args.gate_warmup_epochs}, ramp={args.gate_ramp_epochs}, max={args.gate_l1_max})")
        if args.show_gpu_mem:
            print(f"[GPU MEM] {gpu_mem_str(device)}")
        
        print(f"\n>>> 开始训练 (预计{estimated_batches}批) <<<")
        tr_stats = train_one_epoch(
            model=model,
            dl=train_dl,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            device=device,
            use_amp=use_amp,
            ret_w=args.ret_w,
            ce_w=args.ce_w,
            gate_lam=lam,
            gate_reduction=args.gate_reduction,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            log_interval=args.log_interval,
            show_gpu_mem=args.show_gpu_mem,
            max_batches=(args.max_train_batches if args.max_train_batches > 0 else None),
            debug_first_batch=True,
            total_batches_estimate=estimated_batches,
        )
        print(f"✓ 训练完成: {tr_stats['epoch_time_s']:.1f}秒\n")

        print(f">>> 开始验证 <<<")
        va_stats = eval_one_epoch(
            model=model,
            dl=val_dl,
            device=device,
            use_amp=use_amp,
            max_batches=(args.max_eval_batches if args.max_eval_batches > 0 else None),
            verbose=args.eval_debug,
            show_progress=True,
        )
        print(f"✓ 验证完成: {va_stats.get('eval_time_s', 0):.1f}秒\n")

        epoch_total_time = time.time() - epoch_start_time
        print(f"{'-'*70}")
        print(f"[EPOCH {epoch:03d} 总结]")
        print(f"  训练 - loss:{tr_stats['loss']:.6f} ret:{tr_stats['ret_loss']:.6f} ce:{tr_stats['ce_loss']:.6f} gate:{tr_stats['gate_l1']:.6f}")
        print(f"  验证 - loss:{va_stats['loss']:.6f} ret:{va_stats['ret_loss']:.6f} ce:{va_stats['ce_loss']:.6f}")
        print(f"  指标 - IC:{va_stats['ic']:.4f} RankIC:{va_stats['rankic']:.4f} Acc:{va_stats['cls_acc']:.4f}")
        print(f"  时间 - 训练:{tr_stats['epoch_time_s']:.1f}s 验证:{va_stats.get('eval_time_s', 0):.1f}s 总计:{epoch_total_time:.1f}s")
        if args.show_gpu_mem:
            print(f"  GPU  - {gpu_mem_str(device)}")
        print(f"{'-'*70}")

        # 每个 epoch 打印 gate top/bot（用于筛特征）
        if args.use_gating:
            gv = model.gate_values()
            if gv is not None:
                gv_np = gv.numpy()
                top_idx = np.argsort(-gv_np)[:10]
                bot_idx = np.argsort(gv_np)[:10]
                print("  [GATE top10] " + ", ".join([f"{feature_names[i]}={gv_np[i]:.3f}" for i in top_idx]))
                print("  [GATE bot10] " + ", ".join([f"{feature_names[i]}={gv_np[i]:.3f}" for i in bot_idx]))

        # 写 log.csv
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{tr_stats['loss']:.8f},{tr_stats['ret_loss']:.8f},{tr_stats['ce_loss']:.8f},{lam:.6e},{tr_stats['gate_l1']:.8f},"
                f"{va_stats['loss']:.8f},{va_stats['ret_loss']:.8f},{va_stats['ce_loss']:.8f},{va_stats['ic']:.8f},{va_stats['rankic']:.8f},{va_stats['cls_acc']:.8f},"
                f"{tr_stats['epoch_time_s']:.2f}\n"
            )

        # early stopping（按 val_loss）
        if va_stats["loss"] < best_val - 1e-9:
            best_val = va_stats["loss"]
            best_epoch = epoch
            bad_epochs = 0

            ckpt_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_val,
                    "cfg": asdict(cfg),
                    "feature_names": feature_names,
                },
                ckpt_path,
            )
            print(f"\n  ⭐ [NEW BEST] Epoch {epoch}: val_loss={best_val:.6f} -> saved to {os.path.basename(ckpt_path)}")
        else:
            bad_epochs += 1
            print(f"\n  ⏸️  [PATIENCE] {bad_epochs}/{args.patience} (best: epoch {best_epoch}, val_loss={best_val:.6f})")
            if bad_epochs >= args.patience:
                print(f"\n{'='*70}")
                print(f"  🛑 EARLY STOPPING")
                print(f"  Best Epoch: {best_epoch}")
                print(f"  Best Val Loss: {best_val:.6f}")
                print(f"{'='*70}\n")
                break

    # ----------------- test -----------------
    print(f"\n{'='*70}")
    print(f"  测试集评估")
    print(f"{'='*70}")
    
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        print(f"\n[STEP] 加载最佳模型: {os.path.basename(best_path)}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"       最佳 epoch={ckpt.get('epoch')}, val_loss={ckpt.get('best_val'):.6f}")

    print(f"\n[STEP] 构建测试数据集 ({te_s}..{te_e}: {te_e-te_s+1}天) ...")
    test_dl = make_loader(te_s, te_e, shuffle=False)

    print("[STEP] 开始测试集评估 ...\n")
    te_stats = eval_one_epoch(
        model=model,
        dl=test_dl,
        device=device,
        use_amp=use_amp,
        max_batches=(args.max_eval_batches if args.max_eval_batches > 0 else None),
        verbose=args.eval_debug,
        show_progress=True,
    )
    
    print(f"\n{'-'*70}")
    print(f"[测试集结果]")
    print(f"  Loss: {te_stats['loss']:.6f} (ret={te_stats['ret_loss']:.6f}, ce={te_stats['ce_loss']:.6f})")
    print(f"  IC: {te_stats['ic']:.4f}")
    print(f"  RankIC: {te_stats['rankic']:.4f}")
    print(f"  Cls Acc: {te_stats['cls_acc']:.4f}")
    print(f"  Time: {te_stats.get('eval_time_s', 0):.1f}s")
    print(f"{'-'*70}\n")

    # 导出 gating 排名
    if args.use_gating:
        gv = model.gate_values()
        if gv is not None:
            out_csv = os.path.join(args.save_dir, "feature_gates.csv")
            save_gate_report(out_csv, feature_names, gv.numpy())
            print(f"[GATE REPORT] 特征门控值已保存: {os.path.basename(out_csv)}")
    
    print(f"\n✅ 训练完成！结果保存在: {args.save_dir}\n")


if __name__ == "__main__":
    main()
