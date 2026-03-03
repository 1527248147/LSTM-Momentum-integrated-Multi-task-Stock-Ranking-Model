# -*- coding: utf-8 -*-
"""
固定时间窗口的 Refit 滚动测试

每年用「固定长度 8 年」的窗口训练，再在当年测试集上评估：
- 测 2017：训练 2009–2016（8 年），测试 2017
- 测 2018：训练 2010–2017，测试 2018
- 测 2019：训练 2011–2018，测试 2019
- 测 2020：训练 2012–2019，测试 2020

每年评估完成后会调用 comprehensive_report 生成完整报告，写入：
  save_dir/refit_YYYY/comprehensive_report.json

用法：
  python run_refit_rolling_window_test.py --dataset_py ..\\panel\\2_dataset_memmap.py --memmap_dir ..\\panel\\memmap_data --save_dir runs/refit_rolling --fixed_epochs 50
  python run_refit_rolling_window_test.py --save_dir runs/refit_rolling --skip_train   # 仅做汇总评估 + 生成报告（已有各年 checkpoint）
"""

import os
import sys
import re
import json
import argparse
import subprocess
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import torch

# 与 4_train_stage2 / generate_report 一致
sys.path.insert(0, os.path.dirname(__file__))
from model_lstm_mtl import LSTMMultiTask, LSTMMTLConfig
from comprehensive_report import generate_comprehensive_report
from generate_report_from_checkpoint import run_evaluation


def _empty_metrics():
    """与 comprehensive_report 期望的 metrics 结构一致（无验证集时用）."""
    return {
        "loss": 0.0, "reg": 0.0, "ce": 0.0, "ndcgL": 0.0, "gate": 0.0,
        "ic": 0.0, "ric": 0.0, "ic_std": 0.0, "ric_std": 0.0,
        "icir": 0.0, "rankicir": 0.0, "cls_acc": 0.0,
        "ndcg": 0.0, "k": 0, "tau": 0,
        "acc_class0": 0.0, "acc_class1": 0.0, "acc_class2": 0.0,
        "acc_class3": 0.0, "acc_class4": 0.0,
    }


# -----------------------------------------------------------------------------
# 常量：固定窗口与测试年
# -----------------------------------------------------------------------------
WINDOW_YEARS = 8
TEST_YEARS_DEFAULT = [2017, 2018, 2019, 2020]


def load_calendar(memmap_dir: str) -> pd.DataFrame:
    """加载日历（与 4_train_stage2 一致：优先 meta/calendar.csv，其次 meta.json）。"""
    memmap_path = Path(memmap_dir)
    # 1) meta/calendar.csv (seq_meta 结构)
    for base in [memmap_path, memmap_path.parent]:
        cal_path = base / "meta" / "calendar.csv"
        if cal_path.exists():
            cal_df = pd.read_csv(cal_path)
            if "date_str" in cal_df.columns:
                cal_df["year"] = pd.to_datetime(cal_df["date_str"]).dt.year
            return cal_df
    # 2) meta.json (panel 结构)
    meta_path = memmap_path / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cal_df = pd.DataFrame({"date_str": meta.get("dates", [])})
        cal_df["year"] = pd.to_datetime(cal_df["date_str"]).dt.year
        return cal_df
    raise FileNotFoundError(
        f"未找到日历：{memmap_path}/meta/calendar.csv 或 {memmap_path}/meta.json"
    )


def get_day_range(cal_df: pd.DataFrame, years: List[int], lookback: int):
    """返回这些年份对应的 day_start, day_end（与 4_train_stage2 一致）。"""
    indices = cal_df[cal_df["year"].isin(years)].index.tolist()
    if not indices:
        return None, None
    start = max(indices[0], lookback - 1)
    end = indices[-1]
    return start, end


def parse_year_range(s: str) -> List[int]:
    """解析 '2017-2020' 或 '2017,2018,2019,2020'。"""
    s = (s or "").strip()
    if not s or s.lower() == "none":
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a.strip()), int(b.strip()) + 1))
    return [int(x.strip()) for x in s.split(",")]


def run_refit_for_year(
    test_year: int,
    base_save_dir: str,
    train_years: List[int],
    extra_argv: List[str],
    script_path: str,
) -> bool:
    """对单个测试年跑一次 4_train_stage2 refit（子进程）。"""
    train_str = f"{min(train_years)}-{max(train_years)}"
    save_dir = os.path.join(base_save_dir, f"refit_{test_year}")
    cmd = [
        sys.executable,
        script_path,
        "--train_years", train_str,
        "--val_years", "none",
        "--test_years", str(test_year),
        "--save_dir", save_dir,
    ] + extra_argv
    print(f"\n{'='*80}")
    print(f"Refit 测试年 {test_year}: 训练 {train_str} -> 测试 {test_year} | save_dir={save_dir}")
    print(f"{'='*80}\n")
    ret = subprocess.run(cmd, cwd=os.path.dirname(script_path))
    return ret.returncode == 0


def load_dataset_module(dataset_py: str):
    spec = importlib.util.spec_from_file_location("ds_mod", dataset_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_feature_indices(memmap_dir: str, exclude_features: str, include_features: str) -> Optional[List[int]]:
    """与 4_train_stage2 一致：根据 exclude/include 得到 feature_indices。"""
    meta_path = Path(memmap_dir) / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    all_features = meta.get("feat_cols", [])
    D_original = meta.get("D", 0)
    F_original = len(all_features)
    if not all_features:
        return None

    if include_features:
        pattern = re.compile(include_features)
        selected = [f for f in all_features if pattern.match(f)]
    elif exclude_features:
        pattern = re.compile(exclude_features)
        selected = [f for f in all_features if not pattern.match(f)]
    else:
        return None

    indices = []
    for i, f in enumerate(all_features):
        if f in selected:
            indices.append(i)
            indices.append(F_original + i)
    indices.append(2 * F_original)
    return indices


def compute_ic_rankic_batch(pred: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """(B,K) -> ic_mean, ric_mean, ic_std, ric_std, icir, rankicir"""
    B, K = pred.shape
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
    ic_std = float(np.std(ics)) if len(ics) > 1 else 0.0
    ric_std = float(np.std(rics)) if len(rics) > 1 else 0.0
    icir = ic_mean / (ic_std + 1e-8)
    rankicir = ric_mean / (ric_std + 1e-8)
    return ic_mean, ric_mean, ic_std, ric_std, icir, rankicir


def make_dataloader_for_years(
    memmap_dir: str,
    dataset_py: str,
    lookback: int,
    k: int,
    batch_size: int,
    years: List[int],
    feature_indices: Optional[List[int]],
):
    """为指定年份区间构建 DataLoader（用于 train 或 test）。"""
    cal_df = load_calendar(memmap_dir)
    day_start, day_end = get_day_range(cal_df, years, lookback)
    if day_start is None:
        return None
    ds_mod = load_dataset_module(dataset_py)
    dataset = ds_mod.MemmapDayWindowDataset(
        memmap_dir=memmap_dir,
        lookback=lookback,
        day_start=day_start,
        day_end=day_end,
        k=k,
        seed=42,
        sample_both_valid_only=False,
        sample_ret_valid_only=False,
        sample_present_only=True,
        feature_indices=feature_indices,
    )
    return ds_mod.build_dataloader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=False
    )


def run_eval_one_year(
    ckpt_path: str,
    memmap_dir: str,
    dataset_py: str,
    lookback: int,
    k: int,
    batch_size: int,
    test_year: int,
    device: torch.device,
    use_gating: bool,
    feature_indices: Optional[List[int]],
    embed_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> Dict[str, float]:
    """加载该年 checkpoint，在当年测试集上跑一次评估，返回指标字典。"""
    cal_df = load_calendar(memmap_dir)
    te_start, te_end = get_day_range(cal_df, [test_year], lookback)
    if te_start is None:
        return {"ic": 0.0, "ric": 0.0, "icir": 0.0, "rankicir": 0.0, "cls_acc": 0.0}

    ds_mod = load_dataset_module(dataset_py)
    dataset = ds_mod.MemmapDayWindowDataset(
        memmap_dir=memmap_dir,
        lookback=lookback,
        day_start=te_start,
        day_end=te_end,
        k=k,
        seed=42,
        sample_both_valid_only=False,
        sample_ret_valid_only=False,
        sample_present_only=True,
        feature_indices=feature_indices,
    )
    dl = ds_mod.build_dataloader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=False
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    first_batch = next(iter(dl))
    D = first_batch["X"].shape[-1]
    raw_F = (D - 1) // 2

    cfg = LSTMMTLConfig(
        input_dim=D,
        raw_feature_dim=raw_F,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=5,
        use_layernorm=True,
        use_gating=use_gating,
        gating_init_logit=2.0,
    )
    model = LSTMMultiTask(cfg, fixed_zero_idx=None)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    all_pred, all_y, all_mask = [], [], []
    all_mom_logits, all_y_mom = [], []
    with torch.inference_mode():
        for batch in dl:
            X = batch["X"].to(device)
            y_ret = batch["y_ret"].to(device)
            y_mom = batch["y_mom"].to(device)
            ret_mask = batch["ret_mask"].to(device)
            B, K, L, D_in = X.shape
            x_flat = X.reshape(B * K, L, D_in)
            pred_ret_flat, mom_logits_flat = model(x_flat)
            pred_ret = pred_ret_flat.reshape(B, K)
            mom_logits = mom_logits_flat.reshape(B, K, -1)
            all_pred.append(pred_ret.cpu().numpy())
            all_y.append(y_ret.cpu().numpy())
            all_mask.append(ret_mask.cpu().numpy())
            all_mom_logits.append(mom_logits.cpu().numpy())
            all_y_mom.append(y_mom.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    y = np.concatenate(all_y, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    ic_mean, ric_mean, ic_std, ric_std, icir, rankicir = compute_ic_rankic_batch(pred, y, mask)

    mom_logits = np.concatenate(all_mom_logits, axis=0)
    all_y_mom = np.concatenate(all_y_mom, axis=0)
    mom_pred = mom_logits.argmax(axis=-1)
    valid = (all_y_mom >= 0)
    cls_acc = (mom_pred[valid] == all_y_mom[valid]).mean() if valid.sum() > 0 else 0.0

    return {
        "test_year": test_year,
        "ic": ic_mean,
        "ric": ric_mean,
        "ic_std": ic_std,
        "ric_std": ric_std,
        "icir": icir,
        "rankicir": rankicir,
        "cls_acc": cls_acc,
    }


def run_refit_eval_and_report(
    test_year: int,
    train_years: List[int],
    ckpt_path: str,
    save_dir: str,
    memmap_dir: str,
    dataset_py: str,
    lookback: int,
    k: int,
    batch_size: int,
    device: torch.device,
    feature_indices: Optional[List[int]],
    args,
) -> Dict[str, Any]:
    """
    加载该年 checkpoint，在训练窗口与测试年上评估，并用 comprehensive_report 生成完整报告。
    返回 te 指标（用于汇总 CSV），并写入 save_dir/comprehensive_report.json。
    """
    cal_df = load_calendar(memmap_dir)
    ds_mod = load_dataset_module(dataset_py)
    dl_train = make_dataloader_for_years(
        memmap_dir, dataset_py, lookback, k, batch_size, train_years, feature_indices
    )
    dl_test = make_dataloader_for_years(
        memmap_dir, dataset_py, lookback, k, batch_size, [test_year], feature_indices
    )
    if dl_train is None or dl_test is None:
        return {"test_year": test_year, "ic": 0.0, "ric": 0.0, "icir": 0.0, "rankicir": 0.0, "cls_acc": 0.0}

    ckpt = torch.load(ckpt_path, map_location="cpu")
    first_batch = next(iter(dl_test))
    D = first_batch["X"].shape[-1]
    raw_F = (D - 1) // 2
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
    model = LSTMMultiTask(cfg, fixed_zero_idx=None)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tr = run_evaluation(model, dl_train, device, use_gating=args.use_gating)
    te = run_evaluation(model, dl_test, device, use_gating=args.use_gating)
    va = _empty_metrics()

    report_args = SimpleNamespace(
        save_dir=save_dir,
        memmap_dir=memmap_dir,
        lookback=lookback,
        k=k,
        batch_size=batch_size,
        seed=args.seed,
        _train_years_actual=train_years,
        _val_years_actual=[],
        _test_years_actual=[test_year],
        use_cqb=args.use_cqb,
        use_gating=args.use_gating,
        beta0=args.beta0,
        b_win=args.b_win,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        ret_w=args.ret_w,
        cls_w=args.cls_w,
        lambda_ce=args.lambda_ce,
        use_lambdarank=getattr(args, "use_lambdarank", False),
        lambdarank_k=getattr(args, "lambdarank_k", 50),
        lambdarank_sigma=getattr(args, "lambdarank_sigma", 1.0),
        lambdarank_bins=getattr(args, "lambdarank_bins", 5),
    )
    epoch = int(ckpt.get("epoch", 0))
    best_rankic = float(ckpt.get("best_rankic", te["ric"]))

    report = generate_comprehensive_report(
        args=report_args,
        model=model,
        tr=tr,
        va=va,
        te=te,
        epoch=epoch,
        best_rankic=best_rankic,
        ckpt_path=ckpt_path,
        device=device,
        dl_train=dl_train,
        dl_val=None,
        dl_test=dl_test,
        cqb=None,
    )
    report_path = os.path.join(save_dir, "comprehensive_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  -> 报告已保存: {report_path}")

    return {**te, "test_year": test_year}


def main():
    ap = argparse.ArgumentParser(description="固定时间窗口 Refit 滚动测试（每年 8 年窗口训练 + 当年测试）")
    # 滚动窗口与测试年
    ap.add_argument("--window_years", type=int, default=WINDOW_YEARS, help="训练窗口长度（年）")
    ap.add_argument("--test_years", type=str, default="2017-2020", help="测试年范围，如 2017-2020 或 2017,2018,2019,2020")
    ap.add_argument("--skip_train", action="store_true", help="跳过训练，仅用已有 refit_YYYY 下 best.pt 做汇总评估")

    # 与 4_train_stage2 一致的路径与数据
    ap.add_argument("--dataset_py", type=str, default="", help="与 4_train_stage2 一致")
    ap.add_argument("--memmap_dir", type=str, default="", help="与 4_train_stage2 一致")
    ap.add_argument("--save_dir", type=str, default="runs/refit_rolling", help="根目录，各年存到 save_dir/refit_YYYY/")
    ap.add_argument("--fixed_epochs", type=int, default=0,
                    help="Refit 时每轮训练的 epoch 数。0 = 自动从 --reference_ckpt 读取最佳 epoch")
    ap.add_argument("--reference_ckpt", type=str, default="",
                    help="参考 checkpoint 路径（如 runs/stage2_lambdarank_alpha158/best.pt），"
                         "用于自动获取最佳 epoch 数。若设置且 fixed_epochs=0，则从中读取 epoch 作为 fixed_epochs")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    # 特征 / 模型（与 4_train_stage2 一致，用于子进程与本地 eval）
    ap.add_argument("--exclude_features", type=str, default="")
    ap.add_argument("--exclude_features_file", type=str, default="")
    ap.add_argument("--include_features", type=str, default="")
    ap.add_argument("--use_gating", action="store_true")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)

    # 其余 4_train_stage2 参数（传给子进程，不在此列出全部，用额外列表）
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--use_cqb", action="store_true", default=True)
    ap.add_argument("--beta0", type=float, default=0.5)
    ap.add_argument("--b_win", type=int, default=6)
    ap.add_argument("--ret_w", type=float, default=1.0)
    ap.add_argument("--cls_w", type=float, default=1.0)
    ap.add_argument("--lambda_ce", type=float, default=0.5)
    ap.add_argument("--tau_ratio", type=float, default=0.2)
    ap.add_argument("--temp_rank", type=float, default=1.0)
    ap.add_argument("--temp_topk", type=float, default=1.0)
    ap.add_argument("--gate_warmup_epochs", type=int, default=5)
    ap.add_argument("--gate_ramp_epochs", type=int, default=20)
    ap.add_argument("--gate_l1_max", type=float, default=5e-3)
    ap.add_argument("--gate_reduction", type=str, default="mean")
    ap.add_argument("--use_lambdarank", action="store_true")
    ap.add_argument("--lambdarank_k", type=int, default=50)
    ap.add_argument("--lambdarank_sigma", type=float, default=0.5)
    ap.add_argument("--lambdarank_bins", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()

    # ---- 自动从参考 checkpoint 获取最佳 epoch ----
    if args.fixed_epochs <= 0:
        if not args.reference_ckpt:
            raise ValueError(
                "--fixed_epochs=0 时必须指定 --reference_ckpt（已有正常训练的 best.pt），"
                "用于自动读取最佳 epoch 数"
            )
        ref_path = args.reference_ckpt
        best_epoch = None

        # 方式1：从 checkpoint 文件读取 epoch 字段
        if ref_path.endswith(".pt") and os.path.isfile(ref_path):
            ckpt = torch.load(ref_path, map_location="cpu")
            best_epoch = int(ckpt.get("epoch", 0))
            print(f"从 checkpoint 读取最佳 epoch: {best_epoch}  ({ref_path})")

        # 方式2：若 checkpoint 同目录有 comprehensive_report.json，也可以验证
        if best_epoch is None or best_epoch <= 0:
            report_path = os.path.join(os.path.dirname(ref_path), "comprehensive_report.json")
            if os.path.isfile(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    rpt = json.load(f)
                best_epoch = int(rpt.get("training_summary", {}).get("total_epochs", 0))
                print(f"从 comprehensive_report.json 读取最佳 epoch: {best_epoch}")

        if not best_epoch or best_epoch <= 0:
            raise ValueError(f"无法从 {ref_path} 中获取有效的 epoch 数（读到 {best_epoch}）")
        args.fixed_epochs = best_epoch
        print(f"Refit 将使用 fixed_epochs = {args.fixed_epochs}（来自参考训练的最佳 epoch）\n")

    test_years = parse_year_range(args.test_years)
    if not test_years:
        test_years = TEST_YEARS_DEFAULT
    window_years = max(1, args.window_years)
    base_save_dir = args.save_dir
    os.makedirs(base_save_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "4_train_stage2.py")
    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"未找到训练脚本: {train_script}")

    # 构建传给 4_train_stage2 的额外参数（与 resume_stage2_stable.bat 等对齐）
    extra_argv = [
        "--dataset_py", args.dataset_py or os.path.join(script_dir, "..", "panel", "2_dataset_memmap.py"),
        "--memmap_dir", args.memmap_dir or os.path.join(script_dir, "..", "panel", "memmap_data"),
        "--lookback", str(args.lookback),
        "--k", str(args.k),
        "--batch_size", str(args.batch_size),
        "--fixed_epochs", str(args.fixed_epochs),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--grad_clip", str(args.grad_clip),
        *(["--use_cqb"] if args.use_cqb else []),
        "--beta0", str(args.beta0),
        "--b_win", str(args.b_win),
        "--ret_w", str(args.ret_w),
        "--cls_w", str(args.cls_w),
        "--lambda_ce", str(args.lambda_ce),
        "--tau_ratio", str(args.tau_ratio),
        "--temp_rank", str(args.temp_rank),
        "--temp_topk", str(args.temp_topk),
        "--embed_dim", str(args.embed_dim),
        "--hidden_size", str(args.hidden_size),
        "--num_layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--num_workers", str(args.num_workers),
    ]
    if args.use_gating:
        extra_argv += [
            "--use_gating",
            "--gate_warmup_epochs", str(args.gate_warmup_epochs),
            "--gate_ramp_epochs", str(args.gate_ramp_epochs),
            "--gate_l1_max", str(args.gate_l1_max),
            "--gate_reduction", args.gate_reduction,
        ]
    if args.use_lambdarank:
        extra_argv += [
            "--use_lambdarank",
            "--lambdarank_k", str(args.lambdarank_k),
            "--lambdarank_sigma", str(args.lambdarank_sigma),
            "--lambdarank_bins", str(args.lambdarank_bins),
        ]
    if args.exclude_features:
        extra_argv += ["--exclude_features", args.exclude_features]
    if args.exclude_features_file:
        extra_argv += ["--exclude_features_file", args.exclude_features_file]

    # 1) 若不跳过训练：按年 refit
    if not args.skip_train:
        if not args.dataset_py or not args.memmap_dir:
            # 若未传，用上面默认路径（可能相对路径）
            pass
        for test_year in test_years:
            train_years = list(range(test_year - window_years, test_year))
            ok = run_refit_for_year(
                test_year=test_year,
                base_save_dir=base_save_dir,
                train_years=train_years,
                extra_argv=extra_argv,
                script_path=train_script,
            )
            if not ok:
                print(f"Warning: refit for test year {test_year} exited with non-zero code.")
    else:
        print("Skip training (--skip_train). Will only run aggregate evaluation.")

    # 2) 汇总评估：对每个 test_year 用 refit_YYYY/best.pt 在当年测试集上评估
    memmap_dir = args.memmap_dir or os.path.join(script_dir, "..", "panel", "memmap_data")
    dataset_py = args.dataset_py or os.path.join(script_dir, "..", "panel", "2_dataset_memmap.py")
    if not os.path.isabs(memmap_dir):
        memmap_dir = os.path.normpath(os.path.abspath(os.path.join(script_dir, memmap_dir)))
    if not os.path.isabs(dataset_py):
        dataset_py = os.path.normpath(os.path.abspath(os.path.join(script_dir, dataset_py)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.exclude_features_file and not args.exclude_features:
        try:
            with open(args.exclude_features_file, "r", encoding="utf-8") as f:
                args.exclude_features = f.read().strip()
        except Exception:
            pass
    feature_indices = get_feature_indices(memmap_dir, args.exclude_features or "", getattr(args, "include_features", "") or "")

    results = []
    window_years = max(1, args.window_years)
    for test_year in test_years:
        ckpt_path = os.path.join(base_save_dir, f"refit_{test_year}", "best.pt")
        save_dir_year = os.path.join(base_save_dir, f"refit_{test_year}")
        if not os.path.isfile(ckpt_path):
            print(f"Skip eval {test_year}: no checkpoint at {ckpt_path}")
            continue
        train_years = list(range(test_year - window_years, test_year))
        print(f"\n[Refit {test_year}] 评估 + 生成 comprehensive_report ...")
        row = run_refit_eval_and_report(
            test_year=test_year,
            train_years=train_years,
            ckpt_path=ckpt_path,
            save_dir=save_dir_year,
            memmap_dir=memmap_dir,
            dataset_py=dataset_py,
            lookback=args.lookback,
            k=args.k,
            batch_size=args.batch_size,
            device=device,
            feature_indices=feature_indices,
            args=args,
        )
        results.append(row)
        print(f"  {test_year}: IC={row['ic']:.4f} RankIC={row['ric']:.4f} ICIR={row['icir']:.4f} RankICIR={row['rankicir']:.4f} Acc={row['cls_acc']:.4f}")

    if not results:
        print("No checkpoint found for any test year. Exiting.")
        return

    # 3) 保存汇总表
    df = pd.DataFrame(results)
    summary_path = os.path.join(base_save_dir, "refit_test_summary.csv")
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总已保存: {summary_path}")
    print("\nRefit 滚动窗口测试汇总:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
