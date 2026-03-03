# 2_dataset_memmap.py
# -*- coding: utf-8 -*-
"""
基于 memmap 面板数据的高速 Dataset

特点：
1. 训练时直接 slice 窗口，不再读 parquet
2. 支持多进程 DataLoader（num_workers>0）
3. 内存占用低（memmap 按需加载）

Usage:
    from dataset_memmap import MemmapDayWindowDataset, build_dataloader
    
    ds = MemmapDayWindowDataset(
        memmap_dir="panel/memmap_data",
        lookback=60,
        day_start=60,
        day_end=5000,
        k=512,
        seed=42
    )
    
    loader = build_dataloader(
        ds,
        batch_size=4,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MemmapDayWindowDataset(Dataset):
    """
    基于 memmap 的滑动窗口 Dataset
    
    Args:
        memmap_dir: memmap 数据目录
        lookback: 窗口长度
        day_start: 起始交易日索引（需 >= lookback-1）
        day_end: 结束交易日索引
        k: 每日采样股票数
        seed: 随机种子
        sample_both_valid_only: 是否仅从 both_valid=1 的股票中采样 (⚠️ 用到 t+1 信息)
        sample_ret_valid_only: 是否仅从 ret_valid=1 的股票中采样 (⚠️ 用到 t+1 信息)
        sample_present_only: 是否仅从 row_present=1 的股票中采样 (✓ 仅用 t 时刻信息, 无泄露)
    """
    
    def __init__(
        self, 
        memmap_dir, 
        lookback, 
        day_start, 
        day_end, 
        k, 
        seed=42,
        sample_both_valid_only=True,
        sample_ret_valid_only=False,
        sample_present_only=False,
        feature_indices=None,  # 新增：用于特征选择
    ):
        self.memmap_dir = memmap_dir
        self.lookback = lookback
        self.k = k
        self.sample_both_valid_only = sample_both_valid_only
        self.sample_ret_valid_only = sample_ret_valid_only
        self.sample_present_only = sample_present_only
        self.rng = np.random.RandomState(seed)
        self.feature_indices = feature_indices  # 特征选择索引
        
        # 确保有足够的历史
        assert day_start >= lookback - 1, f"day_start={day_start} must >= lookback-1={lookback-1}"
        
        self.day_idxs = np.arange(day_start, day_end + 1, dtype=np.int32)
        
        # 加载元数据
        with open(os.path.join(memmap_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        self.T = meta["T"]
        self.N = meta["N"]
        self.D = meta["D"]
        self.dates = meta["dates"]
        self.instruments = meta["instruments"]
        
        # 加载特征列名（用于特征重要性输出）
        self.feature_cols = meta.get("feat_cols", [])
        self.X_cols = meta.get("X_cols", [])
        
        print(f"[Dataset] T={self.T}, N={self.N}, D={self.D}")
        print(f"[Dataset] Lookback={lookback}, Days={len(self.day_idxs)} ({day_start}→{day_end})")
        if sample_present_only:
            print(f"[Dataset] Sample K={k}, Present_only=True (\u2713 no future info)")
        elif sample_ret_valid_only:
            print(f"[Dataset] Sample K={k}, Ret_valid_only=True (\u26a0\ufe0f uses t+1 info)")
        else:
            print(f"[Dataset] Sample K={k}, Both_valid_only={sample_both_valid_only}")
        
        # row_present 在 X 的最后一列: index = 2F = D-1
        self._row_present_idx = self.D - 1
        
        # Windows多进程需延迟打开memmap（避免pickle问题）
        self._X = None
        self._yret = None
        self._ymom = None
        self._rm = None
        self._mm = None
        self._bm = None
    
    def _lazy_open(self):
        """延迟打开 memmap 文件（每个 worker 进程独立打开）"""
        if self._X is not None:
            return
        
        def mm(name, dtype, shape):
            path = os.path.join(self.memmap_dir, name)
            return np.memmap(path, dtype=dtype, mode="r", shape=shape)
        
        self._X = mm("X_f16.mmap", np.float16, (self.T, self.N, self.D))
        self._yret = mm("y_ret_f32.mmap", np.float32, (self.T, self.N))
        self._ymom = mm("y_mom_i8.mmap", np.int8, (self.T, self.N))
        self._rm = mm("ret_mask_u8.mmap", np.uint8, (self.T, self.N))
        self._mm = mm("mom_mask_u8.mmap", np.uint8, (self.T, self.N))
        self._bm = mm("both_mask_u8.mmap", np.uint8, (self.T, self.N))
    
    def __len__(self):
        return len(self.day_idxs)
    
    def __getitem__(self, i):
        """
        返回第 i 个样本（对应某个交易日）
        
        Returns:
            dict:
                X: [K, L, D] torch.FloatTensor
                y_ret: [K] torch.FloatTensor
                y_mom: [K] torch.LongTensor
                ret_mask: [K] torch.FloatTensor
                mom_mask: [K] torch.FloatTensor
                both_mask: [K] torch.FloatTensor
                date_idx: int
        """
        self._lazy_open()
        
        t = int(self.day_idxs[i])
        L = self.lookback
        
        # 采样策略
        if self.sample_present_only:
            # 模式0（推荐）：只从 row_present=1 的股票中采样
            # row_present 仅取决于当日(t)数据是否存在，不依赖 t+1，无未来信息泄露
            present = self._X[t, :, self._row_present_idx]  # [N] float16
            valid_idx = np.flatnonzero(present > 0)

            if len(valid_idx) >= self.k:
                idx = self.rng.choice(valid_idx, size=self.k, replace=False)
            else:
                if len(valid_idx) > 0:
                    n_valid = len(valid_idx)
                    n_other = self.k - n_valid
                    other_idx = np.setdiff1d(np.arange(self.N), valid_idx)
                    other_sample = self.rng.choice(other_idx, size=min(n_other, len(other_idx)), replace=False)
                    idx = np.concatenate([valid_idx, other_sample])
                    if len(idx) < self.k:
                        idx = np.concatenate([idx, self.rng.choice(self.N, size=self.k - len(idx), replace=False)])
                else:
                    idx = self.rng.choice(self.N, size=self.k, replace=False)
        elif self.sample_ret_valid_only:
            # 模式1：只从 ret_valid=1 的股票中采样
            ret_valid = self._rm[t]  # [N]
            valid_idx = np.flatnonzero(ret_valid > 0)
            
            if len(valid_idx) >= self.k:
                idx = self.rng.choice(valid_idx, size=self.k, replace=False)
            else:
                # 不足时补充其他股票（但保证至少有 ret_valid 的优先）
                if len(valid_idx) > 0:
                    n_valid = len(valid_idx)
                    n_other = self.k - n_valid
                    other_idx = np.setdiff1d(np.arange(self.N), valid_idx)
                    other_sample = self.rng.choice(other_idx, size=n_other, replace=False)
                    idx = np.concatenate([valid_idx, other_sample])
                else:
                    idx = self.rng.choice(self.N, size=self.k, replace=False)
        elif self.sample_both_valid_only:
            # 模式2：从 both_valid=1 的股票中采样
            both = self._bm[t]  # [N]
            valid_idx = np.flatnonzero(both > 0)
            
            if len(valid_idx) >= self.k:
                idx = self.rng.choice(valid_idx, size=self.k, replace=False)
            else:
                # 不足时补充其他股票
                if len(valid_idx) > 0:
                    n_valid = len(valid_idx)
                    n_other = self.k - n_valid
                    other_idx = np.setdiff1d(np.arange(self.N), valid_idx)
                    other_sample = self.rng.choice(other_idx, size=n_other, replace=False)
                    idx = np.concatenate([valid_idx, other_sample])
                else:
                    idx = self.rng.choice(self.N, size=self.k, replace=False)
        else:
            # 模式3：从所有股票中随机采样
            idx = self.rng.choice(self.N, size=self.k, replace=False)
        
        # Slice 窗口：[t-L+1:t+1, idx, :] -> [L, K, D] -> [K, L, D]
        x = self._X[t - L + 1: t + 1, idx, :]  # memmap slice（只触发需要的部分）
        x = np.transpose(x, (1, 0, 2)).astype(np.float32, copy=False)  # [K, L, D]
        
        # 应用特征选择（如果指定）
        if self.feature_indices is not None:
            x = x[:, :, self.feature_indices]  # [K, L, D_selected]
        
        # 标签和mask（target day = t）
        y_ret = self._yret[t, idx].astype(np.float32, copy=False)
        y_mom = self._ymom[t, idx].astype(np.int64, copy=False)  # CE需要long
        ret_mask = self._rm[t, idx].astype(np.float32, copy=False)
        mom_mask = self._mm[t, idx].astype(np.float32, copy=False)
        both_mask = self._bm[t, idx].astype(np.float32, copy=False)
        
        return {
            "X": torch.from_numpy(x),                # [K, L, D]
            "y_ret": torch.from_numpy(y_ret),        # [K]
            "y_mom": torch.from_numpy(y_mom),        # [K]
            "ret_mask": torch.from_numpy(ret_mask),  # [K]
            "mom_mask": torch.from_numpy(mom_mask),  # [K]
            "both_mask": torch.from_numpy(both_mask),# [K]
            "date_idx": t,
        }


def build_dataloader(
    dataset,
    batch_size=4,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
):
    """
    构建 DataLoader
    
    Args:
        dataset: MemmapDayWindowDataset
        batch_size: 批次大小（B个交易日）
        num_workers: 数据加载进程数
        shuffle: 是否打乱
        pin_memory: 是否锁页内存
        persistent_workers: 是否保持worker进程
        prefetch_factor: 预取批次数
    
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


# ==================== 测试代码 ====================
def test_dataset():
    """快速测试 Dataset 是否正常工作"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap_dir", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=3)
    args = parser.parse_args()
    
    print("=" * 80)
    print("Testing Memmap Dataset")
    print("=" * 80)
    
    # 加载元数据并验证结构
    import json
    meta_path = os.path.join(args.memmap_dir, "meta.json")
    print(f"\n[CHECK] 验证元数据: {meta_path}")
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    T = meta["T"]
    N = meta["N"]
    D = meta["D"]
    feat_cols = meta.get("feat_cols", [])
    isna_cols = meta.get("isna_cols", [])
    has_row_present = meta.get("has_row_present", False)
    dates = meta.get("dates", [])
    instruments = meta.get("instruments", [])
    
    print(f"  T (交易日数): {T}")
    print(f"  N (股票数): {N}")
    print(f"  D (特征维度): {D}")
    print(f"  日期范围: {dates[0] if dates else 'N/A'} → {dates[-1] if dates else 'N/A'}")
    print(f"  股票示例: {', '.join(instruments[:5]) if instruments else 'N/A'}")
    
    # 验证维度结构
    print(f"\n[CHECK] 验证维度结构 (D = 2F + 1)")
    F = len(feat_cols)
    expected_D = F * 2 + (1 if has_row_present else 0)
    
    print(f"  Features: {F}")
    print(f"  IsNA flags: {len(isna_cols)}")
    print(f"  Row present: {1 if has_row_present else 0}")
    print(f"  期望 D: {expected_D}")
    print(f"  实际 D: {D}")
    
    if D == expected_D == 947:
        print(f"  ✅ 维度正确！D = {F} + {len(isna_cols)} + 1 = {D}")
    elif D == expected_D:
        print(f"  ✅ 维度一致！D = {D}")
    else:
        print(f"  ❌ 维度不匹配！期望 {expected_D}，实际 {D}")
    
    if F == 473 and len(isna_cols) == 473 and has_row_present:
        print(f"  ✅ 完美匹配：473 features + 473 isna + 1 row_present = 947")
    
    # 验证特征列命名
    print(f"\n[CHECK] 验证 IsNA 列对应关系")
    isna_match_count = 0
    for feat in feat_cols[:5]:  # 检查前5个
        expected_isna = feat + "__isna"
        if expected_isna in isna_cols:
            isna_match_count += 1
    
    if isna_match_count == 5:
        print(f"  ✅ IsNA 列命名正确（{feat_cols[0]} → {feat_cols[0]}__isna）")
    else:
        print(f"  ⚠️ 部分 IsNA 列可能命名不一致")
    
    # 验证标签列
    print(f"\n[CHECK] 验证标签文件")
    label_files = ["y_ret_f32.mmap", "y_mom_i8.mmap", 
                   "ret_mask_u8.mmap", "mom_mask_u8.mmap", "both_mask_u8.mmap"]
    
    for lf in label_files:
        path = os.path.join(args.memmap_dir, lf)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  ✅ {lf}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {lf}: 不存在！")
    
    # 创建dataset
    print(f"\n[INFO] 创建 Dataset...")
    ds = MemmapDayWindowDataset(
        memmap_dir=args.memmap_dir,
        lookback=args.lookback,
        day_start=args.lookback,
        day_end=args.lookback + 100,  # 测试100天
        k=args.k,
        seed=42,
        sample_both_valid_only=False,  # 改为False以采样到row_present=0的样本
    )
    
    print(f"\n[INFO] Dataset length: {len(ds)}")
    
    # 测试单个样本
    print("\n[CHECK] 测试单个样本...")
    sample = ds[0]
    X_shape = sample['X'].shape  # [K, L, D]
    
    print(f"  X shape: {X_shape}")
    print(f"  y_ret shape: {sample['y_ret'].shape}")
    print(f"  y_mom shape: {sample['y_mom'].shape}")
    
    # 验证 D 维度
    expected_D = 947 if F == 473 else (933 if F == 466 else D)
    if X_shape[2] == expected_D:
        print(f"  ✅ X 最后一维 = {X_shape[2]}（正确）")
    else:
        print(f"  ⚠️ X 最后一维 = {X_shape[2]}（期望{expected_D}）")
    
    # 测试单个样本（保留原逻辑）
    print(f"\n[CHECK] 测试单个样本...")
    X_sample = sample['X']  # [K, L, D]
    row_present_sample = X_sample[:, :, -1]  # [K, L]
    print(f"  样本中 row_present 均值: {row_present_sample.mean():.3f}")
    print(f"  样本中 row_present 范围: [{row_present_sample.min():.0f}, {row_present_sample.max():.0f}]")
    
    # 验证标签和mask
    print(f"\n[CHECK] 验证标签和 Mask 数值范围")
    print(f"  y_ret 范围: [{sample['y_ret'].min():.4f}, {sample['y_ret'].max():.4f}]")
    print(f"  y_mom 范围: [{sample['y_mom'].min()}, {sample['y_mom'].max()}]")
    print(f"  ret_mask 有效数: {sample['ret_mask'].sum():.0f}/{len(sample['ret_mask'])}")
    print(f"  mom_mask 有效数: {sample['mom_mask'].sum():.0f}/{len(sample['mom_mask'])}")
    print(f"  both_mask 有效数: {sample['both_mask'].sum():.0f}/{len(sample['both_mask'])}")
    
    # 检查 row_present（最后一列）- 完整统计
    print(f"\n[CHECK] Row Present 完整统计（扫描所有 {T}x{N} 数据）")
    print(f"  这可能需要几分钟...")
    
    # 直接从memmap读取完整的row_present列
    X_path = os.path.join(args.memmap_dir, "X_f16.mmap")
    X_full = np.memmap(X_path, dtype=np.float16, mode='r', shape=(T, N, D))
    
    # row_present是最后一列
    row_present_idx = D - 1
    row_present_full = X_full[:, :, row_present_idx]  # [T, N]
    
    total_cells = T * N
    count_zero = (row_present_full == 0).sum()
    count_one = (row_present_full == 1).sum()
    count_other = total_cells - count_zero - count_one
    mean_val = row_present_full.mean()
    
    print(f"\n  总格子数: {total_cells:,}")
    print(f"  row_present=0: {count_zero:,} ({count_zero/total_cells*100:.2f}%)")
    print(f"  row_present=1: {count_one:,} ({count_one/total_cells*100:.2f}%)")
    if count_other > 0:
        print(f"  异常值（非0/1）: {count_other:,} ❌")
    print(f"  均值: {mean_val:.4f}")
    
    print(f"\n  预期统计（基于覆盖分析）:")
    print(f"    - 总格子: {T} x {N} = {T*N:,}")
    print(f"    - 有数据: ~15,311,925 (55.4%)")
    print(f"    - 缺失: ~12,322,693 (44.6%)")
    
    if count_one > 0 and count_zero > 0:
        print(f"\n  ✅ 同时包含0和1（补齐逻辑正常）")
        if 0.50 <= mean_val <= 0.60:
            print(f"  ✅ 均值在预期范围内")
        else:
            print(f"  ⚠️ 均值 {mean_val:.4f} 偏离预期 0.554")
            if mean_val < 0.50:
                print(f"      可能原因：某些年份数据缺失较多")
            else:
                print(f"      可能原因：某些年份数据覆盖较好")
    else:
        print(f"\n  ❌ 补齐逻辑异常！")
    
    # 按年份统计（如果有时间信息）
    if dates:
        print(f"\n  按年份统计 row_present:")
        year_stats = {}
        for t in range(T):
            date_str = dates[t]
            year = date_str[:4]
            if year not in year_stats:
                year_stats[year] = {"total": 0, "present": 0}
            year_stats[year]["total"] += N
            year_stats[year]["present"] += (row_present_full[t] == 1).sum()
        
        for year in sorted(year_stats.keys())[:5]:  # 显示前5年
            stats = year_stats[year]
            ratio = stats["present"] / stats["total"]
            print(f"    {year}: {stats['present']:,}/{stats['total']:,} ({ratio*100:.1f}%)")
        print(f"    ... (仅显示前5年)")
    
    del X_full, row_present_full  # 释放内存
    
    # 测试单个样本（保留原逻辑）
    print(f"\n[CHECK] 测试单个样本...")
    X_sample = sample['X']  # [K, L, D]
    row_present_sample = X_sample[:, :, -1]  # [K, L]
    
    # 检查日期信息
    date_idx = sample['date_idx']
    if date_idx < len(dates):
        print(f"\n[CHECK] 日期信息")
        print(f"  样本对应日期: {dates[date_idx]}")
        print(f"  ✅ 日期索引有效")
    else:
        print(f"\n  ❌ 日期索引超出范围！")
    
    # 测试dataloader
    print(f"\n[INFO] Testing DataLoader with {args.num_workers} workers...")
    loader = build_dataloader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    
    import time
    times = []
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        
        start = time.time()
        X = batch["X"]  # [B, K, L, D]
        y_ret = batch["y_ret"]  # [B, K]
        y_mom = batch["y_mom"]  # [B, K]
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"  Batch {i}: X={X.shape}, y_ret={y_ret.shape}, y_mom={y_mom.shape}, time={elapsed:.3f}s")
    
    avg_time = np.mean(times) if times else 0
    print(f"\n[INFO] Average batch time: {avg_time:.3f}s")
    print("=" * 80)
    print("[DONE] Test passed!")

if __name__ == "__main__":
    test_dataset()
