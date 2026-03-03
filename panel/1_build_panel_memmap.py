# 0_build_panel_memmap.py
# -*- coding: utf-8 -*-
"""
将 parquet 数据预处理成 memmap 面板格式，用于高速训练

输出格式：
- X_f16.mmap: [T, N, D] float16 - 特征面板
- y_ret_f32.mmap: [T, N] float32 - 回归标签
- y_mom_i8.mmap: [T, N] int8 - 分类标签
- ret_mask_u8.mmap: [T, N] uint8 - 回归mask
- mom_mask_u8.mmap: [T, N] uint8 - 分类mask
- both_mask_u8.mmap: [T, N] uint8 - 两者都有效mask
- meta.json: 元数据（日期、股票、列名等）

Usage:
    python 0_build_panel_memmap.py
"""
import os
import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_calendar(calendar_csv):
    """加载交易日历"""
    cal = pd.read_csv(calendar_csv)
    date_col = [c for c in cal.columns if "date" in c.lower()][0]
    dates = pd.to_datetime(cal[date_col]).dt.strftime("%Y-%m-%d").tolist()
    return dates

def list_year_parquets(parquet_dir):
    """列出所有年度parquet文件"""
    files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) 
             if f.endswith(".parquet")]
    files.sort()
    return files

def detect_numeric_feature_cols(sample_df, exclude_regex=None):
    """
    检测数值特征列
    
    排除：
    1. 元信息列（datetime, instrument等）
    2. 标签列（label_*）
    3. isna列（会单独处理）
    4. 可选的正则匹配（如 dividend）
    """
    non_feature = {
        "datetime", "instrument", "order_book_id", "date_norm", "instrument_norm",
        "label_ret_1d", "label_mom_cls", "label_ret_valid", "label_mom_valid", "label_both_valid"
    }
    
    cols = []
    for c in sample_df.columns:
        if c in non_feature:
            continue
        # 排除 __isna 列（会单独处理）
        if "__isna" in c or "_isna" in c:
            continue
        if pd.api.types.is_numeric_dtype(sample_df[c]):
            cols.append(c)
    
    # 排除 dividend 相关列
    if exclude_regex:
        pat = re.compile(exclude_regex)
        cols = [c for c in cols if not pat.search(c)]
    
    return cols

def collect_all_instruments(year_files):
    """收集所有股票代码"""
    print("[INFO] Collecting all instruments...")
    inst_set = set()
    for f in tqdm(year_files, desc="Scanning files"):
        part = pd.read_parquet(f, columns=["instrument"])
        inst_set.update(part["instrument"].unique().tolist())
    instruments = sorted(list(inst_set))
    return instruments

def main():
    # ==================== 配置 ====================
    parquet_dir = r"C:\AI_STOCK\model\Momentum-integrated Multi-task Stock Recommendation\label\alpha158_plus_fund_yearly_parquet_labeled_mimstocr"
    calendar_csv = r"C:\AI_STOCK\dataset\ohlcv_ricequant_2004-2025_parquet_suspension_false\trading_calendar_from_merged.csv"
    out_dir = r"C:\AI_STOCK\model\Momentum-integrated Multi-task Stock Recommendation\panel\memmap_data"
    
    # 可选：排除某些特征（如dividend相关）
    exclude_regex = r"(?i)dividend"
    
    # 数据写入优化：每处理N个文件后flush（防止缓存过多）
    flush_interval = 2  # 每2个年度文件flush一次
    
    # ==================== 初始化 ====================
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("Building Panel Memmap Data")
    print("=" * 80)
    
    # 1. 加载日历
    dates = load_calendar(calendar_csv)
    T = len(dates)
    date2t = {d: i for i, d in enumerate(dates)}
    print(f"[INFO] T = {T} trading days")
    
    # 2. 扫描特征列
    year_files = list_year_parquets(parquet_dir)
    print(f"[INFO] Found {len(year_files)} year files")
    
    df_sample = pd.read_parquet(year_files[0])
    feat_cols = detect_numeric_feature_cols(df_sample, exclude_regex)
    print(f"[INFO] Detected {len(feat_cols)} feature columns")
    
    # 3. 收集所有股票代码（补齐的关键！）
    instruments = collect_all_instruments(year_files)
    N = len(instruments)
    inst2i = {inst: i for i, inst in enumerate(instruments)}
    print(f"[INFO] N = {N} unique instruments (stocks)")
    print(f"       This means we'll create a complete [T={T}, N={N}] panel")
    print(f"       where missing (day, stock) combinations have row_present=0")
    
    # 4. 检测 isna 列
    isna_cols = []
    excluded_isna_count = 0
    
    for c in feat_cols:
        # 尝试双下划线（常见格式）
        if (c + "__isna") in df_sample.columns:
            isna_cols.append(c + "__isna")
        # 尝试单下划线
        elif (c + "_isna") in df_sample.columns:
            isna_cols.append(c + "_isna")
    
    # 如果排除了 dividend 特征，也要排除对应的 isna 列
    if exclude_regex:
        pat = re.compile(exclude_regex)
        original_isna_count = len(isna_cols)
        isna_cols = [c for c in isna_cols if not pat.search(c)]
        excluded_isna_count = original_isna_count - len(isna_cols)
    
    # row_present 列：标记数据是否真实存在
    # 注意：不是从 parquet 读取，而是在构建 panel 时根据数据存在性动态计算
    row_present_col = "row_present"
    
    # X 列顺序：[特征 0:F, isna F:2F, row_present 2F]
    X_cols = feat_cols + isna_cols + [row_present_col]
    
    D = len(X_cols)
    F = len(feat_cols)
    
    print(f"\n[INFO] 输入维度 D = {D} = 2F + 1")
    print(f"  - Features (0:{F}): {F} 列")
    print(f"  - IsNA flags ({F}:{2*F}): {len(isna_cols)} 列")
    print(f"  - Row present ({2*F}): 1 列")
    
    if excluded_isna_count > 0:
        print(f"  - 已排除 {excluded_isna_count} 个 isna 列（对应被排除的特征）")
    
    if len(isna_cols) == 0:
        print(f"[WARNING] 未找到isna列，将不包含缺失值标志")
    else:
        print(f"[INFO] IsNA列格式示例: {isna_cols[0]}")
    
    print(f"\n[INFO] Row present 说明:")
    print(f"  - 标记该 (交易日, 股票) 是否在 parquet 中真实存在")
    print(f"  - row_present=1: 有真实数据（parquet 中存在该行）")
    print(f"  - row_present=0: 补齐的空行（未上市/退市/停牌/缺失）")
    print(f"  - 当 row_present=0 时，所有 mask 自动设为 0（不参与训练）")
    print(f"\n[INFO] Panel 补齐说明:")
    print(f"  - Parquet 是稀疏的：只包含有数据的 (day, stock) 行")
    print(f"  - Panel 是完整的：创建所有 [{T} x {N}] 组合")
    print(f"  - 缺失的行：特征填0, isna填1, row_present=0, mask=0")
    
    if len(isna_cols) == 0:
        print(f"[WARNING] 未找到isna列，将不包含缺失值标志")
    else:
        print(f"[INFO] IsNA列格式示例: {isna_cols[0]}")
    
    # 检查磁盘空间
    import shutil
    disk_stat = shutil.disk_usage(out_dir)
    required_gb = (T * N * D * 2 + T * N * 4 + T * N * 1 * 4) / 1024**3  # 约52GB
    available_gb = disk_stat.free / 1024**3
    print(f"\n[DISK] 所需空间: ~{required_gb:.1f}GB")
    print(f"[DISK] 可用空间: {available_gb:.1f}GB")
    if available_gb < required_gb + 10:  # 至少多留10GB
        print(f"[ERROR] 磁盘空间不足！需要至少 {required_gb+10:.0f}GB 可用空间")
        return
    
    # 5. 创建memmap文件（优化版：不立即分配空间）
    print("\n[INFO] Creating memmap files (progressive allocation)...")
    X_path = os.path.join(out_dir, "X_f16.mmap")
    yret_path = os.path.join(out_dir, "y_ret_f32.mmap")
    ymom_path = os.path.join(out_dir, "y_mom_i8.mmap")
    rmask_path = os.path.join(out_dir, "ret_mask_u8.mmap")
    mmask_path = os.path.join(out_dir, "mom_mask_u8.mmap")
    bmask_path = os.path.join(out_dir, "both_mask_u8.mmap")
    
    # 使用渐进式创建，避免一次性分配大内存
    # 方法：先创建空文件，再用r+模式打开（不会立即分配所有空间）
    def create_memmap_progressive(path, dtype, shape):
        """渐进式创建memmap，避免Windows一次性分配大空间"""
        # 计算需要的字节数
        itemsize = np.dtype(dtype).itemsize
        total_bytes = np.prod(shape) * itemsize
        
        # 创建空文件（不分配空间）
        with open(path, 'wb') as f:
            # 只写入最后一个字节，让文件系统知道文件大小
            f.seek(total_bytes - 1)
            f.write(b'\x00')
        
        # 用r+模式打开（读写已存在的文件）
        return np.memmap(path, dtype=dtype, mode='r+', shape=shape)
    
    print("  Creating X_f16.mmap (~52GB)...")
    X_mm = create_memmap_progressive(X_path, np.float16, (T, N, D))
    print("  Creating y_ret_f32.mmap...")
    yret_mm = create_memmap_progressive(yret_path, np.float32, (T, N))
    print("  Creating y_mom_i8.mmap...")
    ymom_mm = create_memmap_progressive(ymom_path, np.int8, (T, N))
    print("  Creating ret_mask_u8.mmap...")
    rmask_mm = create_memmap_progressive(rmask_path, np.uint8, (T, N))
    print("  Creating mom_mask_u8.mmap...")
    mmask_mm = create_memmap_progressive(mmask_path, np.uint8, (T, N))
    print("  Creating both_mask_u8.mmap...")
    bmask_mm = create_memmap_progressive(bmask_path, np.uint8, (T, N))
    
    # 6. 默认填充（缺失行）- 分块填充避免内存爆炸
    print("[INFO] Initializing with default values (chunked)...")
    print("  - 特征值初始化为 0")
    print("  - isna flags 初始化为 1（假设缺失）")
    print("  - row_present 初始化为 0（假设不存在）")
    print("  - 所有 mask 初始化为 0（不参与训练）")
    
    chunk_size = 100  # 每次处理100个交易日
    row_present_idx = len(feat_cols) + len(isna_cols)  # row_present 在最后一列
    
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        
        # 初始化特征为 0
        X_mm[t_start:t_end, :, :len(feat_cols)] = 0
        
        # 初始化 isna flags 为 1（假设全部缺失）
        if len(isna_cols) > 0:
            isna_start = len(feat_cols)
            isna_end = isna_start + len(isna_cols)
            X_mm[t_start:t_end, :, isna_start:isna_end] = 1
        
        # 初始化 row_present 为 0（假设全部不存在）
        X_mm[t_start:t_end, :, row_present_idx] = 0
        
        # 初始化标签和mask
        yret_mm[t_start:t_end] = 0
        ymom_mm[t_start:t_end] = -1  # ignore_index
        rmask_mm[t_start:t_end] = 0
        mmask_mm[t_start:t_end] = 0
        bmask_mm[t_start:t_end] = 0
        
        if (t_end) % 500 == 0:
            print(f"  Initialized {t_end}/{T} days...")
    
    # 7. 逐年文件填充（补齐逻辑）
    print("\n[INFO] Filling memmap from parquet files...")
    print("  - Parquet 中存在的行：设置 row_present=1, 填充真实数据和mask")
    print("  - Parquet 中不存在的行：保持 row_present=0, mask=0（不参与训练）")
    print(f"  - 预期补齐率: ~{(1 - 15311925/27634618)*100:.1f}%（基于之前的覆盖分析）")
    
    label_cols = ["label_ret_1d", "label_mom_cls", "label_ret_valid", 
                  "label_mom_valid", "label_both_valid"]
    
    # use_cols 只需要包含真实存在于 parquet 中的列
    # row_present 不在 parquet 中，是动态计算的
    use_cols_base = ["datetime", "instrument"] + feat_cols + isna_cols + label_cols
    use_cols = [c for c in use_cols_base if c != row_present_col]
    
    # 统计信息
    overflow_count = 0
    overflow_warned = False
    nan_feature_count = 0
    nan_label_count = 0
    
    total_rows = 0
    for file_idx, yf in enumerate(tqdm(year_files, desc="Processing years")):
        # 读取年度文件
        df = pd.read_parquet(yf)
        
        # 过滤需要的列
        available_cols = [c for c in use_cols if c in df.columns]
        df = df[available_cols]
        
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
        
        # 按日期分组写入
        for d, sub in df.groupby("datetime"):
            if d not in date2t:
                continue
            
            t = date2t[d]
            
            # 获取股票索引
            idx = sub["instrument"].map(inst2i).dropna().astype(int).to_numpy()
            if len(idx) == 0:
                continue
            
            # 过滤对应行
            sub = sub.loc[sub["instrument"].map(inst2i).notna()].reset_index(drop=True)
            
            # 【关键】设置 row_present=1（标记该行真实存在）
            X_mm[t, idx, row_present_idx] = 1
            
            # 填充特征和isna
            X_cols_to_fill = [c for c in (feat_cols + isna_cols) if c in sub.columns]
            if len(X_cols_to_fill) > 0:
                X_day = sub[X_cols_to_fill].to_numpy(dtype=np.float32)
                
                # 【关键】处理 NaN：LSTM 无法处理 NaN，必须填充为 0
                nan_mask = np.isnan(X_day)
                if nan_mask.any():
                    nan_feature_count += nan_mask.sum()
                    X_day[nan_mask] = 0.0
                
                # 检查并处理溢出（float16 范围: -65504 到 65504）
                overflow_mask = (np.abs(X_day) > 65500)
                if overflow_mask.any():
                    overflow_count += overflow_mask.sum()
                    if not overflow_warned:
                        print(f"\n[WARNING] 检测到特征值超出float16范围，将自动裁剪到 [-65500, 65500]")
                        print(f"          这不会影响训练，因为极端值会被标准化")
                        overflow_warned = True
                    # Clip到安全范围
                    X_day = np.clip(X_day, -65500, 65500)
                
                # 找到对应的列索引（只填充特征和isna，不填充row_present）
                col_idx = [X_cols.index(c) for c in X_cols_to_fill]
                X_mm[t, idx[:, None], col_idx] = X_day.astype(np.float16)
            
            # 填充标签和mask（只有 row_present=1 的行才会有有效标签）
            if "label_ret_1d" in sub.columns:
                ret_vals = sub["label_ret_1d"].to_numpy(np.float32)
                # 处理标签中的 NaN：设为 0，并将对应 mask 设为 0（不参与训练）
                ret_nan_mask = np.isnan(ret_vals)
                if ret_nan_mask.any():
                    nan_label_count += ret_nan_mask.sum()
                    ret_vals[ret_nan_mask] = 0.0
                yret_mm[t, idx] = ret_vals
                
            if "label_mom_cls" in sub.columns:
                mom_vals = sub["label_mom_cls"].to_numpy(np.float32)  # 先用float读取
                # 处理标签中的 NaN：设为 -1（ignore_index）
                mom_nan_mask = np.isnan(mom_vals)
                if mom_nan_mask.any():
                    mom_vals[mom_nan_mask] = -1
                ymom_mm[t, idx] = mom_vals.astype(np.int8)
                
            if "label_ret_valid" in sub.columns:
                rmask_mm[t, idx] = sub["label_ret_valid"].fillna(0).to_numpy(np.uint8)
            if "label_mom_valid" in sub.columns:
                mmask_mm[t, idx] = sub["label_mom_valid"].fillna(0).to_numpy(np.uint8)
            if "label_both_valid" in sub.columns:
                bmask_mm[t, idx] = sub["label_both_valid"].fillna(0).to_numpy(np.uint8)
            
            total_rows += len(idx)
        
        # 定期flush到磁盘，释放内存
        if (file_idx + 1) % flush_interval == 0:
            X_mm.flush()
            yret_mm.flush()
            ymom_mm.flush()
            rmask_mm.flush()
            mmask_mm.flush()
            bmask_mm.flush()
            # 清理DataFrame内存
            import gc
            gc.collect()
        
        del df
    
    print(f"\n[INFO] Filled {total_rows:,} total rows")
    if overflow_count > 0:
        print(f"[INFO] 处理了 {overflow_count:,} 个溢出值（已裁剪到float16范围）")
        print(f"       占比: {overflow_count/total_rows/len(X_cols)*100:.3f}%")
    if nan_feature_count > 0:
        print(f"[INFO] 处理了 {nan_feature_count:,} 个特征NaN值（已填充为0）")
        print(f"       占比: {nan_feature_count/total_rows/len(X_cols)*100:.3f}%")
    if nan_label_count > 0:
        print(f"[INFO] 处理了 {nan_label_count:,} 个标签NaN值（已填充为0）")
        print(f"       占比: {nan_label_count/total_rows*100:.3f}%")
    
    # 8. Flush到磁盘
    print("\n[INFO] Flushing to disk...")
    X_mm.flush()
    yret_mm.flush()
    ymom_mm.flush()
    rmask_mm.flush()
    mmask_mm.flush()
    bmask_mm.flush()
    
    # 9. 保存元数据
    print("[INFO] Saving metadata...")
    meta = {
        "T": T,
        "N": N,
        "D": D,
        "dates": dates,
        "instruments": instruments,
        "X_cols": X_cols,
        "feat_cols": feat_cols,
        "isna_cols": isna_cols,
        "has_row_present": row_present_col is not None,
        "paths": {
            "X": "X_f16.mmap",
            "y_ret": "y_ret_f32.mmap",
            "y_mom": "y_mom_i8.mmap",
            "ret_mask": "ret_mask_u8.mmap",
            "mom_mask": "mom_mask_u8.mmap",
            "both_mask": "both_mask_u8.mmap",
        }
    }
    
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    # 10. 统计信息
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Shape: [T={T}, N={N}, D={D}]")
    print(f"Total size: ~{T*N*D*2 / 1e9:.2f} GB (X_f16)")
    print(f"Files created:")
    for name, path in meta["paths"].items():
        full_path = os.path.join(out_dir, path)
        size_mb = os.path.getsize(full_path) / 1e6
        print(f"  - {path}: {size_mb:.1f} MB")
    print("=" * 80)
    print("[DONE] Panel memmap data ready for training!")

if __name__ == "__main__":
    main()
