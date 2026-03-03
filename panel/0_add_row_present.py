# 0_add_row_present.py
# -*- coding: utf-8 -*-
"""
数据覆盖分析 - 理解 row_present 的作用

功能：
1. 分析 parquet 数据的覆盖情况
2. 统计每个交易日有多少股票有数据
3. 计算需要补齐的比例
4. 说明 row_present 在 panel 构建中的作用

关键概念：
- Parquet 是稀疏的：只包含有数据的 (day, stock) 行
- Panel 是完整的：创建所有 [T, N] 组合
- row_present=1: 该行在 parquet 中存在（真实数据）
- row_present=0: 该行不存在，需要补齐（特征填0, isna填1, mask=0）

输出结构：
X[t, i, 0:F]      - 原始特征（缺失行填0）
X[t, i, F:2F]     - isna flags（缺失行填1）
X[t, i, 2F]       - row_present（parquet存在=1，否则=0）

Usage:
    python 0_add_row_present.py
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def load_calendar(calendar_csv):
    """加载交易日历"""
    cal = pd.read_csv(calendar_csv)
    date_col = [c for c in cal.columns if "date" in c.lower()][0]
    dates = pd.to_datetime(cal[date_col]).dt.strftime("%Y-%m-%d").tolist()
    return dates

def analyze_data_coverage(parquet_dir, calendar_csv):
    """
    分析数据覆盖情况，统计每个交易日有数据的股票数
    """
    print("=" * 80)
    print("数据覆盖分析 - Row Presence Analysis")
    print("=" * 80)
    
    # 1. 加载交易日历
    dates = load_calendar(calendar_csv)
    print(f"\n[INFO] 交易日历: {len(dates)} 个交易日")
    print(f"  起始: {dates[0]}")
    print(f"  结束: {dates[-1]}")
    
    # 2. 扫描所有 parquet 文件
    year_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) 
                  if f.endswith(".parquet")]
    year_files.sort()
    print(f"\n[INFO] 找到 {len(year_files)} 个年度文件")
    
    # 3. 统计每日股票数和覆盖率
    date_stock_count = defaultdict(set)  # {date: set of instruments}
    all_instruments = set()
    
    print("\n[INFO] 扫描数据...")
    for yf in tqdm(year_files, desc="Processing files"):
        df = pd.read_parquet(yf, columns=["datetime", "instrument"])
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
        
        for d, sub in df.groupby("datetime"):
            instruments = sub["instrument"].unique()
            date_stock_count[d].update(instruments)
            all_instruments.update(instruments)
    
    print(f"\n[INFO] 总共收集到 {len(all_instruments)} 个股票")
    print(f"[INFO] 有数据的交易日: {len(date_stock_count)} 个")
    
    # 4. 计算覆盖率统计
    print("\n" + "=" * 80)
    print("覆盖率统计")
    print("=" * 80)
    
    coverage_stats = []
    for d in dates:
        if d in date_stock_count:
            count = len(date_stock_count[d])
            coverage = count / len(all_instruments) * 100
            coverage_stats.append({
                "date": d,
                "stock_count": count,
                "coverage_pct": coverage
            })
    
    if coverage_stats:
        df_stats = pd.DataFrame(coverage_stats)
        
        print(f"\n平均每日股票数: {df_stats['stock_count'].mean():.0f}")
        print(f"最少: {df_stats['stock_count'].min()} ({df_stats.loc[df_stats['stock_count'].idxmin(), 'date']})")
        print(f"最多: {df_stats['stock_count'].max()} ({df_stats.loc[df_stats['stock_count'].idxmax(), 'date']})")
        
        print(f"\n覆盖率分布:")
        print(f"  <50%: {(df_stats['coverage_pct'] < 50).sum()} 天")
        print(f"  50-80%: {((df_stats['coverage_pct'] >= 50) & (df_stats['coverage_pct'] < 80)).sum()} 天")
        print(f"  80-95%: {((df_stats['coverage_pct'] >= 80) & (df_stats['coverage_pct'] < 95)).sum()} 天")
        print(f"  ≥95%: {(df_stats['coverage_pct'] >= 95).sum()} 天")
        
        # 5. 输出示例
        print("\n" + "=" * 80)
        print("示例：某些交易日的数据情况")
        print("=" * 80)
        sample_dates = df_stats.sample(min(5, len(df_stats)))
        for _, row in sample_dates.iterrows():
            print(f"  {row['date']}: {row['stock_count']:4d} 只股票 ({row['coverage_pct']:.1f}%)")
        
        # 6. row_present 概念说明
        print("\n" + "=" * 80)
        print("Row Present 概念")
        print("=" * 80)
        print("\n在 panel 格式中：")
        print("  - 形状: [T, N, D]")
        print(f"  - T = {len(dates)} 个交易日")
        print(f"  - N = {len(all_instruments)} 个股票")
        print(f"  - D = 2F + 1 个维度（F个特征 + F个isna + 1个row_present）")
        
        total_cells = len(dates) * len(all_instruments)
        filled_cells = sum(df_stats['stock_count'])
        empty_cells = total_cells - filled_cells
        
        print(f"\n总格子数: {total_cells:,}")
        print(f"有数据格子: {filled_cells:,} (row_present=1)")
        print(f"缺失格子: {empty_cells:,} (row_present=0, {empty_cells/total_cells*100:.1f}%)")
        
        print("\nrow_present=0 的原因：")
        print("  - 股票尚未上市")
        print("  - 股票已退市")
        print("  - 当日停牌")
        print("  - 数据源缺失")
        
        print("\nrow_present 的作用：")
        print("  1. 标记真实 vs 补齐数据")
        print("  2. 当 row_present=0 时：")
        print("     - ret_mask[t,i] = 0")
        print("     - mom_mask[t,i] = 0")
        print("     - both_mask[t,i] = 0")
        print("     → 该样本不参与 loss 计算")
    
    return coverage_stats, all_instruments


def main():
    # 配置
    parquet_dir = r"C:\AI_STOCK\model\Momentum-integrated Multi-task Stock Recommendation\label\alpha158_plus_fund_yearly_parquet_labeled_mimstocr"
    calendar_csv = r"C:\AI_STOCK\dataset\ohlcv_ricequant_2004-2025_parquet_suspension_false\trading_calendar_from_merged.csv"
    
    # 分析数据覆盖
    coverage_stats, all_instruments = analyze_data_coverage(parquet_dir, calendar_csv)
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print("\n下一步：运行 1_build_panel_memmap.py 构建 panel")
    print("  → 会自动根据数据存在性设置 row_present")
    print("  → row_present=0 时自动设置所有 mask=0")


if __name__ == "__main__":
    main()
