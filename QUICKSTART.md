# 快速开始指南

本指南提供 MiM-StocR 完整流程的快速上手步骤。

## 前置要求

- Python 3.8+
- PyTorch 1.12+
- pandas, numpy, pyarrow
- 原始股票数据（Parquet 格式，包含日期、股票代码、OHLCV等）

## 完整流程（三步走）

### 步骤 1: 生成标签 (label/)

```bash
cd label

python label_mimstocr_labels.py \
    --input_dir "../../data/raw_stock_data" \
    --output_dir "../../data/labeled_data"
```

**说明**:
- 为每只股票生成次日收益率标签 (`label_ret_1d`)
- 生成20日动量分类标签 (`label_mom_cls`, 0-4五分类)
- 生成标签有效性掩码（过滤停牌、涨跌停、缺失数据等）

**输出**: 带标签的 Parquet 文件

---

### 步骤 2: 构建面板数据 (panel/)

#### 2.1 添加 row_present 列

```bash
cd ../panel

python 0_add_row_present.py \
    --data_dir "../../data/labeled_data" \
    --output_dir "../../data/labeled_with_present"
```

**说明**: 添加 `row_present` 列，标记当日数据是否存在（避免数据泄露）

#### 2.2 构建 memmap 格式数据

```bash
python 1_build_panel_memmap.py \
    --parquet_dir "../../data/labeled_with_present" \
    --output_dir "./memmap_data" \
    --start_year 2007 \
    --end_year 2020
```

**说明**: 
- 将 Parquet 转换为高效的 memmap 格式
- 输出: `X_f16.mmap`, `y_ret_f32.mmap`, `y_mom_i8.mmap`, 掩码文件, `meta.json`

#### 2.3 验证数据（可选）

```bash
# Windows
2_test_memmap_dataset.bat

# 或手动运行
python -c "import sys; sys.path.insert(0, '.'); from 2_dataset_memmap import test_dataset; test_dataset()" \
    --memmap_dir ./memmap_data \
    --lookback 60 \
    --k 512
```

**输出**: `panel/memmap_data/` 包含所有面板数据

---

### 步骤 3: 训练模型 (model_training/)

#### 3.1 Stage 2 训练（主流程）

```bash
cd ../model_training

python 4_train_stage2.py \
    --dataset_py ../panel/2_dataset_memmap.py \
    --memmap_dir ../panel/memmap_data \
    --save_dir ./runs/my_experiment \
    --use_lambdarank \
    --use_cqb \
    --use_gating \
    --exclude_features "^fund" \
    --lookback 60 \
    --k 512 \
    --batch_size 4 \
    --epochs 50 \
    --patience 10 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --dropout 0.3
```

**关键参数**:
- `--use_lambdarank`: 启用 LambdaRank NDCG 排序损失
- `--use_cqb`: 启用 CQB 自适应任务平衡
- `--use_gating`: 启用输入特征门控
- `--exclude_features "^fund"`: 排除基金相关特征
- `--patience 10`: 早停耐心（10个epoch无改善则停止）

**训练监控**:
- 每个 epoch 打印 Train/Val 的 IC, RankIC, ICIR, 分类准确率等
- 自动保存最佳模型（基于 val RankIC）到 `runs/my_experiment/best.pt`

#### 3.2 生成测试报告

```bash
python generate_report_from_checkpoint.py \
    --checkpoint ./runs/my_experiment/best.pt \
    --dataset_py ../panel/2_dataset_memmap.py \
    --memmap_dir ../panel/memmap_data \
    --eval_splits "test" \
    --exclude_features "^fund"
```

**输出**: `runs/my_experiment/comprehensive_report.json`

报告包含:
- Test IC, RankIC, ICIR, RankICIR
- 分类准确率（总体 + 各类别）
- 特征门控值分布
- 模型配置信息

---

## 常见问题

### Q1: 内存不足 (OOM)

**解决方案**:
1. 减小 batch_size（例如从 4 降到 2）
2. 减小 k（每天采样的股票数，例如从 512 降到 256）
3. 使用 AMP 混合精度训练（默认已启用）
4. 在 `1_build_panel_memmap.py` 中使用 float16 存储特征

### Q2: Windows 多进程报错

**解决方案**: 
- 训练脚本中设置 `--num_workers 0`
- Dataset 已实现 `_lazy_open()` 延迟加载 memmap，避免 pickle 错误

### Q3: 训练很慢

**优化建议**:
1. 确保使用 GPU（检查 `torch.cuda.is_available()`）
2. 增大 batch_size（如果显存足够）
3. 减少 lookback（例如从 60 降到 30）
4. 使用 SSD 存储 memmap 数据

### Q4: Val RankIC 比 Train RankIC 高

**原因**: 
- Dropout 在训练时启用，评估时关闭
- 2015-2016 市场结构可能更有利于动量策略
- 这是正常现象，不是过拟合

### Q5: 如何调整数据划分年份？

**方法**:
```bash
python 4_train_stage2.py \
    --train_years "2011-2016" \
    --val_years "2017,2018" \
    --test_years "2019-2020" \
    ...
```

---

## 高级用法

### 特征选择

```bash
# 基于门控值进行特征选择
python 5_feature_selection.py \
    --checkpoint ./runs/my_experiment/best.pt \
    --threshold 0.5 \
    --output selected_features.txt

# 使用选定的特征重新训练
python 4_train_stage2.py \
    --feature_file selected_features.txt \
    ...
```

### 滚动窗口回测

```bash
python run_refit_rolling_window_test.py \
    --dataset_py ../panel/2_dataset_memmap.py \
    --memmap_dir ../panel/memmap_data \
    --output_dir ./runs/rolling_window \
    --train_window 8 \
    --test_window 1
```

### Refit Mode（无验证集）

```bash
python 4_train_stage2.py \
    --train_years "2007-2016" \
    --val_years "none" \
    --test_years "2017-2020" \
    --fixed_epochs 30 \
    ...
```

说明: Refit mode 将 train+val 合并训练，适合最终模型部署。

---

## 性能基准

典型配置（alpha158特征，排除fund，LambdaRank+CQB+Gating）:

| 指标 | Train | Val | Test |
|------|-------|-----|------|
| IC | ~0.03 | ~0.04 | ~0.025 |
| RankIC | ~0.02 | ~0.09 | ~0.07 |
| ICIR | ~1.5 | ~2.5 | ~1.8 |
| Cls Acc | ~87% | ~87% | ~87% |

**注**: 性能会因市场环境、特征工程、超参数而变化。

---

## 下一步

- 阅读 `model_training/STAGE2_README.md` 了解详细的训练配置
- 阅读 `model_training/LAMBDARANK_README.md` 了解 LambdaRank 原理
- 阅读 `model_training/FEATURE_SELECTION.md` 了解特征选择策略
- 查看 `runs/` 目录中的训练日志和报告

---

**祝训练顺利！** 🚀
