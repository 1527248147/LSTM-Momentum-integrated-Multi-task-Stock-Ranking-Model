# Core - MiM-StocR 核心流程文件

本文件夹包含 MiM-StocR (Momentum-integrated Multi-task Stock Recommendation) 项目的核心流程文件，已整理为清晰的三阶段结构。

## 📥 数据集下载

- `alpha158+ricequant_finance+lgbm` 数据集
- 链接: https://pan.baidu.com/s/1yIGTYrIe21nmIMGytfFmeg?pwd=6h8a
- 提取码: `6h8a`

## 📁 文件夹结构

```
Core/
├── label/              # 阶段1: 标签生成
├── panel/              # 阶段2: 数据面板构建
└── model_training/     # 阶段3: 模型训练与评估
```

---

## 🔄 完整流程

### 阶段 1: 标签生成 (`label/`)

**目标**: 为原始股票数据生成回归标签和动量分类标签

**核心文件**:
- `label_mimstocr_labels.py` - 主标签生成脚本
  - 生成 `label_ret_1d` (次日收益率，回归任务)
  - 生成 `label_mom_cls` (20日动量分类，0-4五分类)
  - 生成标签有效性标记 (`label_ret_valid`, `label_mom_valid`)

**输入**: 原始股票数据 Parquet 文件（包含日期、股票代码、收盘价等）

**输出**: 带标签的 Parquet 文件 (`alpha158_plus_fund_yearly_parquet_labeled_mimstocr/`)

**使用方法**:
```bash
python label/label_mimstocr_labels.py \
    --input_dir <原始数据目录> \
    --output_dir <输出目录>
```

---

### 阶段 2: 数据面板构建 (`panel/`)

**目标**: 将带标签的 Parquet 数据转换为高效的 memmap 格式面板数据

**核心文件**:
1. **`0_add_row_present.py`** - 添加 `row_present` 列（标记当日数据是否存在，避免使用未来信息）
   - 批处理: `0_add_row_present.bat`

2. **`1_build_panel_memmap.py`** - 构建 memmap 格式面板数据
   - 批处理: `1_build_panel_memmap.bat`
   - 输出结构:
     - `X_f16.mmap`: 特征矩阵 [T, N, D], D = 2F+1 (特征 + isna + row_present)
     - `y_ret_f32.mmap`: 回归标签 [T, N]
     - `y_mom_i8.mmap`: 分类标签 [T, N]
     - `ret_mask_u8.mmap`, `mom_mask_u8.mmap`: 标签有效性掩码
     - `meta.json`: 元数据（日期、股票列表、特征名等）

3. **`2_dataset_memmap.py`** - PyTorch Dataset 数据加载器
   - 支持滑动窗口采样 (lookback=60)
   - 支持多种采样模式 (`sample_present_only`, `sample_ret_valid_only`)
   - 内存安全的 memmap 读取（避免OOM）

**文档**:
- `MEMMAP_USAGE_GUIDE.md` - memmap使用指南
- `MEMORY_SAFETY_GUIDE.md` - 内存安全指南

**使用方法**:
```bash
# Step 1: 添加 row_present 列
python panel/0_add_row_present.py \
    --data_dir <标签数据目录> \
    --output_dir <输出目录>

# Step 2: 构建 memmap
python panel/1_build_panel_memmap.py \
    --parquet_dir <带row_present的数据目录> \
    --output_dir panel/memmap_data

# Step 3: 测试数据加载 (可选)
python panel/2_test_memmap_dataset.bat
```

**输出**: `panel/memmap_data/` 包含所有 memmap 文件和元数据

---

### 阶段 3: 模型训练与评估 (`model_training/`)

**目标**: 训练 LSTM 多任务模型，评估预测性能

#### 核心模型文件

- **`model_lstm_mtl.py`** - LSTM 多任务学习模型
  - 输入特征门控 (Input Feature Gating)
  - 双任务头: 回归 (收益率预测) + 分类 (动量分类)
  - LayerNorm + Dropout 正则化

- **`loss_adaptivek_approxndcg.py`** - ApproxNDCG 排序损失（自适应 k）

#### 训练脚本

1. **`3_train_stage1.py`** - Stage 1 预训练（可选）
   - 简单 MSE + CE 损失
   - 不使用 CQB / LambdaRank

2. **`4_train_stage2.py`** - **Stage 2 主训练脚本** ⭐
   - **CQB (Cycle-based Quad-Balancing)**: 自适应任务平衡
     - 根据验证集 V_n 动态调整 β_r, β_c, weight_decay
     - Eq.(13)-(15): V_n = ΔL_val / ΔL_train
     - β_n = β₀^(sigmoid(V_n))
   - **LambdaRank NDCG Loss**: 排序优化
     - 使用 ΔNDCG@k 权重的成对损失
     - 将连续收益率转换为离散相关性等级 (5-bins)
   - **Input Feature Gating**: L1正则化的特征门控
   - **早停**: 基于 val RankIC
   - **AMP**: 混合精度训练（节省显存）

#### 辅助脚本

- **`5_feature_selection.py`** - 基于门控值的特征选择
- **`generate_report_from_checkpoint.py`** - 从检查点生成测试报告
- **`comprehensive_report.py`** - 生成综合性能报告
- **`run_refit_rolling_window_test.py`** - 滚动窗口回测

#### 批处理示例

- `train_stage2_lambdarank_alpha158.bat` - 训练示例
- `generate_report.bat` - 报告生成示例
- `run_feature_selection.bat` - 特征选择示例

#### 文档

- `STAGE2_README.md` - Stage2 训练详细说明
- `LAMBDARANK_README.md` - LambdaRank 原理与实现
- `REPORT_GENERATION_README.md` - 报告生成指南
- `FEATURE_SELECTION.md` - 特征选择指南

#### 使用方法

```bash
# 训练 Stage2（完整流程）
python model_training/4_train_stage2.py \
    --dataset_py ../panel/2_dataset_memmap.py \
    --memmap_dir ../panel/memmap_data \
    --save_dir runs/my_experiment \
    --use_lambdarank \
    --use_cqb \
    --use_gating \
    --exclude_features "^fund" \
    --epochs 50 \
    --patience 10

# 生成测试报告
python model_training/generate_report_from_checkpoint.py \
    --checkpoint runs/my_experiment/best.pt \
    --dataset_py ../panel/2_dataset_memmap.py \
    --memmap_dir ../panel/memmap_data \
    --eval_splits "test"
```

---

## 📊 关键指标

- **IC (Information Coefficient)**: Pearson 相关系数 (预测 vs 真实收益)
- **RankIC**: Spearman 秩相关系数（更稳健）
- **ICIR / RankICIR**: IC / RankIC 除以其标准差（夏普比率的类比）
- **Classification Accuracy**: 动量分类准确率
- **NDCG@k**: 排序质量（归一化折损累计增益）

---

## 🎯 最终结果

基于 Alpha158 特征集（2007-2020年中国A股市场）的完整训练结果：

**数据集划分**:
- 训练集: 2007-2014 (8年)
- 验证集: 2015-2016 (2年)
- 测试集: 2017-2020 (4年)

**性能表现**:
- **验证集 (2015-2016)**: RankIC ≈ **0.09**
- **测试集第一年 (2017)**: RankIC ≈ **0.07**

**说明**:
- 验证集用于早停和超参数选择，测试集为完全未见数据
- RankIC > 0.05 在中国A股市场被认为是具有实用价值的预测能力
- 2017年测试集RankIC略低可能与市场结构变化相关（2015-2016年股灾后监管环境变化）

---

## 🔑 关键技术

1. **CQB (Cycle-based Quad-Balancing)** - 自适应多任务平衡
   - 无需手动调参，自动根据验证集表现调整任务权重
   - 论文 Eq.(10)-(15), Eq.(21)

2. **LambdaRank + NDCG Loss** - 排序优化
   - 直接优化股票排序质量（投资组合构建的核心）
   - ΔNDCG 加权的成对损失

3. **Input Feature Gating** - 可解释性特征选择
   - 学习特征重要性权重
   - L1 正则化促进稀疏性

---

## ⚠️ 注意事项

1. **数据泄露预防**:
   - 使用 `sample_present_only=True`（仅用当日 row_present 采样，无未来信息）
   - 特征归一化必须在面板构建时完成（逐日截面标准化）
   - 标签构造不使用未来信息

2. **内存管理**:
   - memmap 延迟加载（`_lazy_open()`）
   - Windows 多进程需设置 `num_workers=0`
   - 大数据集建议 float16 存储特征

3. **训练稳定性**:
   - LambdaRank loss 需要 numerical clipping (score_diff, weights)
   - CQB 需要至少 2b=12 epochs 历史才启用
   - 早停基于 RankIC（而非 loss）
---

## 🎯 快速开始

完整流程三步走：

```bash
# 1. 生成标签
cd label
python label_mimstocr_labels.py --input_dir <数据路径> --output_dir <输出路径>

# 2. 构建面板
cd ../panel
python 0_add_row_present.py --data_dir <标签数据> --output_dir <输出路径>
python 1_build_panel_memmap.py --parquet_dir <上一步输出> --output_dir memmap_data

# 3. 训练模型
cd ../model_training
python 4_train_stage2.py \
    --dataset_py ../panel/2_dataset_memmap.py \
    --memmap_dir ../panel/memmap_data \
    --save_dir runs/experiment_1 \
    --use_lambdarank --use_cqb --use_gating \
    --exclude_features "^fund"
```

训练完成后，检查点保存在 `runs/experiment_1/best.pt`，报告保存在 `runs/experiment_1/comprehensive_report.json`。

---

**版本**: 2026-02  
**维护者**: 孙杰一
