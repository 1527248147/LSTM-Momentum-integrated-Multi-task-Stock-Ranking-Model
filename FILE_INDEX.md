# Core 文件清单

本文档列出 Core 文件夹中所有文件及其用途。

## 📁 label/ - 标签生成阶段

| 文件名 | 类型 | 用途 | 依赖 |
|--------|------|------|------|
| `label_mimstocr_labels.py` | Python脚本 | 为原始股票数据生成回归标签（次日收益率）和分类标签（20日动量） | pandas, numpy |

**输入**: 原始 Parquet 数据（包含 OHLCV 等）  
**输出**: 带标签的 Parquet 数据

---

## 📁 panel/ - 数据面板构建阶段

| 文件名 | 类型 | 用途 | 依赖 |
|--------|------|------|------|
| `0_add_row_present.py` | Python脚本 | 添加 `row_present` 列（标记当日数据存在性，防止数据泄露） | pandas, pyarrow |
| `0_add_row_present.bat` | 批处理 | 运行 `0_add_row_present.py` 的快捷方式 | - |
| `1_build_panel_memmap.py` | Python脚本 | 将 Parquet 数据转换为 memmap 格式面板数据 | pandas, numpy, pyarrow |
| `1_build_panel_memmap.bat` | 批处理 | 运行 `1_build_panel_memmap.py` 的快捷方式 | - |
| `2_dataset_memmap.py` | Python模块 | PyTorch Dataset 和 DataLoader，支持滑动窗口、K采样、特征选择 | torch, numpy |
| `2_test_memmap_dataset.bat` | 批处理 | 测试 memmap 数据加载是否正常 | - |
| `MEMMAP_USAGE_GUIDE.md` | 文档 | memmap 使用指南（数据结构、读取方式等） | - |
| `MEMORY_SAFETY_GUIDE.md` | 文档 | 内存安全指南（避免 OOM、延迟加载等） | - |

**输入**: 带标签且有 `row_present` 列的 Parquet 数据  
**输出**: `memmap_data/` 文件夹，包含:
- `X_f16.mmap`: 特征 [T, N, D]
- `y_ret_f32.mmap`: 回归标签 [T, N]
- `y_mom_i8.mmap`: 分类标签 [T, N]
- `ret_mask_u8.mmap`, `mom_mask_u8.mmap`, `both_mask_u8.mmap`: 标签有效性掩码
- `meta.json`: 元数据

---

## 📁 model_training/ - 模型训练与评估阶段

### 核心训练脚本

| 文件名 | 类型 | 用途 | 关键参数 |
|--------|------|------|---------|
| `3_train_stage1.py` | Python脚本 | Stage 1 预训练（简单 MSE+CE 损失，无 CQB/LambdaRank） | `--dataset_py`, `--memmap_dir`, `--save_dir` |
| `4_train_stage2.py` | **主训练脚本** | Stage 2 主训练（CQB + LambdaRank + Gating） | `--use_cqb`, `--use_lambdarank`, `--use_gating`, `--exclude_features` |

### 模型与损失

| 文件名 | 类型 | 用途 | 说明 |
|--------|------|------|------|
| `model_lstm_mtl.py` | Python模块 | LSTM多任务模型定义（InputFeatureGating + 双任务头） | 包含 `LSTMMTLConfig`, `LSTMMultiTask` 类 |
| `loss_adaptivek_approxndcg.py` | Python模块 | ApproxNDCG 排序损失（自适应 k） | 用于分类任务的 listwise ranking loss |

### 辅助脚本

| 文件名 | 类型 | 用途 | 输入/输出 |
|--------|------|------|----------|
| `5_feature_selection.py` | Python脚本 | 基于门控值的特征选择 | 输入: checkpoint; 输出: 选定特征列表 |
| `generate_report_from_checkpoint.py` | Python脚本 | 从检查点生成测试报告 | 输入: `best.pt`; 输出: `comprehensive_report.json` |
| `comprehensive_report.py` | Python模块 | 生成综合性能报告的辅助函数 | 被 `generate_report_from_checkpoint.py` 调用 |
| `run_refit_rolling_window_test.py` | Python脚本 | 滚动窗口回测 | 在多个测试窗口上评估模型表现 |

### 批处理示例

| 文件名 | 类型 | 用途 | 说明 |
|--------|------|------|------|
| `train_stage2_lambdarank_alpha158.bat` | 批处理 | Stage2 训练示例（LambdaRank + alpha158特征） | 可作为训练脚本模板 |
| `generate_report.bat` | 批处理 | 报告生成示例 | 从 `best.pt` 生成测试报告 |
| `run_feature_selection.bat` | 批处理 | 特征选择示例 | 基于门控值筛选特征 |

### 文档

| 文件名 | 类型 | 内容概要 |
|--------|------|---------|
| `STAGE2_README.md` | 文档 | Stage2 训练详细说明：CQB、LambdaRank、早停策略、超参数等 |
| `LAMBDARANK_README.md` | 文档 | LambdaRank 原理、实现细节、数值稳定性处理 |
| `REPORT_GENERATION_README.md` | 文档 | 报告生成指南：如何解读报告、指标含义、诊断建议 |
| `FEATURE_SELECTION.md` | 文档 | 特征选择策略：门控值分析、阈值选择、重要性排序 |

---

## 💡 核心流程依赖关系

```
label_mimstocr_labels.py (label/)
    ↓ (生成带标签的 Parquet)
0_add_row_present.py (panel/)
    ↓ (添加 row_present 列)
1_build_panel_memmap.py (panel/)
    ↓ (转换为 memmap 格式)
memmap_data/ (输出)
    ↓ (用于训练)
4_train_stage2.py (model_training/)
    ↓ (依赖)
    ├─ model_lstm_mtl.py (模型定义)
    ├─ loss_adaptivek_approxndcg.py (损失函数)
    └─ 2_dataset_memmap.py (数据加载)
    ↓ (输出 checkpoint)
best.pt
    ↓ (用于评估)
generate_report_from_checkpoint.py
    ↓ (输出报告)
comprehensive_report.json
```

---

## 📊 关键技术实现位置

| 技术 | 实现文件 | 关键函数/类 |
|------|---------|-----------|
| **CQB自适应平衡** | `4_train_stage2.py` | `CQBState`, `compute_Vn()`, `beta_from_V()`, `weight_decay_from_V()` |
| **LambdaRank NDCG Loss** | `4_train_stage2.py` | `lambdarank_ndcg_loss()`, `returns_to_relevance()` |
| **Input Feature Gating** | `model_lstm_mtl.py` | `InputFeatureGating` 类 |
| **ApproxNDCG Loss** | `loss_adaptivek_approxndcg.py` | `approx_ndcg_loss_batch()` |
| **Memmap 数据加载** | `2_dataset_memmap.py` | `MemmapDayWindowDataset`, `_lazy_open()` |
| **IC/RankIC 计算** | `4_train_stage2.py` | `compute_ic_rankic_batch()` |
| **早停策略** | `4_train_stage2.py` | 基于 val RankIC，`patience=10` |

---

## 🔑 重要配置文件

虽然没有独立的配置文件，但关键参数通过命令行传递。推荐的配置组合：

### 标准训练配置

```bash
python 4_train_stage2.py \
    --dataset_py ../panel/2_dataset_memmap.py \
    --memmap_dir ../panel/memmap_data \
    --save_dir ./runs/experiment_name \
    --use_lambdarank \
    --lambdarank_k 50 \
    --lambdarank_sigma 0.5 \
    --lambdarank_bins 5 \
    --use_cqb \
    --beta0 0.5 \
    --b_win 6 \
    --use_gating \
    --gate_lambda_init 0.001 \
    --exclude_features "^fund" \
    --lookback 60 \
    --k 512 \
    --batch_size 4 \
    --epochs 50 \
    --patience 10 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --dropout 0.3 \
    --embed_dim 128 \
    --hidden_size 256 \
    --num_layers 2
```

---

## ⚠️ 数据文件说明

**Core 文件夹中不包含以下大文件**（需单独存储）：
- 原始 Parquet 数据
- 面板 memmap 数据 (`memmap_data/`)
- 训练日志与检查点 (`runs/`)

这些文件应存储在项目根目录或专门的数据目录中。

---

## 📝 版本信息

- **创建日期**: 2026-02
- **适用版本**: MiM-StocR v1.0
- **论文**: arXiv:2509.10461v2

---

## 🔗 快速链接

- **快速开始**: 参见 `QUICKSTART.md`
- **完整流程**: 参见 `README.md`
- **技术细节**: 参见各阶段文档（`STAGE2_README.md`, `LAMBDARANK_README.md` 等）
