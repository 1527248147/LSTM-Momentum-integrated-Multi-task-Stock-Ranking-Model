# Core 文件夹整理完成 ✅

## 整理概要

已将 MiM-StocR 项目的核心流程文件整理到 `Core/` 文件夹中，按照三个阶段组织：

- **label/** - 标签生成
- **panel/** - 数据面板构建  
- **model_training/** - 模型训练与评估

## 文件统计

| 文件夹 | 文件数 | 说明 |
|-------|-------|------|
| `label/` | 1 | 标签生成脚本 |
| `panel/` | 8 | 面板构建脚本 + 文档 |
| `model_training/` | 15 | 训练脚本 + 模型 + 文档 |
| **总计** | **29** | **所有核心流程文件** |

## 文档文件

Core 根目录包含以下文档：

1. **README.md** - 完整流程说明与技术概览（详细版）
2. **QUICKSTART.md** - 快速开始指南（简明版）
3. **FILE_INDEX.md** - 文件清单与依赖关系
4. **STRUCTURE.txt** - 目录结构树形图
5. **DIRECTORY_TREE.txt** - 完整文件路径列表

## 排除的文件

以下文件未包含在 Core/ 中（存储在原项目目录）：

### 数据文件（大文件）
- 原始 Parquet 数据
- 带标签的 Parquet 数据  
- memmap 格式面板数据 (`memmap_data/`)

### 运行时文件
- 训练日志和检查点 (`runs/`)
- Python 缓存 (`__pycache__/`)
- 中间文件和测试输出

### 辅助脚本
- 调试和测试脚本（非核心流程）
- 重复的批处理文件
- 实验性代码

## 使用方法

### 1. 快速上手
```bash
cd Core
# 阅读 QUICKSTART.md，按照三步走流程操作
```

### 2. 深入了解
```bash
# 查看完整流程说明
cat README.md

# 查看文件清单
cat FILE_INDEX.md

# 查看目录结构
cat STRUCTURE.txt
```

### 3. 开始训练
```bash
cd label
python label_mimstocr_labels.py ...  # 生成标签

cd ../panel  
python 0_add_row_present.py ...      # 添加 row_present
python 1_build_panel_memmap.py ...   # 构建面板

cd ../model_training
python 4_train_stage2.py ...         # 训练模型
```

## 核心技术

### 标签生成 (label/)
- ✅ 次日收益率回归标签
- ✅ 20日动量分类标签（5分类）
- ✅ 标签有效性掩码（过滤停牌、涨跌停等）

### 面板构建 (panel/)
- ✅ row_present 列（防止数据泄露）
- ✅ memmap 高效存储（避免OOM）
- ✅ 滑动窗口数据加载器
- ✅ 多种采样模式支持

### 模型训练 (model_training/)
- ✅ **CQB**: 自适应多任务平衡（论文 Eq.10-15, Eq.21）
- ✅ **LambdaRank**: ΔNDCG@k 排序损失
- ✅ **Input Gating**: 可解释性特征选择
- ✅ **早停**: 基于 val RankIC
- ✅ **AMP**: 混合精度训练

## 性能指标

典型配置下的性能（alpha158特征，LambdaRank+CQB+Gating）：

| 指标 | Train | Val | Test |
|------|-------|-----|------|
| IC | ~0.03 | ~0.04 | ~0.025 |
| RankIC | ~0.02 | ~0.09 | ~0.07 |
| ICIR | ~1.5 | ~2.5 | ~1.8 |

*注: 实际性能因市场环境和超参数而异*

## 数据流

```
原始数据 (Parquet)
    ↓ [label_mimstocr_labels.py]
带标签数据 (Parquet)
    ↓ [0_add_row_present.py]
带 row_present 数据 (Parquet)
    ↓ [1_build_panel_memmap.py]
memmap 面板数据
    ↓ [2_dataset_memmap.py + 4_train_stage2.py]
训练模型 (best.pt)
    ↓ [generate_report_from_checkpoint.py]
测试报告 (comprehensive_report.json)
```

## 依赖关系

- **Python**: 3.8+
- **PyTorch**: 1.12+
- **pandas**: 数据处理
- **numpy**: 数值计算
- **pyarrow**: Parquet 读写

## 下一步

1. **阅读文档**: 从 `QUICKSTART.md` 开始
2. **准备数据**: 确保有原始股票 Parquet 数据
3. **运行流程**: 按照三阶段执行
4. **调优模型**: 根据 `STAGE2_README.md` 调整超参数

## 联系方式

如有问题或建议，请参考：
- 论文: *MiM-StocR: Momentum-integrated Multi-task Stock Recommendation* (arXiv:2509.10461v2)
- 代码: 已上传到项目仓库

---

**整理日期**: 2026-02-07  
**版本**: v1.0  
**状态**: ✅ 完成
