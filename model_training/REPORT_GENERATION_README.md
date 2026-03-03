# Generate Report from Checkpoint

快速从已有checkpoint生成comprehensive_report.json，无需重新训练。

## 📁 文件说明

- **generate_report_from_checkpoint.py** - Python脚本，核心功能
- **generate_report.bat** - 快捷批处理，使用默认checkpoint路径
- **generate_report_custom.bat** - 自定义checkpoint路径版本

## 🚀 使用方法

### 方法1：使用默认路径（推荐）

```bash
generate_report.bat
```

自动加载 `runs/stage2_lambdarank_alpha158/best.pt` 并生成报告。

### 方法2：自定义checkpoint路径

```bash
generate_report_custom.bat runs/your_run_name/best.pt
```

### 方法3：直接调用Python脚本

```bash
python generate_report_from_checkpoint.py \
  --checkpoint runs/stage2_lambdarank_alpha158/best.pt \
  --dataset_py ../panel/2_dataset_memmap.py \
  --memmap_dir ../panel/memmap_data \
  --lookback 60 \
  --k 512 \
  --batch_size 4 \
  --exclude_features "^fund" \
  --seed 42 \
  --use_lambdarank \
  --use_gating \
  --use_cqb
```

## 📊 输出

脚本会生成：
- **comprehensive_report.json** - 完整训练报告（保存在checkpoint同目录）

报告内容包括：
1. **实验信息**：配置、设备、数据路径
2. **数据统计**：训练/验证/测试集信息
3. **Loss分解**：各loss组件的权重
4. **核心指标**：
   - **Train集**：IC, ICIR, RankIC, RankICIR, 分类准确率等
   - **Val集**：验证集指标
   - **Test集**：测试集指标
5. **CQB状态**：动态平衡状态
6. **Gating结果**：特征重要性排序

## 📈 输出示例

```
================================================================================
SUMMARY
================================================================================
Checkpoint: Epoch 38

IC / ICIR:
  Train:  0.1234 /  1.234
  Val:    0.1001 /  1.001
  Test:   0.0862 /  0.774

RankIC / RankICIR:
  Train:  0.0567 /  0.567
  Val:    0.0489 /  0.489
  Test:   0.0409 /  0.421

Classification Accuracy:
  Train: 96.34%
  Val:   95.12%
  Test:  94.89%

✓ Complete! See runs/stage2_lambdarank_alpha158/comprehensive_report.json for full details.
```

## ⚡ 优势

1. **快速**：无需重新训练，直接评估
2. **完整**：包含Train/Val/Test三个数据集指标
3. **灵活**：随时可以重新生成报告
4. **便捷**：一键运行.bat文件

## 🔧 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必需 | checkpoint文件路径（best.pt） |
| `--dataset_py` | `../panel/2_dataset_memmap.py` | 数据集模块路径 |
| `--memmap_dir` | `../panel/memmap_data` | Memmap数据目录 |
| `--lookback` | 60 | 历史窗口长度 |
| `--k` | 512 | 每日采样股票数 |
| `--batch_size` | 4 | 评估批次大小 |
| `--exclude_features` | `^fund` | 排除特征正则表达式 |
| `--seed` | 42 | 随机种子 |
| `--use_lambdarank` | flag | 是否使用LambdaRank |
| `--use_gating` | flag | 是否使用特征门控 |
| `--use_cqb` | flag | 是否使用CQB |

## 📝 注意事项

1. **checkpoint路径**：确保best.pt存在
2. **数据集路径**：确保memmap_data目录存在
3. **特征过滤**：确保exclude_features与训练时一致
4. **评估时间**：取决于数据集大小，通常1-3分钟

## 🎯 典型使用场景

1. **训练中断**：快速查看当前best checkpoint性能
2. **对比实验**：对比不同checkpoint的指标
3. **过拟合检测**：查看Train/Val/Test指标差异
4. **论文写作**：获取完整实验数据

## 🐛 故障排查

**问题：找不到checkpoint**
```
解决：检查checkpoint路径是否正确，确保best.pt存在
```

**问题：找不到dataset模块**
```
解决：检查dataset_py路径，确保2_dataset_memmap.py存在
```

**问题：内存不足**
```
解决：降低batch_size参数（例如改为2或1）
```

**问题：特征维度不匹配**
```
解决：确保exclude_features与训练时一致
```
