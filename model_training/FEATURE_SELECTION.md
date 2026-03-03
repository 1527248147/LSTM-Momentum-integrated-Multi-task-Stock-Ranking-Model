# 特征选择功能使用说明

## 功能介绍

现在可以在训练Stage2时通过命令行参数动态选择要使用的特征，无需重新构建memmap数据。

## 参数说明

### `--exclude_features <pattern>`
使用正则表达式排除匹配的特征

### `--include_features <pattern>`
使用正则表达式只包含匹配的特征

## 使用示例

### 1. 排除所有fund特征（只用alpha158）

```bash
python 4_train_stage2.py \
  --dataset_py ../panel/2_dataset_memmap.py \
  --memmap_dir ../panel/memmap_data \
  --exclude_features "^fund" \
  --use_gating \
  --epochs 50 \
  --save_dir runs/stage2_alpha158_only
```

**说明**：`^fund` 匹配所有以"fund"开头的特征

### 2. 只用fund特征

```bash
python 4_train_stage2.py \
  --dataset_py ../panel/2_dataset_memmap.py \
  --memmap_dir ../panel/memmap_data \
  --include_features "^fund" \
  --use_gating \
  --epochs 50 \
  --save_dir runs/stage2_fund_only
```

### 3. 排除多类特征（正则表达式或）

```bash
python 4_train_stage2.py \
  --exclude_features "fund|dividend|ILLIQUIDITY" \
  ...
```

**说明**：排除包含 "fund" 或 "dividend" 或 "ILLIQUIDITY" 的特征

### 4. 只用特定前缀的特征

```bash
python 4_train_stage2.py \
  --include_features "^(OPEN|CLOSE|HIGH|LOW|VOLUME|VWAP)" \
  ...
```

**说明**：只使用OHLCV基础价格特征

### 5. 组合使用（先include再exclude）

```bash
python 4_train_stage2.py \
  --include_features "^fund" \
  --exclude_features "dividend" \
  ...
```

**说明**：使用所有fund特征，但排除包含"dividend"的

## 特征选择原理

1. **读取meta.json**：从memmap数据的meta.json获取所有特征名列表
2. **应用过滤规则**：
   - 如果指定 `--exclude_features`：排除匹配的特征
   - 如果指定 `--include_features`：只保留匹配的特征
   - 两者都指定时：先include再exclude
3. **维度映射**：自动处理对应的isna列和row_present列
   - X结构：[features(F), isna(F), row_present(1)] = 2F+1
   - 选择特征时会同时选择对应的isna列
4. **模型适配**：自动调整模型输入维度D_in

## 输出信息

运行时会显示详细的特征选择信息：

```
================================================================================
Feature Selection
================================================================================
Original features: 466
Original D (total dim): 933

Exclude pattern: '^fund'
  Excluded 158 features

✓ Selected features: 308 / 466
  First 5: ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2']
  Last 5: ['VSUMP', 'RSI', 'BETA', 'ILLIQUIDITY', 'ATR']

✓ Input dimension adjusted: D = 617 (was 933)
  Structure: 308 features + 308 isna + 1 row_present
```

## 注意事项

1. **正则表达式语法**：使用Python re模块语法，默认不区分大小写
2. **特征名来源**：特征名从 `memmap_data/meta.json` 的 `feat_cols` 字段读取
3. **Gate数量**：模型的gate数量会自动匹配选择后的特征数
4. **特征重要性**：输出的feature_importance.csv会显示选择后的特征名
5. **内存优势**：虽然memmap仍包含所有特征，但只有选中的特征会被实际加载和计算

## 常用正则表达式模式

| 模式 | 说明 | 示例 |
|------|------|------|
| `^fund` | 以fund开头 | fund_net_value, fund_nav |
| `fund$` | 以fund结尾 | total_fund |
| `fund` | 包含fund | fund_x, some_fund |
| `^(A\|B)` | 以A或B开头 | A_feat, B_feat |
| `[0-9]` | 包含数字 | OPEN0, MA5 |
| `(?!fund)` | 不包含fund（负向前瞻） | 复杂，建议用--exclude |

## 推荐工作流

1. **第一轮**：用所有特征训练，查看gate值
2. **分析**：检查 `feature_importance.csv`，找出低gate值特征
3. **第二轮**：排除低重要性特征，重新训练
4. **对比**：使用 `comprehensive_report.json` 对比性能

## 快速测试

运行测试脚本验证功能：

```bash
# Windows
test_exclude_fund.bat

# Linux/Mac
./test_exclude_fund.sh
```

这会训练1个epoch，快速验证特征选择是否正常工作。
