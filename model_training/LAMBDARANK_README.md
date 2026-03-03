# LambdaRank Integration for Stage2 Training

## 概述

在Stage2训练中添加了**LambdaRank NDCG损失**作为MSE的替代方案，用于优化收益预测任务的排序性能。

## 背景与动机

### 原有方法（MSE）的局限
- **MSE损失**关注的是"预测值与真实值的点对点误差"
- 在金融场景中，我们更关心"哪些股票排名靠前"而非精确的收益预测
- MSE对离群值敏感，可能导致模型过度关注极端收益的股票

### LambdaRank的优势
- **排序导向**：直接优化NDCG@k等排序指标
- **关注Top-K**：重点优化最重要的前K只股票
- **鲁棒性**：通过分位数离散化降低噪声影响
- **理论保证**：有明确的梯度推导和收敛性证明

## 实现细节

### 核心函数

#### 1. `lambdarank_ndcg_loss`
```python
def lambdarank_ndcg_loss(scores, rel, mask=None, k=50, sigma=1.0, eps=1e-8)
```

**参数：**
- `scores` (B, N): 模型预测的股票打分（越大越好）
- `rel` (B, N): 相关性等级（0..n_bins-1，由收益率离散化得到）
- `mask` (B, N): 有效性掩码（True=有效，False=停牌/缺失）
- `k`: NDCG@k中的k值（默认50）
- `sigma`: pairwise差异的缩放因子（默认1.0）

**原理：**
1. 计算当前排序下的NDCG权重
2. 构造所有rel_i > rel_j的股票对
3. 计算ΔNDCG（交换两个股票位置后NDCG的变化）
4. 用ΔNDCG加权pairwise logistic loss
5. 优化目标：让高相关性的股票排名更靠前

**数学公式：**
```
Loss = Σ(i,j) w_ij * log(1 + exp(-σ * (s_i - s_j)))
w_ij = |ΔNDCG_ij| / IDCG
```

#### 2. `returns_to_relevance`
```python
def returns_to_relevance(ret, mask=None, n_bins=5)
```

**参数：**
- `ret` (B, N): 未来收益率
- `mask` (B, N): 有效性掩码
- `n_bins`: 离散化的等级数（默认5）

**原理：**
- 按**每天截面**分别计算分位点
- 将收益率映射到0..n_bins-1的整数等级
- 这种离散化方式能自适应不同市场环境的收益分布

**为什么是每天分位数？**
- 股票收益在不同日期可能有不同的分布（波动率变化、市场环境）
- 使用全局分位数会让模型在高波动期和低波动期产生不同偏好
- 每天独立分位数确保每天都有均匀分布的0-4等级

### 与MSE对比

| 特性 | MSE | LambdaRank |
|------|-----|------------|
| 优化目标 | 最小化预测误差 | 最大化排序准确性 |
| 关注点 | 全部股票均等 | Top-K股票优先 |
| 对噪声 | 敏感（平方惩罚） | 鲁棒（分位数离散化） |
| 离群值 | 强烈惩罚 | 适度惩罚 |
| 可解释性 | 直观（预测vs真实） | 抽象（排序正确性） |
| 计算复杂度 | O(N) | O(N²) pairwise |

## 使用方法

### 命令行参数

添加了以下新参数：

```bash
--use_lambdarank              # 启用LambdaRank（否则使用MSE）
--lambdarank_k 50             # NDCG@k的k值（默认50）
--lambdarank_sigma 1.0        # pairwise缩放因子（默认1.0）
--lambdarank_bins 5           # 相关性等级数（默认5）
```

### 示例1：基础LambdaRank训练

```bash
train_stage2_lambdarank.bat
```

配置：
- 使用LambdaRank NDCG@50损失
- 5个相关性等级（0-4）
- 包含所有466个特征

### 示例2：LambdaRank + Alpha158特征

```bash
train_stage2_lambdarank_alpha158.bat
```

配置：
- 使用LambdaRank NDCG@50损失
- 排除fund特征（仅保留158个alpha158特征）
- 5个相关性等级

### 自定义调用

```bash
python 4_train_stage2.py \
  --dataset_py "2_dataset_memmap.py" \
  --memmap_dir "seq_meta" \
  --use_lambdarank \
  --lambdarank_k 100 \
  --lambdarank_bins 10 \
  --save_dir "runs/stage2_lambdarank_custom"
```

## 超参数调优建议

### `lambdarank_k` (NDCG@k的k值)
- **较小k (20-50)**：适合选股策略（只关心最优股票）
- **较大k (100-200)**：适合组合构建（需要更多候选）
- **推荐**：从k=50开始，观察验证集NDCG@k曲线

### `lambdarank_sigma` (pairwise缩放因子)
- **较小sigma (0.5-1.0)**：平滑梯度，稳定训练
- **较大sigma (2.0-5.0)**：锐化梯度，快速收敛（可能过拟合）
- **推荐**：保持默认1.0，除非发现收敛问题

### `lambdarank_bins` (相关性等级数)
- **较少bins (3-5)**：粗粒度分类，鲁棒但信息损失
- **较多bins (7-10)**：细粒度分类，信息丰富但可能引入噪声
- **推荐**：
  - bins=5：适合日频数据（与动量五分类对齐）
  - bins=10：适合高质量标签或特殊场景

## 测试与验证

### 运行单元测试

```bash
python test_lambdarank.py
```

测试内容：
1. 基本功能（返回值、形状、mask处理）
2. 梯度计算（可微分性）
3. 极端情况（完美排序、完全反序）

### 预期结果

```
Test 1: Basic functionality
Loss value: ~0.3-0.8 (随机排序)

Test 2: Perfect ranking
Loss (near-perfect ranking): ~0.001-0.01 (接近0)

Test 3: Reversed ranking  
Loss (reversed ranking): ~1.5-3.0 (很大)
```

## 训练过程监控

### Loss变化
- **L_reg**：LambdaRank损失（不再是MSE）
- 典型范围：0.1-1.0（取决于σ和k）
- 应该随训练下降

### 指标监控
LambdaRank虽然优化排序，但仍可监控：
- **IC/RankIC**：相关性指标（主要关注）
- **MSE**：点预测误差（参考）
- **Accuracy**：动量分类准确率

### 预期效果
- RankIC应该**显著提升**（相比MSE训练）
- IC可能略有下降（因为不直接优化点预测）
- 验证集Loss可能比训练集低（由于Dropout，正常现象）

## 与现有功能的兼容性

### ✅ 完全兼容
- CQB多任务平衡
- 特征选择（--exclude_features, --include_features）
- Gating特征门控
- 综合报告生成

### 🔄 部分调整
- `comprehensive_report.json`：
  - `mse_loss` → `lambdarank_ndcg_loss`
  - 添加`lambdarank_config`字段

### ❌ 不适用
- MSE相关的早停（需要基于RankIC或IC）

## 理论参考

1. **LambdaRank原论文**：
   Burges, C. J., et al. (2007). "Learning to Rank using Gradient Descent"

2. **LambdaLoss改进**：
   Wang, X., et al. (2018). "LambdaLoss: Differentiable Listwise Loss for Learning to Rank"

3. **NDCG指标**：
   Järvelin, K., & Kekäläinen, J. (2002). "Cumulated gain-based evaluation of IR techniques"

## 常见问题

### Q1: LambdaRank比MSE慢多少？
A: 约2-3倍（由于O(N²) pairwise计算）。但可通过减小k或采样来优化。

### Q2: 为什么验证集指标更好？
A: 因为模型使用了Dropout（训练时20%神经元被丢弃）。这是正常现象，不是bug。

### Q3: 如何选择bins数量？
A: 默认bins=5已经够用。增加bins不一定更好，可能引入噪声。

### Q4: 能否混合MSE和LambdaRank？
A: 目前不支持，但可以用`ret_w`调整回归任务权重：
```bash
--ret_w 0.5  # 降低回归任务权重
--cls_w 1.0  # 保持分类任务权重
```

### Q5: 训练不收敛怎么办？
A: 尝试：
1. 降低sigma（0.5）
2. 增大k（100）
3. 减少bins（3）
4. 降低学习率

## 后续改进方向

1. **计算优化**：
   - Top-k采样（只计算score最高的k对）
   - GPU并行化pairwise计算

2. **损失改进**：
   - Attention-based NDCG（动态k）
   - 自适应sigma调度

3. **混合训练**：
   - 前N个epoch用MSE预热
   - 后续切换到LambdaRank精调

## 总结

LambdaRank是**排序导向**的损失函数，特别适合：
- ✅ 选股策略（关注Top-K）
- ✅ 组合构建（关注排序准确性）
- ✅ 鲁棒训练（分位数离散化）

不太适合：
- ❌ 精确收益预测（MSE更合适）
- ❌ 极小数据集（O(N²)复杂度）
- ❌ 需要快速实验（计算较慢）

**推荐使用场景**：当你发现MSE训练的模型虽然预测误差小，但Top股票选择效果不佳时，尝试LambdaRank！
