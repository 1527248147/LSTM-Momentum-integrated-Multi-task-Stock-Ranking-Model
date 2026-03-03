# Stage 2 Training: CQB + Adaptive-k ApproxNDCG

基于论文 arXiv:2509.10461v2 的完整实现。

## 关键改进 vs Stage 1

### 1. 分类损失 (Eq.9)
```
L_cls = lambda_ce * CE + (1 - lambda_ce) * L_NDCG
```
- `lambda_ce = 0.5` (固定，论文推荐)
- CE: 交叉熵，`ignore_index=-1` 处理无效标签
- L_NDCG: Adaptive-k ApproxNDCG 损失

### 2. Adaptive-k ApproxNDCG (Eq.3-8)

**核心思想**: 动态确定 top-k 截断位置
- `tau = 20%` 股票池大小 (最小阈值)
- 从高到低逐组加入 momentum 标签 (4→0)，直到 k ≥ tau
- 不拆分同一组别 (保持完整性)

**NDCG 计算**:
```python
# Eq.5-6: 平滑排名近似
rank_i = 1 + Σ_{j≠i} sigmoid((s_j - s_i) / temp)

# Eq.3-4: DCG
gain = 2^relevance - 1
discount = 1 / log2(1 + rank)
DCG = Σ gain * discount * topk_gate

# Eq.8: 损失
L_NDCG = exp(-NDCG)
```

### 3. CQB (Cycle-based Quad-Balancing)

**目标**: 平衡回归和分类任务的梯度冲突

#### 3.1 EMA 平滑 (Eq.10)
```
ḡ_r^n = β_r^n * ḡ_r^{n-1} + (1 - β_r^n) * g_r^n
ḡ_c^n = β_c^n * ḡ_c^{n-1} + (1 - β_c^n) * g_c^n
```

#### 3.2 L2归一化 + 幅度平衡 (Eq.11-12)
```
u_r = ḡ_r / ||ḡ_r||_2
u_c = ḡ_c / ||ḡ_c||_2
α = max(||ḡ_r||_2, ||ḡ_c||_2)
g_shared = α * (u_r + u_c)
```

#### 3.3 自适应遗忘率 beta (Eq.13-15)
```
# 验证损失 vs 训练损失的变化率比值
V_n = ΔL_valid / ΔL_train
ΔL = L_{n-1} - mean([L_{n-2b}, ..., L_{n-b-1}])

# 自适应调整 beta
β_n = β_0 ^ sigmoid(V_n)
```
- `b = 6` (窗口大小)
- `beta0 = 0.5` (初始遗忘率)
- 需要 2b=12 个 epoch 历史才开始计算 V_n

#### 3.4 正则化平衡 (Eq.21, 论文补充)
```
weight_decay_n = base_wd * sigmoid(-mean(V_{n-1}))
```
根据模型过拟合/欠拟合状态动态调整权重衰减。

## 文件说明

### 核心文件
1. **4_train_stage2.py** - Stage 2 主训练脚本
   - 集成 CQB + ApproxNDCG
   - 自适应 beta 和 weight_decay 调节
   - 完整的 IC/RankIC 评估

2. **loss_adaptivek_approxndcg.py** - ApproxNDCG 损失
   - `approx_ndcg_loss_one_day`: 单日计算
   - `approx_ndcg_loss_batch`: 批次平均
   - `_adaptive_k_from_mom`: adaptive-k 规则

3. **5_train_stage2.bat** - 完整训练启动脚本
   - 100 epochs, patience=10
   - 标准超参数设置

4. **5_train_stage2_quicktest.bat** - 快速测试 (5 epochs)
   - 验证代码正确性
   - 检查 CQB 指标 (V_n, beta_r, beta_c)

### 启动器配置

**完整训练 (5_train_stage2.bat)**:
```bash
lookback=60, k=512, batch_size=4
lambda_ce=0.5, tau_ratio=0.2
gate_l1_max=5e-3 (比 Stage1 更强的 gating)
beta0=0.5, b_win=6
lr=2e-4, weight_decay=1e-3
epochs=100, patience=10
embed_dim=128, hidden_size=256
```

**快速测试 (5_train_stage2_quicktest.bat)**:
```bash
epochs=5, k=256, batch_size=2
embed_dim=64, hidden_size=128 (更小模型)
其他参数同上
```

## 使用步骤

### 1. 快速验证
```cmd
cd model_training
5_train_stage2_quicktest.bat
```
检查:
- ✅ CQB 指标正常输出 (V_r, V_c, β_r, β_c)
- ✅ NDCG loss 收敛
- ✅ k/tau 值合理 (tau≈20%*valid_stocks, k≥tau)
- ✅ 无内存/梯度错误

### 2. 完整训练
```cmd
5_train_stage2.bat
```
监控:
- **前 12 epochs**: beta_r = beta_c = 0.5 (固定，积累历史)
- **12+ epochs**: beta 开始自适应调整
  - 如果 V > 0: 过拟合倾向 → 增大 beta (保留更多历史)
  - 如果 V < 0: 欠拟合倾向 → 减小 beta (更快响应新梯度)
- **weight_decay**: 根据 mean(V) 自动调节正则化强度
- **NDCG**: 应逐步提升 (0.3-0.5 良好范围)
- **IC/RankIC**: 最终测试集应 > 0.05

### 3. 继续训练
```cmd
python 4_train_stage2.py --resume_ckpt runs\stage2_cqb_adaptivek\best.pt [其他参数]
```

## 理论依据

### 为什么 CQB 有效？
1. **EMA 平滑**: 减少梯度噪声，稳定训练
2. **L2 归一化**: 消除任务梯度的尺度差异
3. **幅度平衡**: 保留重要信息（较大梯度的幅度）
4. **自适应 beta**: 根据泛化信号动态调整保守程度

### 为什么 Adaptive-k？
- **固定 k**: 可能包含过多噪声股票或遗漏潜力股
- **Adaptive-k**: 
  - 自动适应每日有效股票数量
  - 保持 momentum 分组完整性
  - tau=20% 保证最小覆盖范围

### 为什么 lambda_ce=0.5？
论文消融实验显示 0.5 在 CE 和 NDCG 之间取得最佳平衡:
- CE: 提供细粒度类别监督
- NDCG: 优化排序性能 (投资关键指标)

## 调试建议

### 如果 NDCG 不收敛:
1. 检查 `temp_rank` (默认 1.0，可尝试 0.5-2.0)
2. 检查 `temp_topk` (控制 top-k 截断平滑度)
3. 确认 momentum 标签分布 (不应全是 -1)

### 如果 beta 振荡:
1. 增大 `b_win` (6→8 或 10)
2. 调整 `beta0` (0.5→0.6 更保守)

### 如果过拟合严重:
1. 增大 `weight_decay` (1e-3→2e-3)
2. 增大 `dropout` (0.3→0.5)
3. 减小 `gate_l1_max` (让更多特征参与)

### 如果欠拟合:
1. 增大模型容量 (hidden_size 256→512)
2. 减小 weight_decay
3. 训练更多 epochs

## 预期结果

**Stage 1 baseline** (sample_ret_valid_only):
- Val IC: 0.03-0.05
- Test RankIC: 0.04-0.06
- Mom Acc: ~30-40% (更真实，不再 99%)

**Stage 2 CQB+ApproxNDCG** (目标提升):
- Val IC: 0.05-0.08 (+40-60%)
- Test RankIC: 0.06-0.10 (+50-67%)
- NDCG: 0.35-0.50
- Mom Acc: 35-45% (排序优先，准确率次要)

**关键指标优先级**:
1. **RankIC** (投资收益最相关)
2. **NDCG** (排序质量)
3. **IC** (线性相关)
4. Mom Acc (分类准确率，参考价值有限)

## 参考

- 论文: Multi-task Learning for Stock Recommendation (arXiv:2509.10461v2)
- CQB: Eq.(10)-(15)
- ApproxNDCG: Eq.(3)-(9)
- Regularization balancing: Eq.(21)

---

**开始训练**: 运行 `5_train_stage2_quicktest.bat` 验证 → `5_train_stage2.bat` 完整训练

**预计时间**: 
- Quick test: ~10-15分钟 (5 epochs)
- Full training: ~8-12小时 (100 epochs, RTX 5080)
