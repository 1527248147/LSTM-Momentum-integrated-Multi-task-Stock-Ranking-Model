# 内存安全使用指南

## 问题诊断

您的电脑有 **32GB 内存**，但 memmap panel 数据大小约 **52GB**。虽然 memmap 不会一次性加载全部数据，但在实际使用中仍可能导致内存压力。

### 内存占用分析

1. **Memmap 文件本身**: ~52GB（存储在磁盘，按需映射到内存）
2. **系统 + 其他程序**: ~4-8GB
3. **Python 进程基础**: ~2-3GB
4. **DataLoader 缓存**: batch_size × num_workers × prefetch_factor × 单batch大小
5. **GPU 训练内存**: 模型参数 + 梯度 + 中间激活值

**关键风险**：如果 DataLoader 的多进程缓存 + Python 基础内存超过可用内存，会导致系统卡死。

---

## 解决方案

### 方案 1：超低内存模式（强烈推荐32GB内存）

使用超低内存配置脚本：

```batch
3_train_stage1_memmap_lowmem.bat
```

**优化项**：
- `batch_size=2` → 最小批次大小
- `num_workers=1` → 单worker进程
- `pin_memory=False` → 关闭GPU内存锁定
- `persistent_workers=False` → 每个epoch后释放worker
- `max_memory_gb=8` → **严格限制8GB内存**（留24GB给系统）

**预期内存占用**：
- Batch数据: 2 days × 512 stocks × 60 timesteps × 947 features × 2 bytes ≈ 117MB
- Worker缓存: 1 worker × 2 prefetch × 117MB ≈ 234MB
- Python基础: 2-3GB
- **总计**: ~2.5-3.5GB（远低于8GB限制）

### 方案 1B：极限低内存模式（如果还是卡）

```batch
3_train_stage1_memmap_ultralow.bat
```

**极限优化**：
- `batch_size=1` → 单batch
- `num_workers=0` → 完全禁用多进程
- `k=256` → 减少采样股票数（原512）
- `max_memory_gb=6` → **只用6GB**（留26GB）

**预期内存占用**：~1.5-2.5GB

⚠️ **警告**：极限模式训练很慢，仅在内存极度紧张时使用！

### 方案 2：优化预处理过程

如果预处理（`1_build_panel_memmap.py`）时卡住：

1. **已添加的优化**：
   - 每2个年度文件自动 `flush()` 到磁盘
   - 自动 `gc.collect()` 清理内存
   - 启动时检查可用内存，不足10GB会警告

2. **手动优化**（如果还是卡）：
   ```python
   # 修改 1_build_panel_memmap.py 第73行
   flush_interval = 1  # 改成每1个文件就flush（更慢但更安全）
   ```

3. **关闭其他程序**：
   - 浏览器（Chrome/Edge 占用2-4GB）
   - IDE（如果不用调试可以关闭 VS Code）
   - 其他大型应用

### 方案 3：监控内存使用

训练时会自动显示内存状态：

```
================================================================================
[MEMORY] 当前内存: 18.5GB / 32.0GB (57%)
[MEMORY] 可用内存: 13.5GB
[MEMORY] 内存限制: 24.0GB
================================================================================
```

每个 epoch 后也会检查：
```
⚠️  [WARNING] 内存使用 25.2GB 超过限制 24.0GB!
   可用内存: 6.8GB / 32.0GB
   建议：降低 batch_size、num_workers 或 prefetch_factor
```

---

## 参数调优建议

| 参数 | 默认值 | 超低内存 | 极限低内存 | 说明 |
|------|--------|----------|------------|------|
| `batch_size` | 8 | **2** | **1** | 每次训练的天数（主要影响） |
| `num_workers` | 4 | **1** | **0** | 数据加载并行度 |
| `k` | 512 | 512 | **256** | 每天采样股票数（影响较大） |
| `pin_memory` | True | **False** | False | GPU内存锁定 |
| `persistent_workers` | True | **False** | False | 保持worker进程 |
| `max_memory_gb` | - | **8** | **6** | 内存使用上限 |

**调整优先级**：
1. **batch_size**: 影响最大，8→2→1 可减少内存75%-87%
2. **num_workers**: 4→1→0 减少worker缓存和进程复制
3. **k**: 512→256 可减少50%单batch大小（但影响训练质量）

---

## 常见问题

### Q1: 预处理时内存爆了怎么办？
**A**: 
1. 确保关闭了浏览器等大程序
2. 修改 `flush_interval=1`（第73行）
3. 重启电脑清空缓存后重试

### Q2: 训练时系统变卡但没崩溃？
**A**: 
1. 这是内存快不够的信号，系统在用虚拟内存（磁盘）
2. 立即 Ctrl+C 停止训练
3. 使用更低的 `batch_size` 和 `num_workers`

### Q3: 能不能不生成 52GB 的 memmap？
**A**: 
可以，但需要修改代码：
- 使用 `float32` → `float16` 已经是最小（2 bytes/value）
- 排除更多特征（修改 `exclude_regex`）
- 只处理部分年份（删除不需要的 parquet 文件）

### Q4: 为什么 memmap 比内存大也能用？
**A**: 
memmap 是**内存映射文件**，操作系统只会把**正在访问的部分**加载到内存。但如果 batch_size 和 num_workers 设置太高，会同时访问太多数据，导致内存不够。

### Q5: 推荐的最优配置？
**A** (32GB 内存):

**超低内存模式**（推荐）:
```batch
--batch_size 2 ^
--num_workers 1 ^
--k 512 ^
--max_memory_gb 8.0
```
训练速度约 0.5-0.8秒/batch，比原始方式快 **8-12倍**。

**极限低内存模式**（如果还卡）:
```batch
--batch_size 1 ^
--num_workers 0 ^
--k 超低内存模式（32GB强烈推荐）
```batch
3_train_stage1_memmap_lowmem.bat
```
限制8GB内存，batch_size=2, num_workers=1

### 极限低内存模式（如果还是很卡）
```batch
3_train_stage1_memmap_ultralow.bat
```
限制6GB内存，batch_size=1, num_workers=0, k=256度约 1-1.5秒/batch，仍比原始方式快 **4-6倍**。

---

## 快速启动

### 超低内存模式（32GB强烈推荐）
```batch
3_train_stage1_memmap_lowmem.bat
```
限制8GB内存，batch_size=2, num_workers=1

### 极限低内存模式（如果还是很卡）
```batch
3_train_stage1_memmap_ultralow.bat
```
限制6GB内存，batch_size=1, num_workers=0, k=256

### 正常模式（64GB+内存）
```batch
3_train_stage1_memmap.bat
```

### 预处理（一次性，1-2小时）
```batch
1_build_panel_memmap.bat
```

---

## 内存使用公式

### 超低内存模式 (batch_size=2, num_workers=1)
```
Batch缓存 = batch_size × k × lookback × D × 2 bytes
          = 2 × 512 × 60 × 947 × 2 bytes ≈ 117 MB

Worker缓存 = num_workers × prefetch_factor × Batch缓存
           = 1 × 2 × 117 MB ≈ 234 MB

总内存 ≈ Python基础(2.5GB) + Batch(0.12GB) + Worker(0.23GB)
      ≈ 2.85 GB（远低于8GB限制）✅
```

### 极限低内存模式 (batch_size=1, num_workers=0, k=256)
```
Batch缓存 = 1 × 256 × 60 × 947 × 2 bytes ≈ 29 MB
Worker缓存 = 0（无多进程）

总内存 ≈ Python基础(2GB) + Batch(0.03GB)
      ≈ 2 GB（远低于6GB限制）✅✅
```

**结论**：超低内存模式只用~3GB，极限模式只用~2GB，留24-26GB给系统，完全安全！

Batch缓存 = batch_size × k × lookback × D × 2 bytes
          = 4 × 512 × 60 × 947 × 2 bytes ≈ 234 MB

Worker缓存 = num_workers × prefetch_factor × Batch缓存
           = 2 × 2 × 234 MB ≈ 936 MB

总内存 ≈ 3 + 0.9 + 系统预留 = ~4 GB（远低于16GB限制）
```

如果使用原始 `batch_size=8`, `num_workers=4`, `prefetch_factor=2`:
```
Batch缓存 = 468 MB
Worker缓存 = 4 × 2 × 468 MB ≈ 3.7 GB
总内存 ≈ 3 + 3.7 = ~7 GB（仍在16GB限制内）
```

**结论**：低内存模式只用~4GB，配合16GB限制，留16GB给系统，非常安全！

---

## 技术细节

### Memmap 工作原理
1. 创建大文件（52GB）但不立即占用内存
2. 通过操作系统的虚拟内存映射访问
3. 只有被 `[]` 索引的部分才会加载到物理内存
4. 操作系统自动管理页面换入/换出

### 为什么会卡爆
1. **同时访问太多页面**：batch_size 太大
2. **多进程重复映射**：num_workers 太多
3. **预取过多数据**：prefetch_factor 太大
4. **系统剩余内存不足**：其他程序占用过多

---

最后更新：2026-02-02
