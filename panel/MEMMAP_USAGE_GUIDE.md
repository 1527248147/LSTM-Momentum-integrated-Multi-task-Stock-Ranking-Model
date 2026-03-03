# Memmap 高速训练方案使用指南

⚠️ **内存重要提示**: 如果您的电脑有32GB内存，请先阅读 [MEMORY_SAFETY_GUIDE.md](MEMORY_SAFETY_GUIDE.md)

## 📊 方案对比

### 原始方案（2_dataset_day_grouped.py）
- **存储**: Yearly parquet files
- **训练时**: 每个step读取60天parquet → pandas过滤拼接 → numpy → torch
- **瓶颈**: 数据加载占97%时间，GPU仅3%利用率
- **速度**: ~6s/step, GPU利用率10-30%
- **num_workers**: Windows上必须=0（pickle问题）

### Memmap方案（本方案）
- **存储**: [T,N,D]面板memmap（一次性预处理）
- **训练时**: 直接slice窗口，操作系统自动缓存
- **优势**: 数据加载占<10%时间，GPU可达80%+利用率
- **预期速度**: ~0.3s/step（**20x加速**）
- **num_workers**: 支持多进程（Windows/Linux均可）

---

## 🚀 快速开始

**运行顺序**：按文件名前缀数字依次执行

### 方式1：一键运行（推荐新手，32GB内存）

```bash
cd panel
0_quick_start_memmap.bat
```

这会自动完成：预处理 → 测试 → 训练（低内存模式）

### 方式2：分步运行

#### Step 1: 预处理数据（一次性，可能需要1-2小时）

```bash
cd panel
1_build_panel_memmap.bat
```

这会创建 `panel/memmap_data/` 目录，包含：
- `X_f16.mmap`: [T,N,D] 特征面板（~52GB）
- `y_ret_f32.mmap`: [T,N] 回归标签
- `y_mom_i8.mmap`: [T,N] 分类标签
- `*_mask_u8.mmap`: 各种有效性mask
- `meta.json`: 元数据

**注意**: 预处理慢没关系，它只需要跑一次！之后训练会飞快。

#### Step 2: 测试数据（确保正常工作）

```bash
2_test_memmap_dataset.bat
```

应该看到：
```
[INFO] Dataset length: 101
  X shape: torch.Size([512, 60, 947])
  Batch 0: X=torch.Size([4, 512, 60, 947]), time=0.xxx s
  Average batch time: 0.xxx s
[DONE] Test passed!
```

#### Step 3: 开始训练

**32GB内存（推荐）**:
```bash
3_train_stage1_memmap_lowmem.bat
```

**64GB+内存**:
```bash
3_train_stage1_memmap.bat
```

**预期效果**:
```
Step 10/500 | Loss=0.xxxx | Speed=3.xx batch/s | 
⏱️data:0.5s(8%) gpu:5.2s(92%)
```

注意数据时间占比从97%降到<10%，GPU时间占比从3%升到90%+！

---

## ⚙️ 配置调优

### 内存安全配置

详细说明请查看 [MEMORY_SAFETY_GUIDE.md](MEMORY_SAFETY_GUIDE.md)

**32GB内存推荐配置**:
```bash
--batch_size 4           # 降低内存占用
--num_workers 2          # 减少worker进程
--max_memory_gb 16.0     # 内存警告阈值（留16GB给系统）
# 不加 --pin_memory      # 关闭GPU内存锁定
# 不加 --persistent_workers  # 每epoch后释放worker
```

**64GB+内存配置**:
```bash
--batch_size 8
--num_workers 4
--pin_memory
--persistent_workers
--prefetch_factor 2
```

### DataLoader参数调优

**调优建议**:
1. 先用低内存配置测试，观察内存使用
2. 如果内存充足，逐步增加 batch_size 和 num_workers
3. 如果看到内存警告，立即降低参数
4. Windows上通常2-4最优，Linux可以更多

### 内存优化

如果内存不够，可以：
1. 减小 `--k`（每日采样股票数）: 512→256
2. 减小 `--batch_size`: 8→4
3. 关闭 `--persistent_workers`

---

## 📈 性能对比

### 原始方案
```
Epoch 1/50: 80 min
  Step time: 6.1s (data:5.9s, gpu:0.2s)
  GPU util: 10-30%
  Bottleneck: pandas + parquet I/O
```

### Memmap方案（预期）
```
Epoch 1/50: 4-8 min (10-20x faster!)
  Step time: 0.3s (data:0.03s, gpu:0.27s)
  GPU util: 80-95%
  Bottleneck: GPU计算（合理）
```

---

## 🛠️ 故障排查

### 1. 预处理报错

**问题**: `KeyError: 'datetime'` 或列名错误
**解决**: 检查 `1_build_panel_memmap.py` 中的列名是否匹配你的parquet文件

### 2. 测试报错

**问题**: `FileNotFoundError: memmap_data/X_f16.mmap`
**解决**: 先运行 `1_build_panel_memmap.bat` 完成预处理

### 3. 训练慢/num_workers警告

**问题**: 速度没提升
**检查**:
- `--num_workers` 是否>0？Windows建议2-4
- 查看timing输出，data_time是否<10%？
- GPU是否被其他程序占用？（用 `nvidia-smi` 检查）

### 4. 内存不足

**问题**: `RuntimeError: [enforce fail at alloc_cpu.cpp:xxx]`
**解决**:
- 减小 `--batch_size`
- 减小 `--k`
- 关闭 `--persistent_workers`

---

## 📝 技术细节

### 为什么Memmap快？

1. **面板存储**: [T,N,D]连续内存，slice窗口只需简单索引
2. **操作系统缓存**: 常用数据自动缓存在内存，第二次访问几乎无延迟
3. **避免pandas**: 不再需要每次 `read_parquet() + groupby() + concat()`
4. **多进程友好**: memmap文件可以被多个进程共享（不需要pickle）

### 存储空间

```
T=5093, N=5426, D=947
X_f16: 5093 × 5426 × 947 × 2 bytes ≈ 52 GB
其他mask文件: ~1 GB
Total: ~53 GB
```

硬盘够的话完全可以接受（换来10-20x训练速度提升）

### 与原方案兼容性

- **模型**: 完全兼容，输入输出格式相同
- **数据**: 使用相同的parquet源数据
- **训练代码**: 只是dataset换了，其余逻辑一样

---

## 🎯 下一步优化

如果还想更快：

1. **SSD**: 把memmap_data放在SSD上（而不是机械硬盘）
2. **PyArrow**: 预处理阶段用PyArrow替代pandas（IO更快）
3. **混合精度**: 训练时用 `torch.cuda.amp`（GPU计算更快）
4. **模型并行**: 如果有多张GPU

但对于大多数情况，memmap方案已经足够快了！

---

## ✅ 检查清单

训练前确认：
- [ ] `panel/memmap_data/meta.json` 存在
- [ ] `2_test_memmap_dataset.bat` 测试通过
- [ ] `--num_workers` > 0（Windows建议2-4）
- [ ] GPU驱动正常（`nvidia-smi` 能看到GPU）
- [ ] 有足够硬盘空间（~60GB）

---

## 📋 文件清单

**Batch文件（一键执行）**：
- `0_quick_start_memmap.bat` - 🚀 一键运行全流程（推荐新手）

**分步执行**：
- `1_build_panel_memmap.bat` → 调用 `1_build_panel_memmap.py` - ⚙️ 预处理数据
- `2_test_memmap_dataset.bat` → 测试 `2_dataset_memmap.py` - 🧪 验证Dataset
- `3_train_stage1_memmap.bat` → 运行 `3_train_stage1_memmap.py` - 🎯 开始训练

**按数字顺序执行即可！**

---

**祝训练愉快！有问题随时查看这个文档。** 🎉
