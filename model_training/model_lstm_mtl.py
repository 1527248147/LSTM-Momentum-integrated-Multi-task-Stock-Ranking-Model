# -*- coding: utf-8 -*-
"""
model_lstm_mtl.py

LSTM 多任务模型：
- 回归头：预测次日收益 label_ret_1d
- 分类头：预测动量五分类 label_mom_cls
- 输入 gating：对“原始特征”维度做可学习门控 x' = g ⊙ x，并对 g 加 L1（用于自动稀疏化/筛特征）

输入 X_step 的结构（与你的数据管线一致）：
X_step = [features, isna_flags, row_present]
其中：
- features: F 维（截面标准化后）
- isna_flags: F 维（对应每个特征的缺失标记）
- row_present: 1 维（该日该股是否原始有记录）

所以总维度 D = 2F + 1

配对删除机制：
- 每个特征有一个 gate 权重 g_i ∈ (0,1)
- gate 同时作用于原始特征和对应的 isna 标记：
  * feat'_i = g_i × feat_i
  * isna'_i = g_i × isna_i
- 这样当 g_i → 0 时，特征和其缺失标记同时被抑制
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputFeatureGating(nn.Module):
    """
    对原始特征做门控：x' = sigmoid(logits) ⊙ x
    - logits 是可学习参数（每个特征一个）
    - 可提供 fixed_zero_mask：对某些特征强制 gate=0（例如 dividend 噪声因子）
    - 配对删除：同一个 gate 同时作用于原始特征和对应的 isna 标记
    """
    def __init__(
        self,
        num_features: int,
        init_logit: float = 2.0,               # sigmoid(2)=0.88，初期尽量不压特征
        fixed_zero_idx: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_features = int(num_features)

        logits = torch.full((self.num_features,), float(init_logit), dtype=torch.float32, device=device)
        self.gate_logits = nn.Parameter(logits)

        fixed_mask = torch.zeros((self.num_features,), dtype=torch.float32, device=device)
        if fixed_zero_idx is not None:
            for i in fixed_zero_idx:
                if 0 <= int(i) < self.num_features:
                    fixed_mask[int(i)] = 1.0
        # buffer：不会被优化器更新
        self.register_buffer("fixed_zero_mask", fixed_mask, persistent=True)

    def gates(self) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logits)  # [F] in (0,1)
        if self.fixed_zero_mask is not None and self.fixed_zero_mask.sum() > 0:
            # 被固定为 0 的维度：gate=0，且梯度也会被乘 0，等价于冻结
            g = g * (1.0 - self.fixed_zero_mask)
        return g

    def forward(self, x_feat: torch.Tensor) -> torch.Tensor:
        """
        x_feat: [N, L, F] or [*, F]
        """
        g = self.gates()
        # broadcast：最后一维对齐 F
        return x_feat * g

    def l1(self, reduction: str = "mean") -> torch.Tensor:
        """
        对 gate 做 L1，注意 gate 在 [0,1]，这里用 abs 只是形式一致
        reduction:
          - "mean": 与特征数无关，调参更稳（推荐）
          - "sum": 与 F 成正比（F 大时数值很大，不推荐）
        """
        g = self.gates().abs()
        if reduction == "sum":
            return g.sum()
        return g.mean()

    def l1_logit(self, reduction: str = "mean") -> torch.Tensor:
        """
        对 logit w 做 L1（而非对 sigmoid(w) 做 L1）。
        梯度不会被 sigmoid 饱和压扁，gate 更容易被推到负值区域（g→0）。
        L_gate = λ * mean(|w|)
        """
        w = self.gate_logits.abs()
        if self.fixed_zero_mask is not None and self.fixed_zero_mask.sum() > 0:
            w = w * (1.0 - self.fixed_zero_mask)
        if reduction == "sum":
            return w.sum()
        return w.mean()


@dataclass
class LSTMMTLConfig:
    input_dim: int                 # D = 2F+1
    raw_feature_dim: int           # F
    embed_dim: int = 128           # 先把 D 投影到 embed_dim，减轻 LSTM 计算
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 5           # 动量五分类
    use_layernorm: bool = True
    use_gating: bool = True
    gating_init_logit: float = 2.0


class LSTMMultiTask(nn.Module):
    def __init__(
        self,
        cfg: LSTMMTLConfig,
        fixed_zero_idx: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # 校验维度关系：D = 2F + 1
        if cfg.input_dim != 2 * cfg.raw_feature_dim + 1:
            raise ValueError(f"input_dim 必须等于 2*raw_feature_dim+1。当前 input_dim={cfg.input_dim}, raw_feature_dim={cfg.raw_feature_dim}")

        self.raw_F = cfg.raw_feature_dim
        self.D = cfg.input_dim

        # gating 只作用于原始特征（前 F 维）
        self.gating = None
        if cfg.use_gating:
            self.gating = InputFeatureGating(
                num_features=self.raw_F,
                init_logit=cfg.gating_init_logit,
                fixed_zero_idx=fixed_zero_idx,
            )

        # 输入投影：D -> embed_dim
        self.input_proj = nn.Linear(self.D, cfg.embed_dim)

        self.in_ln = nn.LayerNorm(cfg.embed_dim) if cfg.use_layernorm else nn.Identity()
        self.in_drop = nn.Dropout(cfg.dropout)

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=(cfg.dropout if cfg.num_layers > 1 else 0.0),
            bidirectional=False,
        )

        self.post_ln = nn.LayerNorm(cfg.hidden_size) if cfg.use_layernorm else nn.Identity()
        self.post_drop = nn.Dropout(cfg.dropout)

        # heads
        self.ret_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, 1),
        )

        self.mom_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [N, L, D]  (N = B*K)
        return:
          - pred_ret: [N]
          - mom_logits: [N, C]
        """
        # split: [features, isna_flags, row_present]
        feat = x[..., :self.raw_F]                  # [N,L,F]
        isna = x[..., self.raw_F: 2*self.raw_F]     # [N,L,F]
        rp = x[..., 2*self.raw_F:]                  # [N,L,1]

        # 配对删除：gate 同时作用于原始特征和对应的 isna 标记
        if self.gating is not None:
            feat = self.gating(feat)
            isna = self.gating(isna)  # 使用同一个 gate 控制 isna

        x2 = torch.cat([feat, isna, rp], dim=-1)    # [N,L,D]

        z = self.input_proj(x2)
        z = self.in_ln(z)
        z = self.in_drop(z)

        out, _ = self.lstm(z)                       # [N,L,H]
        h = out[:, -1, :]                           # 取最后一步
        h = self.post_ln(h)
        h = self.post_drop(h)

        pred_ret = self.ret_head(h).squeeze(-1)     # [N]
        mom_logits = self.mom_head(h)               # [N,C]
        return pred_ret, mom_logits

    @torch.no_grad()
    def gate_values(self) -> Optional[torch.Tensor]:
        if self.gating is None:
            return None
        return self.gating.gates().detach().cpu()

    def gate_l1(self, reduction: str = "mean") -> torch.Tensor:
        if self.gating is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.gating.l1(reduction=reduction)

    def gate_l1_logit(self, reduction: str = "mean") -> torch.Tensor:
        """L1 on raw logits (not sigmoid output). Stronger sparsity pressure."""
        if self.gating is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.gating.l1_logit(reduction=reduction)
