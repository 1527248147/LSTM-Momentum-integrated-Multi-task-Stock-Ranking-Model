# -*- coding: utf-8 -*-
"""
Adaptive-k ApproxNDCG loss (MiM-StocR, arXiv:2509.10461v2)

Key equations:
- NDCG / DCG / rank approximation: Eq.(3)-(6)
- Adaptive-k rule: Eq.(7), tau default = 20% of stock pool
- NDCG loss: Eq.(8)
- Combine with CE: Eq.(9), lambda_ce=0.5

We implement:
- per-day ApproxNDCG@k with sigmoid-smoothed ranks
- adaptive k computed from momentum label groups (4->0), not splitting same group
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch


@dataclass
class ApproxNDCGInfo:
    k: int
    tau: int
    ndcg: float
    valid_cnt: int


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _adaptive_k_from_mom(y_mom_valid: torch.Tensor, tau: int, num_classes: int = 5) -> int:
    """
    y_mom_valid: (M,) int in [0..4]
    tau: minimum truncation threshold
    Paper: start from highest momentum level, include whole groups until k >= tau.
    """
    k = 0
    # from highest (4) down to 0
    for j in range(num_classes - 1, -1, -1):
        k += int((y_mom_valid == j).sum().item())
        if k >= tau:
            break
    # if still 0 (should not), fallback
    k = max(k, 1)
    # cannot exceed valid count
    k = min(k, int(y_mom_valid.numel()))
    return k


def approx_ndcg_loss_one_day(
    scores: torch.Tensor,                 # (K,) float
    y_mom: torch.Tensor,                  # (K,) int, -1 means invalid
    tau_ratio: float = 0.2,               # paper sets minimum threshold to 20% of stock pool
    fixed_tau: Optional[int] = None,      # if provided, override tau
    temp_rank: float = 1.0,               # sigmoid temperature for rank approx (Eq.6 has no temp; default 1.0)
    temp_topk: float = 1.0,               # softness of top-k truncation (implementation choice)
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, ApproxNDCGInfo]:
    """
    Returns:
      loss: scalar tensor
      info: ApproxNDCGInfo
    """
    device = scores.device
    # valid momentum indices
    valid = (y_mom >= 0)
    M = int(valid.sum().item())
    if M < 2:
        info = ApproxNDCGInfo(k=0, tau=0, ndcg=0.0, valid_cnt=M)
        return torch.zeros((), device=device), info

    s = scores[valid].float()  # (M,)
    w = y_mom[valid].float()   # relevance weights in {0..4}

    # tau (minimum k threshold)
    if fixed_tau is not None:
        tau = int(fixed_tau)
    else:
        tau = max(1, int(round(tau_ratio * M)))

    # adaptive k (Eq.7): include groups from 4->0 until >= tau
    k = _adaptive_k_from_mom(y_mom[valid], tau=tau, num_classes=5)

    # ===== Approx rank (Eq.5-6) =====
    # If(i)<f(j) = 1 / (1 + exp(f(i)-f(j))) = sigmoid(f(j)-f(i))
    # rank_i = 1 + sum_{j!=i} If(i)<f(j)
    # We compute pairwise matrix: P_ij = sigmoid((s_j - s_i)/temp_rank)
    diff = (s.unsqueeze(0) - s.unsqueeze(1)) / max(temp_rank, 1e-6)  # (M,M): s_i - s_j
    P = torch.sigmoid(-diff)  # (M,M): sigmoid(s_j - s_i)
    # remove diagonal (i==j) - use non-inplace operation for gradient compatibility
    mask = 1.0 - torch.eye(M, device=device, dtype=P.dtype)
    P = P * mask
    rank = 1.0 + P.sum(dim=1)  # (M,)

    # ===== DCG (Eq.4) =====
    gain = torch.pow(2.0, w) - 1.0
    discount = 1.0 / torch.log2(1.0 + rank + eps)

    # ApproxNDCG@k: add a soft truncation gate for rank<=k
    # (paper does not specify exact differentiable truncation; this is a standard smooth proxy)
    topk_gate = torch.sigmoid((float(k) + 0.5 - rank) / max(temp_topk, 1e-6))
    dcg_pred = (gain * discount * topk_gate).sum()

    # Ideal DCG@k (exact, constant wrt scores)
    w_sorted, _ = torch.sort(w, descending=True)
    kk = min(k, int(w_sorted.numel()))
    pos = torch.arange(1, kk + 1, device=device, dtype=torch.float32)
    ideal_gain = torch.pow(2.0, w_sorted[:kk]) - 1.0
    ideal_discount = 1.0 / torch.log2(1.0 + pos)
    dcg_ideal = (ideal_gain * ideal_discount).sum().clamp_min(eps)

    ndcg = (dcg_pred / dcg_ideal).clamp(0.0, 1.0)
    loss = torch.exp(-ndcg)  # Eq.(8)

    info = ApproxNDCGInfo(k=int(k), tau=int(tau), ndcg=float(ndcg.detach().item()), valid_cnt=M)
    return loss, info


def approx_ndcg_loss_batch(
    scores: torch.Tensor,     # (B,K)
    y_mom: torch.Tensor,      # (B,K)
    tau_ratio: float = 0.2,
    fixed_tau: Optional[int] = None,
    temp_rank: float = 1.0,
    temp_topk: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute mean loss over B trading days. Also returns some logging stats.
    """
    B = scores.shape[0]
    losses = []
    ks, taus, ndcgs, valids = [], [], [], []
    for b in range(B):
        loss_b, info = approx_ndcg_loss_one_day(
            scores[b], y_mom[b],
            tau_ratio=tau_ratio,
            fixed_tau=fixed_tau,
            temp_rank=temp_rank,
            temp_topk=temp_topk,
        )
        losses.append(loss_b)
        if info.k > 0:
            ks.append(info.k)
            taus.append(info.tau)
            ndcgs.append(info.ndcg)
            valids.append(info.valid_cnt)

    if len(losses) == 0:
        return torch.zeros((), device=scores.device), {"ndcg": 0.0}

    loss = torch.stack(losses).mean()

    stats = {}
    if len(ndcgs) > 0:
        stats = {
            "ndcg": float(sum(ndcgs) / len(ndcgs)),
            "k_mean": float(sum(ks) / len(ks)),
            "tau_mean": float(sum(taus) / len(taus)),
            "mom_valid_mean": float(sum(valids) / len(valids)),
        }
    else:
        stats = {"ndcg": 0.0, "k_mean": 0.0, "tau_mean": 0.0, "mom_valid_mean": 0.0}

    return loss, stats
