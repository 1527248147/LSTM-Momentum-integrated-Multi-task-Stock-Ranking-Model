# -*- coding: utf-8 -*-
"""
Stage 2 Training: MiM-StocR with CQB + Adaptive-k ApproxNDCG
Paper: arXiv:2509.10461v2

What we add vs Stage1:
- L_cls = 0.5*CE + 0.5*Adaptive-k ApproxNDCG (Eq.9, lambda_ce=0.5)
- CQB on shared-layer gradients:
    * EMA smoothing (Eq.10)
    * L2 normalization + magnitude balancing (Eq.11-12)
    * adaptive forgetting rate beta_n using V_n (Eq.13-15), b=6, 12 epoch 后才开始算 Vₙ
    * regularization balancing: adjust weight_decay by mean(V_{n-1}) (Eq.21 in paper)

Notes:
- momentum label invalid is -1. We use ignore_index=-1 for CE
- Sampling: require row_present=1 via sample_present_only=True (无 t+1 泄露)
- Windows compatible (num_workers=0)
"""

import os
import sys
import time
import math
import json
import argparse
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Import model (reuse from Stage1)
from model_lstm_mtl import LSTMMTLConfig, LSTMMultiTask

# Import ApproxNDCG loss
from loss_adaptivek_approxndcg import approx_ndcg_loss_batch


# =============================================================================
# Utils
# =============================================================================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_mem_str(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserv = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return f"GPU MEM: {alloc:.0f}/{reserv:.0f} MB"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50), -50)))  # numerical stability


def compute_ic_rankic_batch(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    per-day IC / RankIC averaged across batch days.
    pred,y,mask shape: (B,K)
    Returns: ic_mean, ric_mean, ic_std, ric_std, icir, rankicir
    
    Optimized: vectorized correlation using numpy broadcasting.
    """
    B = pred.shape[0]
    ics, rics = [], []
    for b in range(B):
        m = mask[b] > 0.5
        if m.sum() < 3:
            continue
        pb = pred[b][m]
        yb = y[b][m]
        if np.std(pb) < 1e-12 or np.std(yb) < 1e-12:
            continue
        # IC: Pearson correlation
        pb_c = pb - pb.mean()
        yb_c = yb - yb.mean()
        ic = np.dot(pb_c, yb_c) / (np.linalg.norm(pb_c) * np.linalg.norm(yb_c) + 1e-12)
        # RankIC: Spearman = Pearson on ranks (use argsort-based ranking)
        n = len(pb)
        pr = np.empty(n, dtype=np.float64)
        yr = np.empty(n, dtype=np.float64)
        pr[pb.argsort()] = np.arange(n, dtype=np.float64)
        yr[yb.argsort()] = np.arange(n, dtype=np.float64)
        pr_c = pr - pr.mean()
        yr_c = yr - yr.mean()
        ric = np.dot(pr_c, yr_c) / (np.linalg.norm(pr_c) * np.linalg.norm(yr_c) + 1e-12)
        if np.isfinite(ic):
            ics.append(ic)
        if np.isfinite(ric):
            rics.append(ric)
    
    ic_mean = float(np.mean(ics)) if len(ics) else 0.0
    ric_mean = float(np.mean(rics)) if len(rics) else 0.0
    ic_std = float(np.std(ics)) if len(ics) > 1 else 0.0
    ric_std = float(np.std(rics)) if len(rics) > 1 else 0.0
    
    # ICIR = IC_mean / IC_std (Information Coefficient Information Ratio)
    icir = ic_mean / ic_std if ic_std > 1e-6 else 0.0
    rankicir = ric_mean / ric_std if ric_std > 1e-6 else 0.0
    
    return ic_mean, ric_mean, ic_std, ric_std, icir, rankicir


# =============================================================================
# CQB (Cycle-based Quad-Balancing)
# =============================================================================
@dataclass
class CQBHyper:
    beta0: float = 0.5      # initial forgetting rate beta
    b_win: int = 6          # window size b (paper uses 6)
    eps: float = 1e-12


class CQBState:
    """
    Keep EMA gradients for two tasks on shared params:
      - regression task: L_reg (MSE on next-day return)
      - classification task: L_cls (0.5 CE + 0.5 NDCG_loss)
    """
    def __init__(self, shared_params: List[torch.nn.Parameter], hyper: CQBHyper, device: torch.device):
        self.shared_params = shared_params
        self.hyper = hyper
        self.device = device

        # flattened EMA gradients
        self.ema_r: Optional[torch.Tensor] = None
        self.ema_c: Optional[torch.Tensor] = None

        # current adaptive betas (updated per-epoch)
        self.beta_r: float = hyper.beta0
        self.beta_c: float = hyper.beta0

        # debug info
        self._last_nr: float = 0.0
        self._last_nc: float = 0.0
        self._last_alpha: float = 0.0

    @staticmethod
    def _flatten_grads(grads: List[Optional[torch.Tensor]], params: List[torch.nn.Parameter]) -> torch.Tensor:
        flats = []
        for g, p in zip(grads, params):
            if g is None:
                flats.append(torch.zeros_like(p, device=p.device).view(-1))
            else:
                flats.append(g.detach().view(-1))
        return torch.cat(flats, dim=0)

    @staticmethod
    def _unflatten(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        out = []
        idx = 0
        for p in params:
            n = p.numel()
            out.append(vec[idx:idx+n].view_as(p))
            idx += n
        return out

    def set_epoch_betas(self, beta_r: float, beta_c: float):
        self.beta_r = float(beta_r)
        self.beta_c = float(beta_c)

    def step(
        self,
        grads_r: List[Optional[torch.Tensor]],
        grads_c: List[Optional[torch.Tensor]],
        grads_extra: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """
        Apply Eq.(10)-(12) on flattened gradients, then unflatten to per-param grads.
        """
        g_r = self._flatten_grads(grads_r, self.shared_params).to(self.device)
        g_c = self._flatten_grads(grads_c, self.shared_params).to(self.device)

        if self.ema_r is None:
            self.ema_r = torch.zeros_like(g_r)
        if self.ema_c is None:
            self.ema_c = torch.zeros_like(g_c)

        # Eq.(10): EMA smoothing
        br = self.beta_r
        bc = self.beta_c
        self.ema_r = br * self.ema_r + (1.0 - br) * g_r
        self.ema_c = bc * self.ema_c + (1.0 - bc) * g_c

        # Eq.(11): normalize
        nr = torch.norm(self.ema_r, p=2).clamp_min(self.hyper.eps)
        nc = torch.norm(self.ema_c, p=2).clamp_min(self.hyper.eps)
        u_r = self.ema_r / nr
        u_c = self.ema_c / nc

        # Eq.(12): magnitude balancing
        alpha = torch.maximum(nr, nc)
        g_e = alpha * (u_r + u_c)

        # add extra gradients (e.g., gating regularization) if provided
        if grads_extra is not None:
            g_x = self._flatten_grads(grads_extra, self.shared_params).to(self.device)
            g_e = g_e + g_x

        # Debug: store magnitudes for reporting
        self._last_nr = float(nr.item())
        self._last_nc = float(nc.item())
        self._last_alpha = float(alpha.item())

        return self._unflatten(g_e, self.shared_params)


def compute_Vn(train_hist: List[float], val_hist: List[float], b: int = 6, eps: float = 1e-12) -> float:
    """
    Eq.(13)-(14):
      ΔL = L_{n-1} - mean([L_{n-2b},...,L_{n-b-1}])
      Vn = ΔL_valid / ΔL_train
    Need enough history: after 12 epochs when b=6.
    """
    if len(train_hist) < 2 * b or len(val_hist) < 2 * b:
        return 1.0
    base_tr = float(np.mean(train_hist[-2*b:-b]))
    base_va = float(np.mean(val_hist[-2*b:-b]))
    d_tr = float(train_hist[-1] - base_tr)
    d_va = float(val_hist[-1] - base_va)
    if abs(d_tr) < eps:
        return 1.0
    return float(d_va / (d_tr + eps))


def beta_from_V(beta0: float, Vn: float) -> float:
    """
    Paper Eq.(15): beta_n = beta0^(sigmoid(Vn))
    """
    s = sigmoid(Vn)
    beta_n = math.exp(s * math.log(max(beta0, 1e-6)))
    return float(min(max(beta_n, 1e-4), 0.9999))


def weight_decay_from_V(base_wd: float, Vmean_prev: float) -> float:
    """
    Regularization balancing: decay_n = decay * sigmoid(-mean(V_{n-1}))
    """
    return float(base_wd * sigmoid(-Vmean_prev))


# =============================================================================
# Loss computation
# =============================================================================
def lambdarank_ndcg_loss(scores, rel, mask=None, k=50, sigma=1.0, eps=1e-8):
    """
    LambdaRank/LambdaLoss style objective: ΔNDCG@k-weighted pairwise logistic loss.
    scores: (B, N) float - model scores for ranking
    rel:    (B, N) int/float - graded relevance (e.g. 0..4 or 0..9). Larger = better.
    mask:   (B, N) bool - True means valid
    k:      top-k for NDCG computation
    sigma:  scaling factor for pairwise differences
    """
    B, N = scores.shape
    device = scores.device

    if mask is None:
        mask = torch.ones((B, N), dtype=torch.bool, device=device)

    rel = rel.float()

    # ---------- 1) Current ranking: rank -> discount (only top-k; others discount=0) ----------
    order = torch.argsort(scores, dim=1, descending=True)              # (B, N)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(N, device=device).expand(B, N))  # ranks[b, idx] = position

    discounts = 1.0 / torch.log2(ranks.float() + 2.0)                 # (B, N)
    discounts = torch.where(ranks < k, discounts, torch.zeros_like(discounts))

    # ---------- 2) Gain & IDCG@k ----------
    gains = torch.pow(2.0, rel) - 1.0                                  # (B, N)

    ideal_rel, _ = torch.sort(rel, dim=1, descending=True)             # (B, N)
    ideal_gains = torch.pow(2.0, ideal_rel) - 1.0

    pos = torch.arange(N, device=device).float()
    ideal_discounts = 1.0 / torch.log2(pos + 2.0)
    ideal_discounts = torch.where(pos < k, ideal_discounts, torch.zeros_like(ideal_discounts))  # (N,)

    idcg = (ideal_gains * ideal_discounts.unsqueeze(0)).sum(dim=1) + eps  # (B,)

    # ---------- 3) Pairwise construction: only for rel_i > rel_j pairs ----------
    s_i = scores.unsqueeze(2)  # (B, N, 1)
    s_j = scores.unsqueeze(1)  # (B, 1, N)
    score_diff = s_i - s_j     # (B, N, N)

    r_i = rel.unsqueeze(2)
    r_j = rel.unsqueeze(1)
    rel_diff = r_i - r_j       # (B, N, N)

    # pair mask: rel_i > rel_j and both valid
    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)                 # (B, N, N)
    pair_mask = (rel_diff > 0) & valid_pair                            # (B, N, N)

    # ---------- 4) ΔNDCG weight (swap approximation with current ranks) ----------
    g_i = gains.unsqueeze(2)
    g_j = gains.unsqueeze(1)
    d_i = discounts.unsqueeze(2)
    d_j = discounts.unsqueeze(1)

    delta_dcg = (g_i - g_j).abs() * (d_i - d_j).abs()                  # (B, N, N)
    w = delta_dcg / idcg.view(B, 1, 1)                                 # (B, N, N)
    
    # Numerical stability: clip weights to prevent extreme values
    w = torch.clamp(w, max=10.0)  # Prevent weight explosion

    # ---------- 5) Weighted pairwise logistic loss ----------
    # When rel_i > rel_j, we want score_i > score_j, loss = log(1 + exp(-sigma*(s_i - s_j)))
    # Clip score_diff to prevent softplus overflow
    score_diff_clipped = torch.clamp(score_diff, min=-10.0, max=10.0)
    pair_loss = w * F.softplus(-sigma * score_diff_clipped)            # (B, N, N)

    denom = pair_mask.float().sum() + eps
    loss = (pair_loss * pair_mask.float()).sum() / denom
    
    # Safety check: return 0 if loss is invalid
    if not torch.isfinite(loss):
        return torch.zeros((), device=device, requires_grad=True)
    
    return loss


def returns_to_relevance(ret, mask=None, n_bins=5):
    """
    Convert continuous returns to discrete relevance levels (0..n_bins-1).
    Uses per-day rank-based binning.
    
    ret:  (B, N) future returns
    mask: (B, N) valid stocks
    Returns: rel (B, N) long tensor with values 0..n_bins-1
    
    Fully vectorized for GPU efficiency - no Python loops.
    """
    B, N = ret.shape
    device = ret.device
    if mask is None:
        mask = torch.ones((B, N), dtype=torch.bool, device=device)

    # Fill invalid positions with -inf so they rank lowest
    ret_masked = ret.clone()
    ret_masked[~mask] = float('-inf')
    
    # Argsort to get ranking (descending: best returns first)
    _, indices = torch.sort(ret_masked, dim=1, descending=True)  # (B, N)
    
    # Inverse permutation: ranks[b, i] = position of i-th stock in sorted order
    ranks = torch.empty_like(indices)
    ranks.scatter_(1, indices, torch.arange(N, device=device).unsqueeze(0).expand(B, N))
    
    # Count valid stocks per sample
    n_valid = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
    n_valid = torch.clamp(n_valid, min=1)  # Avoid division by zero
    
    # Map ranks to bins: bin_id = floor(rank / bin_size)
    # Top-ranked (rank=0) -> highest bin (n_bins-1)
    # We reverse so higher returns get higher relevance
    bin_size = n_valid / n_bins  # (B, 1)
    rel = (n_bins - 1) - torch.floor(ranks.float() / bin_size.expand_as(ranks))  # (B, N)
    rel = torch.clamp(rel, min=0, max=n_bins - 1).long()
    
    # Zero out invalid positions
    rel[~mask] = 0
    
    return rel


def regression_mse(pred_ret: torch.Tensor, y_ret: torch.Tensor, ret_mask: torch.Tensor) -> torch.Tensor:
    """
    pred_ret,y_ret,ret_mask: (B,K)
    """
    m = (ret_mask > 0.5)
    if m.sum() == 0:
        return torch.zeros((), device=pred_ret.device)
    return F.mse_loss(pred_ret[m], y_ret[m])


def classification_ce(mom_logits: torch.Tensor, y_mom: torch.Tensor) -> torch.Tensor:
    """
    mom_logits: (B,K,5), y_mom: (B,K) with -1 invalid
    Safe CE with ignore_index=-1
    """
    B, K, C = mom_logits.shape
    y_flat = y_mom.reshape(-1).long()
    logit_flat = mom_logits.reshape(-1, C)
    return F.cross_entropy(logit_flat, y_flat, ignore_index=-1, reduction="mean")


def compute_losses_stage2(
    pred_ret: torch.Tensor,            # (B,K)
    mom_logits: torch.Tensor,          # (B,K,5)
    y_ret: torch.Tensor,              # (B,K)
    y_mom: torch.Tensor,              # (B,K)  -1 invalid
    ret_mask: torch.Tensor,           # (B,K)
    lambda_ce: float = 0.5,
    tau_ratio: float = 0.2,
    temp_rank: float = 1.0,
    temp_topk: float = 1.0,
    use_lambdarank: bool = False,
    lambdarank_k: int = 50,
    lambdarank_sigma: float = 1.0,
    lambdarank_bins: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Returns: L_reg, L_ce, L_ndcg, stats
    
    If use_lambdarank=True:
      - L_reg becomes LambdaRank NDCG loss on returns (ranking loss)
      - Converts continuous returns to discrete relevance levels
    Otherwise:
      - L_reg is standard MSE loss on returns
    """
    if use_lambdarank:
        # Convert returns to relevance levels (0..n_bins-1)
        mask_bool = (ret_mask > 0.5)
        rel = returns_to_relevance(y_ret, mask_bool, n_bins=lambdarank_bins)
        # Use LambdaRank NDCG loss instead of MSE
        L_reg = lambdarank_ndcg_loss(
            scores=pred_ret,
            rel=rel,
            mask=mask_bool,
            k=lambdarank_k,
            sigma=lambdarank_sigma,
        )
    else:
        # Standard MSE regression loss
        L_reg = regression_mse(pred_ret, y_ret, ret_mask)
    
    L_ce = classification_ce(mom_logits, y_mom)

    # ApproxNDCG: ranking induced by predicted scores (use pred_ret as score)
    L_ndcg, ndcg_stats = approx_ndcg_loss_batch(
        scores=pred_ret,
        y_mom=y_mom,
        tau_ratio=tau_ratio,
        fixed_tau=None,
        temp_rank=temp_rank,
        temp_topk=temp_topk,
    )

    stats = {"L_cls": float(lambda_ce * L_ce.item() + (1-lambda_ce) * L_ndcg.item()), **ndcg_stats}
    return L_reg, L_ce, L_ndcg, stats


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--dataset_py", type=str, required=True)
    ap.add_argument("--memmap_dir", type=str, required=True)

    # data
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    
    # feature selection
    ap.add_argument("--exclude_features", type=str, default="", help="Regex pattern to exclude features (e.g., 'fund' to exclude all fund features)")
    ap.add_argument("--exclude_features_file", type=str, default="", help="Path to text file containing exclude regex (one line, avoids CMD escaping issues)")
    ap.add_argument("--include_features", type=str, default="", help="Regex pattern to include only matching features")
    ap.add_argument("--include_features_file", type=str, default="", help="Path to text file containing include regex (one line)")

    # loss params (Stage2 specific)
    ap.add_argument("--lambda_ce", type=float, default=0.5, help="CE weight in L_cls = lambda_ce*CE + (1-lambda_ce)*NDCG")
    ap.add_argument("--tau_ratio", type=float, default=0.2, help="Adaptive-k tau = 20% of stock pool")
    ap.add_argument("--temp_rank", type=float, default=1.0)
    ap.add_argument("--temp_topk", type=float, default=1.0)
    ap.add_argument("--ret_w", type=float, default=1.0, help="Weight for regression loss")
    ap.add_argument("--cls_w", type=float, default=1.0, help="Weight for classification loss")
    
    # LambdaRank options (alternative to MSE for return prediction)
    ap.add_argument("--use_lambdarank", action="store_true", help="Use LambdaRank NDCG loss instead of MSE for return prediction")
    ap.add_argument("--lambdarank_k", type=int, default=50, help="Top-k for LambdaRank NDCG@k (default: 50)")
    ap.add_argument("--lambdarank_sigma", type=float, default=1.0, help="Scaling factor for pairwise differences in LambdaRank")
    ap.add_argument("--lambdarank_bins", type=int, default=5, help="Number of relevance bins for return discretization (default: 5)")

    # gating
    ap.add_argument("--use_gating", action="store_true")
    ap.add_argument("--gate_warmup_epochs", type=int, default=5)
    ap.add_argument("--gate_ramp_epochs", type=int, default=20)
    ap.add_argument("--gate_l1_max", type=float, default=5e-3, help="Max gate L1 (increased from Stage1)")
    ap.add_argument("--gate_reduction", type=str, default="mean", choices=["mean", "sum"])

    # CQB
    ap.add_argument("--use_cqb", action="store_true", default=True)
    ap.add_argument("--beta0", type=float, default=0.5)
    ap.add_argument("--b_win", type=int, default=6)

    # optimizer / training
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=5.0)

    # data split overrides (default: train=2007-2014, val=2015-2016, test=2017-2020)
    ap.add_argument("--train_years", type=str, default="",
                    help="Override train years, e.g. '2011-2016' or '2011,2012,2013,2014,2015,2016'")
    ap.add_argument("--val_years", type=str, default="",
                    help="Override val years. Use 'none' to disable validation (refit mode)")
    ap.add_argument("--test_years", type=str, default="",
                    help="Override test years, e.g. '2017-2020'")
    ap.add_argument("--fixed_epochs", type=int, default=0,
                    help="Train exactly this many epochs (no early stopping). 0=use patience-based early stopping")

    # model
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)

    # perf
    ap.add_argument("--num_workers", type=int, default=0, help="Windows: must be 0")
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--amp", action="store_true")

    # save
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--resume_ckpt", type=str, default="")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Stage 2: CQB + Adaptive-k ApproxNDCG")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True

    # dataset module
    spec = importlib.util.spec_from_file_location("dataset_memmap", args.dataset_py)
    ds_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds_mod)
    
    # =============================================================================
    # Feature Selection (if specified)
    # =============================================================================
    feature_indices = None
    feature_names_selected = None
    D_original = None

    # Load regex from file if specified (avoids CMD escaping issues)
    if args.exclude_features_file and not args.exclude_features:
        with open(args.exclude_features_file, 'r', encoding='utf-8') as f:
            args.exclude_features = f.read().strip()
        print(f"Loaded exclude regex from file: {args.exclude_features_file}")
    if args.include_features_file and not args.include_features:
        with open(args.include_features_file, 'r', encoding='utf-8') as f:
            args.include_features = f.read().strip()
        print(f"Loaded include regex from file: {args.include_features_file}")
    
    if args.exclude_features or args.include_features:
        print(f"\n{'='*80}")
        print("Feature Selection")
        print(f"{'='*80}")
        
        # Load feature names from meta.json
        meta_path = os.path.join(args.memmap_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        feature_names_all = meta.get("feat_cols", [])
        D_original = meta.get("D", 0)
        F_original = len(feature_names_all)
        
        if not feature_names_all:
            raise ValueError("No feat_cols found in meta.json")
        
        print(f"Original features: {F_original}")
        print(f"Original D (total dim): {D_original}")
        
        # Apply filters
        import re
        selected_mask = np.ones(F_original, dtype=bool)
        
        if args.exclude_features:
            pattern = re.compile(args.exclude_features, re.IGNORECASE)
            excluded = [i for i, name in enumerate(feature_names_all) if pattern.search(name)]
            selected_mask[excluded] = False
            print(f"\nExclude pattern: '{args.exclude_features}'")
            print(f"  Excluded {len(excluded)} features")
            if len(excluded) > 0 and len(excluded) <= 20:
                print(f"  Examples: {[feature_names_all[i] for i in excluded[:10]]}")
        
        if args.include_features:
            pattern = re.compile(args.include_features, re.IGNORECASE)
            included = [i for i, name in enumerate(feature_names_all) if pattern.search(name)]
            mask_include = np.zeros(F_original, dtype=bool)
            mask_include[included] = True
            selected_mask &= mask_include
            print(f"\nInclude pattern: '{args.include_features}'")
            print(f"  Included {len(included)} features")
        
        # Get selected feature indices
        feature_indices_feat = np.where(selected_mask)[0]
        feature_names_selected = [feature_names_all[i] for i in feature_indices_feat]
        F_selected = len(feature_names_selected)
        
        print(f"\n✓ Selected features: {F_selected} / {F_original}")
        print(f"  First 5: {feature_names_selected[:5]}")
        print(f"  Last 5: {feature_names_selected[-5:]}")
        
        # Map feature indices to X dimension indices
        # X structure: [features(F), isna(F), row_present(1)] = 2F+1
        # We need to select both feature and its isna column
        feature_indices_list = []
        for i in feature_indices_feat:
            feature_indices_list.append(i)           # feature column
            feature_indices_list.append(F_original + i)  # corresponding isna column
        feature_indices_list.append(2 * F_original)  # always keep row_present
        
        feature_indices = np.array(feature_indices_list, dtype=np.int32)
        D_selected = len(feature_indices)
        
        print(f"\n✓ Input dimension adjusted: D = {D_selected} (was {D_original})")
        print(f"  Structure: {F_selected} features + {F_selected} isna + 1 row_present")
    
    # load calendar for date splits
    # Support both absolute and relative paths for memmap_dir
    memmap_path = Path(args.memmap_dir)
    if not memmap_path.is_absolute():
        memmap_path = Path.cwd() / memmap_path
    
    # Try multiple sources for calendar/dates:
    # 1. meta.json (panel structure) - has "dates" field
    # 2. meta/calendar.csv (seq_meta structure) - has "date_str" column
    meta_json_path = memmap_path / "meta.json"
    
    if meta_json_path.exists():
        # Load from meta.json (panel structure)
        with open(meta_json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if "dates" in meta:
            dates = meta["dates"]
            cal_df = pd.DataFrame({
                "date_str": dates,
                "date_idx": range(len(dates))
            })
            cal_df["year"] = pd.to_datetime(cal_df["date_str"]).dt.year
        else:
            raise ValueError(f"meta.json exists but missing 'dates' field")
    else:
        # Try to find calendar.csv (seq_meta structure)
        if (memmap_path / "meta" / "calendar.csv").exists():
            meta_dir = memmap_path / "meta"
        elif (memmap_path.parent / "meta" / "calendar.csv").exists():
            meta_dir = memmap_path.parent / "meta"
        else:
            raise FileNotFoundError(
                f"Cannot find calendar data in:\n"
                f"  - {meta_json_path} (meta.json with 'dates')\n"
                f"  - {memmap_path}/meta/calendar.csv\n"
                f"  - {memmap_path.parent}/meta/calendar.csv"
            )
        cal_df = pd.read_csv(meta_dir / "calendar.csv")
        cal_df["year"] = pd.to_datetime(cal_df["date_str"]).dt.year

    # ---- Parse year ranges ----
    def parse_year_range(s):
        """Parse '2011-2016' or '2011,2012,...' into list of ints."""
        s = s.strip()
        if not s or s.lower() == 'none':
            return []
        if '-' in s and ',' not in s:
            parts = s.split('-')
            return list(range(int(parts[0]), int(parts[1]) + 1))
        return [int(x.strip()) for x in s.split(',')]

    # splits (paper default, overridable)
    train_years = parse_year_range(args.train_years) if args.train_years else list(range(2007, 2015))
    val_years = parse_year_range(args.val_years) if args.val_years else [2015, 2016]
    test_years = parse_year_range(args.test_years) if args.test_years else list(range(2017, 2021))

    # Store actual years for report
    args._train_years_actual = train_years
    args._val_years_actual = val_years if val_years else 'none'
    args._test_years_actual = test_years

    refit_mode = (len(val_years) == 0)  # no validation = refit mode
    if refit_mode:
        print("\n*** REFIT MODE: No validation set, will train for fixed epochs ***")
        if args.fixed_epochs <= 0:
            raise ValueError("In refit mode (--val_years none), --fixed_epochs must be > 0")

    train_indices = cal_df[cal_df["year"].isin(train_years)].index.tolist()
    test_indices = cal_df[cal_df["year"].isin(test_years)].index.tolist()

    tr_start, tr_end = max(train_indices[0], args.lookback-1), train_indices[-1]
    te_start, te_end = max(test_indices[0], args.lookback-1), test_indices[-1]

    if not refit_mode:
        val_indices = cal_df[cal_df["year"].isin(val_years)].index.tolist()
        va_start, va_end = max(val_indices[0], args.lookback-1), val_indices[-1]
    else:
        va_start, va_end = None, None

    print(f"\nData splits:")
    print(f"  Train: years={train_years} -> idx {tr_start}-{tr_end} ({len(range(tr_start, tr_end+1))} days)")
    if not refit_mode:
        print(f"  Val:   years={val_years} -> idx {va_start}-{va_end} ({len(range(va_start, va_end+1))} days)")
    else:
        print(f"  Val:   DISABLED (refit mode, fixed_epochs={args.fixed_epochs})")
    print(f"  Test:  years={test_years} -> idx {te_start}-{te_end} ({len(range(te_start, te_end+1))} days)")
    
    # Loss configuration
    print(f"\n{'='*80}")
    print("Loss Configuration")
    print(f"{'='*80}")
    if args.use_lambdarank:
        print(f"Return Prediction Loss: LambdaRank NDCG@{args.lambdarank_k}")
        print(f"  - Relevance bins: {args.lambdarank_bins}")
        print(f"  - Sigma (pairwise scale): {args.lambdarank_sigma}")
        print(f"  - Loss type: Ranking-based (quantile binning)")
    else:
        print(f"Return Prediction Loss: MSE (Mean Squared Error)")
    print(f"Classification Loss: {args.lambda_ce:.2f}*CE + {1-args.lambda_ce:.2f}*NDCG")
    print(f"Task weights: ret_w={args.ret_w}, cls_w={args.cls_w}")

    # create datasets
    def make_loader(day_s, day_e, shuffle):
        dataset = ds_mod.MemmapDayWindowDataset(
            memmap_dir=args.memmap_dir,
            lookback=args.lookback,
            day_start=day_s,
            day_end=day_e,
            k=args.k,
            seed=args.seed,
            sample_both_valid_only=False,
            sample_ret_valid_only=False,
            sample_present_only=True,   # ✓ 仅用当日row_present采样，不依赖t+1
            feature_indices=feature_indices,  # 应用特征选择
        )
        return ds_mod.build_dataloader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=shuffle,
            pin_memory=args.pin_memory,
            persistent_workers=False,
            prefetch_factor=2,
        )

    dl_train = make_loader(tr_start, tr_end, shuffle=True)
    dl_val = make_loader(va_start, va_end, shuffle=False) if not refit_mode else None
    dl_test = make_loader(te_start, te_end, shuffle=False)

    # infer dims
    first_batch = next(iter(dl_train))
    X = first_batch["X"]
    _, _, _, D = X.shape
    raw_F = (D - 1) // 2
    print(f"\nData dims: D={D}, F={raw_F} (expect D=2F+1={2*raw_F+1})")
    if 2*raw_F+1 != D:
        raise ValueError(f"Dimension mismatch!")

    # model
    cfg = LSTMMTLConfig(
        input_dim=D,
        raw_feature_dim=raw_F,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=5,
        use_layernorm=True,
        use_gating=args.use_gating,
        gating_init_logit=2.0,
    )

    model = LSTMMultiTask(cfg, fixed_zero_idx=None).to(device)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    print(f"AMP: {use_amp}")

    # CQB: identify shared params (exclude task heads AND gating params)
    # Gating params must use normal backprop so L1 regularization gradient is preserved.
    # If CQB overwrites gating grads, gate_loss gradient gets replaced by CQB's
    # synthesized gradient (which is ~0 for gate params), making L1 ineffective.
    shared_params = []
    for name, p in model.named_parameters():
        if "ret_head" in name or "mom_head" in name:
            continue
        if "gating" in name:
            continue
        shared_params.append(p)
    print(f"CQB: {len(shared_params)} shared params (excluding heads & gating)")

    cqb_h = CQBHyper(beta0=args.beta0, b_win=args.b_win)
    cqb = CQBState(shared_params, cqb_h, device=device) if args.use_cqb else None

    # history for CQB
    tr_reg_hist, tr_cls_hist = [], []
    va_reg_hist, va_cls_hist = [], []

    # gate lambda schedule
    def gate_lambda(epoch: int) -> float:
        if epoch < args.gate_warmup_epochs:
            return 0.0
        elif epoch < args.gate_warmup_epochs + args.gate_ramp_epochs:
            t = epoch - args.gate_warmup_epochs
            return args.gate_l1_max * (t / args.gate_ramp_epochs)
        else:
            return args.gate_l1_max

    # resume
    best_rankic = float("-inf")  # Monitor RankIC (higher is better)
    start_epoch = 1
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        best_rankic = float(ckpt.get("best_rankic", best_rankic))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"\nResumed from {args.resume_ckpt}, epoch={start_epoch}, best_rankic={best_rankic:.6f}")

    # training
    bad_epochs = 0
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda"))

    def run_epoch(dl, train: bool, epoch: int, beta_r: float, beta_c: float, wd_n: float):
        if train:
            model.train()
        else:
            model.eval()

        # update weight decay
        for pg in optimizer.param_groups:
            pg["weight_decay"] = wd_n

        total_loss = total_reg = total_ce = total_ndcg = total_gate = 0.0
        n_batches = 0
        ndcg_vals, k_vals, tau_vals, mom_valid_vals = [], [], [], []

        # During eval: collect on GPU, transfer once at end
        all_pred_ret, all_y_ret, all_ret_mask = [], [], []
        all_mom_logits, all_y_mom = [], []

        # Use inference_mode for eval (faster than no_grad)
        ctx = torch.inference_mode() if not train else torch.enable_grad()
        with ctx:
            for bi, batch in enumerate(dl):
                X = batch["X"].to(device, non_blocking=True)
                y_ret = batch["y_ret"].to(device, non_blocking=True)
                y_mom = batch["y_mom"].to(device, non_blocking=True)
                ret_mask = batch["ret_mask"].to(device, non_blocking=True)

                B, K, L, D_in = X.shape
                x_flat = X.reshape(B*K, L, D_in)

                with amp_ctx:
                    pred_ret_flat, mom_logits_flat = model(x_flat)
                    pred_ret = pred_ret_flat.reshape(B, K)
                    mom_logits = mom_logits_flat.reshape(B, K, -1)

                    L_reg, L_ce, L_ndcg, ndcg_stats = compute_losses_stage2(
                        pred_ret, mom_logits, y_ret, y_mom, ret_mask,
                        lambda_ce=args.lambda_ce,
                        tau_ratio=args.tau_ratio,
                        temp_rank=args.temp_rank,
                        temp_topk=args.temp_topk,
                        use_lambdarank=args.use_lambdarank,
                        lambdarank_k=args.lambdarank_k,
                        lambdarank_sigma=args.lambdarank_sigma,
                        lambdarank_bins=args.lambdarank_bins,
                    )

                    L_cls = args.lambda_ce * L_ce + (1 - args.lambda_ce) * L_ndcg

                    current_gate_lam = gate_lambda(epoch)
                    gate_loss = model.gate_l1(reduction=args.gate_reduction) if args.use_gating else torch.zeros((), device=device)

                    loss = args.ret_w * L_reg + args.cls_w * L_cls + current_gate_lam * gate_loss

                if train:
                    if args.use_cqb and cqb is not None:
                        # CQB: compute per-task grads on shared params only
                        # (shared_params excludes heads and gating)
                        Lr_scaled = args.ret_w * L_reg
                        Lc_scaled = args.cls_w * L_cls

                        grads_r = torch.autograd.grad(Lr_scaled, shared_params, retain_graph=True, allow_unused=True)
                        grads_c = torch.autograd.grad(Lc_scaled, shared_params, retain_graph=True, allow_unused=True)

                        # Normal backward for full loss (heads + gating get correct grads)
                        optimizer.zero_grad(set_to_none=True)
                        if use_amp:
                            scaler.scale(loss).backward()
                            scale = scaler.get_scale()
                        else:
                            loss.backward()
                            scale = 1.0

                        # Overwrite ONLY shared params grads with CQB-balanced grads
                        # Heads and gating params keep their normal backprop grads
                        cqb.set_epoch_betas(beta_r, beta_c)
                        g_shared = cqb.step(list(grads_r), list(grads_c), None)

                        with torch.no_grad():
                            for p, g in zip(shared_params, g_shared):
                                if p.grad is None:
                                    p.grad = torch.zeros_like(p)
                                if use_amp:
                                    p.grad.copy_(g * scale)
                                else:
                                    p.grad.copy_(g)

                    else:
                        # standard backward
                        optimizer.zero_grad(set_to_none=True)
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    # grad clip & step
                    if use_amp:
                        scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                total_loss += loss.item()
                total_reg += L_reg.item()
                total_ce += L_ce.item()
                total_ndcg += L_ndcg.item()
                total_gate += gate_loss.item()
                n_batches += 1

                if ndcg_stats.get("ndcg", 0) > 0:
                    ndcg_vals.append(ndcg_stats["ndcg"])
                    k_vals.append(ndcg_stats.get("k_mean", 0))
                    tau_vals.append(ndcg_stats.get("tau_mean", 0))
                    mom_valid_vals.append(ndcg_stats.get("mom_valid_mean", 0))

                # collect for IC and classification accuracy (stay on GPU, transfer later)
                all_pred_ret.append(pred_ret.detach())
                all_y_ret.append(y_ret.detach())
                all_ret_mask.append(ret_mask.detach())
                all_mom_logits.append(mom_logits.detach())
                all_y_mom.append(y_mom.detach())

        # Batch transfer GPU -> CPU once (avoid per-batch sync)
        all_pred_ret = torch.cat(all_pred_ret, dim=0).cpu().numpy()
        all_y_ret = torch.cat(all_y_ret, dim=0).cpu().numpy()
        all_ret_mask = torch.cat(all_ret_mask, dim=0).cpu().numpy()
        ic, ric, ic_std, ric_std, icir, rankicir = compute_ic_rankic_batch(all_pred_ret, all_y_ret, all_ret_mask)

        # compute classification accuracy (batch transfer)
        all_mom_logits = torch.cat(all_mom_logits, dim=0).cpu().numpy()  # [N, K, num_classes]
        all_y_mom = torch.cat(all_y_mom, dim=0).cpu().numpy()  # [N, K]
        mom_pred = np.argmax(all_mom_logits, axis=-1)  # [N, K]
        valid_mask = (all_y_mom >= 0)  # 忽略label=-1的样本
        
        # 总体准确率
        if valid_mask.sum() > 0:
            cls_acc = (mom_pred[valid_mask] == all_y_mom[valid_mask]).mean()
        else:
            cls_acc = 0.0
        
        # 各类别准确率
        num_classes = all_mom_logits.shape[-1]
        cls_acc_per_class = {}
        for c in range(num_classes):
            class_mask = valid_mask & (all_y_mom == c)
            if class_mask.sum() > 0:
                cls_acc_per_class[f'acc_class{c}'] = (mom_pred[class_mask] == all_y_mom[class_mask]).mean()
            else:
                cls_acc_per_class[f'acc_class{c}'] = 0.0

        return {
            "loss": total_loss / n_batches,
            "reg": total_reg / n_batches,
            "ce": total_ce / n_batches,
            "ndcgL": total_ndcg / n_batches,
            "gate": total_gate / n_batches,
            "ndcg": np.mean(ndcg_vals) if ndcg_vals else 0.0,
            "k": np.mean(k_vals) if k_vals else 0.0,
            "tau": np.mean(tau_vals) if tau_vals else 0.0,
            "mom_valid": np.mean(mom_valid_vals) if mom_valid_vals else 0.0,
            "ic": ic,
            "ric": ric,
            "ic_std": ic_std,
            "ric_std": ric_std,
            "icir": icir,
            "rankicir": rankicir,
            "cls_acc": cls_acc,
            **cls_acc_per_class,
        }

    # training loop
    print(f"\n{'='*80}")
    print("Training Start")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # In refit mode, stop at fixed_epochs
        if refit_mode and epoch > args.fixed_epochs:
            print(f"  Refit: reached fixed_epochs={args.fixed_epochs}, stopping.")
            break

        # CQB: use fixed beta (no adaptive Vn)
        beta_r = args.beta0
        beta_c = args.beta0
        wd_n = args.weight_decay

        print(f"[Epoch {epoch:03d}] CQB: β_r={beta_r:.4f} β_c={beta_c:.4f} wd={wd_n:.2e}")

        tr = run_epoch(dl_train, train=True, epoch=epoch, beta_r=beta_r, beta_c=beta_c, wd_n=wd_n)

        # update history (train side always available)
        tr_reg_hist.append(tr["reg"])
        tr_cls_hist.append(args.lambda_ce * tr["ce"] + (1-args.lambda_ce) * tr["ndcgL"])

        print(f"  Train: loss={tr['loss']:.4f} | Reg: {tr['reg']:.4f} IC={tr['ic']:.4f} RIC={tr['ric']:.4f} ICIR={tr['icir']:.2f} | Cls: CE={tr['ce']:.4f} Acc={tr['cls_acc']:.3f} NDCG={tr['ndcg']:.3f} | Gate: {tr['gate']:.4f} (λ={gate_lambda(epoch):.1e})")

        if not refit_mode:
            va = run_epoch(dl_val, train=False, epoch=epoch, beta_r=beta_r, beta_c=beta_c, wd_n=wd_n)
            va_reg_hist.append(va["reg"])
            va_cls_hist.append(args.lambda_ce * va["ce"] + (1-args.lambda_ce) * va["ndcgL"])

            print(f"  Val:   loss={va['loss']:.4f} | Reg: {va['reg']:.4f} IC={va['ic']:.4f} RIC={va['ric']:.4f} ICIR={va['icir']:.2f} | Cls: CE={va['ce']:.4f} Acc={va['cls_acc']:.3f} NDCG={va['ndcg']:.3f} | Gate: {va['gate']:.4f} {gpu_mem_str(device)}")
            print(f"  Class Acc: [0]={va.get('acc_class0',0):.3f} [1]={va.get('acc_class1',0):.3f} [2]={va.get('acc_class2',0):.3f} [3]={va.get('acc_class3',0):.3f} [4]={va.get('acc_class4',0):.3f}")
        else:
            # Refit mode: use train loss history for CQB val side too
            va_reg_hist.append(tr["reg"])
            va_cls_hist.append(args.lambda_ce * tr["ce"] + (1-args.lambda_ce) * tr["ndcgL"])
            print(f"  (refit mode: no val) {gpu_mem_str(device)}")
        
        # CQB Debug Info (只在前几个epoch打印)
        if args.use_cqb and cqb is not None and epoch <= 3:
            print(f"  CQB: grad_ratio={cqb._last_nc/max(cqb._last_nr,1e-12):.1f} β_r={beta_r:.3f} β_c={beta_c:.3f}")
        print()

        if not refit_mode:
            # early stopping - monitor RankIC (higher is better for ranking)
            val_rankic = va["ric"]
            if val_rankic > best_rankic + 1e-6:
                best_rankic = val_rankic
                bad_epochs = 0
                ckpt_path = os.path.join(args.save_dir, "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "best_rankic": best_rankic,
                }, ckpt_path)
                print(f"  ✓ Saved best: {ckpt_path} (val_RankIC={best_rankic:.6f})")
            else:
                bad_epochs += 1
                print(f"  No improve: {bad_epochs}/{args.patience}")
                if bad_epochs >= args.patience:
                    print(f"  Early stop!")
                    break
        else:
            # Refit mode: save checkpoint every epoch (final epoch = best)
            ckpt_path = os.path.join(args.save_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "best_rankic": tr["ric"],  # use train RankIC as reference
            }, ckpt_path)
            if epoch == args.fixed_epochs:
                print(f"  ✓ Saved final refit model: {ckpt_path} (train_RankIC={tr['ric']:.6f})")

    # Final evaluation on all splits with best checkpoint
    print(f"\n{'='*80}")
    print("Final Evaluation with Best Checkpoint (All Splits)")
    print(f"{'='*80}\n")
    ckpt_path = os.path.join(args.save_dir, "best.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)

    # Evaluate on train set
    print("Evaluating on Train set...")
    tr_final = run_epoch(dl_train, train=False, epoch=0, beta_r=args.beta0, beta_c=args.beta0, wd_n=args.weight_decay)
    print(f"Train Results:")
    print(f"  Total Loss: {tr_final['loss']:.4f}")
    print(f"  Regression: MSE={tr_final['reg']:.4f} | IC={tr_final['ic']:.4f} (±{tr_final['ic_std']:.4f}) | RankIC={tr_final['ric']:.4f} (±{tr_final['ric_std']:.4f})")
    print(f"  IC Ratios: ICIR={tr_final['icir']:.4f} | RankICIR={tr_final['rankicir']:.4f}")
    print(f"  Classification: CE={tr_final['ce']:.4f} | Accuracy={tr_final['cls_acc']:.4f} | NDCG={tr_final['ndcg']:.4f}")
    print(f"  Per-Class Acc: [0]={tr_final.get('acc_class0',0):.3f} [1]={tr_final.get('acc_class1',0):.3f} [2]={tr_final.get('acc_class2',0):.3f} [3]={tr_final.get('acc_class3',0):.3f} [4]={tr_final.get('acc_class4',0):.3f}")

    # Evaluate on val set
    if not refit_mode:
        print(f"\nEvaluating on Val set...")
        va_final = run_epoch(dl_val, train=False, epoch=0, beta_r=args.beta0, beta_c=args.beta0, wd_n=args.weight_decay)
        print(f"Val Results:")
        print(f"  Total Loss: {va_final['loss']:.4f}")
        print(f"  Regression: MSE={va_final['reg']:.4f} | IC={va_final['ic']:.4f} (±{va_final['ic_std']:.4f}) | RankIC={va_final['ric']:.4f} (±{va_final['ric_std']:.4f})")
        print(f"  IC Ratios: ICIR={va_final['icir']:.4f} | RankICIR={va_final['rankicir']:.4f}")
        print(f"  Classification: CE={va_final['ce']:.4f} | Accuracy={va_final['cls_acc']:.4f} | NDCG={va_final['ndcg']:.4f}")
        print(f"  Per-Class Acc: [0]={va_final.get('acc_class0',0):.3f} [1]={va_final.get('acc_class1',0):.3f} [2]={va_final.get('acc_class2',0):.3f} [3]={va_final.get('acc_class3',0):.3f} [4]={va_final.get('acc_class4',0):.3f}")
    else:
        va_final = None
        print(f"\n  (refit mode: val evaluation skipped)")

    # Evaluate on test set
    print(f"\nEvaluating on Test set...")
    te = run_epoch(dl_test, train=False, epoch=0, beta_r=args.beta0, beta_c=args.beta0, wd_n=args.weight_decay)
    print(f"Test Results:")
    print(f"  Total Loss: {te['loss']:.4f}")
    print(f"  Regression: MSE={te['reg']:.4f} | IC={te['ic']:.4f} (±{te['ic_std']:.4f}) | RankIC={te['ric']:.4f} (±{te['ric_std']:.4f})")
    print(f"  IC Ratios: ICIR={te['icir']:.4f} | RankICIR={te['rankicir']:.4f}")
    print(f"  Classification: CE={te['ce']:.4f} | Accuracy={te['cls_acc']:.4f} | NDCG={te['ndcg']:.4f}")
    print(f"  Per-Class Acc: [0]={te.get('acc_class0',0):.3f} [1]={te.get('acc_class1',0):.3f} [2]={te.get('acc_class2',0):.3f} [3]={te.get('acc_class3',0):.3f} [4]={te.get('acc_class4',0):.3f}")

    # =============================================================================
    # Save Feature Importance (if gating enabled)
    # =============================================================================
    if args.use_gating:
        print(f"\n{'='*80}")
        print("Extracting Feature Importance")
        print(f"{'='*80}\n")
        
        gates = model.gate_values().cpu().numpy()
        
        # 使用过滤后的特征名（如果做了特征选择）
        if feature_names_selected is not None:
            feature_names = feature_names_selected
            print(f"✓ 使用选择后的 {len(feature_names)} 个特征名")
            print(f"  前5个特征: {feature_names[:5]}")
            print(f"  后5个特征: {feature_names[-5:]}")
        else:
            # 从meta.json获取特征名（这是构建memmap时实际使用的特征列）
            meta_path = os.path.join(args.memmap_dir, "meta.json")
            feature_names = None
            
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    feature_names = meta.get("feat_cols", None)
                    
                    if feature_names and len(feature_names) == len(gates):
                        print(f"✓ 从meta.json加载到 {len(feature_names)} 个特征名")
                        print(f"  前5个特征: {feature_names[:5]}")
                        print(f"  后5个特征: {feature_names[-5:]}")
                    elif feature_names:
                        print(f"⚠️  警告: 特征名数量不匹配！")
                        print(f"  Gate数: {len(gates)}, feat_cols数: {len(feature_names)}")
                        feature_names = None
        
        if feature_names is None or len(feature_names) != len(gates):
            print(f"Warning: 无法获取特征名，使用默认命名")
            print(f"  Gate数: {len(gates)}")
            feature_names = [f"feat_{i:03d}" for i in range(len(gates))]
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature_name': feature_names,
            'gate_value': gates
        })
        feature_importance = feature_importance.sort_values('gate_value', ascending=False).reset_index(drop=True)
        feature_importance['rank'] = feature_importance.index + 1
        
        # Classify importance level
        def classify_importance(gate_val):
            if gate_val >= 0.9:
                return "Very High"
            elif gate_val >= 0.7:
                return "High"
            elif gate_val >= 0.5:
                return "Medium"
            elif gate_val >= 0.3:
                return "Low"
            else:
                return "Very Low"
        
        feature_importance['importance_level'] = feature_importance['gate_value'].apply(classify_importance)
        
        # Save feature importance
        feat_dir = os.path.join(args.save_dir, "feature_importance")
        os.makedirs(feat_dir, exist_ok=True)
        
        csv_path = os.path.join(feat_dir, "feature_importance.csv")
        feature_importance.to_csv(csv_path, index=False)
        print(f"✓ Saved feature importance: {csv_path}")
        
        # Save top and bottom features
        top_path = os.path.join(feat_dir, "top_50_features.csv")
        feature_importance.head(50).to_csv(top_path, index=False)
        print(f"✓ Saved top 50 features: {top_path}")
        
        bottom_path = os.path.join(feat_dir, "bottom_50_features.csv")
        feature_importance.tail(50).to_csv(bottom_path, index=False)
        print(f"✓ Saved bottom 50 features: {bottom_path}")
        
        # Print summary statistics
        print(f"\nFeature Importance Summary:")
        print(f"  Total Features: {len(gates)}")
        print(f"  Gate Value Range: [{gates.min():.4f}, {gates.max():.4f}]")
        print(f"  Mean Gate Value: {gates.mean():.4f} ± {gates.std():.4f}")
        print(f"\nTop 5 Features:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"  {i+1}. {row['feature_name']:<30} gate={row['gate_value']:.4f}")
        print(f"\nSparsity Metrics:")
        print(f"  Gate < 0.1: {(gates < 0.1).sum():4d} ({(gates < 0.1).mean()*100:5.1f}%)")
        print(f"  Gate < 0.3: {(gates < 0.3).sum():4d} ({(gates < 0.3).mean()*100:5.1f}%)")
        print(f"  Gate > 0.9: {(gates > 0.9).sum():4d} ({(gates > 0.9).mean()*100:5.1f}%)")

    # =============================================================================
    # Save Comprehensive Final Results
    # =============================================================================
    print(f"\n{'='*80}")
    print("Saving Comprehensive Final Results")
    print(f"{'='*80}\n")
    
    # 使用comprehensive_report生成完整报告
    from comprehensive_report import generate_comprehensive_report, extract_loss_breakdown
    
    report = generate_comprehensive_report(
        args=args,
        model=model,
        tr=tr_final,
        va=va_final,
        te=te,
        epoch=epoch,
        best_rankic=best_rankic,
        ckpt_path=ckpt_path,
        device=device,
        dl_train=dl_train,
        dl_val=dl_val,
        dl_test=dl_test,
        cqb=cqb if args.use_cqb else None,
    )
    
    # 保存完整JSON报告
    report_path = os.path.join(args.save_dir, "comprehensive_report.json")
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved comprehensive report: {report_path}")
    
    # 读取训练历史（用于后续打印）
    log_path = os.path.join(args.save_dir, "log.csv")
    
    # Print comprehensive final summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE FINAL SUMMARY")
    print(f"{'='*80}")
    
    exp_info = report["1_experiment_info"]
    print(f"\n📁 Experiment Info:")
    print(f"  Run: {exp_info['run_name']}")
    print(f"  Time: {exp_info['export_time']}")
    print(f"  Seed: {exp_info['seed']}")
    print(f"  Device: {exp_info['device']['device_type']}, {exp_info['device'].get('gpu_name', 'N/A')}")
    print(f"  Command Hash: {exp_info['command_hash']}")
    
    arch = report["model_architecture"]
    print(f"\n🏗️  Model Architecture:")
    print(f"  Total Params: {arch['total_params']:,} ({arch['total_params']/1e6:.2f}M)")
    print(f"  Hidden: {arch['hidden_size']}, Layers: {arch['num_layers']}, Dropout: {arch['dropout']}")
    
    print(f"\n📊 Core Metrics (Test Set):")
    cm = report["4_core_metrics"]["test"]
    print(f"\n  回归/排序 (Regression/Ranking):")
    print(f"    IC:       {cm['regression_ranking']['IC']:.4f} (±{cm['regression_ranking']['IC_std']:.4f})")
    print(f"    ICIR:     {cm['regression_ranking']['ICIR']:.4f}")
    print(f"    RankIC:   {cm['regression_ranking']['RankIC']:.4f} (±{cm['regression_ranking']['RankIC_std']:.4f})")
    print(f"    RankICIR: {cm['regression_ranking']['RankICIR']:.4f}")
    print(f"    MSE:      {cm['regression_ranking']['MSE']:.4f}")
    
    print(f"\n  分类 (Classification):")
    print(f"    Accuracy: {cm['classification']['accuracy_overall']:.4f}")
    print(f"    CE Loss:  {cm['classification']['ce_loss']:.4f}")
    print(f"    Per-Class Accuracy:")
    for cls, acc in cm['classification']['accuracy_per_class'].items():
        print(f"      {cls}: {acc:.3f}")
    
    print(f"\n  Listwise排序 (NDCG):")
    print(f"    NDCG:      {cm['ranking_listwise']['ndcg']:.4f}")
    print(f"    Avg K:     {cm['ranking_listwise']['avg_k']:.1f}")
    print(f"    Avg Tau:   {cm['ranking_listwise']['avg_tau']:.1f}")
    
    # Loss分解
    print(f"\n💰 Loss Decomposition (Test Set):")
    loss_dec = report["3_loss_decomposition"]["best_checkpoint"]["test"]
    print(f"  Total Loss: {loss_dec['total_loss']:.4f}")
    print(f"  Components:")
    for name, val in loss_dec['components'].items():
        print(f"    {name}: {val:.4f}")
    print(f"  Weighted Contributions:")
    for name, val in loss_dec['weighted_contributions'].items():
        print(f"    {name}: {val:.4f}")
    
    # CQB状态
    cqb_info = report["5_cqb_state"]
    if cqb_info['enabled']:
        print(f"\n⚖️  CQB State:")
        print(f"  Enabled: True (beta0={cqb_info['config']['beta0']}, b_win={cqb_info['config']['b_win']})")
    
    # Gating结果
    gating = report["6_gating_results"]
    if gating['enabled']:
        print(f"\n🎯 Feature Selection (Gating):")
        dist = gating['distribution']
        print(f"  Total Features: {dist['total_features']}")
        print(f"  Gate: mean={dist['gate_mean']:.3f}, std={dist['gate_std']:.3f}, "
              f"median={dist['gate_median']:.3f}")
        print(f"  Sparsity:")
        sp = gating['sparsity']
        print(f"    <0.1: {sp['pct_lt_0.1']:.1f}%, <0.3: {sp['pct_lt_0.3']:.1f}%, "
              f"<0.5: {sp['pct_lt_0.5']:.1f}%")
        print(f"    >0.9: {sp['pct_gt_0.9']:.1f}%")
        
        rec = gating['recommendations']
        print(f"\n  💡 Feature Drop Recommendations:")
        print(f"    Gate <0.1: {rec['drop_candidates_gate_lt_0.1']['count']} features")
        print(f"    Gate <0.3: {rec['drop_candidates_gate_lt_0.3']['count']} features")
        if rec['drop_candidates_gate_lt_0.3']['count'] > 0:
            print(f"    Consider removing low-gate features for efficiency")
    
    # 诊断信息
    if report["4_core_metrics"].get("diagnostics"):
        print(f"\n⚠️  Diagnostics:")
        for diag in report["4_core_metrics"]["diagnostics"]:
            print(f"  {diag}")
    
    print(f"\n📄 Output Files:")
    print(f"  - best.pt                          : Model checkpoint")
    print(f"  - comprehensive_report.json        : Complete analysis (JSON)")
    print(f"  - final_results.json               : Legacy format")
    print(f"  - final_results.csv                : Key metrics (CSV)")
    if os.path.isfile(log_path):
        print(f"  - log.csv                          : Training log")
    if gating['enabled']:
        print(f"  - feature_importance/*.csv         : Feature importance details")
        feat_dir = os.path.join(args.save_dir, "feature_importance")
        if os.path.exists(feat_dir):
            drop_01 = os.path.join(feat_dir, "drop_candidates_gate_lt_0.1.txt")
            drop_03 = os.path.join(feat_dir, "drop_candidates_gate_lt_0.3.txt")
            
            # 保存推荐drop的特征列表
            with open(drop_01, 'w', encoding='utf-8') as f:
                for feat in rec['drop_candidates_gate_lt_0.1']['features']:
                    f.write(feat + '\n')
            print(f"  - feature_importance/drop_candidates_gate_lt_0.1.txt")
            
            with open(drop_03, 'w', encoding='utf-8') as f:
                for feat in rec['drop_candidates_gate_lt_0.3']['features']:
                    f.write(feat + '\n')
            print(f"  - feature_importance/drop_candidates_gate_lt_0.3.txt")

    print(f"\n{'='*80}")
    print("✅ Stage 2 Training Complete with Comprehensive Report!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

