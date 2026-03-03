# comprehensive_report.py
"""
Comprehensive report generation for Stage 2 training
包含实验信息、数据统计、Loss分解、核心指标、CQB状态、特征筛选等
"""
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import pandas as pd


def generate_comprehensive_report(
    args,
    model,
    tr: Dict,
    va: Dict,
    te: Dict,
    epoch: int,
    best_rankic: float,
    ckpt_path: str,
    device,
    dl_train=None,
    dl_val=None,
    dl_test=None,
    cqb=None,
) -> Dict[str, Any]:
    """
    生成全面的训练报告
    
    包含6大部分：
    1. Experiment信息（可复现必备）
    2. Data & Sampling统计（避免采样偏差）
    3. Loss分解（训练过程是否合理）
    4. Core Metrics（真正关心的指标）
    5. CQB运行状态（是否真的在工作）
    6. Gating/特征筛选结果（后续删特征用）
    """
    
    ###################################################################################
    # 1. EXPERIMENT INFORMATION - 可复现必备
    ###################################################################################
    
    # 计算command hash (用于版本控制)
    import sys
    command_str = " ".join(sys.argv)
    command_hash = hashlib.md5(command_str.encode()).hexdigest()[:8]
    
    # 计算脚本文件hash
    script_path = sys.argv[0]
    script_hash = "unknown"
    if os.path.isfile(script_path):
        with open(script_path, 'rb') as f:
            script_hash = hashlib.md5(f.read()).hexdigest()[:8]
    
    # Device信息
    import torch
    device_info = {
        "device_type": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
        })
    
    experiment_info = {
        "run_name": os.path.basename(args.save_dir),
        "export_time": datetime.now().isoformat(),
        "command_line": command_str,
        "command_hash": command_hash,
        "script_file": script_path,
        "script_hash": script_hash,
        "seed": args.seed,
        "device": device_info,
        "data_paths": {
            "memmap_dir": args.memmap_dir,
            "lookback": args.lookback,
            "k": args.k,
            "batch_size": args.batch_size,
        },
        "split_config": {
            "train_years": str(getattr(args, '_train_years_actual', '[2007-2014]')),
            "val_years": str(getattr(args, '_val_years_actual', '[2015-2016]')),
            "test_years": str(getattr(args, '_test_years_actual', '[2017-2020]')),
        }
    }
    
    ###################################################################################
    # 2. DATA & SAMPLING STATISTICS - 避免采样偏差
    ###################################################################################
    
    # 这部分需要从dataloaders统计
    # 由于统计整个dataloader比较耗时，这里提供骨架，实际使用时按需开启
    data_stats = {
        "train": {"note": "Run with --compute_data_stats to get detailed statistics"},
        "val": {"note": "Run with --compute_data_stats to get detailed statistics"},
        "test": {"note": "Run with --compute_data_stats to get detailed statistics"},
    }
    
    # 如果需要详细统计，取消注释以下代码
    # for split_name, dl in [("train", dl_train), ("val", dl_val), ("test", dl_test)]:
    #     if dl is not None:
    #         data_stats[split_name] = compute_split_statistics(dl)
    
    ###################################################################################
    # 3. LOSS DECOMPOSITION - 训练过程是否合理
    ###################################################################################
    
    # 从log.csv读取训练历史
    log_path = os.path.join(args.save_dir, "log.csv")
    loss_decomposition = {
        "final_epoch": {},
        "best_epoch": {},
        "history_available": os.path.isfile(log_path),
    }
    
    if os.path.isfile(log_path):
        import csv
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # 最后一个epoch的loss分解
            if rows:
                last_train = [r for r in rows if r.get('phase') == 'train'][-1] if rows else {}
                last_val = [r for r in rows if r.get('phase') == 'val'][-1] if rows else {}
                
                loss_decomposition["final_epoch"] = {
                    "train": extract_loss_breakdown(last_train, args),
                    "val": extract_loss_breakdown(last_val, args),
                }
    
    # Best checkpoint evaluation on all splits
    best_ckpt_losses = {
        "train": extract_loss_breakdown(tr, args),
        "test": extract_loss_breakdown(te, args),
    }
    if va is not None:
        best_ckpt_losses["val"] = extract_loss_breakdown(va, args)
    loss_decomposition["best_checkpoint"] = best_ckpt_losses
    
    ###################################################################################
    # 4. CORE METRICS - 真正关心的指标
    ###################################################################################
    
    def extract_metrics(metrics_dict):
        """Extract metrics from a phase result dict"""
        return {
            "regression_ranking": {
                "IC": float(metrics_dict.get('ic', 0)),
                "IC_std": float(metrics_dict.get('ic_std', 0)),
                "ICIR": float(metrics_dict.get('icir', 0)),
                "RankIC": float(metrics_dict.get('ric', 0)),
                "RankIC_std": float(metrics_dict.get('ric_std', 0)),
                "RankICIR": float(metrics_dict.get('rankicir', 0)),
                "MSE": float(metrics_dict.get('reg', 0)),
            },
            "classification": {
                "accuracy_overall": float(metrics_dict.get('cls_acc', 0)),
                "accuracy_per_class": {
                    f"class_{i}": float(metrics_dict.get(f'acc_class{i}', 0))
                    for i in range(5)
                },
                "ce_loss": float(metrics_dict.get('ce', 0)),
            },
            "ranking_listwise": {
                "ndcg": float(metrics_dict.get('ndcg', 0)),
                "ndcg_loss": float(metrics_dict.get('ndcgL', 0)),
                "avg_k": float(metrics_dict.get('k', 0)),
                "avg_tau": float(metrics_dict.get('tau', 0)),
            }
        }
    
    core_metrics = {
        "train": extract_metrics(tr),
        "test": extract_metrics(te),
    }
    if va is not None:
        core_metrics["val"] = extract_metrics(va)
    
    # 添加建议（基于test集）
    ic = core_metrics["test"]["regression_ranking"]["IC"]
    ric = core_metrics["test"]["regression_ranking"]["RankIC"]
    icir = core_metrics["test"]["regression_ranking"]["ICIR"]
    rankicir = core_metrics["test"]["regression_ranking"]["RankICIR"]
    acc = core_metrics["test"]["classification"]["accuracy_overall"]
    
    # 过拟合检测（对比train和test）
    ic_train = core_metrics["train"]["regression_ranking"]["IC"]
    ic_val = core_metrics.get("val", core_metrics["train"])["regression_ranking"]["IC"]
    acc_train = core_metrics["train"]["classification"]["accuracy_overall"]
    acc_val = core_metrics.get("val", core_metrics["train"])["classification"]["accuracy_overall"]
    
    diagnostics = []
    
    # 过拟合检测
    if ic_train > 0 and ic > 0 and (ic_train - ic) / ic_train > 0.5:
        diagnostics.append("⚠️  IC: Train远高于Test，可能存在过拟合")
    if acc_train > 0 and acc > 0 and (acc_train - acc) > 0.15:
        diagnostics.append("⚠️  分类准确率: Train远高于Test，可能存在过拟合")
    
    # 信号质量检测
    if ic < 0.01:
        diagnostics.append("⚠️  IC很低，回归任务可能没有学到有用的信号")
    if ric < 0.01:
        diagnostics.append("⚠️  RankIC很低，排序能力较差")
    if icir < 0.5:
        diagnostics.append("⚠️  ICIR较低，IC信号不够稳定")
    if rankicir < 0.5:
        diagnostics.append("⚠️  RankICIR较低，RankIC信号不够稳定")
    if acc > 0.9:
        diagnostics.append("⚠️  分类准确率异常高，可能存在数据泄露或类别不平衡")
    if acc < 0.25:
        diagnostics.append("⚠️  分类准确率接近随机，分类任务可能没有学到")
    
    core_metrics["diagnostics"] = diagnostics
    
    ###################################################################################
    # 5. CQB RUNNING STATE - 是否真的在工作
    ###################################################################################
    
    cqb_state = {
        "enabled": args.use_cqb,
        "config": {
            "beta0": args.beta0,
            "b_win": args.b_win,
        } if args.use_cqb else None,
    }
    
    if args.use_cqb and cqb is not None:
        # 从CQB对象获取状态（需要CQB类暴露这些属性）
        cqb_state["runtime_state"] = {
            "note": "CQB state logging requires modifications to CQB class",
            # "V_reg": cqb.V_reg if hasattr(cqb, 'V_reg') else None,
            # "V_cls": cqb.V_cls if hasattr(cqb, 'V_cls') else None,
            # "beta_reg": cqb.beta_reg if hasattr(cqb, 'beta_reg') else None,
            # "beta_cls": cqb.beta_cls if hasattr(cqb, 'beta_cls') else None,
        }
    
    ###################################################################################
    # 6. GATING / FEATURE SELECTION - 特征筛选结果
    ###################################################################################
    
    gating_results = {
        "enabled": args.use_gating,
    }
    
    if args.use_gating:
        gates = model.gate_values().cpu().numpy()
        
        # 从meta.json获取特征名
        meta_path = os.path.join(args.memmap_dir, "meta.json")
        feature_names = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                feature_names = meta.get("feat_cols", None)
        
        if feature_names is None or len(feature_names) != len(gates):
            feature_names = [f"feat_{i:03d}" for i in range(len(gates))]
        
        # Gate分布统计
        gating_results["distribution"] = {
            "total_features": int(len(gates)),
            "gate_mean": float(gates.mean()),
            "gate_std": float(gates.std()),
            "gate_min": float(gates.min()),
            "gate_max": float(gates.max()),
            "gate_median": float(np.median(gates)),
            "gate_q25": float(np.percentile(gates, 25)),
            "gate_q75": float(np.percentile(gates, 75)),
        }
        
        # 稀疏性指标
        gating_results["sparsity"] = {
            "pct_lt_0.1": float((gates < 0.1).mean() * 100),
            "pct_lt_0.3": float((gates < 0.3).mean() * 100),
            "pct_lt_0.5": float((gates < 0.5).mean() * 100),
            "pct_gt_0.5": float((gates > 0.5).mean() * 100),
            "pct_gt_0.7": float((gates > 0.7).mean() * 100),
            "pct_gt_0.9": float((gates > 0.9).mean() * 100),
        }
        
        # 推荐drop的特征
        threshold_01 = gates < 0.1
        threshold_03 = gates < 0.3
        
        gating_results["recommendations"] = {
            "drop_candidates_gate_lt_0.1": {
                "count": int(threshold_01.sum()),
                "features": [feature_names[i] for i in np.where(threshold_01)[0]],
            },
            "drop_candidates_gate_lt_0.3": {
                "count": int(threshold_03.sum()),
                "features": [feature_names[i] for i in np.where(threshold_03)[0]],
            },
            "note": "Consider removing features with gate < 0.3 for efficiency"
        }
    
    ###################################################################################
    # FINAL ASSEMBLY
    ###################################################################################
    
    report = {
        "1_experiment_info": experiment_info,
        "2_data_sampling": data_stats,
        "3_loss_decomposition": loss_decomposition,
        "4_core_metrics": core_metrics,
        "5_cqb_state": cqb_state,
        "6_gating_results": gating_results,
        "model_architecture": {
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "embed_dim": args.embed_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "training_summary": {
            "total_epochs": epoch,
            "best_val_rankic": float(best_rankic),
            "note": "Early stopping monitors validation RankIC (higher is better)"
        }
    }
    
    return report


def extract_loss_breakdown(metrics_dict: Dict, args) -> Dict:
    """从metrics字典提取loss分解"""
    # Determine if using LambdaRank or MSE
    reg_loss_type = "lambdarank_ndcg" if getattr(args, 'use_lambdarank', False) else "mse"
    
    breakdown = {
        "total_loss": float(metrics_dict.get('loss', 0)),
        "components": {
            f"{reg_loss_type}_loss": float(metrics_dict.get('reg', 0)),
            "ce_loss": float(metrics_dict.get('ce', 0)),
            "ndcg_loss": float(metrics_dict.get('ndcgL', 0)),
            "gate_loss": float(metrics_dict.get('gate', 0)) if args.use_gating else 0,
        },
        "weighted_contributions": {
            f"{reg_loss_type}_contribution": float(metrics_dict.get('reg', 0)) * args.ret_w,
            "cls_contribution": (
                args.lambda_ce * float(metrics_dict.get('ce', 0)) + 
                (1 - args.lambda_ce) * float(metrics_dict.get('ndcgL', 0))
            ) * args.cls_w,
        }
    }
    
    # Add LambdaRank config if applicable
    if getattr(args, 'use_lambdarank', False):
        breakdown["lambdarank_config"] = {
            "k": getattr(args, 'lambdarank_k', 50),
            "sigma": getattr(args, 'lambdarank_sigma', 1.0),
            "bins": getattr(args, 'lambdarank_bins', 5),
        }
    
    return breakdown


def compute_split_statistics(dataloader):
    """计算某个split的详细统计（可选，比较耗时）"""
    import torch
    
    all_ret_mask = []
    all_mom_mask = []
    all_both_mask = []
    all_y_mom = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            all_ret_mask.append(batch['ret_mask'].cpu().numpy())
            all_mom_mask.append(batch['mom_mask'].cpu().numpy())
            all_both_mask.append(batch['both_mask'].cpu().numpy())
            all_y_mom.append(batch['y_mom'].cpu().numpy())
            num_batches += 1
    
    ret_mask = np.concatenate(all_ret_mask)
    mom_mask = np.concatenate(all_mom_mask)
    both_mask = np.concatenate(all_both_mask)
    y_mom = np.concatenate(all_y_mom)
    
    # 只在mom_valid上统计动量分布
    mom_valid_indices = mom_mask.flatten() > 0
    y_mom_valid = y_mom.flatten()[mom_valid_indices]
    
    mom_dist = {}
    for i in range(5):
        mom_dist[f"class_{i}_pct"] = float((y_mom_valid == i).mean() * 100)
    
    stats = {
        "num_batches": num_batches,
        "total_samples": int(ret_mask.size),
        "ret_valid_rate": float(ret_mask.mean()),
        "mom_valid_rate": float(mom_mask.mean()),
        "both_valid_rate": float(both_mask.mean()),
        "momentum_distribution": mom_dist,
    }
    
    return stats
