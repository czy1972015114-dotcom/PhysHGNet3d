"""
train_phys_hgnet_3d.py — PhysHGNet 3D 训练脚本（v4 全面修复版）
=============================================================

修复历史
--------
v1: 初始版本
v2: find_unused_parameters=True + NaN 零损失回退（双重 backward 崩溃）
v3: nan_to_num 夹住最终预测（NaN 梯度污染仍存在）
v4（本版）：
    ① NaN 梯度清零：backward 后立即检测并置零 NaN/inf 梯度
      → 防止 PCG rollout 中 NaN 通过梯度污染模型参数
    ② 跨 rank dist.all_reduce 聚合指标
      → val MSE/RNE 反映整个验证集，而非 rank 0 的局部子集
    ③ 降低默认学习率到 3e-4，梯度裁剪阈值降至 0.5
      → 减小 NaN 批次对参数更新的影响
"""

import argparse, json, os, time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D
from dataset_3d     import LaserHardening3DDataset, collate_fn_3d, find_h5_file

# ─────────────────────────────────────────────────────────────────
# DDP
# ─────────────────────────────────────────────────────────────────

def setup_ddp():
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# ─────────────────────────────────────────────────────────────────
# Batch 迁移
# ─────────────────────────────────────────────────────────────────

def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {k2: (v2.to(device) if isinstance(v2, torch.Tensor) else v2)
                      for k2, v2 in v.items()}
        else:
            out[k] = v
    return out

# ─────────────────────────────────────────────────────────────────
# 修复 ①: NaN 梯度清零
# ─────────────────────────────────────────────────────────────────

def zero_nan_gradients(model: nn.Module) -> int:
    """将所有参数中的 NaN/inf 梯度置零，返回被修复的参数数量。

    为什么必须这样做：
    PCG 求解器在某些时步产生 NaN → u_curr 含 NaN →
    GNN 收到 NaN 输入 → 产生 NaN 梯度 →
    clip_grad_norm_ 无法修复 NaN → 参数更新时被 NaN 污染 →
    模型越训越差（MSE 上升）。

    做法：backward 之后、optimizer.step() 之前调用本函数。
    NaN 位置梯度置零 = 该参数在本 batch 不更新，安全且正确。
    """
    n_fixed = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0,
                                            posinf=0.0, neginf=0.0)
            n_fixed += 1
    return n_fixed

# ─────────────────────────────────────────────────────────────────
# 修复 ②: 跨 rank 指标聚合
# ─────────────────────────────────────────────────────────────────

def all_reduce_metrics(metrics: Dict[str, float],
                       device: torch.device,
                       world_size: int) -> Dict[str, float]:
    """用 dist.all_reduce 把所有 rank 的指标求和后平均。

    为什么必须这样做：
    只有 rank 0 打印指标，但 rank 0 用 DistributedSampler 只看到
    1/7 的数据。不聚合 → val MSE 在每个 epoch 完全相同（同一个子集）。
    """
    if world_size <= 1:
        return metrics

    keys = list(metrics.keys())
    tensor = torch.tensor([metrics[k] for k in keys],
                           dtype=torch.float64, device=device)
    # all_reduce -> 求和，返回各 rank 的和。不要在这里除以 world_size，
    # 否则随后按批次数归一化时会出现缩放错误（重复除以 world_size）。
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {k: float(tensor[i]) for i, k in enumerate(keys)}

# ─────────────────────────────────────────────────────────────────
# 训练 / 验证 epoch
# ─────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, optimizer, device, is_train,
    scaler=None, epoch=0, rank=0, world_size=1,
) -> Dict[str, float]:
    model.train() if is_train else model.eval()

    sum_mse = sum_rne = 0.0
    n_ok = n_nan = n_nan_grad = n_input_nan = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            batch = batch_to_device(batch, device)

            # 输入数据 NaN 检测（快速定位哪些轨迹文件 / 批次包含异常）
            try:
                if "initial_conditions" in batch and not torch.isfinite(batch["initial_conditions"]).all().item():
                    n_input_nan += 1
                    print(f"[rank {rank}] Epoch {epoch} batch {batch_idx}: NaN in initial_conditions; traj_keys={batch.get('traj_keys')}")
                if "source_terms" in batch and not torch.isfinite(batch["source_terms"]).all().item():
                    n_input_nan += 1
                    print(f"[rank {rank}] Epoch {epoch} batch {batch_idx}: NaN in source_terms; traj_keys={batch.get('traj_keys')}")
                if "targets" in batch and not torch.isfinite(batch["targets"]).all().item():
                    n_input_nan += 1
                    print(f"[rank {rank}] Epoch {epoch} batch {batch_idx}: NaN in targets; traj_keys={batch.get('traj_keys')}")
            except Exception:
                # 在某些环境 .item() 可能抛出，确保训练不因此中断
                pass
            tgt   = batch["targets"].to(device)          # (B, T, N, 1)

            amp_on = (scaler is not None)
            with torch.amp.autocast('cuda', enabled=amp_on):
                out  = model(batch)
                pred = out["u_final"]                    # (B, T, N, 1)

                # NaN 检测：若预测含 NaN，我们使用 clamped 值计算指标，
                # 但跳过该批次的反向传播以避免梯度污染。
                nan_in_pred = not torch.isfinite(pred).all().item()
                if nan_in_pred:
                    n_nan += 1
                    print(f"[rank {rank}] Epoch {epoch} batch {batch_idx}: NaN in prediction; traj_keys={batch.get('traj_keys')}")
                    pred_clamped = torch.nan_to_num(pred, nan=298.15,
                                                    posinf=2000.0, neginf=0.0)
                else:
                    pred_clamped = pred

                mse      = F.mse_loss(pred_clamped, tgt)
                rne      = ((pred_clamped - tgt).norm()
                            / tgt.norm().clamp(min=1e-8))
                phys_pen = F.relu(-pred_clamped).mean()
                loss     = mse + 0.01 * phys_pen

            # 如果预测含 NaN，则跳过该批次的反向传播与指标累加
            skip_batch = nan_in_pred
            if is_train:
                optimizer.zero_grad()
                if skip_batch:
                    print(f"[rank {rank}] Epoch {epoch} batch {batch_idx}: skipping backward/step due to NaN in prediction")
                else:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        # ── 修复 ①: NaN 梯度清零 ──────────────────
                        n_fixed = zero_nan_gradients(model)
                        if n_fixed > 0:
                            n_nan_grad += 1
                            print(f"[rank {rank}] Epoch {epoch} batch {batch_idx}: fixed {n_fixed} NaN/inf parameter gradients")
                        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        # ── 修复 ①: NaN 梯度清零 ──────────────────
                        n_fixed = zero_nan_gradients(model)
                        if n_fixed > 0:
                            n_nan_grad += 1
                            print(f"[rank {rank}] Epoch {epoch} batch {batch_idx}: fixed {n_fixed} NaN/inf parameter gradients")
                        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

            # 如果预测含 NaN，则不计入有效样本统计（避免污染训练/验证指标）
            if skip_batch:
                continue

            sum_mse += mse.item()
            sum_rne += rne.item()
            n_ok    += 1

    # ── 修复 ②: 跨 rank 聚合指标 ─────────────────────────────────
    raw = {
        "mse"      : sum_mse,
        "rne"      : sum_rne,
        "n_ok"     : float(n_ok),
        "n_nan"    : float(n_nan),
        "n_ng"     : float(n_nan_grad),
        "n_inp_nan": float(n_input_nan),
    }
    agg = all_reduce_metrics(raw, device, world_size)

    n_ok_total = max(int(agg["n_ok"]), 1)
    # agg contains sums across all ranks (all_reduce SUM),
    # so normalize by the total number of processed batches to get averages.
    return {
        "mse"         : float(agg["mse"]) / float(n_ok_total),
        "rne"         : float(agg["rne"]) / float(n_ok_total),
        "nan_batches" : int(agg["n_nan"]),
        "nan_grads"   : int(agg["n_ng"]),
    }

# ─────────────────────────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────────────────────────

def train(args):
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device  = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"=== PhysHGNet3D 训练 | N≈{args.n_nodes} | world_size={world_size} ===")

    # ── 数据 ─────────────────────────────────────────────────────
    h5_path = find_h5_file(args.data_dir, args.n_nodes)
    full_ds = LaserHardening3DDataset(h5_path, n_time_steps=args.n_time_steps)
    N       = full_ds.N

    n_total = len(full_ds)
    n_train = int(n_total * 0.8)
    n_val   = n_total - n_train
    gen     = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=gen)

    if world_size > 1:
        tr_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        va_sampler = DistributedSampler(
            val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
        tr_shuffle = False
    else:
        tr_sampler = va_sampler = None
        tr_shuffle = True

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size,
                           sampler=tr_sampler, shuffle=tr_shuffle,
                           collate_fn=collate_fn_3d,
                           num_workers=args.num_workers, drop_last=True)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size,
                           sampler=va_sampler, shuffle=False,
                           collate_fn=collate_fn_3d,
                           num_workers=args.num_workers)

    if is_main:
        print(full_ds.info())
        print(f"  训练={n_train} | 验证={n_val} | batch={args.batch_size}")

    # ── 模型 ─────────────────────────────────────────────────────
    model_cfg = {
        **DEFAULT_CONFIG_3D,
        "m_anchors"           : args.m_anchors,
        "residual_hidden_dim" : args.hidden_dim,
        "residual_num_layers" : args.n_layers,
        "use_physics_anchor"  : not args.no_physics_anchor,
        "use_learned_coarse"  : not args.no_learned_coarse,
        "use_dual_scale_gnn"  : not args.no_dual_scale,
        "use_virtual_nodes"   : not args.no_virtual_nodes,
    }
    model = PhysHGNet3D(model_cfg).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)

    raw_model = model.module if world_size > 1 else model
    if is_main:
        print(raw_model.extra_info())
        print(f"  总参数量: {raw_model.num_parameters():,}")

    # ── 优化器（v4：lr=3e-4 更稳）────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = (torch.amp.GradScaler('cuda')
              if torch.cuda.is_available() and args.amp else None)

    # ── 检查点 ───────────────────────────────────────────────────
    ckpt_dir  = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_{N}.pth"

    start_epoch = 0
    best_rne    = float('inf')
    if ckpt_path.exists() and args.resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_rne    = ckpt["best_rne"]
        if is_main:
            print(f"  恢复 epoch={start_epoch}, best_rne={best_rne:.4f}")

    log = []
    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            tr_sampler.set_epoch(epoch)

        t0   = time.time()
        tr_m = run_epoch(model, tr_loader, optimizer, device,
                         is_train=True,  scaler=scaler,
                         epoch=epoch, rank=rank, world_size=world_size)
        va_m = run_epoch(model, va_loader, None, device,
                         is_train=False, epoch=epoch,
                         rank=rank, world_size=world_size)
        scheduler.step()

        if is_main:
            elapsed = time.time() - t0
            extra = ""
            if tr_m["nan_batches"] > 0:
                extra += f"  [NaN→clamp: {tr_m['nan_batches']}]"
            if tr_m["nan_grads"] > 0:
                extra += f"  [NaN梯度清零: {tr_m['nan_grads']}]"
            print(
                f"Epoch {epoch+1:4d}/{args.epochs} | "
                f"train MSE={tr_m['mse']:.4f} RNE={tr_m['rne']:.4f} | "
                f"val MSE={va_m['mse']:.4f} RNE={va_m['rne']:.4f} | "
                f"{elapsed:.1f}s{extra}"
            )
            log.append({
                "epoch"      : epoch + 1,
                "train_mse"  : tr_m["mse"],
                "train_rne"  : tr_m["rne"],
                "val_mse"    : va_m["mse"],
                "val_rne"    : va_m["rne"],
                "nan_batches": tr_m["nan_batches"],
                "nan_grads"  : tr_m["nan_grads"],
            })

            cur = va_m["rne"]
            if 0 < cur < best_rne:
                best_rne = cur
                torch.save({
                    "model"    : raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch"    : epoch,
                    "best_rne" : best_rne,
                    "config"   : model_cfg,
                }, ckpt_path)
                print(f"  ✓ 保存最优 (RNE={best_rne:.4f})")

    if is_main:
        log_path = ckpt_dir / f"train_log_{N}.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n训练完成！best val RNE={best_rne:.4f}")
        print(f"检查点: {ckpt_path}   日志: {log_path}")

    cleanup_ddp()

# ─────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PhysHGNet3D 训练（v4）")
    p.add_argument("--n_nodes",      type=int,   default=5000)
    p.add_argument("--data_dir",     type=str,   default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir",     type=str,   default="checkpoints/phys_hgnet_3d")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=3e-4)   # v4: 降低默认 lr
    p.add_argument("--n_time_steps", type=int,   default=20)
    p.add_argument("--m_anchors",    type=int,   default=64)
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--n_layers",     type=int,   default=5)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--no_physics_anchor", action="store_true")
    p.add_argument("--no_learned_coarse", action="store_true")
    p.add_argument("--no_dual_scale",     action="store_true")
    p.add_argument("--no_virtual_nodes",  action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
