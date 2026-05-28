"""
train_dgnet_3d.py — DGNet3D 基线训练脚本
=========================================
修复：all_reduce_metrics 后 d 不再多除 world_size（指标放大 bug）

与 train_phys_hgnet_3d.py 接口完全对称，确保对比实验控制变量一致。
"""

import argparse, json, os, time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from dgnet_3d    import DGNet3D, DEFAULT_CONFIG_DGNET3D
from dataset_3d  import LaserHardening3DDataset, collate_fn_3d, find_h5_file


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
# 工具函数
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


def zero_nan_gradients(model: nn.Module) -> int:
    n = 0
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad.data = torch.nan_to_num(p.grad.data,
                                           nan=0.0, posinf=0.0, neginf=0.0)
            n += 1
    return n


def all_reduce_metrics(metrics, device, world_size):
    if world_size <= 1:
        return metrics
    keys   = list(metrics.keys())
    tensor = torch.tensor([metrics[k] for k in keys],
                           dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return {k: float(tensor[i]) for i, k in enumerate(keys)}

# ─────────────────────────────────────────────────────────────────
# Epoch（含进度条）
# ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, is_train,
              scaler=None, rank=0, world_size=1, epoch_desc="train"):
    model.train() if is_train else model.eval()
    sum_mse = sum_rne = 0.0
    n_ok = n_nan = 0

    show_bar = (rank == 0 and HAS_TQDM)
    bar = tqdm(loader, desc=epoch_desc, leave=False,
               ncols=90, disable=not show_bar)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in bar:
            batch = batch_to_device(batch, device)
            tgt   = batch["targets"]

            amp_on = (scaler is not None)
            with torch.amp.autocast('cuda', enabled=amp_on):
                out  = model(batch)
                pred = out["u_final"]
                if not torch.isfinite(pred).all().item():
                    n_nan += 1
                    pred = torch.nan_to_num(pred, nan=298.15,
                                            posinf=2000.0, neginf=0.0)
                mse  = F.mse_loss(pred, tgt)
                rne  = (pred - tgt).norm() / tgt.norm().clamp(min=1e-8)
                loss = mse + 0.01 * F.relu(-pred).mean()

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    zero_nan_gradients(model)
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    zero_nan_gradients(model)
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            sum_mse += mse.item()
            sum_rne += rne.item()
            n_ok    += 1

            if show_bar:
                bar.set_postfix(mse=f"{mse.item():.3f}",
                                rne=f"{rne.item():.4f}")

    if show_bar:
        bar.close()

    agg = all_reduce_metrics(
        {"mse": sum_mse, "rne": sum_rne,
         "n_ok": float(n_ok), "n_nan": float(n_nan)},
        device, world_size)
    d = max(int(agg["n_ok"]), 1)   # 修复：不除 world_size
    return {
        "mse"        : agg["mse"] / d,
        "rne"        : agg["rne"] / d,
        "nan_batches": int(agg["n_nan"]),
    }

# ─────────────────────────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────────────────────────

def train(args):
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device  = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    h5_path = find_h5_file(args.data_dir, args.n_nodes)

    import h5py
    with h5py.File(str(h5_path), 'r') as f:
        all_keys = sorted([k for k in f.keys() if k.startswith('trajectory_')],
                          key=lambda x: int(x.split('_')[1]))
    n_traj  = len(all_keys)
    n_train = int(n_traj * 0.8)
    train_ti = list(range(n_train))
    val_ti   = list(range(n_train, n_traj))

    train_ds = LaserHardening3DDataset(
        str(h5_path), window_size=args.window_size, stride=args.stride,
        traj_indices=train_ti)
    val_ds   = LaserHardening3DDataset(
        str(h5_path), window_size=args.window_size, stride=args.stride,
        traj_indices=val_ti)

    if is_main:
        print(f"╔{'═'*64}╗")
        print(f"║  DGNet3D (baseline)  |  N={train_ds.N}  |  {world_size} GPU(s)")
        print(f"║  window={args.window_size}  stride={args.stride}  "
              f"lr={args.lr}  bs={args.batch_size}")
        print(f"╚{'═'*64}╝")
        print(train_ds.info())
        print(f"  训练窗口={len(train_ds)} | val窗口={len(val_ds)}"
              f" | batch={args.batch_size}")

    if world_size > 1:
        tr_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        va_sampler = DistributedSampler(
            val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
        tr_shuffle = False
    else:
        tr_sampler = va_sampler = None
        tr_shuffle = True

    tr_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=tr_sampler, shuffle=tr_shuffle,
        collate_fn=collate_fn_3d, num_workers=args.num_workers, drop_last=True)
    va_loader = DataLoader(
        val_ds,   batch_size=args.batch_size,
        sampler=va_sampler, shuffle=False,
        collate_fn=collate_fn_3d, num_workers=args.num_workers)

    model_cfg = {
        **DEFAULT_CONFIG_DGNET3D,
        "residual_hidden_dim" : args.hidden_dim,
        "residual_num_layers" : args.n_layers,
    }
    model = DGNet3D(model_cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)
    raw_model = model.module if world_size > 1 else model
    if is_main:
        print(raw_model.extra_info())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = (torch.amp.GradScaler('cuda')
              if torch.cuda.is_available() and args.amp else None)

    ckpt_dir  = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_{train_ds.N}.pth"

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
            print(f"  ↩ 恢复 epoch={start_epoch}  best_rne={best_rne:.4f}")

    log = []
    epoch_bar = (tqdm(range(start_epoch, args.epochs),
                      desc="DGNet3D", unit="ep", ncols=90)
                 if (is_main and HAS_TQDM) else range(start_epoch, args.epochs))

    for epoch in epoch_bar:
        if world_size > 1:
            tr_sampler.set_epoch(epoch)

        t0   = time.time()
        tr_m = run_epoch(model, tr_loader, optimizer, device,
                         is_train=True,  scaler=scaler,
                         rank=rank, world_size=world_size,
                         epoch_desc=f"  train ep{epoch+1}")
        va_m = run_epoch(model, va_loader, None, device,
                         is_train=False,
                         rank=rank, world_size=world_size,
                         epoch_desc=f"  val   ep{epoch+1}")
        scheduler.step()

        if is_main:
            elapsed = time.time() - t0
            nan_s   = f" NaN={tr_m['nan_batches']}" if tr_m["nan_batches"] else ""
            line    = (f"Ep {epoch+1:>4}/{args.epochs} | "
                       f"tr MSE={tr_m['mse']:>10.3f} RNE={tr_m['rne']:.4f} | "
                       f"va MSE={va_m['mse']:>10.3f} RNE={va_m['rne']:.4f} | "
                       f"{elapsed:.1f}s{nan_s}")
            if HAS_TQDM:
                epoch_bar.set_postfix(
                    va_rne=f"{va_m['rne']:.4f}",
                    tr_rne=f"{tr_m['rne']:.4f}")
                tqdm.write(line)
            else:
                print(line)

            log.append({
                "epoch"    : epoch + 1,
                "train_mse": tr_m["mse"], "train_rne": tr_m["rne"],
                "val_mse"  : va_m["mse"], "val_rne"  : va_m["rne"],
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
                msg = f"  ✓ 保存最优 (RNE={best_rne:.4f})"
                tqdm.write(msg) if HAS_TQDM else print(msg)

    if is_main:
        with open(ckpt_dir / f"train_log_{train_ds.N}.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n训练完成！best val RNE={best_rne:.4f}")
        print(f"检查点: {ckpt_path}")

    cleanup_ddp()

# ─────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DGNet3D 基线训练")
    p.add_argument("--n_nodes",     type=int,   default=1000)
    p.add_argument("--data_dir",    type=str,   default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir",    type=str,   default="checkpoints/dgnet_3d")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--window_size", type=int,   default=10)
    p.add_argument("--stride",      type=int,   default=None)
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--n_layers",    type=int,   default=5)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--amp",         action="store_true")
    p.add_argument("--resume",      action="store_true")
    args = p.parse_args()
    if args.stride is None:
        args.stride = max(1, args.window_size // 2)
    return args

if __name__ == "__main__":
    train(parse_args())
