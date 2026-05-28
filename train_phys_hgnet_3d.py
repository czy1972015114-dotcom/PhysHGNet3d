"""
train_phys_hgnet_3d.py — PhysHGNet 3D 训练脚本（含锚点聚焦损失）
=================================================================
相比旧版的核心改动：

1. anchor_focus_loss（新增）
   让 anchor_selector 的 MLP 打分与热源强度正相关，
   使锚点真正学会跟随激光位置。
   loss = -mean( scores[topk] )  s.t. scores 由 source_terms 监督

2. m_anchors 自动缩放（保留）
   max(64, N // 15) ≈ 6-7% of N

3. 所有其他修复保留（DDP、NaN 处理、指标计算等）
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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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


def all_reduce_metrics(metrics: Dict[str, float],
                       device: torch.device,
                       world_size: int) -> Dict[str, float]:
    if world_size <= 1:
        return metrics
    keys   = list(metrics.keys())
    tensor = torch.tensor([metrics[k] for k in keys],
                           dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return {k: float(tensor[i]) for i, k in enumerate(keys)}


# ─────────────────────────────────────────────────────────────────
# 锚点聚焦辅助损失
# ─────────────────────────────────────────────────────────────────

def compute_anchor_focus_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    m_anchors: int,
) -> Optional[torch.Tensor]:
    """
    让 anchor_selector 的 MLP 打分与热源强度正相关。

    做法：
      1. 取当前 batch 最后时刻的 source_terms 作为监督信号 q (N,)
      2. 用 anchor_selector.score_nodes() 得到可微打分 scores (N,)
      3. 损失 = -cosine_similarity(scores, q)
         等价于让 scores 的排序与 q 的排序对齐

    这个损失直接作用在 MLP 参数上，使得打分高的节点
    对应热源强度大的位置（即激光附近）。
    """
    raw = model.module if hasattr(model, "module") else model
    selector = getattr(raw, "anchor_selector", None)
    if selector is None or not hasattr(selector, "score_nodes"):
        return None

    # 提取热源强度（最后时刻）
    if "source_terms" not in batch:
        return None

    st = batch["source_terms"]     # (B, T, N, 1) 或 (T, N, 1)
    if st.dim() == 4:              # (B, T, N, C)  ← DDP 常见形状
        q = st[0, -1, :, 0]       # (N,)
    elif st.dim() == 3:            # (T, N, C) 或 (B, T, N)
        # C 维通常很小（1），N 维很大；用 shape[-1] 区分
        if st.shape[-1] <= 4:      # 最后维是 C → (T, N, C)
            q = st[-1, :, 0]       # (N,)
        else:                      # 最后维是 N → (B, T, N)
            q = st[0, -1]          # (N,)
    elif st.dim() == 2:            # (T, N)
        q = st[-1]                 # (N,)
    else:
        q = st.squeeze()

    if q.max() < 1e-8:            # 该时间步没有激光输入
        return None

    # 节点坐标
    nodes = batch.get("nodes", batch.get("pos", None))
    if nodes is None:
        return None
    if nodes.dim() == 3:
        nodes = nodes.squeeze(0)   # (N, 3)

    # 可选：温度场
    temperature = None
    if "initial_conditions" in batch:
        ic = batch["initial_conditions"].squeeze()
        temperature = ic[:, 0] if ic.dim() == 2 else ic

    # 可微打分
    scores = selector.score_nodes(
        nodes=nodes,
        source_q=q,
        temperature=temperature,
    )                              # (N,) 可微

    # 归一化后计算 cosine 相似度（让 scores 排序对齐 q）
    s_norm = F.normalize(scores.unsqueeze(0), dim=1)
    q_norm = F.normalize(q.float().unsqueeze(0), dim=1)
    anchor_focus_loss = 1.0 - (s_norm * q_norm).sum()  # ∈ [0, 2]，越小越好

    return anchor_focus_loss


# ─────────────────────────────────────────────────────────────────
# Epoch
# ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, is_train,
              scaler=None, rank=0, world_size=1, epoch_desc="train",
              anchor_loss_weight=0.1):
    model.train() if is_train else model.eval()
    sum_mse = sum_rne = sum_aloss = 0.0
    n_ok = n_nan = n_aloss = 0

    show_bar = (rank == 0 and HAS_TQDM)
    bar = tqdm(loader, desc=epoch_desc, leave=False,
               ncols=100, disable=not show_bar)

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

            # ── 锚点聚焦辅助损失（仅训练阶段）────────────────────
            aloss_val = 0.0
            if is_train and anchor_loss_weight > 0:
                with torch.enable_grad():
                    af = compute_anchor_focus_loss(
                        model, batch, device, m_anchors=0)
                if af is not None and torch.isfinite(af):
                    loss = loss + anchor_loss_weight * af
                    aloss_val = af.item()
                    sum_aloss += aloss_val
                    n_aloss   += 1

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
                bar.set_postfix(
                    mse=f"{mse.item():.2f}",
                    rne=f"{rne.item():.4f}",
                    af=f"{aloss_val:.3f}" if aloss_val else "-")

    if show_bar:
        bar.close()

    agg = all_reduce_metrics(
        {"mse": sum_mse, "rne": sum_rne,
         "aloss": sum_aloss,
         "n_ok": float(n_ok), "n_nan": float(n_nan),
         "n_aloss": float(n_aloss)},
        device, world_size)
    d      = max(int(agg["n_ok"]), 1)
    d_al   = max(int(agg["n_aloss"]), 1)
    return {
        "mse"          : agg["mse"]   / d,
        "rne"          : agg["rne"]   / d,
        "anchor_loss"  : agg["aloss"] / d_al,
        "nan_batches"  : int(agg["n_nan"]),
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
        all_keys = sorted(
            [k for k in f.keys() if k.startswith('trajectory_')],
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

    # 自动缩放锚点数
    if args.m_anchors is None:
        args.m_anchors = max(64, train_ds.N // 15)

    if is_main:
        print(f"╔{'═'*68}╗")
        print(f"║  PhysHGNet3D（新版 anchor selector）  "
              f"N={train_ds.N}  {world_size} GPU(s)")
        print(f"║  m_anchors={args.m_anchors} "
              f"({args.m_anchors/train_ds.N*100:.1f}% of N)  "
              f"anchor_loss_weight={args.anchor_loss_weight}")
        print(f"╚{'═'*68}╝")
        print(train_ds.info())

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
        n_total = raw_model.num_parameters()
        n_sel   = sum(p.numel() for p in
                      raw_model.anchor_selector.parameters())
        print(f"  总参数量: {n_total:,}  其中 anchor_selector: {n_sel:,}")

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
                      desc="PhysHGNet3D", unit="ep", ncols=100)
                 if (is_main and HAS_TQDM) else range(start_epoch, args.epochs))

    for epoch in epoch_bar:
        if world_size > 1:
            tr_sampler.set_epoch(epoch)

        # anchor_loss_weight 线性预热：前 10 epoch 从 0 增到目标值
        # 避免训练初期 anchor loss 破坏主损失收敛
        warmup_epochs = min(10, args.epochs // 5)
        eff_alw = args.anchor_loss_weight * min(
            1.0, (epoch + 1) / max(warmup_epochs, 1))

        t0   = time.time()
        tr_m = run_epoch(model, tr_loader, optimizer, device,
                         is_train=True,  scaler=scaler,
                         rank=rank, world_size=world_size,
                         epoch_desc=f"  train ep{epoch+1}",
                         anchor_loss_weight=eff_alw)
        va_m = run_epoch(model, va_loader, None, device,
                         is_train=False,
                         rank=rank, world_size=world_size,
                         epoch_desc=f"  val   ep{epoch+1}",
                         anchor_loss_weight=0.0)
        scheduler.step()

        # ── VIZ PATCH：周期性保存锚点快照 ────────────────────────
        _VIZ_EPOCHS = set(
            list(range(1, 6)) +
            list(range(5, args.epochs + 1, max(1, args.epochs // 8))))
        if is_main and (epoch + 1) in _VIZ_EPOCHS:
            _snap_dir = Path(args.ckpt_dir) / f"anchor_snapshots_{train_ds.N}"
            _snap_dir.mkdir(exist_ok=True)
            _snap_p = _snap_dir / f"epoch_{epoch+1:04d}_{train_ds.N}.pth"
            torch.save({
                "model"    : raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch"    : epoch,
                "best_rne" : best_rne,
                "config"   : model_cfg,
            }, _snap_p)
            _msg = f"  📸 锚点快照 → {_snap_p.name}"
            tqdm.write(_msg) if HAS_TQDM else print(_msg)

        if is_main:
            elapsed = time.time() - t0
            nan_s   = f" NaN={tr_m['nan_batches']}" if tr_m["nan_batches"] else ""
            al_s    = f" af={tr_m['anchor_loss']:.4f}"
            line    = (f"Ep {epoch+1:>4}/{args.epochs} | "
                       f"tr MSE={tr_m['mse']:>9.2f} RNE={tr_m['rne']:.4f}{al_s} | "
                       f"va MSE={va_m['mse']:>9.2f} RNE={va_m['rne']:.4f} | "
                       f"{elapsed:.1f}s{nan_s}")
            if HAS_TQDM:
                epoch_bar.set_postfix(
                    va_rne=f"{va_m['rne']:.4f}",
                    af=f"{tr_m['anchor_loss']:.3f}")
                tqdm.write(line)
            else:
                print(line)

            log.append({
                "epoch"       : epoch + 1,
                "train_mse"   : tr_m["mse"],
                "train_rne"   : tr_m["rne"],
                "anchor_loss" : tr_m["anchor_loss"],
                "val_mse"     : va_m["mse"],
                "val_rne"     : va_m["rne"],
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
                msg = f"  ✓ 保存最优 (val RNE={best_rne:.4f})"
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
    p = argparse.ArgumentParser(
        description="PhysHGNet3D 训练（新版可学习 anchor selector）")
    p.add_argument("--n_nodes",      type=int,   default=4000)
    p.add_argument("--data_dir",     type=str,   default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir",     type=str,   default="checkpoints/phys_hgnet_3d")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--window_size",  type=int,   default=10)
    p.add_argument("--stride",       type=int,   default=None)
    p.add_argument("--m_anchors",    type=int,   default=None,
                   help="None=自动 max(64, N//15)")
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--n_layers",     type=int,   default=5)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--resume",       action="store_true")
    # 锚点聚焦损失权重
    p.add_argument("--anchor_loss_weight", type=float, default=0.05,
                   help="anchor_focus_loss 的权重，0 = 关闭。"
                        "建议范围 0.01~0.1，前 10 epoch 线性预热")
    p.add_argument("--no_physics_anchor", action="store_true")
    p.add_argument("--no_learned_coarse", action="store_true")
    p.add_argument("--no_dual_scale",     action="store_true")
    p.add_argument("--no_virtual_nodes",  action="store_true")
    args = p.parse_args()
    if args.stride is None:
        args.stride = max(1, args.window_size // 2)
    return args

if __name__ == "__main__":
    train(parse_args())
