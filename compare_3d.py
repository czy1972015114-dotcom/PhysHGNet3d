"""
compare_3d.py — DGNet3D vs PhysHGNet3D 统一对比评测
=====================================================

在完全相同的测试集、相同设备、相同 batch 接口下，
对两个模型进行精度 / 效率的对比评测。

输出指标
--------
  MSE  ± std        : 均方误差
  RNE  ± std        : 相对 L2 误差
  推理时间 (s/batch) : 每个 batch 的平均推理时间
  GPU 峰值显存 (MiB) : 推理阶段 GPU 显存峰值
  参数量             : 可训练参数总数

输出文件
--------
  comparison_results_3d_{N}.json

使用方法
--------
  # 基本用法
  python compare_3d.py --n_nodes 5000

  # 指定检查点目录
  python compare_3d.py --n_nodes 5000 \\
      --phys_ckpt_dir checkpoints/phys_hgnet_3d \\
      --dgnet_ckpt_dir checkpoints/dgnet_3d

  # 不加载检查点（用随机权重验证接口一致性）
  python compare_3d.py --n_nodes 5000 --no_checkpoint
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ── 两个模型 ──────────────────────────────────────────────────────
from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D
from dgnet_3d      import DGNet3D, DEFAULT_CONFIG_DGNET3D

# ── 数据 ─────────────────────────────────────────────────────────
from dataset_3d import (
    LaserHardening3DDataset,
    collate_fn_3d,
    find_h5_file,
)
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────
# 工具：将 batch dict 迁移到 device
# ─────────────────────────────────────────────────────────────────

def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            # 递归处理嵌套 dict（boundary_info / L_physics）
            out[k] = {
                k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2
                for k2, v2 in v.items()
            }
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────
# 评测单个模型（接口统一，model 只需实现 model(batch) → {"u_final":...}）
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    n_warmup: int = 2,
    label:    str = "model",
) -> Dict[str, Any]:
    """
    在 loader 上评测模型，收集：
      - MSE / RNE（每个 batch）
      - 推理时间（跳过 warmup batch）
      - GPU 峰值显存
    """
    model.eval()
    model.to(device)

    mse_list:  List[float] = []
    rne_list:  List[float] = []
    time_list: List[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for i, batch in enumerate(loader):
        batch  = to_device(batch, device)
        target = batch["targets"]      # (B, T, N, 1)

        # ── warm-up：不计时，但跑推理以预热 CUDA ─────────────────
        if i < n_warmup:
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            continue

        # ── 正式计时 ─────────────────────────────────────────────
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        out  = model(batch)
        pred = out["u_final"]           # (B, T, N, 1)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        # ── 指标 ─────────────────────────────────────────────────
        mse = ((pred - target) ** 2).mean().item()
        rne = ((pred - target).norm() / target.norm().clamp(min=1e-8)).item()

        mse_list.append(mse)
        rne_list.append(rne)
        time_list.append(elapsed)

    peak_mem_mib = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device.type == "cuda" else 0.0)

    mse_a  = np.array(mse_list)
    rne_a  = np.array(rne_list)
    time_a = np.array(time_list)

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "label"         : label,
        "n_params"      : n_params,
        "n_batches_eval": len(mse_list),
        "mse_mean"      : float(mse_a.mean())  if len(mse_a)  > 0 else 0.,
        "mse_std"       : float(mse_a.std())   if len(mse_a)  > 0 else 0.,
        "rne_mean"      : float(rne_a.mean())  if len(rne_a)  > 0 else 0.,
        "rne_std"       : float(rne_a.std())   if len(rne_a)  > 0 else 0.,
        "infer_time_s"  : float(time_a.mean()) if len(time_a) > 0 else 0.,
        "peak_mem_mib"  : float(peak_mem_mib),
    }


# ─────────────────────────────────────────────────────────────────
# 模型加载 & 构建
# ─────────────────────────────────────────────────────────────────

def load_or_init_phys_hgnet_3d(
    ckpt_path: Optional[str],
    device: torch.device,
    no_checkpoint: bool = False,
) -> PhysHGNet3D:
    """加载 PhysHGNet3D 检查点，若不存在则用随机初始化（用于接口验证）。"""
    if no_checkpoint or not (ckpt_path and Path(ckpt_path).exists()):
        print(f"  [PhysHGNet3D] 检查点不存在，使用随机初始化")
        m = PhysHGNet3D(DEFAULT_CONFIG_3D)
    else:
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        cfg   = {**DEFAULT_CONFIG_3D, **ckpt.get("config", {}), "spatial_dim": 3}
        m     = PhysHGNet3D(cfg)
        m.load_state_dict(ckpt["model"])
        print(f"  [PhysHGNet3D] 加载检查点: {ckpt_path}")

    m.to(device).eval()
    print(f"  参数量: {sum(p.numel() for p in m.parameters()):,}")
    return m


def load_or_init_dgnet_3d(
    ckpt_path: Optional[str],
    device: torch.device,
    no_checkpoint: bool = False,
) -> DGNet3D:
    """加载 DGNet3D 检查点，若不存在则用随机初始化。"""
    if no_checkpoint or not (ckpt_path and Path(ckpt_path).exists()):
        print(f"  [DGNet3D] 检查点不存在，使用随机初始化")
        m = DGNet3D(DEFAULT_CONFIG_DGNET3D)
    else:
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        cfg   = {**DEFAULT_CONFIG_DGNET3D, **ckpt.get("config", {})}
        m     = DGNet3D(cfg)
        m.load_state_dict(ckpt["model"])
        print(f"  [DGNet3D] 加载检查点: {ckpt_path}")

    m.to(device).eval()
    print(f"  参数量: {sum(p.numel() for p in m.parameters()):,}")
    return m


# ─────────────────────────────────────────────────────────────────
# 打印对比表格
# ─────────────────────────────────────────────────────────────────

def print_comparison_table(results: List[Dict[str, Any]]):
    sep = "─" * 80
    print(f"\n{sep}")
    print(f"  {'指标':<22}{'DGNet3D':>18}{'PhysHGNet3D':>18}{'改善幅度':>16}")
    print(sep)

    if len(results) < 2:
        print("  （仅一个模型可用，无法对比）")
        return

    # 约定 results[0]=PhysHGNet3D, results[1]=DGNet3D
    our  = next((r for r in results if "PhysHGNet3D" in r["label"]), results[0])
    base = next((r for r in results if "DGNet3D"     in r["label"]), results[1])

    def pct(b, o, higher_is_better=False):
        if b == 0:
            return "—"
        delta = (b - o) / b * 100
        if higher_is_better:
            delta = -delta
        sign = "↓" if delta > 0 else "↑"
        return f"{abs(delta):.1f}% {sign}"

    rows = [
        ("MSE (均值)",
         f"{base['mse_mean']:.4f}±{base['mse_std']:.4f}",
         f"{our['mse_mean']:.4f}±{our['mse_std']:.4f}",
         pct(base['mse_mean'], our['mse_mean'])),
        ("RNE (均值)",
         f"{base['rne_mean']:.4f}±{base['rne_std']:.4f}",
         f"{our['rne_mean']:.4f}±{our['rne_std']:.4f}",
         pct(base['rne_mean'], our['rne_mean'])),
        ("推理时间 (s/batch)",
         f"{base['infer_time_s']:.3f}",
         f"{our['infer_time_s']:.3f}",
         f"{base['infer_time_s']/max(our['infer_time_s'],1e-9):.1f}× 加速"),
        ("GPU 峰值显存 (MiB)",
         f"{base['peak_mem_mib']:.1f}",
         f"{our['peak_mem_mib']:.1f}",
         pct(base['peak_mem_mib'], our['peak_mem_mib'])),
        ("参数量",
         f"{base['n_params']:,}",
         f"{our['n_params']:,}",
         pct(base['n_params'], our['n_params'])),
    ]

    for name, bval, oval, improv in rows:
        print(f"  {name:<22}{bval:>18}{oval:>18}{improv:>16}")

    print(sep)


# ─────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(
        f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu"
    )
    print(f"=== 3D 对比评测 | N≈{args.n_nodes} | device={device} ===\n")

    # ── 数据 ─────────────────────────────────────────────────────
    h5_path = find_h5_file(args.data_dir, args.n_nodes)
    ds      = LaserHardening3DDataset(
        str(h5_path),
        n_time_steps = args.n_time_steps,
    )
    loader  = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collate_fn_3d,
        num_workers = 0,
    )
    N = ds.N
    print(ds.info())

    # ── 检查点路径 ────────────────────────────────────────────────
    phys_ckpt  = Path(args.phys_ckpt_dir)  / f"best_{N}.pth"
    dgnet_ckpt = Path(args.dgnet_ckpt_dir) / f"best_{N}.pth"

    # ── 模型 ─────────────────────────────────────────────────────
    print("\n[1/2] 加载 PhysHGNet3D...")
    model_phys  = load_or_init_phys_hgnet_3d(
        str(phys_ckpt), device, no_checkpoint=args.no_checkpoint
    )

    print("\n[2/2] 加载 DGNet3D...")
    model_dgnet = load_or_init_dgnet_3d(
        str(dgnet_ckpt), device, no_checkpoint=args.no_checkpoint
    )

    # ── 评测 ─────────────────────────────────────────────────────
    print(f"\n── 评测 PhysHGNet3D (warmup={args.warmup}) ──")
    res_phys  = evaluate_model(model_phys,  loader, device,
                                n_warmup=args.warmup, label="PhysHGNet3D")

    print(f"\n── 评测 DGNet3D (warmup={args.warmup}) ──")
    res_dgnet = evaluate_model(model_dgnet, loader, device,
                                n_warmup=args.warmup, label="DGNet3D")

    results = [res_phys, res_dgnet]

    # ── 打印对比表格 ──────────────────────────────────────────────
    print_comparison_table(results)

    # ── 保存 JSON ────────────────────────────────────────────────
    out = {
        "n_nodes"    : N,
        "device"     : str(device),
        "n_time_steps": args.n_time_steps,
        "batch_size" : args.batch_size,
        "results"    : results,
    }
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comparison_results_3d_{N}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 对比结果已保存至 {out_path}")


# ─────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DGNet3D vs PhysHGNet3D 对比评测")
    p.add_argument("--n_nodes",         type=int,   default=5000)
    p.add_argument("--data_dir",        type=str,   default="data_laser_hardening_3d")
    p.add_argument("--phys_ckpt_dir",   type=str,   default="checkpoints/phys_hgnet_3d")
    p.add_argument("--dgnet_ckpt_dir",  type=str,   default="checkpoints/dgnet_3d")
    p.add_argument("--out_dir",         type=str,   default="results_3d")
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--n_time_steps",    type=int,   default=20)
    p.add_argument("--warmup",          type=int,   default=2)
    p.add_argument("--gpu",             type=int,   default=0)
    p.add_argument("--no_checkpoint",   action="store_true",
                   help="不加载检查点，使用随机初始化（仅验证接口）")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
