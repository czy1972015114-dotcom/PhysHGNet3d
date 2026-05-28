"""
compare_3d.py — DGNet3D vs PhysHGNet3D 统一对比评测
=====================================================

修复：LaserHardening3DDataset 参数从 n_time_steps 改为 window_size。

使用方法
--------
  python compare_3d.py --n_nodes 1000 \\
      --phys_ckpt_dir  checkpoints/phys_hgnet_3d \\
      --dgnet_ckpt_dir checkpoints/dgnet_3d
"""

import argparse, json, time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D
from dgnet_3d      import DGNet3D, DEFAULT_CONFIG_DGNET3D
from dataset_3d    import LaserHardening3DDataset, collate_fn_3d, find_h5_file
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────

def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2
                      for k2, v2 in v.items()}
        else:
            out[k] = v
    return out

# ─────────────────────────────────────────────────────────────────
# 评测单模型
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    n_warmup: int = 2,
    label:    str = "model",
) -> Dict[str, Any]:
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
        target = batch["targets"]

        if i < n_warmup:
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            continue

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        out  = model(batch)
        pred = out["u_final"]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        mse = ((pred - target) ** 2).mean().item()
        rne = ((pred - target).norm() / target.norm().clamp(min=1e-8)).item()
        mse_list.append(mse)
        rne_list.append(rne)
        time_list.append(elapsed)

    peak_mem = (torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else 0.0)

    mse_a  = np.array(mse_list)
    rne_a  = np.array(rne_list)
    time_a = np.array(time_list)
    n_par  = sum(p.numel() for p in model.parameters())

    return {
        "label"         : label,
        "n_params"      : n_par,
        "n_batches_eval": len(mse_list),
        "mse_mean"      : float(mse_a.mean())  if len(mse_a)  > 0 else 0.,
        "mse_std"       : float(mse_a.std())   if len(mse_a)  > 0 else 0.,
        "rne_mean"      : float(rne_a.mean())  if len(rne_a)  > 0 else 0.,
        "rne_std"       : float(rne_a.std())   if len(rne_a)  > 0 else 0.,
        "infer_time_s"  : float(time_a.mean()) if len(time_a) > 0 else 0.,
        "peak_mem_mib"  : float(peak_mem),
    }

# ─────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────

def load_or_init_phys_hgnet_3d(ckpt_path: Optional[str], device, no_checkpoint=False):
    if no_checkpoint or not (ckpt_path and Path(ckpt_path).exists()):
        print(f"  [PhysHGNet3D] 检查点不存在，使用随机初始化")
        m = PhysHGNet3D(DEFAULT_CONFIG_3D)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = {**DEFAULT_CONFIG_3D, **ckpt.get("config", {}), "spatial_dim": 3}
        m    = PhysHGNet3D(cfg)
        m.load_state_dict(ckpt["model"])
        print(f"  [PhysHGNet3D] 加载: {ckpt_path}")
    m.to(device).eval()
    print(f"  参数量: {sum(p.numel() for p in m.parameters()):,}")
    return m


def load_or_init_dgnet_3d(ckpt_path: Optional[str], device, no_checkpoint=False):
    if no_checkpoint or not (ckpt_path and Path(ckpt_path).exists()):
        print(f"  [DGNet3D] 检查点不存在，使用随机初始化")
        m = DGNet3D(DEFAULT_CONFIG_DGNET3D)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = {**DEFAULT_CONFIG_DGNET3D, **ckpt.get("config", {})}
        m    = DGNet3D(cfg)
        m.load_state_dict(ckpt["model"])
        print(f"  [DGNet3D] 加载: {ckpt_path}")
    m.to(device).eval()
    print(f"  参数量: {sum(p.numel() for p in m.parameters()):,}")
    return m

# ─────────────────────────────────────────────────────────────────
# 打印对比表格
# ─────────────────────────────────────────────────────────────────

def print_comparison_table(results: List[Dict[str, Any]]):
    sep = "─" * 82
    print(f"\n{sep}")
    print(f"  {'指标':<24}{'DGNet3D (基线)':>18}{'PhysHGNet3D':>18}{'改善':>16}")
    print(sep)

    if len(results) < 2:
        print("  （仅一个模型可用，无法对比）")
        return

    our  = next((r for r in results if "PhysHGNet3D" in r["label"]), results[0])
    base = next((r for r in results if "DGNet3D"     in r["label"]), results[1])

    def pct(b, o):
        if b == 0: return "—"
        d = (b - o) / b * 100
        return f"{abs(d):.1f}% {'↓' if d > 0 else '↑'}"

    rows = [
        ("MSE (均值±std)",
         f"{base['mse_mean']:.3f}±{base['mse_std']:.3f}",
         f"{our['mse_mean']:.3f}±{our['mse_std']:.3f}",
         pct(base['mse_mean'], our['mse_mean'])),
        ("RNE (均值±std)",
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
        print(f"  {name:<24}{bval:>18}{oval:>18}{improv:>16}")
    print(sep)

# ─────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(
        f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    print(f"=== 3D 对比评测 | N≈{args.n_nodes} | device={device} ===\n")

    # ── 数据（使用完整轨迹作为评测样本，window_size=full）──────────
    h5_path = find_h5_file(args.data_dir, args.n_nodes)
    ds      = LaserHardening3DDataset(
        str(h5_path),
        window_size = args.window_size,   # 修复：原 n_time_steps → window_size
        stride      = args.window_size,   # 评测时不重叠（stride=window_size）
    )
    loader  = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_3d, num_workers=0)
    N = ds.N
    print(ds.info())

    # ── 检查点路径 ────────────────────────────────────────────────
    phys_ckpt  = Path(args.phys_ckpt_dir)  / f"best_{N}.pth"
    dgnet_ckpt = Path(args.dgnet_ckpt_dir) / f"best_{N}.pth"

    # ── 加载模型 ──────────────────────────────────────────────────
    print("\n[1/2] 加载 PhysHGNet3D...")
    model_phys  = load_or_init_phys_hgnet_3d(
        str(phys_ckpt), device, no_checkpoint=args.no_checkpoint)

    print("\n[2/2] 加载 DGNet3D...")
    model_dgnet = load_or_init_dgnet_3d(
        str(dgnet_ckpt), device, no_checkpoint=args.no_checkpoint)

    # ── 评测 ─────────────────────────────────────────────────────
    print(f"\n── 评测 PhysHGNet3D (warmup={args.warmup}) ──")
    res_phys  = evaluate_model(model_phys,  loader, device,
                                n_warmup=args.warmup, label="PhysHGNet3D")

    print(f"\n── 评测 DGNet3D (warmup={args.warmup}) ──")
    res_dgnet = evaluate_model(model_dgnet, loader, device,
                                n_warmup=args.warmup, label="DGNet3D")

    results = [res_phys, res_dgnet]
    print_comparison_table(results)

    # ── 保存 JSON ────────────────────────────────────────────────
    out = {
        "n_nodes"    : N,
        "device"     : str(device),
        "window_size": args.window_size,
        "batch_size" : args.batch_size,
        "results"    : results,
    }
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comparison_results_3d_{N}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存至 {out_path}")

# ─────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DGNet3D vs PhysHGNet3D 对比")
    p.add_argument("--n_nodes",         type=int,   default=1000)
    p.add_argument("--data_dir",        type=str,   default="data_laser_hardening_3d")
    p.add_argument("--phys_ckpt_dir",   type=str,   default="checkpoints/phys_hgnet_3d")
    p.add_argument("--dgnet_ckpt_dir",  type=str,   default="checkpoints/dgnet_3d")
    p.add_argument("--out_dir",         type=str,   default="results_3d")
    p.add_argument("--batch_size",      type=int,   default=2)
    # 修复：window_size 替代旧的 n_time_steps
    p.add_argument("--window_size",     type=int,   default=10,
                   help="评测时使用的时步数（原 n_time_steps）")
    p.add_argument("--warmup",          type=int,   default=2)
    p.add_argument("--gpu",             type=int,   default=0)
    p.add_argument("--no_checkpoint",   action="store_true",
                   help="不加载检查点（仅验证接口）")
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
