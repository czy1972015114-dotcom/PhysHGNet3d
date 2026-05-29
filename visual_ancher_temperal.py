#!/usr/bin/env python3
"""
visual_ancher_temperal.py — PhysHGNet3D 锚点时序可视化

功能
----
1. 加载已训练的 PhysHGNet3D checkpoint
2. 从测试轨迹中读取若干时步
3. 在每个时步重置图缓存（强制重建），捕获锚点坐标 + 残差强度
4. 生成 3D 散点图序列（锚点位置 + 热源强度热图）
5. 保存为独立 PNG 帧，并合成 GIF 动画

用法
----
# 基本用法
python visual_ancher_temperal.py \
    --ckpt checkpoints/phys_hgnet_3d/best_6000.pth \
    --h5   data_laser_hardening_3d/pde_trajectories_3d_N6000.h5 \
    --traj_idx 32 \
    --out  figs/anchor_temporal

# 自定义时步范围
python visual_ancher_temperal.py \
    --ckpt checkpoints/phys_hgnet_3d/best_6000.pth \
    --h5   data_laser_hardening_3d/pde_trajectories_3d_N6000.h5 \
    --traj_idx 32 --t_start 0 --t_end 60 --t_step 5 \
    --out  figs/anchor_temporal
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠ Pillow 未安装，将跳过 GIF 合成。pip install Pillow")

# ─────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", DEFAULT_CONFIG_3D)
    cfg["spatial_dim"] = 3

    model = PhysHGNet3D(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    print(f"✓ 加载 checkpoint: {ckpt_path}")
    print(f"  best_rne={ckpt.get('best_rne', 'N/A'):.4f}  epoch={ckpt.get('epoch', 'N/A')}")
    return model


# ─────────────────────────────────────────────────────────────────
# 数据加载（单条轨迹）
# ─────────────────────────────────────────────────────────────────

def load_trajectory(h5_path: str, traj_idx: int, device: torch.device):
    """读取单条测试轨迹的全部时步。"""
    with h5py.File(h5_path, "r") as f:
        meta    = f["mesh_meta"]
        nodes   = torch.tensor(meta["nodes"][:],       dtype=torch.float32)
        edges   = torch.tensor(meta["edges"][:],       dtype=torch.long)
        tets    = torch.tensor(meta["tets"][:],        dtype=torch.long)
        nvol    = torch.tensor(meta["node_volumes"][:],dtype=torch.float32)
        tp      = torch.tensor(meta["time_points"][:], dtype=torch.float32)

        all_keys = sorted(
            [k for k in f.keys() if k.startswith("trajectory_")],
            key=lambda x: int(x.split("_")[1]))
        key  = all_keys[traj_idx]
        g    = f[key]
        u    = torch.tensor(g["node_features"][:], dtype=torch.float32)  # (T,N,1)
        src  = torch.tensor(g["source_terms"][:],  dtype=torch.float32)  # (T,N,1)

        bnd_idx = torch.tensor(g["boundary_info"]["dirichlet"]["indices"][:], dtype=torch.long)
        bnd_val = torch.tensor(g["boundary_info"]["dirichlet"]["values"][:],  dtype=torch.float32)

    print(f"✓ 轨迹 {key}: T={u.shape[0]}, N={u.shape[1]}")
    return (nodes.to(device), edges.to(device), tets.to(device),
            nvol.to(device), tp.to(device), u.to(device), src.to(device),
            {"dirichlet": {"indices": bnd_idx.to(device),
                           "values": bnd_val.to(device)}})


# ─────────────────────────────────────────────────────────────────
# 逐时步提取锚点
# ─────────────────────────────────────────────────────────────────

def extract_anchors_temporal(
    model,
    nodes, edges, tets, nvol, tp, u_traj, src_traj, bnd_info,
    t_indices: List[int],
    device: torch.device,
) -> List[Dict]:
    """
    对每个时步 t，以 u[t] 为初始条件、src[t:t+2] 为源项，
    单独运行一步 forward，读取此时的锚点状态。

    返回：每个时步的字典，包含：
      - anchor_coords   (m, 3) numpy
      - residual_score  (m,)   numpy，每锚点的平均残差大小
      - source_at_anchors (m,) numpy，锚点处的热源强度
      - t               时步索引
    """
    from physics_3d import build_operator_3d

    # 预构建 L_physics（全轨迹共享）
    with torch.no_grad():
        L_phys = build_operator_3d(nodes.float(), tets.long())

    N = nodes.shape[0]
    node_type = torch.zeros(N, dtype=torch.long, device=device)

    snapshots = []
    T = u_traj.shape[0]

    for t in t_indices:
        if t >= T - 1:
            continue

        # 构建单步 batch
        batch = {
            "nodes"             : nodes,
            "edges"             : edges,
            "tets"              : tets,
            "node_volumes"      : nvol,
            "node_type"         : node_type,
            "L_physics"         : L_phys,
            "initial_conditions": u_traj[t].unsqueeze(0),        # (1,N,1)
            "source_terms"      : src_traj[t:t+2].unsqueeze(0),  # (1,2,N,1)
            "time_points"       : tp[t:t+2],
            "boundary_info"     : bnd_info,
            "targets"           : u_traj[t:t+2].unsqueeze(0),
        }

        # ★ 强制重建图缓存，捕获此时步的锚点
        model._graph_cache = None
        model._residual_cache = None
        model._grad_norm_cache = None

        with torch.no_grad():
            # 触发 _build_graph
            from physics_3d import build_operator_3d
            nodes_   = batch["nodes"]
            edges_   = batch["edges"]
            L_phy_   = batch["L_physics"]
            nvol_    = batch.get("node_volumes")
            nt_      = batch.get("node_type")
            u_init_  = batch["initial_conditions"]

            gc = model._build_graph(
                nodes_, edges_, L_phy_,
                node_volumes=nvol_,
                node_type=nt_,
                u_init=u_init_[0])

        anchor_idx    = gc["anchor_idx"].cpu().numpy()          # (m,)
        anchor_coords = gc["anchor_coords"].cpu().numpy()       # (m, 3)
        m = anchor_idx.shape[0]

        # 残差得分（用当前 u 的 L*u 近似）
        with torch.no_grad():
            anchor_feats = model.fine_encoder(
                nodes_, gc["fine_ei"], gc["fine_ea"], nvol_, nt_
            )[gc["anchor_idx"]]
            u0 = u_traj[t, :, 0].float()
            try:
                Lu = model._Leff_matvec(u0, gc, anchor_feats.detach())
                res = (Lu + src_traj[t, :, 0]).abs()
            except Exception:
                res = torch.zeros(N, device=device)

        res_at_anchors = res[gc["anchor_idx"]].cpu().numpy()

        # 热源强度（节点平均）
        q_all = src_traj[t, :, 0].cpu().numpy()
        q_anchors = q_all[anchor_idx]

        snapshots.append({
            "t"                  : t,
            "anchor_coords"      : anchor_coords,
            "residual_score"     : res_at_anchors,
            "source_at_anchors"  : q_anchors,
            "q_all"              : q_all,
            "u_all"              : u_traj[t, :, 0].cpu().numpy(),
            "nodes_np"           : nodes.cpu().numpy(),
        })

        print(f"  t={t:3d}  m={m}  "
              f"res_max={res_at_anchors.max():.3f}  "
              f"q_max={q_anchors.max():.3f}")

    return snapshots


# ─────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────

def _make_axis_equal_3d(ax, nodes_np):
    """让 3D 图的三个轴比例相等。"""
    xyz_min = nodes_np.min(0)
    xyz_max = nodes_np.max(0)
    ranges  = xyz_max - xyz_min
    center  = (xyz_max + xyz_min) / 2
    R       = ranges.max() / 2 * 1.1
    ax.set_xlim(center[0] - R, center[0] + R)
    ax.set_ylim(center[1] - R, center[1] + R)
    ax.set_zlim(center[2] - R, center[2] + R)


def plot_snapshot(snap: Dict, out_path: str,
                  elev: float = 25, azim: float = -60,
                  colorby: str = "residual"):
    """绘制单帧：全部节点（灰色小点）+ 锚点（彩色大点）。"""
    nodes_np      = snap["nodes_np"]          # (N,3)
    anchor_coords = snap["anchor_coords"]     # (m,3)
    t             = snap["t"]

    if colorby == "residual":
        scores = snap["residual_score"]
        cbar_label = "残差强度"
    elif colorby == "source":
        scores = snap["source_at_anchors"]
        cbar_label = "热源强度"
    else:
        scores = snap["u_all"][snap["nodes_np"].shape[0]  # placeholder
                               if False else 0]
        scores = np.ones(anchor_coords.shape[0])
        cbar_label = ""

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # 全部节点（均匀稀疏采样显示，最多 3000 个，否则太慢）
    N = nodes_np.shape[0]
    step = max(1, N // 3000)
    ax.scatter(nodes_np[::step, 0], nodes_np[::step, 1], nodes_np[::step, 2],
               s=0.3, c="lightgray", alpha=0.25, depthshade=False)

    # 锚点（按 score 着色）
    s_norm = scores / (scores.max() + 1e-8)
    colors = cm.hot(s_norm)
    sc = ax.scatter(anchor_coords[:, 0], anchor_coords[:, 1], anchor_coords[:, 2],
                    s=30, c=s_norm, cmap="hot", vmin=0, vmax=1,
                    depthshade=True, edgecolors="none", zorder=5)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label(cbar_label, fontsize=10)

    _make_axis_equal_3d(ax, nodes_np)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"锚点分布  t={t}  (m={anchor_coords.shape[0]})", fontsize=12)

    ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_gif(frame_paths: List[str], gif_path: str, fps: int = 4):
    if not HAS_PIL:
        print("  跳过 GIF（无 Pillow）")
        return
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True, append_images=frames[1:],
        optimize=False,
        duration=int(1000 / fps),
        loop=0)
    print(f"  ✓ GIF → {gif_path}")


# ─────────────────────────────────────────────────────────────────
# 摘要图（锚点数随时间变化 + 残差最大值）
# ─────────────────────────────────────────────────────────────────

def plot_summary(snapshots: List[Dict], out_path: str):
    ts     = [s["t"] for s in snapshots]
    ms     = [s["anchor_coords"].shape[0] for s in snapshots]
    res_mx = [s["residual_score"].max() for s in snapshots]
    q_mx   = [s["source_at_anchors"].max() for s in snapshots]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(ts, ms, "o-", color="steelblue", ms=4)
    axes[0].set_ylabel("锚点数 m", color="steelblue")
    axes[0].tick_params(axis="y", colors="steelblue")
    axes[0].set_title("PhysHGNet3D 锚点时序摘要")
    axes[0].grid(True, alpha=0.3)

    ax1b = axes[0].twinx()
    ax1b.plot(ts, q_mx, "s--", color="orangered", ms=4)
    ax1b.set_ylabel("锚点最大热源强度", color="orangered")
    ax1b.tick_params(axis="y", colors="orangered")

    axes[1].plot(ts, res_mx, "^-", color="purple", ms=4)
    axes[1].set_ylabel("锚点最大残差强度", color="purple")
    axes[1].set_xlabel("时步 t")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 摘要图 → {out_path}")


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PhysHGNet3D 锚点时序可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt",     required=True,
                   help="PhysHGNet3D checkpoint 路径")
    p.add_argument("--h5",       required=True,
                   help="测试数据 HDF5 文件路径")
    p.add_argument("--traj_idx", type=int, default=32,
                   help="使用第几条轨迹（建议用 val 集轨迹）")
    p.add_argument("--t_start",  type=int, default=0)
    p.add_argument("--t_end",    type=int, default=-1,
                   help="-1 表示全部时步")
    p.add_argument("--t_step",   type=int, default=5,
                   help="每隔几步采样一帧")
    p.add_argument("--out",      default="figs/anchor_temporal",
                   help="输出目录（帧 PNG + GIF）")
    p.add_argument("--colorby",  choices=["residual", "source"],
                   default="residual",
                   help="锚点着色依据：residual=残差强度，source=热源强度")
    p.add_argument("--elev",  type=float, default=25,  help="3D 视角仰角")
    p.add_argument("--azim",  type=float, default=-60, help="3D 视角方位角")
    p.add_argument("--fps",   type=int,   default=4,   help="GIF 帧率")
    p.add_argument("--device",            default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # 加载模型
    model = load_model(args.ckpt, device)

    # 加载轨迹
    (nodes, edges, tets, nvol, tp,
     u_traj, src_traj, bnd_info) = load_trajectory(
        args.h5, args.traj_idx, device)

    T = u_traj.shape[0]
    t_end = T - 1 if args.t_end < 0 else min(args.t_end, T - 1)
    t_indices = list(range(args.t_start, t_end, args.t_step))

    print(f"\n将在 {len(t_indices)} 个时步提取锚点：{t_indices[:5]}{'...' if len(t_indices)>5 else ''}")

    # 提取锚点快照
    print("\n── 提取锚点快照 ──")
    snapshots = extract_anchors_temporal(
        model, nodes, edges, tets, nvol, tp,
        u_traj, src_traj, bnd_info,
        t_indices, device)

    # 绘制每帧
    print("\n── 绘制帧 ──")
    frame_paths = []
    for snap in snapshots:
        t   = snap["t"]
        out_png = frames_dir / f"frame_{t:04d}.png"
        plot_snapshot(snap, str(out_png),
                      elev=args.elev, azim=args.azim,
                      colorby=args.colorby)
        frame_paths.append(str(out_png))
        print(f"  帧 t={t:3d} → {out_png.name}")

    # 合成 GIF
    if frame_paths:
        gif_path = out_dir / "anchor_temporal.gif"
        make_gif(frame_paths, str(gif_path), fps=args.fps)

    # 摘要图
    summary_path = out_dir / "anchor_summary.png"
    plot_summary(snapshots, str(summary_path))

    print(f"\n✓ 完成！输出目录：{out_dir}")
    print(f"  {len(frame_paths)} 帧 PNG  →  {frames_dir}/")
    print(f"  GIF 动画               →  {out_dir}/anchor_temporal.gif")
    print(f"  摘要图                 →  {summary_path}")


if __name__ == "__main__":
    main()
