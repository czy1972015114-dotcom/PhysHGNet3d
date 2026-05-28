"""
visualize_anchors_temporal.py — PhysHGNet3D 锚点时序可视化
============================================================
正确的可视化思路：
  固定一个训练好的模型（best checkpoint），沿一条轨迹的时间步推进，
  展示锚点位置如何随温度场演变而动态移动。

  这才能说明方法有效性：锚点跟随激光路径/高梯度区域移动，
  而非随机均匀分布。

生成的图：
  1. anchor_temporal_grid_<N>.png   — 多时间步对比拼图（论文主图）
  2. anchor_temporal_single_t<T>_<N>.png — 每个时间步的精细四视图
  3. anchor_temporal_trajectory_<N>.png  — 锚点重心轨迹图（量化移动）
  4. anchor_temporal_<N>.html       — 交互式动画（需要 plotly）

使用方法
--------
  python visualize_anchors_temporal.py --n_nodes 10000
  python visualize_anchors_temporal.py --n_nodes 4000 --traj_idx 0 --timesteps 0 2 4 6 8 9
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D
    from dataset_3d import LaserHardening3DDataset, collate_fn_3d, find_h5_file
    from torch.utils.data import DataLoader
    PROJECT_IMPORTED = True
except ImportError as e:
    print(f"[错误] 无法导入项目模块：{e}")
    PROJECT_IMPORTED = False


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def to_device(x, device):
    """递归将 batch 中所有 tensor 移到指定设备（包括嵌套 dict）"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def load_best_model(ckpt_dir: str, n_nodes: int,
                    device: torch.device) -> tuple:
    """加载 best checkpoint，返回 (model, config)"""
    ckpt_path = Path(ckpt_dir) / f"best_{n_nodes}.pth"
    if not ckpt_path.exists():
        # 尝试找最新的 epoch checkpoint
        candidates = sorted(Path(ckpt_dir).glob(f"*{n_nodes}*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"在 {ckpt_dir} 中未找到 N={n_nodes} 的检查点")
        ckpt_path = candidates[-1]
        print(f"  未找到 best checkpoint，使用: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = {**DEFAULT_CONFIG_3D, **ckpt.get("config", {})}
    model = PhysHGNet3D(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"  加载检查点: {ckpt_path.name}")
    print(f"  m_anchors={cfg.get('m_anchors')}  "
          f"best_rne={ckpt.get('best_rne', 'N/A')}")
    return model, cfg


# ═══════════════════════════════════════════════════════════════
# 锚点提取：对每个时间步单独推断
# ═══════════════════════════════════════════════════════════════

def extract_anchor_per_timestep(
    model: nn.Module,
    h5_path: str,
    n_nodes: int,
    traj_idx: int,
    timesteps: List[int],   # 窗口内的时间步索引（0-based）
    window_size: int,
    device: torch.device,
) -> List[Dict]:
    """
    对同一条轨迹的不同时间步，逐步提取锚点索引和温度场。

    策略：
      - 数据集以 stride=1 构建，每个 sample 对应轨迹中一个长度为
        window_size 的窗口。
      - 对每个指定时间步 t，取包含 t 的窗口（窗口末尾 = t），
        这样 targets[-1] 就是 t 时刻的温度。
      - 用 hook 截取 anchor_selector 输出。
    """
    import h5py

    # 重建只含目标轨迹的 dataset，stride=1 以便精细控制
    ds = LaserHardening3DDataset(
        h5_path, window_size=window_size, stride=1,
        traj_indices=[traj_idx])

    results = []
    captured_idx = []

    # 注册 hook
    def hook_fn(module, inputs, output):
        if isinstance(output, torch.Tensor) and output.dtype == torch.int64:
            captured_idx.clear()
            captured_idx.append(output.detach().cpu())
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor) and o.dtype == torch.int64:
                    captured_idx.clear()
                    captured_idx.append(o.detach().cpu())
                    break

    handle = None
    for name, mod in model.named_modules():
        if "anchor_selector" in name.lower():
            handle = mod.register_forward_hook(hook_fn)
            break

    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn_3d, num_workers=0)

    # 收集所有 sample
    all_samples = list(loader)
    T_total = len(all_samples)

    # 坐标从第一个 sample 读取（同一轨迹坐标不变）
    first_batch = to_device(all_samples[0], device)
    coords = first_batch["nodes"].cpu().numpy()  # (N, 3)

    # 归一化坐标到 [0,1]（方便可视化）
    coords_norm = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-8)

    for t in timesteps:
        # 选取窗口末尾 = t 的 sample
        # window 末尾时间步 = sample_index + window_size - 1
        # 所以 sample_index = t - window_size + 1（最小为 0）
        sample_idx = max(0, min(t, T_total - 1))
        batch = to_device(all_samples[sample_idx], device)

        # 清除图缓存，强制每个时间步重新计算锚点
        # （否则模型复用第一次的缓存，anchor_selector 不会再执行）
        raw_model = model.module if hasattr(model, "module") else model
        for attr in ("_graph_cache", "_cache_key",
                     "_residual_cache", "_grad_norm_cache"):
            if hasattr(raw_model, attr):
                setattr(raw_model, attr, None)

        captured_idx.clear()
        with torch.no_grad():
            out = model(batch)

        # 读取当前时间步温度
        tgt = batch["targets"].cpu().numpy()  # (1, T, N, 1)
        tgt = tgt.squeeze()                    # (T, N, 1) or (T, N)
        if tgt.ndim == 3:
            temp = tgt[-1, :, 0]               # 最后时刻，(N,)
        elif tgt.ndim == 2:
            temp = tgt[-1]                      # (N,)
        else:
            temp = tgt

        # 读取激光位置：从 H5 直接按时间步读 source_terms，避免 window 错位
        laser_pos = None
        heat_intensity = None   # 每个节点的热源强度，用于锚点着色
        import h5py as _h5
        with _h5.File(h5_path, "r") as _f:
            traj_key_h5 = f"trajectory_{traj_idx}"
            if "source_terms" in _f[traj_key_h5]:
                q_all = _f[traj_key_h5]["source_terms"][:]   # (T_total, N, 1)
                t_real = min(t, q_all.shape[0] - 1)
                q = q_all[t_real, :, 0]                       # (N,)
                heat_intensity = q
                if q.max() > 0:
                    hot_node = int(np.argmax(q))
                    laser_pos = coords[hot_node]

        # 获取锚点
        if captured_idx:
            anchor_idx = captured_idx[0].numpy().flatten().astype(int)
            anchor_idx = anchor_idx[anchor_idx < len(coords)]
        else:
            print(f"  ⚠ t={t}: hook 未捕获到锚点，跳过")
            continue

        results.append({
            "t"              : t,
            "label"          : f"t = {t+1}",
            "coords"         : coords,
            "coords_norm"    : coords_norm,
            "anchor_idx"     : anchor_idx,
            "temp"           : temp,
            "heat_intensity" : heat_intensity,   # (N,) 热源强度，用于锚点着色
            "laser_pos"      : laser_pos,
            "n_anchors"      : len(anchor_idx),
        })
        print(f"  t={t+1:>3}: 锚点数={len(anchor_idx):>4}  "
              f"温度范围=[{temp.min():.1f}, {temp.max():.1f}] K"
              + (f"  激光位置≈{laser_pos.round(3)}" if laser_pos is not None else ""))

    if handle is not None:
        handle.remove()

    return results


# ═══════════════════════════════════════════════════════════════
# 可视化函数
# ═══════════════════════════════════════════════════════════════

CMAP_TEMP   = "plasma"
COLOR_NODE  = "#C8D6E5"
COLOR_LASER = "#00FF88"


def plot_temporal_grid(snapshots: List[Dict], out_path: str,
                       n_nodes: int, elev=28.0, azim=-55.0):
    """
    多时间步对比拼图：
    上排 — XY 俯视（显示激光扫描路径方向）
    下排 — 3D 透视
    锚点颜色 = 该节点温度，大小 ∝ 局部温度梯度
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    n = len(snapshots)
    fig = plt.figure(figsize=(4.2 * n, 8.5), dpi=150, facecolor="white")
    fig.suptitle(
        f"PhysHGNet3D — Anchor Positions along Trajectory  (N={n_nodes})\n"
        "Anchors follow the high-gradient region as the laser sweeps",
        fontsize=12, fontweight="bold", y=0.99)

    gs_top = gridspec.GridSpec(1, n, top=0.93, bottom=0.52,
                                left=0.03, right=0.96, wspace=0.04)
    gs_bot = gridspec.GridSpec(1, n, top=0.47, bottom=0.04,
                                left=0.03, right=0.96, wspace=0.04)

    # 统一色域
    all_temp = np.concatenate([s["temp"] for s in snapshots])
    norm = Normalize(vmin=all_temp.min(), vmax=all_temp.max())
    heats = [s.get("heat_intensity") for s in snapshots if s.get("heat_intensity") is not None]
    all_temp_heat_max = float(np.concatenate(heats).max()) if heats else 1.0

    for col, snap in enumerate(snapshots):
        coords    = snap["coords"]
        aidx      = snap["anchor_idx"]
        temp      = snap["temp"]
        # 优先用热源强度着色（信号更强），否则用温度
        heat = snap.get("heat_intensity")
        if heat is not None and heat.max() > 0:
            t_anchor  = heat[aidx]
            cmap_use  = "hot"
            norm_heat = Normalize(vmin=0, vmax=all_temp_heat_max)
            norm_use  = norm_heat
        else:
            t_anchor  = temp[aidx]
            cmap_use  = CMAP_TEMP
            norm_use  = norm
        laser_pos = snap["laser_pos"]

        # ── 俯视图 (XY) ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs_top[0, col])
        ax2.scatter(coords[:, 0], coords[:, 1],
                    s=0.5, c=COLOR_NODE, alpha=0.20,
                    linewidths=0, rasterized=True)
        ax2.scatter(coords[aidx, 0], coords[aidx, 1],
                    s=22, c=t_anchor, cmap=cmap_use, norm=norm_use,
                    zorder=5, edgecolors="black", linewidths=0.35, alpha=0.9)
        if laser_pos is not None:
            ax2.scatter(laser_pos[0], laser_pos[1],
                        s=120, marker="*", c=COLOR_LASER, zorder=10,
                        edgecolors="black", linewidths=0.5)
        ax2.set_title(f"{snap['label']}\nM={snap['n_anchors']}",
                       fontsize=10, pad=3)
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_facecolor("#F7F9FB")

        # ── 3D 透视 ───────────────────────────────────────────────
        ax3 = fig.add_subplot(gs_bot[0, col], projection="3d")
        ax3.scatter(*[coords[:, i] for i in range(3)],
                    s=0.3, c=COLOR_NODE, alpha=0.10,
                    linewidths=0, rasterized=True)
        ax3.scatter(*[coords[aidx, i] for i in range(3)],
                    s=25, c=t_anchor, cmap=cmap_use, norm=norm_use,
                    edgecolors="black", linewidths=0.35, alpha=0.9)
        if laser_pos is not None:
            ax3.scatter(*laser_pos, s=150, marker="*", c=COLOR_LASER,
                        zorder=10, edgecolors="black", linewidths=0.5)
        ax3.view_init(elev=elev, azim=azim)
        ax3.set_facecolor("#F7F9FB")
        for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#DDDDDD")
        ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

    # 色条
    sm = plt.cm.ScalarMappable(cmap=CMAP_TEMP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.38, pad=0.005,
                        fraction=0.008, orientation="vertical")
    cbar.set_label("Temperature (K)", fontsize=9)

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#E74C3C", markersize=8, label="Anchor node"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=COLOR_LASER, markersize=10, label="Laser position"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.00))

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  [图] 时序对比图 → {out_path}")


def plot_anchor_centroid_trajectory(snapshots: List[Dict],
                                    out_path: str, n_nodes: int):
    """
    锚点重心轨迹图：
    - X 轴 = 时间步
    - Y 轴 = 锚点重心的 x/y/z 坐标
    - 辅线 = 激光位置（如果有）
    量化地展示锚点重心确实在跟随激光移动。
    """
    import matplotlib.pyplot as plt

    ts        = [s["t"] + 1 for s in snapshots]
    centroids = np.array([s["coords"][s["anchor_idx"]].mean(0)
                          for s in snapshots])
    laser_pos = [s["laser_pos"] for s in snapshots]
    has_laser = any(p is not None for p in laser_pos)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True,
                              facecolor="white", dpi=150)
    fig.suptitle(f"Anchor Centroid vs Laser Position over Time  (N={n_nodes})",
                 fontsize=12, fontweight="bold")

    labels = ["X", "Y", "Z"]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]
    for i, (ax, lab, col) in enumerate(zip(axes, labels, colors)):
        ax.plot(ts, centroids[:, i], "o-", color=col, linewidth=2,
                markersize=5, label=f"Anchor centroid {lab}")
        if has_laser:
            lp = [p[i] if p is not None else np.nan for p in laser_pos]
            ax.plot(ts, lp, "s--", color="gray", linewidth=1.5,
                    markersize=4, alpha=0.7, label=f"Laser {lab}")
        ax.set_ylabel(f"{lab} coord", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFAFA")

    axes[-1].set_xlabel("Time step", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  [图] 锚点重心轨迹图 → {out_path}")


def make_temporal_html(snapshots: List[Dict], out_path: str, n_nodes: int):
    """交互式 HTML：用滑块切换时间步，3D 可旋转"""
    if not HAS_PLOTLY:
        print("  [跳过] plotly 未安装 (pip install plotly)")
        return

    all_temp = np.concatenate([s["temp"] for s in snapshots])
    tmin, tmax = float(all_temp.min()), float(all_temp.max())

    frames = []
    for snap in snapshots:
        coords   = snap["coords"]
        aidx     = snap["anchor_idx"]
        t_anchor = snap["temp"][aidx].tolist()

        traces = [
            go.Scatter3d(
                x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
                z=coords[:, 2].tolist(),
                mode="markers",
                marker=dict(size=1.0, color="#AABBCC", opacity=0.12),
                name="All nodes", showlegend=False, hoverinfo="skip"),
            go.Scatter3d(
                x=coords[aidx, 0].tolist(), y=coords[aidx, 1].tolist(),
                z=coords[aidx, 2].tolist(),
                mode="markers",
                marker=dict(size=4.5, color=t_anchor,
                            colorscale="Plasma",
                            cmin=tmin, cmax=tmax,
                            showscale=True,
                            colorbar=dict(title="Temp (K)", thickness=14),
                            opacity=0.9, line=dict(width=0.5, color="black")),
                name=f"Anchors M={snap['n_anchors']}",
                hovertemplate=(
                    f"<b>{snap['label']}</b><br>"
                    "x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<br>"
                    "T=%{marker.color:.1f} K<extra></extra>")),
        ]
        if snap["laser_pos"] is not None:
            lp = snap["laser_pos"]
            traces.append(go.Scatter3d(
                x=[lp[0]], y=[lp[1]], z=[lp[2]],
                mode="markers",
                marker=dict(size=10, symbol="diamond",
                            color=COLOR_LASER, opacity=1.0,
                            line=dict(width=1, color="black")),
                name="Laser", showlegend=False))

        frames.append(go.Frame(data=traces, name=snap["label"]))

    # 初始帧
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=f"PhysHGNet3D Anchor Dynamics  N={n_nodes}",
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                zaxis=dict(visible=False), bgcolor="#F5F7FA"),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.05, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 800},
                                      "transition": {"duration": 300},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0},
                                        "mode": "immediate"}]),
                ])],
            sliders=[dict(
                steps=[dict(args=[[f.name],
                                  {"frame": {"duration": 400},
                                   "mode": "immediate"}],
                            label=f.name, method="animate")
                       for f in frames],
                x=0.05, len=0.90, y=0.02,
                currentvalue=dict(prefix="Time step: ", font={"size": 13}),
            )],
            margin=dict(l=0, r=60, t=80, b=60),
            width=960, height=720,
        ))

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  [图] 交互式动画 → {out_path}")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main(args):
    if not PROJECT_IMPORTED:
        sys.exit(1)

    device  = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"╔{'═'*60}╗")
    print(f"║  锚点时序可视化  N={args.n_nodes}  traj={args.traj_idx}  device={device}")
    print(f"╚{'═'*60}╝\n")

    # 加载最优模型
    model, cfg = load_best_model(args.ckpt_dir, args.n_nodes, device)
    window_size = cfg.get("window_size", args.window_size)

    # 找 H5 文件，确定轨迹长度
    h5_path = find_h5_file(args.data_dir, args.n_nodes)
    import h5py
    with h5py.File(str(h5_path), "r") as f:
        traj_key = f"trajectory_{args.traj_idx}"
        if traj_key not in f:
            traj_key = sorted(f.keys())[args.traj_idx]
        T_total = f[traj_key]["node_features"].shape[0]  # 总时间步数

    print(f"\n轨迹 {args.traj_idx}: 共 {T_total} 个时间步\n")

    # 确定要可视化的时间步
    if args.timesteps:
        timesteps = [t for t in args.timesteps if t < T_total]
    else:
        # 默认：均匀选取 6 个时间步（含首尾）
        timesteps = list(np.linspace(0, T_total - 1, 6, dtype=int))
    print(f"可视化时间步: {[t+1 for t in timesteps]}\n")

    # 提取每个时间步的锚点
    print("── 提取各时间步锚点 ──")
    snapshots = extract_anchor_per_timestep(
        model, str(h5_path), args.n_nodes,
        traj_idx=args.traj_idx,
        timesteps=timesteps,
        window_size=window_size,
        device=device)

    if not snapshots:
        print("✗ 无法提取锚点，退出。")
        sys.exit(1)

    print(f"\n── 生成可视化图 ──")

    # 1. 多时间步对比拼图（论文主图）
    plot_temporal_grid(
        snapshots,
        out_path=str(out_dir / f"anchor_temporal_grid_{args.n_nodes}.png"),
        n_nodes=args.n_nodes,
        elev=args.elev, azim=args.azim)

    # 2. 每个时间步的精细四视图
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    all_temp = np.concatenate([s["temp"] for s in snapshots])
    norm = Normalize(vmin=all_temp.min(), vmax=all_temp.max())

    for snap in snapshots:
        coords = snap["coords"]
        aidx   = snap["anchor_idx"]
        t_a    = snap["temp"][aidx]

        fig = plt.figure(figsize=(14, 10), facecolor="white", dpi=150)
        fig.suptitle(
            f"Anchor Distribution — {snap['label']}  "
            f"(N={args.n_nodes}, M={snap['n_anchors']})",
            fontsize=12, fontweight="bold")

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 4, figure=fig, wspace=0.08, left=0.02, right=0.95)
        proj = [("XY top", 0, 1), ("XZ front", 0, 2), ("YZ side", 1, 2)]

        for i, (title, xi, yi) in enumerate(proj):
            ax = fig.add_subplot(gs[0, i])
            ax.scatter(coords[:, xi], coords[:, yi],
                        s=0.6, c=COLOR_NODE, alpha=0.18,
                        linewidths=0, rasterized=True)
            ax.scatter(coords[aidx, xi], coords[aidx, yi],
                        s=28, c=t_a, cmap=CMAP_TEMP, norm=norm,
                        zorder=5, edgecolors="black", linewidths=0.35, alpha=0.9)
            if snap["laser_pos"] is not None:
                lp = snap["laser_pos"]
                ax.scatter(lp[xi], lp[yi], s=150, marker="*",
                            c=COLOR_LASER, zorder=10, edgecolors="black")
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            ax.set_facecolor("#F7F9FB")
            ax.tick_params(labelsize=7)

        ax3 = fig.add_subplot(gs[0, 3], projection="3d")
        ax3.scatter(*[coords[:, i] for i in range(3)],
                    s=0.4, c=COLOR_NODE, alpha=0.10, rasterized=True)
        ax3.scatter(*[coords[aidx, i] for i in range(3)],
                    s=28, c=t_a, cmap=CMAP_TEMP, norm=norm,
                    edgecolors="black", linewidths=0.35, alpha=0.9)
        ax3.view_init(elev=args.elev, azim=args.azim)
        ax3.set_title("3D", fontsize=10)
        ax3.set_facecolor("#F7F9FB")
        ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

        single_out = out_dir / f"anchor_temporal_single_t{snap['t']+1}_{args.n_nodes}.png"
        plt.savefig(str(single_out), bbox_inches="tight", dpi=200)
        plt.close()

    print(f"  [图] 各时间步精细图已保存")

    # 3. 锚点重心轨迹图
    plot_anchor_centroid_trajectory(
        snapshots,
        out_path=str(out_dir / f"anchor_temporal_trajectory_{args.n_nodes}.png"),
        n_nodes=args.n_nodes)

    # 4. 交互式 HTML
    if not args.no_html:
        make_temporal_html(
            snapshots,
            out_path=str(out_dir / f"anchor_temporal_{args.n_nodes}.html"),
            n_nodes=args.n_nodes)

    print(f"\n✓ 所有文件已保存至: {out_dir}")
    print("\n生成文件：")
    for f in sorted(out_dir.glob(f"*temporal*{args.n_nodes}*")):
        print(f"  {f.name:<55} {f.stat().st_size/1024:>7.1f} KB")


def parse_args():
    p = argparse.ArgumentParser(description="PhysHGNet3D 锚点时序可视化（沿轨迹）")
    p.add_argument("--n_nodes",    type=int,   default=4000)
    p.add_argument("--data_dir",   type=str,   default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints/phys_hgnet_3d")
    p.add_argument("--out_dir",    type=str,   default="results_viz/anchors")
    p.add_argument("--traj_idx",   type=int,   default=0,
                   help="使用哪条轨迹（默认第 0 条）")
    p.add_argument("--timesteps",  type=int,   nargs="*", default=None,
                   help="指定时间步（0-based），默认均匀取 6 步")
    p.add_argument("--window_size",type=int,   default=10)
    p.add_argument("--elev",       type=float, default=28.0)
    p.add_argument("--azim",       type=float, default=-55.0)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--no_html",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
