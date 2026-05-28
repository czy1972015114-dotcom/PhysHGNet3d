"""
visualize_anchors_temporal.py
==============================
沿轨迹时序展示 PhysHGNet3D 锚点的动态变化。

核心设计：
  固定 best checkpoint，对同一条轨迹的多个时间步分别推断，
  每步清除图缓存以强制 anchor_selector 重新打分，
  展示锚点如何跟随激光热源移动。

锚点着色方案：
  优先用 source_terms（热源强度）着色，信号远强于温度梯度（89 K）。
  热源峰值节点 = 激光位置，用绿星标注。

生成文件：
  anchor_temporal_grid_<N>.png      多时步对比拼图（论文主图）
  anchor_temporal_single_t<T>_<N>.png  各时步精细四视图
  anchor_temporal_trajectory_<N>.png   锚点重心 vs 激光位置轨迹
  anchor_temporal_<N>.html          交互式动画（需 plotly）

用法：
  python visualize_anchors_temporal.py --n_nodes 4000
  python visualize_anchors_temporal.py --n_nodes 4000 \\
      --timesteps 0 10 20 30 40 50 60 --traj_idx 2
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import h5py
import numpy as np
import torch
import torch.nn as nn

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
    print(f"[错误] 无法导入项目模块：{e}\n请在项目目录下运行。")
    PROJECT_IMPORTED = False

# ─────────────────────────────────────────────────────────────────
# 颜色常量
# ─────────────────────────────────────────────────────────────────
COLOR_NODE  = "#C8D6E5"   # 普通节点（浅蓝灰）
COLOR_LASER = "#00FF88"   # 激光位置（亮绿）

# ─────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────

def to_device(x, device):
    """递归将所有 tensor 搬到 device，支持嵌套 dict。"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x

# ─────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str, n_nodes: int, device: torch.device):
    """加载 best checkpoint，返回 (model, cfg)。"""
    ckpt_path = Path(ckpt_dir) / f"best_{n_nodes}.pth"
    if not ckpt_path.exists():
        candidates = sorted(Path(ckpt_dir).glob(f"*{n_nodes}*.pth"))
        if not candidates:
            raise FileNotFoundError(f"在 {ckpt_dir} 中找不到 N={n_nodes} 的检查点")
        ckpt_path = candidates[-1]
        print(f"  未找到 best checkpoint，使用: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = {**DEFAULT_CONFIG_3D, **ckpt.get("config", {})}
    model = PhysHGNet3D(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"  加载: {ckpt_path.name}")
    print(f"  m_anchors={cfg.get('m_anchors')}  "
          f"best_rne={ckpt.get('best_rne', 'N/A'):.4f}")
    return model, cfg

# ─────────────────────────────────────────────────────────────────
# 核心：按时间步提取锚点
# ─────────────────────────────────────────────────────────────────

def extract_anchors_over_time(
    model: nn.Module,
    h5_path: str,
    traj_idx: int,
    timesteps: List[int],
    window_size: int,
    device: torch.device,
) -> List[Dict]:
    """
    对每个时间步单独推断，提取锚点索引、温度场和热源强度。

    关键细节：
    - 每步推断前清除 _graph_cache，强制 anchor_selector 重新打分
    - 热源从 H5 直接读，避免 window 边界错位
    - Hook 截取 anchor_selector 的输出（long tensor of indices）
    """
    # 以 stride=1 构建只含目标轨迹的 dataset，方便精细索引
    ds = LaserHardening3DDataset(
        h5_path, window_size=window_size, stride=1,
        traj_indices=[traj_idx])
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn_3d, num_workers=0)
    all_samples = list(loader)
    T_avail = len(all_samples)

    # 从第一个 sample 读节点坐标（整条轨迹共享）
    first = to_device(all_samples[0], device)
    coords = first["nodes"].cpu().numpy()    # (N, 3)
    N = len(coords)

    # 读 H5 中完整的热源序列（用于获取真实时刻的激光位置）
    with h5py.File(h5_path, "r") as f:
        traj_key = f"trajectory_{traj_idx}"
        src_all  = f[traj_key]["source_terms"][:]   # (T_total, N, 1)
        nf_all   = f[traj_key]["node_features"][:]  # (T_total, N, 1)
    T_total = src_all.shape[0]

    # 注册 hook 截取 anchor_selector 的输出
    captured = []
    def hook_fn(module, inputs, output):
        if isinstance(output, torch.Tensor) and output.dtype == torch.int64:
            captured.clear()
            captured.append(output.detach().cpu().numpy().flatten())
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor) and o.dtype == torch.int64:
                    captured.clear()
                    captured.append(o.detach().cpu().numpy().flatten())
                    break

    handle = None
    raw = model.module if hasattr(model, "module") else model
    for name, mod in raw.named_modules():
        if "anchor_selector" in name.lower():
            handle = mod.register_forward_hook(hook_fn)
            break
    if handle is None:
        print("  ⚠ 未找到 anchor_selector 模块，将用输出 dict 的 anchor_scores 替代")

    results = []
    for t in timesteps:
        if t >= T_total:
            print(f"  跳过 t={t+1}（超出轨迹长度 {T_total}）")
            continue

        # 清除图缓存，强制每步重新选锚点
        for attr in ("_graph_cache", "_cache_key",
                     "_residual_cache", "_grad_norm_cache",
                     "_source_q_cache"):
            if hasattr(raw, attr):
                setattr(raw, attr, None)

        # 选取 sample：让窗口末尾尽量落在 t（但不超出 dataset 范围）
        sample_idx = max(0, min(t - window_size + 1, T_avail - 1))

        batch = to_device(all_samples[sample_idx], device)
        q_t = torch.tensor(src_all[t, :, 0], dtype=torch.float32, device=device)
        raw._source_q_cache = q_t
        captured.clear()
        with torch.no_grad():
            out = model(batch)

        # 获取锚点索引
        anchor_idx = None
        if captured:
            anchor_idx = captured[0].astype(int)
        elif "anchor_scores" in out and out["anchor_scores"] is not None:
            # 新版 forward 返回可微 scores，取 top-m
            scores = out["anchor_scores"].cpu().numpy()
            m = raw.m_anchors
            anchor_idx = np.argsort(scores)[-m:]
        if anchor_idx is None:
            print(f"  ⚠ t={t+1}: 无法获取锚点，跳过")
            continue
        anchor_idx = anchor_idx[anchor_idx < N]

        # 从 H5 读真实时刻的热源和温度
        t_real     = min(t, T_total - 1)
        q_map      = src_all[t_real, :, 0]    # (N,) 热源强度
        temp_map   = nf_all[t_real, :, 0]     # (N,) 温度（K）

        # 激光位置 = 热源最强节点
        laser_pos = coords[int(np.argmax(q_map))] if q_map.max() > 0 else None

        results.append({
            "t"          : t,
            "label"      : f"t = {t+1}",
            "coords"     : coords,
            "anchor_idx" : anchor_idx,
            "n_anchors"  : len(anchor_idx),
            "temp"       : temp_map,
            "heat"       : q_map,
            "laser_pos"  : laser_pos,
        })
        print(f"  t={t+1:>3}: 锚点={len(anchor_idx):>4}  "
              f"T=[{temp_map.min():.1f}, {temp_map.max():.1f}] K  "
              f"热源峰值={q_map.max():.2f}"
              + (f"  激光≈({laser_pos[0]:.3f},{laser_pos[1]:.3f})"
                 if laser_pos is not None else ""))

    if handle is not None:
        handle.remove()

    return results

# ─────────────────────────────────────────────────────────────────
# 图 1：多时步对比拼图（论文主图）
# ─────────────────────────────────────────────────────────────────

def plot_temporal_grid(snapshots: List[Dict], out_path: str,
                       n_nodes: int, elev=28.0, azim=-55.0):
    """
    上排：XY 俯视图（最直观，激光扫描路径清晰可见）
    下排：3D 透视图
    锚点颜色：热源强度（红→黄表示激光区域，蓝表示低热源）
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    n = len(snapshots)
    if n == 0:
        print("  没有有效快照，跳过对比图")
        return

    fig = plt.figure(figsize=(4.0 * n, 9.0), dpi=150, facecolor="white")
    fig.suptitle(
        f"PhysHGNet3D — Anchor Positions along Trajectory  (N={n_nodes})\n"
        "Red/yellow anchors concentrate near the laser heat source",
        fontsize=12, fontweight="bold", y=0.995)

    gs_top = gridspec.GridSpec(1, n, top=0.92, bottom=0.52,
                                left=0.03, right=0.95, wspace=0.04)
    gs_bot = gridspec.GridSpec(1, n, top=0.47, bottom=0.03,
                                left=0.03, right=0.95, wspace=0.04)

    # 统一热源色域
    all_heat = np.concatenate([s["heat"] for s in snapshots])
    h_max = float(all_heat.max()) if all_heat.max() > 0 else 1.0
    norm_heat = Normalize(vmin=0, vmax=h_max)

    for col, snap in enumerate(snapshots):
        coords    = snap["coords"]
        aidx      = snap["anchor_idx"]
        q_map     = snap["heat"]
        laser_pos = snap["laser_pos"]

        # 锚点处的热源强度（决定颜色）
        q_anchor = q_map[aidx]

        # ── 俯视图 ──────────────────────────────────────────────
        ax2 = fig.add_subplot(gs_top[0, col])
        # 全体节点（淡灰背景）
        ax2.scatter(coords[:, 0], coords[:, 1],
                    s=0.4, c=COLOR_NODE, alpha=0.18,
                    linewidths=0, rasterized=True)
        # 锚点（热源着色）
        sc = ax2.scatter(coords[aidx, 0], coords[aidx, 1],
                         s=25, c=q_anchor, cmap="hot", norm=norm_heat,
                         zorder=5, edgecolors="black", linewidths=0.3,
                         alpha=0.9)
        # 激光位置（绿星）
        if laser_pos is not None:
            ax2.scatter(laser_pos[0], laser_pos[1],
                        s=140, marker="*", c=COLOR_LASER, zorder=10,
                        edgecolors="black", linewidths=0.5)

        ax2.set_title(f"{snap['label']}\nM={snap['n_anchors']}",
                      fontsize=10, pad=3)
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_facecolor("#F7F9FB")

        # ── 3D 透视图 ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs_bot[0, col], projection="3d")
        ax3.scatter(*[coords[:, i] for i in range(3)],
                    s=0.3, c=COLOR_NODE, alpha=0.09,
                    linewidths=0, rasterized=True)
        ax3.scatter(*[coords[aidx, i] for i in range(3)],
                    s=28, c=q_anchor, cmap="hot", norm=norm_heat,
                    edgecolors="black", linewidths=0.3, alpha=0.9)
        if laser_pos is not None:
            ax3.scatter(*laser_pos, s=180, marker="*",
                        c=COLOR_LASER, zorder=10,
                        edgecolors="black", linewidths=0.5)
        ax3.view_init(elev=elev, azim=azim)
        ax3.set_facecolor("#F7F9FB")
        for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#DDDDDD")
        ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

    # 热源强度色条
    sm = plt.cm.ScalarMappable(cmap="hot", norm=norm_heat)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.35, pad=0.005,
                        fraction=0.008, orientation="vertical")
    cbar.set_label("Heat source intensity\n(W/m³ normalized)", fontsize=8)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#FF4444", markersize=8,
               label="Anchor (high heat)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#2233AA", markersize=8,
               label="Anchor (low heat)"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=COLOR_LASER, markersize=11,
               label="Laser position"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, framealpha=0.85,
               bbox_to_anchor=(0.5, 0.001))

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  [图] 多时步对比图 → {out_path}")

# ─────────────────────────────────────────────────────────────────
# 图 2：锚点重心 vs 激光位置轨迹
# ─────────────────────────────────────────────────────────────────

def plot_centroid_trajectory(snapshots: List[Dict],
                             out_path: str, n_nodes: int):
    """
    定量展示锚点重心坐标随时间的变化，
    以及与激光位置的跟随关系。
    """
    import matplotlib.pyplot as plt

    ts   = [s["t"] + 1 for s in snapshots]
    cen  = np.array([s["coords"][s["anchor_idx"]].mean(0)
                     for s in snapshots])
    lpos = [s["laser_pos"] for s in snapshots]
    has_laser = any(p is not None for p in lpos)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True,
                              facecolor="white", dpi=150)
    fig.suptitle(
        f"Anchor Centroid vs Laser Position  (N={n_nodes})\n"
        "Closer tracking = better physics awareness",
        fontsize=11, fontweight="bold")

    axes_labels = ["X (m)", "Y (m)", "Z (m)"]
    axes_colors = ["#E74C3C", "#27AE60", "#2980B9"]

    for i, (ax, lab, col) in enumerate(zip(axes, axes_labels, axes_colors)):
        ax.plot(ts, cen[:, i], "o-", color=col, lw=2, ms=5,
                label=f"Anchor centroid {lab.split()[0]}")
        if has_laser:
            lv = [p[i] if p is not None else np.nan for p in lpos]
            ax.plot(ts, lv, "s--", color="gray", lw=1.5, ms=4,
                    alpha=0.7, label=f"Laser {lab.split()[0]}")
        ax.set_ylabel(lab, fontsize=9)
        ax.legend(fontsize=8, loc="best", framealpha=0.7)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#FAFAFA")

    axes[-1].set_xlabel("Time step", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  [图] 重心轨迹图 → {out_path}")

# ─────────────────────────────────────────────────────────────────
# 图 3：各时步精细四视图
# ─────────────────────────────────────────────────────────────────

def plot_single_timestep(snap: Dict, out_path: str,
                         n_nodes: int, norm_heat,
                         elev=28.0, azim=-55.0):
    """XY / XZ / YZ 三视图 + 3D 透视，单时步精细展示。"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    coords = snap["coords"]
    aidx   = snap["anchor_idx"]
    q_a    = snap["heat"][aidx]
    lp     = snap["laser_pos"]

    fig = plt.figure(figsize=(14, 9), facecolor="white", dpi=150)
    fig.suptitle(
        f"Anchor Distribution — {snap['label']}  "
        f"(N={n_nodes}, M={snap['n_anchors']})\n"
        "Color = heat source intensity at anchor node",
        fontsize=11, fontweight="bold")

    gs = GridSpec(1, 4, figure=fig, wspace=0.06, left=0.02, right=0.95)
    proj = [("XY (top)", 0, 1), ("XZ (front)", 0, 2), ("YZ (side)", 1, 2)]

    for i, (title, xi, yi) in enumerate(proj):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(coords[:, xi], coords[:, yi],
                   s=0.5, c=COLOR_NODE, alpha=0.18,
                   linewidths=0, rasterized=True)
        ax.scatter(coords[aidx, xi], coords[aidx, yi],
                   s=30, c=q_a, cmap="hot", norm=norm_heat,
                   zorder=5, edgecolors="black", linewidths=0.3,
                   alpha=0.9)
        if lp is not None:
            ax.scatter(lp[xi], lp[yi], s=180, marker="*",
                       c=COLOR_LASER, zorder=10, edgecolors="black")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_facecolor("#F7F9FB")
        ax.tick_params(labelsize=7)

    ax3 = fig.add_subplot(gs[0, 3], projection="3d")
    ax3.scatter(*[coords[:, j] for j in range(3)],
                s=0.4, c=COLOR_NODE, alpha=0.09, rasterized=True)
    ax3.scatter(*[coords[aidx, j] for j in range(3)],
                s=30, c=q_a, cmap="hot", norm=norm_heat,
                edgecolors="black", linewidths=0.3, alpha=0.9)
    if lp is not None:
        ax3.scatter(*lp, s=180, marker="*", c=COLOR_LASER,
                    zorder=10, edgecolors="black", linewidths=0.5)
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_title("3D perspective", fontsize=10)
    ax3.set_facecolor("#F7F9FB")
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

    # 统计注释
    fig.text(0.50, 0.01,
             f"N_total={len(coords):,}  |  M_anchors={snap['n_anchors']:,}  "
             f"({snap['n_anchors']/len(coords)*100:.1f}%)  |  {snap['label']}",
             ha="center", fontsize=8, color="#555555")

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()

# ─────────────────────────────────────────────────────────────────
# 图 4：交互式 HTML 动画
# ─────────────────────────────────────────────────────────────────

def make_html_animation(snapshots: List[Dict], out_path: str, n_nodes: int):
    """带时间步滑块、可旋转 3D 的交互式 HTML。"""
    if not HAS_PLOTLY:
        print("  [跳过] plotly 未安装 (pip install plotly)")
        return

    all_heat = np.concatenate([s["heat"] for s in snapshots])
    q_min, q_max = 0.0, float(all_heat.max())

    frames = []
    for snap in snapshots:
        coords = snap["coords"]
        aidx   = snap["anchor_idx"]
        q_a    = snap["heat"][aidx].tolist()

        data = [
            go.Scatter3d(
                x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
                z=coords[:, 2].tolist(), mode="markers",
                marker=dict(size=1.0, color="#AABBCC", opacity=0.12),
                showlegend=False, hoverinfo="skip"),
            go.Scatter3d(
                x=coords[aidx, 0].tolist(),
                y=coords[aidx, 1].tolist(),
                z=coords[aidx, 2].tolist(), mode="markers",
                marker=dict(size=5, color=q_a, colorscale="hot",
                            cmin=q_min, cmax=q_max, showscale=True,
                            colorbar=dict(title="Heat src.", thickness=14),
                            opacity=0.92,
                            line=dict(width=0.5, color="black")),
                name=f"Anchors M={snap['n_anchors']}",
                hovertemplate=(
                    f"<b>{snap['label']}</b><br>"
                    "x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<br>"
                    "heat=%{marker.color:.2f}<extra></extra>")),
        ]
        if snap["laser_pos"] is not None:
            lp = snap["laser_pos"]
            data.append(go.Scatter3d(
                x=[lp[0]], y=[lp[1]], z=[lp[2]], mode="markers",
                marker=dict(size=12, symbol="diamond",
                            color=COLOR_LASER, opacity=1.0,
                            line=dict(width=1, color="black")),
                name="Laser", showlegend=False))

        frames.append(go.Frame(data=data, name=snap["label"]))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=f"PhysHGNet3D Anchor Dynamics  N={n_nodes}<br>"
                     "<sup>Anchors colored by heat source intensity — "
                     "red/yellow = near laser</sup>",
                font=dict(size=13)),
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                zaxis=dict(visible=False), bgcolor="#F5F7FA"),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.08, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play",  method="animate",
                         args=[None, {"frame": {"duration": 700},
                                      "transition": {"duration": 250},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
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
                currentvalue=dict(prefix="Time step: ",
                                  font={"size": 13}),
            )],
            margin=dict(l=0, r=60, t=90, b=60),
            width=960, height=720,
        ))

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  [图] 交互式动画 → {out_path}")

# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main(args):
    if not PROJECT_IMPORTED:
        sys.exit(1)

    device  = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n╔{'═'*62}╗")
    print(f"║  PhysHGNet3D 锚点时序可视化")
    print(f"║  N={args.n_nodes}  traj={args.traj_idx}  device={device}")
    print(f"╚{'═'*62}╝\n")

    # 加载模型
    model, cfg = load_model(args.ckpt_dir, args.n_nodes, device)
    window_size = args.window_size

    # 读轨迹长度
    h5_path = str(find_h5_file(args.data_dir, args.n_nodes))
    with h5py.File(h5_path, "r") as f:
        traj_key = f"trajectory_{args.traj_idx}"
        T_total  = f[traj_key]["node_features"].shape[0]

    print(f"轨迹 {args.traj_idx}：共 {T_total} 时步\n")

    # 确定时间步
    if args.timesteps:
        timesteps = [t for t in args.timesteps if t < T_total]
    else:
        n_frames  = min(args.n_frames, T_total)
        timesteps = list(np.linspace(0, T_total - 1, n_frames, dtype=int))
    print(f"可视化时步: {[t+1 for t in timesteps]}\n")

    # ── 提取锚点 ─────────────────────────────────────────────────
    print("── 提取各时步锚点 ──")
    snapshots = extract_anchors_over_time(
        model, h5_path, args.traj_idx,
        timesteps, window_size, device)

    if not snapshots:
        print("\n✗ 无有效快照，退出。")
        sys.exit(1)

    # 统一热源色域（供单图使用）
    from matplotlib.colors import Normalize
    all_heat  = np.concatenate([s["heat"] for s in snapshots])
    norm_heat = Normalize(vmin=0, vmax=float(all_heat.max()))

    print(f"\n── 生成可视化图（共 {len(snapshots)} 个时步）──")

    # 图 1：多时步对比拼图
    plot_temporal_grid(
        snapshots,
        out_path=str(out_dir / f"anchor_temporal_grid_{args.n_nodes}.png"),
        n_nodes=args.n_nodes,
        elev=args.elev, azim=args.azim)

    # 图 2：重心轨迹
    plot_centroid_trajectory(
        snapshots,
        out_path=str(out_dir / f"anchor_temporal_centroid_{args.n_nodes}.png"),
        n_nodes=args.n_nodes)

    # 图 3：各时步精细四视图
    for snap in snapshots:
        fo = out_dir / f"anchor_temporal_t{snap['t']+1:03d}_{args.n_nodes}.png"
        plot_single_timestep(snap, str(fo), args.n_nodes, norm_heat,
                             elev=args.elev, azim=args.azim)
    print(f"  [图] 各时步精细图已保存（{len(snapshots)} 张）")

    # 图 4：HTML 动画
    if not args.no_html:
        make_html_animation(
            snapshots,
            out_path=str(out_dir / f"anchor_temporal_{args.n_nodes}.html"),
            n_nodes=args.n_nodes)

    print(f"\n✓ 所有文件已保存至: {out_dir}")
    print("\n生成文件列表：")
    for fp in sorted(out_dir.iterdir()):
        if f"{args.n_nodes}" in fp.name:
            print(f"  {fp.name:<58} {fp.stat().st_size/1024:>7.1f} KB")

# ─────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PhysHGNet3D 锚点时序可视化")
    p.add_argument("--n_nodes",    type=int,   default=4000)
    p.add_argument("--data_dir",   type=str,   default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints/phys_hgnet_3d")
    p.add_argument("--out_dir",    type=str,   default="results_viz/anchors")
    p.add_argument("--traj_idx",   type=int,   default=0,
                   help="轨迹编号（0-based），不同轨迹激光路径不同")
    p.add_argument("--timesteps",  type=int,   nargs="*", default=None,
                   help="手动指定时步（0-based），如 --timesteps 0 10 20 30 40 50 60")
    p.add_argument("--n_frames",   type=int,   default=6,
                   help="自动均匀采样的帧数（timesteps 未指定时生效）")
    p.add_argument("--window_size",type=int,   default=10)
    p.add_argument("--elev",       type=float, default=28.0)
    p.add_argument("--azim",       type=float, default=-55.0)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--no_html",    action="store_true",
                   help="跳过 plotly HTML 生成")
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
