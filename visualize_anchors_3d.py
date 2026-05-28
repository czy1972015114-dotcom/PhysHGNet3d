"""
visualize_anchors_3d.py  (v2 — 白色背景 + 时步对齐修复)
=========================================================
改动说明：
  1. 背景改为白色
  2. 修复 _source_q_cache 时步错位：推断前手动注入 src_all[t]，
     使锚点选择与可视化时步严格对齐，消除 t=25/33 等帧锚点"跑偏"现象

用法：
  python visualize_anchors_3d.py --n_nodes 4000
  python visualize_anchors_3d.py --n_nodes 4000 \\
      --timesteps 0 10 20 30 40 50 60 --traj_idx 0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

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

# ─────────────────────────────────────────────────────────────────
# 自定义 colormap（黑→深红→橙红→橙→亮黄）
# ─────────────────────────────────────────────────────────────────
_HEAT_COLORS = [
    (0.04, 0.04, 0.08),
    (0.40, 0.02, 0.05),
    (0.85, 0.18, 0.05),
    (0.98, 0.55, 0.10),
    (0.99, 0.95, 0.30),
]
CMAP_HEAT = LinearSegmentedColormap.from_list("heat_dark", _HEAT_COLORS, N=512)

# 白色背景下配色
BG_COLOR   = "white"
PANE_COLOR = (0.94, 0.95, 0.97, 1.0)
PANE_EDGE  = (0.75, 0.78, 0.83, 0.8)
GRID_COLOR = "#CACED6"
NODE_COLOR = "#B0BAC8"
TITLE_COLOR= "#1A2030"
SUB_COLOR  = "#5A6478"
CBAR_LABEL = "#1A2030"

# ─────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def load_model(ckpt_dir: str, n_nodes: int, device: torch.device):
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
    print(f"  加载: {ckpt_path.name}  |  best_rne={ckpt.get('best_rne', 'N/A')}")
    return model, cfg

# ─────────────────────────────────────────────────────────────────
# 按时间步提取锚点（修复时步对齐）
# ─────────────────────────────────────────────────────────────────

def extract_anchors(model, h5_path, traj_idx, timesteps, window_size, device):
    ds = LaserHardening3DDataset(
        h5_path, window_size=window_size, stride=1,
        traj_indices=[traj_idx])
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn_3d, num_workers=0)
    all_samples = list(loader)
    T_avail = len(all_samples)

    first  = to_device(all_samples[0], device)
    coords = first["nodes"].cpu().numpy()   # (N, 3)
    N = len(coords)

    with h5py.File(h5_path, "r") as f:
        traj_key = f"trajectory_{traj_idx}"
        src_all  = f[traj_key]["source_terms"][:]   # (T, N, 1)
        nf_all   = f[traj_key]["node_features"][:]  # (T, N, 1)
    T_total = src_all.shape[0]

    # hook 截取 anchor_selector 输出的锚点索引
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

    raw = model.module if hasattr(model, "module") else model
    handle = None
    for _, mod in raw.named_modules():
        if type(mod).__name__ == "PhysicsAwareAnchorSelector":
            handle = mod.register_forward_hook(hook_fn)
            break
    if handle is None:
        # 兜底：匹配名称含 anchor_selector 的第一个子模块
        for name, mod in raw.named_modules():
            if "anchor_selector" in name.lower() and name.count(".") == 1:
                handle = mod.register_forward_hook(hook_fn)
                break

    results = []
    for t in timesteps:
        if t >= T_total:
            continue

        # ── 清除所有缓存 ──────────────────────────────────────────
        for attr in ("_graph_cache", "_cache_key",
                     "_residual_cache", "_grad_norm_cache",
                     "_source_q_cache", "_step_counter"):
            if hasattr(raw, attr):
                setattr(raw, attr, None if attr != "_step_counter" else 0)

        # ── 关键修复：直接注入 t 时刻的热源 ──────────────────────
        # 原来的逻辑：model 用 batch 窗口末尾（t+window_size-1）的热源
        # 来选锚点，但可视化时用 t 时刻的激光位置 → 两者错位最多 9 步。
        # 修复：在 forward 前手动将 _source_q_cache 设为 src_all[t]，
        # 强制 anchor_selector 以 t 时刻的热源为信号选锚点。
        q_t = torch.tensor(
            src_all[t, :, 0], dtype=torch.float32, device=device)
        raw._source_q_cache = q_t.detach()

        sample_idx = max(0, min(t, T_avail - 1))
        batch = to_device(all_samples[sample_idx], device)

        captured.clear()
        with torch.no_grad():
            out = model(batch)

        # 获取锚点索引
        anchor_idx = None
        if captured:
            anchor_idx = captured[0].astype(int)
        elif "anchor_scores" in out and out["anchor_scores"] is not None:
            scores    = out["anchor_scores"].cpu().numpy()
            m         = raw.m_anchors
            anchor_idx = np.argsort(scores)[-m:]
        if anchor_idx is None:
            print(f"  ⚠ t={t+1}: 无法获取锚点，跳过")
            continue
        anchor_idx = anchor_idx[anchor_idx < N]

        # t 时刻的热源和温度
        t_real    = min(t, T_total - 1)
        q_map     = src_all[t_real, :, 0]
        temp_map  = nf_all[t_real, :, 0]
        laser_pos = coords[int(np.argmax(q_map))] if q_map.max() > 0 else None

        results.append({
            "t": t, "label": f"t = {t+1}",
            "coords": coords,
            "anchor_idx": anchor_idx,
            "n_anchors": len(anchor_idx),
            "heat": q_map,
            "temp": temp_map,
            "laser_pos": laser_pos,
        })
        print(f"  t={t+1:>3}: 锚点={len(anchor_idx):>4}  "
              f"T=[{temp_map.min():.0f}, {temp_map.max():.0f}] K  "
              f"热源峰={q_map.max():.2f}"
              + (f"  激光≈({laser_pos[0]:.3f},{laser_pos[1]:.3f})"
                 if laser_pos is not None else ""))

    if handle is not None:
        handle.remove()
    return results

# ─────────────────────────────────────────────────────────────────
# 单格 3D 渲染（白色背景版）
# ─────────────────────────────────────────────────────────────────

def _render_3d_panel(ax, snap, norm_heat, elev, azim):
    coords    = snap["coords"]
    aidx      = snap["anchor_idx"]
    q_anchor  = snap["heat"][aidx]
    laser_pos = snap["laser_pos"]

    # ── 背景节点：浅蓝灰、极低 alpha ────────────────────────────
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               s=0.5, c=NODE_COLOR, alpha=0.10,
               linewidths=0, rasterized=True, depthshade=False)

    # ── 锚点：大小 ∝ 热源强度，颜色 = 热源强度 ──────────────────
    q_norm = norm_heat(q_anchor)
    sizes  = 14 + 65 * q_norm
    colors = CMAP_HEAT(q_norm)

    ax.scatter(coords[aidx, 0], coords[aidx, 1], coords[aidx, 2],
               s=sizes, c=colors, alpha=0.88,
               linewidths=0.4, edgecolors="#555566",
               rasterized=False, depthshade=True)

    # ── 激光位置：光晕 + 实心星 ──────────────────────────────────
    if laser_pos is not None:
        ax.scatter(*laser_pos, s=700, marker="*",
                   c="#22CC66", alpha=0.15,
                   linewidths=0, depthshade=False)
        ax.scatter(*laser_pos, s=240, marker="*",
                   c="#00BB55", alpha=1.0,
                   linewidths=0.8, edgecolors="#005533",
                   depthshade=False, zorder=10)

    # ── 坐标轴美化（白色背景） ────────────────────────────────────
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(BG_COLOR)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])
        axis.set_tick_params(size=0)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor(PANE_COLOR)
        pane.set_edgecolor(PANE_EDGE)

    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    ax.set_title(snap["label"],
                 fontsize=10, color=TITLE_COLOR,
                 pad=4, fontweight="bold",
                 fontfamily="monospace")

# ─────────────────────────────────────────────────────────────────
# 主图：多时步 3D 对比拼图
# ─────────────────────────────────────────────────────────────────

def plot_3d_grid(snapshots: List[Dict], out_path: str,
                 n_nodes: int, n_cols: int = 3,
                 elev: float = 28.0, azim: float = -55.0):
    n      = len(snapshots)
    n_rows = (n + n_cols - 1) // n_cols

    all_heat  = np.concatenate([s["heat"] for s in snapshots])
    q_max     = float(all_heat.max()) if all_heat.max() > 0 else 1.0
    norm_heat = Normalize(vmin=0, vmax=q_max)

    fig = plt.figure(
        figsize=(5.2 * n_cols, 5.0 * n_rows + 1.5),
        facecolor=BG_COLOR, dpi=180)

    fig.text(0.50, 0.990,
             "PhysHGNet3D  ·  Anchor Distribution over Time",
             ha="center", va="top",
             fontsize=15, color=TITLE_COLOR, fontweight="bold",
             fontfamily="monospace")
    fig.text(0.50, 0.971,
             f"N = {n_nodes:,}  |  M = {snapshots[0]['n_anchors']}  |  "
             "Color / size ∝ heat source intensity",
             ha="center", va="top",
             fontsize=9, color=SUB_COLOR)

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        top=0.940, bottom=0.09,
        left=0.03, right=0.90,
        wspace=0.05, hspace=0.10)

    for i, snap in enumerate(snapshots):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col], projection="3d")
        _render_3d_panel(ax, snap, norm_heat, elev, azim)

    for i in range(len(snapshots), n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    # 颜色条
    sm = ScalarMappable(cmap=CMAP_HEAT, norm=norm_heat)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.915, 0.12, 0.018, 0.79])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Heat source intensity",
                   color=CBAR_LABEL, fontsize=9, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color=CBAR_LABEL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=CBAR_LABEL)
    cbar.outline.set_edgecolor("#AABBCC")

    # 图例
    legend_elements = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=CMAP_HEAT(0.95), markersize=9,
               label="Anchor  (high heat)"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=CMAP_HEAT(0.08), markersize=6,
               markeredgecolor="#666677", markeredgewidth=0.5,
               label="Anchor  (low heat)"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor="#00BB55", markersize=11,
               label="Laser position"),
    ]
    fig.legend(handles=legend_elements,
               loc="lower center", ncol=3,
               fontsize=9, framealpha=0.85,
               facecolor="white", edgecolor="#CACED6",
               labelcolor=TITLE_COLOR,
               bbox_to_anchor=(0.46, 0.005))

    plt.savefig(out_path, bbox_inches="tight",
                dpi=200, facecolor=BG_COLOR)
    plt.close()
    print(f"  [图] 3D 拼图 → {out_path}")

# ─────────────────────────────────────────────────────────────────
# 单帧精细大图（主视角 + 俯视）
# ─────────────────────────────────────────────────────────────────

def plot_3d_single(snap: Dict, out_path: str,
                   n_nodes: int, norm_heat,
                   elev: float = 28.0, azim: float = -55.0):
    fig = plt.figure(figsize=(13, 6), facecolor=BG_COLOR, dpi=180)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _render_3d_panel(ax1, snap, norm_heat, elev, azim)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _render_3d_panel(ax2, snap, norm_heat, elev=88, azim=azim)
    ax2.set_title("Top view", fontsize=10, color=TITLE_COLOR,
                  pad=4, fontweight="bold", fontfamily="monospace")

    fig.suptitle(
        f"Anchor Distribution — {snap['label']}  "
        f"(N={n_nodes}, M={snap['n_anchors']})",
        fontsize=13, color=TITLE_COLOR, fontweight="bold",
        fontfamily="monospace", y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, bbox_inches="tight",
                dpi=200, facecolor=BG_COLOR)
    plt.close()

# ─────────────────────────────────────────────────────────────────
# 交互式 HTML（plotly）
# ─────────────────────────────────────────────────────────────────

def make_html_animation(snapshots: List[Dict], out_path: str, n_nodes: int):
    if not HAS_PLOTLY:
        print("  [跳过] plotly 未安装 (pip install plotly)")
        return

    all_heat = np.concatenate([s["heat"] for s in snapshots])
    q_min, q_max = 0.0, float(all_heat.max())

    x_vals = np.linspace(0, 1, 10)
    plotly_cmap = []
    for x in x_vals:
        r, g, b, _ = CMAP_HEAT(x)
        plotly_cmap.append([float(x),
                            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])

    frames = []
    for snap in snapshots:
        coords = snap["coords"]
        aidx   = snap["anchor_idx"]
        q_a    = snap["heat"][aidx].tolist()
        q_norm = np.array(q_a) / max(q_max, 1e-8)
        sizes  = (8 + 28 * q_norm).tolist()

        data = [
            go.Scatter3d(
                x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
                z=coords[:, 2].tolist(), mode="markers",
                marker=dict(size=1.0, color="#99AABB", opacity=0.10),
                showlegend=False, hoverinfo="skip"),
            go.Scatter3d(
                x=coords[aidx, 0].tolist(),
                y=coords[aidx, 1].tolist(),
                z=coords[aidx, 2].tolist(),
                mode="markers",
                marker=dict(
                    size=sizes, color=q_a,
                    colorscale=plotly_cmap,
                    cmin=q_min, cmax=q_max,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Heat src.",
                                   font=dict(color="#1A2030")),
                        tickfont=dict(color="#1A2030"),
                        thickness=14),
                    opacity=0.90,
                    line=dict(width=0.4, color="rgba(80,80,100,0.5)")),
                name=f"M={snap['n_anchors']}",
                hovertemplate=(
                    f"<b>{snap['label']}</b><br>"
                    "(%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
                    "heat=%{marker.color:.3f}<extra></extra>")),
        ]
        if snap["laser_pos"] is not None:
            lp = snap["laser_pos"]
            data.append(go.Scatter3d(
                x=[lp[0]], y=[lp[1]], z=[lp[2]],
                mode="markers",
                marker=dict(size=14, symbol="diamond",
                            color="#00BB55", opacity=1.0,
                            line=dict(width=1.5, color="#005533")),
                name="Laser", showlegend=False))

        frames.append(go.Frame(data=data, name=snap["label"]))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=(f"PhysHGNet3D  ·  Anchor Dynamics  N={n_nodes}<br>"
                      "<sup>Anchor size & color ∝ heat source intensity</sup>"),
                font=dict(size=14, color="#1A2030", family="monospace"),
                x=0.5, xanchor="center"),
            paper_bgcolor="white",
            scene=dict(
                bgcolor="white",
                xaxis=dict(showticklabels=False, gridcolor="#CACED6",
                           showbackground=True, backgroundcolor="#F0F2F5"),
                yaxis=dict(showticklabels=False, gridcolor="#CACED6",
                           showbackground=True, backgroundcolor="#F0F2F5"),
                zaxis=dict(showticklabels=False, gridcolor="#CACED6",
                           showbackground=True, backgroundcolor="#F0F2F5"),
                camera=dict(eye=dict(x=1.4, y=1.2, z=0.8)),
            ),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.08, x=0.5, xanchor="center",
                font=dict(color="#1A2030"),
                bgcolor="white",
                bordercolor="#CACED6",
                buttons=[
                    dict(label="▶ Play", method="animate",
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
                font=dict(color="#1A2030"),
                bgcolor="white",
                activebgcolor="#DDE4F0",
                bordercolor="#AABBCC",
                currentvalue=dict(
                    prefix="Time step: ",
                    font={"size": 13, "color": "#1A2030"}),
            )],
            margin=dict(l=0, r=80, t=100, b=70),
            width=1000, height=760,
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
    print(f"║  PhysHGNet3D 锚点时序可视化（白色背景版）")
    print(f"║  N={args.n_nodes}  traj={args.traj_idx}  device={device}")
    print(f"╚{'═'*62}╝\n")

    model, cfg = load_model(args.ckpt_dir, args.n_nodes, device)
    window_size = args.window_size

    h5_path = str(find_h5_file(args.data_dir, args.n_nodes))
    with h5py.File(h5_path, "r") as f:
        traj_key = f"trajectory_{args.traj_idx}"
        T_total  = f[traj_key]["node_features"].shape[0]

    print(f"轨迹 {args.traj_idx}：共 {T_total} 时步\n")

    if args.timesteps:
        timesteps = [t for t in args.timesteps if t < T_total]
    else:
        n_frames  = min(args.n_frames, T_total)
        timesteps = list(np.linspace(0, T_total - 1, n_frames, dtype=int))
    print(f"可视化时步: {[t+1 for t in timesteps]}\n")

    print("── 提取各时步锚点 ──")
    snapshots = extract_anchors(
        model, h5_path, args.traj_idx,
        timesteps, window_size, device)

    if not snapshots:
        print("\n✗ 无有效快照，退出。")
        sys.exit(1)

    all_heat  = np.concatenate([s["heat"] for s in snapshots])
    norm_heat = Normalize(vmin=0, vmax=float(all_heat.max()))

    print(f"\n── 生成可视化图 ──")

    n_cols = min(args.n_cols, len(snapshots))
    plot_3d_grid(
        snapshots,
        out_path=str(out_dir / f"anchor_3d_grid_{args.n_nodes}.png"),
        n_nodes=args.n_nodes,
        n_cols=n_cols,
        elev=args.elev, azim=args.azim)

    for snap in snapshots:
        fo = out_dir / f"anchor_3d_t{snap['t']+1:03d}_{args.n_nodes}.png"
        plot_3d_single(snap, str(fo), args.n_nodes,
                       norm_heat, elev=args.elev, azim=args.azim)
    print(f"  [图] 各时步精细图已保存（{len(snapshots)} 张）")

    if not args.no_html:
        make_html_animation(
            snapshots,
            out_path=str(out_dir / f"anchor_3d_{args.n_nodes}.html"),
            n_nodes=args.n_nodes)

    print(f"\n✓ 全部保存至: {out_dir}")
    print("\n文件列表：")
    for fp in sorted(out_dir.iterdir()):
        if str(args.n_nodes) in fp.name:
            print(f"  {fp.name:<56} {fp.stat().st_size/1024:>8.1f} KB")


def parse_args():
    p = argparse.ArgumentParser(description="PhysHGNet3D 锚点时序可视化（纯 3D，白色背景）")
    p.add_argument("--n_nodes",    type=int,   default=4000)
    p.add_argument("--data_dir",   type=str,   default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints/phys_hgnet_3d")
    p.add_argument("--out_dir",    type=str,   default="results_viz/anchors")
    p.add_argument("--traj_idx",   type=int,   default=0)
    p.add_argument("--timesteps",  type=int,   nargs="*", default=None,
                   help="手动指定时步（0-based），如 --timesteps 0 10 20 30 40 50 60")
    p.add_argument("--n_frames",   type=int,   default=6,
                   help="自动均匀采样帧数（timesteps 未指定时生效）")
    p.add_argument("--n_cols",     type=int,   default=3,
                   help="拼图每行列数（默认 3）")
    p.add_argument("--window_size",type=int,   default=10)
    p.add_argument("--elev",       type=float, default=28.0)
    p.add_argument("--azim",       type=float, default=-55.0)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--no_html",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
