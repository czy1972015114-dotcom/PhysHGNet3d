"""
visualize_anchors_3d.py — PhysHGNet3D 锚点时序可视化
=====================================================

功能
----
1. 读取 checkpoints/<ckpt_dir>/best_{N}.pth 中保存的最佳模型
2. 随机生成一条含移动热源的 3D 激光淬火轨迹
3. 逐时间步做推理，记录每步选出的锚点坐标
4. 输出：
   - anchor_step_000.png … anchor_step_T.png   每步快照
   - anchor_summary.png                         所有步叠加对比图
   - anchor_evolution.gif                        动态演化 GIF

运行指令（见文末）：
  python visualize_anchors_3d.py --n_nodes 1000 --ckpt_dir checkpoints/phys_hgnet_3d
"""

from __future__ import annotations

import argparse
import math
import sys
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

import torch

# ── 可选依赖 ─────────────────────────────────────────────────────
try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

# ── 项目模块 ──────────────────────────────────────────────────────
try:
    from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D
except ImportError as e:
    sys.exit(f"[错误] 无法导入 PhysHGNet3D，请在项目根目录运行：{e}")


# ═══════════════════════════════════════════════════════════════════
# Part 1 — 合成 3D 网格与轨迹生成
# ═══════════════════════════════════════════════════════════════════

def _make_3d_mesh(N: int, seed: int = 0):
    """
    在 [0,1]×[0,0.4]×[0,0.06] 长方体内生成 N 个节点的四面体网格。
    返回 (nodes_np, tets_np, edges_np)
    """
    if not HAS_SCIPY:
        sys.exit("[错误] 网格生成需要 scipy：pip install scipy")

    rng = np.random.default_rng(seed)
    pts = rng.uniform([0, 0, 0], [1.0, 0.4, 0.06], size=(N, 3)).astype(np.float32)

    tri     = Delaunay(pts)
    tets_np = tri.simplices.astype(np.int64)

    # 提取无向边
    edge_set: set = set()
    for tet in tets_np:
        for i in range(4):
            for j in range(i + 1, 4):
                edge_set.add((min(tet[i], tet[j]), max(tet[i], tet[j])))
    edges_np = np.array(sorted(edge_set), dtype=np.int64)

    return pts, tets_np, edges_np


def _make_laser_trajectory(pts: np.ndarray, T: int, B: int = 1,
                            seed: int = 42) -> tuple:
    """
    生成一条含移动激光热源的合成轨迹。

    热源：从 x=0.1 匀速移动到 x=0.9，沿 y=0.2 中心线，
          高斯分布 σ=0.08，强度 5000 W/m³。
    初始温度：298.15 K（室温）+ 左侧预加热。
    """
    N = pts.shape[0]
    # ── 初始温度 ─────────────────────────────────────────────────
    T_init = np.full((B, N, 1), 298.15, dtype=np.float32)
    # 左侧 x<0.2 略微预加热
    hot_mask = pts[:, 0] < 0.2
    T_init[:, hot_mask, :] += 100.0

    # ── 热源逐步移动 ─────────────────────────────────────────────
    src = np.zeros((B, T, N, 1), dtype=np.float32)
    laser_y = 0.2
    laser_z = 0.03  # 表面中部
    sigma   = 0.08
    power   = 5000.0

    for t in range(T):
        frac   = t / max(T - 1, 1)
        lx     = 0.1 + frac * 0.8          # 从 x=0.1 移动到 x=0.9
        dx     = pts[:, 0] - lx
        dy     = pts[:, 1] - laser_y
        dz     = pts[:, 2] - laser_z
        dist2  = dx**2 + dy**2 + dz**2
        gauss  = power * np.exp(-dist2 / (2 * sigma**2))
        src[:, t, :, 0] = gauss

    time_pts = np.linspace(0.0, 2.0, T, dtype=np.float32)
    return T_init, src, time_pts


# ═══════════════════════════════════════════════════════════════════
# Part 2 — 模型加载
# ═══════════════════════════════════════════════════════════════════

def load_model(ckpt_dir: str, N: int,
               device: torch.device,
               res_upd_freq: int = 1) -> PhysHGNet3D:
    """
    从 <ckpt_dir>/best_{N}.pth 加载已训练的 PhysHGNet3D。

    res_upd_freq 强制设为 1，确保每步都重选锚点（可视化需要）。
    """
    ckpt_path = Path(ckpt_dir) / f"best_{N}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到检查点：{ckpt_path}\n"
            f"请先运行训练脚本生成 best_{N}.pth，"
            f"或使用 --no_pretrain 跳过加载（随机初始化权重）。"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("config", {**DEFAULT_CONFIG_3D})

    # 强制每步更新锚点，让可视化更有动态性
    cfg["residual_update_freq"] = res_upd_freq
    cfg["use_physics_anchor"]   = True

    model = PhysHGNet3D(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch    = ckpt.get("epoch",    "?")
    best_rne = ckpt.get("best_rne", float("nan"))
    print(f"  ✓ 已加载 {ckpt_path.name}  "
          f"(epoch={epoch}, best_val_RNE={best_rne:.4f})")
    return model


def build_random_model(N: int, device: torch.device,
                       res_upd_freq: int = 1) -> PhysHGNet3D:
    """随机初始化模型（用于无预训练权重时的演示）。"""
    cfg = {
        **DEFAULT_CONFIG_3D,
        "m_anchors"           : max(16, N // 60),
        "residual_update_freq": res_upd_freq,
        "use_physics_anchor"  : True,
        "cg_max_iter"         : 10,
    }
    model = PhysHGNet3D(cfg).to(device)
    model.eval()
    print("  ⚠ 使用随机初始化权重（无预训练）")
    return model


# ═══════════════════════════════════════════════════════════════════
# Part 3 — 逐步推理，捕获锚点轨迹
# ═══════════════════════════════════════════════════════════════════

def run_inference_and_capture(
    model:    PhysHGNet3D,
    batch:    dict,
    T_steps:  int,
    device:   torch.device,
) -> tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    逐时间步 forward（窗口=2），记录每步结束后的锚点坐标。

    返回
    ----
    anchor_list : List[(m, 3)]   每步锚点坐标（numpy）
    u_history   : (T, N)          每步节点温度场
    laser_pos   : (T, 3)          每步激光中心坐标（供可视化叠加）
    """
    # 重置模型状态
    model._graph_cache     = None
    model._cache_key       = None
    model._L_physics_cache = None
    model._residual_cache  = None
    model._grad_norm_cache = None
    model._step_counter    = 0

    nodes_t  = batch["nodes"].to(device)
    src_t    = batch["source_terms"].to(device)   # (B, T, N, 1)
    tp_t     = batch["time_points"].to(device)    # (T,)
    u_curr   = batch["initial_conditions"].to(device)  # (B, N, 1)

    N        = nodes_t.shape[0]
    B        = u_curr.shape[0]

    anchor_list: List[np.ndarray] = []
    u_history   = np.zeros((T_steps, N), dtype=np.float32)
    laser_pos   = np.zeros((T_steps, 3), dtype=np.float32)

    u_history[0] = u_curr[0, :, 0].cpu().numpy()

    # 激光位置从 source_terms 中反推质心
    src_np = src_t[0].cpu().numpy()  # (T, N, 1)
    nodes_np = nodes_t.cpu().numpy()
    for t in range(T_steps):
        w = src_np[t, :, 0]
        if w.sum() > 1e-8:
            laser_pos[t] = (nodes_np * w[:, None]).sum(0) / w.sum()
        else:
            laser_pos[t] = nodes_np.mean(0)

    # 捕获 t=0 时的初始锚点（第一次 forward 时建立）
    with torch.no_grad():
        mini = {
            **batch,
            "nodes"              : nodes_t,
            "initial_conditions" : u_curr,
            "source_terms"       : src_t[:, 0:2],
            "time_points"        : tp_t[0:2],
        }
        _out = model(mini)

    if model._graph_cache is not None:
        anchor_list.append(model._graph_cache["anchor_coords"].cpu().numpy())
    else:
        anchor_list.append(nodes_np[:1])

    u_curr = _out["u_final"][:, 1:2, :, :].squeeze(1)

    # t = 1 … T-1
    for t in range(1, T_steps - 1):
        with torch.no_grad():
            mini = {
                **batch,
                "nodes"              : nodes_t,
                "initial_conditions" : u_curr,
                "source_terms"       : src_t[:, t:t + 2],
                "time_points"        : tp_t[t:t + 2],
            }
            out = model(mini)

        u_curr = out["u_final"][:, 1:2, :, :].squeeze(1)
        u_history[t + 1] = u_curr[0, :, 0].cpu().numpy()

        if model._graph_cache is not None:
            anchor_list.append(model._graph_cache["anchor_coords"].cpu().numpy())
        else:
            anchor_list.append(anchor_list[-1])

    # 补齐最后一步（如果缺少）
    while len(anchor_list) < T_steps:
        anchor_list.append(anchor_list[-1])

    return anchor_list, u_history, laser_pos


# ═══════════════════════════════════════════════════════════════════
# Part 4 — 可视化绘图
# ═══════════════════════════════════════════════════════════════════

# 全局调色板
CMAP_TEMP   = cm.get_cmap("hot")
CMAP_TIME   = cm.get_cmap("plasma")
ANCHOR_CMAP = cm.get_cmap("cool")


def _temp_colors(u: np.ndarray) -> np.ndarray:
    """将温度场归一化为 RGBA 颜色数组。"""
    vmin, vmax = u.min(), u.max()
    if vmax - vmin < 1.0:
        vmax = vmin + 1.0
    return CMAP_TEMP((u - vmin) / (vmax - vmin))


def draw_single_step(
    nodes:     np.ndarray,   # (N, 3)
    u:         np.ndarray,   # (N,)
    anchors:   np.ndarray,   # (m, 3)
    laser_pos: np.ndarray,   # (3,)
    step:      int,
    T_total:   int,
    out_path:  Path,
    elev: int  = 25,
    azim: int  = -60,
) -> None:
    """
    绘制单个时间步的四格图：
      左上：3D 散点（温度场 + 锚点）
      右上：XY 投影
      左下：XZ 投影
      右下：锚点密度热图（XY）
    """
    fig = plt.figure(figsize=(14, 10), facecolor="#111111")
    fig.suptitle(
        f"PhysHGNet3D — 锚点可视化   时间步 {step:>3d} / {T_total - 1}",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )

    node_c = _temp_colors(u)[:, :3]          # (N, 3) RGB
    t_frac = step / max(T_total - 1, 1)
    a_color = ANCHOR_CMAP(t_frac)             # 单色：按时步着色锚点

    # ── 左上：3D 散点 ─────────────────────────────────────────────
    ax3d = fig.add_subplot(2, 2, 1, projection="3d",
                           facecolor="#1a1a2e")
    ax3d.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                 c=node_c, s=3, alpha=0.25, linewidths=0)
    ax3d.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2],
                 c=[a_color], s=120, marker="*",
                 edgecolors="white", linewidths=0.4,
                 zorder=5, label=f"锚点 m={len(anchors)}")
    # 激光位置
    ax3d.scatter(*laser_pos, c="yellow", s=200, marker="X",
                 edgecolors="white", linewidths=0.8, zorder=6, label="激光")
    ax3d.set_xlabel("X", color="gray", fontsize=8)
    ax3d.set_ylabel("Y", color="gray", fontsize=8)
    ax3d.set_zlabel("Z", color="gray", fontsize=8)
    ax3d.tick_params(colors="gray", labelsize=6)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_title("3D 视图（温度 + 锚点）", color="white", fontsize=9, pad=4)
    ax3d.legend(loc="upper left", fontsize=7, labelcolor="white",
                facecolor="#1a1a2e", edgecolor="gray")
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False

    # ── 右上：XY 投影 ─────────────────────────────────────────────
    ax_xy = fig.add_subplot(2, 2, 2, facecolor="#1a1a2e")
    ax_xy.scatter(nodes[:, 0], nodes[:, 1],
                  c=node_c, s=3, alpha=0.3, linewidths=0)
    ax_xy.scatter(anchors[:, 0], anchors[:, 1],
                  c=[a_color], s=120, marker="*",
                  edgecolors="white", linewidths=0.5, zorder=5)
    ax_xy.scatter(laser_pos[0], laser_pos[1],
                  c="yellow", s=200, marker="X",
                  edgecolors="white", linewidths=0.8, zorder=6)
    ax_xy.set_xlabel("X", color="gray", fontsize=8)
    ax_xy.set_ylabel("Y", color="gray", fontsize=8)
    ax_xy.set_title("XY 投影", color="white", fontsize=9)
    ax_xy.tick_params(colors="gray", labelsize=7)
    for sp in ax_xy.spines.values():
        sp.set_color("gray")

    # ── 左下：XZ 投影 ─────────────────────────────────────────────
    ax_xz = fig.add_subplot(2, 2, 3, facecolor="#1a1a2e")
    ax_xz.scatter(nodes[:, 0], nodes[:, 2],
                  c=node_c, s=3, alpha=0.3, linewidths=0)
    ax_xz.scatter(anchors[:, 0], anchors[:, 2],
                  c=[a_color], s=120, marker="*",
                  edgecolors="white", linewidths=0.5, zorder=5)
    ax_xz.scatter(laser_pos[0], laser_pos[2],
                  c="yellow", s=200, marker="X",
                  edgecolors="white", linewidths=0.8, zorder=6)
    ax_xz.set_xlabel("X", color="gray", fontsize=8)
    ax_xz.set_ylabel("Z（深度）", color="gray", fontsize=8)
    ax_xz.set_title("XZ 投影（深度方向）", color="white", fontsize=9)
    ax_xz.tick_params(colors="gray", labelsize=7)
    for sp in ax_xz.spines.values():
        sp.set_color("gray")

    # ── 右下：温度 colorbar + 锚点统计 ──────────────────────────
    ax_info = fig.add_subplot(2, 2, 4, facecolor="#1a1a2e")
    ax_info.axis("off")

    # 温度渐变条
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax_info.imshow(grad, aspect="auto", cmap="hot",
                   extent=[0, 1, 0.55, 0.7], transform=ax_info.transAxes)
    ax_info.text(0.05, 0.52, f"{u.min():.0f} K",
                 transform=ax_info.transAxes,
                 color="white", fontsize=8, ha="left")
    ax_info.text(0.95, 0.52, f"{u.max():.0f} K",
                 transform=ax_info.transAxes,
                 color="white", fontsize=8, ha="right")
    ax_info.text(0.5, 0.74, "温度色标",
                 transform=ax_info.transAxes,
                 color="lightgray", fontsize=9, ha="center")

    # 统计信息
    stats = [
        f"时间步：{step} / {T_total - 1}",
        f"进度：{100 * t_frac:.1f}%",
        f"锚点数：{len(anchors)}",
        f"节点数：{len(nodes)}",
        f"温度 min：{u.min():.1f} K",
        f"温度 max：{u.max():.1f} K",
        f"温度 mean：{u.mean():.1f} K",
        f"激光位置 X：{laser_pos[0]:.3f}",
    ]
    for i, s in enumerate(stats):
        ax_info.text(0.08, 0.45 - i * 0.055, s,
                     transform=ax_info.transAxes,
                     color="white", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_summary(
    nodes:        np.ndarray,    # (N, 3)
    anchor_list:  List[np.ndarray],
    laser_pos:    np.ndarray,    # (T, 3)
    T_steps:      int,
    out_path:     Path,
) -> None:
    """
    总览图：在一张图上用时间颜色叠加所有步骤的锚点位置，
    体现锚点随热源移动而迁移的轨迹。
    """
    fig = plt.figure(figsize=(16, 6), facecolor="#111111")
    fig.suptitle(
        "PhysHGNet3D — 锚点位置演化总览（颜色深→浅 = 时间早→晚）",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    axes = [
        fig.add_subplot(1, 3, 1, projection="3d", facecolor="#1a1a2e"),
        fig.add_subplot(1, 3, 2, facecolor="#1a1a2e"),
        fig.add_subplot(1, 3, 3, facecolor="#1a1a2e"),
    ]
    titles = ["3D 全景", "XY 投影", "XZ 投影"]

    # 底层：灰色网格节点
    for ax, title in zip(axes, titles):
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.tick_params(colors="gray", labelsize=6)
        if hasattr(ax, "view_init"):
            ax.view_init(elev=25, azim=-55)
            ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                       c="gray", s=2, alpha=0.15, linewidths=0)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
        else:
            for sp in ax.spines.values():
                sp.set_color("#444")

    # 绘制底层灰色节点（2D）
    axes[1].scatter(nodes[:, 0], nodes[:, 1], c="gray", s=2, alpha=0.15, linewidths=0)
    axes[2].scatter(nodes[:, 0], nodes[:, 2], c="gray", s=2, alpha=0.15, linewidths=0)

    # 逐步叠加锚点（颜色随时间变化）
    norm  = Normalize(vmin=0, vmax=T_steps - 1)
    cmap  = cm.get_cmap("plasma")

    for t, ac in enumerate(anchor_list):
        color  = cmap(norm(t))
        alpha  = 0.3 + 0.7 * (t / max(T_steps - 1, 1))   # 越晚越亮
        size   = 30 + 60 * (t / max(T_steps - 1, 1))

        axes[0].scatter(ac[:, 0], ac[:, 1], ac[:, 2],
                        c=[color], s=size * 0.5, alpha=alpha,
                        marker="*", edgecolors="none", linewidths=0)
        axes[1].scatter(ac[:, 0], ac[:, 1],
                        c=[color], s=size, alpha=alpha,
                        marker="*", edgecolors="none", linewidths=0)
        axes[2].scatter(ac[:, 0], ac[:, 2],
                        c=[color], s=size, alpha=alpha,
                        marker="*", edgecolors="none", linewidths=0)

    # 激光轨迹（黄色折线）
    axes[1].plot(laser_pos[:, 0], laser_pos[:, 1],
                 "y--", lw=1.5, alpha=0.7, label="激光路径")
    axes[2].plot(laser_pos[:, 0], laser_pos[:, 2],
                 "y--", lw=1.5, alpha=0.7, label="激光路径")
    axes[1].legend(fontsize=8, labelcolor="white",
                   facecolor="#1a1a2e", edgecolor="gray")

    # 轴标签
    for ax in axes[1:]:
        for sp in ax.spines.values():
            sp.set_color("#444")
    axes[1].set_xlabel("X", color="gray", fontsize=8)
    axes[1].set_ylabel("Y", color="gray", fontsize=8)
    axes[2].set_xlabel("X", color="gray", fontsize=8)
    axes[2].set_ylabel("Z", color="gray", fontsize=8)

    # 颜色条
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("时间步", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ 总览图 → {out_path}")


def make_gif(frame_paths: List[Path], gif_path: Path,
             fps: int = 4) -> None:
    """将 PNG 序列合成为 GIF。"""
    if not HAS_IMAGEIO:
        print("  ⚠ imageio 未安装，跳过 GIF 生成。"
              "安装方法：pip install imageio")
        return
    frames = [imageio.imread(str(p)) for p in frame_paths]
    duration = 1.0 / fps  # seconds per frame
    imageio.mimsave(str(gif_path), frames, duration=duration,
                    loop=0)
    print(f"  ✓ GIF → {gif_path}  "
          f"（{len(frames)} 帧, {fps} fps）")


# ═══════════════════════════════════════════════════════════════════
# Part 5 — 主程序
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="PhysHGNet3D 锚点时序可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_nodes",    type=int,   default=500,
                   help="节点数，与模型保存时一致")
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints/phys_hgnet_3d",
                   help="存放 best_{N}.pth 的目录")
    p.add_argument("--T_steps",    type=int,   default=20,
                   help="可视化时间步数")
    p.add_argument("--out_dir",    type=str,   default="anchor_viz_output",
                   help="输出目录（截图 + GIF）")
    p.add_argument("--fps",        type=int,   default=4,
                   help="GIF 帧率")
    p.add_argument("--res_freq",   type=int,   default=1,
                   help="锚点刷新频率（1=每步刷新，最能体现动态）")
    p.add_argument("--seed",       type=int,   default=42,
                   help="随机种子（控制网格和轨迹生成）")
    p.add_argument("--no_pretrain", action="store_true",
                   help="不加载预训练权重，使用随机初始化（调试用）")
    p.add_argument("--device",     type=str,   default="cpu",
                   choices=["cpu", "cuda"],
                   help="推理设备")
    p.add_argument("--azim",       type=int,   default=-60,
                   help="3D 子图方位角")
    p.add_argument("--elev",       type=int,   default=25,
                   help="3D 子图仰角")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device
                          if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = args.n_nodes
    T = args.T_steps
    B = 1

    print("=" * 65)
    print(f"  PhysHGNet3D 锚点可视化")
    print(f"  节点数 N={N}  时间步 T={T}  输出目录: {out_dir}")
    print("=" * 65)

    # ── Step 1: 生成合成网格 & 轨迹 ──────────────────────────────
    print("\n[1/4] 生成合成 3D 网格与激光轨迹…")
    nodes_np, tets_np, edges_np = _make_3d_mesh(N, seed=args.seed)
    T_init, src_np, time_pts_np = _make_laser_trajectory(
        nodes_np, T=T, B=B, seed=args.seed)

    nodes  = torch.tensor(nodes_np,   dtype=torch.float32)
    tets   = torch.tensor(tets_np,    dtype=torch.long)
    edges  = torch.tensor(edges_np,   dtype=torch.long)
    u_init = torch.tensor(T_init,     dtype=torch.float32)   # (B, N, 1)
    src    = torch.tensor(src_np,     dtype=torch.float32)   # (B, T, N, 1)
    tps    = torch.tensor(time_pts_np, dtype=torch.float32)  # (T,)

    batch = {
        "nodes"              : nodes,
        "edges"              : edges,
        "tets"               : tets,
        "initial_conditions" : u_init,
        "source_terms"       : src,
        "time_points"        : tps,
        "node_type"          : torch.zeros(N, dtype=torch.long),
        "boundary_info"      : {},
    }

    print(f"  网格：{N} 节点，{len(tets_np)} 四面体，{len(edges_np)} 边")

    # ── Step 2: 加载模型 ─────────────────────────────────────────
    print(f"\n[2/4] 加载模型…")
    if args.no_pretrain:
        model = build_random_model(N, device, res_upd_freq=args.res_freq)
    else:
        try:
            model = load_model(args.ckpt_dir, N, device,
                               res_upd_freq=args.res_freq)
        except FileNotFoundError as e:
            print(f"  ! {e}")
            print("  → 回退到随机初始化模型（--no_pretrain）")
            model = build_random_model(N, device, res_upd_freq=args.res_freq)

    m_anchors = model.m_anchors
    print(f"  锚点数（期望）m={m_anchors}，刷新频率 res_freq={args.res_freq}")

    # ── Step 3: 逐步推理，捕获锚点 ───────────────────────────────
    print(f"\n[3/4] 逐时间步推理（共 {T} 步）…")
    anchor_list, u_history, laser_pos = run_inference_and_capture(
        model, batch, T, device)

    # 计算锚点变化量（诊断）
    diffs = []
    for i in range(1, len(anchor_list)):
        a1 = np.sort(anchor_list[i],   axis=0)
        a0 = np.sort(anchor_list[i-1], axis=0)
        min_len = min(len(a1), len(a0))
        diffs.append(np.linalg.norm(a1[:min_len] - a0[:min_len]))
    print(f"  锚点平均漂移（相邻步 Frobenius）: {np.mean(diffs):.4f}")
    if np.mean(diffs) < 1e-6:
        print("  ⚠  锚点几乎不变！请确认 phys_hgnet.py 已应用锚点更新修复。")
    else:
        print("  ✓  锚点随时间步正确演化。")

    # ── Step 4: 绘图 ─────────────────────────────────────────────
    print(f"\n[4/4] 生成可视化…")
    frame_paths: List[Path] = []

    for t in range(T):
        frame_p = out_dir / f"anchor_step_{t:03d}.png"
        draw_single_step(
            nodes   = nodes_np,
            u       = u_history[t],
            anchors = anchor_list[min(t, len(anchor_list) - 1)],
            laser_pos = laser_pos[t],
            step    = t,
            T_total = T,
            out_path = frame_p,
            elev    = args.elev,
            azim    = args.azim,
        )
        frame_paths.append(frame_p)

        if t % max(1, T // 5) == 0 or t == T - 1:
            print(f"    步骤 {t:>3d}/{T-1} → {frame_p.name}")

    # 总览图
    summary_path = out_dir / "anchor_summary.png"
    draw_summary(
        nodes       = nodes_np,
        anchor_list = anchor_list,
        laser_pos   = laser_pos,
        T_steps     = T,
        out_path    = summary_path,
    )

    # GIF
    gif_path = out_dir / "anchor_evolution.gif"
    make_gif(frame_paths, gif_path, fps=args.fps)

    print("\n" + "=" * 65)
    print(f"  完成！输出目录：{out_dir.resolve()}")
    print(f"  逐帧截图：anchor_step_000.png … anchor_step_{T-1:03d}.png")
    print(f"  总览图：  anchor_summary.png")
    print(f"  动画：    anchor_evolution.gif")
    print("=" * 65)


if __name__ == "__main__":
    main()
