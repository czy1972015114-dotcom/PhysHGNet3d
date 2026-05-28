"""
visualize_anchors.py — PhysHGNet3D 锚点分布可视化
====================================================
功能：
  1. 加载训练不同阶段的检查点（epoch 1 / 5 / 10 / 20 / best）
  2. 对同一个样本进行前向推断，Hook 截取 C2 锚点选择结果
  3. 生成三种图：
       (a) 3D 散点图：网格节点 + 锚点（matplotlib PDF/PNG）
       (b) 多 epoch 对比图（4 子图拼图，适合论文投稿）
       (c) 交互式 HTML（plotly，可旋转）

使用方法
--------
  # 基本（自动查找所有 epoch 检查点）
  python visualize_anchors.py --n_nodes 4000

  # 手动指定要可视化的 epoch
  python visualize_anchors.py --n_nodes 4000 --epochs 1 5 10 20 40

  # 只生成论文对比图（不需要 plotly）
  python visualize_anchors.py --n_nodes 4000 --no_html

依赖：pip install matplotlib numpy h5py torch
可选：pip install plotly kaleido        (交互式 + 高质量 PNG 导出)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import h5py
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────
# 尝试导入项目模块（必须在同目录或 PYTHONPATH 中）
# ─────────────────────────────────────────────────────────────────
try:
    from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D
    from dataset_3d    import LaserHardening3DDataset, collate_fn_3d, find_h5_file
    from torch.utils.data import DataLoader
    PROJECT_IMPORTED = True
except ImportError as e:
    print(f"[警告] 无法导入项目模块：{e}")
    print("  请在 PhysHGNet3d 项目目录下运行此脚本，或将项目目录加入 PYTHONPATH。")
    PROJECT_IMPORTED = False

# ─────────────────────────────────────────────────────────────────
# 可选 plotly
# ─────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ═════════════════════════════════════════════════════════════════
# Part 1 — 锚点提取（Hook 机制）
# ═════════════════════════════════════════════════════════════════

class AnchorHook:
    """
    用 forward hook 截取 C2（Learned Coarse）模块输出的锚点索引。

    兼容多种实现方式：
      方式A：模块直接返回 (anchor_indices: LongTensor, ...)
      方式B：模块输出 dict，含 'anchor_idx' 键
      方式C：模块输出 anchor_scores，外部 top-k 选取
      方式D：anchor_indices 作为模型的 Buffer 或 Attribute 在 forward 中被写入
    """

    def __init__(self):
        self._captured: List[torch.Tensor] = []
        self._handles  = []

    # ── 注册 Hook ───────────────────────────────────────────────
    def register(self, model: nn.Module) -> bool:
        """
        自动搜索模型中可能含有锚点选择逻辑的子模块并挂 hook。
        返回是否成功找到至少一个候选模块。
        """
        TARGET_NAMES = [
            "learned_coarse", "anchor_select", "anchor_scorer",
            "coarse_pool", "anchor_pooling", "c2_module",
            "coarsen", "hierarchical_pool", "learned_pooling",
        ]
        found = False
        for name, module in model.named_modules():
            lower = name.lower()
            if any(t in lower for t in TARGET_NAMES):
                h = module.register_forward_hook(self._hook_fn)
                self._handles.append(h)
                print(f"  [Hook] 挂载到模块: {name}")
                found = True
        if not found:
            # 退化：直接 hook 整个模型，捕获输出中的锚点信息
            h = model.register_forward_hook(self._model_hook_fn)
            self._handles.append(h)
            print(f"  [Hook] 未找到 C2 子模块，已 hook 整个模型（将尝试从输出提取）")
        return found

    def _hook_fn(self, module, inputs, output):
        """子模块级 hook"""
        idx = self._try_extract_indices(output)
        if idx is not None:
            self._captured.append(idx.detach().cpu())

    def _model_hook_fn(self, module, inputs, output):
        """整模型级 hook（备用）"""
        if isinstance(output, dict):
            for key in ("anchor_idx", "anchor_indices", "anchor_ids",
                        "coarse_idx", "pooling_idx"):
                if key in output:
                    self._captured.append(output[key].detach().cpu())
                    return

    @staticmethod
    def _try_extract_indices(output) -> Optional[torch.Tensor]:
        """尝试从不同格式的输出中提取锚点索引 Tensor"""
        if isinstance(output, torch.Tensor):
            if output.dtype in (torch.long, torch.int32, torch.int64):
                return output.flatten()
            if output.dtype == torch.float32 and output.dim() == 1:
                # 可能是 score，取 top-k 作为近似可视化
                return None  # 不能确定 k，跳过
        if isinstance(output, (tuple, list)) and len(output) >= 1:
            return AnchorHook._try_extract_indices(output[0])
        if isinstance(output, dict):
            for key in ("anchor_idx", "anchor_indices", "indices", "idx"):
                if key in output:
                    return AnchorHook._try_extract_indices(output[key])
        return None

    def get_indices(self) -> Optional[np.ndarray]:
        """返回本次前向传播中捕获到的最后一个锚点索引数组"""
        if not self._captured:
            return None
        idx = self._captured[-1]
        if idx.dim() > 1:
            idx = idx.flatten()
        return idx.numpy().astype(int)

    def clear(self):
        self._captured.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def get_anchor_indices_from_model(model: nn.Module,
                                  batch: Dict[str, Any],
                                  device: torch.device,
                                  n_nodes: int,
                                  m_anchors: int) -> np.ndarray:
    """
    尽力从模型中提取锚点索引，若 hook 无法捕获则降级为坐标聚类估算。
    """
    hook = AnchorHook()
    has_c2 = hook.register(model)

    model.eval()
    with torch.no_grad():
        out = model(batch)

    captured = hook.get_indices()
    hook.remove()

    # ── 策略 1：Hook 成功 ──────────────────────────────────────
    if captured is not None and len(captured) > 0:
        print(f"  [锚点] Hook 捕获到 {len(captured)} 个锚点索引")
        return captured

    # ── 策略 2：模型输出含 anchor 字段 ──────────────────────────
    if isinstance(out, dict):
        for key in ("anchor_idx", "anchor_indices", "anchor_ids"):
            if key in out:
                idx = out[key].detach().cpu().numpy().flatten().astype(int)
                print(f"  [锚点] 从输出字段 '{key}' 获取 {len(idx)} 个锚点")
                return idx

    # ── 策略 3：遍历模型属性查找 buffer/attribute ───────────────
    for attr in ("anchor_indices", "anchor_idx", "_anchor_idx"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, torch.Tensor) and val.numel() > 0:
                idx = val.detach().cpu().numpy().flatten().astype(int)
                print(f"  [锚点] 从模型属性 '{attr}' 获取 {len(idx)} 个锚点")
                return idx

    # ── 策略 4：FPS 降级（仅用于可视化占位，不代表真实锚点）───────
    print(f"  [锚点] ⚠ 无法截取真实锚点，使用 FPS 降级估算（m={m_anchors}）")
    # 自动找坐标键（与 main() 中逻辑一致）
    _coords = None
    for _ck in ["nodes", "pos", "coords", "node_coords", "positions", "xyz", "x"]:
        if _ck in batch:
            _v = batch[_ck]
            if hasattr(_v, "shape") and _v.shape[-1] == 3:
                _coords = _v.reshape(-1, 3).cpu().numpy()
                break
    if _coords is None:
        for _ck, _v in batch.items():
            if hasattr(_v, "shape") and _v.shape[-1] == 3 and _v.ndim >= 2:
                _coords = _v.reshape(-1, 3).cpu().numpy()
                break
    coords = _coords if _coords is not None else np.zeros((n_nodes, 3))
    return fps_downsample(coords, m_anchors)


def fps_downsample(coords: np.ndarray, k: int) -> np.ndarray:
    """最远点采样（Farthest Point Sampling），返回 k 个节点的索引"""
    N = len(coords)
    k = min(k, N)
    selected = [np.random.randint(N)]
    dist = np.full(N, np.inf)

    for _ in range(k - 1):
        d = np.sum((coords - coords[selected[-1]]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        selected.append(int(np.argmax(dist)))

    return np.array(selected, dtype=int)


# ═════════════════════════════════════════════════════════════════
# Part 2 — 检查点管理
# ═════════════════════════════════════════════════════════════════

def find_epoch_checkpoints(ckpt_dir: str, n_nodes: int) -> Dict[str, Path]:
    """
    查找检查点目录中的多 epoch 检查点文件。
    命名约定（支持两种）：
      epoch_{ep}_{N}.pth   —— 训练脚本周期性保存
      best_{N}.pth         —— 最优检查点
    """
    d = Path(ckpt_dir)
    result = {}

    # 周期性保存的检查点
    for p in sorted(d.glob(f"epoch_*_{n_nodes}.pth")):
        parts = p.stem.split("_")
        if len(parts) >= 2:
            try:
                ep = int(parts[1])
                result[f"epoch_{ep}"] = p
            except ValueError:
                pass

    # 最优检查点
    best = d / f"best_{n_nodes}.pth"
    if best.exists():
        result["best"] = best

    return result


def load_model_from_ckpt(ckpt_path: Path,
                          device: torch.device,
                          m_anchors: Optional[int] = None) -> PhysHGNet3D:
    """从检查点文件恢复 PhysHGNet3D 模型"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = {**DEFAULT_CONFIG_3D, **ckpt.get("config", {}), "spatial_dim": 3}
    if m_anchors is not None:
        cfg["m_anchors"] = m_anchors
    model = PhysHGNet3D(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


# ═════════════════════════════════════════════════════════════════
# Part 3 — 可视化
# ═════════════════════════════════════════════════════════════════

COLORS = {
    "mesh_node" : "#C8D6E5",   # 浅蓝灰（普通节点）
    "anchor"    : "#E74C3C",   # 红色（锚点）
    "edge"      : "#D5D8DC",   # 浅灰（网格边）
    "bg"        : "#FAFAFA",
}

# ── 3A: matplotlib 多子图对比 ────────────────────────────────────

def plot_comparison_grid(
    snapshots: List[Dict],   # [{label, coords(N,3), anchor_idx, temp(N)}]
    out_path: str,
    title: str = "Anchor Distribution across Training Epochs",
    elev: float = 25.0,
    azim: float = -55.0,
):
    """
    生成 2×n 或 1×n 的子图拼图（论文图），每个子图显示一个 epoch 的锚点分布。
    上排：俯视图（z 轴垂直纸面）；下排：3D 透视图。
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib import cm

    n = len(snapshots)
    fig = plt.figure(figsize=(4.5 * n, 8), dpi=150)
    fig.patch.set_facecolor("white")

    gs_top = gridspec.GridSpec(1, n, top=0.94, bottom=0.52,
                                left=0.03, right=0.97, wspace=0.05)
    gs_bot = gridspec.GridSpec(1, n, top=0.48, bottom=0.04,
                                left=0.03, right=0.97, wspace=0.05)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    for col, snap in enumerate(snapshots):
        coords   = snap["coords"]       # (N, 3)
        aidx     = snap["anchor_idx"]   # (M,)
        temp     = snap["temp"]         # (N,) or None
        label    = snap["label"]

        # 颜色映射：用温度给锚点着色
        if temp is not None:
            t_anchor = temp[aidx]
            norm  = Normalize(vmin=temp.min(), vmax=temp.max())
            cmap  = cm.plasma
            ac    = cmap(norm(t_anchor))
        else:
            ac = COLORS["anchor"]

        # 锚点占比
        ratio = len(aidx) / len(coords) * 100

        # ── 俯视图（xy 投影）──────────────────────────────────
        ax2d = fig.add_subplot(gs_top[0, col])
        ax2d.scatter(
            coords[:, 0], coords[:, 1],
            s=0.5, c=COLORS["mesh_node"], alpha=0.25, linewidths=0, rasterized=True
        )
        sc = ax2d.scatter(
            coords[aidx, 0], coords[aidx, 1],
            s=18, c=t_anchor if temp is not None else "red",
            cmap="plasma" if temp is not None else None,
            norm=norm if temp is not None else None,
            zorder=5, linewidths=0.3, edgecolors="black", alpha=0.85
        )
        ax2d.set_title(f"{label}\nM={len(aidx)} ({ratio:.1f}%)",
                        fontsize=10, pad=4)
        ax2d.set_aspect("equal")
        ax2d.axis("off")
        ax2d.set_facecolor(COLORS["bg"])

        # ── 3D 透视图 ─────────────────────────────────────────
        ax3d = fig.add_subplot(gs_bot[0, col], projection="3d")
        ax3d.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            s=0.3, c=COLORS["mesh_node"], alpha=0.12,
            linewidths=0, rasterized=True
        )
        ax3d.scatter(
            coords[aidx, 0], coords[aidx, 1], coords[aidx, 2],
            s=22, c=t_anchor if temp is not None else "red",
            cmap="plasma" if temp is not None else None,
            norm=norm if temp is not None else None,
            zorder=5, edgecolors="black", linewidths=0.3, alpha=0.9
        )
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_facecolor(COLORS["bg"])
        for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#CCCCCC")
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])

    # 色条
    if snapshots[0]["temp"] is not None:
        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=Normalize(
                                       vmin=snapshots[0]["temp"].min(),
                                       vmax=snapshots[0]["temp"].max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.4, pad=0.01,
                             orientation="vertical", fraction=0.01)
        cbar.set_label("Temperature (K)", fontsize=9)

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  [图] 对比图已保存: {out_path}")


# ── 3B: 单个 epoch 的精细图（可选） ────────────────────────────

def plot_single_epoch(
    coords: np.ndarray,
    anchor_idx: np.ndarray,
    temp: Optional[np.ndarray],
    label: str,
    out_path: str,
    elev: float = 30.0,
    azim: float = -60.0,
):
    """生成单 epoch 的高质量 3D 图（含 xyz 三视图 + 透视图）"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(14, 10), dpi=150, facecolor="white")
    fig.suptitle(f"Anchor Distribution — {label}", fontsize=13,
                 fontweight="bold", y=0.97)

    specs = [
        dict(type="axes", colspan=1),
        dict(type="axes", colspan=1),
        dict(type="axes", colspan=1),
        dict(type="axes3d", colspan=1),
    ]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, figure=fig, wspace=0.1, left=0.02, right=0.95,
                  top=0.88, bottom=0.06)

    norm  = Normalize(vmin=temp.min(), vmax=temp.max()) if temp is not None else None
    t_a   = temp[anchor_idx] if temp is not None else None

    proj_configs = [
        ("XY  (top view)",    0, 1, "z"),
        ("XZ  (front view)",  0, 2, "y"),
        ("YZ  (side view)",   1, 2, "x"),
    ]
    for i, (title, xi, yi, dropped) in enumerate(proj_configs):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(
            coords[:, xi], coords[:, yi],
            s=0.6, c=COLORS["mesh_node"], alpha=0.20, linewidths=0,
            rasterized=True
        )
        sc = ax.scatter(
            coords[anchor_idx, xi], coords[anchor_idx, yi],
            s=30,
            c=t_a if t_a is not None else "crimson",
            cmap="plasma" if t_a is not None else None,
            norm=norm,
            zorder=5, edgecolors="black", linewidths=0.4, alpha=0.9
        )
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(labelsize=7)

    # 3D 透视
    ax3 = fig.add_subplot(gs[0, 3], projection="3d")
    ax3.scatter(*[coords[:, i] for i in range(3)],
                s=0.4, c=COLORS["mesh_node"], alpha=0.10,
                linewidths=0, rasterized=True)
    ax3.scatter(
        *[coords[anchor_idx, i] for i in range(3)],
        s=30,
        c=t_a if t_a is not None else "crimson",
        cmap="plasma" if t_a is not None else None,
        norm=norm,
        edgecolors="black", linewidths=0.4, alpha=0.9
    )
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_title("3D Perspective", fontsize=10)
    ax3.set_facecolor(COLORS["bg"])

    # 统计注释
    ratio = len(anchor_idx) / len(coords) * 100
    fig.text(0.50, 0.01,
             f"N_total={len(coords):,}  |  M_anchors={len(anchor_idx):,}  "
             f"({ratio:.1f}%)  |  {label}",
             ha="center", fontsize=9, color="#555555")

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  [图] 单 epoch 图已保存: {out_path}")


# ── 3C: 交互式 HTML（plotly）────────────────────────────────────

def make_interactive_html(
    snapshots: List[Dict],
    out_path: str,
    title: str = "PhysHGNet3D — Anchor Evolution",
):
    """
    生成带 epoch 切换按钮的交互式 HTML（可旋转 3D）。
    每个 epoch 是一帧，通过 updatemenus 切换。
    """
    if not HAS_PLOTLY:
        print("  [跳过] plotly 未安装，跳过交互式 HTML 生成。"
              "  安装方式：pip install plotly")
        return

    # 为每个 epoch 创建两个 trace：全体节点（淡）+ 锚点（亮）
    all_traces    = []
    button_list   = []
    traces_per_ep = 2

    for snap_i, snap in enumerate(snapshots):
        coords = snap["coords"]
        aidx   = snap["anchor_idx"]
        temp   = snap["temp"]
        label  = snap["label"]
        t_a    = temp[aidx].tolist() if temp is not None else None

        # 普通节点
        trace_mesh = go.Scatter3d(
            x=coords[:, 0].tolist(),
            y=coords[:, 1].tolist(),
            z=coords[:, 2].tolist(),
            mode="markers",
            marker=dict(size=1.0, color="#AABBCC", opacity=0.15),
            name="All nodes",
            visible=(snap_i == 0),
            showlegend=False,
            hoverinfo="skip",
        )

        # 锚点
        trace_anchor = go.Scatter3d(
            x=coords[aidx, 0].tolist(),
            y=coords[aidx, 1].tolist(),
            z=coords[aidx, 2].tolist(),
            mode="markers",
            marker=dict(
                size=4,
                color=t_a if t_a is not None else "red",
                colorscale="Plasma",
                showscale=(snap_i == 0) and (t_a is not None),
                colorbar=dict(title="Temp (K)", thickness=14, x=1.01)
                         if snap_i == 0 and t_a is not None else None,
                opacity=0.90,
                line=dict(width=0.5, color="black"),
            ),
            name=f"Anchors M={len(aidx)}",
            visible=(snap_i == 0),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                + ("<br>T=%{marker.color:.1f} K" if t_a is not None else "")
                + "<extra></extra>"
            ),
        )
        all_traces.extend([trace_mesh, trace_anchor])

        # 切换按钮
        vis = [False] * (len(snapshots) * traces_per_ep)
        vis[snap_i * traces_per_ep]     = True
        vis[snap_i * traces_per_ep + 1] = True
        button_list.append(dict(
            label=label,
            method="update",
            args=[{"visible": vis},
                  {"title": f"{title} — {label}  "
                            f"(M={len(aidx)}, {len(aidx)/len(coords)*100:.1f}%)"}],
        ))

    layout = go.Layout(
        title=dict(text=f"{title} — {snapshots[0]['label']}",
                   font=dict(size=14)),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#F8F9FA",
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.5, xanchor="center", y=1.12, yanchor="top",
            showactive=True,
            buttons=button_list,
            bgcolor="#EEF2F7",
            bordercolor="#AAAAAA",
            font=dict(size=12),
        )],
        margin=dict(l=0, r=60, t=90, b=0),
        paper_bgcolor="white",
        legend=dict(x=0.02, y=0.98),
        width=960, height=700,
    )

    fig = go.Figure(data=all_traces, layout=layout)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"  [图] 交互式 HTML 已保存: {out_path}")


# ── 3D: 锚点密度热图（2D 投影）──────────────────────────────────

def plot_anchor_density_evolution(
    snapshots: List[Dict],
    out_path: str,
):
    """
    绘制多 epoch 的锚点密度热图（XY 投影 KDE），
    直观展示锚点随训练逐渐聚集到高梯度区域。
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LogNorm

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4),
                              facecolor="white", dpi=150)
    if n == 1:
        axes = [axes]

    fig.suptitle("Anchor Density (XY projection)", fontsize=12,
                 fontweight="bold")

    for ax, snap in zip(axes, snapshots):
        coords = snap["coords"]
        aidx   = snap["anchor_idx"]
        ax_pts = coords[aidx, :2]   # (M, 2)

        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()

        try:
            kde = gaussian_kde(ax_pts.T, bw_method="silverman")
            xi, yi = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
            zi     = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
            ax.contourf(xi, yi, zi, levels=14, cmap="Reds", alpha=0.85)
        except Exception:
            # KDE 失败时退化为散点
            ax.scatter(ax_pts[:, 0], ax_pts[:, 1],
                        s=8, c="red", alpha=0.6)

        ax.scatter(coords[:, 0], coords[:, 1],
                    s=0.4, c=COLORS["mesh_node"], alpha=0.15, zorder=0,
                    rasterized=True)
        ax.scatter(ax_pts[:, 0], ax_pts[:, 1],
                    s=6, c="crimson", alpha=0.7, zorder=5)
        ax.set_title(snap["label"], fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  [图] 密度热图已保存: {out_path}")


# ═════════════════════════════════════════════════════════════════
# Part 4 — 训练脚本 Patch（周期性保存 anchor snapshots）
# ═════════════════════════════════════════════════════════════════

TRAINING_PATCH = '''
# ── 在 train_phys_hgnet_3d.py 的 train() 函数中，epoch 循环末尾添加 ──
# （在 scheduler.step() 之后，if is_main: 块内添加以下代码）

# 每 N 个 epoch 保存一次锚点快照（用于可视化）
ANCHOR_VIZ_EPOCHS = {1, 5, 10, 20, 40}  # 根据实际 epoch 数调整

if is_main and (epoch + 1) in ANCHOR_VIZ_EPOCHS:
    snap_dir = Path(args.ckpt_dir) / f"anchor_snapshots_{train_ds.N}"
    snap_dir.mkdir(exist_ok=True)
    snap_path = snap_dir / f"epoch_{epoch+1:04d}_{train_ds.N}.pth"
    torch.save({
        "model"    : raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch"    : epoch,
        "best_rne" : best_rne,
        "config"   : model_cfg,
    }, snap_path)
    print(f"  📸 锚点快照已保存: {snap_path}")
'''

# ═════════════════════════════════════════════════════════════════
# Part 5 — 主流程
# ═════════════════════════════════════════════════════════════════

def batch_to_device(batch, device):
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


def main(args):
    if not PROJECT_IMPORTED:
        sys.exit(1)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"╔{'═'*60}╗")
    print(f"║  锚点可视化  |  N≈{args.n_nodes}  |  device={device}")
    print(f"╚{'═'*60}╝\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 找数据 ──────────────────────────────────────────────────
    h5_path = find_h5_file(args.data_dir, args.n_nodes)
    ds = LaserHardening3DDataset(
        str(h5_path), window_size=10, stride=10)
    sample_loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn_3d, num_workers=0)
    sample_batch = next(iter(sample_loader))
    sample_batch = batch_to_device(sample_batch, device)

    # ── 自动探测坐标键名 ───────────────────────────────────────
    # ── 读取节点坐标 ────────────────────────────────────────────
    # batch['nodes']: (N, 3)  无 batch 维度（dataset 未加 batch 维）
    COORD_KEYS = ["nodes", "pos", "coords", "node_coords",
                  "positions", "xyz", "node_pos", "x"]
    coords = None
    coord_key_used = None
    for _ck in COORD_KEYS:
        if _ck in sample_batch:
            _v = sample_batch[_ck]
            if not hasattr(_v, "shape"):
                continue
            # 支持 (N,3) 和 (B,N,3) 两种形状
            if _v.ndim == 2 and _v.shape[-1] == 3:
                coords = _v.cpu().numpy()
                coord_key_used = _ck
                break
            if _v.ndim == 3 and _v.shape[-1] == 3:
                coords = _v.squeeze(0).cpu().numpy()
                coord_key_used = _ck
                break
    # 通用兜底：找第一个末维=3的 tensor
    if coords is None:
        for _ck, _v in sample_batch.items():
            if hasattr(_v, "shape") and _v.shape[-1] == 3 and _v.ndim >= 2:
                coords = _v.reshape(-1, 3).cpu().numpy()
                coord_key_used = _ck
                break
    if coords is None:
        print("  ✗ 无法从 batch 中找到节点坐标。")
        print(f"  batch 键: {list(sample_batch.keys())}")
        sys.exit(1)
    print(f"  节点坐标键: {coord_key_used!r}  shape={coords.shape}")

    # ── 读取温度场（用于锚点着色）────────────────────────────────
    # batch['targets']: (1, T, N, 1)  → 取最后时刻 → (N,)
    temp_all = None
    TEMP_KEYS = ["targets", "temperature", "u_seq", "u", "temp", "y"]
    for _tk in TEMP_KEYS:
        if _tk not in sample_batch:
            continue
        _tv = sample_batch[_tk]
        if not hasattr(_tv, "shape") or _tv.numel() == 0:
            continue
        t = _tv.cpu().numpy()
        # 去掉所有大小为 1 的维度，然后取最后一个时间步
        t = t.squeeze()          # (T, N, 1) 或 (T, N) 或 (N,) 等
        if t.ndim == 3:          # (T, N, C)
            t = t[-1, :, 0]      # 最后时刻，第一通道
        elif t.ndim == 2:
            # 判断哪个维度是 N
            if t.shape[0] == len(coords):
                t = t[:, 0] if t.shape[1] < t.shape[0] else t[-1]
            else:
                t = t[-1]        # (T, N) → 最后时刻
        # t 现在应该是 (N,)
        t = t.flatten()
        if len(t) == len(coords):
            temp_all = t
            print(f"  温度键: {_tk!r}  最终 shape={t.shape}  "
                  f"range=[{t.min():.1f}, {t.max():.1f}] K")
            break
    if temp_all is None:
        print("  温度: 未找到匹配，锚点将用统一红色标注")

    print(f"  节点数: {len(coords):,}")

    # ── 找检查点 ─────────────────────────────────────────────────
    # 优先查 anchor_snapshots 子目录
    snap_sub = Path(args.ckpt_dir) / f"anchor_snapshots_{args.n_nodes}"
    if snap_sub.exists():
        ckpt_dict = find_epoch_checkpoints(str(snap_sub), args.n_nodes)
    else:
        ckpt_dict = find_epoch_checkpoints(args.ckpt_dir, args.n_nodes)

    # 如果指定了 epoch 则只用指定的
    if args.epochs:
        filtered = {}
        for ep in args.epochs:
            key = f"epoch_{ep}"
            if key in ckpt_dict:
                filtered[key] = ckpt_dict[key]
        if "best" in ckpt_dict:
            filtered["best"] = ckpt_dict["best"]
        ckpt_dict = filtered

    # 如果一个 snapshot 都没找到，只用 best
    if not ckpt_dict:
        best = Path(args.ckpt_dir) / f"best_{args.n_nodes}.pth"
        if best.exists():
            ckpt_dict = {"best": best}
            print(f"  ⚠ 未找到 epoch 快照，仅使用最优检查点。")
            print(f"  提示：在 train_phys_hgnet_3d.py 中添加如下代码以周期性保存快照：")
            print(TRAINING_PATCH)
        else:
            print("  ✗ 未找到任何检查点！请先训练模型，或指定 --ckpt_dir")
            sys.exit(1)

    print(f"\n找到 {len(ckpt_dict)} 个检查点: {list(ckpt_dict.keys())}\n")

    # ── 逐检查点提取锚点 ─────────────────────────────────────────
    m_anchors_auto = max(64, args.n_nodes // 15)
    snapshots = []

    for label, ckpt_path in sorted(ckpt_dict.items()):
        print(f"── 处理 [{label}]  {ckpt_path.name} ──")
        try:
            model, cfg = load_model_from_ckpt(ckpt_path, device, m_anchors_auto)
            m_anchors = cfg.get("m_anchors", m_anchors_auto)
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            continue

        anchor_idx = get_anchor_indices_from_model(
            model, sample_batch, device, args.n_nodes, m_anchors)

        # 裁剪越界索引（保险）
        anchor_idx = anchor_idx[anchor_idx < len(coords)]

        snapshots.append({
            "label"     : label,
            "coords"    : coords,
            "anchor_idx": anchor_idx,
            "temp"      : temp_all,
            "m_anchors" : m_anchors,
        })

        # 单 epoch 精细图
        single_out = out_dir / f"anchor_single_{label}_{args.n_nodes}.png"
        plot_single_epoch(
            coords, anchor_idx, temp_all,
            label=label, out_path=str(single_out))

        # 保存 JSON 数据（方便后处理）
        json_out = out_dir / f"anchor_data_{label}_{args.n_nodes}.json"
        with open(json_out, "w") as f:
            json.dump({
                "label"           : label,
                "n_nodes"         : int(len(coords)),
                "m_anchors"       : int(m_anchors),
                "anchor_indices"  : anchor_idx.tolist(),
                "anchor_ratio_pct": round(len(anchor_idx) / len(coords) * 100, 2),
            }, f, indent=2)

    if not snapshots:
        print("  ✗ 无可用快照，退出。")
        sys.exit(1)

    # ── 多 epoch 对比图 ───────────────────────────────────────────
    compare_out = out_dir / f"anchor_comparison_{args.n_nodes}.png"
    plot_comparison_grid(
        snapshots,
        out_path=str(compare_out),
        title=f"PhysHGNet3D Anchor Distribution Evolution  (N={args.n_nodes})",
        elev=args.elev, azim=args.azim,
    )

    # ── 密度热图 ──────────────────────────────────────────────────
    try:
        from scipy.stats import gaussian_kde
        density_out = out_dir / f"anchor_density_{args.n_nodes}.png"
        plot_anchor_density_evolution(snapshots, out_path=str(density_out))
    except ImportError:
        print("  [跳过] scipy 未安装，跳过密度热图 (pip install scipy)")

    # ── 交互式 HTML ───────────────────────────────────────────────
    if not args.no_html:
        html_out = out_dir / f"anchor_interactive_{args.n_nodes}.html"
        make_interactive_html(
            snapshots,
            out_path=str(html_out),
            title=f"PhysHGNet3D Anchor Evolution  N={args.n_nodes}",
        )

    print(f"\n✓ 所有可视化文件已保存至: {out_dir}")
    print("\n生成文件列表：")
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<55} {size:>7.1f} KB")


def parse_args():
    p = argparse.ArgumentParser(description="PhysHGNet3D 锚点分布可视化")
    p.add_argument("--n_nodes",   type=int, default=4000)
    p.add_argument("--data_dir",  type=str, default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir",  type=str, default="checkpoints/phys_hgnet_3d")
    p.add_argument("--out_dir",   type=str, default="results_viz/anchors")
    p.add_argument("--epochs",    type=int, nargs="*", default=None,
                   help="要可视化的 epoch 编号，如 --epochs 1 5 10 20。"
                        "默认自动查找所有快照。")
    p.add_argument("--elev",      type=float, default=28.0)
    p.add_argument("--azim",      type=float, default=-55.0)
    p.add_argument("--gpu",       type=int,   default=0)
    p.add_argument("--no_html",   action="store_true",
                   help="不生成交互式 HTML（无需 plotly）")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
