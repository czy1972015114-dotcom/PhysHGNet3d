"""
dataset_3d.py — PhysHGNet 3D 数据集加载器（滑动窗口版）
=========================================================

核心改动：按照 DGNet 仓库的正确实现思路，
对每条长轨迹（T 步）做滑动窗口切分，
每个窗口作为一个独立训练样本。

HDF5 结构
---------
/mesh_meta/
    nodes (N,3)  edges (E,2)  faces (BF,3)  tets (F,4)
    node_volumes (N,)  L_physics_ei (2,2E')  L_physics_ew (2E',)
    time_points (T,)
/trajectory_{i}/
    node_features  (T, N, 1)
    source_terms   (T, N, 1)
    initial_condition (1, N, 1)

样本数量估算（以 n_traj=40, T=61, win=10, stride=5 为例）
-------------------------------------------------------
  每条轨迹窗口数 = floor((61 - 10) / 5) + 1 = 11
  总训练样本    = 40 × 11 = 440 个

参数说明
--------
window_size  : 每个子轨迹包含的时步数（含初始条件，即 T_sub = window_size）
stride       : 相邻窗口的起始步间隔（stride < window_size 时窗口有重叠）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────
# 路径工具
# ─────────────────────────────────────────────────────────────────

def find_h5_file(data_dir: str, n_nodes: int) -> Path:
    data_dir_ = Path(data_dir)
    for pattern in [f"pde_trajectories_3d_N{n_nodes}.h5",
                    "pde_trajectories_3d_N*.h5"]:
        matches = sorted(data_dir_.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"在 {data_dir} 中找不到 N≈{n_nodes} 的 3D HDF5 文件。"
        f"\n请先运行：python generate_laser_data_3d.py --n_nodes {n_nodes}"
    )


# ─────────────────────────────────────────────────────────────────
# 主数据集类（滑动窗口）
# ─────────────────────────────────────────────────────────────────

class LaserHardening3DDataset(Dataset):
    """3D 激光淬火 PDE 数据集，使用滑动窗口切分长轨迹。

    核心设计
    --------
    - 每条长轨迹（T 步）被切分为若干长度为 window_size 的短窗口。
    - 每个窗口 (traj_idx, start_t) 是一个独立的训练样本：
        initial_conditions = u[start_t]
        source_terms       = src[start_t : start_t + window_size]
        targets            = u[start_t : start_t + window_size]
    - 窗口间距由 stride 控制；stride < window_size 时窗口有重叠，
      data augmentation 效果更佳。

    参数
    ----
    h5_path     : HDF5 文件路径
    window_size : 子轨迹时步数（模型 forward 中的 T）
    stride      : 相邻窗口起点间隔，默认 = window_size//2（50% 重叠）
    traj_indices: 指定使用哪些轨迹（None=全部）
    """

    def __init__(
        self,
        h5_path:      str,
        window_size:  int  = 10,
        stride:       Optional[int] = None,
        traj_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.h5_path     = Path(h5_path)
        self.window_size = window_size
        # 默认 stride = window_size // 2（50% 重叠，样本数翻倍）
        self.stride      = stride if stride is not None else max(1, window_size // 2)

        # ── 读取全局网格元数据 ─────────────────────────────────────
        with h5py.File(self.h5_path, 'r') as f:
            meta = f['mesh_meta']
            self.nodes        = torch.tensor(meta['nodes'][:],        dtype=torch.float32)
            self.edges        = torch.tensor(meta['edges'][:],        dtype=torch.long)
            self.faces        = torch.tensor(meta['faces'][:],        dtype=torch.long)
            self.tets         = torch.tensor(meta['tets'][:],         dtype=torch.long)
            self.node_volumes = torch.tensor(meta['node_volumes'][:], dtype=torch.float32)
            self.time_points  = torch.tensor(meta['time_points'][:],  dtype=torch.float32)

            # 预存稀疏 Laplacian
            L_ei = torch.tensor(meta['L_physics_ei'][:], dtype=torch.long)
            L_ew = torch.tensor(meta['L_physics_ew'][:], dtype=torch.float32).clamp(min=0.0)  # 负 FEM 权重（钝角单元）强制为 0
            self.L_physics = {
                "type"         : "sparse",
                "edge_index"   : L_ei,
                "edge_weights" : L_ew,
                "node_volumes" : self.node_volumes,
            }

            # 节点类型（0=内部，1=边界）
            N    = self.nodes.shape[0]
            bset = set(self.faces.numpy().ravel().tolist()) if len(self.faces) > 0 else set()
            nt   = torch.zeros(N, dtype=torch.long)
            if bset:
                nt[list(bset)] = 1
            self.node_type = nt

            # 完整轨迹键列表
            all_keys = sorted(
                [k for k in f.keys() if k.startswith('trajectory_')],
                key=lambda x: int(x.split('_')[1]))
            self.traj_keys = (
                [all_keys[i] for i in traj_indices]
                if traj_indices is not None else all_keys
            )
            T_total = len(self.time_points)

        self.N       = self.nodes.shape[0]
        self.T_total = T_total

        # ── 构建 (轨迹索引, 起始步) 的所有窗口列表 ─────────────────
        self._windows: List[Tuple[int, int]] = []
        for ti in range(len(self.traj_keys)):
            t = 0
            while t + self.window_size <= self.T_total:
                self._windows.append((ti, t))
                t += self.stride
        # 如果没有任何合法窗口（轨迹太短），回退到单窗口
        if not self._windows:
            ws = min(self.window_size, self.T_total)
            for ti in range(len(self.traj_keys)):
                self._windows.append((ti, 0))
            self.window_size = ws

    # ── 统计 ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_idx, start_t = self._windows[idx]
        key = self.traj_keys[traj_idx]
        ws  = self.window_size
        end_t = start_t + ws

        with h5py.File(self.h5_path, 'r') as f:
            g   = f[key]
            # 切取当前窗口
            u   = torch.tensor(g['node_features'][start_t:end_t], dtype=torch.float32)  # (ws,N,1)
            src = torch.tensor(g['source_terms'][start_t:end_t],  dtype=torch.float32)  # (ws,N,1)
            # 窗口的 BC 与全局一致（不随时间变化）
            bnd_g   = g['boundary_info']['dirichlet']
            bnd_idx = torch.tensor(bnd_g['indices'][:], dtype=torch.long)
            bnd_val = torch.tensor(bnd_g['values'][:],  dtype=torch.float32)

        # 本窗口对应的时间坐标
        tp_window = self.time_points[start_t:end_t]

        return {
            # 网格（所有窗口共享）
            "nodes"             : self.nodes,
            "edges"             : self.edges,
            "faces"             : self.faces,
            "tets"              : self.tets,
            "node_volumes"      : self.node_volumes,
            "node_type"         : self.node_type,
            "L_physics"         : self.L_physics,
            # 本窗口数据
            "initial_conditions": u[0],            # (N, 1) — 窗口第 0 步
            "source_terms"      : src,             # (ws, N, 1)
            "time_points"       : tp_window,       # (ws,)
            "targets"           : u,               # (ws, N, 1) — 监督目标
            # BC
            "boundary_info": {
                "dirichlet": {"indices": bnd_idx, "values": bnd_val}
            },
        }

    def info(self) -> str:
        n_windows_per_traj = len(self._windows) / max(len(self.traj_keys), 1)
        return (
            f"LaserHardening3DDataset | 文件: {self.h5_path.name}\n"
            f"  节点 N={self.N} | 四面体={len(self.tets)} | 边={len(self.edges)}\n"
            f"  轨迹数={len(self.traj_keys)} | T_total={self.T_total}\n"
            f"  window_size={self.window_size} | stride={self.stride}\n"
            f"  每条轨迹窗口数≈{n_windows_per_traj:.1f}\n"
            f"  总样本数={len(self._windows)}"
        )


# ─────────────────────────────────────────────────────────────────
# Collate（将 list[dict] → batch dict）
# ─────────────────────────────────────────────────────────────────

def collate_fn_3d(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将多个窗口样本整合为 batch。

    网格数据（nodes/edges/tets/...）来自同一 HDF5，各样本完全相同，
    直接取第一个样本即可（不做 stack）。

    轨迹数据沿 batch 维度 stack：
        initial_conditions : (B, N, 1)
        source_terms       : (B, ws, N, 1)
        targets            : (B, ws, N, 1)
    """
    s0 = samples[0]
    return {
        # 网格（共享）
        "nodes"        : s0["nodes"],
        "edges"        : s0["edges"],
        "faces"        : s0["faces"],
        "tets"         : s0["tets"],
        "node_volumes" : s0["node_volumes"],
        "node_type"    : s0["node_type"],
        "L_physics"    : s0["L_physics"],
        "time_points"  : s0["time_points"],
        "boundary_info": s0["boundary_info"],
        # 轨迹（batch 维度）
        "initial_conditions": torch.stack([s["initial_conditions"] for s in samples], dim=0),
        "source_terms"      : torch.stack([s["source_terms"]       for s in samples], dim=0),
        "targets"           : torch.stack([s["targets"]            for s in samples], dim=0),
    }


# ─────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────

def build_dataloaders_3d(
    data_dir:    str,
    n_nodes:     int,
    batch_size:  int   = 4,
    window_size: int   = 10,
    stride:      Optional[int] = None,
    train_ratio: float = 0.8,
    num_workers: int   = 0,
    seed:        int   = 42,
) -> Tuple[DataLoader, DataLoader]:
    """构建 3D 训练 / 验证 DataLoader。

    注意：按轨迹切分 train/val，避免同一轨迹的不同窗口同时出现在
    两个 split 中导致数据泄露。
    """
    h5_path  = find_h5_file(data_dir, n_nodes)

    # 先统计轨迹总数，再按轨迹编号切分
    with h5py.File(str(h5_path), 'r') as f:
        all_traj_keys = sorted(
            [k for k in f.keys() if k.startswith('trajectory_')],
            key=lambda x: int(x.split('_')[1]))
    n_total  = len(all_traj_keys)
    n_train  = int(n_total * train_ratio)

    train_indices = list(range(n_train))
    val_indices   = list(range(n_train, n_total))

    train_ds = LaserHardening3DDataset(
        str(h5_path), window_size=window_size, stride=stride,
        traj_indices=train_indices)
    val_ds   = LaserHardening3DDataset(
        str(h5_path), window_size=window_size, stride=stride,
        traj_indices=val_indices)

    print(train_ds.info())
    print(f"  → train 窗口={len(train_ds)} | val 窗口={len(val_ds)}")

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_3d, num_workers=num_workers,
        drop_last=True, generator=gen)
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_3d, num_workers=num_workers)

    return train_loader, val_loader


def build_single_dataloader_3d(
    h5_path:     str,
    batch_size:  int  = 4,
    window_size: int  = 10,
    stride:      Optional[int] = None,
    shuffle:     bool = False,
    num_workers: int  = 0,
) -> DataLoader:
    """直接从 h5_path 构建 DataLoader（评测用）。"""
    ds = LaserHardening3DDataset(str(h5_path), window_size=window_size, stride=stride)
    print(ds.info())
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn_3d, num_workers=num_workers)


# ─────────────────────────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    h5 = sys.argv[1] if len(sys.argv) > 1 else \
        "data_laser_hardening_3d/pde_trajectories_3d_N1000.h5"

    for ws, st in [(10, 5), (10, 10), (20, 10)]:
        ds = LaserHardening3DDataset(h5, window_size=ws, stride=st)
        print(ds.info())
        s  = ds[0]
        print(f"  initial_conditions: {s['initial_conditions'].shape}")
        print(f"  source_terms:       {s['source_terms'].shape}")
        print(f"  targets:            {s['targets'].shape}")
        print()

    loader = build_single_dataloader_3d(h5, batch_size=4, window_size=10, stride=5)
    batch  = next(iter(loader))
    print("Batch shapes:")
    print(f"  initial_conditions: {batch['initial_conditions'].shape}")
    print(f"  source_terms:       {batch['source_terms'].shape}")
    print(f"  targets:            {batch['targets'].shape}")
    print("自测通过！")
