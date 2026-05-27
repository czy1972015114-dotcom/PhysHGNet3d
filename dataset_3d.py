"""
dataset_3d.py — PhysHGNet 3D 数据集加载器
==========================================

加载 generate_laser_data_3d.py 生成的 HDF5 文件，
返回与 PhysHGNet3D.forward 期望接口兼容的 batch 字典。

主要与 2D dataset.py 的差异
---------------------------
1. nodes 形状 (N, 3)，spatial_dim = 3
2. tets 键：(F, 4) 四面体连接（取代 faces (F,3) 三角面）
3. node_volumes：直接从 HDF5 读取（generate 时已保存）
4. L_physics：从 HDF5 预存的稀疏 Laplacian 读取，避免每次重建
5. edges 基于四面体提取的无向边

HDF5 文件结构（generate_laser_data_3d.py 输出）
-------------------------------------------------
/mesh_meta/
    nodes        (N, 3)  float32
    edges        (E, 2)  int32
    faces        (BF,3)  int32   边界三角面
    tets         (F, 4)  int32
    node_volumes (N,)    float32
    L_physics_ei (2,2E') int64    稀疏 Laplacian 边索引
    L_physics_ew (2E',)  float32  稀疏 Laplacian 边权重
    time_points  (T,)    float32
/trajectory_{i}/
    node_features     (T, N, 1) float32
    source_terms      (T, N, 1) float32
    initial_condition (1, N, 1) float32
    boundary_info/
        dirichlet/
            indices  int32
            values   float32
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────
# 辅助：路径解析
# ─────────────────────────────────────────────────────────────────

def find_h5_file(data_dir: str, n_nodes: int) -> Path:
    """在 data_dir 中寻找匹配 N 的 HDF5 文件。"""
    data_dir_ = Path(data_dir)
    # 优先精确匹配
    for pattern in [f"pde_trajectories_3d_N{n_nodes}.h5",
                    f"pde_trajectories_3d_N*.h5"]:
        matches = sorted(data_dir_.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"在 {data_dir} 中找不到 N≈{n_nodes} 的 3D HDF5 文件。"
        f"请先运行 generate_laser_data_3d.py --n_nodes {n_nodes}"
    )


# ─────────────────────────────────────────────────────────────────
# 主数据集类
# ─────────────────────────────────────────────────────────────────

class LaserHardening3DDataset(Dataset):
    """3D 激光淬火 PDE 轨迹数据集。

    参数
    ----
    h5_path      : HDF5 文件路径
    n_time_steps : 每个样本使用的时间步数（从第 0 步起）
    device       : 张量设备（通常 'cpu'，DataLoader 后再转 GPU）
    traj_indices : 指定使用哪些轨迹（None = 全部）
    """

    def __init__(
        self,
        h5_path:      str,
        n_time_steps: int = 20,
        device:       str = 'cpu',
        traj_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.h5_path      = Path(h5_path)
        self.n_time_steps = n_time_steps
        self.device       = device

        # ── 读取全局网格元数据 ─────────────────────────────────
        with h5py.File(self.h5_path, 'r') as f:
            meta = f['mesh_meta']

            self.nodes        = torch.tensor(meta['nodes'][:],        dtype=torch.float32)
            self.edges        = torch.tensor(meta['edges'][:],        dtype=torch.long)
            self.faces        = torch.tensor(meta['faces'][:],        dtype=torch.long)
            self.tets         = torch.tensor(meta['tets'][:],         dtype=torch.long)
            self.node_volumes = torch.tensor(meta['node_volumes'][:], dtype=torch.float32)
            self.time_points  = torch.tensor(meta['time_points'][:],  dtype=torch.float32)

            # 稀疏 Laplacian（预存，避免每次重建 FEM）
            L_ei = torch.tensor(meta['L_physics_ei'][:], dtype=torch.long)
            L_ew = torch.tensor(meta['L_physics_ew'][:], dtype=torch.float32)
            self.L_physics = {
                "type"         : "sparse",
                "edge_index"   : L_ei,
                "edge_weights" : L_ew,
                "node_volumes" : self.node_volumes,
            }

            # 节点类型（0=内部, 1=边界）
            N = self.nodes.shape[0]
            bnd_set = set(self.faces.numpy().ravel().tolist()) if len(self.faces) > 0 else set()
            nt = torch.zeros(N, dtype=torch.long)
            if bnd_set:
                nt[list(bnd_set)] = 1
            self.node_type = nt

            # 轨迹列表
            all_traj_keys = sorted([k for k in f.keys() if k.startswith('trajectory_')],
                                    key=lambda x: int(x.split('_')[1]))
            self.traj_keys = (
                [all_traj_keys[i] for i in traj_indices] if traj_indices is not None
                else all_traj_keys
            )

        self.N = self.nodes.shape[0]
        T_avail = len(self.time_points)
        self.n_time_steps = min(n_time_steps, T_avail)

    # ── 统计信息 ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.traj_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.traj_keys[idx]
        T   = self.n_time_steps

        with h5py.File(self.h5_path, 'r') as f:
            g    = f[key]
            u    = torch.tensor(g['node_features'][:T],     dtype=torch.float32)   # (T, N, 1)
            src  = torch.tensor(g['source_terms'][:T],      dtype=torch.float32)   # (T, N, 1)
            u0   = torch.tensor(g['initial_condition'][:1], dtype=torch.float32)   # (1, N, 1)

            # Dirichlet BC
            bnd_g = g['boundary_info']['dirichlet']
            bnd_idx = torch.tensor(bnd_g['indices'][:], dtype=torch.long)
            bnd_val = torch.tensor(bnd_g['values'][:],  dtype=torch.float32)

        batch = {
            # 网格（每个样本均相同，但 DataLoader 需要）
            "nodes"             : self.nodes,             # (N, 3)
            "edges"             : self.edges,             # (E, 2)
            "faces"             : self.faces,             # (BF, 3)
            "tets"              : self.tets,              # (F, 4)
            "node_volumes"      : self.node_volumes,      # (N,)
            "node_type"         : self.node_type,         # (N,)
            # 物理算子（稀疏 Laplacian）
            "L_physics"         : self.L_physics,
            # 轨迹数据
            "initial_conditions": u[0:1].squeeze(0),      # (N, 1)
            "source_terms"      : src.unsqueeze(0),       # (1, T, N, 1)  → batch 维度由 collate 处理
            "time_points"       : self.time_points[:T],   # (T,)
            "targets"           : u,                      # (T, N, 1)
            # 边界
            "boundary_info"     : {
                "dirichlet": {"indices": bnd_idx, "values": bnd_val}
            },
            # 用于调试：返回轨迹键，以便训练时定位出问题的样本
            "traj_key"          : key,
        }
        return batch

    def info(self) -> str:
        return (
            f"LaserHardening3DDataset | 文件: {self.h5_path.name}\n"
            f"  节点 N={self.N} | 四面体={len(self.tets)} | 边={len(self.edges)}\n"
            f"  轨迹数={len(self.traj_keys)} | 时间步={self.n_time_steps}\n"
            f"  空间维度=3 | 特征维度=1"
        )


# ─────────────────────────────────────────────────────────────────
# Collate：将 list[dict] 整理成 batch dict
# ─────────────────────────────────────────────────────────────────

def collate_fn_3d(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将多个样本的轨迹数据组合成 batch。

    网格数据（nodes/edges/tets/...）在同一文件内完全相同，
    直接取第一个样本的值即可（不做 stack）。

    轨迹数据（source_terms/targets/initial_conditions）沿 batch 维度 stack。
    """
    B = len(samples)
    s0 = samples[0]

    batch = {
        # 网格（共享）
        "nodes"       : s0["nodes"],
        "edges"       : s0["edges"],
        "faces"       : s0["faces"],
        "tets"        : s0["tets"],
        "node_volumes": s0["node_volumes"],
        "node_type"   : s0["node_type"],
        "L_physics"   : s0["L_physics"],
        "time_points" : s0["time_points"],
        "boundary_info": s0["boundary_info"],
        # 轨迹（stack 成 batch）
        # initial_conditions: (B, N, 1)
        "initial_conditions": torch.stack([s["initial_conditions"] for s in samples], dim=0),
        # source_terms: (B, T, N, 1)
        "source_terms": torch.stack([s["source_terms"].squeeze(0) for s in samples], dim=0),
        # targets: (B, T, N, 1)
        "targets"     : torch.stack([s["targets"] for s in samples], dim=0),
        # 轨迹键列表（用于调试）
        "traj_keys"   : [s["traj_key"] for s in samples],
    }
    return batch


# ─────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────

def build_dataloaders_3d(
    data_dir:       str,
    n_nodes:        int,
    batch_size:     int   = 4,
    n_time_steps:   int   = 20,
    train_ratio:    float = 0.8,
    num_workers:    int   = 0,
    seed:           int   = 42,
) -> Tuple[DataLoader, DataLoader]:
    """构建 3D 训练集 / 验证集 DataLoader。

    返回
    ----
    train_loader, val_loader
    """
    h5_path = find_h5_file(data_dir, n_nodes)
    ds_full = LaserHardening3DDataset(h5_path, n_time_steps=n_time_steps)

    n_total = len(ds_full)
    n_train = int(n_total * train_ratio)
    n_val   = n_total - n_train

    rng   = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        ds_full, [n_train, n_val], generator=rng
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_fn_3d,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_fn_3d,
    )

    print(ds_full.info())
    print(f"  训练样本={n_train}, 验证样本={n_val}, batch_size={batch_size}")
    return train_loader, val_loader


def build_single_dataloader_3d(
    h5_path:      str,
    batch_size:   int = 4,
    n_time_steps: int = 20,
    shuffle:      bool = False,
    num_workers:  int  = 0,
) -> DataLoader:
    """直接从 h5_path 构建单个 DataLoader（用于评测/对比）。"""
    ds = LaserHardening3DDataset(h5_path, n_time_steps=n_time_steps)
    print(ds.info())
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_fn_3d,
    )


# ─────────────────────────────────────────────────────────────────
# 快速自测
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    h5 = sys.argv[1] if len(sys.argv) > 1 else "data_laser_hardening_3d/pde_trajectories_3d_N5000.h5"
    loader = build_single_dataloader_3d(h5, batch_size=2, n_time_steps=10)
    batch  = next(iter(loader))
    print("Batch keys:", list(batch.keys()))
    print("nodes     :", batch["nodes"].shape,         batch["nodes"].dtype)
    print("tets      :", batch["tets"].shape,          batch["tets"].dtype)
    print("source_terms:", batch["source_terms"].shape)
    print("targets   :", batch["targets"].shape)
    print("L_physics edge_index:", batch["L_physics"]["edge_index"].shape)
    print("自测通过！")
