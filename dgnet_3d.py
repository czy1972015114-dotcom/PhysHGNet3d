"""
dgnet_3d.py — DGNet 3D 基线模型
=================================

DGNet3D = PhysHGNet3D + 关闭全部三项创新（C1/C2/C3）。

设计原则（与 2D 对照实验一致）
---------------------------------
2D 中，DGNet 是 PhysHGNet 的基线。两者差异仅在于：
    C1: 物理感知锚点选取  → False（退化为普通 FPS）
    C2: 可学习粗算子      → False（退化为距离倒数固定权重）
    C3: 双尺度 GNN 修正   → False（退化为单尺度细网格 MPNN）
    虚拟节点              → False

这样对比时，所有超参数（hidden_dim、层数等）相同，
结果差异 100% 来自三项创新，保证控制变量严格对等。

与 DGNet 仓库（structured_dgnet.py）的关系
-------------------------------------------
本模块并不直接调用 structured_dgnet.StructuredDGNet，
而是使用 PhysHGNet3D 的消融开关路径，确保：
  ① 使用相同的 3D FEM 物理算子（physics_3d）
  ② 使用相同的 3D 数据集接口（dataset_3d）
  ③ 相同的 Crank-Nicolson 时间推进 + PCG 求解器
  ④ 唯一差异：无 C1/C2/C3

接口
----
model = DGNet3D(config_override={})
out   = model(batch)    # batch 格式与 PhysHGNet3D.forward 完全一致
out["u_final"]          # (B, T, N, 1)

检查点路径：checkpoints/dgnet_3d/best_{N}.pth
"""

from __future__ import annotations
from copy import deepcopy
from typing import Dict, Any, Optional

from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D

# ─────────────────────────────────────────────────────────────────
# DGNet3D 默认配置（关闭所有创新；其余与 PhysHGNet3D 完全相同）
# ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_DGNET3D: Dict[str, Any] = {
    **deepcopy(DEFAULT_CONFIG_3D),
    # ── 关闭三项创新（这是 DGNet3D 与 PhysHGNet3D 的唯一差异）──
    "use_physics_anchor" : False,   # C1 OFF → 普通 FPS
    "use_learned_coarse" : False,   # C2 OFF → 距离倒数固定边权
    "use_dual_scale_gnn" : False,   # C3 OFF → 仅细网格 MPNN
    "use_virtual_nodes"  : False,   # 虚拟节点 OFF
    # ── 参数量对齐（为公平对比保持与 PhysHGNet3D 相同容量）────
    "residual_hidden_dim" : 128,
    "residual_num_layers" : 5,
    "operator_hidden_dim" : 64,
    "operator_num_layers" : 3,
}


# ─────────────────────────────────────────────────────────────────
# DGNet3D
# ─────────────────────────────────────────────────────────────────

class DGNet3D(PhysHGNet3D):
    """DGNet 3D 基线（Structured DGNet 的 3D 版本）。

    继承 PhysHGNet3D，强制关闭所有三项创新，
    保留相同的 CG 求解器、FEM 物理算子、时间推进逻辑。

    batch 格式与 PhysHGNet3D.forward 完全一致。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {
            **DEFAULT_CONFIG_DGNET3D,
            **(config or {}),
            # 强制关闭创新（不允许外部 config 覆盖）
            "use_physics_anchor": False,
            "use_learned_coarse": False,
            "use_dual_scale_gnn": False,
            "use_virtual_nodes" : False,
        }
        # 注意：PhysHGNet3D.__init__ 会再次强制 spatial_dim=3
        super().__init__(cfg)

    def extra_info(self) -> str:
        return (
            "DGNet3D | spatial_dim=3 | 四面体 FEM Laplacian\n"
            "  C1: OFF (普通 FPS)\n"
            "  C2: OFF (距离倒数固定边权)\n"
            "  C3: OFF (单尺度细网格 MPNN)\n"
            f"  参数量: {self.num_parameters():,}"
        )


# ─────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────

def build_dgnet_3d(
    residual_hidden: int = 128,
    residual_layers: int = 5,
    **extra_kwargs,
) -> DGNet3D:
    """便捷工厂函数，构建 DGNet3D。"""
    cfg = {
        **DEFAULT_CONFIG_DGNET3D,
        "residual_hidden_dim" : residual_hidden,
        "residual_num_layers" : residual_layers,
        **extra_kwargs,
    }
    return DGNet3D(cfg)


# ─────────────────────────────────────────────────────────────────
# 快速自测
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import torch
    from scipy.spatial import Delaunay

    print("=== DGNet3D 快速自测 ===")

    N_test = 200
    rng = np.random.default_rng(42)
    pts = rng.uniform([0, 0, 0], [0.5, 0.3, 0.05], size=(N_test, 3))
    tri = Delaunay(pts)
    tets_np = tri.simplices.astype(np.int64)

    edge_set = set()
    for tet in tets_np:
        for i in range(4):
            for j in range(i+1, 4):
                edge_set.add((min(tet[i], tet[j]), max(tet[i], tet[j])))
    edges_np = np.array(sorted(edge_set), dtype=np.int64)

    T_steps, B = 5, 2
    nodes  = torch.tensor(pts,      dtype=torch.float32)
    tets   = torch.tensor(tets_np,  dtype=torch.long)
    edges  = torch.tensor(edges_np, dtype=torch.long)

    batch = {
        "nodes"             : nodes,
        "edges"             : edges,
        "tets"              : tets,
        "initial_conditions": torch.full((B, N_test, 1), 298.15),
        "source_terms"      : torch.zeros(B, T_steps, N_test, 1),
        "time_points"       : torch.linspace(0, 2, T_steps),
        "node_type"         : torch.zeros(N_test, dtype=torch.long),
        "boundary_info"     : {},
    }

    model = build_dgnet_3d()
    print(model.extra_info())

    with torch.no_grad():
        out = model(batch)

    print(f"输出 u_final 形状: {out['u_final'].shape}")
    assert out['u_final'].shape == (B, T_steps, N_test, 1)
    print("✓ 自测通过！")
