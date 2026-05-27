"""
phys_hgnet_3d.py — PhysHGNet 3D 扩展
======================================

在 PhysHGNet（2D）基础上进行 3D 适配：

与 2D 版本的关键差异
--------------------
维度       2D              3D
-------    ----------      ------------------
空间维     spatial_dim=2   spatial_dim=3
网格单元   三角形 faces(F,3) 四面体 tets(F,4)
节点体积   面积 / 3        tet 体积 / 4（体积加权）
物理算子   cotangent L     FEM 刚度矩阵（build_operator_3d）
物理梯度   2D 面积加权     3D 体积加权 FEM 形函数梯度
边特征     (E, 3)=[Δx,Δy,d]  (E, 4)=[Δx,Δy,Δz,d]
编码器输入  spatial_dim+1+3=6  spatial_dim+1+3=7

三大创新 C1/C2/C3 在 3D 下的适配
---------------------------------
C1 - PhysicsAwareAnchorSelector：
    代价函数 C_i = α·几何分散 + β·PDE 残差 + γ·物理梯度
    空间距离改为三维 Euclidean，残差 ||∇T||₃D 由 physics_3d.compute_pde_residual_3d 提供。
    模块代码无需修改（PhysicsAwareAnchorSelector 已用距离计算，维度无关）。

C2 - LearnableCoarseOperator：
    MLP 输入 = [xᵢ(3), xⱼ(3), hᵢ(hd), hⱼ(hd), dist(1)] = 6+2hd+1
    构造时传入 spatial_dim=3 即自动适配。

C3 - DualScaleGNNCorrector：
    fine_enc  输入 = u(1) + nodes(3) + node_type(3) = 7D
    coarse_enc 输入 = u_c(1) + anchor_coords(3) = 4D
    _MPNN edge_dim  = spatial_dim+1 = 4
    构造时传入 spatial_dim=3 即自动适配。

实现方式：继承 PhysHGNet，覆盖 forward（注入 3D 物理算子）。
PhysHGNet 的 _build_graph / _Leff_matvec / C1/C2/C3 不修改。

使用方法
--------
from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D

model = PhysHGNet3D(DEFAULT_CONFIG_3D)
out   = model(batch)   # batch 含 "tets" 而非 "faces"
"""

from __future__ import annotations
import math
from copy import deepcopy
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 2D 基类 & 工具 ────────────────────────────────────────────────
from phys_hgnet import PhysHGNet, DEFAULT_CONFIG as _DEFAULT_CONFIG_2D
from physics_3d import build_operator_3d, compute_node_volumes_3d

# ─────────────────────────────────────────────────────────────────
# 3D 默认配置（覆盖 2D 版本中与维度相关的参数）
# ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_3D: Dict[str, Any] = {
    **deepcopy(_DEFAULT_CONFIG_2D),
    # ── 核心维度变更 ──────────────────────────────────────────
    "spatial_dim"           : 3,        # 3D 坐标
    "feature_dim"           : 1,        # 温度标量场
    "output_dim"            : 1,
    # ── 适当增大容量以应对 3D 复杂度 ──────────────────────────
    "operator_hidden_dim"   : 64,
    "operator_num_layers"   : 3,
    "residual_hidden_dim"   : 128,
    "residual_num_layers"   : 5,
    "coarse_num_layers"     : 4,
    "k_virtual_nodes"       : 4,
    # ── 锚点参数（3D 场景适当增多）───────────────────────────
    "m_anchors"             : 64,
    "q_local"               : 6,        # 每个锚点关联近邻数（3D 略多）
    "k_coarse"              : 8,        # 粗网格近邻数（3D 更稠密）
    # ── CG 参数（3D 条件数更大，适当增加迭代）────────────────
    "cg_max_iter"           : 80,
    "cg_tol"                : 1e-6,
    # ── 消融开关（默认全开）──────────────────────────────────
    "use_physics_anchor"    : True,
    "use_learned_coarse"    : True,
    "use_dual_scale_gnn"    : True,
    "use_virtual_nodes"     : True,
    "operator_type"         : "laplace",
    "residual_update_freq"  : 5,
    "use_checkpoint"        : False,
}


# ─────────────────────────────────────────────────────────────────
# PhysHGNet3D
# ─────────────────────────────────────────────────────────────────

class PhysHGNet3D(PhysHGNet):
    """Physics-aware Hierarchical Graph Neural Operator — 3D 版本。

    继承 PhysHGNet（2D），覆盖 forward 中与物理算子构建相关的逻辑，
    使用 physics_3d.build_operator_3d（FEM 四面体刚度矩阵）替代
    cotangent Laplacian。

    三大创新模块（C1/C2/C3）无需修改，只需在 __init__ 时传入 spatial_dim=3
    使所有 MLP/MPNN 维度自动适配。

    ─── batch 字典期望格式 ────────────────────────────────────────
    nodes            : (N, 3)
    edges            : (E, 2)
    tets             : (F, 4)   ← 新增（替代 faces(F,3)）
    node_volumes     : (N,)     ← 新增（可选，若缺则内部从 tets 计算）
    L_physics        : dict     ← 可选，缺省则内部从 tets 构建
    initial_conditions : (B, N, 1)
    source_terms     : (B, T, N, 1)
    time_points      : (T,)
    node_type        : (N,)
    boundary_info    : dict
    """

    def __init__(self, config: Dict[str, Any]):
        # 强制 spatial_dim=3
        cfg3d = {**DEFAULT_CONFIG_3D, **config, "spatial_dim": 3}
        super().__init__(cfg3d)

    # ─────────────────────────────────────────────────────────────
    # 覆盖 forward：在调用 super().forward 之前注入 3D 物理算子
    # ─────────────────────────────────────────────────────────────

    def forward(
        self,
        batch: Dict[str, Any],
        use_physics_anchor: Optional[bool] = None,
        use_learned_coarse: Optional[bool] = None,
        use_dual_scale_gnn: Optional[bool] = None,
        use_virtual_nodes:  Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播（3D 版本）。

        主要逻辑与 2D 版本相同，差异在于：
        1. 从 batch["tets"] 获取四面体连接
        2. 若 batch 中无 L_physics，调用 build_operator_3d 构建
        3. 若 batch 中无 node_volumes，从 tets 计算
        """
        nodes  = batch["nodes"]
        device = nodes.device

        # ── 确保 L_physics 存在（3D FEM 版本）──────────────────
        if "L_physics" not in batch or batch["L_physics"] is None:
            tets = batch.get("tets")
            if tets is None:
                raise ValueError(
                    "PhysHGNet3D.forward 需要 batch['tets'] (四面体连接关系) "
                    "或 batch['L_physics'] (预构建稀疏算子)。"
                )
            batch = dict(batch)   # 浅拷贝，避免修改原始 dict
            batch["L_physics"] = build_operator_3d(
                nodes.float(),
                tets.long(),
            )

        # ── 确保 node_volumes 存在 ───────────────────────────────
        if "node_volumes" not in batch or batch["node_volumes"] is None:
            tets = batch.get("tets")
            if tets is not None:
                if not isinstance(batch, dict):
                    batch = dict(batch)
                batch["node_volumes"] = compute_node_volumes_3d(
                    nodes.float(), tets.long()
                )

        # ── 更新图缓存 key（包含 3D 标志，防止与 2D 缓存冲突）──
        # cache_key 在 super().forward 中由 (N, device, use_physics_anchor) 确定；
        # 3D 与 2D 节点数通常不同，故自然隔离。无需额外处理。

        return super().forward(
            batch,
            use_physics_anchor = use_physics_anchor,
            use_learned_coarse = use_learned_coarse,
            use_dual_scale_gnn = use_dual_scale_gnn,
            use_virtual_nodes  = use_virtual_nodes,
        )

    def extra_info(self) -> str:
        return (
            "PhysHGNet3D | spatial_dim=3 | 四面体 FEM Laplacian\n" +
            self.ablation_summary()
        )


# ─────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────

def build_phys_hgnet_3d(
    m_anchors:        int  = 64,
    residual_hidden:  int  = 128,
    residual_layers:  int  = 5,
    use_physics_anchor: bool = True,
    use_learned_coarse: bool = True,
    use_dual_scale_gnn: bool = True,
    use_virtual_nodes:  bool = True,
    **extra_kwargs,
) -> PhysHGNet3D:
    """便捷工厂函数，构建 PhysHGNet3D。"""
    cfg = {
        **DEFAULT_CONFIG_3D,
        "m_anchors"           : m_anchors,
        "residual_hidden_dim" : residual_hidden,
        "residual_num_layers" : residual_layers,
        "use_physics_anchor"  : use_physics_anchor,
        "use_learned_coarse"  : use_learned_coarse,
        "use_dual_scale_gnn"  : use_dual_scale_gnn,
        "use_virtual_nodes"   : use_virtual_nodes,
        **extra_kwargs,
    }
    return PhysHGNet3D(cfg)


# ─────────────────────────────────────────────────────────────────
# 快速自测
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from scipy.spatial import Delaunay

    print("=== PhysHGNet3D 快速自测 ===")

    # 生成一个小型 3D 网格
    N_test = 200
    rng = np.random.default_rng(42)
    pts = rng.uniform([0, 0, 0], [0.5, 0.3, 0.05], size=(N_test, 3))
    tri = Delaunay(pts)
    tets_np = tri.simplices.astype(np.int64)

    # 边
    edge_set = set()
    for tet in tets_np:
        for i in range(4):
            for j in range(i+1, 4):
                edge_set.add((min(tet[i], tet[j]), max(tet[i], tet[j])))
    edges_np = np.array(sorted(edge_set), dtype=np.int64)

    T_steps, B = 5, 2
    nodes  = torch.tensor(pts,       dtype=torch.float32)
    tets   = torch.tensor(tets_np,   dtype=torch.long)
    edges  = torch.tensor(edges_np,  dtype=torch.long)

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

    model = build_phys_hgnet_3d(m_anchors=16)
    print(model.extra_info())
    print(f"参数量: {model.num_parameters():,}")

    with torch.no_grad():
        out = model(batch)

    print(f"输出 u_final 形状: {out['u_final'].shape}")
    assert out['u_final'].shape == (B, T_steps, N_test, 1), "形状错误！"
    print("✓ 自测通过！")
