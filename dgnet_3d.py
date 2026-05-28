"""
dgnet_3d.py — DGNet3D：与 DGNEt 仓库 StructuredDGNet 严格对齐的 3D 基线

对齐原则
--------
本文件不再继承 PhysHGNet3D，而是直接包装 StructuredDGNet，
仅在 forward() 里注入 3D FEM Laplacian（来自 physics_3d），
确保模型架构与 DGNet 仓库完全一致。

DGNet3D 与 PhysHGNet3D 的唯一合法差距来自：
  C1 物理感知锚点选取、C2 可学习粗算子、C3 双尺度 GNN 修正。

架构对齐要点（对应 DGNEt/structured_dgnet.py）
-----------------------------------------------
1. 算子公式：L = L_phys + α_loc·S_θ + α_coarse·P·C·R
             L_phys 永远全强度，S_θ 由 LocalCorrectionHead 学习
2. Alpha 归一化：a_loc = L_scale × softplus(raw)
3. NonlinearDynamicsSolver：r(u^k) 加入 RHS
4. ResidualSolver：数据路径 u_net
5. 训练损失：仅在 t=1 和 t=T-1 计算（与 StructuredDGNetLoss 对齐）
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from structured_dgnet import StructuredDGNet

# ─────────────────────────────────────────────────────────────────
# 默认配置（与 DGNet 仓库对齐，spatial_dim=3）
# ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_DGNET3D: Dict[str, Any] = {
    "spatial_dim"          : 3,    # 3D
    "feature_dim"          : 1,
    "output_dim"           : 1,
    "m_anchors"            : 64,   # 与 DGNet 仓库默认一致
    "q_local"              : 4,
    "tau"                  : 0.1,
    "k_coarse"             : 6,
    "cg_max_iter"          : 50,
    "cg_tol"               : 1e-6,
    "operator_hidden_dim"  : 64,   # FineGraphEncoder / LocalCorrectionHead
    "operator_num_layers"  : 3,
    "residual_hidden_dim"  : 128,  # NonlinearDynamicsSolver + ResidualSolver
    "residual_num_layers"  : 5,
    "coarse_num_layers"    : 2,
    "loss_last_only"       : False,
    "use_checkpoint"       : False,
}


# ─────────────────────────────────────────────────────────────────
# DGNet3D
# ─────────────────────────────────────────────────────────────────

class DGNet3D(nn.Module):
    """
    DGNet 的 3D 版本。

    直接包装 StructuredDGNet（与 DGNEt 仓库完全对齐），
    在 forward() 里自动注入 3D FEM Laplacian（来自 physics_3d）。

    batch 格式与 PhysHGNet3D.forward 完全一致：
      nodes, edges, tets, initial_conditions, source_terms,
      time_points, node_type, boundary_info
    可选：L_physics（若已预计算则直接使用，否则在线计算）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG_DGNET3D, **(config or {})}
        # 强制 3D
        cfg["spatial_dim"] = 3
        self.cfg = cfg
        self._net = StructuredDGNet(cfg)

    # ── forward ──────────────────────────────────────────────────

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        batch = self._inject_physics(batch)
        return self._net(batch)

    def _inject_physics(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """若 batch 中无 L_physics / node_volumes，则在线计算。"""
        need_L = (batch.get("L_physics") is None)
        need_V = (batch.get("node_volumes") is None)

        if not (need_L or need_V):
            return batch

        batch = dict(batch)   # shallow copy，不修改原始 batch
        nodes  = batch["nodes"]
        edges  = batch["edges"]
        tets   = batch.get("tets", None)
        device = nodes.device
        N      = nodes.shape[0]

        if need_V:
            # 均匀节点体积（简单 fallback）
            batch["node_volumes"] = torch.ones(N, device=device,
                                               dtype=nodes.dtype)

        if need_L:
            try:
                from physics_3d import build_fem_laplacian_3d
                L = build_fem_laplacian_3d(nodes, tets)
                # 转为 StructuredDGNet 需要的 sparse dict 格式
                if isinstance(L, torch.Tensor) and L.dim() == 2:
                    sp = L.to_sparse()
                    idx = sp.indices()
                    vals = sp.values()
                    L_scale = vals.abs().max().clamp(min=1.0)
                    batch["L_physics"] = {
                        "type"         : "sparse",
                        "edge_index"   : idx,
                        "edge_weights" : vals,
                        "N"            : N,
                        "diag"         : L.diag(),
                        "L_scale"      : float(L_scale),
                    }
                elif isinstance(L, dict):
                    batch["L_physics"] = L
                else:
                    raise ValueError("unexpected L type")
            except Exception as e:
                # Fallback: 零矩阵（训练会依赖纯数据路径）
                print(f"[DGNet3D] L_physics 计算失败 ({e})，使用零矩阵")
                ei = edges.T  # (2, E)
                fwd = torch.cat([ei, ei.flip(0)], dim=1)
                batch["L_physics"] = {
                    "type"         : "sparse",
                    "edge_index"   : fwd,
                    "edge_weights" : torch.zeros(fwd.shape[1], device=device),
                    "N"            : N,
                    "diag"         : torch.zeros(N, device=device),
                    "L_scale"      : 1.0,
                }

        return batch

    # ── 便捷属性 ─────────────────────────────────────────────────

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_info(self) -> str:
        return (
            f"DGNet3D | spatial_dim=3 | StructuredDGNet backbone\n"
            f"  算子: L = L_phys + α_loc·S_θ + α_coarse·P·C·R\n"
            f"  NonlinearDynamicsSolver: ON (r(u^k) in RHS)\n"
            f"  ResidualSolver (data path): ON\n"
            f"  隐层: {self.cfg['residual_hidden_dim']} × "
            f"{self.cfg['residual_num_layers']} layers\n"
            f"  参数量: {self.num_parameters():,}"
        )


# ─────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────

def build_dgnet_3d(
    residual_hidden: int = 128,
    residual_layers: int = 5,
    m_anchors: int = 64,
    **extra_kwargs,
) -> DGNet3D:
    cfg = {
        **DEFAULT_CONFIG_DGNET3D,
        "residual_hidden_dim": residual_hidden,
        "residual_num_layers": residual_layers,
        "m_anchors"          : m_anchors,
        **extra_kwargs,
    }
    return DGNet3D(cfg)


# ─────────────────────────────────────────────────────────────────
# 快速自测
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from scipy.spatial import Delaunay

    print("=== DGNet3D 快速自测 ===")
    N_test = 200
    rng  = np.random.default_rng(42)
    pts  = rng.uniform([0, 0, 0], [0.5, 0.3, 0.05], size=(N_test, 3))
    tri  = Delaunay(pts)
    tets_np = tri.simplices.astype(np.int64)
    edge_set = set()
    for tet in tets_np:
        for i in range(4):
            for j in range(i+1, 4):
                edge_set.add((min(tet[i], tet[j]), max(tet[i], tet[j])))
    edges_np = np.array(sorted(edge_set), dtype=np.int64)

    T_steps, B = 5, 2
    nodes = torch.tensor(pts, dtype=torch.float32)
    tets  = torch.tensor(tets_np, dtype=torch.long)
    edges = torch.tensor(edges_np, dtype=torch.long)
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
    print("✓ 自测通过！")
