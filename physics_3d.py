"""
physics_3d.py — PhysHGNet 的 3D FEM 物理算子模块
================================================

替换 2D 的 cotangent Laplacian（三角网格面积权重），
改用 3D 线性四面体 FEM 刚度矩阵（体积权重）。

公式
----
单元刚度矩阵（线性四面体，形函数梯度在单元内为常数）：
    K_e^{ij} = k_th · V_e · (∇φ_i · ∇φ_j)

质量归一化离散扩散算子：
    (L_h u)_i = diffCoeff · Σ_{j≠i} [K_{ij} / M_i] · (u_j - u_i)

其中：
    diffCoeff = k_th / (ρ·cp)
    M_i       = ρ·cp · Σ_{e⊃i} V_e/4

接口
----
build_operator_3d(nodes, tets, ...) -> dict
    返回 PhysHGNet._build_graph 期望的 L_physics 字典：
    {
        "type"         : "sparse",
        "edge_index"   : LongTensor (2, 2·E_off),
        "edge_weights" : FloatTensor (2·E_off,),
        "node_volumes" : FloatTensor (N,)    # 纯几何体积，不含 ρ·cp
    }

compute_node_volumes_3d(nodes, tets) -> FloatTensor (N,)
    V_i = Σ_{e⊃i} V_e / 4

compute_tet_volumes(nodes, tets) -> FloatTensor (F,)
    V_e = |det J_e| / 6

注意：本模块在 CPU 上进行 scipy sparse 运算，返回 torch.Tensor。
"""

import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

# ─── 物理常数（与 generate_laser_data_3d.py 一致）────────────────
K_TH      = 50.0        # W/(m·K)
RHO       = 7850.0      # kg/m³
CP        = 450.0       # J/(kg·K)
DIFF_COEFF = K_TH / (RHO * CP)
MIN_TET_VOL = 1e-18     # 退化四面体阈值

# ─────────────────────────────────────────────────────────────────
# 底层：numpy/scipy FEM 组装
# ─────────────────────────────────────────────────────────────────

def _assemble_tet_stiffness_np(coords_np: np.ndarray,
                                tets_np: np.ndarray) -> sp.csr_matrix:
    """组装 3D FEM 刚度矩阵 K（scipy CSR）。

    K[i,j] = Σ_{e⊃{i,j}} k_th · V_e · (∇φ_i · ∇φ_j)

    仅用于 N ≤ 50000 的情形；更大规模可用 chunked 版本。
    """
    N     = coords_np.shape[0]
    n_tet = tets_np.shape[0]

    x0 = coords_np[tets_np[:, 0]]
    x1 = coords_np[tets_np[:, 1]]
    x2 = coords_np[tets_np[:, 2]]
    x3 = coords_np[tets_np[:, 3]]

    J    = np.stack([x1 - x0, x2 - x0, x3 - x0], axis=1)   # (F, 3, 3)
    detJ = np.linalg.det(J)                                   # (F,)
    vol  = np.abs(detJ) / 6.0                                 # (F,)

    # 过滤退化单元
    valid = vol > MIN_TET_VOL
    if not np.all(valid):
        n_bad = int(np.sum(~valid))
        warnings.warn(f"[physics_3d] 发现 {n_bad} 个退化四面体，已忽略其刚度贡献。")
        vol  = np.where(valid, vol, 0.0)

    # J 的逆转置（形函数梯度变换）
    J_inv  = np.zeros_like(J)
    J_inv[valid] = np.linalg.inv(J[valid])
    J_invT = np.transpose(J_inv, (0, 2, 1))

    # 形函数梯度（(F, 4, 3) 数组，常数每单元）
    grad = np.zeros((n_tet, 4, 3))
    grad[:, 1, :] = J_invT[:, :, 0]
    grad[:, 2, :] = J_invT[:, :, 1]
    grad[:, 3, :] = J_invT[:, :, 2]
    grad[:, 0, :] = -(grad[:, 1, :] + grad[:, 2, :] + grad[:, 3, :])

    rows_l, cols_l, vals_l = [], [], []
    for i in range(4):
        for j in range(4):
            dot_ij = np.einsum('ed,ed->e', grad[:, i, :], grad[:, j, :])
            rows_l.append(tets_np[:, i])
            cols_l.append(tets_np[:, j])
            vals_l.append(K_TH * vol * dot_ij)

    K_csr = sp.csr_matrix(
        (np.concatenate(vals_l),
         (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(N, N)
    )
    K_csr.sum_duplicates()
    K_csr.eliminate_zeros()
    return K_csr


def _compute_node_volumes_np(coords_np: np.ndarray,
                              tets_np: np.ndarray) -> np.ndarray:
    """节点控制体积（纯几何，不含 ρ·cp）：V_i = Σ_{e⊃i} V_e / 4。"""
    N = coords_np.shape[0]
    x0, x1, x2, x3 = (coords_np[tets_np[:, k]] for k in range(4))
    J   = np.stack([x1 - x0, x2 - x0, x3 - x0], axis=1)
    vol = np.abs(np.linalg.det(J)) / 6.0
    nv  = np.zeros(N)
    for k in range(4):
        np.add.at(nv, tets_np[:, k], vol / 4.0)
    return np.maximum(nv, 1e-14)


# ─────────────────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────────────────

def compute_node_volumes_3d(nodes: torch.Tensor,
                             tets: torch.Tensor) -> torch.Tensor:
    """计算节点控制体积 V_i [m³]，返回 (N,) FloatTensor。

    V_i = Σ_{e⊃i} |det J_e| / (6 × 4)
    """
    coords_np = nodes.detach().cpu().numpy().astype(np.float64)
    tets_np   = tets.detach().cpu().numpy().astype(np.int32)
    nv_np     = _compute_node_volumes_np(coords_np, tets_np)
    return torch.tensor(nv_np, dtype=torch.float32, device=nodes.device)


def compute_tet_volumes(nodes: torch.Tensor,
                        tets: torch.Tensor) -> torch.Tensor:
    """计算每个四面体体积，返回 (F,) FloatTensor。"""
    x0 = nodes[tets[:, 0]]
    x1 = nodes[tets[:, 1]]
    x2 = nodes[tets[:, 2]]
    x3 = nodes[tets[:, 3]]
    # J = [x1-x0, x2-x0, x3-x0]，形状 (F, 3, 3)
    J   = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=1)
    vol = torch.linalg.det(J).abs() / 6.0
    return vol


def build_operator_3d(
    nodes: torch.Tensor,
    tets:  torch.Tensor,
    node_volumes: Optional[torch.Tensor] = None,
    operator_type: str = 'laplace',
) -> Dict[str, Any]:
    """从四面体网格构建 3D FEM 扩散算子。

    参数
    ----
    nodes        : (N, 3) 节点坐标
    tets         : (F, 4) 四面体连接关系（int long）
    node_volumes : (N,) 预计算节点体积（可选，若 None 则内部计算）
    operator_type: 目前仅支持 'laplace'

    返回
    ----
    dict 含以下键：
        "type"         : "sparse"
        "edge_index"   : LongTensor (2, 2·E_off)  有向边（正反各一）
        "edge_weights" : FloatTensor (2·E_off,)    w_{ij} = diffCoeff·K_{ij}/M_i
        "node_volumes" : FloatTensor (N,)           纯几何节点体积

    语义：(L_h u)_i = Σ_{j: (i,j)∈E} w_{ij} · (u_j - u_i)
    """
    if operator_type != 'laplace':
        raise NotImplementedError(f"3D operator_type='{operator_type}' 尚未实现。")

    device    = nodes.device
    coords_np = nodes.detach().cpu().numpy().astype(np.float64)
    tets_np   = tets.detach().cpu().numpy().astype(np.int32)

    # ── FEM 组装（numpy/scipy）─────────────────────────────────
    K_sp   = _assemble_tet_stiffness_np(coords_np, tets_np)
    nv_np  = _compute_node_volumes_np(coords_np, tets_np)
    M_diag = RHO * CP * nv_np        # ρ·cp·V_i

    # ── 提取非对角元，构建质量归一化权重 ─────────────────────
    K_coo = K_sp.tocoo()
    mask  = K_coo.row != K_coo.col
    rows  = K_coo.row[mask].astype(np.int64)
    cols  = K_coo.col[mask].astype(np.int64)
    kvals = K_coo.data[mask]

    # w_{ij} = diffCoeff · K_{ij} / M_i
    w = DIFF_COEFF * kvals / M_diag[rows]

    ei = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long,  device=device)
    ew = torch.tensor(w,                              dtype=torch.float32, device=device)
    nv = torch.tensor(nv_np,                          dtype=torch.float32, device=device)

    return {
        "type"         : "sparse",
        "edge_index"   : ei,
        "edge_weights" : ew,
        "node_volumes" : nv,
    }


def compute_pde_residual_3d(
    u:           torch.Tensor,
    nodes:       torch.Tensor,
    tets:        torch.Tensor,
    source_term: Optional[torch.Tensor] = None,
    dt:          float = 1.0,
) -> torch.Tensor:
    """计算当前时步的 3D PDE 残差幅值（用于 C1 物理感知锚点选取）。

    残差定义（简化版）：
        r_i ≈ |∇²_h u_i| = |(L_h u)_i|

    也可选传入 source_term f 后计算 |(L_h u + f)|。

    参数
    ----
    u           : (N,) 或 (N, C) 当前温度场
    source_term : (N,) 或 (N, C) 热源项（可选）

    返回
    ----
    residual : (N,) 残差幅值（非负）
    """
    L_dict = build_operator_3d(nodes, tets)
    ei = L_dict["edge_index"]
    ew = L_dict["edge_weights"]

    # (L_h u)_i = Σ_{j: (i,j)∈E} w_{ij} · (u_j - u_i)
    src_e = ei[0]
    dst_e = ei[1]

    if u.dim() == 1:
        msg  = ew * (u[dst_e] - u[src_e])
        Lu   = torch.zeros_like(u)
        Lu.scatter_add_(0, src_e, msg)
    else:
        msg  = ew.unsqueeze(1) * (u[dst_e] - u[src_e])
        Lu   = torch.zeros_like(u)
        idx  = src_e.unsqueeze(1).expand_as(msg)
        Lu.scatter_add_(0, idx, msg)

    if source_term is not None:
        residual = (Lu + source_term).norm(dim=-1) if Lu.dim() > 1 else (Lu + source_term).abs()
    else:
        residual = Lu.norm(dim=-1) if Lu.dim() > 1 else Lu.abs()

    return residual
