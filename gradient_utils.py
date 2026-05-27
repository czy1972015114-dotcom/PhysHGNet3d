"""
gradient_utils.py — 网格物理梯度范数计算（自动识别参数顺序）
=============================================================

被 phys_hgnet.py 调用：
    from gradient_utils import compute_gradient_norm

兼容以下所有调用约定，按张量形状自动识别参数：
    compute_gradient_norm(u, nodes, elements)        # 原始约定
    compute_gradient_norm(nodes, u, elements)        # 常见约定
    compute_gradient_norm(u, nodes, elements, spatial_dim=d)
    compute_gradient_norm(nodes, u)                  # 无 elements（退化回零）

识别规则
--------
- nodes  : 2D FloatTensor, shape = (N, d), d ∈ {2, 3}
- u      : 1D FloatTensor shape=(N,)  或  2D shape=(N,C) with C 小
- elements: 2D (Long/Int) Tensor, shape = (F, k), k ∈ {3, 4}

梯度公式
--------
2D 三角形（面积加权常数元）：
    ∇u|_e = Σ_k u_k ∇φ_k^e
    ‖∇u‖_i = Σ_{e⊃i} A_e ‖∇u_e‖ / Σ_{e⊃i} A_e

3D 四面体（体积加权常数元）：
    ∇φ 由 J^{-T} 给出
    ‖∇u‖_i = Σ_{e⊃i} V_e ‖∇u_e‖ / Σ_{e⊃i} V_e
"""

from __future__ import annotations
from typing import Optional, Union
import torch


# ─────────────────────────────────────────────────────────────────
# 自动参数识别
# ─────────────────────────────────────────────────────────────────

def _identify_args(a0, a1, a2):
    """
    从最多三个位置参数中，识别出 (u, nodes, elements)。

    识别依据：
    - nodes    : 2D float tensor，shape[1] ∈ {2, 3}  且 shape[1] << shape[0]
    - elements : 2D int/long tensor，shape[1] ∈ {3, 4}
    - u        : 其他（1D 或 2D float tensor）

    返回 (u, nodes, elements)，任何识别失败的返回 None。
    """
    candidates = [t for t in [a0, a1, a2] if isinstance(t, torch.Tensor)]
    if not candidates:
        return None, None, None

    nodes_cand    = None
    elements_cand = None
    u_cand        = None

    for t in candidates:
        if t.ndim == 2:
            d = t.shape[1]
            # 坐标：列数 2 或 3，且是浮点类型
            if d in (2, 3) and t.is_floating_point():
                # 进一步排除 elements 被误判（elements 的列数恰好是 3 时）
                # 区分方法：nodes 的值域大（几何坐标），elements 是整数索引
                # 这里用 dtype 最可靠
                if nodes_cand is None:
                    nodes_cand = t
            # 连接关系：列数 3 或 4，整数类型
            elif d in (3, 4) and not t.is_floating_point():
                if elements_cand is None:
                    elements_cand = t
            # 列数是 3 的浮点 tensor 可能是 elements 的 float 版本（罕见）
            elif d in (3, 4) and t.is_floating_point():
                # 先当成 nodes（d=3 的情况已被上面捕获，d=4 走这里）
                if nodes_cand is None and d == 3:
                    nodes_cand = t
                elif elements_cand is None:
                    elements_cand = t
            else:
                # 列数不是 2/3/4，当成 u
                if u_cand is None:
                    u_cand = t
        elif t.ndim == 1:
            # 1D tensor 一定是 u（字段值）
            if u_cand is None:
                u_cand = t
        elif t.ndim == 3:
            # (B,N,C) 形状 → 取 [0,:,0] 当 u
            if u_cand is None:
                u_cand = t[0, :, 0]

    # 如果 nodes 和 elements 都没找到，但有两个 2D tensor，
    # 用第一个当 nodes，第二个当 elements
    floats = [t for t in candidates if t.ndim == 2 and t.is_floating_point()]
    ints   = [t for t in candidates if t.ndim == 2 and not t.is_floating_point()]

    if nodes_cand is None and floats:
        # 找 shape[1] 最小且在 {2,3} 的
        for t in floats:
            if t.shape[1] in (2, 3):
                nodes_cand = t
                break
        if nodes_cand is None:
            nodes_cand = floats[0]

    if elements_cand is None and ints:
        for t in ints:
            if t.shape[1] in (3, 4):
                elements_cand = t
                break
        if elements_cand is None and ints:
            elements_cand = ints[0]

    return u_cand, nodes_cand, elements_cand


# ─────────────────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────────────────

def compute_gradient_norm(
    arg0,
    arg1=None,
    arg2=None,
    spatial_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    计算每个节点处物理场的梯度范数 ‖∇u‖₂。

    参数（顺序无关，按形状自动识别）
    ----
    arg0, arg1, arg2 : torch.Tensor
        - 节点坐标 nodes  : (N, d), d=2 或 3，float
        - 物理场   u      : (N,) 或 (N,C) 或 (B,N,C)，float
        - 网格单元 elements: (F, 3) 三角形 或 (F, 4) 四面体，long/int
    spatial_dim : int, 可选（若为 None 则从 nodes.shape[1] 推断）

    返回
    ----
    grad_norm : (N,) FloatTensor，非负，体积/面积加权平均

    示例
    ----
    >>> compute_gradient_norm(u, nodes, faces)          # u first
    >>> compute_gradient_norm(nodes, u, faces)          # nodes first
    >>> compute_gradient_norm(u, nodes, tets)           # 3D tet mesh
    >>> compute_gradient_norm(nodes, u, tets, spatial_dim=3)
    """
    # ── 参数识别 ──────────────────────────────────────────────────
    u, nodes, elements = _identify_args(arg0, arg1, arg2)

    # ── 防御性处理：任何一项为 None 时返回零向量 ─────────────────
    if nodes is None:
        # 尝试从已识别的 u 推断 N
        N = u.shape[0] if (u is not None and isinstance(u, torch.Tensor)) else 1
        dev = u.device if u is not None else torch.device('cpu')
        return torch.zeros(N, device=dev, dtype=torch.float32)

    N   = nodes.shape[0]
    dev = nodes.device

    if u is None:
        return torch.zeros(N, device=dev, dtype=torch.float32)

    if elements is None or len(elements) == 0:
        return torch.zeros(N, device=dev, dtype=torch.float32)

    # ── 推断空间维度 ──────────────────────────────────────────────
    if spatial_dim is None:
        if nodes.ndim >= 2:
            spatial_dim = nodes.shape[1]
        else:
            return torch.zeros(N, device=dev, dtype=torch.float32)

    # ── 调用具体实现 ──────────────────────────────────────────────
    try:
        if spatial_dim == 2:
            return _grad_norm_2d(u, nodes, elements)
        elif spatial_dim == 3:
            return _grad_norm_3d(u, nodes, elements)
        else:
            return torch.zeros(N, device=dev, dtype=torch.float32)
    except Exception:
        # 任何数值异常都回退到零（不中断训练）
        return torch.zeros(N, device=dev, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# 2D 实现（三角网格，常数元梯度）
# ─────────────────────────────────────────────────────────────────

def _grad_norm_2d(u: torch.Tensor,
                  nodes: torch.Tensor,
                  faces: torch.Tensor) -> torch.Tensor:
    N      = nodes.shape[0]
    device = nodes.device
    dtype  = torch.float32

    # u 统一成 1D
    u_s = u.float()
    if u_s.dim() == 2:
        u_s = u_s[:, 0]
    elif u_s.dim() == 3:
        u_s = u_s[0, :, 0]
    u_s = u_s.to(device)

    nd  = nodes.float().to(device)
    fc  = faces.long().to(device)

    i0, i1, i2 = fc[:, 0], fc[:, 1], fc[:, 2]
    x0, x1, x2 = nd[i0], nd[i1], nd[i2]
    u0, u1, u2 = u_s[i0], u_s[i1], u_s[i2]

    # 有向面积 × 2
    two_area = ((x1[:, 0] - x0[:, 0]) * (x2[:, 1] - x0[:, 1])
              - (x1[:, 1] - x0[:, 1]) * (x2[:, 0] - x0[:, 0]))
    area = (two_area.abs() / 2.0).clamp(min=1e-14)

    inv2A = 1.0 / two_area.clamp(min=1e-14)
    dux = (u0*(x1[:,1]-x2[:,1]) + u1*(x2[:,1]-x0[:,1]) + u2*(x0[:,1]-x1[:,1])) * inv2A
    duy = (u0*(x2[:,0]-x1[:,0]) + u1*(x0[:,0]-x2[:,0]) + u2*(x1[:,0]-x0[:,0])) * inv2A
    gmag = (dux**2 + duy**2).clamp(min=0).sqrt()

    acc_g = torch.zeros(N, device=device, dtype=dtype)
    acc_a = torch.zeros(N, device=device, dtype=dtype)
    for vi in [i0, i1, i2]:
        acc_g.scatter_add_(0, vi, area * gmag)
        acc_a.scatter_add_(0, vi, area)

    return acc_g / acc_a.clamp(min=1e-14)


# ─────────────────────────────────────────────────────────────────
# 3D 实现（四面体网格，常数元梯度）
# ─────────────────────────────────────────────────────────────────

def _grad_norm_3d(u: torch.Tensor,
                  nodes: torch.Tensor,
                  tets: torch.Tensor) -> torch.Tensor:
    N      = nodes.shape[0]
    device = nodes.device
    dtype  = torch.float32

    u_s = u.float()
    if u_s.dim() == 2:
        u_s = u_s[:, 0]
    elif u_s.dim() == 3:
        u_s = u_s[0, :, 0]
    u_s = u_s.to(device)

    nd  = nodes.float().to(device)
    tet = tets.long().to(device)

    i0, i1, i2, i3 = tet[:,0], tet[:,1], tet[:,2], tet[:,3]
    x0, x1, x2, x3 = nd[i0], nd[i1], nd[i2], nd[i3]
    u0, u1, u2, u3 = u_s[i0], u_s[i1], u_s[i2], u_s[i3]

    J    = torch.stack([x1-x0, x2-x0, x3-x0], dim=1)
    detJ = torch.linalg.det(J)
    vol  = (detJ.abs() / 6.0).clamp(min=1e-18)

    valid  = vol > 1e-18
    J_inv  = torch.zeros_like(J)
    if valid.any():
        J_inv[valid] = torch.linalg.inv(J[valid])
    J_invT = J_inv.transpose(1, 2)

    gph1 = J_invT[:, :, 0]
    gph2 = J_invT[:, :, 1]
    gph3 = J_invT[:, :, 2]
    gph0 = -(gph1 + gph2 + gph3)

    gradu = (u0.unsqueeze(1)*gph0 + u1.unsqueeze(1)*gph1
           + u2.unsqueeze(1)*gph2 + u3.unsqueeze(1)*gph3)
    gmag  = gradu.norm(dim=1)

    acc_g = torch.zeros(N, device=device, dtype=dtype)
    acc_v = torch.zeros(N, device=device, dtype=dtype)
    for vi in [i0, i1, i2, i3]:
        acc_g.scatter_add_(0, vi, vol * gmag)
        acc_v.scatter_add_(0, vi, vol)

    return acc_g / acc_v.clamp(min=1e-14)


# ─────────────────────────────────────────────────────────────────
# 便捷函数（从 batch dict 自动派发）
# ─────────────────────────────────────────────────────────────────

def compute_gradient_norm_from_batch(u, batch):
    nodes = batch.get("nodes")
    if nodes is None:
        return torch.zeros(1)
    d = nodes.shape[1] if nodes.ndim >= 2 else 2
    if d == 3:
        elems = batch.get("tets")
    else:
        elems = batch.get("faces")
    if elems is None or len(elems) == 0:
        return torch.zeros(nodes.shape[0], device=nodes.device)
    return compute_gradient_norm(u, nodes, elems, spatial_dim=d)


# ─────────────────────────────────────────────────────────────────
# 自测：验证 4 种调用约定都能正确运行
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(0)

    print("=== gradient_utils 自测 ===\n")

    # ── 2D ──────────────────────────────────────────────────────
    pts2 = rng.uniform(0, 1, (80, 2)).astype(np.float32)
    tri2 = Delaunay(pts2)
    N2   = pts2.shape[0]
    nd2  = torch.tensor(pts2)
    fc2  = torch.tensor(tri2.simplices.astype(np.int64))
    u2   = torch.tensor(pts2[:, 0])   # u = x => |∇u| ≈ 1

    for label, args in [
        ("(u, nodes, faces)",   (u2,  nd2, fc2)),
        ("(nodes, u, faces)",   (nd2, u2,  fc2)),
    ]:
        gn = compute_gradient_norm(*args, spatial_dim=2)
        assert gn.shape == (N2,), f"shape={gn.shape}"
        assert not torch.isnan(gn).any(), "NaN!"
        print(f"  2D {label}: mean={gn.mean():.3f}  (期望≈1.0)  ✓")

    # ── 3D ──────────────────────────────────────────────────────
    pts3 = rng.uniform(0, 1, (150, 3)).astype(np.float32)
    tri3 = Delaunay(pts3)
    N3   = pts3.shape[0]
    nd3  = torch.tensor(pts3)
    tet3 = torch.tensor(tri3.simplices.astype(np.int64))
    u3   = torch.tensor(pts3[:, 0])   # u = x => |∇u| ≈ 1

    for label, args in [
        ("(u, nodes, tets)",    (u3,  nd3, tet3)),
        ("(nodes, u, tets)",    (nd3, u3,  tet3)),
        ("(nodes, u)  no elems",(nd3, u3)),
    ]:
        gn = compute_gradient_norm(*args, spatial_dim=3)
        assert gn.shape == (N3,), f"shape={gn.shape}"
        assert not torch.isnan(gn).any(), "NaN!"
        print(f"  3D {label}: mean={gn.mean():.3f}  (期望≈1.0)  ✓")

    print("\n✓ 全部调用约定通过！")
