"""
phys_hgnet.py — PhysHGNet: Physics-aware Hierarchical Graph Neural Operator.
修复版（v2）：
  Fix 1: 去掉 _build_graph 中 256 的硬上限，让 m_anchors 真正生效
  Fix 2: graph_cache_key 加入 rebuild_epoch，使锚点定期随残差/梯度刷新
  Fix 3: _step_counter 移到 if 外，每时间步都递增（保证 rebuild 频率正确）
  Fix 4: DEFAULT_CONFIG 新增 graph_rebuild_freq=20
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from structured_models import (
    FineGraphEncoder, build_bidirectional_edges, build_edge_features, build_knn_graph,
    farthest_point_sampling,
)
from structured_dgnet import ImplicitCGSolve, _precond_cg_solve, ResidualSolver as _OrigResidualSolver
from phys_hgnet_modules import (
    PhysicsAwareAnchorSelector, LearnableCoarseOperator, DualScaleGNNCorrector,
    build_restriction_prolongation, build_coarse_edge_attr,
)
from gradient_utils import compute_gradient_norm

DEFAULT_CONFIG: Dict[str, Any] = {
    "spatial_dim": 2, "feature_dim": 1, "output_dim": 1,
    "m_anchors": 64, "q_local": 4, "k_coarse": 6,
    "operator_hidden_dim": 64, "operator_num_layers": 3,
    "residual_hidden_dim": 128, "residual_num_layers": 5,
    "coarse_num_layers": 4, "k_virtual_nodes": 4,
    "cg_max_iter": 50, "cg_tol": 1e-6,
    "use_physics_anchor": True,
    "use_learned_coarse": True,
    "use_dual_scale_gnn": True,
    "use_virtual_nodes": True,
    "residual_update_freq": 5,
    "graph_rebuild_freq": 20,   # ← Fix 4: 新增，每 20 步重建一次图（刷新锚点）
    "operator_type": "laplace",
}


def _sparse_Lv(v, Lp_w, fine_ei):
    src, dst = fine_ei
    if v.dim() == 1:
        msg = Lp_w * (v[dst] - v[src])
        out = torch.zeros_like(v)
        out.scatter_add_(0, src, msg)
    else:
        msg = Lp_w.unsqueeze(1) * (v[dst] - v[src])
        out = torch.zeros_like(v)
        idx = src.unsqueeze(1).expand(-1, v.shape[1])
        out.scatter_add_(0, idx, msg)
    return out


def _build_jacobi_precond(L_local_weights, fine_ei, N, dt):
    diag_L = torch.zeros(N, device=L_local_weights.device, dtype=L_local_weights.dtype)
    diag_L.scatter_add_(0, fine_ei[0], L_local_weights)
    diag_A = 1.0 + (dt / 2.0) * diag_L
    return 1.0 / diag_A.clamp(min=1e-8)


def _match_weights_to_edge_index(sp_ei, sp_w, fine_ei, N):
    device = sp_ei.device
    E = fine_ei.shape[1]
    sp_key   = sp_ei[0] * N + sp_ei[1]
    fine_key = fine_ei[0] * N + fine_ei[1]
    sorted_key, perm = sp_key.sort()
    sorted_w = sp_w[perm]
    idx   = torch.searchsorted(sorted_key, fine_key).clamp(max=sorted_key.shape[0] - 1)
    found = sorted_key[idx] == fine_key
    weights = torch.zeros(E, device=device, dtype=sp_w.dtype)
    weights[found] = sorted_w[idx[found]]
    return weights


def _apply_bcs(u, boundary_info):
    if not isinstance(boundary_info, dict):
        return u
    di = boundary_info.get("dirichlet")
    if di is None:
        return u
    idx = di.get("indices")
    val = di.get("values")
    if idx is None:
        return u
    device = u.device
    idx = idx.to(device)
    val = val.to(device)
    u = u.clone()
    if val is not None:
        u[:, idx] = val.unsqueeze(0).unsqueeze(-1).expand(u.shape[0], -1, u.shape[-1])
    return u


class PhysHGNet(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **config}
        self.cfg = cfg
        self.spatial_dim  = cfg["spatial_dim"]
        self.feature_dim  = cfg["feature_dim"]
        self.output_dim   = cfg["output_dim"]
        self.m_anchors    = cfg["m_anchors"]
        self.q_local      = cfg["q_local"]
        self.k_coarse     = cfg["k_coarse"]
        self.cg_max_iter  = cfg["cg_max_iter"]
        self.cg_tol       = cfg["cg_tol"]
        self.res_upd_freq = cfg["residual_update_freq"]
        self.graph_rebuild_freq = cfg.get("graph_rebuild_freq", 20)   # ← Fix 4

        self.use_physics_anchor  = cfg["use_physics_anchor"]
        self.use_learned_coarse  = cfg["use_learned_coarse"]
        self.use_dual_scale_gnn  = cfg["use_dual_scale_gnn"]
        self.use_virtual_nodes   = cfg["use_virtual_nodes"]

        op_hd = cfg["operator_hidden_dim"]
        op_nl = cfg["operator_num_layers"]
        self.fine_encoder = FineGraphEncoder(self.spatial_dim, op_hd, op_nl)

        init_raw = math.log(max(math.exp(0.001) - 1.0, 1e-10))
        self.raw_alpha_loc    = nn.Parameter(torch.tensor(init_raw))
        self.raw_alpha_coarse = nn.Parameter(torch.tensor(init_raw))

        self.anchor_selector = PhysicsAwareAnchorSelector(
            init_lambda=0.3,
            init_weights=(2.0, 1.0, 1.0),
        )
        self.learnable_coarse = LearnableCoarseOperator(
            spatial_dim=self.spatial_dim, feat_dim=op_hd,
            hidden_dim=op_hd, k_coarse=self.k_coarse)
        self.dual_scale_corrector = DualScaleGNNCorrector(
            spatial_dim=self.spatial_dim, feature_dim=self.feature_dim,
            output_dim=self.output_dim, hidden_dim=cfg["residual_hidden_dim"],
            num_fine_layers=cfg["residual_num_layers"],
            num_coarse_layers=cfg["coarse_num_layers"],
            k_virtual_nodes=cfg["k_virtual_nodes"])

        self._graph_cache:   Any = None
        self._cache_key:     Any = None
        self._residual_cache: Optional[torch.Tensor] = None
        self._grad_norm_cache: Optional[torch.Tensor] = None
        self._step_counter:  int = 0

    @property
    def alpha_loc(self):
        return F.softplus(self.raw_alpha_loc)

    @property
    def alpha_coarse_op(self):
        return F.softplus(self.raw_alpha_coarse)

    # ── graph construction ────────────────────────────────────
    def _build_graph(self, nodes, edges, L_physics, node_volumes=None,
                     node_type=None, u_init=None):
        N = nodes.shape[0]
        device = nodes.device

        fine_ei = build_bidirectional_edges(edges)
        fine_ea = build_edge_features(nodes, fine_ei)

        if isinstance(L_physics, dict) and L_physics.get("type") == "sparse":
            sp_ei, sp_w = L_physics["edge_index"], L_physics["edge_weights"]
        else:
            L_dense = L_physics if isinstance(L_physics, torch.Tensor) else torch.zeros(N, N, device=device)
            sp_mask = (L_dense != 0)
            sp_idx  = sp_mask.nonzero(as_tuple=True)
            sp_ei   = torch.stack(sp_idx, dim=0)
            sp_w    = L_dense[sp_idx]

        Llocal_weights = _match_weights_to_edge_index(sp_ei, sp_w, fine_ei, N)

        # ── Fix 1: 去掉 256 硬上限 ──────────────────────────────
        # 原来: m_target = min(self.m_anchors, max(8, N // 16), 256)
        # N=4000 → 250 anchors (6.25%)
        # N=6000 → 256 anchors (4.27%)  ← 被 256 截断，覆盖率下降！
        # 修复: 让 m_anchors 完全控制，仅保留下限 8 防止退化
        m_target = min(self.m_anchors, N)
        m_target = max(8, m_target)
        # ──────────────────────────────────────────────────────────

        # ── C1: three-term FPS ───────────────────────────────────
        with torch.no_grad():   # ← 加这一行：FPS 返回 long 索引，不可微，no_grad 正确
            anchor_idx = self.anchor_selector(
                nodes, m_target,
                residual=self._residual_cache,
                grad_norm=self._grad_norm_cache,
                use_physics_anchor=self.use_physics_anchor,
            )
        anchor_coords = nodes[anchor_idx]
        m = anchor_idx.shape[0]

        R, P = build_restriction_prolongation(nodes, anchor_idx, q=self.q_local)

        k_c = min(self.k_coarse, m - 1)
        coarse_ei, _ = build_knn_graph(anchor_coords, k_c)
        coarse_ea    = build_coarse_edge_attr(anchor_coords, coarse_ei)

        _nv = node_volumes if node_volumes is not None else torch.ones(N, device=device, dtype=nodes.dtype)
        _nt = node_type    if node_type    is not None else torch.zeros(N, device=device, dtype=torch.long)

        with torch.no_grad():
            node_feats_enc = self.fine_encoder(nodes, fine_ei, fine_ea, _nv, _nt)
            anchor_feats   = node_feats_enc[anchor_idx]

        if os.environ.get('PHGNET_DEBUG', '0') in ('1', 'true', 'True'):
            print(f"DEBUG _build_graph: N={N}, m_target={m_target}, m={m}, "
                  f"ratio={m/N:.3f}, residual_valid={self._residual_cache is not None}")

        return {
            "N": N, "m": m,
            "fine_ei": fine_ei, "fine_ea": fine_ea,
            "Llocal_weights": Llocal_weights,
            "anchor_idx": anchor_idx, "anchor_coords": anchor_coords,
            "anchor_feats_static": anchor_feats,
            "R": R, "P": P,
            "coarse_ei": coarse_ei, "coarse_ea": coarse_ea,
            "m_target": m_target,
        }

    def _Leff_matvec(self, v, gc, anchor_feats):
        Lv_loc = _sparse_Lv(v, gc["Llocal_weights"], gc["fine_ei"])
        R, P   = gc["R"], gc["P"]
        coarse_ei    = gc["coarse_ei"]
        anchor_coords = gc["anchor_coords"]

        v_c = R @ v if v.dim() == 1 else R @ v
        L_hat  = self.learnable_coarse(
            anchor_coords, anchor_feats, coarse_ei,
            use_learned_coarse=self.use_learned_coarse)
        Lv_c      = L_hat @ v_c
        Lv_coarse = P @ Lv_c

        return self.alpha_loc * Lv_loc + self.alpha_coarse_op * Lv_coarse

    # ── forward ───────────────────────────────────────────────
    def forward(self, batch: Dict[str, Any],
                use_physics_anchor=None, use_learned_coarse=None,
                use_dual_scale_gnn=None, use_virtual_nodes=None,
                ) -> Dict[str, torch.Tensor]:

        _pa = use_physics_anchor if use_physics_anchor is not None else self.use_physics_anchor
        _lc = use_learned_coarse if use_learned_coarse is not None else self.use_learned_coarse
        _ds = use_dual_scale_gnn if use_dual_scale_gnn is not None else self.use_dual_scale_gnn
        _vn = use_virtual_nodes  if use_virtual_nodes  is not None else self.use_virtual_nodes

        nodes     = batch["nodes"]
        edges     = batch["edges"]
        faces     = batch.get("faces")
        node_type = batch.get("node_type",
                              torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device))
        bnd_info  = batch.get("boundary_info", {})
        L_physics = batch.get("L_physics", None)
        src_terms = batch["source_terms"]
        time_pts  = batch["time_points"]
        u_init    = batch["initial_conditions"]

        B, T, N, C = src_terms.shape
        device = nodes.device
        dt = float(time_pts[1] - time_pts[0]) if T > 1 else 0.0

        if L_physics is None:
            try:
                from physics import build_operator
                L_physics = build_operator(
                    nodes=nodes, edges=edges, faces=faces,
                    node_volumes=batch.get("node_volumes"),
                    operator_type=self.cfg.get("operator_type", "laplace"))
            except Exception:
                L_physics = torch.zeros(N, N, device=device)

        # ── Fix 2: cache_key 加入 rebuild_epoch，使锚点定期刷新 ─────
        # 原来: cache_key = (N, str(device), _pa)
        # 问题: 整个训练周期 cache_key 从不变化，锚点永远是第一个 batch
        #       用 residual=None/grad_norm=None 选出来的 FPS 结果
        # 修复: 每 graph_rebuild_freq 步重建一次图，使用最新的残差信息
        _rebuild_epoch = self._step_counter // self.graph_rebuild_freq
        cache_key = (N, str(device), _pa, _rebuild_epoch)
        if self._graph_cache is None or self._cache_key != cache_key:
            self._graph_cache = self._build_graph(
                nodes, edges, L_physics,
                node_volumes=batch.get("node_volumes"),
                node_type=node_type, u_init=u_init[:, 0])
            self._cache_key = cache_key
        # ──────────────────────────────────────────────────────────────

        gc = self._graph_cache

        _nv = batch.get("node_volumes")
        if _nv is None:
            _nv = torch.ones(nodes.shape[0], device=device, dtype=nodes.dtype)

        anchor_feats = self.fine_encoder(
            nodes, gc["fine_ei"], gc["fine_ea"], _nv, node_type)[gc["anchor_idx"]]

        precond_inv = _build_jacobi_precond(gc["Llocal_weights"], gc["fine_ei"], N, dt)

        u_hist = torch.zeros(B, T, N, C, device=device)
        u_curr = u_init
        u_hist[:, 0] = u_curr

        cg_warm_start = None

        for t in range(T - 1):
            f_cur  = src_terms[:, t]
            f_next = src_terms[:, t + 1]

            # ── Fix 3: _step_counter 移到 if 外，每时间步都递增 ──────
            # 原来 _step_counter 在 if 内部递增，导致每 res_upd_freq 步才加 1
            # 改为每个时间步都递增，保证 graph_rebuild_freq 的计数正确
            if _pa and (self._step_counter % self.res_upd_freq == 0):
                with torch.no_grad():
                    u0 = u_curr[0, :, 0]
                    Lu = self._Leff_matvec(u0, gc, anchor_feats.detach())
                    self._residual_cache = (Lu + f_cur[0, :, 0]).abs().detach()

                    if faces is not None:
                        self._grad_norm_cache = compute_gradient_norm(
                            nodes, u0, faces).detach()
                    else:
                        src_e, dst_e = gc["fine_ei"]
                        du = (u0[dst_e] - u0[src_e]).abs()
                        g = torch.zeros(N, device=device, dtype=u0.dtype)
                        g.scatter_add_(0, src_e, du)
                        cnt = torch.zeros(N, device=device, dtype=u0.dtype)
                        cnt.scatter_add_(0, src_e, torch.ones_like(du))
                        self._grad_norm_cache = (g / cnt.clamp(min=1)).detach()

            self._step_counter += 1   # ← Fix 3: 移到 if 外
            # ──────────────────────────────────────────────────────

            # ── Physics CG solve ──────────────────────────────────
            u_phys_next = torch.zeros_like(u_curr)
            for b in range(B):
                u_b = u_curr[b]
                def Bop_mv(v):
                    return v + (dt / 2.0) * self._Leff_matvec(v, gc, anchor_feats)
                rhs_b = (Bop_mv(u_b[:, 0]) +
                         (dt / 2.0) * (f_cur[b, :, 0] + f_next[b, :, 0])).unsqueeze(-1)
                def A_mv_scalar(v):
                    return v - (dt / 2.0) * self._Leff_matvec(v, gc, anchor_feats)
                x0 = cg_warm_start[b][:, 0] if cg_warm_start is not None else u_b[:, 0]
                x_sol = _precond_cg_solve(A_mv_scalar, rhs_b[:, 0],
                                          precond_inv=precond_inv,
                                          max_iter=self.cg_max_iter,
                                          tol=self.cg_tol, x0=x0)
                u_phys_next[b] = x_sol.unsqueeze(-1)
            cg_warm_start = u_phys_next.detach()

            # ── C3: dual-scale GNN correction ────────────────────
            u_corr = torch.zeros_like(u_curr)
            for b in range(B):
                u_corr[b] = self.dual_scale_corrector(
                    u=u_curr[b], nodes=nodes,
                    edge_index=gc["fine_ei"], edge_attr=gc["fine_ea"],
                    node_type=node_type,
                    anchor_idx=gc["anchor_idx"], anchor_coords=gc["anchor_coords"],
                    R=gc["R"], P=gc["P"],
                    coarse_edge_index=gc["coarse_ei"], coarse_edge_attr=gc["coarse_ea"],
                    use_dual_scale=_ds, use_virtual_nodes=_vn)

            u_next = u_phys_next + u_corr
            if bnd_info:
                u_next = _apply_bcs(u_next, bnd_info)

            u_hist[:, t + 1] = u_next
            u_curr = u_next.detach()

        return {"u_final": u_hist}

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def ablation_summary(self):
        anchor_w = self.anchor_selector.weight_summary()
        lines = [
            "=== PhysHGNet Ablation Config ===",
            f"  C1 Physics Anchor  : {'ON' if self.use_physics_anchor else 'OFF'}",
            f"  Anchor weights     : {anchor_w}",
            f"  C2 Learned Coarse  : {'ON' if self.use_learned_coarse else 'OFF'}",
            f"  C3 Dual-Scale GNN  : {'ON' if self.use_dual_scale_gnn else 'OFF'}",
            f"  C3 Virtual Nodes   : {'ON' if self.use_virtual_nodes else 'OFF'}",
            f"  graph_rebuild_freq : {self.graph_rebuild_freq}",
            f"  Total Parameters   : {self.num_parameters():,}",
        ]
        return "\n".join(lines)
