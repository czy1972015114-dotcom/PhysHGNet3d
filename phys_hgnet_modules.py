"""
phys_hgnet_modules.py  ← 锚点选择器重写版（形状保护加固）
================================================
相比上一版的改动（仅 PhysicsAwareAnchorSelector）：

新增 _to_node_field(t, N)：
  无论调用方（phys_hgnet.py 或 train_phys_hgnet_3d.py）传入的张量
  是 (N,) / (N,C) / (B,N,C) / (T,) / (B,T,N,C) 等任意形状，
  都能安全提取出 (N,) 的节点标量场；
  形状无法匹配时返回 None 而不是崩溃。

其余模块（LearnableCoarseOperator、DualScaleGNNCorrector 等）
完全保留原版（含 Fix A/A-extra 数值稳定修复）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple


def _pairwise_sq_dist(a, b):
    a2 = (a * a).sum(-1, keepdim=True)
    b2 = (b * b).sum(-1, keepdim=True).t()
    return (a2 + b2 - 2.0 * a @ b.t()).clamp(min=0.0)


def build_knn_edges(src_coords, dst_coords, k, eps=1e-8):
    N_src = src_coords.shape[0]
    N_dst = dst_coords.shape[0]
    k = min(k, N_dst)
    dists = _pairwise_sq_dist(src_coords, dst_coords)
    topk_d, topk_idx = dists.topk(k, dim=-1, largest=False)
    row = torch.arange(N_src, device=src_coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
    col = topk_idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    inv_d = 1.0 / (topk_d.clamp(min=eps).sqrt() + eps)
    weights = inv_d / inv_d.sum(-1, keepdim=True)
    return edge_index, weights.reshape(-1)


def _to_node_field(t: Optional[torch.Tensor], N: int) -> Optional[torch.Tensor]:
    """
    从任意形状张量中提取 (N,) 的节点标量场。

    支持的输入形状（按优先顺序匹配）：
      (N,)           → 直接返回
      (N, C)         → 取第 0 列
      (B, N, C)      → 取 batch=0, channel=0
      (B, N)         → 取 batch=0
      (B, T, N, C)   → 取 batch=0, t=-1, channel=0
    其他形状 → 返回 None（不崩溃）
    """
    if t is None:
        return None
    t = t.detach().float()
    if t.dim() == 1:
        if t.shape[0] == N:
            return t
    elif t.dim() == 2:
        if t.shape[0] == N:          # (N, C)
            return t[:, 0]
        if t.shape[1] == N:          # (B, N)
            return t[0]
    elif t.dim() == 3:
        if t.shape[1] == N:          # (B, N, C)
            return t[0, :, 0]
        if t.shape[0] == N:          # (N, ?, ?)
            return t[:, 0, 0]
    elif t.dim() == 4:
        if t.shape[2] == N:          # (B, T, N, C)
            return t[0, -1, :, 0]
    return None


# ──────────────────────────────────────────────────────────────────
# C1: Physics-Aware Anchor Selector（MLP 打分版）
# ──────────────────────────────────────────────────────────────────

class PhysicsAwareAnchorSelector(nn.Module):
    """
    可学习的物理感知锚点选择器（MLP 打分版）

    输入特征（7 维）：
      coords(3) + heat_source_q(1) + temperature(1)
      + residual_norm(1) + grad_norm(1)

    所有可选输入均通过 _to_node_field 做形状保护，
    无论调用方传入什么形状都不会崩溃。
    """

    def __init__(self, feat_dim: int = 7, hidden_dim: int = 32,
                 init_lambda: float = 0.3,
                 init_weights: Tuple[float, float, float] = (2.0, 1.0, 1.0)):
        super().__init__()
        self.feat_dim = feat_dim
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / x.abs().max().clamp(min=1e-8)

    def score_nodes(
        self,
        nodes: torch.Tensor,                          # (N, 3)
        source_q: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        grad_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回每个节点的标量得分 (N,)，完全可微。"""
        N, device = nodes.shape[0], nodes.device
        z = torch.zeros(N, device=device)

        # 所有输入都经过 _to_node_field 统一处理，保证形状为 (N,)
        q_t   = _to_node_field(source_q,   N)
        t_t   = _to_node_field(temperature, N)
        # residual 可能是 (N, C)，先处理多通道
        if residual is not None and residual.dim() > 1:
            residual = residual.norm(dim=-1)
        r_t   = _to_node_field(residual, N)
        g_t   = _to_node_field(grad_norm, N)

        coords_n = self._norm(nodes.float())
        q_n   = self._norm(q_t.to(device))   if q_t   is not None else z
        t_n   = self._norm(t_t.to(device))   if t_t   is not None else z
        r_n   = self._norm(r_t.to(device))   if r_t   is not None else z
        g_n   = self._norm(g_t.to(device))   if g_t   is not None else z

        feat = torch.stack(
            [coords_n[:, 0], coords_n[:, 1], coords_n[:, 2],
             q_n, t_n, r_n, g_n], dim=1)           # (N, 7)
        return self.scorer(feat).squeeze(-1)        # (N,)

    def forward(
        self,
        nodes: torch.Tensor,
        m: int,
        source_q: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        grad_norm: Optional[torch.Tensor] = None,
        use_physics_anchor: bool = True,
    ) -> torch.Tensor:
        N = nodes.shape[0]
        m = min(m, N)

        if use_physics_anchor:
            scores = self.score_nodes(
                nodes, source_q=source_q, temperature=temperature,
                residual=residual, grad_norm=grad_norm)
        else:
            scores = torch.zeros(N, device=nodes.device)

        _, anchor_idx = torch.topk(scores, m, sorted=False)
        return anchor_idx.long()

    def weight_summary(self) -> str:
        norms = [f"layer{i}_wnorm={p.norm().item():.4f}"
                 for i, (name, p) in enumerate(self.scorer.named_parameters())
                 if "weight" in name]
        return "  ".join(norms)


# ──────────────────────────────────────────────────────────────────
# C2: Learnable Coarse Operator (Graph Structure Learning)
# ──────────────────────────────────────────────────────────────────

class LearnableCoarseOperator(nn.Module):
    """
    GSL-based coarse operator.
    Fix A: Gershgorin 谱归一化，Fix A-extra: softplus 输出硬夹 max=10。
    """

    def __init__(self, spatial_dim, feat_dim, hidden_dim=64, k_coarse=6, eps=1e-8):
        super().__init__()
        self.k_coarse = k_coarse
        self.eps = eps
        in_dim = 2 * spatial_dim + 2 * feat_dim + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.constant_(self.edge_mlp[-1].bias, -2.0)

    def _fixed_weights(self, anchor_coords, edge_index):
        src, dst = edge_index
        dist = (anchor_coords[src] - anchor_coords[dst]).norm(dim=-1).clamp(min=self.eps)
        return 1.0 / dist

    def _learned_weights(self, anchor_coords, anchor_feats, edge_index):
        src, dst = edge_index
        xi, xj = anchor_coords[src], anchor_coords[dst]
        hi, hj = anchor_feats[src], anchor_feats[dst]
        dist = (xi - xj).norm(dim=-1, keepdim=True).clamp(min=self.eps)
        edge_feat = torch.cat([xi, xj, hi, hj, dist], dim=-1)
        raw = self.edge_mlp(edge_feat).squeeze(-1)
        w = F.softplus(raw)
        return w.clamp(max=10.0)

    def forward(self, anchor_coords, anchor_feats, edge_index, use_learned_coarse=True):
        m = anchor_coords.shape[0]
        w = (self._learned_weights(anchor_coords, anchor_feats, edge_index)
             if use_learned_coarse
             else self._fixed_weights(anchor_coords, edge_index))

        src, dst = edge_index
        row_oh = F.one_hot(src, num_classes=m).to(dtype=w.dtype)
        col_oh = F.one_hot(dst, num_classes=m).to(dtype=w.dtype)
        L_off     = row_oh.t() @ (w.unsqueeze(1) * col_oh)
        L_off_sym = L_off + L_off.t()
        diag_vals = -L_off_sym.sum(dim=1)
        L_hat     = L_off_sym + torch.diag(diag_vals)

        spec_bound = (2.0 * diag_vals.abs().max()).clamp(min=1e-8)
        return L_hat / spec_bound


# ──────────────────────────────────────────────────────────────────
# C3: Dual-Scale GNN Corrector
# ──────────────────────────────────────────────────────────────────

class _MPNN(MessagePassing):
    def __init__(self, dim, edge_dim, aggr='mean'):
        super().__init__(aggr=aggr)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.upd_norm = nn.LayerNorm(dim)
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.upd_norm(x + out)

    def message(self, x_i, x_j, edge_attr):
        if edge_attr.shape[-1] < self.edge_dim:
            edge_attr = F.pad(edge_attr, (0, self.edge_dim - edge_attr.shape[-1]))
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr[:, :self.edge_dim]], dim=-1))


class DualScaleGNNCorrector(nn.Module):
    def __init__(self, spatial_dim, feature_dim, output_dim,
                 hidden_dim=128, num_fine_layers=5, num_coarse_layers=4,
                 k_virtual_nodes=4, num_node_types=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k_vn = k_virtual_nodes

        self.fine_enc = nn.Sequential(
            nn.Linear(spatial_dim + feature_dim + num_node_types, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fine_layers = nn.ModuleList(
            [_MPNN(hidden_dim, spatial_dim + 1) for _ in range(num_fine_layers)])
        self.fine_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

        coarse_in = spatial_dim + feature_dim
        self.coarse_enc = nn.Sequential(
            nn.Linear(coarse_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.coarse_layers = nn.ModuleList(
            [_MPNN(hidden_dim, spatial_dim + 1) for _ in range(num_coarse_layers)])

        self.virtual_embed  = nn.Parameter(torch.randn(k_virtual_nodes, hidden_dim) * 0.01)
        self.vn_agg_mlp  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.vn_bcast_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.coarse_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

        self.raw_alpha_fine   = nn.Parameter(torch.zeros(1))
        self.raw_alpha_coarse = nn.Parameter(torch.zeros(1))

        for dec in [self.fine_dec, self.coarse_dec]:
            nn.init.normal_(dec[-1].weight, std=1e-3)
            nn.init.zeros_(dec[-1].bias)

    @property
    def alpha_fine(self):
        return F.softplus(self.raw_alpha_fine)

    @property
    def alpha_coarse(self):
        return F.softplus(self.raw_alpha_coarse)

    def _fine_forward(self, u, nodes, edge_index, edge_attr, node_type):
        nt = F.one_hot(node_type.long(), num_classes=3).float()
        h = self.fine_enc(torch.cat([u, nodes, nt], dim=-1))
        for layer in self.fine_layers:
            h = layer(h, edge_index, edge_attr)
        return self.fine_dec(h)

    def _virtual_node_step(self, h):
        vn     = self.virtual_embed
        pooled = h.mean(dim=0, keepdim=True)
        vn_updated = self.vn_agg_mlp(vn + pooled)
        vn_msg = vn_updated.mean(dim=0, keepdim=True).expand(h.shape[0], -1)
        return self.vn_bcast_mlp(torch.cat([h, vn_msg], dim=-1))

    def _coarse_forward(self, u_coarse, anchor_coords, coarse_ei, coarse_ea,
                        use_virtual_nodes=True):
        h = self.coarse_enc(torch.cat([u_coarse, anchor_coords], dim=-1))
        for i, layer in enumerate(self.coarse_layers):
            if use_virtual_nodes and (i % 2 == 0):
                h = self._virtual_node_step(h)
            h = layer(h, coarse_ei, coarse_ea)
        return self.coarse_dec(h)

    def forward(self, u, nodes, edge_index, edge_attr, node_type,
                anchor_idx, anchor_coords, R, P,
                coarse_edge_index, coarse_edge_attr,
                use_dual_scale=True, use_virtual_nodes=True):
        r_fine = self._fine_forward(u, nodes, edge_index, edge_attr, node_type)
        if not use_dual_scale:
            return r_fine
        u_coarse   = (torch.sparse.mm(R, u) if R.is_sparse else R @ u)
        r_coarse_m = self._coarse_forward(u_coarse, anchor_coords,
                                          coarse_edge_index, coarse_edge_attr,
                                          use_virtual_nodes=use_virtual_nodes)
        r_coarse_n = (torch.sparse.mm(P, r_coarse_m) if P.is_sparse else P @ r_coarse_m)
        return self.alpha_fine * r_fine + self.alpha_coarse * r_coarse_n


# ──────────────────────────────────────────────────────────────────
# R / P helpers
# ──────────────────────────────────────────────────────────────────

def build_restriction_prolongation(nodes, anchor_idx, q=4, eps=1e-8):
    N = nodes.shape[0]
    anchor_coords = nodes[anchor_idx]
    m = anchor_coords.shape[0]
    device = nodes.device
    edge_idx_R, w_R = build_knn_edges(anchor_coords, nodes, q, eps)
    R = torch.zeros(m, N, device=device, dtype=nodes.dtype)
    R[edge_idx_R[0], edge_idx_R[1]] = w_R
    edge_idx_P, w_P = build_knn_edges(nodes, anchor_coords, q, eps)
    P = torch.zeros(N, m, device=device, dtype=nodes.dtype)
    P[edge_idx_P[0], edge_idx_P[1]] = w_P
    return R, P


def build_coarse_edge_attr(anchor_coords, coarse_edge_index):
    src, dst = coarse_edge_index
    delta = anchor_coords[dst] - anchor_coords[src]
    dist  = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([delta, dist], dim=-1)
