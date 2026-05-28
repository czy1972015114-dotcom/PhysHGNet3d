"""
phys_hgnet_modules.py — PhysHGNet C1 / C2 / C3 modules.
修复版（v2）：
  Fix: 为 PhysicsAwareAnchorSelector 添加 score_nodes() 方法，
       使 train_phys_hgnet_3d.py 中的 anchor_focus_loss 真正生效。
  其余代码不变。
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
    inv_d  = 1.0 / (topk_d.clamp(min=eps).sqrt() + eps)
    weights = inv_d / inv_d.sum(-1, keepdim=True)
    return edge_index, weights.reshape(-1)


# ─────────────────────────────────────────────────────────────
# C1: Physics-Aware Anchor Selector
# ─────────────────────────────────────────────────────────────

class PhysicsAwareAnchorSelector(nn.Module):
    """
    Three-term weighted FPS:
      C_i = α · d_min_norm  +  β · residual_norm  +  γ · grad_norm
    """

    def __init__(self, init_lambda: float = 0.3,
                 init_weights: Tuple[float, float, float] = (2.0, 1.0, 1.0)):
        super().__init__()
        self.raw_weights = nn.Parameter(
            torch.tensor(list(init_weights), dtype=torch.float32))

    @property
    def weights(self) -> torch.Tensor:
        return F.softmax(self.raw_weights, dim=0)

    @property
    def lam(self) -> torch.Tensor:
        return self.weights[1]

    # ── Fix: 新增 score_nodes()，供 anchor_focus_loss 使用 ────────
    def score_nodes(
        self,
        nodes: torch.Tensor,           # (N, d)
        source_q: torch.Tensor,        # (N,) 热源强度（可微）
        temperature: Optional[torch.Tensor] = None,  # (N,) 温度（可选）
    ) -> torch.Tensor:
        """
        可微的节点打分函数，供 compute_anchor_focus_loss() 调用。

        返回 (N,) 分数，分数越高越倾向于被选为锚点。
        梯度可回传到 self.raw_weights（α/β/γ），使锚点偏好与物理场对齐。

        实现思路：
          score_i = β * q_norm_i  +  γ * temp_norm_i
          （α 的几何项在可微打分中省略，因为 d_min 依赖离散 argmax，不可微）
        """
        _, beta, gamma = self.weights.unbind()

        q_norm = source_q / source_q.max().clamp(min=1e-8)
        score  = beta * q_norm

        if temperature is not None:
            t_norm = temperature / temperature.max().clamp(min=1e-8)
            score  = score + gamma * t_norm

        return score  # (N,)，可微
    # ── End Fix ───────────────────────────────────────────────────

    def forward(self,
                nodes: torch.Tensor,
                m: int,
                residual: Optional[torch.Tensor] = None,
                grad_norm: Optional[torch.Tensor] = None,
                use_physics_anchor: bool = True,
                ) -> torch.Tensor:
        N = nodes.shape[0]
        m = min(m, N)
        device = nodes.device

        def _normalize(x):
            return x / x.max().clamp(min=1e-8)

        if use_physics_anchor:
            alpha, beta, gamma = self.weights.unbind()
            r_norm = (_normalize(
                residual.norm(dim=-1) if residual.dim() > 1 else residual.abs())
                if residual is not None
                else torch.zeros(N, device=device))
            g_norm = (_normalize(grad_norm)
                if grad_norm is not None
                else torch.zeros(N, device=device))
        else:
            alpha = torch.ones(1, device=device)
            beta  = torch.zeros(1, device=device)
            gamma = torch.zeros(1, device=device)
            r_norm = torch.zeros(N, device=device)
            g_norm = torch.zeros(N, device=device)

        if use_physics_anchor and (residual is not None or grad_norm is not None):
            seed_score = beta * r_norm + gamma * g_norm
            first = int(seed_score.argmax().item())
        else:
            first = 0

        selected = [first]
        d_min = _pairwise_sq_dist(nodes, nodes[first:first + 1]).squeeze(1).sqrt()

        for _ in range(m - 1):
            d_norm = _normalize(d_min)
            cost = alpha * d_norm + beta * r_norm + gamma * g_norm
            cost[selected] = -float('inf')
            next_idx = int(cost.argmax().item())
            selected.append(next_idx)
            d_new = _pairwise_sq_dist(
                nodes, nodes[next_idx:next_idx + 1]).squeeze(1).sqrt()
            d_min = torch.minimum(d_min, d_new)

        return torch.tensor(selected, dtype=torch.long, device=device)

    def weight_summary(self) -> str:
        alpha, beta, gamma = self.weights.detach().unbind()
        return f"α(geo)={alpha:.3f} β(res)={beta:.3f} γ(grad)={gamma:.3f}"


# ─────────────────────────────────────────────────────────────
# C2: Learnable Coarse Operator
# ─────────────────────────────────────────────────────────────

class LearnableCoarseOperator(nn.Module):
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
        return F.softplus(self.edge_mlp(edge_feat).squeeze(-1))

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
        return L_off_sym + torch.diag(diag_vals)


# ─────────────────────────────────────────────────────────────
# C3: Dual-Scale GNN Corrector
# ─────────────────────────────────────────────────────────────

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
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim
        self.k_vn        = k_virtual_nodes

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
        self.vn_agg_mlp     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.vn_bcast_mlp   = nn.Sequential(
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
        h  = self.fine_enc(torch.cat([u, nodes, nt], dim=-1))
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
        u_coarse    = (torch.sparse.mm(R, u) if R.is_sparse else R @ u)
        r_coarse_m  = self._coarse_forward(u_coarse, anchor_coords,
                                            coarse_edge_index, coarse_edge_attr,
                                            use_virtual_nodes=use_virtual_nodes)
        r_coarse_n  = (torch.sparse.mm(P, r_coarse_m) if P.is_sparse else P @ r_coarse_m)
        return self.alpha_fine * r_fine + self.alpha_coarse * r_coarse_n


# ─────────────────────────────────────────────────────────────
# R / P helpers
# ─────────────────────────────────────────────────────────────

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
