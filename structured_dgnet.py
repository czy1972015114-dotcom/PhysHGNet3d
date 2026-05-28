"""
structured_dgnet.py  ← 数值上溢修复版
=======================================
修改点（仅 _precond_cg_solve 函数）：

Fix E1: 每次迭代后对 x, r 做渐进式夹值（±1e5 K）
  原因：非正定系统中 x 和残差向量会指数增长。
        逐步夹值比最后一次性夹值更安全，且不影响已收敛的情形。

Fix E2: 更完善的有限性检查
  原因：原始代码只检查 pAp/rz 是否有限，
        但 x 或 r 中的 inf 有时会传播到 pAp 之前就造成溢出。

Fix E3: 改进 fallback：返回 x0.clamp(T_MIN, T_MAX)（物理上合理）
  原因：原始代码返回 x.clamp(-1e12, 1e12) 过于宽松，
        1e12 K 会让后续 float32 运算立即溢出。

以下为 _precond_cg_solve 函数的完整修复版。
其余内容（结构体、类定义等）与原始文件一致，此处完整复制以直接替换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Dict, Any
import os
import math

from structured_models import (
    FineGraphEncoder, LocalCorrectionHead, ProlongationNet, CoarseGraphModule,
    farthest_point_sampling, build_bidirectional_edges, build_edge_features,
    build_reverse_edge_map, build_knn_graph,
    sparse_laplacian_matvec, structured_L_matvec,
)

# 物理合理温度范围
_T_MIN = 0.0
_T_MAX = 1e5   # 比 phys_hgnet.py 宽松一点，兼容一般 PDE

# ══════════════════════════════════════════════════════════════
# Preconditioned CG with warm start  ← 修复版
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def _precond_cg_solve(matvec_fn, b, precond_inv=None, max_iter=50, tol=1e-6, x0=None):
    """Preconditioned CG for SPD system Ax=b.
    （Fix E1/E2/E3：数值稳定性增强版）
    """
    b  = b.float()
    x  = x0.float().clone() if x0 is not None else torch.zeros_like(b)
    x0_init = x0.float().clone() if x0 is not None else None

    r = b - matvec_fn(x) if x0 is not None else b.clone()

    if precond_inv is not None:
        precond_inv = precond_inv.float().clamp(max=1e3)
        z = precond_inv * r
    else:
        z = r.clone()

    p  = z.clone()
    rz = r.dot(z)
    b_norm = b.dot(b).clamp(min=1e-30)
    thr = tol * tol * b_norm

    for it in range(max_iter):
        # ── Fix E2: 提前检查向量本身是否有限 ─────────────────────────
        if not (torch.isfinite(x).all() and torch.isfinite(r).all()
                and torch.isfinite(p).all()):
            if os.environ.get('PHGNET_DEBUG', '0') in ('1', 'true', 'True'):
                print(f"DEBUG: CG early exit (non-finite vector) at iter={it}")
            break

        if rz.abs() < thr:
            break

        Ap  = matvec_fn(p)
        pAp = p.dot(Ap)

        if os.environ.get('PHGNET_DEBUG', '0') in ('1', 'true', 'True'):
            print(f"DEBUG: CG iter={it} rz={float(rz):.3e} "
                  f"pAp={float(pAp):.3e} x_norm={float(x.norm()):.3e}")

        if (not torch.isfinite(pAp)) or (not torch.isfinite(rz)):
            if os.environ.get('PHGNET_DEBUG', '0') in ('1', 'true', 'True'):
                print(f"DEBUG: CG non-finite at iter={it} pAp={pAp} rz={rz}")
            break

        if pAp.abs() < 1e-30:
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        # ── Fix E1: 渐进式夹值防止指数增长 ────────────────────────────
        # 只在真正超出范围时才夹，避免影响已正常收敛的情形。
        x_max = x.abs().max()
        if x_max > _T_MAX:
            x = x * (_T_MAX / x_max.clamp(min=1.0))
            r = b - matvec_fn(x)   # 重新计算残差（接受一步开销）
            if os.environ.get('PHGNET_DEBUG', '0') in ('1', 'true', 'True'):
                print(f"DEBUG: CG clamped x at iter={it} (x_max={float(x_max):.3e})")

        z      = precond_inv * r if precond_inv is not None else r
        rz_new = r.dot(z)
        beta   = rz_new / (rz + 1e-30)
        p      = z + beta * p
        rz     = rz_new

    # ── Fix E3: 改进 fallback ─────────────────────────────────────────
    # 使用物理合理值而非 ±1e12 K（会立即造成后续 float32 溢出）
    if not torch.isfinite(x).all():
        if x0_init is not None and torch.isfinite(x0_init).all():
            return x0_init
        # 最后兜底：用 b 本身（相当于把 A 近似为 I）
        return b.clamp(_T_MIN, _T_MAX)

    return x


class ImplicitCGSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, s_w, c_w, P, R, a_loc, a_coarse,
                Lp_w, fine_ei, coarse_ei, N, m, dt, max_iter, tol, precond_inv, x0):
        b       = b.float()
        s_w_f   = s_w.detach().float()
        c_w_f   = c_w.detach().float()
        P_f     = P.detach().float()
        R_f     = R.detach().float()
        a_loc_f = a_loc.detach().float()
        a_coarse_f = a_coarse.detach().float()
        Lp_w_f  = Lp_w.detach().float()

        def A_mv(v):
            Lv = structured_L_matvec(v, Lp_w_f, fine_ei, s_w_f, a_loc_f,
                                     P_f, R_f, c_w_f, coarse_ei, a_coarse_f, N, m)
            return v - (dt / 2.0) * Lv

        pc   = precond_inv.detach().float() if precond_inv is not None else None
        warm = x0.detach().float() if x0 is not None else None
        x    = _precond_cg_solve(A_mv, b.detach(), pc, max_iter, tol, warm)

        ctx.save_for_backward(x, s_w, c_w, P, R, a_loc, a_coarse, Lp_w, fine_ei, coarse_ei)
        ctx.N, ctx.m, ctx.dt = N, m, dt
        ctx.max_iter, ctx.tol = max_iter, tol
        ctx.precond_inv = pc
        return x

    @staticmethod
    def backward(ctx, grad_x):
        x, sw, cw, P, R, al, ac, Lpw, fei, cei = ctx.saved_tensors
        N, m, dt = ctx.N, ctx.m, ctx.dt
        grad_x = grad_x.float()
        sw_f, cw_f = sw.detach().float(), cw.detach().float()
        P_f, R_f   = P.detach().float(),  R.detach().float()
        al_f, ac_f = al.detach().float(), ac.detach().float()
        Lpw_f = Lpw.detach().float()

        def A_mv(v):
            Lv = structured_L_matvec(v, Lpw_f, fei, sw_f, al_f, P_f, R_f,
                                     cw_f, cei, ac_f, N, m)
            return v - (dt / 2.0) * Lv

        lam = _precond_cg_solve(A_mv, grad_x.detach(), ctx.precond_inv,
                                ctx.max_iter, ctx.tol)

        with torch.enable_grad():
            sw_g  = sw.detach().float().requires_grad_(True)
            cw_g  = cw.detach().float().requires_grad_(True)
            P_g   = P.detach().float().requires_grad_(True)
            al_g  = al.detach().float().requires_grad_(True)
            ac_g  = ac.detach().float().requires_grad_(True)
            col_sum = P_g.sum(0).clamp(min=1e-6)
            R_g   = (P_g / col_sum.unsqueeze(0)).t()
            Lt_x  = structured_L_matvec(
                x.detach().float(), Lpw_f, fei, sw_g, al_g, P_g, R_g, cw_g, cei, ac_g, N, m)
            grads = torch.autograd.grad(
                Lt_x, [sw_g, cw_g, P_g, al_g, ac_g],
                grad_outputs=(dt / 2.0) * lam.detach(), allow_unused=True)

        return (lam, grads[0], grads[1], grads[2], None,
                grads[3], grads[4], None, None, None,
                None, None, None, None, None, None, None)


# ══════════════════════════════════════════════════════════════
# MPNN 和残差求解器（原始代码，未修改）
# ══════════════════════════════════════════════════════════════

class _MPNNSimple(MessagePassing):
    def __init__(self, dim, aggr='mean'):
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        if edge_attr.shape[-1] != x_i.shape[-1]:
            edge_attr = F.pad(edge_attr, (0, x_i.shape[-1] - edge_attr.shape[-1]))
        return self.mlp(torch.cat([x_i, x_j, edge_attr], -1))


class SimpleNonlinearSolver(nn.Module):
    def __init__(self, spatial_dim, feature_dim, output_dim, hidden_dim=128,
                 num_layers=5, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.enc = nn.Sequential(
            nn.Linear(spatial_dim + feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([_MPNNSimple(hidden_dim) for _ in range(num_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, nodes, edge_index, edge_attr, u):
        h = self.enc(torch.cat([nodes, u], -1))
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h, edge_index, edge_attr))
        return self.dec(h)


class ResidualSolver(nn.Module):
    def __init__(self, spatial_dim, feature_dim, output_dim, hidden_dim=128,
                 num_layers=5, num_node_types=3, use_checkpoint=False):
        super().__init__()
        self.num_node_types = num_node_types
        self.enc = nn.Sequential(
            nn.Linear(spatial_dim + feature_dim + num_node_types, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([_MPNNSimple(hidden_dim) for _ in range(num_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, nodes, edge_index, edge_attr, u, node_type, boundary_info=None):
        nt = F.one_hot(node_type.long(), num_classes=self.num_node_types).float()
        h  = self.enc(torch.cat([u, nodes, nt], dim=-1))
        h_anchor = h.clone()
        dirichlet_idx = None
        if boundary_info and isinstance(boundary_info, dict):
            di = boundary_info.get('dirichlet', None)
            if di is not None and 'indices' in di:
                dirichlet_idx = di['indices']
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h, edge_index, edge_attr))
            if dirichlet_idx is not None:
                h[dirichlet_idx] = h_anchor[dirichlet_idx]
        return self.dec(h)


# ══════════════════════════════════════════════════════════════
# StructuredDGNet（原始代码，未修改）
# ══════════════════════════════════════════════════════════════

class StructuredDGNet(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.spatial_dim   = config.get('spatial_dim', 2)
        self.feature_dim   = config.get('feature_dim', 1)
        self.output_dim    = config.get('output_dim',  1)
        self.m_anchors     = config.get('m_anchors', 64)
        self.q_local       = config.get('q_local', 4)
        self.tau           = config.get('tau', 0.1)
        self.k_coarse      = config.get('k_coarse', 6)
        self.cg_max_iter   = config.get('cg_max_iter', 50)
        self.cg_tol        = config.get('cg_tol', 1e-6)

        op_hd = config.get('operator_hidden_dim', 64)
        op_nl = config.get('operator_num_layers', 3)
        self.fine_encoder  = FineGraphEncoder(self.spatial_dim, op_hd, op_nl)
        self.local_head    = LocalCorrectionHead(op_hd, self.spatial_dim + 1, op_hd)
        self.prolong_net   = ProlongationNet(op_hd, op_hd, self.spatial_dim,
                                             self.q_local, self.tau)
        self.coarse_module = CoarseGraphModule(
            op_hd, self.spatial_dim + 1, op_hd,
            config.get('coarse_num_layers', 2))

        init_raw = math.log(max(math.exp(0.001) - 1.0, 1e-10))
        self.raw_alpha_loc    = nn.Parameter(torch.tensor(init_raw))
        self.raw_alpha_coarse = nn.Parameter(torch.tensor(init_raw))

        res_hd   = config.get('residual_hidden_dim', 128)
        res_nl   = config.get('residual_num_layers', 5)
        use_ckpt = config.get('use_checkpoint', False)
        self.nonlinear_solver = SimpleNonlinearSolver(
            self.spatial_dim, self.feature_dim, self.output_dim,
            hidden_dim=res_hd, num_layers=res_nl, use_checkpoint=use_ckpt)
        self.residual_solver = ResidualSolver(
            self.spatial_dim, self.feature_dim, self.output_dim,
            hidden_dim=res_hd, num_layers=res_nl, use_checkpoint=use_ckpt)

        self._gc     = None
        self._gc_key = None

    def _get_graph_cache(self, nodes, edges, node_volumes, node_type, L_physics):
        N = nodes.shape[0]
        device    = nodes.device
        cache_key = (N, str(device))
        if self._gc is not None and self._gc_key == cache_key:
            return self._gc

        fine_ei  = build_bidirectional_edges(edges)
        fine_ea  = build_edge_features(nodes, fine_ei)
        fine_rev = build_reverse_edge_map(fine_ei)
        m = min(self.m_anchors, N // 2)
        anchor_idx    = farthest_point_sampling(nodes, m)
        anchor_coords = nodes[anchor_idx]
        k_c = min(self.k_coarse, m - 1)
        coarse_ei, coarse_ea = build_knn_graph(anchor_coords, k_c)
        coarse_rev = build_reverse_edge_map(coarse_ei)

        if isinstance(L_physics, dict) and L_physics.get('type') == 'sparse':
            sp_ei  = L_physics['edge_index']
            sp_w   = L_physics['edge_weights']
            sp_N   = L_physics.get('N', N)
            sp_key = sp_ei[0] * sp_N + sp_ei[1]
            fine_key = fine_ei[0] * sp_N + fine_ei[1]
            sorted_sp_key, sp_perm = sp_key.sort()
            pos     = torch.searchsorted(sorted_sp_key, fine_key).clamp(max=len(sp_key) - 1)
            matched = (sorted_sp_key[pos] == fine_key)
            Lp_w    = torch.zeros(fine_ei.shape[1], device=device)
            Lp_w[matched] = sp_w[sp_perm[pos[matched]]]
            Lp_w    = Lp_w.detach()
            # 兼容旧版（无 diag / L_scale 键）
            if 'diag' in L_physics:
                L_diag  = L_physics['diag'].to(device).detach()
                L_scale = torch.tensor(L_physics['L_scale'], device=device).detach()
            else:
                L_diag  = torch.zeros(N, device=device)
                L_diag.scatter_add_(0, fine_ei[0], Lp_w)
                L_scale = Lp_w.abs().max().clamp(min=1.0).detach()
        else:
            Lp_w    = L_physics[fine_ei[0], fine_ei[1]].detach()
            L_diag  = L_physics.diag().detach()
            L_scale = L_physics.abs().max().detach().clamp(min=1.0)

        self._gc = {
            'N': N, 'm': m,
            'fine_ei': fine_ei, 'fine_ea': fine_ea, 'fine_rev': fine_rev,
            'anchor_idx': anchor_idx, 'anchor_coords': anchor_coords,
            'coarse_ei': coarse_ei, 'coarse_ea': coarse_ea, 'coarse_rev': coarse_rev,
            'Lp_w': Lp_w, 'L_diag': L_diag, 'L_scale': L_scale,
        }
        self._gc_key = cache_key
        return self._gc

    def _compute_components(self, gc, nodes, node_volumes, node_type):
        h    = self.fine_encoder(nodes, gc['fine_ei'], gc['fine_ea'], node_volumes, node_type)
        s_w  = self.local_head(h, gc['fine_ei'], gc['fine_ea'], gc['fine_rev'])
        h_anc = h[gc['anchor_idx']]
        P, R  = self.prolong_net(h, h_anc, nodes, gc['anchor_coords'])
        Pt_h  = P.t() @ h
        Pt_1  = P.sum(0).clamp(min=1e-6).unsqueeze(1)
        g_c   = Pt_h / Pt_1
        c_w, _ = self.coarse_module(g_c, gc['coarse_ei'], gc['coarse_ea'], gc['coarse_rev'])
        L_scale = gc['L_scale']
        a_loc    = L_scale * F.softplus(self.raw_alpha_loc)
        a_coarse = L_scale * F.softplus(self.raw_alpha_coarse)
        return s_w.float(), c_w.float(), P.float(), R.float(), a_loc.float(), a_coarse.float()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        nodes       = batch['nodes']
        edges       = batch['edges']
        src_term    = batch['source_terms']
        init_cond   = batch['initial_conditions']
        t_pts       = batch['time_points']
        vol         = batch['node_volumes']
        L_phys      = batch.get('L_physics', None)
        device      = nodes.device
        B, T, N, C  = src_term.shape
        dt = float(t_pts[1] - t_pts[0]) if T > 1 else 0.01
        node_type   = batch.get('node_type', torch.zeros(N, dtype=torch.long, device=device))
        boundary_info = batch.get('boundary_info', None)

        if L_phys is None:
            from physics import build_operator
            L_phys = build_operator(
                nodes=nodes, edges=edges, faces=batch.get('faces'),
                node_volumes=vol, operator_type='laplace')

        gc = self._get_graph_cache(nodes, edges, vol, node_type, L_phys)
        s_w, c_w, P, R, al, ac = self._compute_components(gc, nodes, vol, node_type)
        Lp_w, fei, fea, cei, m = gc['Lp_w'], gc['fine_ei'], gc['fine_ea'], gc['coarse_ei'], gc['m']
        L_diag = gc['L_diag']

        precond_diag = 1.0 + (dt / 2.0) * L_diag.abs()
        precond_diag = precond_diag.clamp(min=1e-6)
        precond_inv  = (1.0 / precond_diag).clamp(max=1e3)

        u_cur = init_cond[:, 0] if init_cond.dim() == 4 else init_cond
        step_first  = u_cur
        step_second = None
        step_last   = None

        for t in range(T - 1):
            u_cur = u_cur.float()
            f0 = src_term[:, t].float()
            f1 = src_term[:, t + 1].float()

            r_list = [self.nonlinear_solver(nodes, fei, fea, u_cur[b]) for b in range(B)]
            r_uk   = torch.stack(r_list, 0).float()

            Lt_u = torch.zeros_like(u_cur)
            for c in range(C):
                for b in range(B):
                    Lt_u[b, :, c] = structured_L_matvec(
                        u_cur[b, :, c], Lp_w, fei, s_w, al, P, R, c_w, cei, ac, N, m)

            rhs = u_cur + (dt / 2) * Lt_u + (dt / 2) * (f0 + f1) + dt * r_uk

            u_phys_parts = []
            for c in range(C):
                b_parts = [ImplicitCGSolve.apply(
                    rhs[b, :, c], s_w, c_w, P, R, al, ac, Lp_w, fei, cei, N, m, dt,
                    self.cg_max_iter, self.cg_tol, precond_inv, u_cur[b, :, c].detach())
                    for b in range(B)]
                u_phys_parts.append(torch.stack(b_parts, 0))
            u_phys_next = torch.stack(u_phys_parts, -1)

            u_net_list  = [self.residual_solver(nodes, fei, fea, u_cur[b],
                                                node_type, boundary_info)
                           for b in range(B)]
            u_net_next  = torch.stack(u_net_list, 0).float()
            u_next      = u_phys_next + u_net_next

            if boundary_info and isinstance(boundary_info, dict) and 'dirichlet' in boundary_info:
                di  = boundary_info['dirichlet']
                idx = di['indices'].to(device)
                val = di['values'].to(device)
                for c in range(C):
                    u_next[:, idx, c] = val.unsqueeze(0) if val.dim() == 1 else val

            u_next = u_next.clamp(_T_MIN, _T_MAX)  # 物理范围保护

            if t == 0:
                step_second = u_next
            step_last = u_next
            u_cur = u_next.detach()

        steps = [step_first]
        steps.append(step_second if step_second is not None else step_first)
        if T > 2 and step_last is not None:
            for _ in range(T - 3):
                steps.append(u_cur.detach())
            steps.append(step_last)
        elif step_last is not None:
            steps.append(step_last)

        return {'u_final': torch.stack(steps, 1)}


class StructuredDGNetLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        u  = pred['u_final']
        lT = self.mse(u[:, -1].float(), target[:, -1].float())
        l1 = self.mse(u[:, 1].float(),  target[:, 1].float())
        return {'total_loss': l1 + lT,
                'first_step_loss': l1.item(), 'final_step_loss': lT.item()}
