"""
generate_laser_data_3d.py — 3D 激光淬火热方程数据生成器（PhysHGNet 版本）

在 DGNet 3D 版本基础上额外保存：
  - node_volumes  : 每个节点的控制体积（tet 体积 / 4 的节点聚合）
  - L_physics_ei  : 稀疏 FEM Laplacian 的 edge_index（供 PhysHGNet3D 预加载）
  - L_physics_ew  : 稀疏 FEM Laplacian 的 edge_weights

输出目录：data_laser_hardening_3d/
文件格式：pde_trajectories_3d_N{N}.h5

使用方法：
    python generate_laser_data_3d.py --n_nodes 5000
    python generate_laser_data_3d.py --all 2000,5000,10000 --n_traj 30
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay

# ═══════════════════════════════════════════════════════════════
# 物理参数（钢，与 2D 版本一致）
# ═══════════════════════════════════════════════════════════════
BLOCK_X, BLOCK_Y, BLOCK_Z = 0.5, 0.3, 0.05   # 3D 钢块尺寸（m）
RHO        = 7850.0    # 密度 kg/m³
CP         = 450.0     # 比热容 J/(kg·K)
K_TH       = 50.0      # 热导率 W/(m·K)
H_CONV     = 25.0      # 对流换热系数 W/(m²·K)
T_AMB      = 298.15    # 环境温度 K
T_SIM      = 30.0      # 模拟时长 s
DT         = 0.5       # 时间步长 s
NUM_LASERS = 5
DIFF_COEFF = K_TH / (RHO * CP)   # 热扩散系数 m²/s
MIN_TET_VOL = 1e-18               # 退化四面体体积阈值

# ═══════════════════════════════════════════════════════════════
# 网格生成：Delaunay 四面体网格
# ═══════════════════════════════════════════════════════════════

def generate_tet_mesh(n_target: int, seed: int = 42):
    """生成 ~n_target 节点的 3D Delaunay 四面体网格。

    Returns
    -------
    coords   : (N, 3) float64
    tets     : (F, 4) int32
    edges    : (E, 2) int32  唯一无向边
    bnd_faces: (BF, 3) int32 边界三角面
    """
    rng = np.random.default_rng(seed)

    n_face = max(6, int(np.cbrt(n_target) * 0.5))
    xs = np.linspace(0, BLOCK_X, n_face)
    ys = np.linspace(0, BLOCK_Y, n_face)
    zs = np.linspace(0, BLOCK_Z, max(3, n_face // 3))

    face_pts = []
    for z in [0.0, BLOCK_Z]:
        for x in xs:
            for y in ys:
                face_pts.append([x, y, z])
    for y in [0.0, BLOCK_Y]:
        for x in xs:
            for z in zs[1:-1]:
                face_pts.append([x, y, z])
    for x in [0.0, BLOCK_X]:
        for y in ys[1:-1]:
            for z in zs[1:-1]:
                face_pts.append([x, y, z])

    face_pts = np.array(face_pts, dtype=np.float64)
    jitter = min(BLOCK_X, BLOCK_Y, BLOCK_Z) * 1e-6
    face_pts += rng.uniform(-jitter, jitter, face_pts.shape)
    face_pts[:, 0] = np.clip(face_pts[:, 0], 0.0, BLOCK_X)
    face_pts[:, 1] = np.clip(face_pts[:, 1], 0.0, BLOCK_Y)
    face_pts[:, 2] = np.clip(face_pts[:, 2], 0.0, BLOCK_Z)

    n_int = max(0, n_target - len(face_pts))
    interior = rng.uniform(
        low =[0.005, 0.005, 0.002],
        high=[BLOCK_X - 0.005, BLOCK_Y - 0.005, BLOCK_Z - 0.002],
        size=(n_int, 3),
    )
    coords = np.concatenate([face_pts, interior], axis=0)
    coords = np.unique(np.round(coords, decimals=8), axis=0).astype(np.float64)

    tri   = Delaunay(coords)
    tets  = tri.simplices.astype(np.int32)
    tets, n_rem = _filter_degenerate_tets(coords, tets)
    if n_rem > 0:
        print(f"  [mesh] 移除 {n_rem} 个退化四面体 (vol < {MIN_TET_VOL:.0e})")

    # 重新索引，剔除孤立节点
    used = np.unique(tets.ravel())
    if len(used) < len(coords):
        old2new = np.full(len(coords), -1, dtype=np.int32)
        old2new[used] = np.arange(len(used), dtype=np.int32)
        coords = coords[used]
        tets   = old2new[tets]
        print(f"  [mesh] 重新索引: {len(used)} 个活跃节点")

    # 提取无向边（每个 tet 有 6 条边）
    edge_set = set()
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                edge_set.add((min(tet[i], tet[j]), max(tet[i], tet[j])))
    edges = np.array(sorted(edge_set), dtype=np.int32)

    bnd_faces = _extract_boundary_faces(tets)
    return coords, tets, edges, bnd_faces


def _filter_degenerate_tets(coords: np.ndarray, tets: np.ndarray):
    x0, x1, x2, x3 = coords[tets[:,0]], coords[tets[:,1]], coords[tets[:,2]], coords[tets[:,3]]
    J   = np.stack([x1-x0, x2-x0, x3-x0], axis=1)
    vol = np.abs(np.linalg.det(J)) / 6.0
    good = vol > MIN_TET_VOL
    return tets[good], int(np.sum(~good))


def _extract_boundary_faces(tets: np.ndarray) -> np.ndarray:
    face_count = {}
    for tet in tets:
        for fn in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
            f = tuple(sorted([tet[k] for k in fn]))
            face_count[f] = face_count.get(f, 0) + 1
    return np.array([f for f, c in face_count.items() if c == 1], dtype=np.int32)

# ═══════════════════════════════════════════════════════════════
# 3D FEM 刚度矩阵 & 节点体积（热方程离散）
# ═══════════════════════════════════════════════════════════════

def assemble_stiffness_mass_3d(coords: np.ndarray, tets: np.ndarray):
    """组装 3D FEM 刚度矩阵 K 和集中质量向量 M_diag。

    线性四面体单元，每个单元的形函数梯度为常数：
        K_e[i,j] = k_th · V_e · (∇φ_i · ∇φ_j)
        M_i      = ρ·cp · Σ_{e: i∈e} V_e/4

    Returns
    -------
    K      : scipy sparse CSR (N, N)
    M_diag : (N,) float64   集中质量（ρ·cp·V_i）
    """
    N     = coords.shape[0]
    n_tet = tets.shape[0]

    x0, x1, x2, x3 = coords[tets[:,0]], coords[tets[:,1]], coords[tets[:,2]], coords[tets[:,3]]
    J    = np.stack([x1-x0, x2-x0, x3-x0], axis=1)  # (n_tet, 3, 3)
    detJ = np.linalg.det(J)
    vol  = np.abs(detJ) / 6.0

    assert np.all(vol > 0), "存在体积为零的四面体，请先过滤退化单元！"

    J_inv  = np.linalg.inv(J)                         # (n_tet, 3, 3)
    J_invT = np.transpose(J_inv, (0, 2, 1))

    # 形函数梯度（局部坐标到全局坐标变换）
    grad = np.zeros((n_tet, 4, 3))
    grad[:, 1, :] = J_invT[:, :, 0]
    grad[:, 2, :] = J_invT[:, :, 1]
    grad[:, 3, :] = J_invT[:, :, 2]
    grad[:, 0, :] = -(grad[:, 1, :] + grad[:, 2, :] + grad[:, 3, :])

    rows_l, cols_l, vals_l = [], [], []
    for i in range(4):
        for j in range(4):
            dot_ij = np.sum(grad[:, i, :] * grad[:, j, :], axis=1)
            rows_l.append(tets[:, i])
            cols_l.append(tets[:, j])
            vals_l.append(K_TH * vol * dot_ij)

    K = sp.csr_matrix(
        (np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(N, N)
    )

    # 集中质量 M_i = ρ·cp·V_i，其中 V_i = Σ_{e⊃i} V_e / 4
    M_diag = np.zeros(N)
    for k in range(4):
        np.add.at(M_diag, tets[:, k], vol / 4.0)
    M_diag = np.maximum(M_diag, 1e-14)
    return K, M_diag


def compute_node_volumes(coords: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """节点控制体积 V_i = Σ_{e⊃i} V_e / 4（不含 ρ·cp）。"""
    N = coords.shape[0]
    x0, x1, x2, x3 = coords[tets[:,0]], coords[tets[:,1]], coords[tets[:,2]], coords[tets[:,3]]
    J   = np.stack([x1-x0, x2-x0, x3-x0], axis=1)
    vol = np.abs(np.linalg.det(J)) / 6.0
    nv  = np.zeros(N)
    for k in range(4):
        np.add.at(nv, tets[:, k], vol / 4.0)
    return np.maximum(nv, 1e-14)


def build_sparse_laplacian_3d(K_sp: "sp.csr_matrix", M_diag: np.ndarray) -> tuple:
    """将 FEM 刚度矩阵转化为 PhysHGNet 期望的稀疏 Laplacian 格式。

    离散扩散算子（质量归一化）：
        (L_h u)_i = Σ_{j≠i} [diffCoeff · K_{ij} / (ρ·cp·V_i)] · (u_j - u_i)

    Returns
    -------
    ei : (2, 2·E_off) int64  有向边索引（包含正反两个方向）
    ew : (2·E_off,) float32  边权重 w_{ij} = diffCoeff · K_{ij} / M_i
    """
    K_coo = K_sp.tocoo()
    mask  = K_coo.row != K_coo.col          # 仅保留非对角元
    rows  = K_coo.row[mask].astype(np.int64)
    cols  = K_coo.col[mask].astype(np.int64)
    vals  = K_coo.data[mask]

    # 归一化：w_{ij} = diffCoeff · K_{ij} / M_i
    w = DIFF_COEFF * vals / M_diag[rows]

    ei = np.stack([rows, cols], axis=0)   # (2, E_off)
    ew = w.astype(np.float32)
    return ei, ew

# ═══════════════════════════════════════════════════════════════
# 边界质量矩阵（对流 BC）
# ═══════════════════════════════════════════════════════════════

def assemble_boundary_mass_3d(coords: np.ndarray, bnd_faces: np.ndarray):
    N = coords.shape[0]
    if len(bnd_faces) == 0:
        return sp.csr_matrix((N, N)), np.array([], dtype=np.int32)
    rows, cols, vals = [], [], []
    bnd_set = set()
    for face in bnd_faces:
        i, j, k = int(face[0]), int(face[1]), int(face[2])
        v1   = coords[j] - coords[i]
        v2   = coords[k] - coords[i]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        for n in [i, j, k]:
            rows.append(n); cols.append(n); vals.append(area / 3.0)
            bnd_set.add(n)
    B = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return B, np.array(sorted(bnd_set), dtype=np.int32)

# ═══════════════════════════════════════════════════════════════
# 激光热源（顶面加热，xy 高斯 + z 指数衰减）
# ═══════════════════════════════════════════════════════════════

def initialize_lasers_3d(seed: int) -> list:
    rng = np.random.default_rng(seed)
    lasers = []
    for _ in range(NUM_LASERS):
        lasers.append({
            'cx': rng.uniform(0.1*BLOCK_X, 0.9*BLOCK_X),
            'cy': rng.uniform(0.1*BLOCK_Y, 0.9*BLOCK_Y),
            'vx': rng.uniform(-0.01, 0.01),
            'vy': rng.uniform(-0.01, 0.01),
            'power':  rng.uniform(5e7, 1e8),
            'radius': rng.uniform(0.02, 0.04),
        })
    return lasers


def compute_laser_source(coords: np.ndarray, lasers: list, t: float) -> np.ndarray:
    """3D 激光体积热源 Q(x,t) [W/m³]。"""
    Q = np.zeros(coords.shape[0])
    for ls in lasers:
        lx = ls['cx'] + ls['vx'] * t
        ly = ls['cy'] + ls['vy'] * t
        r2 = (coords[:, 0] - lx)**2 + (coords[:, 1] - ly)**2
        depth = BLOCK_Z - coords[:, 2]          # 从顶面向下深度
        Q += ls['power'] * np.exp(-r2 / (2 * ls['radius']**2)) \
                         * np.exp(-depth / 0.01)  # 穿透深度 ~10mm
    return Q

# ═══════════════════════════════════════════════════════════════
# 模拟单条轨迹
# ═══════════════════════════════════════════════════════════════

def simulate_trajectory_3d(coords, lu, RHS_op, M_diag, B_amb_vec,
                            n_steps, time_points, traj_seed: int):
    N = coords.shape[0]
    T   = np.full(N, T_AMB, dtype=np.float64)
    traj = np.zeros((n_steps, N, 1), dtype=np.float32)
    src  = np.zeros((n_steps, N, 1), dtype=np.float32)
    traj[0, :, 0] = T_AMB

    lasers = initialize_lasers_3d(traj_seed)
    for step in range(1, n_steps):
        t_now = float(time_points[step])
        Q     = compute_laser_source(coords, lasers, t_now)
        F_vec = Q * M_diag
        rhs   = RHS_op @ T + B_amb_vec + F_vec
        T     = lu.solve(rhs)
        traj[step, :, 0] = T.astype(np.float32)
        src [step, :, 0] = (Q / (RHO * CP)).astype(np.float32)
    return traj, src

# ═══════════════════════════════════════════════════════════════
# 主生成函数
# ═══════════════════════════════════════════════════════════════

def generate_for_n(n_target: int, n_traj: int, out_dir: str, seed: int = 42):
    print(f"\n{'='*64}")
    print(f"  3D 数据生成: 目标 N ≈ {n_target}")
    print(f"{'='*64}")

    t0 = time.time()
    coords, tets, edges, bnd_faces = generate_tet_mesh(n_target, seed=seed)
    N = coords.shape[0]
    print(f"  网格: N={N}, 四面体={len(tets)}, 边={len(edges)}, "
          f"边界面={len(bnd_faces)}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    K, M_diag    = assemble_stiffness_mass_3d(coords, tets)
    B, bnd_idx   = assemble_boundary_mass_3d(coords, bnd_faces)
    node_vol     = compute_node_volumes(coords, tets)
    L_ei, L_ew   = build_sparse_laplacian_3d(K, M_diag)
    print(f"  FEM: K.nnz={K.nnz}, L.E={L_ei.shape[1]}, "
          f"边界节点={len(bnd_idx)}  ({time.time()-t0:.1f}s)")

    # Crank-Nicolson 系统矩阵
    M_sp_over_dt = sp.diags(M_diag * RHO * CP / DT)
    LHS          = M_sp_over_dt + 0.5*K + 0.5*H_CONV*B
    RHS_op       = M_sp_over_dt - 0.5*K - 0.5*H_CONV*B
    B_amb_vec    = H_CONV * (B @ np.full(N, T_AMB))

    print(f"  LU 分解 (N={N})...")
    t0 = time.time()
    lu = splu(LHS.tocsc())
    print(f"  LU 完成 ({time.time()-t0:.1f}s)")

    out_dir_  = Path(out_dir)
    out_dir_.mkdir(parents=True, exist_ok=True)
    out_file  = out_dir_ / f"pde_trajectories_3d_N{N}.h5"

    n_steps      = int(T_SIM / DT) + 1
    time_points  = np.arange(n_steps, dtype=np.float64) * DT

    print(f"  生成 {n_traj} 条轨迹 → {out_file}")
    t_gen = time.time()
    with h5py.File(out_file, "w") as f:
        # 全局网格元数据（所有轨迹共享）
        meta = f.create_group("mesh_meta")
        meta.create_dataset("nodes",        data=coords.astype(np.float32))
        meta.create_dataset("edges",        data=edges)
        meta.create_dataset("faces",        data=bnd_faces)
        meta.create_dataset("tets",         data=tets)
        meta.create_dataset("node_volumes", data=node_vol.astype(np.float32))
        meta.create_dataset("L_physics_ei", data=L_ei)     # (2, 2E_off)
        meta.create_dataset("L_physics_ew", data=L_ew)     # (2E_off,)
        meta.create_dataset("time_points",  data=time_points.astype(np.float32))

        for ti in range(n_traj):
            t1   = time.time()
            traj, src = simulate_trajectory_3d(
                coords, lu, RHS_op, M_diag, B_amb_vec,
                n_steps, time_points, traj_seed=seed + 1000*(ti+1)
            )
            t_range = (traj[:,:,0].min(), traj[:,:,0].max())
            print(f"  轨迹 {ti+1:3d}/{n_traj} ({time.time()-t1:.1f}s) "
                  f"T∈[{t_range[0]:.0f}, {t_range[1]:.0f}]K")

            g = f.create_group(f"trajectory_{ti}")
            g.create_dataset("node_features",     data=traj)
            g.create_dataset("source_terms",      data=src)
            g.create_dataset("initial_condition", data=traj[0])

            bg = g.create_group("boundary_info")
            dg = bg.create_group("dirichlet")
            dg.create_dataset("indices", data=np.array([], dtype=np.int32))
            dg.create_dataset("values",  data=np.array([], dtype=np.float32))

    print(f"  ✓ 已保存 {out_file}  (共 {time.time()-t_gen:.1f}s)")
    return out_file


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="3D 激光淬火数据生成器（PhysHGNet 版）")
    p.add_argument("--n_nodes", type=int, default=5000,  help="目标节点数")
    p.add_argument("--all",     type=str, default=None,  help="逗号分隔的多个 N，如 2000,5000,10000")
    p.add_argument("--n_traj",  type=int, default=20,    help="轨迹数量")
    p.add_argument("--out_dir", type=str, default="data_laser_hardening_3d")
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()

    if args.all:
        for n in [int(x) for x in args.all.split(",")]:
            generate_for_n(n, args.n_traj, args.out_dir, seed=args.seed)
    else:
        generate_for_n(args.n_nodes, args.n_traj, args.out_dir, seed=args.seed)

    print("\n全部完成。")
