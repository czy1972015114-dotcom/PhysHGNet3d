"""
plot_anchor_density_comparison.py
对比三种采样策略在激光附近的锚点密度：
  1. PhysHGNet anchor_selector（学到的）
  2. FPS（最远点采样，空间均匀基线）
  3. Random（随机采样基线）
横轴：到激光位置的距离 r
纵轴：距激光 r 以内的锚点占比（CDF）

如果 PhysHGNet 曲线在小 r 处高于 FPS/Random，
说明它确实把更多锚点放在了物理上重要的区域。
"""
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import h5py
sys.path.insert(0, '.')

from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D
from dataset_3d import LaserHardening3DDataset, collate_fn_3d, find_h5_file
from torch.utils.data import DataLoader


def to_device(x, device):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, dict): return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(to_device(v, device) for v in x)
    return x


def fps(coords, k):
    N = len(coords)
    k = min(k, N)
    sel = [np.random.randint(N)]
    dist = np.full(N, np.inf)
    for _ in range(k - 1):
        d = ((coords - coords[sel[-1]]) ** 2).sum(1)
        dist = np.minimum(dist, d)
        sel.append(int(np.argmax(dist)))
    return np.array(sel)


def cdf_vs_radius(coords, anchor_idx, laser_pos, r_vals):
    """锚点到激光的距离 CDF"""
    d = np.linalg.norm(coords[anchor_idx] - laser_pos, axis=1)
    return np.array([(d <= r).mean() for r in r_vals])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_nodes",  type=int, default=4000)
    p.add_argument("--data_dir", type=str, default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/phys_hgnet_3d")
    p.add_argument("--traj_idx", type=int, default=0)
    p.add_argument("--n_timesteps", type=int, default=8,
                   help="均匀取多少个时间步做平均")
    p.add_argument("--out",      type=str,
                   default="results_viz/anchors/anchor_density_comparison.png")
    p.add_argument("--gpu",      type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── 加载模型 ──────────────────────────────────────────────────
    import os; os.makedirs("results_viz/anchors", exist_ok=True)
    ckpt_path = f"{args.ckpt_dir}/best_{args.n_nodes}.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = {**DEFAULT_CONFIG_3D, **ckpt.get("config", {})}
    model = PhysHGNet3D(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    m_anchors = cfg.get("m_anchors", 250)
    print(f"模型加载完成  m_anchors={m_anchors}")

    # ── 读数据 ────────────────────────────────────────────────────
    h5_path = str(find_h5_file(args.data_dir, args.n_nodes))
    with h5py.File(h5_path, "r") as f:
        coords    = f["mesh_meta"]["nodes"][:]              # (N, 3)
        traj_key  = f"trajectory_{args.traj_idx}"
        src_all   = f[traj_key]["source_terms"][:]          # (T, N, 1)
        T_total   = src_all.shape[0]

    # 均匀取时间步（跳过 t=0，激光未启动）
    timesteps = list(np.linspace(1, T_total - 1,
                                  args.n_timesteps, dtype=int))
    print(f"使用时间步: {[t+1 for t in timesteps]}")

    # 半径范围（以工件尺寸为基准）
    domain_size = np.linalg.norm(coords.max(0) - coords.min(0))
    r_vals = np.linspace(0, domain_size * 0.3, 200)

    # ── 注册 hook ─────────────────────────────────────────────────
    captured = []
    def hook_fn(module, inputs, output):
        if isinstance(output, torch.Tensor) and output.dtype == torch.int64:
            captured.clear(); captured.append(output.detach().cpu().numpy().flatten())

    for name, mod in model.named_modules():
        if "anchor_selector" in name.lower():
            mod.register_forward_hook(hook_fn)
            break

    ds = LaserHardening3DDataset(h5_path, window_size=10, stride=1,
                                  traj_indices=[args.traj_idx])
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn_3d, num_workers=0)
    all_samples = list(loader)

    # ── 逐时间步收集 CDF ─────────────────────────────────────────
    cdfs_model  = []
    cdfs_fps    = []
    cdfs_random = []
    laser_moves = []

    np.random.seed(42)
    for t in timesteps:
        # 激光位置（从 H5 读）
        q = src_all[t, :, 0]
        if q.max() == 0: continue
        laser_pos = coords[int(np.argmax(q))]
        laser_moves.append(laser_pos)

        # 清缓存
        raw = model.module if hasattr(model, "module") else model
        for attr in ("_graph_cache", "_cache_key",
                     "_residual_cache", "_grad_norm_cache"):
            if hasattr(raw, attr): setattr(raw, attr, None)

        # 模型推断，捕获锚点
        sample_idx = min(t, len(all_samples) - 1)
        batch = to_device(all_samples[sample_idx], device)
        captured.clear()
        with torch.no_grad():
            model(batch)

        if not captured: continue
        anchor_idx_model = captured[0].astype(int)
        anchor_idx_model = anchor_idx_model[anchor_idx_model < len(coords)]

        # FPS 基线（每次重新算，保证公平）
        anchor_idx_fps = fps(coords, m_anchors)

        # Random 基线
        anchor_idx_rnd = np.random.choice(len(coords), m_anchors, replace=False)

        cdfs_model.append( cdf_vs_radius(coords, anchor_idx_model,  laser_pos, r_vals))
        cdfs_fps.append(   cdf_vs_radius(coords, anchor_idx_fps,    laser_pos, r_vals))
        cdfs_random.append(cdf_vs_radius(coords, anchor_idx_rnd,    laser_pos, r_vals))

        d_to_laser = np.linalg.norm(
            coords[anchor_idx_model] - laser_pos, axis=1).min()
        print(f"  t={t+1}: 最近锚点距激光={d_to_laser:.4f}  "
              f"域尺寸={domain_size:.3f}")

    if not cdfs_model:
        print("✗ 未收集到有效数据"); return

    # 平均 CDF
    cdf_m = np.mean(cdfs_model, axis=0)
    cdf_f = np.mean(cdfs_fps,   axis=0)
    cdf_r = np.mean(cdfs_random,axis=0)

    # ── 绘图 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              facecolor="white", dpi=150)
    fig.suptitle(
        f"Anchor Distribution Quality  (N={args.n_nodes}, M={m_anchors})\n"
        "Higher CDF at small radius = more anchors near the laser (physics-focused)",
        fontsize=11, fontweight="bold")

    # 左图：平均 CDF 曲线
    ax = axes[0]
    r_norm = r_vals / domain_size   # 归一化为域尺寸的比例
    ax.plot(r_norm, cdf_m, "-",  color="#E74C3C", lw=2.5,
             label=f"PhysHGNet (learned)")
    ax.plot(r_norm, cdf_f, "--", color="#3498DB", lw=2.0,
             label="FPS (spatial uniform)")
    ax.plot(r_norm, cdf_r, ":",  color="#95A5A6", lw=2.0,
             label="Random")
    ax.axvline(0.05, color="gray", ls="--", lw=1, alpha=0.5)
    ax.text(0.05, 0.02, "r=5%", fontsize=8, color="gray", ha="center")
    ax.set_xlabel("Distance to laser / domain size", fontsize=11)
    ax.set_ylabel("Fraction of anchors within radius r", fontsize=11)
    ax.set_title("Cumulative anchor density near laser\n(averaged over timesteps)",
                  fontsize=10)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    # 右图：各时间步激光路径 + 锚点密度热力图（XY 投影）
    ax2 = axes[1]
    if laser_moves:
        lm = np.array(laser_moves)
        ax2.plot(lm[:, 0], lm[:, 1], "g-o", lw=2, ms=6,
                  label="Laser path", zorder=10)

    # 用最后一个时间步的模型锚点画密度
    sample_idx = min(timesteps[-1], len(all_samples) - 1)
    batch = to_device(all_samples[sample_idx], device)
    raw = model.module if hasattr(model, "module") else model
    for attr in ("_graph_cache", "_cache_key"):
        if hasattr(raw, attr): setattr(raw, attr, None)
    captured.clear()
    with torch.no_grad(): model(batch)
    if captured:
        aidx = captured[0].astype(int)
        aidx = aidx[aidx < len(coords)]
        ax2.scatter(coords[:, 0], coords[:, 1],
                    s=0.5, c="#C8D6E5", alpha=0.2, rasterized=True)
        ax2.scatter(coords[aidx, 0], coords[aidx, 1],
                    s=20, c="#E74C3C", alpha=0.8, zorder=5,
                    label=f"Anchors (t={timesteps[-1]+1})")
        if laser_moves:
            lp = laser_moves[-1]
            ax2.scatter(lp[0], lp[1], s=200, marker="*",
                        c="#00CC66", zorder=15, edgecolors="black",
                        label="Laser (final)")

    ax2.set_aspect("equal")
    ax2.set_title(f"Anchor positions at t={timesteps[-1]+1} (XY view)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.axis("off")
    ax2.set_facecolor("#F7F9FB")

    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"\n✓ 密度对比图已保存: {args.out}")

    # 打印数值汇总
    print(f"\n在 r=5% 域尺寸内的锚点占比（越高越好）：")
    idx5 = np.argmin(np.abs(r_norm - 0.05))
    print(f"  PhysHGNet : {cdf_m[idx5]*100:.1f}%")
    print(f"  FPS       : {cdf_f[idx5]*100:.1f}%")
    print(f"  Random    : {cdf_r[idx5]*100:.1f}%")

if __name__ == "__main__":
    main()
