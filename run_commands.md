# PhysHGNet3D 实验运行命令大全

所有命令默认在项目根目录 `/home/caiziyue/.local/PhysHGNet3d` 下执行。

---

## 0. 准备工作：接口修复

在开始任何实验前，先应用以下两个修复：

```bash
cd /home/caiziyue/.local/PhysHGNet3d

# 修复 1：inspect_forward.py（n_time_steps → window_size）
cp inspect_forward.py inspect_forward.py.bak
# 用仓库里新版的 inspect_forward.py 覆盖

# 修复 2：dgnet_3d.py L_physics 字段补全
python3 patch_dgnet_l_physics.py

# 修复 3：将 visual_ancher_temperal.py 放入项目根目录
cp visual_ancher_temperal.py .
```

---

## 1. 数据生成

```bash
# N=1000（快速测试用）
python generate_laser_data_3d.py --n_nodes 1000 --n_traj 40 \
    --out_dir data_laser_hardening_3d

# N=2000
python generate_laser_data_3d.py --n_nodes 2000 --n_traj 40 \
    --out_dir data_laser_hardening_3d

# N=4000（主力规模）
python generate_laser_data_3d.py --n_nodes 4000 --n_traj 40 \
    --out_dir data_laser_hardening_3d

# N=6000（大规模）
python generate_laser_data_3d.py --n_nodes 6000 --n_traj 40 \
    --out_dir data_laser_hardening_3d

# 验证数据完整性
python check_h5_nans.py data_laser_hardening_3d/pde_trajectories_3d_N6000.h5
```

---

## 2. 实验一：Scaling 对比（DGNet3D vs PhysHGNet3D）

每个 N 值分别训练 DGNet3D 和 PhysHGNet3D，观察随 N 增大时的性能变化。

### 2a. DGNet3D 训练（baseline）

```bash
# N=1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
  --nproc_per_node=7 --master_port=29701 \
  train_dgnet_3d.py --n_nodes 1000 --epochs 100 \
  --ckpt_dir checkpoints/dgnet_3d

# N=2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
  --nproc_per_node=7 --master_port=29701 \
  train_dgnet_3d.py --n_nodes 2000 --epochs 100 \
  --ckpt_dir checkpoints/dgnet_3d

# N=4000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
  --nproc_per_node=7 --master_port=29701 \
  train_dgnet_3d.py --n_nodes 4000 --epochs 100 \
  --ckpt_dir checkpoints/dgnet_3d

# N=6000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
  --nproc_per_node=7 --master_port=29701 \
  train_dgnet_3d.py --n_nodes 6000 --epochs 40 \
  --ckpt_dir checkpoints/dgnet_3d
```

### 2b. PhysHGNet3D 训练（完整模型）

```bash
# N=1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
  --nproc_per_node=7 --master_port=29700 \
  train_phys_hgnet_3d.py --n_nodes 1000 --epochs 100 \
  --ckpt_dir checkpoints/phys_hgnet_3d

# N=2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
  --nproc_per_node=7 --master_port=29700 \
  train_phys_hgnet_3d.py --n_nodes 2000 --epochs 100 \
  --ckpt_dir checkpoints/phys_hgnet_3d

# N=4000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
  --nproc_per_node=7 --master_port=29700 \
  train_phys_hgnet_3d.py --n_nodes 4000 --epochs 100 \
  --ckpt_dir checkpoints/phys_hgnet_3d

# N=6000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
  --nproc_per_node=7 --master_port=29700 \
  train_phys_hgnet_3d.py --n_nodes 6000 --epochs 40 \
  --ckpt_dir checkpoints/phys_hgnet_3d
```

### 2c. Scaling 评测对比

```bash
# 对每个 N 分别调用 compare_3d.py
for N in 1000 2000 4000 6000; do
  echo "=== N=$N ==="
  python compare_3d.py \
    --n_nodes $N \
    --phys_ckpt checkpoints/phys_hgnet_3d/best_${N}.pth \
    --dg_ckpt   checkpoints/dgnet_3d/best_${N}.pth \
    --data_dir  data_laser_hardening_3d
done
```

---

## 3. 实验二：PhysHGNet 消融实验

固定 N=4000，逐一关闭创新，分析每项贡献。

### 3a. 训练消融变体

```bash
N=4000
GPUS="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 --master_port=29700"

# Full（全部打开，与 scaling 实验共用）
# 已在 checkpoints/phys_hgnet_3d/best_4000.pth

# w/o C1（关闭物理感知锚点）
$GPUS train_phys_hgnet_3d.py --n_nodes $N --epochs 100 \
  --no_physics_anchor \
  --ckpt_dir checkpoints/phys_hgnet_3d_no_c1

# w/o C2（关闭可学习粗算子）
$GPUS train_phys_hgnet_3d.py --n_nodes $N --epochs 100 \
  --no_learned_coarse \
  --ckpt_dir checkpoints/phys_hgnet_3d_no_c2

# w/o C3（关闭双尺度 GNN → 退化为单尺度 MPNN）
$GPUS train_phys_hgnet_3d.py --n_nodes $N --epochs 100 \
  --no_dual_scale \
  --ckpt_dir checkpoints/phys_hgnet_3d_no_c3

# w/o VN（关闭虚拟节点，C3 保留）
$GPUS train_phys_hgnet_3d.py --n_nodes $N --epochs 100 \
  --no_virtual_nodes \
  --ckpt_dir checkpoints/phys_hgnet_3d_no_vn

# w/o C1+C2（只剩 C3）
$GPUS train_phys_hgnet_3d.py --n_nodes $N --epochs 100 \
  --no_physics_anchor --no_learned_coarse \
  --ckpt_dir checkpoints/phys_hgnet_3d_no_c1c2

# w/o C1+C3（只剩 C2）
$GPUS train_phys_hgnet_3d.py --n_nodes $N --epochs 100 \
  --no_physics_anchor --no_dual_scale \
  --ckpt_dir checkpoints/phys_hgnet_3d_no_c1c3

# w/o C2+C3（只剩 C1）
$GPUS train_phys_hgnet_3d.py --n_nodes $N --epochs 100 \
  --no_learned_coarse --no_dual_scale \
  --ckpt_dir checkpoints/phys_hgnet_3d_no_c2c3
```

### 3b. 消融评测（每个变体 vs DGNet3D baseline）

```bash
N=4000
for variant in "" "_no_c1" "_no_c2" "_no_c3" "_no_vn" "_no_c1c2" "_no_c1c3" "_no_c2c3"; do
  ckpt="checkpoints/phys_hgnet_3d${variant}/best_${N}.pth"
  [ -f "$ckpt" ] || { echo "跳过 $ckpt（不存在）"; continue; }
  echo "=== variant=${variant:-full} ==="
  python compare_3d.py \
    --n_nodes $N \
    --phys_ckpt "$ckpt" \
    --dg_ckpt   checkpoints/dgnet_3d/best_${N}.pth \
    --data_dir  data_laser_hardening_3d \
    --label     "PhysHGNet${variant}"
done
```

---

## 4. 实验三：锚点时序可视化

### 4a. 基础用法（默认每 5 步采样一帧）

```bash
python visual_ancher_temperal.py \
  --ckpt   checkpoints/phys_hgnet_3d/best_6000.pth \
  --h5     data_laser_hardening_3d/pde_trajectories_3d_N6000.h5 \
  --traj_idx 32 \
  --out    figs/anchor_temporal_N6000
```

### 4b. 高密度采样（每步一帧，适合论文展示）

```bash
python visual_ancher_temperal.py \
  --ckpt   checkpoints/phys_hgnet_3d/best_6000.pth \
  --h5     data_laser_hardening_3d/pde_trajectories_3d_N6000.h5 \
  --traj_idx 32 \
  --t_start 0 --t_end 60 --t_step 2 \
  --colorby residual \
  --fps 6 \
  --out figs/anchor_temporal_N6000_dense
```

### 4c. 按热源强度着色（展示锚点跟随激光）

```bash
python visual_ancher_temperal.py \
  --ckpt   checkpoints/phys_hgnet_3d/best_6000.pth \
  --h5     data_laser_hardening_3d/pde_trajectories_3d_N6000.h5 \
  --traj_idx 33 \
  --colorby source \
  --elev 30 --azim -45 \
  --out figs/anchor_source_N6000
```

### 4d. 对不同 N 生成可视化，观察锚点覆盖率

```bash
for N in 1000 4000 6000; do
  python visual_ancher_temperal.py \
    --ckpt   checkpoints/phys_hgnet_3d/best_${N}.pth \
    --h5     data_laser_hardening_3d/pde_trajectories_3d_N${N}.h5 \
    --traj_idx $((N/200)) \
    --t_step 5 \
    --out    figs/anchor_temporal_N${N}
done
```

---

## 5. 调试与检查工具

```bash
# 检查单条轨迹的前向传播是否正常
python inspect_forward.py \
  data_laser_hardening_3d/pde_trajectories_3d_N4000.h5 \
  0

# 检查 h5 文件是否有 NaN
python check_h5_nans.py \
  data_laser_hardening_3d/pde_trajectories_3d_N6000.h5

# 打开调试模式（打印 L_hat / Lv 数值统计）
PHGNET_DEBUG=1 python inspect_forward.py \
  data_laser_hardening_3d/pde_trajectories_3d_N4000.h5 1
```

---

## 6. 额外建议实验

### 实验四：训练收敛速度对比（可选）

直接用训练日志文件绘图，不需要重新训练：

```bash
python - << 'EOF'
import json, glob
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for path in sorted(glob.glob("checkpoints/*/train_log_4000.json")):
    name = path.split("/")[-2].replace("phys_hgnet_3d", "PhysHGNet").replace("dgnet_3d", "DGNet")
    d = json.load(open(path))
    ep  = [x["epoch"]     for x in d]
    rne = [x["val_rne"]   for x in d]
    mse = [x["val_mse"]   for x in d]
    axes[0].plot(ep, rne, label=name)
    axes[1].plot(ep, mse, label=name)

for ax, title, ylabel in zip(axes,
    ["Val RNE (N=4000)", "Val MSE (N=4000)"],
    ["RNE", "MSE"]):
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title);    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figs/convergence_N4000.png", dpi=150)
print("→ figs/convergence_N4000.png")
EOF
```

### 实验五：锚点权重演化（α/β/γ 训练过程）

训练时 PhysHGNet3D 会自动记录 anchor_weights（通过 ablation_summary），
加上以下代码片段到 `train_phys_hgnet_3d.py` 的 epoch 结束处：

```python
# 在 epoch 循环末尾加入（is_main 保护下）
if is_main:
    w = raw_model.anchor_selector.weight_summary()
    log[-1]["anchor_weights"] = w
    tqdm.write(f"  锚点权重: {w}")
```

---

## 7. 结果汇总表脚本

训练和评测全部完成后，运行此脚本生成汇总：

```bash
python - << 'EOF'
import json, os, glob
from pathlib import Path

results = {}
for log in sorted(glob.glob("checkpoints/*/train_log_*.json")):
    model = Path(log).parent.name
    N_str = Path(log).stem.split("_")[-1]
    try:
        N = int(N_str)
        d = json.load(open(log))
        best_rne = min(x["val_rne"] for x in d)
        best_ep  = min(d, key=lambda x: x["val_rne"])["epoch"]
        results.setdefault(N, {})[model] = {
            "best_val_rne": round(best_rne, 5),
            "best_epoch"  : best_ep,
        }
    except Exception as e:
        print(f"跳过 {log}: {e}")

print("\n=== Scaling 结果汇总 ===")
print(f"{'N':>6}  {'Model':<30}  {'Best Val RNE':>14}  {'Best Ep':>8}")
print("-" * 65)
for N in sorted(results.keys()):
    for model, stats in sorted(results[N].items()):
        tag = "★" if "phys_hgnet_3d" == model else " "
        print(f"{N:>6}  {tag} {model:<28}  {stats['best_val_rne']:>14.5f}  {stats['best_epoch']:>8}")
EOF
```

---

## 快速参考：关键超参

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `--n_nodes` | 网格节点数 | 1000/2000/4000/6000 |
| `--epochs` | 训练轮数 | N≤4000: 100，N=6000: 40 |
| `--batch_size` | 批大小 | 2（每 GPU） |
| `--window_size` | 时间窗口步数 | 10 |
| `--stride` | 窗口滑动步长 | 5（默认 window//2） |
| `--graph_rebuild_freq` | 图重建频率（步） | 20 |
| `--m_anchors` | 锚点数（PhysHGNet） | 自动 N//12 |
| `--m_anchors` | 锚点数（DGNet） | 自动 max(32,min(64,N//32)) |
