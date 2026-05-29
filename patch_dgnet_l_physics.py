"""
patch_dgnet_l_physics.py

修复 dgnet_3d.py 中 _inject_physics 方法：
当 batch 已包含 L_physics（来自 dataset_3d）但缺少 N/diag/L_scale 时，
自动补全这些字段，使 StructuredDGNet._get_graph_cache 不会 KeyError。

运行：
    cd /home/caiziyue/.local/PhysHGNet3d
    python3 patch_dgnet_l_physics.py
"""
from pathlib import Path

OLD = '''    def _inject_physics(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """若 batch 中无 L_physics / node_volumes，则在线计算。"""
        need_L = (batch.get("L_physics") is None)
        need_V = (batch.get("node_volumes") is None)

        if not (need_L or need_V):
            return batch'''

NEW = '''    def _inject_physics(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """确保 L_physics 包含 StructuredDGNet 需要的所有字段（N/diag/L_scale）。"""
        import torch

        # 检查 L_physics 是否存在且完整
        lp = batch.get("L_physics")
        need_L = (lp is None)
        # 已有 L_physics 但缺少 N/diag/L_scale（来自 dataset_3d 的简化格式）
        need_upgrade = (
            lp is not None
            and isinstance(lp, dict)
            and lp.get("type") == "sparse"
            and ("N" not in lp or "diag" not in lp or "L_scale" not in lp)
        )
        need_V = (batch.get("node_volumes") is None)

        if not (need_L or need_upgrade or need_V):
            return batch

        batch = dict(batch)  # shallow copy
        nodes = batch["nodes"]
        N     = nodes.shape[0]

        if need_V:
            batch["node_volumes"] = torch.ones(N, device=nodes.device,
                                               dtype=nodes.dtype)

        if need_upgrade:
            # 补全 N / diag / L_scale
            lp   = dict(lp)  # shallow copy
            ei   = lp["edge_index"]
            ew   = lp["edge_weights"]
            # 对角 = 每行权重之和（L 是正半定）
            diag = torch.zeros(N, device=ei.device, dtype=ew.dtype)
            diag.scatter_add_(0, ei[0], ew.abs())
            L_scale = float(ew.abs().max().clamp(min=1.0).item())
            lp["N"]       = N
            lp["diag"]    = diag
            lp["L_scale"] = L_scale
            batch["L_physics"] = lp
            need_L = False  # 已处理，不用重新算

        if need_L:
            tets = batch.get("tets", None)'''

content = Path("dgnet_3d.py").read_text()
if OLD in content:
    Path("dgnet_3d.py").write_text(content.replace(OLD, NEW))
    print("✓ dgnet_3d.py 补丁应用成功")
else:
    print("⚠ 未找到目标代码，可能已修复或格式不同")
    print("  请手动在 _inject_physics 开头加入 L_physics 升级逻辑")
