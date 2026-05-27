#!/usr/bin/env python3
import sys
import torch
from pathlib import Path

from dataset_3d import LaserHardening3DDataset, collate_fn_3d, find_h5_file
from phys_hgnet_3d import PhysHGNet3D, DEFAULT_CONFIG_3D


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {k2: (v2.to(device) if isinstance(v2, torch.Tensor) else v2)
                      for k2, v2 in v.items()}
        else:
            out[k] = v
    return out


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: inspect_forward.py <h5_path> <traj_key_or_index>')
        sys.exit(2)
    h5 = sys.argv[1]
    key = sys.argv[2]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ds = LaserHardening3DDataset(h5, n_time_steps=20)
    print(ds.info())

    # locate index
    if key.startswith('trajectory_'):
        try:
            idx = ds.traj_keys.index(key)
        except ValueError:
            print('traj_key not found:', key)
            sys.exit(2)
    else:
        idx = int(key)

    sample = ds[idx]
    batch = collate_fn_3d([sample])
    batch = to_device(batch, device)

    # build model
    cfg = {**DEFAULT_CONFIG_3D}
    model = PhysHGNet3D(cfg).to(device)
    model.eval()

    nodes = batch['nodes']
    edges = batch['edges']
    node_type = batch.get('node_type', None)
    L_physics = batch.get('L_physics', None)
    u_init = batch['initial_conditions'] if 'initial_conditions' in batch else None

    # build graph cache
    with torch.no_grad():
        gc = model._build_graph(nodes, edges, L_physics,
                                node_volumes=batch.get('node_volumes', None),
                                node_type=batch.get('node_type', None),
                                u_init=u_init[0] if u_init is not None else None)
    print('Graph cache keys:', list(gc.keys()))
    print('anchor count m=', gc['m'])

    # anchor feats
    with torch.no_grad():
        nv = batch.get('node_volumes')
        if nv is None:
            nv = torch.ones(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
        anchor_feats = model.fine_encoder(nodes, gc['fine_ei'], gc['fine_ea'], nv, node_type)[gc['anchor_idx']]
    print('anchor_feats finite:', torch.isfinite(anchor_feats).all().item(), 'shape', anchor_feats.shape)

    # compute L_hat
    L_hat = model.learnable_coarse(gc['anchor_coords'], anchor_feats, gc['coarse_ei'], use_learned_coarse=True)
    print('L_hat dtype:', L_hat.dtype, 'shape:', tuple(L_hat.shape))
    print('L_hat finite:', torch.isfinite(L_hat).all().item())
    print('L_hat stats: min', float(L_hat.min().item()), 'max', float(L_hat.max().item()), 'abs_max', float(L_hat.abs().max().item()), 'norm', float(L_hat.norm().item()))

    # Leff matvec on initial condition (pass anchor_feats, not L_hat)
    u_curr = batch['initial_conditions']
    u0 = u_curr[0, :, 0]
    with torch.no_grad():
        Lu = model._Leff_matvec(u0, gc, anchor_feats)
    print('Lu finite:', torch.isfinite(Lu).all().item())
    print('Lu stats: min', float(Lu.min().item()), 'max', float(Lu.max().item()), 'norm', float(Lu.norm().item()))

    # print alpha
    try:
        print('alpha_loc', float(model.alpha_loc.item()), 'alpha_coarse_op', float(model.alpha_coarse_op.item()))
    except Exception:
        pass

    print('Done')
