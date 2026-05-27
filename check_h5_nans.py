#!/usr/bin/env python3
import sys
import h5py
import numpy as np
from pathlib import Path

p = sys.argv[1] if len(sys.argv) > 1 else "data_laser_hardening_3d/pde_trajectories_3d_N1000.h5"
path = Path(p)
if not path.exists():
    print(f"File not found: {p}")
    sys.exit(2)

bad = []
with h5py.File(path, 'r') as f:
    # check mesh_meta
    if 'mesh_meta' in f:
        for name in ['nodes','edges','faces','tets','node_volumes','time_points','L_physics_ei','L_physics_ew']:
            if name in f['mesh_meta']:
                arr = f['mesh_meta'][name][:]
                if np.isnan(arr).any() or np.isinf(arr).any():
                    bad.append(('mesh_meta', name))
    # check trajectories
    traj_keys = sorted([k for k in f.keys() if k.startswith('trajectory_')], key=lambda x: int(x.split('_')[1]))
    for k in traj_keys:
        g = f[k]
        for name in ['node_features','source_terms','initial_condition']:
            if name in g:
                arr = g[name][:]
                if np.isnan(arr).any() or np.isinf(arr).any():
                    bad.append((k, name))

if not bad:
    print('No NaN/Inf found in HDF5 file.')
else:
    print('Found NaN/Inf in the following datasets:')
    for item in bad:
        print(' -', item[0], '/', item[1])

# Print basic stats for a few trajectories for quick inspection
with h5py.File(path, 'r') as f:
    traj_keys = sorted([k for k in f.keys() if k.startswith('trajectory_')], key=lambda x: int(x.split('_')[1]))
    for k in traj_keys[:8]:
        g = f[k]
        if 'node_features' in g:
            arr = g['node_features'][:]
            print(f"{k} node_features: shape={arr.shape}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}")
        if 'source_terms' in g:
            arr = g['source_terms'][:]
            print(f"{k} source_terms: shape={arr.shape}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}")
        if 'initial_condition' in g:
            arr = g['initial_condition'][:]
            print(f"{k} initial_condition: shape={arr.shape}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}")

print('Done.')
