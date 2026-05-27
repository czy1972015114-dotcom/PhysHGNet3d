"""train_dgnet_3d.py — DGNet 3D 训练（v4，与 train_phys_hgnet_3d.py 逻辑一致）"""
import argparse, json, os, time
from pathlib import Path
from typing import Dict, Any
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dgnet_3d    import DGNet3D, DEFAULT_CONFIG_DGNET3D
from dataset_3d  import LaserHardening3DDataset, collate_fn_3d, find_h5_file

def setup_ddp():
    if "RANK" not in os.environ: return 0,0,1
    dist.init_process_group(backend="nccl")
    rank,local_rank,world_size=dist.get_rank(),int(os.environ.get("LOCAL_RANK",0)),dist.get_world_size()
    torch.cuda.set_device(local_rank); return rank,local_rank,world_size

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()

def batch_to_device(batch, device):
    out={}
    for k,v in batch.items():
        if isinstance(v,torch.Tensor): out[k]=v.to(device)
        elif isinstance(v,dict): out[k]={k2:v2.to(device) if isinstance(v2,torch.Tensor) else v2 for k2,v2 in v.items()}
        else: out[k]=v
    return out

def zero_nan_gradients(model):
    n=0
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad.data=torch.nan_to_num(p.grad.data,nan=0.,posinf=0.,neginf=0.); n+=1
    return n

def all_reduce_metrics(metrics, device, world_size):
    if world_size<=1: return metrics
    keys=list(metrics.keys())
    t=torch.tensor([metrics[k] for k in keys],dtype=torch.float64,device=device)
    dist.all_reduce(t,op=dist.ReduceOp.SUM); t/=world_size
    return {k:float(t[i]) for i,k in enumerate(keys)}

def run_epoch(model,loader,optimizer,device,is_train,scaler=None,epoch=0,rank=0,world_size=1):
    model.train() if is_train else model.eval()
    sum_mse=sum_rne=0.; n_ok=n_nan=n_ng=0
    ctx=torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch=batch_to_device(batch,device); tgt=batch["targets"].to(device)
            with torch.amp.autocast('cuda',enabled=scaler is not None):
                out=model(batch); pred=out["u_final"]
                if not torch.isfinite(pred).all().item():
                    n_nan+=1; pred=torch.nan_to_num(pred,nan=298.15,posinf=2000.,neginf=0.)
                mse=F.mse_loss(pred,tgt); rne=(pred-tgt).norm()/tgt.norm().clamp(min=1e-8)
                loss=mse+0.01*F.relu(-pred).mean()
            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    if zero_nan_gradients(model)>0: n_ng+=1
                    nn.utils.clip_grad_norm_(model.parameters(),0.5)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    if zero_nan_gradients(model)>0: n_ng+=1
                    nn.utils.clip_grad_norm_(model.parameters(),0.5); optimizer.step()
            sum_mse+=mse.item(); sum_rne+=rne.item(); n_ok+=1
    agg=all_reduce_metrics({"mse":sum_mse,"rne":sum_rne,"n_ok":float(n_ok),"n_nan":float(n_nan),"n_ng":float(n_ng)},device,world_size)
    d=max(int(agg["n_ok"]),1)/world_size
    return {"mse":agg["mse"]/d,"rne":agg["rne"]/d,"nan_batches":int(agg["n_nan"]),"nan_grads":int(agg["n_ng"])}

def train(args):
    rank,local_rank,world_size=setup_ddp(); is_main=(rank==0)
    device=torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main: print(f"=== DGNet3D 训练 | N≈{args.n_nodes} | world_size={world_size} ===")
    h5_path=find_h5_file(args.data_dir,args.n_nodes)
    full_ds=LaserHardening3DDataset(h5_path,n_time_steps=args.n_time_steps); N=full_ds.N
    n_total=len(full_ds); n_train=int(n_total*0.8); n_val=n_total-n_train
    gen=torch.Generator().manual_seed(args.seed)
    train_ds,val_ds=torch.utils.data.random_split(full_ds,[n_train,n_val],generator=gen)
    if world_size>1:
        tr_s=DistributedSampler(train_ds,num_replicas=world_size,rank=rank,shuffle=True)
        va_s=DistributedSampler(val_ds,  num_replicas=world_size,rank=rank,shuffle=False)
        tr_sh=False
    else: tr_s=va_s=None; tr_sh=True
    tr_loader=DataLoader(train_ds,batch_size=args.batch_size,sampler=tr_s,shuffle=tr_sh,
                         collate_fn=collate_fn_3d,num_workers=args.num_workers,drop_last=True)
    va_loader=DataLoader(val_ds,  batch_size=args.batch_size,sampler=va_s,shuffle=False,
                         collate_fn=collate_fn_3d,num_workers=args.num_workers)
    if is_main: print(full_ds.info()); print(f"  训练={n_train} | 验证={n_val} | batch={args.batch_size}")
    model_cfg={**DEFAULT_CONFIG_DGNET3D,"residual_hidden_dim":args.hidden_dim,"residual_num_layers":args.n_layers}
    model=DGNet3D(model_cfg).to(device)
    if world_size>1: model=DDP(model,device_ids=[local_rank],find_unused_parameters=True)
    raw_model=model.module if world_size>1 else model
    if is_main: print(raw_model.extra_info())
    optimizer=optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler('cuda') if torch.cuda.is_available() and args.amp else None
    ckpt_dir=Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True,exist_ok=True)
    ckpt_path=ckpt_dir/f"best_{N}.pth"
    start_epoch=0; best_rne=float('inf')
    if ckpt_path.exists() and args.resume:
        ckpt=torch.load(ckpt_path,map_location=device)
        raw_model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"]); start_epoch=ckpt["epoch"]+1; best_rne=ckpt["best_rne"]
        if is_main: print(f"  恢复 epoch={start_epoch}, best_rne={best_rne:.4f}")
    log=[]
    for epoch in range(start_epoch,args.epochs):
        if world_size>1: tr_s.set_epoch(epoch)
        t0=time.time()
        tr_m=run_epoch(model,tr_loader,optimizer,device,is_train=True, scaler=scaler,epoch=epoch,rank=rank,world_size=world_size)
        va_m=run_epoch(model,va_loader,None,      device,is_train=False,epoch=epoch,rank=rank,world_size=world_size)
        scheduler.step()
        if is_main:
            extra=""
            if tr_m["nan_batches"]>0: extra+=f"  [NaN→clamp:{tr_m['nan_batches']}]"
            if tr_m["nan_grads"]>0:   extra+=f"  [NaN梯度:{tr_m['nan_grads']}]"
            print(f"Epoch {epoch+1:4d}/{args.epochs} | train MSE={tr_m['mse']:.4f} RNE={tr_m['rne']:.4f} | val MSE={va_m['mse']:.4f} RNE={va_m['rne']:.4f} | {time.time()-t0:.1f}s{extra}")
            log.append({"epoch":epoch+1,"train_mse":tr_m["mse"],"train_rne":tr_m["rne"],"val_mse":va_m["mse"],"val_rne":va_m["rne"]})
            cur=va_m["rne"]
            if 0<cur<best_rne:
                best_rne=cur
                torch.save({"model":raw_model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"best_rne":best_rne,"config":model_cfg},ckpt_path)
                print(f"  ✓ 保存最优 (RNE={best_rne:.4f})")
    if is_main:
        with open(ckpt_dir/f"train_log_{N}.json","w") as f: json.dump(log,f,indent=2)
        print(f"\n训练完成！best RNE={best_rne:.4f}  检查点:{ckpt_path}")
    cleanup_ddp()

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--n_nodes",type=int,default=5000); p.add_argument("--data_dir",type=str,default="data_laser_hardening_3d")
    p.add_argument("--ckpt_dir",type=str,default="checkpoints/dgnet_3d"); p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch_size",type=int,default=4); p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--n_time_steps",type=int,default=20); p.add_argument("--hidden_dim",type=int,default=128)
    p.add_argument("--n_layers",type=int,default=5); p.add_argument("--num_workers",type=int,default=0)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--amp",action="store_true"); p.add_argument("--resume",action="store_true")
    return p.parse_args()

if __name__=="__main__": train(parse_args())
