"""Bounded frozen-feature source heads; the old R3 policy and head are reused."""
from __future__ import annotations
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans
from .runtime import ROOT, digest, canonical, write_json, read_json, record
from .model_math import ConditionalCountModel, small_head, marginal_probabilities
from routers.bench_coe_query.heads import AuditedBenchCoEHead

DEFAULT_TRAIN=dict(batch_size=256,epochs=30,min_epochs=6,patience=5,learning_rate=.0003,weight_decay=.001,dropout=.15,threads=4)


class MIRTAdapted(nn.Module):
    def __init__(self,dim=2048,experts=17,rank=8):
        super().__init__()
        self.discrimination=small_head(dim,rank,rank)
        self.ability=nn.Parameter(torch.randn(rank,experts)/rank**.5)
        self.expert_bias=nn.Parameter(torch.zeros(experts))
        self.difficulty=small_head(dim,rank,1)
    def forward(self,x):
        return self.discrimination(x)@self.ability+self.expert_bias-self.difficulty(x)


def make_model(kind,rank=8):
    if kind in ("csc","joint_bce"):
        return ConditionalCountModel(rank=rank)
    if kind=="mirt": return MIRTAdapted(rank=rank)
    if kind in ("bce17","anchor_hard","cost_soft"):
        return AuditedBenchCoEHead(2048,17,.15)
    raise ValueError("unknown model "+kind)


def strongest_labels(y,accuracy):
    labels=np.where(y>0,accuracy[None,:],-1).argmax(1)
    labels[y.sum(1)==0]=-100
    return labels.astype(np.int64)


def fit_head(data,train_indices,path,*,protocol,kind,seed,rank=8,config=None):
    cfg=dict(DEFAULT_TRAIN,**(config or {}))
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    indices=np.asarray(train_indices,dtype=int)
    uids=data["uids"][indices].tolist()
    identity=dict(protocol=protocol,kind=kind,seed=seed,rank=rank,config=cfg,fit_uid_hash=canonical(uids),
        source_hashes={name:digest(ROOT/"src/next_round_x"/name) for name in ("models.py","model_math.py")})
    meta_path=path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        meta=read_json(meta_path)
        if meta["identity"] != identity or digest(path)!=meta["checkpoint"]["sha256"]:
            raise ValueError("existing checkpoint identity changed: "+str(path))
        model=make_model(kind,rank)
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        return model,meta
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Checkpoint selection has its own group holdout, never the outer OOF fold.
    stop_mask=np.array([int(canonical({"group":str(g),"seed":seed,"purpose":"early_stop"})[:8],16)%10==0 for g in data["groups"][indices]])
    avail=data["available"][indices]
    fit=indices[~stop_mask & avail]
    stop=indices[stop_mask & avail]
    if len(stop)<32 or len(fit)<64: raise ValueError("insufficient independent inner stop data")
    assert not set(data["groups"][fit]) & set(data["groups"][stop])
    mean_y=data["y"][fit].mean(0)
    hard=strongest_labels(data["y"],mean_y)
    xfit=torch.from_numpy(np.asarray(data["x"][fit],dtype=np.float32).copy())
    yfit=torch.from_numpy(data["y"][fit].copy())
    hfit=torch.from_numpy(hard[fit])
    xstop=torch.from_numpy(np.asarray(data["x"][stop],dtype=np.float32).copy())
    ystop=torch.from_numpy(data["y"][stop].copy())
    hstop=torch.from_numpy(hard[stop])
    kappa=float(np.median(data["c"][fit][data["c"][fit]>0]))
    weight=np.exp(-.1*data["c"][fit]/kappa)*data["y"][fit]
    soft=torch.from_numpy((weight/np.maximum(weight.sum(1,keepdims=True),1e-30)).astype(np.float32))
    model=make_model(kind,rank)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    generator=torch.Generator().manual_seed(seed)
    best=float("inf");best_epoch=0;bad=0;state=None;history=[]
    started=time.perf_counter();cpu_started=time.process_time()
    for epoch in range(1,cfg["epochs"]+1):
        model.train();perm=torch.randperm(len(fit),generator=generator)
        loss_sum=0.;seen=0
        for local in perm.split(cfg["batch_size"]):
            x=xfit[local];y=yfit[local]
            opt.zero_grad(set_to_none=True)
            if kind in ("csc","joint_bce"):
                loss=model.loss(x,y,joint_bce=kind=="joint_bce")
            else:
                logits=model(x)
                if kind=="anchor_hard":
                    valid=hfit[local]!=-100
                    if not valid.any(): continue
                    loss=F.cross_entropy(logits[valid],hfit[local][valid])
                elif kind=="cost_soft":
                    valid=soft[local].sum(1)>0
                    if not valid.any(): continue
                    loss=-(soft[local][valid]*F.log_softmax(logits[valid],dim=1)).sum(1).mean()
                else: loss=F.binary_cross_entropy_with_logits(logits,y)
            if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
            loss.backward();opt.step()
            loss_sum+=loss.item()*len(local);seen+=len(local)
        model.eval()
        with torch.no_grad():
            if kind in ("csc","joint_bce"):
                # Same held-out joint objective for the two matched structures.
                # Full marginals only at the saved checkpoint, not every epoch.
                values=[]
                for ii in torch.arange(len(stop)).split(512):
                    theta,cl=model(xstop[ii]); q=(cl.softmax(1)*torch.arange(18)).sum(1)/17
                    logits=theta+(torch.log(q)-torch.log1p(-q))[:,None]
                    values.append((F.binary_cross_entropy_with_logits(logits,ystop[ii])+F.cross_entropy(cl,ystop[ii].sum(1).long())/17).item()*len(ii))
                monitor=sum(values)/len(stop)
            else:
                logits=torch.cat([model(part) for part in xstop.split(1024)])
                valid=hstop!=-100
                monitor=F.cross_entropy(logits[valid],hstop[valid]).item()
        history.append(dict(epoch=epoch,train_loss=loss_sum/max(seen,1),inner_stop_loss=monitor))
        if monitor<best-1e-6:
            best=monitor;best_epoch=epoch;bad=0;state=copy.deepcopy(model.state_dict())
        else: bad+=1
        if epoch>=cfg["min_epochs"] and bad>=cfg["patience"]: break
    if state is None: raise RuntimeError("no finite checkpoint")
    model.load_state_dict(state);model.eval()
    torch.save(state,path)
    meta=dict(identity=identity,protocol=protocol,fitted=True,fit_uids=uids,
        gradient_fit_uids=data["uids"][fit].tolist(),inner_selection_uids=data["uids"][stop].tolist(),
        ancestors=[dict(kind="generic_frozen_features",fitted=False)],checkpoint=record(path),
        count=len(indices),gradient_fit_n=len(fit),selection_n=len(stop),parameters=sum(p.numel() for p in model.parameters()),
        epochs_run=len(history),best_epoch=best_epoch,history=history,wall_seconds=time.perf_counter()-started,
        cpu_seconds=time.process_time()-cpu_started,gpu_seconds=0,router_dollar_conversion=None,
        monitor="inner held-out BCE+count CE" if kind in ("csc","joint_bce") else "inner dev_hard_label_loss (legacy R3 criterion)")
    write_json(meta_path,meta)
    print(f"trained {protocol} {path.parent.name} {kind} rank={rank} seed={seed} epochs={len(history)} best={best_epoch} seconds={meta['wall_seconds']:.1f}",flush=True)
    return model,meta


def predict_head(model,x,*,kind):
    model.eval();out=[];thetas=[];pis=[]
    started=time.perf_counter();cpu_started=time.process_time()
    with torch.no_grad():
        for start in range(0,len(x),1024):
            batch=torch.from_numpy(np.array(x[start:start+1024],dtype=np.float32))
            if kind in ("csc","joint_bce"):
                theta,cl=model(batch);pi=cl.softmax(1)
                if kind=="csc": p=marginal_probabilities(theta.double(),pi.double())
                else:
                    q=(pi*torch.arange(18)).sum(1)/17
                    p=(theta+(q.log()-torch.log1p(-q))[:,None]).sigmoid()
                thetas.append(theta.numpy());pis.append(pi.numpy());out.append(p.numpy())
            else:
                logits=model(batch)
                out.append((logits.softmax(1) if kind in ("anchor_hard","cost_soft") else logits.sigmoid()).numpy())
    result=dict(p=np.concatenate(out),wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started)
    if thetas: result.update(theta=np.concatenate(thetas),pi=np.concatenate(pis))
    return result


def normalized(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def predict_knn(xtrain,ytrain,xquery,k=32):
    xt=normalized(xtrain);xq=normalized(xquery);out=[]
    k=min(k,len(xt))
    for start in range(0,len(xq),128):
        sim=xq[start:start+128]@xt.T
        near=np.argpartition(-sim,k-1,axis=1)[:,:k]
        out.append(ytrain[near].mean(1))
    return np.concatenate(out)


def cluster_quality(xtrain,ytrain,xquery,k,seed=0):
    km=MiniBatchKMeans(n_clusters=k,random_state=seed,n_init=3,batch_size=1024,max_iter=100)
    labels=km.fit_predict(normalized(xtrain))
    global_y=ytrain.mean(0)
    rates=np.array([(ytrain[labels==j].sum(0)+64*global_y)/((labels==j).sum()+64) for j in range(k)])
    return rates[km.predict(normalized(xquery))],km,rates
