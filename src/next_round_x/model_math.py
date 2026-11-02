"""Classical conditional subset likelihood and exact elementary symmetric DP.

All-count partition: O(B M^2); leave-one-out marginals: O(B M^3).
No subset enumeration in training or inference. Ground-truth S is label-only.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def log_partitions(theta: torch.Tensor) -> torch.Tensor:
    if theta.ndim != 2 or not torch.isfinite(theta).all():
        raise ValueError("theta must be finite B x M")
    z=theta.new_zeros((theta.shape[0],1))
    for j in range(theta.shape[1]):
        t=theta[:,j:j+1]
        # Only structurally reachable states are stored. No logaddexp(-inf,
        # -inf), whose undefined gradient can silently poison autograd.
        middle=torch.logaddexp(z[:,1:],z[:,:-1]+t)
        z=torch.cat((z[:,:1],middle,z[:,-1:]+t),dim=1)
    return z


def validate_labels(y, theta):
    if y.shape != theta.shape or not torch.isfinite(y).all() or not ((y==0)|(y==1)).all():
        raise ValueError("complete binary labels of matching shape required")


def conditional_nll(theta, y):
    validate_labels(y,theta)
    centered=theta-theta.mean(dim=1,keepdim=True)
    s=y.sum(1).long()
    z=log_partitions(centered)
    loss=z.gather(1,s[:,None]).squeeze(1)-(y*centered).sum(1)
    return torch.where((s==0)|(s==theta.shape[1]),centered.sum(1)*0,loss)


def conditional_marginals(theta):
    theta=theta-theta.mean(1,keepdim=True)
    m=theta.shape[1]
    if m<1: raise ValueError("at least one expert required")
    z=log_partitions(theta)
    cols=[]
    for j in range(m):
        other=torch.cat((theta[:,:j],theta[:,j+1:]),1)
        minus=log_partitions(other)
        positive=(theta[:,j:j+1]+minus-z[:,1:]).exp()
        # Exact boundary has no gradients: conditioned on S=M all are present.
        positive=torch.cat((positive[:,:-1],torch.ones_like(positive[:,-1:])),1)
        cols.append(torch.cat((torch.zeros_like(positive[:,:1]),positive),1))
    return torch.stack(cols,2)  # B x (M+1) x M


def marginal_probabilities(theta, count_probabilities):
    pi=count_probabilities
    if pi.shape != (len(theta),theta.shape[1]+1) or not torch.isfinite(pi).all() or (pi<0).any():
        raise ValueError("invalid count probabilities")
    if not torch.allclose(pi.sum(1),torch.ones(len(pi),device=pi.device,dtype=pi.dtype),atol=1e-6,rtol=1e-6):
        raise ValueError("count distribution must normalize")
    return (conditional_marginals(theta)*pi[:,:,None]).sum(1)


def pair_transitions(theta,pi,anchor:int,candidate:int):
    if anchor==candidate:
        zero=theta[:,0]*0
        return zero,zero
    m=theta.shape[1]
    centered=theta-theta.mean(1,keepdim=True)
    z=log_partitions(centered)
    reduced=torch.stack([centered[:,j] for j in range(m) if j not in (anchor,candidate)],1) if m>2 else centered[:,:0]
    zz=log_partitions(reduced)
    both=(centered[:,anchor:anchor+1]+centered[:,candidate:candidate+1]+zz-z[:,2:]).exp()
    both=torch.cat((both[:,:-1],torch.ones_like(both[:,-1:])),1)
    both=(both*pi[:,2:]).sum(1)
    p=marginal_probabilities(centered,pi)
    return p[:,anchor]-both,p[:,candidate]-both  # joint harm, joint rescue


def small_head(dim,rank,out):
    return nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,rank),nn.Tanh(),nn.Linear(rank,out))


class ConditionalCountModel(nn.Module):
    def __init__(self,dim=2048,experts=17,rank=8):
        super().__init__()
        self.relative=small_head(dim,rank,experts)
        self.count=small_head(dim,rank,experts+1)

    def forward(self,x):
        theta=self.relative(x)
        theta=theta-theta.mean(1,keepdim=True)
        count_logits=self.count(x)
        return theta,count_logits

    def loss(self,x,y,*,joint_bce=False):
        theta,count_logits=self(x)
        validate_labels(y,theta)
        count_loss=F.cross_entropy(count_logits,y.sum(1).long())
        if joint_bce:
            # Same two networks and parameter count. The count distribution's
            # mean supplies a shared intercept; BCE jointly trains both heads.
            pi=count_logits.softmax(1)
            q=(pi*torch.arange(pi.shape[1],device=x.device)).sum(1)/theta.shape[1]
            intercept=torch.log(q)-torch.log1p(-q)
            relative_loss=F.binary_cross_entropy_with_logits(theta+intercept[:,None],y)
        else:
            relative_loss=conditional_nll(theta,y).mean()/theta.shape[1]
        return relative_loss+count_loss/theta.shape[1]

