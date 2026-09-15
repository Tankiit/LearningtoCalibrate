"""Finite report families: observed decisions, paired contrasts and CDF envelopes."""
import numpy as np


def decisions(values, threshold):
    a=np.asarray(values,dtype=float)
    if a.ndim!=2:raise ValueError('Expected question by legend matrix')
    valid=np.isfinite(a).all(1)
    lower=np.min(a,axis=1);upper=np.max(a,axis=1)
    return dict(valid=valid,lower=lower,upper=upper,
                all_accept=valid&(lower>=threshold),all_reject=valid&(upper<threshold),
                legend_dependent=valid&(lower<threshold)&(upper>=threshold),invalid=~valid)


def paired_ordering(positive,negative,tolerance=0.):
    p=np.asarray(positive,dtype=float);n=np.asarray(negative,dtype=float)
    if p.shape!=n.shape or p.ndim!=2:raise ValueError('Paired matrices must have identical shape')
    d=p-n;valid=np.isfinite(d).all(1);z=np.where(abs(d)<=tolerance,0.,d)
    lower=z.min(1);upper=z.max(1)
    categories=np.full(len(p),'invalid',dtype=object)
    categories[valid&(lower>0)]='consistently_correct'
    categories[valid&(upper<0)]='consistently_incorrect'
    categories[valid&(lower<0)&(upper>0)]='sign_change'
    categories[valid&(lower==0)&(upper==0)]='tie_all'
    categories[valid&(lower==0)&(upper>0)]='tie_and_correct'
    categories[valid&(lower<0)&(upper==0)]='tie_and_incorrect'
    # Cartesian contrast interval deliberately forgets the within-legend pairing.
    cart_lower=p.min(1)-n.max(1);cart_upper=p.max(1)-n.min(1)
    return dict(contrasts=d,lower=d.min(1),upper=d.max(1),category=categories,
                cartesian_lower=cart_lower,cartesian_upper=cart_upper)


def pbox(probabilities,grid,aggregate):
    """aggregate is rsuq.core.beliefs.cluster_prob_mass; no DS mass reinterpretation."""
    import torch
    p=np.asarray(probabilities,dtype=float);v=np.asarray(grid,dtype=float)
    if p.ndim!=3 or p.shape[-1]!=len(v) or np.any(np.diff(v)<=0):raise ValueError('Invalid aligned family')
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-6):raise ValueError('Invalid distributions')
    # Remove saved floating-point normalization residuals for valid CDF endpoints.
    p=p/p.sum(-1,keepdims=True);t=torch.as_tensor(p,dtype=torch.float64)
    cdf=np.stack([aggregate(t,torch.tensor((v>x).astype(np.int64)),2)[...,0].numpy() for x in v],axis=-1)
    lower=cdf.min(1);upper=cdf.max(1)
    return dict(aligned=p,cdf=cdf,cdf_lower=lower,cdf_upper=upper,
                family_mean_lower=(p@v).min(1),family_mean_upper=(p@v).max(1),
                pbox_mean_lower=v[0]+((1-upper[:,:-1])*np.diff(v)).sum(1),
                pbox_mean_upper=v[0]+((1-lower[:,:-1])*np.diff(v)).sum(1))
