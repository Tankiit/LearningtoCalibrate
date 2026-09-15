"""Fixed-answer DINCO arithmetic, matching the public TriviaQA runner.

Candidate 0 is the supplied answer, never a newly selected generation. NLI order
is entailment/neutral/contradiction. Distractor diagonal entailment is 1.
Auxiliary candidates/NLI/SC must be cached once and reused across report formats.
"""
import numpy as np

def scores(confidences,nli,self_consistency):
    p=np.asarray(confidences,float);nli=np.asarray(nli,float)
    if p.ndim!=1 or nli.shape!=(len(p),len(p),3):raise ValueError('Candidate/NLI shape mismatch')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid emission: retain invalid estimator status, do not impute')
    if not np.isfinite(nli).all() or np.any((nli<0)|(nli>1)) or not np.allclose(nli.sum(-1),1):raise ValueError('Invalid NLI distribution')
    if not 0<=self_consistency<=1:raise ValueError('Invalid SC')
    entail=nli[:,:,0].copy();np.fill_diagonal(entail,1.)
    # Degree excludes the supplied answer, includes each distractor's self-edge.
    degree=entail[1:,1:].sum(0)
    weights=.5*(nli[0,1:,2]+nli[1:,0,2])/degree
    denominator=max(1.,float(p[0]+p[1:]@weights))
    nvc=float(p[0]/denominator)
    return dict(raw=float(p[0]),normalized=nvc,self_consistency=float(self_consistency),dinco=.5*(nvc+self_consistency),normalizer=denominator,distractor_weights=weights.tolist())

def sc_from_entailment(bidirectional_entailment,exact_matches):
    a=np.asarray(bidirectional_entailment,float);exact=np.asarray(exact_matches,bool)
    if a.shape!=(len(exact),2) or not len(exact):raise ValueError('SC sample dimensions')
    return float(np.where(exact,1.,a.mean(1)).mean())

def confidence_dispersion_score(values):
    """SteerConf confidence factor with answer fixed (answer agreement=1).

Applying this to legend variants is an aggregation-rule adaptation, not a
reproduction of its cautious/confident steering prompt intervention.
"""
    values=np.asarray(values,float);mu=values.mean();sd=values.std()
    return float(mu*mu/(mu+sd)) if mu+sd>0 else 0.
