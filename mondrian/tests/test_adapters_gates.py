import json
import numpy as np
import pytest
from mondrian.audit import require_gate,write_json,source_hash,manifest
from data.llm_cache import load_cache,one_row_per_question
from models.scores import ScoreModel
from data.tabular import load_communities


def test_gates_fail_closed(tmp_path):
    path=tmp_path/'gate.json'
    write_json(path,{'stage':'x1','status':'HOLD','source_sha256':source_hash()})
    with pytest.raises(RuntimeError):require_gate(path,'x1')
    write_json(path,{'stage':'x1','status':'PASS','source_sha256':'stale'})
    with pytest.raises(RuntimeError):require_gate(path,'x1')
    write_json(path,{'stage':'x1','status':'PASS','source_sha256':source_hash()})
    assert require_gate(path,'x1')['status']=='PASS'
    with pytest.raises(ValueError):manifest(M=1)


def test_strict_cache_and_question_unit(tmp_path):
    meta=dict(schema_version=1,model='test',dataset='test',layer_pool='last',representation='residual_stream',basis_id='one')
    (tmp_path/'meta.json').write_text(json.dumps(meta))
    np.savez(tmp_path/'cache.npz',X=np.arange(24).reshape(12,2),correct=np.tile([0,1],6),question_id=np.repeat(['a','b','c','d'],3))
    X,y,q=load_cache(tmp_path,'test','test','last')
    Z,v,ids,idx=one_row_per_question(X,y,q)
    assert len(ids)==len(set(ids))==4
    assert np.array_equal(Z,X[idx])
    with pytest.raises(ValueError):load_cache(tmp_path,'test','test','mean')
    meta['schema_version']=2;(tmp_path/'meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):load_cache(tmp_path,'test','test','last')


def test_score_preprocessing_is_fit_only():
    X=np.array([[1.,np.nan],[2.,1.],[3.,3.],[4.,2.]])
    score=ScoreModel().fit(X,np.array([1.,2.,3.,4.]))
    before=score.features(X).copy()
    score.features(np.array([[1e9,-1e9]]))
    np.testing.assert_array_equal(before,score.features(X))
    assert np.isfinite(before).all()


def test_communities_schema_excludes_identifiers_and_target(tmp_path):
    path=tmp_path/'communities.data'
    rows=[]
    for i in range(6):
        rows.append(','.join(['1','?','?','Town','1']+['0.2']*122+['0.7']))
    path.write_text('\n'.join(rows))
    X,y,g,cluster,meta=load_communities(path)
    assert X.shape==(6,122) and np.all(y==.7) and cluster is None
    assert np.all(X==.2) and meta['given_group']=='state'


def test_x4_interval_bins_frozen_and_answerable_closed_form():
    from experiments.x4_selection import IntervalBins, sample
    rng=np.random.default_rng(0)
    b=IntervalBins(rng.uniform(size=500),5)
    v=rng.uniform(size=50)
    assert np.array_equal(b(v),np.concatenate([b(x[None]) for x in v]))
    X,a,Y,S,V=sample(20000,0.,np.random.default_rng(1))
    assert np.all(S[Y==1]<=1) and np.all(S[Y==0]>=1)


def test_e_llm_forecast_and_label_scores():
    from experiments.e_llm import binned_forecast, label_scores
    cc=np.array([0]*30+[1]*5); yc=np.array([1]*15+[0]*15+[1]*5,float)
    pt,rate,n,below=binned_forecast(cc,yc,np.array([0,1]),2)
    assert rate[0]==.5 and below[1] and rate[1]==yc.mean()
    p=np.array([.9,.2])
    assert np.allclose(label_scores(p,np.array([1,0])),[.1,.2])
