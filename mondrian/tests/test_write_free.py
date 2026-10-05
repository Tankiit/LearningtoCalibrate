import numpy as np
import pytest
from mondrian.splits import make_splits,assert_disjoint,freeze_splits,split_hash
from mondrian.exemplars import select_exemplars
from mondrian.cells import assign_voronoi,assign_power,assign_topk,assign_with_residual,fit_power_weights
from mondrian.conformal import per_cell_quantile
from mondrian.audit import write_free_check
from models.partition import ExemplarPartition


def test_splits_disjoint(tmp_path):
    groups=np.repeat(np.arange(30),3)
    splits=make_splits(len(groups),seed=17,groups=groups)
    assert_disjoint(splits,n=len(groups),groups=groups)
    assert all(not v.flags.writeable for v in splits.values())
    repeated=make_splits(len(groups),seed=17,groups=groups)
    assert all(np.array_equal(splits[k],repeated[k]) for k in splits)
    with pytest.raises(ValueError):
        assert_disjoint({'fit':np.array([0]),'cal':np.array([0]),'test':np.array([1])})
    freeze_splits(tmp_path/'split.json',splits,'data-A')
    freeze_splits(tmp_path/'split.json',splits,'data-A')
    with pytest.raises(ValueError):
        freeze_splits(tmp_path/'split.json',splits,'data-B')
    assert split_hash(np.array([1,2],dtype='>i8'))==split_hash(np.array([1,2],dtype='<i8'))


@pytest.mark.parametrize('method',['uniform','kmeans','voronoi'])
def test_exemplars_from_fit_only(method):
    X=np.random.default_rng(1).normal(size=(100,3))
    idx,meta=select_exemplars(X[:40],8,method,np.random.default_rng(4))
    assert np.all(idx<40) and len(set(idx))==8 and meta['source']=='D_fit'
    part=ExemplarPartition.fit(X[:40],8,method,4)
    E=part.exemplars.copy()
    X[:40]=999  # fitted rule owns a copy, not a live view
    np.testing.assert_array_equal(E,part.exemplars)
    assert not part.exemplars.flags.writeable


def test_power_weights_use_fit_only():
    X=np.random.default_rng(4).uniform(size=(120,2))
    a=ExemplarPartition.fit(X[:80],4,seed=9,rule='power')
    before=a.weights.copy()
    a(X[80:]);a(np.full((30,2),1e6))
    np.testing.assert_array_equal(before,a.weights)
    X[80:]=999
    b=ExemplarPartition.fit(X[:80],4,seed=9,rule='power')
    np.testing.assert_array_equal(a.weights,b.weights)
    assert a.metadata['power_fit']['source']=='D_fit'


def test_assignment_batch_invariant():
    rng=np.random.default_rng(5)
    X=rng.normal(size=(30,2));other=rng.normal(30,4,size=(50,2));E=X[:4]
    w=fit_power_weights(X,E)
    for rule in (lambda x:assign_voronoi(x,E),lambda x:assign_power(x,E,w),
                 lambda x:assign_topk(x,E,2),lambda x:assign_with_residual(x,E,np.ones(4))):
        assert write_free_check(rule,X,other)
    with pytest.raises(AssertionError):
        write_free_check(lambda x:(x[:,0]>x[:,0].mean()).astype(int),X,other)


def test_floor_rule_same_across_arms():
    X=np.arange(60,dtype=float).reshape(-1,1);E=X[[5,30,55]]
    for labels in (assign_voronoi(X,E),assign_power(X,E,np.array([0.,100.,-100.]))):
        q,n=per_cell_quantile(labels,np.arange(60)/60,3,.1,25)
        assert np.all(np.isinf(q[n<25]))
        assert n.sum()==60
    q,n=per_cell_quantile(np.array([0,0]),np.array([.1,.2]),3,.1,20)
    assert np.isinf(q).all() and n.tolist()==[2,0,0]
