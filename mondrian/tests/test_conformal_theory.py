import numpy as np
import pytest
from mondrian.conformal import conformal_rank,per_cell_quantile,coverage_by_cell,functionals
from mondrian.theory import beta_law_cell_moments,Mstar,optimal_power_balance
from experiments.x1_synthetic import decompose,exponent_test,_smoothed_argmin,gate_decision
from experiments.topk import pooled_quantiles
from mondrian.cells import assign_topk,assign_with_residual,fit_power_weights,assign_power,assign_voronoi


def test_finite_sample_rank_and_infinity():
    assert conformal_rank(9,.1)==9
    assert conformal_rank(19,.05)==19
    q,_=per_cell_quantile(np.zeros(9,int),np.arange(9.),1,.1,0)
    assert q[0]==8
    q,_=per_cell_quantile(np.zeros(8,int),np.arange(8.),1,.1,0)
    assert np.isinf(q[0])
    with pytest.raises(ValueError):
        per_cell_quantile(np.array([0]),np.array([np.nan]),1,.1,0)


def test_coverage_and_empty_cells():
    c,n=coverage_by_cell(np.array([0,0,1]),np.array([1.,3.,9.]),np.array([2.,np.inf,1.]))
    np.testing.assert_allclose(c[:2],[.5,1.])
    assert np.isnan(c[2]) and n.tolist()==[2,1,0]
    f=functionals(c,n,.1)
    assert f['coverage']==pytest.approx(2/3)
    assert f['msce']==pytest.approx((2*.4**2+.1**2)/3)
    assert f['n_empty_test_cells']==1


def test_beta_law_against_independent_order_statistics():
    rng=np.random.default_rng(99)
    n,alpha=49,.1
    expected=beta_law_cell_moments(n,alpha)
    samples=rng.uniform(size=(30000,n))
    coverage=np.partition(samples,expected['rank']-1,axis=1)[:,expected['rank']-1]
    assert abs(coverage.mean()-expected['mean'])<5*np.sqrt(expected['variance']/len(samples))
    assert abs(coverage.var()-expected['variance'])<.04*expected['variance']
    empty=beta_law_cell_moments(0,alpha)
    assert empty['mean']==1 and empty['variance']==0
    assert beta_law_cell_moments(100,.1,floor=101)['mean']==1


def test_orthogonal_decomposition():
    c=np.array([0,0,1,1]);p=np.array([.2,.8,.6,1.])
    a,b,total,_,_=decompose(p,c,2,.9)
    assert total==pytest.approx(a+b,abs=1e-14)
    # Cell means hide nonzero within-cell error even under exact average coverage.
    a,b,total,_,_=decompose(np.array([.8,1.]),np.array([0,0]),1,.9)
    assert b==0 and total==pytest.approx(.01)


def test_Mstar_derivative_and_unsupported_functionals():
    for d in (1,2,4):
        m=Mstar(1000,d,'pointwise_msce',A=2,B=.09)
        derivative=-(2/d)*2*m**(-2/d-1)+.09/1000
        assert abs(derivative)<1e-12
    assert Mstar(8000,1,'pointwise_msce')/Mstar(1000,1,'pointwise_msce')==pytest.approx(2)
    for f in ('msce','spread','concentrated'):
        with pytest.raises(ValueError):
            Mstar(1000,2,f)


def test_topk_deduplicates_and_k1_matches_partition():
    E=np.array([[0.],[1.]])
    X=np.linspace(0,1,50)[:,None];scores=np.arange(50.)
    cal=assign_topk(X,E,1);test=assign_topk(X,E,1)
    q,n=pooled_quantiles(cal,scores,test,.1,0)
    expected,counts=per_cell_quantile(cal[:,0],scores,2,.1,0)
    np.testing.assert_array_equal(q,expected[test[:,0]])
    np.testing.assert_array_equal(n,counts[test[:,0]])
    both=np.tile([0,1],(50,1))
    _,n=pooled_quantiles(both,scores,both[:2],.1,0)
    assert n.tolist()==[50,50]  # not 100 duplicated score rows


def test_residual_uses_nearest_eligible_not_nearest_ineligible():
    E=np.array([[0.],[3.]])
    assert assign_with_residual(np.array([[1.],[10.]]),E,np.array([.1,3.])).tolist()==[1,2]


def test_power_dual_improves_mass_balance():
    X=np.linspace(0,1,1001)[:,None];E=np.array([[.05],[.2],[.9]])
    before=np.max(np.abs(np.bincount(assign_voronoi(X,E),minlength=3)/len(X)-1/3))
    w,info=fit_power_weights(X,E,return_diagnostics=True)
    after=np.max(np.abs(np.bincount(assign_power(X,E,w),minlength=3)/len(X)-1/3))
    assert after<before/5 and after==pytest.approx(info['max_mass_error'])


def test_paired_bootstrap_preserves_functional_identity():
    rows=[]
    for seed in range(4):
        for n,best in ((100,2),(400,4),(1600,8)):
            for m in (1,2,4,8,16):
                value=(np.log2(m/best))**2+seed*.001
                rows.append(dict(seed=seed,n_cal=n,M=m,msce=value,spread=value,
                                 concentrated=value,pointwise_msce=value))
    result=exponent_test(rows,n_boot=50)
    assert result['functionals']['msce']['slope']==pytest.approx(.5)
    for v in result['paired_slope_differences'].values():
        assert v['estimate']==0 and v['ci95']==[0,0]


def test_smoothed_argmin_recovers_vertex_and_flags_boundary():
    log_m=np.log([1,2,3,4,5,6,8,11,14,18])
    v,b=_smoothed_argmin((log_m-np.log(5.5))**2,log_m)
    assert v==pytest.approx(np.log(5.5)) and not b
    v,b=_smoothed_argmin(-log_m,log_m)
    assert b


def test_gate_decision_three_outcomes():
    ok=dict(interior=True,mechanism=True,b0_argmin_M1=True,b0_beta=True,decomposition=True)
    assert gate_decision(ok,.05,.2)=='PASS'
    assert gate_decision({**ok,'mechanism':False},.25,.2)=='INCONCLUSIVE'
    assert gate_decision({**ok,'mechanism':False},.05,.2)=='FAIL'
    assert gate_decision({**ok,'mechanism':False,'interior':False},.25,.2)=='FAIL'
