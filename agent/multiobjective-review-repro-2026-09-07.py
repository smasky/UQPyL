import json, warnings
from types import SimpleNamespace
import numpy as np
from UQPyL.optimization import Population
from UQPyL.optimization.runtime.result import OptState
from UQPyL.optimization.core import NDSort
from UQPyL.optimization.moea import MOEAD, RVEA
from UQPyL.optimization.metric import HV
from UQPyL.problem import Problem
quiet=dict(verboseFlag=False,logFlag=False,saveFlag=False)
def pop(objs,cons):
    return Population(np.arange(len(objs)).reshape(-1,1),objs,cons)
s=OptState(SimpleNamespace())
a=s._updateMulti(pop([[1,1]],[[1]]),1,0)
b=s._updateMulti(pop([[10,10]],[[0]]),2,1)
print('first_feasible',dict(improved=b,ref=s.hvRefPoint.tolist(),hv=s.bestMetric,feasible=s.bestFeasible,appearFEs=s.appearFEs))
s=OptState(SimpleNamespace())
s._updateMulti(pop([[-10,-10]],[[0]]),1,0)
print('negative_objectives',dict(ref=s.hvRefPoint.tolist(),hv=s.bestMetric))
p=pop([[1,1],[2,2],[3,3]],[[1],[2],[3]])
print('infeasible_pareto',p.getParetoFront().cons.ravel().tolist())
print('equal_cv_ranks',NDSort([[0,2],[1,1],[2,0]],[[1],[1],[1]])[0].tolist())
# Fixed raw reference, strictly improved singleton can have smaller normalized HV.
print('hv_order',[(x,HV([[x,x]],refPoint=[10,10]),HV([[x,x]],refPoint=[10,10],normalize=False)) for x in [-1,-2]])
for agg in ['PBI','TCH_N']:
    p=Problem(nInput=1,nObj=2,nCon=1,lb=0,ub=1,objFunc=lambda X: np.zeros((len(X),2)),conFunc=lambda X: -np.ones((len(X),1)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r=MOEAD(aggregation=agg,nPop=20,maxFEs=40,maxIters=1,**quiet).run(p,seed=3)
    print('moead',agg,'warnings',len(w),sorted(set(str(x.message) for x in w)))
# Independent non-dominated sorting for finite unconstrained inputs.
rng=np.random.default_rng(7)
for trial in range(200):
    objs=rng.integers(-3,4,size=(20,3))
    remaining=list(range(len(objs))); expected=np.zeros(len(objs)); level=0
    while remaining:
        level+=1
        front=[i for i in remaining if not any(np.all(objs[j]<=objs[i]) and np.any(objs[j]<objs[i]) for j in remaining)]
        expected[front]=level; remaining=[i for i in remaining if i not in front]
    np.testing.assert_array_equal(NDSort(objs)[0],expected)
print('independent_unconstrained_sort: 200 passed')
p=Problem(nInput=1,nObj=2,nCon=1,lb=0,ub=1,objFunc=lambda X: np.column_stack((X[:,0],1-X[:,0])),conFunc=lambda X: X-.5)
r=MOEAD(nPop=8,maxFEs=24,maxIters=1,**quiet).run(p,seed=3)
print('moead_small_population',dict(FEs=r.FEs,historyFEs=r.history.iterToFEs))
v=np.array([[1.,0.],[.5,.5],[0.,1.]])
rv=RVEA(**quiet)
newV=rv.updateReferenceVector(np.array([[1.,1.]]),v)
print('rvea_single_feasible_vectors',newV.tolist())
print('rvea_zero_vectors_selection',rv.environmentSelection(np.array([[0.,2.],[1.,1.],[2.,0.]]),newV,.5, np.zeros((3,1))).tolist())
from UQPyL.optimization.moea import NSGAII, NSGAIII
for trial in range(100):
    objs=rng.random((20,3)); cons=rng.random((20,2))-.5
    weights=np.array([10.,1.]); cv=np.maximum(cons*weights,0).sum(axis=1)
    ranks,_=NDSort(objs,cons,conWgt=weights)
    for i in range(20):
        for j in range(20):
            if cv[i]<cv[j]: assert ranks[i]<ranks[j]
print('weighted_feasibility_order: 100 passed')
