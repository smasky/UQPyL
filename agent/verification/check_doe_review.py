"""DOE read-only structural/analytic audit, run with PYTHONPATH=."""
import argparse
import json
import warnings
from pathlib import Path
from itertools import product
import numpy as np
from scipy.stats import qmc
from UQPyL.doe import LHS, Random, FFD, Sobol, SaltelliDesign, FASTDesign, MorrisDesign
from UQPyL.problem import Problem
from UQPyL.analysis import Sobol as SobolAnalysis, FAST, Morris

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='agent/verification/1003-doe-review.json')
args = parser.parse_args()
records = []
def record(case, **values):
    records.append(dict(case=case, **values))
def problem(d=3):
    return Problem(nInput=d, nObj=1, lb=0., ub=1., objFunc=lambda x: (x @ np.arange(1,d+1))[:,None])
def attempt(case, fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value=fn()
            record(case, result=value, warnings=[str(w.message) for w in caught])
        except Exception as error:
            record(case, error=type(error).__name__, message=str(error), warnings=[str(w.message) for w in caught])

for d,n,seed in product([1,3,8],[2,16],[5,17,41]):
    p=problem(d)
    for criterion in ['classic','center','maximin','center_maximin','correlation']:
        sampler=LHS(criterion, iterations=5)
        x=sampler.sample(p,n,seed,output='unit')
        assert np.array_equal(np.sort(np.floor(x*n).astype(int),axis=0),np.tile(np.arange(n)[:,None],(1,d)))
        assert np.array_equal(x,sampler.sample(p,n,seed,output='unit'))
        if criterion in ['center','center_maximin']:
            np.testing.assert_allclose(np.sort(x,axis=0),(np.arange(n)[:,None]+.5)/n+np.zeros((1,d)),atol=1e-15)
        record('lhs_structure',dimension=d,n=n,seed=seed,criterion=criterion,passed=True)

for d,seed in product([1,3,8],[5,17,41]):
    p=problem(d)
    for sampler,n in [(Random(),256),(Sobol(),256),(SaltelliDesign(),256),(SaltelliDesign(secondOrder=True),256),(FASTDesign(),1025),(MorrisDesign(4),30),(MorrisDesign(8),30)]:
        before=np.random.get_state()
        x,meta=sampler.sampleWithMeta(p,n,seed=seed,output='unit')
        assert np.array_equal(x,sampler.sample(p,n,seed,output='unit'))
        after=np.random.get_state()
        assert all(np.array_equal(a,b) for a,b in zip(before,after))
        assert np.all(np.isfinite(x)) and np.all((x>=0)&(x<=1))
        if isinstance(sampler,SaltelliDesign):
            blocks=x.reshape(n,meta['blockSize'],d)
            for block in blocks:
                for j in range(d):
                    expected=block[0].copy()
                    expected[j]=block[-1,j]
                    np.testing.assert_array_equal(block[j+1],expected)
                    if sampler.secondOrder:
                        expected=block[-1].copy()
                        expected[j]=block[0,j]
                        np.testing.assert_array_equal(block[d+j+1],expected)
        if isinstance(sampler,FASTDesign):
            phaseRng=np.random.default_rng(seed)
            high=(n-1)//(2*sampler.M)
            maximum=high//(2*sampler.M)
            low=np.floor(np.linspace(1,maximum,d-1))
            for focus in range(d):
                phase=2*np.pi*phaseRng.random()
                frequencies=np.insert(low,focus,high)
                angle=np.arange(n)[:,None]*(2*np.pi/n)*frequencies+phase
                expected=1-np.abs(((angle+np.pi/2)%(2*np.pi))/np.pi-1)
                np.testing.assert_allclose(x[focus*n:(focus+1)*n],expected,atol=2e-12,rtol=0)
        if isinstance(sampler,MorrisDesign):
            delta=sampler.numLevels/(2*(sampler.numLevels-1))
            difference=np.diff(x.reshape(n,d+1,d),axis=1)
            active=np.abs(difference)>1e-12
            assert np.all(active.sum(axis=1)==1) and np.all(active.sum(axis=2)==1)
            np.testing.assert_allclose(np.abs(difference[active]),delta,atol=1e-14)
            np.testing.assert_allclose(x*(sampler.numLevels-1),np.round(x*(sampler.numLevels-1)),atol=1e-14)
        record('design_structure',method=type(sampler).__name__,dimension=d,seed=seed,passed=True)

p=problem()
for criterion in ['classic','center','maximin','center_maximin','correlation']:
    attempt('lhs_one_'+criterion,lambda c=criterion:LHS(c).sample(p,1,17).tolist())
for criterion,iterations in product(['maximin','center_maximin'],[0,-1]):
    attempt(f'lhs_iterations_{criterion}_{iterations}',lambda c=criterion,i=iterations:LHS(c,i).sample(p,4,17).tolist())
levels=[2,3,4]
x,meta=FFD().sampleWithMeta(p,levels)
np.testing.assert_array_equal(x,np.array(list(product(*[np.linspace(0,1,level) for level in levels]))))
levels[0]=9
record('ffd_metadata_alias',rows=len(x),metadata_levels=meta['levels'],metadata_implied_rows=int(np.prod(meta['levels'])))
for cls in [Sobol,SaltelliDesign]:
    attempt(cls.__name__+'_skip_exceeds_count',lambda cls=cls:cls(scramble=False,skipValue=32).sample(p,16).shape)
    with warnings.catch_warnings(record=True) as caught:
        x=cls(scramble=False,skipValue=4).sample(p,16,output='unit')
    base=x if cls is Sobol else x.reshape(16,5,3)[:,0]
    counts=np.bincount(np.floor(base[:,0]*16).astype(int),minlength=16)
    record(cls.__name__+'_unaligned_skip',bin_counts=counts.tolist(),warnings=[str(w.message) for w in caught])
engine=qmc.Sobol(3,scramble=False)
engine.fast_forward(32)
x=engine.random(16)
record('scipy_skip_control',rows=len(x),bin_counts=np.bincount((x[:,0]*16).astype(int),minlength=16).tolist())

mixed=Problem(nInput=4,nObj=1,lb=[-3,-2,0,7],ub=[9,3,1,7],varType=[0,1,2,0],varSet={2:[.1,2.5,8.]},objFunc=lambda x:np.zeros((len(x),1)))
for sampler,n in [(LHS(),120),(Random(),120),(Sobol(),128),(SaltelliDesign(),128),(FASTDesign(),257),(MorrisDesign(),20),(FFD(),[3,6,3,1])]:
    x,meta=sampler.sampleWithMeta(mixed,n,seed=17)
    assert np.all((x[:,0]>=-3)&(x[:,0]<=9))
    assert np.all(np.isin(x[:,1],np.arange(-2,4))) and np.all(np.isin(x[:,2],[.1,2.5,8.])) and np.all(x[:,3]==7)
    record('mixed_mapping',method=type(sampler).__name__,rows=len(x),passed=True)

for seed in [5,17,41]:
    for design,analysis,n in [(SaltelliDesign(secondOrder=True),SobolAnalysis,4096),(FASTDesign(),FAST,4097),(MorrisDesign(),Morris,100)]:
        x,meta=design.sampleWithMeta(p,n,seed=seed)
        result=analysis(verboseFlag=False,saveFlag=False).analyze(p,x,p.evaluate(x).objs,meta=meta)
        metrics={m.name:np.asarray(m.values).tolist() for m in result.metrics}
        record('linear_end_to_end',method=analysis.__name__,seed=seed,metrics=metrics,reference_sensitivity=[1/14,4/14,9/14],reference_morris=[1.,2.,3.])
# Nonlinear interaction: y=x0*x1, independent U(0,1) inputs.
pProduct=Problem(nInput=3,nObj=1,lb=0.,ub=1.,objFunc=lambda x:(x[:,0]*x[:,1])[:,None])
for seed in [5,17,41]:
    for design,analysis,n in [(SaltelliDesign(secondOrder=True),SobolAnalysis,4096),(FASTDesign(),FAST,4097)]:
        x,meta=design.sampleWithMeta(pProduct,n,seed=seed)
        result=analysis(verboseFlag=False,saveFlag=False).analyze(pProduct,x,pProduct.evaluate(x).objs,meta=meta)
        metrics={m.name:np.asarray(m.values).tolist() for m in result.metrics}
        error=max(float(np.max(np.abs(np.asarray(metrics[k])-reference))) for k,reference in [('S1',[3/7,3/7,0]),('ST',[4/7,4/7,0])])
        assert error<.015
        record('product_end_to_end',method=analysis.__name__,seed=seed,metrics=metrics,max_abs_error=error)
for scramble,skip in product([False,True],[0,16]):
    sampler=Sobol(scramble=scramble,skipValue=skip)
    x=sampler.sample(problem(),16,17,output='unit')
    seed=np.random.default_rng(17).integers(1,1000000) if scramble else None
    reference=qmc.Sobol(3,scramble=scramble,seed=seed)
    if skip:
        reference.fast_forward(skip)
    np.testing.assert_array_equal(x,reference.random(16))
    record('scipy_sequence_reference',scramble=scramble,skip=skip,passed=True)
for row in records:
    if row['case']=='linear_end_to_end':
        if row['method']=='Morris':
            np.testing.assert_allclose(row['metrics']['mu_star'],[[1,2,3]],atol=1e-13)
        else:
            for key in ['S1','ST']:
                np.testing.assert_allclose(row['metrics'][key],[[1/14,4/14,9/14]],rtol=0,atol=.002)

Path(args.output).write_text(json.dumps(records,indent=2))
print('records',len(records))
for row in records:
    if row['case'] not in ['lhs_structure','design_structure','mixed_mapping']:
        print(row)
