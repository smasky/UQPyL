"""Read-only public Problem/ModelProblem evaluation and conversion audit."""
import argparse
import json
import warnings
from pathlib import Path
from itertools import product
import numpy as np
from UQPyL.problem import Problem, ModelProblem, singleFunc
from UQPyL.doe import LHS

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='agent/verification/1003-problem-review.json')
args=parser.parse_args()
records=[]
def record(case, **values):
    records.append(dict(case=case,**values))
def capture(case, callback):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value=callback()
            record(case,value=value,warnings=[str(w.message) for w in caught])
        except Exception as error:
            record(case,error=type(error).__name__,message=str(error),warnings=[str(w.message) for w in caught])
def basic(**kw):
    args=dict(nInput=2,nObj=1,lb=[0.,10.],ub=[1.,20.],objFunc=lambda x:x.sum(axis=1)[:,None])
    args.update(kw)
    return Problem(**args)

for rows,target,direction in product([1,2,7],[None,'objs','cons'],['min','max',['min','max']]):
    x=np.arange(rows*2,dtype=float).reshape(rows,2)/10
    calls=[]
    def objective(x):
        calls.append('obj')
        return np.column_stack([x[:,0]+2*x[:,1],x[:,0]-x[:,1]])
    def constraint(x):
        calls.append('con')
        return (x[:,0]-.5)[:,None]
    p=basic(nObj=2,nCon=1,lb=0.,ub=10.,objFunc=objective,conFunc=constraint,optType=direction)
    result=p.evaluate(x[0] if rows==1 else x,target=target)
    expectedCalls=[]
    if target in [None,'objs']:
        np.testing.assert_array_equal(result.objs,np.column_stack([x[:,0]+2*x[:,1],x[:,0]-x[:,1]]))
        expectedCalls.append('obj')
    else:
        assert result.objs is None
    if target in [None,'cons']:
        np.testing.assert_array_equal(result.cons,(x[:,0]-.5)[:,None])
        expectedCalls.append('con')
    else:
        assert result.cons is None
    assert calls==expectedCalls and result.sims is None
    record('problem_target',rows=rows,target=target,direction=direction,passed=True)

for rows,target in product([1,2,7],[None,'objs','cons','sims']):
    x=np.arange(rows*2,dtype=float).reshape(rows,2)/10
    calls=[]
    def simulation(x):
        calls.append('sim')
        return np.stack([x,x+1],axis=1)
    def objective(x,context):
        calls.append('obj')
        return context.sims.sum(axis=(1,2))[:,None]
    def constraint(x,context):
        calls.append('con')
        return context.sims[:,0,:1]-1
    p=ModelProblem(nInput=2,nObj=1,nCon=1,lb=0.,ub=10.,simFunc=simulation,objFunc=objective,conFunc=constraint,optType='max')
    result=p.evaluate(x[0] if rows==1 else x,target=target)
    expectedSim=np.stack([x,x+1],axis=1)
    assert calls==['sim']+(['obj'] if target in [None,'objs'] else [])+(['con'] if target in [None,'cons'] else [])
    for key,expected in [('sims',expectedSim),('objs',expectedSim.sum(axis=(1,2))[:,None]),('cons',x[:,:1]-1)]:
        if target in [None,key]:
            np.testing.assert_array_equal(getattr(result,key),expected)
        else:
            assert getattr(result,key) is None
    record('model_target',rows=rows,target=target,passed=True)

p=basic(nInput=4,lb=[-3,-2,0,7],ub=[9,3,1,7],varType=[0,1,2,0],varSet={2:[.1,2.5,8.]})
for seed in [5,17,41]:
    unit=np.random.default_rng(seed).random((40,4))
    real=p.unit_to_space(unit)
    expected=np.column_stack([-3+unit[:,0]*12,-2+np.floor(unit[:,1]*6),np.array([.1,2.5,8.])[np.floor(unit[:,2]*3).astype(int)],np.full(40,7.)])
    np.testing.assert_array_equal(real,expected)
    np.testing.assert_allclose(p.unit_to_space(p.space_to_unit(real)),real,atol=1e-14)
    np.testing.assert_allclose(p.canonicalize_unit(unit),p.space_to_unit(real),atol=1e-14)
    record('mixed_mapping',seed=seed,passed=True)

for shape in ['vector','row','column']:
    lower=np.array([0.,10.])
    upper=np.array([1.,20.])
    if shape=='row':
        lower,upper=lower[None,:],upper[None,:]
    if shape=='column':
        lower,upper=lower[:,None],upper[:,None]
    p=basic(lb=lower,ub=upper)
    for rows in [1,2,3]:
        capture(f'bounds_{shape}_{rows}',lambda p=p,rows=rows:p.unit_to_space(np.full((rows,2),.5)).tolist())
    capture('lhs_'+shape,lambda p=p:LHS().sample(p,2,seed=17).tolist())

extreme=basic(nInput=1,lb=-1e308,ub=1e308)
capture('extreme_decode',lambda:extreme.unit_to_space(np.array([[0.],[.25],[.5],[.75],[1.]])).tolist())
capture('extreme_encode',lambda:extreme.space_to_unit(np.array([[-1e308],[-5e307],[0.],[5e307],[1e308]])).tolist())

# Reusing one output work buffer is common in expensive model adapters.
buffer=np.zeros(2)
@singleFunc
def reusedBuffer(x):
    buffer[:]=[x[0],2*x[0]]
    return buffer
p=basic(nInput=1,nObj=2,lb=0.,ub=5.,objFunc=reusedBuffer)
capture('single_func_reused_buffer',lambda:p.evaluate([[1.],[2.],[3.]]).objs.tolist())

for masked in [False,True]:
    mask=np.array([[False],[True]]) if masked else None
    p=ModelProblem(nInput=1,lb=0.,ub=1.,obs=np.ones((2,1)),mask=mask,simFunc=lambda x:np.tile([[[1.],[np.nan]]],(len(x),1,1)))
    capture('masked_nan_'+str(masked),lambda p=p:p.evaluate([[.5]],target='sims').sims.tolist())
for invalid in [np.zeros((3,1)),np.zeros((2,2)),np.array(['a','b'])]:
    p=basic(objFunc=lambda x,invalid=invalid:invalid)
    capture('malformed_output_'+str(invalid.shape),lambda p=p:p.evaluate([[0.,10.],[1.,20.]]).objs.tolist())

def sanitize(value):
    if isinstance(value,float) and not np.isfinite(value):
        return str(value)
    if isinstance(value,list):
        return [sanitize(item) for item in value]
    if isinstance(value,dict):
        return {key:sanitize(item) for key,item in value.items()}
    return value
Path(args.output).write_text(json.dumps(sanitize(records),indent=2,allow_nan=False))
print('records',len(records))
for row in records:
    if 'passed' not in row:
        print(row)
