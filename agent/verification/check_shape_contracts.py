"""Read-only audit of sample and observation axes. Run from repository root."""
import ast
import argparse
import json
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from UQPyL.problem import Problem, ModelProblem, SimContext
from UQPyL.calibration import GLUE, SUFI2, ES, IES, CalReader
from UQPyL.surrogate.regression import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='agent/verification/1004-shape-contracts.json')
args = parser.parse_args()
records = []


def record(name, fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value = fn()
            result = {'case': name, 'value': value}
        except Exception as error:
            result = {'case': name, 'error': type(error).__name__, 'message': str(error)}
        result['warnings'] = [str(w.message) for w in caught]
    records.append(result)
    return result


for dimension in (1, 2):
    p = Problem(nInput=dimension, nObj=1, lb=-5, ub=5, objFunc=lambda x: np.sum(x*x, axis=1, keepdims=True))
    for label, x in [('scalar', 1.), ('vector', np.array([1.,2.])), ('row', np.array([[1.,2.]])),
                     ('column', np.array([[1.],[2.]])), ('batch', np.ones((3,dimension))),
                     ('tensor',np.ones((2,1,dimension)))]:
        record(f'problem_x_{dimension}_{label}',lambda p=p,x=x: {'validated_shape':list(p.validate(x).shape), 'result':p.evaluate(x).objs.tolist()})

obsGrid = np.array([[1.,2.],[3.,4.]])
for label, obs in [('vector',obsGrid.ravel()),('column',obsGrid.reshape(4,1)),('row',obsGrid.reshape(1,4)),('grid',obsGrid)]:
    record('obs_'+label,lambda obs=obs: {'obs_shape':list(ModelProblem(nInput=1,lb=-5,ub=5,obs=obs,simFunc=lambda x: x).obs.shape)})

for label, mask in [('none',None),('vector',np.zeros(4,dtype=bool)),('grid',np.array([[False,True],[False,False]])),
                    ('numeric',np.array([[0.,np.nan],[2.,0.]]))]:
    record('mask_'+label,lambda mask=mask: {'mask':ModelProblem(nInput=1,lb=-5,ub=5,obs=obsGrid,mask=mask,simFunc=lambda x:x).flattenMask().tolist()})

# The same observation positions, flattened or structured, with and without a masked NaN.
mask = np.array([[False,True],[False,False]])
for layout in ('grid','flat','transposed_grid','wrong_count'):
    for missing in (False,True):
        def simulate(x, layout=layout, missing=missing):
            base = np.broadcast_to(obsGrid, (len(x),2,2)).copy()
            if missing:
                base[:,0,1] = np.nan
            if layout=='flat': return base.reshape(len(x),4)
            if layout=='transposed_grid': return base.transpose(0,2,1)
            if layout=='wrong_count': return np.ones((len(x),3))
            return base
        p=ModelProblem(nInput=1,lb=-5,ub=5,obs=obsGrid,mask=mask,simFunc=simulate)
        record(f'sim_{layout}_nan_{missing}',lambda p=p: {'shape':list(p.evaluate([[1.]],target='sims').sims.shape)})
        record(f'glue_{layout}_nan_{missing}',lambda p=p: {'scores':GLUE().run(p,[[1.],[2.]],threshold=100).scores.tolist()})

# Mathematical parity: an explicit reshape of the SAME ordering changes no update.
prior=np.linspace(-1.,1.,24)[:,None]
for cls in (GLUE,SUFI2,ES,IES):
    for masked in (False,True):
        def parity(cls=cls,masked=masked):
            results=[]
            for flat in (False,True):
                def simulate(x,flat=flat):
                    values=x*np.array([[1.,2.,3.,4.]])
                    return values if flat else values.reshape(len(x),2,2)
                p=ModelProblem(nInput=1,lb=-5,ub=5,obs=obsGrid*.2,mask=mask if masked else None,simFunc=simulate)
                method=cls(maxIters=2,seed=17) if cls is IES else cls()
                options={'r':np.eye(3 if masked else 4)} if cls in (ES,IES) else ({'threshold':100} if cls is GLUE else {'eliteSize':8,'seed':17})
                results.append(method.run(p,prior,**options))
            np.testing.assert_allclose(results[0].samples,results[1].samples,rtol=0,atol=0)
            np.testing.assert_allclose(results[0].simulations,results[1].simulations,rtol=0,atol=0)
            np.testing.assert_allclose(results[0].scores,results[1].scores,rtol=0,atol=0)
            return {'identical':True,'obs_shape':list(results[0].obs.shape),'simulation_shape':list(results[0].simulations.shape)}
        outcome=record(f'{cls.__name__}_reshape_parity_mask_{masked}',parity)
        assert 'error' not in outcome, outcome

for cls in (ES,IES):
    for size in (3,4):
        def covariance(cls=cls,size=size):
            p=ModelProblem(nInput=1,lb=-5,ub=5,obs=obsGrid,mask=mask,simFunc=lambda x:(x*np.arange(1,5)[None,:]).reshape(len(x),2,2))
            method=cls(maxIters=1,seed=17) if cls is IES else cls()
            return {'shape':list(method.run(p,prior,r=np.eye(size)).samples.shape)}
        record(f'{cls.__name__}_masked_R_{size}',covariance)

model=LinearRegression()
model.fit(np.array([0.,1.,2.]),np.array([0.,2.,4.]))
record('surrogate_fit_vector',lambda: {'x_train_shape':list(model.xTrain.shape)})
record('surrogate_predict_vector',lambda: {'prediction':model.predict(np.array([0.,1.,2.])).tolist()})
record('surrogate_predict_column',lambda: {'prediction':model.predict(np.array([[0.],[1.],[2.]])).tolist()})

# Execute the actual documented function, rather than a copied reproduction.
source=Path('docs_v2/cn/problem.md').read_text()
start=source.index('def objFunc(X, simContext):')
end=source.index('\n\n',start)
namespace={'np':np}
exec(compile(ast.parse(source[start:end]),'<documented-objFunc>','exec'),namespace)
record('documented_masked_objective',lambda: namespace['objFunc'](np.ones((2,1)),SimContext(np.ones((2,2,2)),obsGrid,mask)).tolist())

with tempfile.TemporaryDirectory() as directory:
    p=ModelProblem(nInput=1,lb=-5,ub=5,obs=obsGrid,mask=mask,simFunc=lambda x:x*np.arange(1,5)[None,:])
    p.workDir=directory
    res=GLUE(saveFlag=True).run(p,prior,threshold=100)
    with CalReader(next(Path(directory).rglob('*.sqlite3'))) as reader:
        restored=reader.load_result()
    np.testing.assert_array_equal(res.simulations,restored.simulations)
    np.testing.assert_array_equal(res.obs,restored.obs)
    record('sqlite_shapes',lambda:{'obs':list(restored.obs.shape),'mask':list(restored.mask.shape),'simulations':list(restored.simulations.shape),'n_time':restored.nTime,'n_series':restored.nSeries,'n_obs':restored.nObs})

Path(args.output).write_text(json.dumps(records,indent=2)+'\n')
print(f'{len(records)} audit records saved (expected rejection cases included).')
