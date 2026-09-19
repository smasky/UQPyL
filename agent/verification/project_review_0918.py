"""Read-only edge-case probes; exceptions are recorded, not fixed."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import warnings
import numpy as np
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF
from UQPyL.surrogate.scaler import StandardScaler
from UQPyL.surrogate.metric import mse, rank_score
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.analysis import DeltaTest, RSA
from UQPyL.problem import Problem
from UQPyL.inference import DEMC
from UQPyL.viz.common import smooth_curve

results = {}
def capture(name, func):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value = func()
            results[name] = {'value': value}
        except Exception as error:
            results[name] = {'error':type(error).__name__, 'message':str(error)}
        results[name]['warnings'] = [str(w.message) for w in caught]

def fixed(**kwargs):
    return GPR(kernel=RBF(length_scale=.3,length_attr=None), C_attr=None, **kwargs)
x = np.linspace(0,1,8)[:,None]
y = np.sin(3*x)
def sharedScaler():
    scaler = StandardScaler()
    a = fixed(scalers=(scaler,None)).fit(x,y)
    before = a.predict(x).copy()
    fixed(scalers=(scaler,None)).fit(10+10*x,y)
    return float(np.max(abs(a.predict(x)-before)))
capture('shared_scaler_prediction_delta', sharedScaler)
def failedRefit():
    model = fixed(scalers=(StandardScaler(),None)).fit(x,y)
    before = model.predict(x).copy()
    def fail(values):
        raise ValueError('injected component initialization failure')
    model._prepare_training_components = fail
    try:
        model.fit(10+10*x,y)
    except ValueError:
        pass
    return float(np.max(abs(model.predict(x)-before)))
capture('failed_refit_stale_prediction_delta', failedRefit)
capture('mse_equal_values_mixed_shapes',lambda:mse(np.arange(1.,4.),np.arange(1.,4.)[:,None]).tolist())
capture('rank_second_output_reversed',lambda:float(rank_score(np.array([[0.,10.],[1.,20.],[2.,30.]]),np.array([[0.,30.],[1.,20.],[2.,10.]]))))
capture('grid_single_validation_sample',lambda:str(AutoTuner(fixed()).gridTune(x,y,{'C':[1e-6]},seed=4)))
capture('smooth_three_samples',lambda:smooth_curve(np.arange(3.),10).tolist())
p = Problem(nInput=1,nObj=1,lb=0,ub=1,objFunc=lambda x:x)
p2 = Problem(nInput=2,nObj=1,lb=0,ub=1,objFunc=lambda x:x[:,:1])
capture('delta_one_input',lambda:DeltaTest(verboseFlag=False).analyze(p,x,y))
capture('delta_two_samples',lambda:DeltaTest(verboseFlag=False).analyze(p2,np.ones((2,2)),np.ones((2,1))))
capture('demc_default',lambda:DEMC(warmUp=0,maxIterTimes=1,verboseFlag=False,saveFlag=False).run(p,seed=1))
xRsa = np.linspace(0,1,40)[:,None]
capture('rsa_threshold_output',lambda:RSA(verboseFlag=False).analyze(p,xRsa,(xRsa>.5).astype(float)).getMetric('S1').values.tolist())
def singleObjectiveVectors():
    code = 'from UQPyL.optimization.core.uniform_point import uniformPoint; print("entered",flush=True); uniformPoint(8,1)'
    try:
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=5)
        return {'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    except subprocess.TimeoutExpired as error:
        return {'timeout_seconds':5,'stdout':str(error.stdout)}
capture('reference_vectors_single_objective',singleObjectiveVectors)
files = list(Path('UQPyL').rglob('*.py'))
for path in files:
    ast.parse(path.read_text(), filename=str(path))
results['python_files_parsed'] = len(files)
Path('agent/verification/0918-project-review-probes.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(results,indent=2,ensure_ascii=False))
