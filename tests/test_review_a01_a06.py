from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import cramervonmises_2samp

from UQPyL.optimization.core.uniform_point import uniformPoint
from UQPyL.optimization.moea import MOEAD, NSGAIII, RVEA
from UQPyL.problem import Problem
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF
from UQPyL.surrogate.scaler import StandardScaler, MinMaxScaler
from UQPyL.surrogate.metric import mse, r_square, nse
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.analysis import RSA


def fixedModel(**kwargs):
    return GPR(kernel=RBF(length_scale=.3, length_attr=None), C_attr=None, **kwargs)


def testSingleObjectiveReferencePointTerminates():
    code = 'from UQPyL.optimization.core.uniform_point import uniformPoint; w,n=uniformPoint(8,1); assert n==1 and w.tolist()==[[1.0]]'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('method', ['NBI', 'grid'])
@pytest.mark.parametrize('n,m', [(0,2), (8,0), (-1,2), (True,2), (8,False), (2.5,2), (8,1.5)])
def testReferencePointInputValidation(method,n,m):
    with pytest.raises(ValueError, match='positive integer'):
        uniformPoint(n,m,method)


@pytest.mark.parametrize('algorithmClass', [MOEAD, NSGAIII, RVEA])
def testMoeaRejectsSingleObjectiveBeforeEvaluation(algorithmClass):
    calls = []
    problem = Problem(nInput=1,nObj=1,lb=0,ub=1,objFunc=lambda x:calls.append(x))
    with pytest.raises(ValueError, match='at least two objectives'):
        algorithmClass(nPop=8,verboseFlag=False,logFlag=False,saveFlag=False).run(problem,seed=1)
    assert not calls


@pytest.mark.parametrize('scalerClass', [StandardScaler, MinMaxScaler])
@pytest.mark.parametrize('axis', [0,1])
def testScalerOwnership(scalerClass,axis):
    shared = scalerClass()
    scalers = [None,None]; scalers[axis] = shared
    x = np.linspace(0,1,8)[:,None]; y = np.sin(3*x)
    a,b = fixedModel(scalers=scalers),fixedModel(scalers=scalers)
    a.fit(x,y); before = a.predict(x,returnVar=True)
    b.fit(10+10*x,10+20*y)
    after = a.predict(x,returnVar=True)
    for left,right in zip(before,after):
        np.testing.assert_array_equal(left,right)
    assert not shared.fitted
    assert a.xScaler is not b.xScaler if axis==0 else a.yScaler is not b.yScaler


@pytest.mark.parametrize('stage', ['prepare','fit','input'])
def testFailedRefitInvalidatesAndCanRecover(monkeypatch,stage):
    x=np.linspace(0,1,8)[:,None]; y=np.sin(3*x)
    model=fixedModel(scalers=(StandardScaler(),None)).fit(x,y)
    before=model.predict(x)
    def fail(*args):
        model.fitState['partial'] = 1
        raise ValueError('injected failure')
    with monkeypatch.context() as patch:
        if stage != 'input':
            patch.setattr(model, '_prepare_training_components' if stage=='prepare' else 'fitHyper', fail)
        with pytest.raises(ValueError):
            model.fit(10+10*x, y[:-1] if stage=='input' else y)
    assert not model.fitState
    with pytest.raises(RuntimeError,match='fitted'):
        model.predict(x)
    model.fit(x,y)
    np.testing.assert_allclose(model.predict(x),before)


@pytest.mark.parametrize('metric', [mse,r_square,nse])
def testMetricMixedSingleOutputShapes(metric):
    y=np.arange(1.,5.)
    np.testing.assert_allclose(metric(y,y[:,None]),metric(y[:,None],y[:,None]))
    np.testing.assert_allclose(metric(y[:,None],y),metric(y[:,None],y[:,None]))


@pytest.mark.parametrize('metric', [mse,r_square,nse])
@pytest.mark.parametrize('other', [np.zeros((3,2)),np.zeros((2,1)),np.zeros((3,1,1)),np.full((3,1),np.nan)])
def testMetricRejectsIncompatibleData(metric,other):
    with pytest.raises(ValueError):
        metric(np.arange(3.),other)


class CandidateOptimizer:
    def run(self,problem,seed=None):
        points=np.array([[.1],[.2]])
        scores=problem.evaluate(points).objs
        assert not np.isnan(scores).any()
        best=int(np.argmax(scores[:,0]))
        return SimpleNamespace(bestDecs=points[best],bestObjs=scores[best])


def tune(tuner,entry,x,y,**kwargs):
    options={'paraGrid':{'C':[.1,.2]}} if entry=='gridTune' else {'paraList':['C']}
    return getattr(tuner,entry)(x,y,seed=1,tuneMode='joint',**options,**kwargs)


@pytest.mark.parametrize('entry', ['gridTune','optTune'])
@pytest.mark.parametrize('badData', ['one_test_point','constant'])
def testTunerRejectsUndefinedValidation(entry,badData):
    x=np.arange(8.)[:,None]; y=np.ones_like(x) if badData=='constant' else x*x
    tuner=AutoTuner(LinearRegression(lossType='Ridge'),CandidateOptimizer())
    with pytest.raises(ValueError,match='validation'):
        tune(tuner,entry,x,y,ratio=50 if badData=='constant' else 10)


@pytest.mark.parametrize('entry', ['gridTune','optTune'])
@pytest.mark.parametrize('failure', ['exception','nan'])
def testTunerAllCandidatesFailClearly(monkeypatch,entry,failure):
    x=np.arange(10.)[:,None]; y=x*x
    model=LinearRegression(lossType='Ridge')
    if failure=='exception':
        def fail(*args): raise np.linalg.LinAlgError('bad candidate')
        monkeypatch.setattr(model,'fitModel',fail)
    else:
        monkeypatch.setattr(model,'predict',lambda x:np.full((len(x),1),np.nan))
    with pytest.raises(RuntimeError,match='No candidate'):
        tune(AutoTuner(model,CandidateOptimizer()),entry,x,y,ratio=30)
    assert not model.fitState


def testRsaBinaryOutputMatchesIndependentTwoSampleStatistic():
    x=np.linspace(0,1,40)[:,None]; y=(x>.5).astype(float)
    p=Problem(nInput=1,nObj=1,lb=0,ub=1,objFunc=lambda x:(x>.5).astype(float))
    result=RSA(verboseFlag=False).analyze(p,x,y)
    expected=cramervonmises_2samp(x[:20,0],x[20:,0]).statistic
    assert result['S1'].values[0,0] == pytest.approx(expected)
    assert result['S1_norm'].values[0,0] == 1


def testRsaConstantOutputRemainsFiniteZero():
    x=np.linspace(0,1,8)[:,None]
    p=Problem(nInput=1,nObj=1,lb=0,ub=1,objFunc=lambda x:np.ones_like(x))
    result=RSA(verboseFlag=False).analyze(p,x,np.ones_like(x))
    np.testing.assert_array_equal(result['S1'].values,[[0.]])
