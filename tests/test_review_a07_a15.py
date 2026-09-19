import builtins
import doctest
import importlib

import numpy as np
import pytest
from scipy.stats import kendalltau

from UQPyL.analysis import DeltaTest
from UQPyL.problem import Problem, Sphere
from UQPyL.inference import DEMC
from UQPyL.surrogate.metric import rank_score
from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.optimization.expensive import ASMO
from UQPyL.optimization.soea import GA
from UQPyL.viz.common import smooth_curve

QUIET = dict(verboseFlag=False,logFlag=False,saveFlag=False)

@pytest.mark.parametrize('count',[0,-1,True,1.5])
def testDeltaRejectsInvalidNeighbors(count):
    with pytest.raises(ValueError,match='positive integer'):
        DeltaTest(nNeighbors=count)

@pytest.mark.parametrize('nSamples,nInput',[(8,1),(2,2)])
def testDeltaReportsUnsupportedData(nSamples,nInput):
    x=np.arange(nSamples*nInput,dtype=float).reshape(nSamples,nInput)
    with pytest.raises(ValueError,match='at least two inputs|smaller than the sample count'):
        DeltaTest(**QUIET).analyze(Sphere(nInput=nInput),x,np.sum(x*x,axis=1)[:,None])


def testRankUsesEveryOutputAndHandlesTies():
    actual=np.array([[0.,10.],[1.,20.],[2.,30.]])
    predicted=np.array([[0.,30.],[1.,20.],[2.,10.]])
    assert rank_score(actual,predicted)==0.
    a=np.array([0.,0.,1.,2.]); b=np.array([0.,1.,1.,2.])
    assert rank_score(a,b)==pytest.approx(kendalltau(a,b,variant='b').statistic)
    assert rank_score(np.ones(4),np.arange(4.))==0.
    with pytest.raises(ValueError,match='two samples'):
        rank_score(np.ones(1),np.ones(1))

@pytest.mark.parametrize('length',[0,1,3,9,10,40,60])
def testSmoothingPreservesLengthAndInputs(length):
    x=np.arange(length,dtype=float)**2; original=x.copy()
    result=smooth_curve(x)
    assert result.shape==x.shape
    np.testing.assert_array_equal(x,original)
    if length<10:
        np.testing.assert_array_equal(result,x)
    if length==60:
        assert np.any(result[20:-20]!=x[20:-20])

@pytest.mark.parametrize('count',[1,2,True,3.5])
def testDemcInvalidChainsAtConstruction(count):
    with pytest.raises(ValueError,match='nChains'):
        DEMC(nChains=count,**QUIET)


def testDemcDefaultRuns():
    model=DEMC(warmUp=0,maxIterTimes=2,**QUIET)
    model.run(Sphere(nInput=1),seed=3)
    assert model.get('nChains')==3


def testMultiSurrogatePropagatesIndependentReproducibleStreams():
    x=np.linspace(0,1,10)[:,None]; y=np.hstack([np.sin(4*x),np.cos(4*x)])
    traces=[]
    for _ in range(2):
        model=MultiSurrogate(2,[KRG(),KRG()])
        model.rng=np.random.default_rng(72)
        model.fit(x,y)
        traces.append((model.predict(x),[m.rng.integers(0,2**32) for m in model.models_list]))
    np.testing.assert_array_equal(traces[0][0],traces[1][0])
    assert traces[0][1]==traces[1][1]
    assert traces[0][1][0]!=traces[0][1][1]


def testAsmoSeedControlsActualKrgFitsAndEvaluations():
    records=[]
    for _ in range(2):
        evaluated=[]
        def objective(x):
            evaluated.append(x.copy())
            return np.sin(5*x)+x*x
        problem=Problem(nInput=1,nObj=1,lb=0,ub=1,objFunc=objective)
        alg=ASMO(nInit=8,maxFEs=10,maxIters=2,surrogate=KRG(),
                 optimizer=GA(nPop=8,maxFEs=16,maxIters=1,**QUIET),**QUIET)
        result=alg.run(problem,seed=12)
        records.append((np.vstack(evaluated),result.bestObjs,alg.surrogate.predict(np.array([[.25],[.75]]))))
    for a,b in zip(*records):
        np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize('entry',['gridTune','optTune'])
@pytest.mark.parametrize('error',[TypeError('bug'),AttributeError('bug'),ValueError('bad shape')])
def testTunerProgrammingErrorsPropagate(monkeypatch,entry,error):
    model=LinearRegression(lossType='Ridge')
    def fail(*args): raise error
    monkeypatch.setattr(model,'fitModel',fail)
    class Optimizer:
        def run(self,problem,seed):
            problem.evaluate(np.array([[.1]]))
    tuner=AutoTuner(model,Optimizer())
    kwargs={'paraGrid':{'C':[.1]}} if entry=='gridTune' else {'paraList':['C']}
    x=np.arange(10.)[:,None]
    with pytest.raises(type(error),match=str(error)):
        getattr(tuner,entry)(x,x*x,ratio=30,tuneMode='joint',seed=1,**kwargs)


def testTunerRecordsNumericalFailuresWithoutPrinting(monkeypatch,capsys):
    model=LinearRegression(lossType='Ridge')
    original=model.fitModel; calls=[]
    def fit(x,y):
        calls.append(1)
        if len(calls)==1: raise np.linalg.LinAlgError('singular candidate')
        return original(x,y)
    monkeypatch.setattr(model,'fitModel',fit)
    tuner=AutoTuner(model); x=np.arange(10.)[:,None]
    _,score=tuner.gridTune(x,x*x,{'C':[.1,.2]},ratio=30,seed=1,tuneMode='joint')
    assert np.isfinite(score)
    assert tuner.candidateFailures==[{'candidate_index':0,'error_type':'LinAlgError','message':'singular candidate'}]
    assert not capsys.readouterr().out

@pytest.mark.parametrize('failure',[RuntimeError('implementation error'),ModuleNotFoundError('unrelated',name='unrelated_dependency')])
def testOptionalMarsDoesNotHideUnexpectedImportFailures(monkeypatch,failure):
    module=importlib.import_module('UQPyL.analysis.methods')
    original=builtins.__import__
    def fail(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='mars' and globals and globals.get('__package__')=='UQPyL.analysis.methods':
            raise failure
        return original(name,globals,locals,fromlist,level)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins,'__import__',fail)
            with pytest.raises(type(failure),match=str(failure)):
                importlib.reload(module)
    finally:
        importlib.reload(module)


def testOptionalMarsMissingExtensionIsRecognized(monkeypatch):
    module=importlib.import_module('UQPyL.analysis.methods'); original=builtins.__import__
    def fail(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='mars' and globals and globals.get('__package__')=='UQPyL.analysis.methods':
            raise ModuleNotFoundError('missing extension',name='UQPyL.surrogate.mars.core._forward')
        return original(name,globals,locals,fromlist,level)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins,'__import__',fail)
            importlib.reload(module)
            assert module.MARS is None
    finally:
        importlib.reload(module)


@pytest.mark.parametrize("name", ["full_fact", "random", "morris", "saltelli", "fast", "sobol", "lhs"])
def testDoeDocumentationExamplesExecute(name):
    module=importlib.import_module("UQPyL.doe.methods."+name)
    assert doctest.testmod(module).failed==0


def testReaderListUsesSnakeCaseFields(tmp_path):
    import sqlite3
    from UQPyL.core.runtime_reader import BaseReader
    path=tmp_path/'run.sqlite3'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE run (runId TEXT, createdAt TEXT, finishedAt TEXT, finalFEs INTEGER, finalIters INTEGER)')
        conn.execute("INSERT INTO run VALUES ('id','start','end',12,2)")
    row=BaseReader.list_runs(tmp_path,'*')[0]
    assert set(row)=={'run_id','created_at','finished_at','final_fes','final_iters','db_path','file_name'}
    assert row['final_fes']==12 and row['file_name']=='run.sqlite3'
