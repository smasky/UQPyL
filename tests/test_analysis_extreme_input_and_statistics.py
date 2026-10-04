"""Extreme input normalization and explicit Morris statistic range diagnostics."""
from decimal import Decimal, localcontext
import warnings

import numpy as np
import pytest

from UQPyL.analysis import DeltaTest, Morris
from UQPyL.analysis.runtime import AnaReader
from UQPyL.doe import LHS
from UQPyL.problem import Problem


def makeProblem(lower, upper):
    return Problem(nInput=len(lower), nObj=1, lb=lower, ub=upper,
                   objFunc=lambda x: np.zeros((len(x), 1)))


@pytest.mark.parametrize('bounds,points', [
    ((-1e308,1e308),[-1e308,-9e307,0.,9e307,1e308]),
    ((-1.7e308,1e308),[-1.7e308,-1e308,1e308,1.7e308]),
    ((-1.7e308,-1e308),[-1.7e308,-1e308,1e308]),
    ((0.,1e-310),[0.,5e-311,1e-310]),
])
def testDeltaInputScalingAgainstDecimal(bounds, points):
    x=np.array(points)[:,None]
    original=x.copy()
    problem=makeProblem([bounds[0]],[bounds[1]])
    with localcontext() as ctx:
        ctx.prec=80
        lo,hi=map(Decimal.from_float,bounds)
        expected=np.array([float((Decimal.from_float(v)-lo)/(hi-lo)) for v in points])[:,None]
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        actual=DeltaTest(verboseFlag=False)._scaleInputs(problem,x)
    np.testing.assert_allclose(actual,expected,rtol=3e-15,atol=0)
    np.testing.assert_array_equal(x,original)


@pytest.mark.parametrize('seed',[0,3,17,41])
def testDeltaAnalysisAndSubsetSelectionSurviveExtremeInputUnits(seed):
    base=makeProblem([-1.,0.],[1.,1.])
    unit=LHS().sample(makeProblem([0.,0.],[1.,1.]),128,seed=seed)
    x=np.column_stack([-.9+.8*unit[:,0],unit[:,1]])
    y=3*unit[:,:1]+1
    method=DeltaTest(verboseFlag=False)
    expected=method.analyze(base,x,y)
    expectedSubset=method.findCombVio(base,x,y)
    extreme=makeProblem([-1e308,0.],[1e308,1.])
    transformed=x.copy()
    transformed[:,0]*=1e308
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        actual=method.analyze(extreme,transformed,y)
        actualSubset=method.findCombVio(extreme,transformed,y)
    for metric in ['S1','S1_norm']:
        np.testing.assert_allclose(actual[metric].values,expected[metric].values,rtol=1e-12,atol=1e-14)
    assert actualSubset==expectedSubset==[base.xLabels[0]]
    expectedSearch=method.findCombEA(base,x,y,FEs=50,verboseFlag=False,saveFlag=False,seed=17)
    actualSearch=method.findCombEA(extreme,transformed,y,FEs=50,verboseFlag=False,saveFlag=False,seed=17)
    np.testing.assert_array_equal(actualSearch.bestDecs,expectedSearch.bestDecs)
    np.testing.assert_array_equal(actualSearch.bestDecs,[[1,0]])
    np.testing.assert_allclose(actualSearch.bestObjs,expectedSearch.bestObjs,rtol=1e-12)



def testDeltaExtremeDiscreteAndFixedColumns():
    problem=Problem(nInput=2,nObj=1,lb=[0.,7.],ub=[1.,7.],varType=[2,0],
                    varSet={0:[-1e308,0.,1e308]},objFunc=lambda x:x[:,:1])
    x=np.array([[-1e308,7.],[0.,7.],[1e308,7.]])
    np.testing.assert_allclose(DeltaTest()._scaleInputs(problem,x),[[0,0],[.5,0],[1,0]])
    x[0,1]=8.
    with pytest.raises(ValueError,match='fixed inputs'):
        DeltaTest()._scaleInputs(problem,x)


@pytest.mark.parametrize('scale',[1.,1e307,1e308])
def testMorrisSigmaRangeIsExplicitAndNormalizedEffectSurvives(scale,tmp_path):
    problem=makeProblem([0.],[1.])
    problem.workDir=str(tmp_path)
    x=np.array([[0.],[2/3],[1/3],[1.]])
    y=np.array([[0.],[scale],[0.],[-scale]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=Morris(verboseFlag=False,saveFlag=True).analyze(problem,x,y,dict(designType='morris',numLevels=4))
    status=result.extra['morris_statistic_status']
    assert status['mu']==status['mu_star']==[['available']]
    np.testing.assert_allclose(result['mu_star'].values/scale,[[1.5]],rtol=1e-15)
    np.testing.assert_array_equal(result['S1_norm'].values,[[1.]])
    if scale==1e308:
        assert len(caught)==1 and 'Morris sigma' in str(caught[0].message)
        assert 'overflow' in str(caught[0].message)
        assert status['sigma']==[['overflow']]
        assert np.isposinf(result['sigma'].values[0,0])
    else:
        assert not caught and status['sigma']==[['available']]
        np.testing.assert_allclose(result['sigma'].values/scale,[[1.5*np.sqrt(2)]],rtol=1e-15)
    with AnaReader(next(tmp_path.rglob('*.sqlite3'))) as reader:
        loaded=reader.load_result()
    assert loaded.extra['morris_statistic_status']==status
    np.testing.assert_array_equal(loaded['sigma'].values,result['sigma'].values)


def testMorrisUnrepresentableMeanIsMarkedWithoutMislabelingExactZero():
    tiny=np.nextafter(0.,1.)
    problem=makeProblem([0.],[1.])
    x=np.tile([[0.],[1.]],(3,1))
    y=np.array([[0.],[tiny],[0.],[-tiny],[0.],[tiny]])
    with pytest.warns(RuntimeWarning,match='Morris mu underflow'):
        result=Morris(verboseFlag=False).analyze(problem,x,y,dict(designType='morris',numLevels=4))
    assert result.extra['morris_statistic_status']['mu']==[['underflow']]
    assert result.extra['morris_statistic_status']['mu_star']==[['available']]
    np.testing.assert_array_equal(result['S1_norm'].values,[[1.]])
    zero=Morris(verboseFlag=False).analyze(problem,x,np.zeros_like(y),dict(designType='morris',numLevels=4))
    assert all(value==[['available']] for value in zero.extra['morris_statistic_status'].values())
