"""Representable errors must survive intermediate overflow; ESS is diagnostic."""
from decimal import Decimal, localcontext
import warnings

import numpy as np
import pytest

from UQPyL.calibration import GLUE, SUFI2, CalReader
from UQPyL.calibration.util import mse, mae
from UQPyL.problem import ModelProblem


def decimalError(obs, sim, power):
    with localcontext() as context:
        context.prec = 120
        values = [abs(Decimal.from_float(float(x))-Decimal.from_float(float(y)))**power for x,y in zip(sim,obs)]
        return float(sum(values)/len(values))


@pytest.mark.parametrize('function,power',[(mse,2),(mae,1)])
@pytest.mark.parametrize('exponent',[-150,-80,0,80,150])
def testErrorMatchesDecimalAcrossUnitsAndMask(function,power,exponent):
    factor = 10.**exponent
    obs = np.array([1., 2., 3., 4.])*factor
    sim = np.array([[1.2, 2.4, 4., 3.], [1., 2., 3., 4.]])*factor
    mask = np.array([False, False, True, False])
    saved = sim.copy()
    sim.flags.writeable = False
    actual = function(obs,sim,mask=mask)
    expected = [decimalError(obs[~mask],row[~mask],power) for row in sim]
    np.testing.assert_allclose(actual,expected,rtol=5e-15,atol=0)
    np.testing.assert_array_equal(sim,saved)


@pytest.mark.parametrize('function,power,obs,sim',[
    (mse,2,[0.,0.],[1.4e154,0.]),
    (mse,2,[0.,0.],[1e154,1e154]),
    (mse,2,[0.,0.],[2e-162,2e-162]),
    (mae,1,[-1e308,1e308],[1e308,1e308]),
    (mae,1,[0.,0.],[1e308,1e308]),
    (mae,1,[0.,0.],[5e-324,5e-324]),
    (mae,1,[1e308,0.],[1e308,1e-300]),
    (mse,2,[1e308,0.],[1e308,1e-150]),
])
def testRecoverableExtremesAndCancellation(function,power,obs,sim):
    actual = function(obs,[sim])[0]
    assert np.isfinite(actual) and actual>0
    assert actual == pytest.approx(decimalError(obs,sim,power),rel=5e-15,abs=0)


@pytest.mark.parametrize('function,obs,sim,expected',[
    (mse,[0.],[[1e200]],np.inf),
    (mse,[0.],[[1e-200]],0.),
    (mae,[-1e308],[[1e308]],np.inf),
    (mae,[0.,0.],[[5e-324,0.]],0.),
])
def testGenuineRangeLimitsWarnOnce(function,obs,sim,expected):
    with pytest.warns(RuntimeWarning,match=f'{function.__name__.upper()} exceeds floating-point range') as caught:
        actual = function(obs,sim)[0]
    assert len(caught)==1
    assert actual==expected


@pytest.mark.parametrize('methodClass',[GLUE,SUFI2])
@pytest.mark.parametrize('metric',['mse','mae'])
def testActualCalibrationSelectionWithLargeUnits(methodClass,metric):
    if metric=='mse':
        obs=np.array([[0.],[1.]])
        values=np.array([[1.4],[1.2]])
        threshold=8e307
        def simulate(x):
            return np.column_stack([x[:, 0] * 1e+154, np.ones(len(x))])
        expected=[9.8e307,7.2e307]
    else:
        obs=np.array([[-1e308],[1e308]])
        values=np.array([[.9],[.8]])
        threshold=9.25e307
        def simulate(x):
            return np.column_stack([x[:, 0] * 1e+308, np.full(len(x), 1e+308)])
        expected=[9.5e307,9e307]
    problem=ModelProblem(nInput=1,lb=0,ub=2,obs=(obs).reshape(-1),simFunc=simulate)
    options=dict(threshold=threshold) if methodClass is GLUE else dict(eliteSize=1)
    result=methodClass(metric=metric).run(problem,values,**options)
    np.testing.assert_array_equal(result.bestDecs,values[1:])
    np.testing.assert_allclose(result.diagnostics['scores'],expected,rtol=5e-15)
    if methodClass is GLUE:
        np.testing.assert_array_equal(result.samples,values[1:])


def weightProblem():
    return ModelProblem(nInput=1,lb=0,ub=1,obs=np.array([0.5, 1.0]),
                        simFunc=lambda x:np.stack([x[:, 0], 2 * x[:, 0]], axis=1))


def testConcentratedGlueWeightsWarnWithoutChangingResults(tmp_path):
    problem=weightProblem()
    problem.workDir=str(tmp_path)
    samples=np.linspace(0,1,50)[:,None]
    with pytest.warns(RuntimeWarning,match='GLUE uncertainty effective sample size is below 20') as caught:
        result=GLUE(saveFlag=True).run(problem,samples,threshold=10.,
            logLikelihood=lambda obs,sim,mask:np.where(np.arange(len(sim))==25,0.,-np.inf))
    assert len(caught)==1
    assert result.diagnostics['uncertaintyStatus']=='low_effective_sample_size'
    assert result.diagnostics['effectiveSampleSize']==1.
    expected=np.zeros(50)
    expected[25]=1
    np.testing.assert_array_equal(result.weights,expected)
    np.testing.assert_array_equal(result.intervals[0]['lower'],result.simulations[25])
    np.testing.assert_array_equal(result.intervals[0]['upper'],result.simulations[25])
    with CalReader(next(tmp_path.rglob('*.sqlite3'))) as reader:
        restored=reader.load_result()
    assert restored.diagnostics['uncertaintyStatus']=='low_effective_sample_size'


@pytest.mark.parametrize('count',[20,21,100])
def testUniformLikelihoodAtOrAboveThresholdDoesNotWarn(count):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=GLUE().run(weightProblem(),np.linspace(0,1,count)[:,None],threshold=10.,
                          logLikelihood=lambda obs,sim,mask:np.zeros(len(sim)))
    assert not caught
    assert result.diagnostics['uncertaintyStatus']=='estimated'
    assert result.diagnostics['effectiveSampleSize']==pytest.approx(count)


def testUniformScreeningRecordsSmallEssWithoutLikelihoodWarning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=GLUE().run(weightProblem(),[[.4],[.6]],threshold=10.)
    assert not caught
    assert result.diagnostics['weighting']=='uniform'
    assert result.diagnostics['uncertaintyStatus']=='low_effective_sample_size'


def testSufiUniformTwentyPriorMembersDoNotSpuriouslyWarn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=SUFI2().run(weightProblem(),[[.4],[.6]],eliteSize=1,
            uncertaintyX=np.linspace(0,1,20)[:,None],logLikelihood=lambda obs,sim,mask:np.zeros(len(sim)))
    assert not caught
    assert result.diagnostics['uncertaintyStatus']=='estimated'
