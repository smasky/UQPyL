"""Uniform result rows, explicit provenance and persisted summaries."""

from copy import deepcopy

import numpy as np
import pytest

from UQPyL.calibration import ES, IES, GLUE, SUFI2, CalReader
from UQPyL.calibration.util import rmse
from UQPyL.problem import ModelProblem


def makeProblem():
    return ModelProblem(nInput=1, lb=-6, ub=6, obs=np.array([1.0, 2.0, 99.0]),
                        mask=np.array([False, False, True]),
                        simFunc=lambda x:np.column_stack([x[:, 0], 2 * x[:, 0], -x[:, 0]]))


def runMethod(method, problem):
    samples=np.array([[0.], [1.2], [1.], [.8]])
    if isinstance(method,GLUE):
        return method.run(problem,samples,threshold=.4)
    if isinstance(method,SUFI2):
        return method.run(problem,samples,eliteSize=2)
    return method.run(problem,samples,r=np.eye(2)*.25)


@pytest.mark.parametrize('methodClass,kind',[(GLUE,'behavioral'),(SUFI2,'sampling_ensemble'),
                                          (ES,'updated_ensemble'),(IES,'updated_ensemble')])
def testUniformRowsAndSqliteSummary(methodClass,kind,tmp_path):
    problem=makeProblem()
    problem.workDir=str(tmp_path)
    options=dict(seed=12,maxIters=2) if methodClass is IES else {}
    method=methodClass(saveFlag=True,**options)
    result=runMethod(method,problem)
    assert result.sample_kind==kind
    count=len(result.samples)
    assert result.samples.shape==(count,1)
    assert result.simulations.shape==(count,3)
    assert result.scores.shape==(count,)
    np.testing.assert_allclose(result.scores,rmse(problem.obs.ravel(),result.simulations,mask=problem.mask.ravel()))
    np.testing.assert_allclose(result.samples[result.best_index:result.best_index+1],result.bestDecs)
    assert result.best_score==pytest.approx(result.scores[result.best_index])
    if methodClass is GLUE:
        assert result.best_index==1  # Original candidate index is 2.
        assert result.extra['bestIdx']==2
        assert result.weights.shape==(count,)
        assert result.weights.sum()==pytest.approx(1.)
    else:
        assert result.weights is None
    with CalReader(next(tmp_path.rglob('*.sqlite3'))) as reader:
        saved=reader.load_result()
        summary=reader.get_run_summary()
    for field in ['samples','simulations','scores','weights']:
        np.testing.assert_equal(getattr(saved,field),getattr(result,field))
    assert summary['sample_kind']==kind
    assert summary['n_samples']==count
    assert summary['best_score']==result.best_score
    assert summary['best_index']==result.best_index
    assert summary['interval_count']==len(result.intervals)


@pytest.mark.parametrize('methodClass',[GLUE,SUFI2,ES,IES])
def testNewResultArraysDoNotAliasStateOrLegacyArrays(methodClass):
    method=methodClass()
    result=runMethod(method,makeProblem())
    legacy=result.behavioralDecs if methodClass is GLUE else result.posteriorDecs
    expected=legacy.copy()
    stateBefore=deepcopy(method.state.buildResult())
    result.samples[:]=-999
    result.scores[:]=-777
    np.testing.assert_array_equal(legacy,expected)
    np.testing.assert_array_equal(method.state.buildResult().samples,stateBefore.samples)
    np.testing.assert_array_equal(method.state.buildResult().scores,stateBefore.scores)
    for interval in result.intervals:
        interval['lower'][:]=-888
    np.testing.assert_array_equal(method.state.buildResult().diagnostics['scores'],stateBefore.diagnostics['scores'])


def testIntervalsKeepMasksAndIndependentPriorProvenance():
    prior=np.linspace(-3,3,2048)[:,None]
    result=SUFI2().run(makeProblem(),[[0.],[1.],[2.]],eliteSize=1,uncertaintyX=prior,
                      logLikelihood=lambda obs,sim,mask:-.5*((sim[:,0]-obs[0])/.5)**2)
    assert result.sample_kind=='sampling_ensemble'
    assert result.weights is None  # Prior weights must NOT be attached to search rows.
    assert result.uncertainty['weights'].shape==(2048,)
    assert len(result.intervals)==3
    assert result.intervals[0]['kind']=='sampling_envelope'
    assert result.intervals[0]['sample_source']=='samples'
    for interval in result.intervals[1:]:
        assert interval['sample_source']=='uncertainty.samples'
        assert interval['kind']=='prior_importance_weighting'
        if interval['space']=='simulation':
            np.testing.assert_array_equal(interval['indices'],[0,1])
            assert interval['lower'].shape==(2,)
        else:
            np.testing.assert_array_equal(interval['indices'],[0])
    result.uncertainty['weights'][:]=0
    assert result.extra['uncertainty']['weights'].sum()==pytest.approx(1.)
    result.intervals[1]['lower'][:]=-999
    assert result.uncertainty['parameter_lower'][0]!=-999


@pytest.mark.parametrize('methodClass',[ES,IES])
def testUnestimatedIntervalsStayEmpty(methodClass):
    result=runMethod(methodClass(),makeProblem())
    assert result.intervals==[]
    assert result.uncertainty is None


@pytest.mark.parametrize('methodClass',[GLUE,SUFI2])
def testBestScoreSummaryUsesBestMemberNotFirstCandidate(methodClass):
    result=runMethod(methodClass(),makeProblem())
    assert result.diagnostics['scores'][0]>0
    assert result.best_score==0
    assert result.summary()['best_score']==0
