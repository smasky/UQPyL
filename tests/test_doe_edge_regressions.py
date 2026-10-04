"""Boundary recovery, Sobol block quality and factorial metadata isolation."""
import warnings

import numpy as np
import pytest
from scipy.stats import qmc
from scipy.spatial.distance import pdist

from UQPyL.doe import FFD, LHS, Sobol, SaltelliDesign
from UQPyL.doe.methods.lhs import _lhs_classic, _lhs_centered
from UQPyL.problem import Problem


def makeProblem():
    return Problem(nInput=3, nObj=1, lb=0., ub=1., objFunc=lambda x: x.sum(axis=1)[:, None])


@pytest.mark.parametrize('criterion,fallback', [('maximin','classic'), ('center_maximin','center')])
def testSinglePointFallsBackWithAccurateMetadata(criterion, fallback):
    sampler = LHS(criterion)
    with pytest.warns(RuntimeWarning, match='no pairwise distance'):
        actual, meta = sampler.sampleWithMeta(makeProblem(), 1, seed=17)
    expected = LHS(fallback).sample(makeProblem(), 1, seed=17)
    np.testing.assert_array_equal(actual, expected)
    assert meta['criterion'] == criterion
    assert meta['effective_criterion'] == fallback
    _, nextMeta = sampler.sampleWithMeta(makeProblem(), 8, seed=17)
    assert nextMeta['effective_criterion'] == criterion


@pytest.mark.parametrize('criterion', ['maximin','center_maximin','correlation'])
@pytest.mark.parametrize('iterations', [0,-1,1.5,True,np.bool_(False),np.nan])
def testInvalidOptimizationCountIsAnExplicitParameterError(criterion, iterations):
    with pytest.raises(ValueError, match='iterations must be a positive integer'):
        LHS(criterion, iterations).sample(makeProblem(), 1, seed=17)


@pytest.mark.parametrize('criterion,generator', [('maximin',_lhs_classic),('center_maximin',_lhs_centered)])
def testOrdinaryMaximinRetainsBestCandidate(criterion, generator):
    rng = np.random.default_rng(17)
    candidates = [generator(8, 3, rng) for _ in range(5)]
    best = max(candidates, key=lambda x: pdist(x).min())
    actual = LHS(criterion, np.int64(5)).sample(makeProblem(), 8, seed=17)
    np.testing.assert_array_equal(actual, best)


@pytest.mark.parametrize('methodClass', [Sobol,SaltelliDesign])
@pytest.mark.parametrize('scramble', [False,True])
@pytest.mark.parametrize('skip', [0,4,16,32,48])
def testSobolBlocksMatchIndependentSequenceAndWarnOnlyOnMisalignment(methodClass, scramble, skip):
    method = methodClass(scramble=scramble,skipValue=skip)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual, meta = method.sampleWithMeta(makeProblem(),16,seed=17,output='unit')
    assert len(caught) == (1 if skip == 4 else 0)
    if caught:
        assert 'not aligned' in str(caught[0].message)
    dimension = 3 if methodClass is Sobol else 6
    seed = np.random.default_rng(17).integers(1,1000000) if scramble else None
    reference = qmc.Sobol(dimension,scramble=scramble,seed=seed)
    if skip:
        reference.fast_forward(skip)
    base = reference.random(16)
    if methodClass is Sobol:
        np.testing.assert_array_equal(actual,base)
    else:
        blocks = actual.reshape(16,5,3)
        np.testing.assert_array_equal(blocks[:,0],base[:,:3])
        np.testing.assert_array_equal(blocks[:,-1],base[:,3:])
        for axis in range(3):
            expected = base[:,:3].copy()
            expected[:,axis] = base[:,axis+3]
            np.testing.assert_array_equal(blocks[:,axis+1],expected)
        actual = blocks[:,0]
    counts = np.bincount((actual[:,0]*16).astype(int),minlength=16)
    assert np.array_equal(counts,np.ones(16)) == (skip != 4)
    assert meta['skipValue'] == skip


@pytest.mark.parametrize('methodClass', [Sobol,SaltelliDesign])
def testNonPowerSizeProducesOneQualityWarningAndStillReturns(methodClass):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = methodClass().sample(makeProblem(),15,seed=17)
    assert len(caught) == 1 and 'not a power of 2' in str(caught[0].message)
    assert actual.shape == (15 if methodClass is Sobol else 75,3)


@pytest.mark.parametrize('methodClass', [Sobol,SaltelliDesign])
@pytest.mark.parametrize('skip', [-1,True,1.5])
def testIllegalSkipFailsBeforeSequenceGeneration(methodClass,skip):
    with pytest.raises((ValueError,TypeError),match='skipValue'):
        methodClass(skipValue=skip).sample(makeProblem(),16)


def testFactorialMetadataIsIndependentOfInputAndOtherResults():
    levels = [2,3,4]
    sampler = FFD()
    x, first = sampler.sampleWithMeta(makeProblem(),levels)
    _, second = sampler.sampleWithMeta(makeProblem(),levels)
    levels[0] = 9
    assert first['levels'] == second['levels'] == [2,3,4]
    first['levels'][1] = 8
    assert second['levels'] == [2,3,4] and levels == [9,3,4]
    assert len(x) == np.prod(second['levels']) == 24
