"""Independent controls for optimizer transitions, incumbents and numerical geometry."""
import math

import numpy as np
import pytest

from UQPyL.optimization import Population
from UQPyL.optimization.core import crowdingDist, gaOperator, gaOperatorHalf, NDSort
from UQPyL.optimization.metric import GD, IGD, HV
from UQPyL.optimization.moea import MOEAD, NSGAII, NSGAIII, RVEA
from UQPyL.optimization.soea import ABC, DE, GA, SCE_UA, ML_SCE_UA
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


@pytest.mark.parametrize('cls', [SCE_UA, ML_SCE_UA])
@pytest.mark.parametrize('direction', ['min', 'max'])
@pytest.mark.parametrize('seed', [7, 19, 31])
def testSceRanksAndAllEvaluatedIncumbents(cls, direction, seed):
    seen = []
    sign = 1 if direction == 'min' else -1
    def objective(x):
        scores = np.sum((x - .3)**2, axis=1, keepdims=True)
        seen.extend(scores[:, 0])
        return sign * scores
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=objective, optType=direction)
    method = cls(ngs=2, maxIters=5, maxFEs=1000, tolerate=None, **QUIET)
    original = method._cce
    def traced(simplex, *args):
        assert np.all(np.diff(simplex.objs[:, 0]) >= 0)
        return original(simplex, *args)
    method._cce = traced
    result = method.run(problem, seed=seed)
    assert result.FEs == len(seen)
    assert float(result.bestObjs[0, 0]) * sign == min(seen)
    np.testing.assert_allclose(np.sum((result.bestDecs - .3)**2, axis=1), min(seen))


@pytest.mark.parametrize('cls', [SCE_UA, ML_SCE_UA])
@pytest.mark.parametrize('dimension,options,expected', [
    (4, {}, 18), (2, dict(npg=8, nps=3, nspl=2), 16),
])
def testScePopulationParametersAreHonored(cls, dimension, options, expected):
    problem = Problem(nInput=dimension, nObj=1, lb=0, ub=1,
                      objFunc=lambda x: np.sum(x*x, axis=1, keepdims=True))
    method = cls(ngs=2, maxIters=0, **options, **QUIET)
    assert method.run(problem, seed=2).FEs == expected


@pytest.mark.parametrize('cls', [SCE_UA, ML_SCE_UA])
@pytest.mark.parametrize('contract', [False, True])
def testSceReflectionAndContractionUseCentroidExcludingWorst(cls, contract):
    evaluated = []
    def objective(x):
        evaluated.extend(x[:, 0].tolist())
        return np.full((len(x), 1), 3. if contract and len(evaluated) == 1 else 1.5)
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=objective)
    method = cls(**QUIET)
    method.setup(problem, 1)
    simplex = Population([[.4], [.5], [.6]], [[0.], [1.], [2.]])
    if cls is ML_SCE_UA:
        method._cce(simplex, simplex[0], 1., .5, 0.)
    else:
        method._cce(simplex, 1., .5)
    np.testing.assert_allclose(evaluated, [.3, .525] if contract else [.3])


def testSingleEvaluatedCandidateSurvivesDiscardWithoutChangingIterationAccounting():
    method = GA(nPop=2, tolerate=1e-6, maxTolerates=5, **QUIET)
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    method.setup(problem, 1)
    pop = method.initPop(2, initialPop=[[.7], [.9]])
    method.update(pop)
    method.evaluate(Population([[.1]]))
    method.evaluate(Population([[.8]]))
    assert method.iters == 0 and method.state.bestObj == .7
    method.update(pop, completed=True)
    result = method.buildResult()
    assert result.bestObjs[0, 0] == .1
    assert result.FEs == 4 and result.iters == 1 and result.appearFEs == 3
    assert result.appearIters == 1 and method.tolerateTimes == 0
    method.evaluate(Population([[0.]]))
    method.setup(problem, 2)
    pop = method.initPop(2, initialPop=[[.7], [.9]])
    method.update(pop)
    assert method.state.bestObj == .7


def testAbcSuccessResetsFailureCount():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    method = ABC(nPop=4, **QUIET)
    method.setup(problem, 0)
    pop = method.initPop(4, initialPop=[[.6], [.9], [.8], [.7]])
    pop, count = method.updateEmployedBees(pop, np.array([1, 0, 0, 0]), np.array([6., 0, 0, 0]))
    assert pop.objs[0, 0] < .6 and count[0] == 0


def testAbcRepeatedSourceFailuresAccumulate():
    class FixedRng:
        def choice(self, a, size, p):
            return np.zeros(size, dtype=int)
        def permutation(self, index):
            return np.array([0, 2, 3, 1])
        def random(self, shape):
            return np.full(shape, .5)
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    method = ABC(nPop=4, **QUIET)
    method.setup(problem, 0)
    pop = method.initPop(4, initialPop=[[.1], [.2], [.3], [.4]])
    method.rng = FixedRng()
    _, _, count = method.updateUnemployedBees(pop, np.array([1, 0, 0, 0]), np.zeros(4))
    assert count[0] == 3


def testAbcAllEmployedAndRecoverableSmallPopulation():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    result = ABC(nPop=4, employedRate=1, maxIters=2, **QUIET).run(problem, seed=1)
    assert result.iters == 2 and result.FEs == 12
    with pytest.warns(RuntimeWarning, match='one employed bee'):
        result = ABC(nPop=2, maxIters=1, **QUIET).run(problem, seed=1)
    assert result.iters == 1


def testPopulationNegativeIndexAndFloatingReplacement():
    pop = Population([[1], [2]], [[3], [4]], [[0], [1]])
    np.testing.assert_array_equal(pop[-1].decs, [[2]])
    with pytest.raises(IndexError):
        pop[-3]
    pop.replace(-1, Population([[1.5]], [[2.5]], [[.5]]))
    np.testing.assert_array_equal(pop[-1].decs, [[1.5]])
    np.testing.assert_array_equal(pop[-1].objs, [[2.5]])
    np.testing.assert_array_equal(pop[-1].cons, [[.5]])


@pytest.mark.parametrize('size', [1, 3, 5, 8])
def testGaOperatorRetainsOddPopulationAndFixedCoordinates(size):
    decs = np.full((size, 2), .4)
    decs[:, 0] = .2
    output = gaOperator(decs, np.array([[.2, 1.]]), np.array([[.2, 0.]]),
                        rng=np.random.default_rng(3))
    assert output.shape == decs.shape
    assert np.all(output[:, 0] == .2) and np.all((output >= 0) & (output <= 1))
    half = gaOperatorHalf(np.vstack([decs, decs]), np.array([[.2, 1.]]), np.array([[.2, 0.]]),
                          1, 20, 1, 20, rng=np.random.default_rng(3))
    assert np.all(half[:, 0] == .2)


def testDeZeroCrossoverStillAttemptsOneDonorCoordinate():
    method = DE(cr=0, **QUIET)
    problem = Problem(nInput=3, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    method.setup(problem, 1)
    parent = np.full((5, 3), .5)
    result = method._deOperator(parent, np.full((5, 3), .7), np.full((5, 3), .3), 0., .5)
    np.testing.assert_array_equal(np.sum(result != parent, axis=1), np.ones(5))


@pytest.mark.parametrize('function', [GD, IGD])
@pytest.mark.parametrize('scale', [1., 1e-200, 1e200, 1e308])
def testDistancesMatchIndependentHypot(function, scale):
    actual = function([[scale, scale]], [[0., 0.]])
    assert math.isclose(actual, math.hypot(scale, scale), rel_tol=5e-15, abs_tol=0)


def testDistanceMeanRepresentableWhenIndividualDistanceOverflows():
    assert GD([[1e308], [-1e308]], [[-1e308]]) == 1e308
    # A global scale must not erase tiny nearest distances beside remote points.
    expected = 5e-201
    actual = GD([[1e-200], [1e200]], [[0.], [1e200]])
    assert math.isclose(actual, expected, rel_tol=5e-15, abs_tol=0)
    with pytest.warns(RuntimeWarning, match='floating-point range'):
        assert math.isinf(GD([[1e308]], [[-1e308]]))


@pytest.mark.parametrize('scale', [1e-200, 1., 1e200, 1e308])
def testCrowdingKeepsDimensionlessDistances(scale):
    points = np.array([[-1., 1.], [-.5, .5], [0., 0.], [.5, -.5], [1., -1.]]) * scale
    np.testing.assert_allclose(crowdingDist(points), [np.inf, 1., 1., 1., np.inf])


@pytest.mark.parametrize('scale', [1e-200, 1., 1e200])
def testRveaKeepsDirectionsAcrossCommonScale(scale):
    points = np.array([[0., 4.], [1., 3.], [2., 2.], [3., 1.], [4., 0.]])
    vectors = np.array([[1., 0.], [.5, .5], [0., 1.]])
    method = RVEA(**QUIET)
    np.testing.assert_array_equal(method.environmentSelection(points * scale, vectors, .5), [4, 2, 0])
    actual = method.updateReferenceVector(points * scale, vectors)
    np.testing.assert_array_equal(method.environmentSelection(points * scale, actual, .5), [4, 2, 0])


@pytest.mark.parametrize('cls,options', [(NSGAII, {}), (NSGAIII, {}), (RVEA, {})] +
                         [(MOEAD, dict(aggregation=kind)) for kind in ['PBI', 'TCH', 'TCH_N', 'TCH_M']])
def testMultiObjectivePublicRunsKeepScaleEquivalentSearch(cls, options):
    runs = []
    for scale in [1., 2.**-650, 2.**650]:
        problem = Problem(nInput=2, nObj=2, lb=0, ub=1,
                          objFunc=lambda x: scale * np.column_stack([np.sum(x*x, axis=1),
                                                                     np.sum((x - 1)**2, axis=1)]))
        result = cls(nPop=12, maxIters=3, maxFEs=1000, hvFlag=False, **options, **QUIET).run(problem, seed=19)
        runs.append(result)
    for result in runs[1:]:
        np.testing.assert_array_equal(result.bestDecs, runs[0].bestDecs)
        assert result.FEs == runs[0].FEs


@pytest.mark.parametrize('widths', [[1e200, 1e200, 1e-300], [1e-200, 1e-200, 1e300],
                                  [1e200, 1e200, 1e-200, 1e-200]])
def testHypervolumeAvoidsIntermediateVolumeRangeFailures(widths):
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 100
        expected = Decimal(1)
        for value in widths:
            expected *= Decimal.from_float(value)
        expected = float(expected)
    actual = HV(np.zeros((1, len(widths))), widths, normalize=False,
                nSamples=32, rng=np.random.default_rng(1))
    assert math.isclose(actual, expected, rel_tol=5e-15, abs_tol=0)


def testHypervolumeNormalizationAtExtremePositiveAndNegativeValues():
    assert HV([[-1e308, -1e308]], [1e308, 1e308]) == 4.
    with pytest.warns(RuntimeWarning, match='floating-point range'):
        assert math.isinf(HV([[0., 0.]], [1e200, 1e200], normalize=False))


@pytest.mark.parametrize('cls,options', [
    (GA, dict(nPop=0)), (DE, dict(cr=-.1)), (RVEA, dict(fr=0)),
    (MOEAD, dict(aggregation='unknown')), (SCE_UA, dict(nps=9, npg=3)),
    (ABC, dict(employedRate=0)), (GA, dict(disM=np.nan)),
])
def testInvalidControlsFailBeforeModelEvaluation(cls, options):
    calls = []
    nObj = 2 if cls in (RVEA, MOEAD) else 1
    problem = Problem(nInput=2, nObj=nObj, lb=0, ub=1, objFunc=lambda x: calls.append(x))
    with pytest.raises(ValueError):
        cls(**options, **QUIET).run(problem, seed=1)
    assert calls == []


@pytest.mark.parametrize('field', ['objs', 'cons'])
@pytest.mark.parametrize('value', [np.nan, -np.inf, 1j])
@pytest.mark.parametrize('preEvaluated', [False, True])
def testNonfiniteOrComplexEvaluationsCannotBecomeOptimizationResults(field, value, preEvaluated):
    y = np.array([[value if field == 'objs' else 1.]])
    c = np.array([[value if field == 'cons' else 0.]])
    problem = Problem(nInput=1, nObj=1, nCon=1, lb=0, ub=1,
                      objFunc=lambda x: np.repeat(y, len(x), axis=0),
                      conFunc=lambda x: np.repeat(c, len(x), axis=0))
    initial = Population([[.5]], y, c) if preEvaluated else None
    with pytest.raises(ValueError, match='finite real'):
        GA(nPop=1, maxIters=0, **QUIET).run(problem, seed=1, initialPop=initial)


def testPreEvaluatedFirstIncumbentRetainsEqualScoreAndCopies():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: np.ones((len(x), 1)))
    initial = Population([[.5]], [[1.]])
    result = GA(nPop=4, maxIters=0, **QUIET).run(problem, seed=1, initialPop=initial)
    np.testing.assert_array_equal(result.bestDecs, [[.5]])
    assert result.appearFEs == 0 and result.FEs == 3
    np.testing.assert_array_equal(initial.decs, [[.5]])


def testSinglePendingIncumbentUsesWeightedFeasibilityFirst():
    problem = Problem(nInput=1, nObj=1, nCon=1, lb=0, ub=1,
                      objFunc=lambda x: -x[:, :1], conFunc=lambda x: x[:, :1] - .5)
    method = GA(nPop=2, **QUIET)
    method.setup(problem, 1)
    pop = method.initPop(2, initialPop=[[.8], [.9]])
    method.update(pop)
    method.evaluate(Population([[.4]]))
    method.evaluate(Population([[1.]]))
    method.update(pop, completed=True)
    result = method.buildResult()
    assert result.bestFeasible and result.bestDecs[0, 0] == .4


def testRveaSupportsIterationOnlyBudget():
    problem = Problem(nInput=1, nObj=2, lb=0, ub=1,
                      objFunc=lambda x: np.column_stack([x[:, 0], 1 - x[:, 0]]))
    result = RVEA(nPop=8, maxFEs=None, maxIters=3, **QUIET).run(problem, seed=1)
    assert result.iters == 3 and result.stopReason == 'max_iters'


def testZeroRequestedFrontsAndEmptyInput():
    ranks, last = NDSort([[1., 2.], [2., 1.]], [[0.], [1.]], nSort=0)
    assert np.all(np.isinf(ranks)) and last == 0
    ranks, last = NDSort(np.empty((0, 2)))
    assert len(ranks) == 0 and last == 0


@pytest.mark.parametrize('direction,sign', [('min', 1), ('max', -1)])
def testScalarWorstDirectionInfinityPenaltyIsPreserved(direction, sign):
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, optType=direction,
                      objFunc=lambda x: sign * np.where(x < .5, np.inf, x))
    result = GA(nPop=4, maxIters=2, **QUIET).run(problem, seed=3,
                                                 initialPop=[[.1], [.3], [.6], [.9]])
    assert np.isfinite(result.bestObjs).all() and result.bestDecs[0, 0] >= .5


def testAutomaticHypervolumeReferenceReportsUnrepresentableMargin():
    with pytest.warns(RuntimeWarning, match='Automatic hypervolume reference'):
        value = HV([[1.7e308, 1.7e308]])
    assert np.isfinite(value) and value > 0


@pytest.mark.parametrize('scores', [np.array([[-2**63], [0]], dtype=np.int64),
                                  np.array([[0], [2**63]], dtype=np.uint64)])
def testPreEvaluatedIntegerScoresConvertBeforeMaximization(scores):
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, optType='max', objFunc=lambda x: x)
    population = Population([[.2], [.8]], scores)
    result = GA(nPop=2, maxIters=0, **QUIET).run(problem, seed=1, initialPop=population)
    assert result.bestDecs[0, 0] == .8 and result.FEs == 0
    assert result.bestObjs[0, 0] == float(scores[1, 0])
    np.testing.assert_array_equal(population.objs, scores)
