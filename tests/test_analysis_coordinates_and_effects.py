"""Analysis coordinates and effects.

Migrated from test_review_c01_c05.py; original regression provenance is retained below.
"""

from copy import deepcopy
import numpy as np
import pytest
from UQPyL.analysis import Sobol, Morris, FAST, RBDFAST, RSA
from UQPyL.doe import SaltelliDesign, MorrisDesign, FASTDesign, LHS
from UQPyL.problem import Problem, ModelProblem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


# Regression source: test_review_c01_c05.py::testAnalysisUnitCoordinatesMatchReal
@pytest.mark.parametrize(
    "methodClass,design,count",
    [
        (Sobol, SaltelliDesign(secondOrder=False), 256),
        (Morris, MorrisDesign(), 20),
        (FAST, FASTDesign(), 129),
        (RBDFAST, LHS(), 128),
        (RSA, LHS(), 100),
    ],
)
@pytest.mark.parametrize("providedY", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def testAnalysisUnitCoordinatesMatchReal(methodClass, design, count, providedY, positional):
    seen = []

    def objective(x):
        seen.append(x.copy())
        return x[:, :1] + 2 * x[:, 1:2]

    problem = Problem(nInput=2, nObj=1, lb=[10.0, -3.0], ub=[100.0, 2.0], objFunc=objective)
    real, realMeta = design.sampleWithMeta(problem, count, seed=2)
    unit, unitMeta = design.sampleWithMeta(problem, count, seed=2, output="unit")
    savedUnit, savedMeta = unit.copy(), deepcopy(unitMeta)
    target = objective(real) if providedY else None
    expected = methodClass(**QUIET).analyze(problem, real, target, realMeta)
    seen.clear()
    method = methodClass(**QUIET)
    actual = (
        method.analyze(problem, unit, target, unitMeta)
        if positional
        else method.analyze(problem, X=unit, Y=target, meta=unitMeta)
    )
    for metric in expected.metrics:
        np.testing.assert_allclose(actual[metric.name].values, metric.values, atol=1e-11)
    np.testing.assert_allclose(actual.X, real)
    assert actual.meta["output"] == "real"
    assert actual.meta["source_output"] == "unit"
    assert unitMeta == savedMeta
    np.testing.assert_array_equal(unit, savedUnit)
    if providedY:
        assert seen == []
    else:
        assert len(seen) == 1
        np.testing.assert_allclose(seen[0], real)


# Regression source: test_review_c01_c05.py::testAnalysisMixedUnitCoordinatesAndUnknownSpace
def testAnalysisMixedUnitCoordinatesAndUnknownSpace():
    calls = []

    def objective(x):
        calls.append(x.copy())
        assert np.all(np.isin(x[:, 0], [0, 1, 2, 3, 4]))
        assert np.all(np.isin(x[:, 1], [10, 20, 30]))
        return x.sum(axis=1)[:, None]

    problem = Problem(
        nInput=2, nObj=1, lb=[0, 0], ub=[4, 1], varType=[1, 2], varSet={1: [10, 20, 30]}, objFunc=objective
    )
    unit, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, 128, seed=1, output="unit")
    result = Sobol(**QUIET).analyze(problem, unit, meta=meta)
    np.testing.assert_array_equal(result.X, problem.unit_to_space(unit))
    calls.clear()
    with pytest.raises(ValueError, match="output"):
        Sobol(**QUIET).analyze(problem, unit, meta={**meta, "output": "unknown"})
    assert calls == []


# Regression source: test_review_c01_c05.py::testMorrisUnitsPreserveNormalizedEffectsAndDimensionalStatistics
@pytest.mark.parametrize("scale", [1.0, 1e-10, -1e-10, 1e-200, -1e200])
def testMorrisUnitsPreserveNormalizedEffectsAndDimensionalStatistics(scale):
    problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=lambda x: x[:, :1] + 2 * x[:, 1:2])
    x, meta = MorrisDesign().sampleWithMeta(problem, 20, seed=2)
    y = problem.evaluate(x).objs * scale
    saved = y.copy()
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = Morris(**QUIET).analyze(problem, x, y, meta=meta)
    np.testing.assert_allclose(result["S1_norm"].values, [[1 / 3, 2 / 3]], rtol=1e-12)
    np.testing.assert_allclose(result["mu"].values / scale, [[1, 2]], rtol=1e-12)
    np.testing.assert_allclose(result["mu_star"].values / abs(scale), [[1, 2]], rtol=1e-12)
    np.testing.assert_allclose(result["sigma"].values / abs(scale), 0.0, atol=1e-12)
    np.testing.assert_array_equal(result.Y, saved)


# Regression source: test_review_c01_c05.py::testMorrisConstantAndMultiOutputScales
def testMorrisConstantAndMultiOutputScales():
    problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=lambda x: x[:, :1] + 2 * x[:, 1:2])
    x, meta = MorrisDesign().sampleWithMeta(problem, 20, seed=4)
    y = problem.evaluate(x).objs
    outputs = np.hstack([y, y * 1e-200, y * -1e200, np.full_like(y, 1e300)])
    result = Morris(**QUIET).analyze(problem, x, outputs, meta=meta)
    np.testing.assert_allclose(result["S1_norm"].values[:3], [[1 / 3, 2 / 3]] * 3, rtol=1e-12)
    for metric in result.metrics:
        np.testing.assert_array_equal(metric.values[-1], 0.0)


# Regression source: test_review_c01_c05.py::testMorrisNonlinearSigmaMatchesAnalyticalEffects
@pytest.mark.parametrize("scale", [1e-200, -1e200])
def testMorrisNonlinearSigmaMatchesAnalyticalEffects(scale):
    # Two valid trajectories for x0**2 + 2*x1 have x0 effects .5 and 1.5.
    x = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [1.0, 1.0], [0.5, 1.0], [0.5, 0.5]])
    y = (x[:, :1] ** 2 + 2 * x[:, 1:2]) * scale
    problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=lambda x: x[:, :1])
    result = Morris(**QUIET).analyze(problem, x, y, meta={"designType": "morris", "numLevels": 4})
    np.testing.assert_allclose(result["mu"].values / scale, [[1.0, 2.0]], atol=1e-14)
    np.testing.assert_allclose(result["sigma"].values / abs(scale), [[np.sqrt(0.5), 0.0]], atol=1e-14)
    np.testing.assert_allclose(result["S1_norm"].values, [[1 / 3, 2 / 3]], atol=1e-14)


# Regression source: test_review_c01_c05.py::testDecodedAnalysisMetadataAndSamplesSurviveSqlite
def testDecodedAnalysisMetadataAndSamplesSurviveSqlite(tmp_path):
    from UQPyL.analysis.runtime import AnaReader

    problem = Problem(nInput=2, nObj=1, lb=[10.0, 0.0], ub=[100.0, 1.0], objFunc=lambda x: x.sum(axis=1)[:, None])
    problem.workDir = str(tmp_path)
    x, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, 128, seed=1, output="unit")
    result = Sobol(verboseFlag=False, logFlag=False, saveFlag=True).analyze(problem, x, meta=meta)
    with AnaReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
    np.testing.assert_array_equal(loaded.X, result.X)
    assert loaded.meta == result.meta
    assert loaded.meta["output"] == "real"
    assert loaded.meta["source_output"] == "unit"
