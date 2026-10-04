"""Observation coordinates stay explicit from model evaluation to persisted results."""

import sqlite3
from contextlib import closing

import numpy as np
import pytest

from UQPyL.problem import ModelProblem
from UQPyL.calibration import ES, IES, GLUE, SUFI2, CalReader


def makeProblem(**options):
    return ModelProblem(**(dict(nInput=2, lb=-5, ub=5, obs=np.array([1.0, 2.0]), simFunc=lambda X: X.copy()) | options))


@pytest.mark.parametrize("obs", [np.array(1.0), np.ones((1, 2)), np.ones((2, 1)), np.ones((1, 1, 2)), np.array([])])
def testObservationsMustBeNonemptyVector(obs):
    with pytest.raises(ValueError, match="1D|at least one"):
        makeProblem(obs=obs)


@pytest.mark.parametrize("mask", [np.array(False), np.zeros((1, 2), bool), np.zeros((2, 1), bool), np.zeros(3, bool)])
def testMaskMustMatchVectorExactly(mask):
    with pytest.raises(ValueError, match="Mask shape"):
        makeProblem(mask=mask)


@pytest.mark.parametrize("shape", [(), (2,), (1, 1, 2), (1, 3), (1, 0), (2, 2)])
@pytest.mark.parametrize("entry", ["evaluate", "simulate", "simFunc", "flattenSim"])
def testAllSimulationEntryPointsRejectInvalidShapes(shape, entry):
    values = np.ones(shape)
    problem = makeProblem(simFunc=lambda X: values)
    # flattenSim does not know the caller's sample count; a valid two-row matrix is legal there.
    if entry == "flattenSim" and shape == (2, 2):
        np.testing.assert_array_equal(problem.flattenSim(values), values)
    else:
        with pytest.raises(ValueError, match="2D|column|first dimension"):
            getattr(problem, entry)(values if entry == "flattenSim" else [1.0, 2.0])


@pytest.mark.parametrize("nSamples,nObs", [(1, 1), (3, 1), (1, 3), (3, 3)])
def testSingletonAxesArePreserved(nSamples, nObs):
    values = np.arange(nSamples * nObs, dtype=float).reshape(nSamples, nObs)
    problem = makeProblem(
        obs=np.arange(nObs, dtype=float),
        simFunc=lambda X: values,
        objFunc=lambda X, ctx: np.mean((ctx.sims - ctx.obs) ** 2, axis=1, keepdims=True),
    )
    result = problem.evaluate(np.zeros((nSamples, 2)))
    assert result.sims.shape == (nSamples, nObs)
    assert result.objs.shape == (nSamples, 1)
    np.testing.assert_allclose(result.objs[:, 0], np.mean((values - np.arange(nObs)) ** 2, axis=1))


def testSimulationOnlyNeedsNoObservationMetadata():
    problem = makeProblem(obs=None)
    np.testing.assert_array_equal(problem.evaluate([1.0, 2.0], target="sims").sims, [[1.0, 2.0]])
    assert problem.nObs is None
    with pytest.raises(ValueError, match="requires observation"):
        makeProblem(obs=None, mask=np.zeros(2, bool))


def testObservationLabelsAreNotPartOfTheInterface():
    import inspect
    from UQPyL.calibration import CalResult

    assert "obsLabels" not in inspect.signature(ModelProblem).parameters
    assert "obsLabels" not in CalResult.__dataclass_fields__
    assert not hasattr(makeProblem(), "obsLabels")


def testCustomEvaluationCannotReturnLegacyTensor():
    from UQPyL.problem import Eval

    class TensorProblem(ModelProblem):
        def evaluate(self, X, target=None):
            return Eval(sims=np.ones((len(X), 2, 1)))

    problem = TensorProblem(nInput=2, lb=0, ub=5, obs=np.ones(2), simFunc=lambda X: X)
    with pytest.raises(ValueError, match="2D"):
        problem.evaluate([1.0, 2.0], target="sims")


@pytest.mark.parametrize("methodClass", [GLUE, SUFI2, ES, IES])
def testObservationPermutationKeepsScoresAndParameterUpdate(methodClass):
    """Move columns, obs, mask and effective R together; parameter answers stay invariant."""
    matrix = np.array([[1.0, 2.0, -1.0, 0.5], [2.0, -0.5, 3.0, 1.0]])
    obs = np.array([0.7, np.nan, -0.8, 0.1])
    mask = np.array([False, True, False, False])
    samples = np.random.default_rng(16).normal(size=(32, 2))
    results = []
    for order in (np.arange(4), np.array([3, 1, 0, 2])):

        def simulate(X):
            values = X @ matrix
            values[:, 1] = np.nan
            return values[:, order]

        problem = makeProblem(obs=obs[order], mask=mask[order], simFunc=simulate)
        method = methodClass(maxIters=2, seed=17) if methodClass is IES else methodClass()
        # Zero R avoids coordinate-dependent random draws in IES; nonzero R alignment is tested below for ES.
        options = (
            {"r": np.zeros((3, 3))}
            if methodClass in (ES, IES)
            else ({"threshold": 100.0} if methodClass is GLUE else {"eliteSize": 12, "seed": 17})
        )
        results.append(method.run(problem, samples, **options))
    np.testing.assert_allclose(results[0].samples, results[1].samples, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(results[0].scores, results[1].scores, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(results[0].simulations[:, [3, 1, 0, 2]], results[1].simulations, atol=2e-13, rtol=2e-13)


def testEsUsesEffectiveObservationCovarianceOrder():
    samples = np.random.default_rng(5).normal(size=(40, 2))
    obs = np.array([0.1, 999.0, 0.7])
    covariance = np.array([[0.2, 0.05], [0.05, 0.6]])
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 100.0, 1.0]])
    results = []
    for order, validOrder in (([0, 1, 2], [0, 1]), ([2, 1, 0], [1, 0])):
        problem = makeProblem(
            obs=obs[order], mask=np.array([False, True, False]), simFunc=lambda X: (X @ matrix)[:, order]
        )
        results.append(ES().run(problem, samples, r=covariance[np.ix_(validOrder, validOrder)]))
    np.testing.assert_allclose(results[0].samples, results[1].samples, rtol=2e-13, atol=2e-13)


def storageSimulate(X):
    return np.column_stack((X[:, 0], X.sum(axis=1), X[:, 1]))


@pytest.mark.parametrize("methodClass", [GLUE, SUFI2, ES, IES])
def testVectorMetadataRoundTripsThroughSqlite(methodClass, tmp_path):
    problem = makeProblem(obs=np.array([1.0, 99.0, 2.0]), mask=np.array([False, True, False]), simFunc=storageSimulate)
    problem.workDir = str(tmp_path)
    samples = np.random.default_rng(10).normal(size=(32, 2))
    method = methodClass(saveFlag=True, maxIters=1, seed=17) if methodClass is IES else methodClass(saveFlag=True)
    options = (
        {"r": np.eye(2)}
        if methodClass in (ES, IES)
        else ({"threshold": 100.0} if methodClass is GLUE else {"eliteSize": 12, "seed": 17})
    )
    result = method.run(problem, samples, **options)
    assert result.obs.shape == result.mask.shape == (3,)
    assert result.nObs == 3
    assert not hasattr(result, "obsLabels")
    assert "obs_labels" not in result.summary()
    assert not hasattr(result, "nTime") and not hasattr(result, "nSeries")
    (path,) = (tmp_path / "Result").glob("*.sqlite3")
    with CalReader(path) as reader:
        loaded = reader.load_result()
        savedProblem = reader.load_problem()
        summary = reader.get_run_summary()
        assert summary["n_obs"] == summary["n_output"] == 3
        assert "obs_labels" not in summary
        assert "n_time" not in summary and "n_series" not in summary
        assert "nTime" not in reader.get_run() and "nSeries" not in reader.get_run()
        np.testing.assert_array_equal(loaded.obs, result.obs)
        np.testing.assert_array_equal(loaded.mask, result.mask)
        np.testing.assert_array_equal(loaded.simulations, result.simulations)
        assert savedProblem.obs.shape == savedProblem.mask.shape == (3,)
        assert not hasattr(savedProblem, "obsLabels")
        assert not hasattr(loaded, "obsLabels")


def testReaderRejectsOldObservationSchema(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE runtimeMeta (name TEXT, value TEXT)")
        conn.execute("INSERT INTO runtimeMeta VALUES ('domain', 'calibration')")
    with pytest.raises(ValueError, match="1D observation.*new run"):
        CalReader(path)
    assert CalReader.list_runs(tmp_path) == []
