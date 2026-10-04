"""Whole user workflows, including physical units, budgets, and persistence."""

import json

import numpy as np
import pytest

from UQPyL.analysis import Sobol
from UQPyL.analysis.runtime import AnaReader
from UQPyL.calibration import IES, CalReader
from UQPyL.doe import LHS, SaltelliDesign
from UQPyL.inference import MH, InfReader
from UQPyL.optimization.soea import GA
from UQPyL.optimization.runtime import OptReader
from UQPyL.problem import Problem, ModelProblem
from UQPyL.surrogate import AutoTuner, StandardScaler, r_square
from UQPyL.surrogate.rbf import RBF


@pytest.mark.parametrize("coordinate", ["real", "unit"])
def testDesignEvaluationAnalysisAndReader(coordinate, tmp_path):
    calls = []

    def objective(x):
        calls.append(len(x))
        assert np.all((x >= [10.0, 0.0]) & (x <= [20.0, 1.0]))
        return x[:, :1] + 2 * x[:, 1:2]

    problem = Problem(
        nInput=2,
        nObj=1,
        lb=[10.0, 0.0],
        ub=[20.0, 1.0],
        objFunc=objective,
        xLabels=["width", "ratio"],
        objLabels=["response"],
    )
    problem.workDir = str(tmp_path)
    x, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, 512, seed=13, output=coordinate)
    real = problem.unit_to_space(x) if coordinate == "unit" else x
    y = problem.evaluate(real).objs
    before = sum(calls)
    result = Sobol(verboseFlag=False, saveFlag=True).analyze(problem, x, Y=y, meta=meta)
    assert sum(calls) == before
    np.testing.assert_allclose(result.X, real)
    np.testing.assert_array_equal(result.Y, y)
    np.testing.assert_allclose(result["S1"].values.ravel(), [100 / 104, 4 / 104], atol=0.01)
    assert result["S1"].colLabels == ["width", "ratio"]
    with AnaReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
        assert reader.get_run_summary()["run_id"] == result.runId
        np.testing.assert_allclose(loaded["S1"].values, result["S1"].values)
        np.testing.assert_allclose(loaded.toDict()["X"], real)


def testTuningOptimizationTrueModelCheckAndReader(tmp_path):
    calls = []

    def objective(x):
        calls.append(len(x))
        return (x - 13.0) ** 2 + 2.0

    realProblem = Problem(nInput=1, nObj=1, lb=10.0, ub=20.0, objFunc=objective)
    x = LHS().sample(realProblem, 24, seed=15)
    y = realProblem.evaluate(x).objs
    model = RBF(
        scalers=(StandardScaler(), StandardScaler()),
        C_smooth_attr={"lb": 0.0, "ub": 0.1, "type": "float", "log": False},
    )
    tuner = AutoTuner(model)
    tuner.gridTune(
        x,
        y,
        paraGrid={"C_smooth": [0.0, 1e-5, 1e-3]},
        splitIndices=(np.arange(18), np.arange(18, 24)),
        tuneMode="joint",
        seed=17,
    )
    assert sum(calls) == 24 and tuner.getReport()["fit_calls"] == 4
    json.dumps(tuner.getReport(), allow_nan=False)
    heldOut = np.linspace(10.5, 19.5, 11)[:, None]
    assert r_square((heldOut - 13.0) ** 2 + 2.0, model.predict(heldOut)) > 0.99
    surrogateProblem = Problem(nInput=1, nObj=1, lb=10.0, ub=20.0, objFunc=model.predict)
    surrogateProblem.workDir = str(tmp_path)
    method = GA(nPop=12, maxIters=3, maxFEs=48, saveFlag=True, saveFreq=1, verboseFlag=False)
    result = method.run(surrogateProblem, seed=19)
    assert result.FEs == 48 and sum(calls) == 24
    np.testing.assert_allclose(result.bestObjs, model.predict(result.bestDecs))
    checked = realProblem.evaluate(result.bestDecs).objs
    assert sum(calls) == 25
    np.testing.assert_allclose(checked, result.bestObjs, atol=0.03)
    with OptReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
        assert reader.get_run_summary()["stop_reason"] == result.stopReason
        np.testing.assert_array_equal(loaded.bestDecs, result.bestDecs)
        np.testing.assert_array_equal(loaded.toDict()["best_objs"], result.bestObjs)
    np.savez(tmp_path / "optimization.npz", **method.state.toNpzPayload())
    with np.load(tmp_path / "optimization.npz", allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["bestDecs"], result.bestDecs)


def testSimulationCalibrationInferenceDiagnosticsAndReaders(tmp_path):
    time = np.linspace(0.0, 2.0, 9)
    obs = (1.2 + 1.8 * time)[:, None]
    mask = np.zeros_like(obs, dtype=bool)
    mask[3] = True
    obs[3] = np.nan
    calls = []

    def simulate(x):
        assert np.all((x >= 0.0) & (x <= 3.0))
        calls.append(len(x))
        sims = (x[:, :1] + x[:, 1:2] * time)[:, :, None]
        sims[:, 3] = np.nan
        return (sims).reshape(len(x), -1)

    def loss(x, context):
        residual = (context.sims - context.obs)[:, ~context.mask]
        return 0.5 * np.sum((residual / 0.2) ** 2, axis=1, keepdims=True)

    problem = ModelProblem(
        nInput=2, nObj=1, lb=0.0, ub=3.0, simFunc=simulate, objFunc=loss, obs=(obs).reshape(-1), mask=None if mask is None else mask.reshape(-1)
    )
    problem.workDir = str(tmp_path)
    initial = LHS().sample(problem, 6, seed=21)
    calibration = IES(maxIters=2, lam=0.1, saveFlag=True).run(problem, initial)
    assert calls == [6, 6, 6]
    assert np.all(np.isfinite(calibration.posteriorDecs))
    np.testing.assert_allclose(calibration.bestDecs, [[1.2, 1.8]], atol=0.08)
    with CalReader(CalReader.list_runs(tmp_path)[0]["db_path"]) as reader:
        restored = reader.load_result()
        np.testing.assert_array_equal(restored.posteriorDecs, calibration.posteriorDecs)
        np.testing.assert_allclose(restored.bestSim, calibration.bestSim, equal_nan=True)
    np.savez(
        tmp_path / "calibration.npz", posterior_decs=calibration.posteriorDecs, posterior_sims=calibration.posteriorSims
    )

    before = sum(calls)
    result = MH(nChains=4, warmUp=2, maxIters=32, saveFlag=True, saveFreq=8, verboseFlag=False).run(
        problem, gamma=0.08, seed=23
    )
    assert result.FEs == sum(calls) - before
    beforeDiagnostics = sum(calls)
    report = result.computeDiagnostics()
    assert sum(calls) == beforeDiagnostics
    json.dumps(report, allow_nan=False)
    assert result.decs.shape == (4, 32, 2)
    evaluated = problem.evaluate(result.decs.reshape(-1, 2)).objs.reshape(4, 32, 1)
    np.testing.assert_allclose(result.objs, evaluated)
    assert len(CalReader.list_runs(tmp_path)) == len(InfReader.list_runs(tmp_path)) == 1
    with InfReader(InfReader.list_runs(tmp_path)[0]["db_path"]) as reader:
        loaded = reader.load_result()
        np.testing.assert_array_equal(loaded.decs, result.decs)
        assert loaded.computeDiagnostics() == report
        assert reader.get_run_summary()["run_id"] == result.runId
        assert reader.load_partial_result()["complete"] is False
    exported = result.toDict()
    np.testing.assert_array_equal(exported["log_prob"], result.logProb)
    assert exported["diagnostics"]["chains"] == report
