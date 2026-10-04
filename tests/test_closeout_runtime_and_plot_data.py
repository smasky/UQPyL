"""Saved public results and the numerical data displayed by plots."""

from dataclasses import fields, is_dataclass
import sqlite3
from contextlib import closing

import matplotlib.pyplot as plt
import numpy as np
import pytest

from UQPyL.analysis import Morris
from UQPyL.analysis.runtime import AnaMetric, AnaResult, AnaReader
from UQPyL.calibration import GLUE, CalReader
from UQPyL.doe import MorrisDesign
from UQPyL.inference import MH, InfReader
from UQPyL.optimization.soea import GA
from UQPyL.optimization.runtime import OptReader
from UQPyL.problem import Problem, ModelProblem
from UQPyL.viz import plot_sa, plot_op_curve, plot_op_curve_stat, plot_op_pareto, plot_infer_trace


@pytest.fixture(autouse=True)
def closeFigures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def compareTree(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        for field in fields(left):
            compareTree(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compareTree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compareTree(a, b)
    elif isinstance(left, float) and np.isnan(left):
        assert np.isnan(right)
    else:
        assert left == right


@pytest.mark.parametrize("domain", ["optimization", "inference", "analysis", "calibration"])
def testCompletePublicResultsRoundtripAndReuse(domain, tmp_path, monkeypatch):
    p = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=lambda x: x[:, :1] + 2 * x[:, 1:2], optType="max")
    flags = dict(verboseFlag=False, saveFlag=True)
    if domain == "optimization":
        method = GA(nPop=6, maxFEs=18, saveFreq=1, **flags)

        def run():
            return method.run(p, seed=17)

        readerClass = OptReader
    elif domain == "inference":
        method = MH(nChains=3, warmUp=2, maxIters=12, saveFreq=2, **flags)

        def run():
            return method.run(p, seed=17)

        readerClass = InfReader
    elif domain == "analysis":
        method = Morris(**flags)
        x, meta = MorrisDesign().sampleWithMeta(p, 6, seed=17)

        def run():
            return method.analyze(p, x, meta=meta)

        readerClass = AnaReader
    else:
        p = ModelProblem(nInput=2, lb=-1.0, ub=1.0, simFunc=lambda x: x, obs=np.zeros(2))
        method = GLUE(**flags)
        x = np.random.default_rng(17).uniform(-1, 1, (30, 2))

        def run():
            return method.run(p, x, threshold=10.0)

        readerClass = CalReader
    p.workDir = str(tmp_path)
    originalReset = method.state.reset

    def delayedReset():
        originalReset()
        method.state.createdAt = "2000-01-01T00:00:00"

    monkeypatch.setattr(method.state, "reset", delayedReset)
    first = run()
    assert method.session is None
    path = next(tmp_path.rglob("*.sqlite3"))
    with readerClass(path) as reader:
        loaded = reader.load_result()
        assert reader.get_run()["status"] == "finished"
        assert reader.get_run()["createdAt"] == first.createdAt
        if domain == "optimization":
            # OptReader restores saved snapshots, not every in-memory population.
            for key in (
                "bestDecs",
                "bestObjs",
                "bestCons",
                "bestMetric",
                "bestFeasible",
                "appearFEs",
                "appearIters",
                "FEs",
                "iters",
                "runtime",
                "stopReason",
                "createdAt",
            ):
                compareTree(getattr(first, key), getattr(loaded, key))
            compareTree(first.history.iterToFEs, loaded.history.iterToFEs)
            compareTree(first.history.bestObjHistory, loaded.history.bestObjHistory)
        else:
            compareTree(first, loaded)
        if domain == "optimization":
            _, ax = plot_op_curve({"saved": reader})
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), first.history.bestObjHistory)
        elif domain == "analysis":
            _, ax = plot_sa({"saved": reader}, metric="mu_star")
            np.testing.assert_array_equal([b.get_height() for b in ax.patches], first["mu_star"].values[0])
        elif domain == "inference":
            _, axes = plot_infer_trace(reader, idx=[1], burnIn=2)
            for chain, line in enumerate(axes[0].lines):
                np.testing.assert_array_equal(line.get_ydata(), first.decs[chain, 2:, 1])
    assert reader.conn is None
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    run()
    assert len(list(tmp_path.rglob("*.sqlite3"))) == 2
    with readerClass(path) as reader:
        compareTree(loaded, reader.load_result())


def sensitivity(values, labels=("a", "b"), name="S1"):
    return AnaResult(
        runId="test",
        method="test",
        problemName="test",
        nInput=2,
        nOutput=1,
        nCon=0,
        target="objs",
        settings={},
        meta={},
        metrics=[AnaMetric(name, np.atleast_2d(values), ["y"], list(labels), "input")],
        X=None,
        Y=None,
        runtime=0.0,
        createdAt="test",
    )


def testSensitivityBarsRetainSignsScaleAndAlignLabels():
    first = sensitivity([-0.2, 0.6])
    second = sensitivity([1.5, 0.3], labels=("b", "a"))
    _, ax = plot_sa({"one": first, "two": second})
    np.testing.assert_allclose([bar.get_height() for bar in ax.patches], [-0.2, 0.6, 0.3, 1.5])
    assert ax.get_ylim()[0] < -0.2 and ax.get_ylim()[1] > 1.5
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b"]
    np.testing.assert_array_equal(first["S1"].values, [[-0.2, 0.6]])


def testSensitivitySelectsOutputAndReportsNonfiniteBars():
    result = sensitivity([[0.2, 0.4], [np.inf, 0.8]])
    with pytest.warns(RuntimeWarning, match="non-finite"):
        _, ax = plot_sa({"one": result}, outputIndex=1)
    assert np.isnan(ax.patches[0].get_height()) and ax.patches[1].get_height() == 0.8


def optimization():
    p = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=lambda x: np.sum(x * x, axis=1, keepdims=True))
    return GA(nPop=4, maxIters=0, saveFlag=False, verboseFlag=False).run(p, seed=17)


def testStatisticsMatchOnlyCommonCoordinatesAndRetainNegativeBand():
    first, second = optimization(), optimization()
    first.history.iterToFEs = [[0, 10], [1, 20], [2, 30]]
    second.history.iterToFEs = [[0, 20], [1, 30], [2, 40]]
    first.history.bestObjHistory = [-2.0, -4.0, -6.0]
    second.history.bestObjHistory = [-8.0, -10.0, -12.0]
    with pytest.warns(RuntimeWarning, match="shared coordinates"):
        _, ax = plot_op_curve_stat({"group": [first, second]}, xCoord="fe")
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [20, 30])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [-6.0, -8.0])
    vertices = ax.collections[0].get_paths()[0].vertices
    assert vertices[:, 1].max() < 0
    assert vertices[:, 1].min() == pytest.approx(-8 - np.sqrt(8))


def testMissingHistoryMetricsKeepTheirOwnCoordinates():
    first, second = optimization(), optimization()
    first.history.iterToFEs = second.history.iterToFEs = [[0, 10], [1, 20], [2, 30]]
    first.history.bestObjHistory = [1.0, None, 3.0]
    second.history.bestObjHistory = [4.0, 5.0, None]
    with pytest.warns(RuntimeWarning, match="shared coordinates"):
        _, ax = plot_op_curve_stat({"group": [first, second]}, xCoord="fe")
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [10])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2.5])


@pytest.mark.parametrize("dimensions", [2, 3])
def testParetoReferenceUsesPointRows(dimensions):
    result = optimization()
    reference = np.arange(12, dtype=float).reshape(-1, dimensions)
    result.bestObjs = reference + 0.5
    _, ax = plot_op_pareto(result, optima=reference)
    if dimensions == 2:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), reference[:, 0])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), reference[:, 1])
    else:
        for observed, expected in zip(ax.collections[1]._offsets3d, reference.T):
            np.testing.assert_array_equal(observed, expected)


def testTraceTitleRetainsOriginalVariableIndex():
    p = Problem(nInput=3, nObj=1, lb=-1.0, ub=1.0, objFunc=lambda x: np.sum(x * x, axis=1, keepdims=True))
    result = MH(nChains=2, warmUp=0, maxIters=5, saveFlag=False, verboseFlag=False).run(p, seed=17)
    _, axes = plot_infer_trace(result, idx=[2, 0], burnIn=1)
    assert [ax.get_title() for ax in axes] == ["Decision Variable 3", "Decision Variable 1"]
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), result.decs[0, 1:, 2])
