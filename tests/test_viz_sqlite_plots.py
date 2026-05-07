from pathlib import Path
import shutil
import uuid

import numpy as np

from UQPyL.analysis.runtime import AnaMetric, AnaResult
from UQPyL.inference.runtime import InfHistory, InfResult
from UQPyL.optimization.runtime import OptHistory, OptReader, OptResult
from UQPyL.viz import (
    plot_infer_stat,
    plot_infer_stat_combined,
    plot_infer_trace,
    plot_op_curve,
    plot_op_curve_stat,
    plot_op_pareto,
    plot_sa,
    plot_surrogate,
)


def _dummy_opt_result():
    history = OptHistory(
        iterToFEs=[[0, 4], [1, 8], [2, 12]],
        bestObjHistory=[3.0, 2.0, 1.0],
        bestMetricHistory=[0.2, 0.3, 0.5],
        numBestHistory=[2, 3, 4],
    )
    return OptResult(
        bestDecs=np.array([[0.1, 0.2]]),
        bestObjs=np.array([[1.0]]),
        bestCons=None,
        bestMetric=None,
        bestFeasible=True,
        appearFEs=12,
        appearIters=2,
        FEs=12,
        iters=2,
        runtime=0.5,
        history=history,
    )


def _dummy_mo_result():
    history = OptHistory(
        iterToFEs=[[0, 4], [1, 8], [2, 12]],
        bestMetricHistory=[0.2, 0.3, 0.5],
        numBestHistory=[2, 3, 4],
    )
    return OptResult(
        bestDecs=np.array([[0.1, 0.2], [0.3, 0.4]]),
        bestObjs=np.array([[1.0, 2.0], [0.8, 2.2]]),
        bestCons=None,
        bestMetric=0.5,
        bestFeasible=True,
        appearFEs=12,
        appearIters=2,
        FEs=12,
        iters=2,
        runtime=0.5,
        history=history,
    )


def _dummy_ana_result():
    return AnaResult(
        runId="ana_001",
        method="FAST",
        problemName="Demo",
        nInput=3,
        nOutput=1,
        nCon=0,
        target="objs",
        settings={"M": 4},
        meta={"designType": "fast"},
        metrics=[
            AnaMetric(
                name="S1",
                values=np.array([[0.5, 0.3, 0.2]]),
                rowLabels=["obj1"],
                colLabels=["x1", "x2", "x3"],
                colDim="input",
            )
        ],
        X=None,
        Y=None,
        runtime=0.1,
        createdAt="2026-05-06T00:00:00",
    )


def _dummy_inf_result():
    decs = np.array(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]],
        ]
    )
    objs = np.sum(decs**2, axis=2, keepdims=True)
    return InfResult(
        runId="inf_001",
        method="MH",
        problemName="Demo",
        nInput=2,
        nOutput=1,
        nCon=0,
        settings={"nChains": 2},
        runtime=0.2,
        createdAt="2026-05-06T00:00:00",
        decs=decs,
        objs=objs,
        cons=None,
        logProb=-objs[..., 0],
        accepted=np.ones((2, 3), dtype=bool),
        feasibleMask=np.ones((2, 3), dtype=bool),
        acceptanceRate=np.array([1.0, 1.0]),
        bestDecs=np.array([[0.0, 0.1]]),
        bestObjs=np.array([[0.01]]),
        bestCons=None,
        bestFeasible=True,
        FEs=6,
        iters=2,
        history=InfHistory(),
    )


def test_viz_exports_work_with_result_objects(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    fig1, ax1 = plot_op_curve({"demo": _dummy_opt_result()})
    fig2, ax2 = plot_op_curve_stat({"demo": _dummy_opt_result()})
    fig3, ax3 = plot_op_pareto(_dummy_mo_result())
    fig4, ax4 = plot_sa({"demo": _dummy_ana_result()})
    fig5, ax5 = plot_surrogate("RBF", np.array([[1.0], [2.0]]), np.array([[1.1], [1.9]]))
    fig6, ax6 = plot_infer_trace(_dummy_inf_result())
    fig7, ax7 = plot_infer_stat(_dummy_inf_result())
    fig8, ax8 = plot_infer_stat_combined(_dummy_inf_result())

    assert fig1 is not None and ax1 is not None
    assert fig2 is not None and ax2 is not None
    assert fig3 is not None and ax3 is not None
    assert fig4 is not None and ax4 is not None
    assert fig5 is not None and ax5 is not None
    assert fig6 is not None and ax6 is not None
    assert fig7 is not None and ax7 is not None
    assert fig8 is not None and ax8 is not None


def test_viz_optimization_accepts_sqlite_path_and_reader(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    from UQPyL.optimization.soea import GA
    from UQPyL.problem.sop.single_simple_problem import Sphere

    work_dir = Path(".cache") / "viz_sqlite_test" / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        problem = Sphere(nInput=2, ub=1.0, lb=-1.0)
        problem.workDir = str(work_dir)

        result_dir = work_dir / "Result"
        before = set(result_dir.glob("*.sqlite3")) if result_dir.exists() else set()
        GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=True, saveFreq=2).run(problem, seed=123)
        after = set(result_dir.glob("*.sqlite3"))
        db_files = sorted(after - before)
        assert len(db_files) == 1

        fig1, ax1 = plot_op_curve({"demo": str(db_files[0])})
        reader = OptReader(str(db_files[0]))
        try:
            fig2, ax2 = plot_op_curve({"demo": reader})
        finally:
            reader.close()

        assert fig1 is not None and ax1 is not None
        assert fig2 is not None and ax2 is not None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
