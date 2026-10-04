from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.optimization.expensive import MOASMO
from UQPyL.optimization.moea import NSGAII
from UQPyL.problem import Problem


QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False, hvFlag=False)


@pytest.mark.parametrize("case", ["spread", "short_front", "few", "duplicates"])
def testAdvancedInfillingSelectsNovelPointsAndCountsEvaluations(case, monkeypatch):
    received = []

    def objective(X):
        received.append(X.copy())
        return np.column_stack((X[:, 0], 20 - X[:, 0]))

    problem = Problem(nInput=1, nObj=2, lb=10, ub=20, objFunc=objective)
    candidates = np.array([[0.2], [0.5], [0.8], [0.9]])
    scores = np.column_stack((10 + 10 * candidates[:, 0], 10 - 10 * candidates[:, 0]))
    if case == "short_front":
        scores = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    elif case == "few":
        candidates, scores = candidates[:1], scores[:1]
    elif case == "duplicates":
        candidates = np.array([[0.0], [0.0], [0.5], [0.5]])
        scores = np.column_stack((10 + 10 * candidates[:, 0], 10 - 10 * candidates[:, 0]))
    inner = NSGAII(nPop=4, **QUIET)
    monkeypatch.setattr(inner, "run", lambda *args, **kwargs: SimpleNamespace(bestDecs=candidates, bestObjs=scores))
    method = MOASMO(nInit=4, pct=0.5, advance_infilling=True, maxIters=1, optimizer=inner, **QUIET)
    result = method.run(problem, initialPop=[[10], [11], [19.5], [20]], seed=7)
    evaluated = np.vstack(received)
    assert result.FEs == len(evaluated) == 6
    assert len(np.unique(evaluated, axis=0)) == 6
    assert np.all((evaluated >= 10) & (evaluated <= 20))
    if case == "spread":
        # Independent geometric expectation: middle first, then the wider gap.
        np.testing.assert_allclose(evaluated[4:, 0], [15, 18])
    if case in ("short_front", "few"):
        assert np.any(np.isclose(evaluated[4:, 0], 12))
    np.testing.assert_allclose(result.bestObjs[:, 1], 20 - result.bestDecs[:, 0])


def testAdvancedInfillingStopsWhenDiscreteDomainIsExhausted(monkeypatch):
    received = []

    def objective(X):
        received.extend(X[:, 0].tolist())
        return np.column_stack((X[:, 0], -X[:, 0]))

    problem = Problem(nInput=1, nObj=2, lb=0, ub=1, varType=[2], varSet={0: [10, 20]}, objFunc=objective)
    inner = NSGAII(nPop=4, **QUIET)
    monkeypatch.setattr(
        inner,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            bestDecs=np.array([[0.1], [0.2], [0.8]]), bestObjs=np.array([[10, -10], [10, -10], [20, -20]])
        ),
    )
    result = MOASMO(nInit=2, pct=0.5, advance_infilling=True, maxIters=2, optimizer=inner, **QUIET).run(
        problem, initialPop=[[10], [20]], seed=3
    )
    assert result.FEs == 2
    assert sorted(received) == [10, 20]
    assert result.stopReason == "no_novel_candidates"


def testAdvancedInfillingWithRealSurrogatesAndInnerSearch():
    received = []

    def objective(X):
        received.append(X.copy())
        return np.column_stack((np.sum(X**2, axis=1), np.sum((X - 1) ** 2, axis=1)))

    problem = Problem(nInput=2, nObj=2, lb=-2, ub=3, objFunc=objective)
    inner = NSGAII(nPop=12, maxIters=2, **QUIET)
    result = MOASMO(nInit=8, pct=0.25, advance_infilling=True, maxIters=1, optimizer=inner, **QUIET).run(
        problem, seed=11
    )
    evaluated = np.vstack(received)
    assert result.FEs == len(evaluated) == 10
    assert len(np.unique(evaluated, axis=0)) == 10
    assert np.all((evaluated >= -2) & (evaluated <= 3))
    np.testing.assert_allclose(result.bestObjs, objective(result.bestDecs))
